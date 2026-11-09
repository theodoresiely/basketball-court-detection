"""Human correction of source A boxes, step 2: one box per court.

Step 1 (`review.py`) sorted the original source A images into single / multiple / unsure.
Step 2 redraws the boxes of the "multiple" and "unsure" images, one image at a time:
    "fixed"    the saved boxes are the right ones (one box per full court)
    "excluded" the image holds a half court or is unusable: it leaves training and evaluation
               (a half court gets no box: labelling it as a court or as background would both
               send a contradictory signal)

To save work, the page proposes a GEOMETRIC split of each original box into n equal parts along
its long (or short) side. The rule does not look at the image: it is right only when the courts
have the same size, are aligned and fill the box. It never uses the model's detections, so the
valid/test boxes do not favour the model being evaluated.

Corrections are written to their own JSON file {"<split>/<file_name>": {"status", "boxes",
"method", "time"}}; the Roboflow annotations are never modified. `apply_corrections` replaces
the boxes of the corrected records when loading.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from courtdet.data.dataset import load_coco_records
from courtdet.data.review import DecisionStore, build_items, source_key

STATUSES = ("fixed", "excluded")
TO_CORRECT = ("multiple", "unsure")
SPLIT_COUNTS = (1, 2, 3, 4)
MIN_SIZE = 2.0  # px: smaller boxes are drawing slips, not courts


def split_box(box: Sequence[float], n: int, along: str) -> list[list[float]]:
    """Cut `box` (x1, y1, x2, y2) into n equal parts along "x" (vertical cuts) or "y" (horizontal cuts)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if along not in ("x", "y"):
        raise ValueError(f"along must be 'x' or 'y', got {along!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    parts = []
    for k in range(n):
        if along == "x":
            parts.append([x1 + (x2 - x1) * k / n, y1, x1 + (x2 - x1) * (k + 1) / n, y2])
        else:
            parts.append([x1, y1 + (y2 - y1) * k / n, x2, y1 + (y2 - y1) * (k + 1) / n])
    return parts


def propose(boxes: Sequence[Sequence[float]], n: int, side: str) -> list[list[float]]:
    """Split every box into n parts along its "long" or "short" side (a square box counts x as long)."""
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    out = []
    for b in boxes:
        x_is_long = (b[2] - b[0]) >= (b[3] - b[1])
        along = "x" if x_is_long == (side == "long") else "y"
        out.extend(split_box(b, n, along))
    return [[round(v, 2) for v in b] for b in out]


def build_edit_items(root: str | Path, decisions: Mapping[str, dict],
                     splits: Sequence[str] = ("valid", "test")) -> list[dict]:
    """One item per original image to correct (step 1 decision "multiple" or "unsure").

    Each item edits ONE image: in the train split the 2 augmented copies have different boxes
    (Roboflow moved the courts), so only the first copy by file name is corrected.
    Item: {"id", "key" (key of the corrections file), "split", "decision", "file_name", "width", "height", "boxes" (original),
    "proposals": {"<n>-<side>": boxes}}.
    """
    items = []
    for it in build_items(root, splits):
        decision = decisions.get(it["id"], {}).get("decision")
        if decision not in TO_CORRECT:
            continue
        im = it["images"][0]
        items.append({
            "id": it["id"], "key": f"{it['split']}/{im['file_name']}", "split": it["split"], "decision": decision,
            "file_name": im["file_name"], "width": im["width"], "height": im["height"], "boxes": im["boxes"],
            "proposals": {f"{n}-{side}": propose(im["boxes"], n, side)
                          for n in SPLIT_COUNTS for side in ("long", "short")},
        })
    return items


def validate_boxes(boxes, width: float, height: float, min_size: float = MIN_SIZE) -> list[list[float]]:
    """Check drawn boxes: 4 numbers each, inside the image, at least `min_size` px on each side."""
    out = []
    for b in boxes:
        if len(b) != 4:
            raise ValueError(f"a box needs 4 numbers, got {b!r}")
        x1, y1, x2, y2 = (float(v) for v in b)
        if not (0 <= x1 and 0 <= y1 and x2 <= width and y2 <= height):
            raise ValueError(f"box {b!r} leaves the {width}x{height} image")
        if x2 - x1 < min_size or y2 - y1 < min_size:
            raise ValueError(f"box {b!r} is smaller than {min_size} px")
        out.append([round(v, 2) for v in (x1, y1, x2, y2)])
    return out


class CorrectionStore(DecisionStore):
    """{"<split>/<file_name>": {"status", "boxes", "method", "time"}}, same atomic writes as DecisionStore."""

    def set(self, key: str, status: str | None, boxes=(), method: str = "") -> None:  # type: ignore[override]
        """status None removes the correction (undo). An excluded image keeps no boxes."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
        if status == "fixed" and not boxes:
            raise ValueError("a fixed image needs at least one box (exclude it otherwise)")
        with self._lock:
            if status is None:
                self.data.pop(key, None)
            else:
                self.data[key] = {"status": status, "boxes": [list(b) for b in boxes] if status == "fixed" else [],
                                  "method": method, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._write()

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(STATUSES, 0)
        for v in self.snapshot().values():
            out[v["status"]] += 1
        return out


def apply_corrections(records: list[dict], corrections: Mapping[str, dict], split: str) -> list[dict]:
    """Records with the corrected boxes; "excluded" images are dropped. The input list is not modified."""
    out = []
    for r in records:
        c = corrections.get(f"{split}/{r['file_name']}")
        if c is None:
            out.append(r)
        elif c["status"] == "fixed":
            out.append({**r, "boxes": np.asarray(c["boxes"], dtype=np.float64).reshape(-1, 4)})
        elif c["status"] != "excluded":
            raise ValueError(f"unknown status {c['status']!r} for {split}/{r['file_name']}")
    return out


def select_records(records: list[dict], split: str, *, corrections: Mapping[str, dict] | None = None,
                   decisions: Mapping[str, dict] | None = None, source_a_fraction: float = 1.0,
                   seed: int = 0) -> list[dict]:
    """Records used for training or evaluation, in three steps (each optional):

    1. `corrections` (step 2 file): corrected boxes replace the Roboflow ones, "excluded" images are dropped.
    2. `decisions` (step 1 file): a source A image whose original was sorted "multiple" / "unsure" and
       that has NO correction is dropped: its boxes follow the old convention (one box for several
       courts). In the train split this also drops the second copy of a corrected source.
    3. `source_a_fraction` < 1: keep a random fraction of the source A ORIGINALS (both copies of a
       source together, so a source is never half kept), drawn with `seed`. For learning curves.
    Source B images are only affected by step 1 (they have no correction and no decision).
    """
    if not 0.0 < source_a_fraction <= 1.0:
        raise ValueError(f"source_a_fraction must be in (0, 1], got {source_a_fraction}")
    out = apply_corrections(records, corrections, split) if corrections else list(records)
    if decisions:
        corrected = set(corrections or {})
        out = [r for r in out
               if r["source"] != "A"
               or f"{split}/{r['file_name']}" in corrected
               or decisions.get(f"{split}/{source_key(r['file_name'])}", {}).get("decision") not in TO_CORRECT]
    if source_a_fraction < 1.0:
        keys = sorted({source_key(r["file_name"]) for r in out if r["source"] == "A"})
        rng = np.random.default_rng(seed)
        kept = set(rng.permutation(keys)[: round(source_a_fraction * len(keys))].tolist())
        out = [r for r in out if r["source"] != "A" or source_key(r["file_name"]) in kept]
    return out


def load_records(root: str | Path, split: str, *, corrections: str | Path | None = None,
                 decisions: str | Path | None = None, source_a_fraction: float = 1.0, seed: int = 0) -> list[dict]:
    """`load_coco_records` of root/split, then `select_records` with the JSON files given by path."""
    records = load_coco_records(Path(root) / split / "_annotations.coco.json")
    return select_records(
        records, split,
        corrections=json.loads(Path(corrections).read_text()) if corrections else None,
        decisions=json.loads(Path(decisions).read_text()) if decisions else None,
        source_a_fraction=source_a_fraction, seed=seed)
