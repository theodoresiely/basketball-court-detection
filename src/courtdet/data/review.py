"""Human review of source A annotations: does each box hold ONE court or SEVERAL? (step 1)

Source A sometimes puts adjacent courts in a single box. Before any
relabelling, a human sorts every ORIGINAL image of source A:
    "single"   every box holds one court (nothing to change)
    "multiple" at least one box holds several courts
    "unsure"   to look at again
In the train split each original image exists in 2 augmented copies (Roboflow); one decision
covers both copies. The copies are grouped by the part of the file name before ".rf.".

Decisions are written to a JSON file after every click (atomic replace), so the review can be
stopped and resumed. Nothing here changes the annotations.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Sequence

from courtdet.data.dataset import load_coco_records

DECISIONS = ("single", "multiple", "unsure")


def source_key(file_name: str) -> str:
    """'basketball_court_523_jpg.rf.eea455...jpg' -> 'basketball_court_523_jpg' (shared by the copies)."""
    return file_name.split(".rf.")[0]


def build_items(root: str | Path, splits: Sequence[str] = ("valid", "test", "train")) -> list[dict]:
    """One item per original source A image: {"id", "split", "images": [{"file_name", "width", "height", "boxes"}]}.

    Order: splits in the given order, then by source name (deterministic).
    """
    items = []
    for split in splits:
        records = load_coco_records(Path(root) / split / "_annotations.coco.json")
        groups: dict[str, list[dict]] = {}
        for r in records:
            if r["source"] == "A":
                groups.setdefault(source_key(r["file_name"]), []).append(r)
        for key in sorted(groups):
            images = sorted(groups[key], key=lambda r: r["file_name"])
            items.append({
                "id": f"{split}/{key}",
                "split": split,
                "images": [{"file_name": r["file_name"], "width": r["width"], "height": r["height"],
                            "boxes": [[round(float(v), 2) for v in b] for b in r["boxes"]]} for r in images],
            })
    return items


class DecisionStore:
    """{item id: {"decision", "time"}} persisted to `path` (JSON), rewritten atomically on each change."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}
        # The server handles requests on several threads: without a lock, a change during
        # json.dump of another thread could corrupt the write or raise.
        self._lock = threading.Lock()

    def set(self, item_id: str, decision: str | None) -> None:
        """decision None removes the item's decision (undo)."""
        if decision is not None and decision not in DECISIONS:
            raise ValueError(f"unknown decision {decision!r}; expected one of {DECISIONS}")
        with self._lock:
            if decision is None:
                self.data.pop(item_id, None)
            else:
                self.data[item_id] = {"decision": decision, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._write()

    def _write(self) -> None:
        # Write to a temporary file then rename: a crash mid-write cannot leave a truncated file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".decisions-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(self.data, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def snapshot(self) -> dict[str, dict]:
        """Copy taken under the lock (safe to serialise while other threads write)."""
        with self._lock:
            return {k: dict(v) for k, v in self.data.items()}

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(DECISIONS, 0)
        for v in self.snapshot().values():
            out[v["decision"]] += 1
        return out
