"""Error analysis from saved detections (output of scripts/evaluate.py), at IoU 0.5.

    python scripts/analyze_errors.py --detections runs/prelim_lr3e-4/eval_valid_best.pt

Writes to <run>/analysis_<split>_<checkpoint>/:
    report.txt / report.json   decomposition of the AP, false alarms, courts by type / size / ratio, scores
    top_false_alarms_B.png     the highest-scoring false alarms of source B (colours: see courtdet.utils.viz)
    top_false_alarms_A.png     the same for source A
    missed_courts.png          the courts of source B that no detection touches at IoU >= 0.5 (in cyan)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from courtdet.data.corrections import load_records
from courtdet.utils.viz import CYAN, sheet, tile
from courtdet.engine.analysis import (
    RATIO_LABELS,
    SIZE_LABELS,
    ap50_without,
    classify_courts,
    classify_detections,
    image_type,
    max_recall,
    quantiles,
    ratio_bins,
    size_bins,
)

FP_KINDS = ("background", "localization", "duplicate")
THRESHOLDS = (0.05, 0.5, 0.9)


# ---------------------------------------------------------------- report


def analyse(records, detections) -> tuple[dict, list[dict]]:
    """All the numbers of the report, plus one row per image (used for the contact sheets)."""
    rows = []
    for r, d in zip(records, detections):
        boxes, scores, labels = classify_detections(d["boxes"], d["scores"], r["boxes"])
        best, court_labels = classify_courts(d["boxes"], r["boxes"])
        rows.append({"record": r, "type": image_type(r["file_name"]), "boxes": boxes, "scores": scores,
                     "labels": labels, "court_best_iou": best, "court_labels": court_labels})

    out = {}
    for source in ("A", "B"):
        part = [x for x in rows if x["record"]["source"] == source]
        cls = [(x["boxes"], x["scores"], x["labels"]) for x in part]
        n_gts = [len(x["record"]["boxes"]) for x in part]
        positives = [i for i, n in enumerate(n_gts) if n > 0]
        court_labels = np.concatenate([x["court_labels"] for x in part])
        out[source] = {
            "n_images": len(part),
            "n_images_with_courts": len(positives),
            "n_courts": int(sum(n_gts)),
            "ap50": ap50_without(cls, n_gts),
            "ap50_images_with_courts_only": ap50_without([cls[i] for i in positives], [n_gts[i] for i in positives]),
            **{f"ap50_without_{k}": ap50_without(cls, n_gts, remove=(k,)) for k in FP_KINDS},
            "ap50_without_any_false_alarm": ap50_without(cls, n_gts, remove=FP_KINDS),
            "max_recall_matched": max_recall(cls, n_gts),
            "courts": dict(Counter(court_labels.tolist())),
            "false_alarms": {
                f"score>={t}": {
                    where: dict(Counter(np.concatenate(
                        [x["labels"][(x["scores"] >= t) & (x["labels"] != "tp")] for x in part
                         if (len(x["record"]["boxes"]) > 0) == (where == "on_images_with_courts")] or [np.array([])]
                    ).tolist()))
                    for where in ("on_images_with_courts", "on_images_without_courts")
                }
                for t in THRESHOLDS
            },
            "scores_tp_quantiles_10_25_50_75_90": quantiles(np.concatenate([x["scores"][x["labels"] == "tp"] for x in part])),
            "scores_fp_quantiles_10_25_50_75_90": quantiles(np.concatenate([x["scores"][x["labels"] != "tp"] for x in part])),
        }

    by_type = {}
    for t in sorted({x["type"] for x in rows}):
        part = [x for x in rows if x["type"] == t]
        labels = np.concatenate([x["court_labels"] for x in part])
        by_type[t] = {
            "n_images": len(part),
            "n_images_with_courts": sum(len(x["record"]["boxes"]) > 0 for x in part),
            "n_courts": int(len(labels)),
            "courts": dict(Counter(labels.tolist())),
            "max_recall_matched": max_recall([(x["boxes"], x["scores"], x["labels"]) for x in part],
                                             [len(x["record"]["boxes"]) for x in part]),
            "false_alarms_per_image_score>=0.5": float(np.mean(
                [((x["scores"] >= 0.5) & (x["labels"] != "tp")).sum() for x in part])),
        }
    out["by_image_type"] = by_type

    for source in ("A", "B"):
        part = [x for x in rows if x["record"]["source"] == source]
        gts = np.concatenate([x["record"]["boxes"].reshape(-1, 4) for x in part])
        labels = np.concatenate([x["court_labels"] for x in part])
        for name, bins, names in (("by_size", size_bins(gts), SIZE_LABELS), ("by_ratio", ratio_bins(gts), RATIO_LABELS)):
            out[source][name] = {
                n: {"n_courts": int((bins == i).sum()), **dict(Counter(labels[bins == i].tolist()))}
                for i, n in enumerate(names)
            }
    return out, rows


def format_report(out: dict) -> str:
    lines = []
    for source in ("A", "B"):
        s = out[source]
        lines += [
            f"=== source {source}: {s['n_images']} images ({s['n_images_with_courts']} with courts), {s['n_courts']} courts",
            f"AP@0.5                                      {s['ap50']:.3f}",
            f"  only images with courts (no FP on negatives) {s['ap50_images_with_courts_only']:.3f}",
            *[f"  without '{k}' false alarms{'':<{14 - len(k)}}      {s[f'ap50_without_{k}']:.3f}" for k in FP_KINDS],
            f"  without any false alarm                   {s['ap50_without_any_false_alarm']:.3f}",
            f"max recall (matched, all detections kept)  {s['max_recall_matched']:.3f}",
            f"courts by best IoU with any detection: {s['courts']}",
            "false alarms (count) by score threshold:",
            *[f"  {t:<11} {v}" for t, v in s["false_alarms"].items()],
            "courts by size (sqrt area, native px): " + str(s["by_size"]),
            "courts by width/height ratio: " + str(s["by_ratio"]),
            f"score quantiles 10/25/50/75/90  TP {[round(v, 3) for v in s['scores_tp_quantiles_10_25_50_75_90']]}",
            f"                                FP {[round(v, 3) for v in s['scores_fp_quantiles_10_25_50_75_90']]}",
            "",
        ]
    lines.append("=== by image type")
    for t, v in out["by_image_type"].items():
        lines.append(f"{t:<11} {v}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", required=True, type=Path, help="eval_<split>_<ckpt>.pt from evaluate.py")
    parser.add_argument("--root", default=None, help="dataset root (default: the one recorded by evaluate.py)")
    parser.add_argument("--n-tiles", type=int, default=16)
    args = parser.parse_args(argv)

    saved = torch.load(args.detections, weights_only=False)
    meta = json.loads(args.detections.with_suffix(".json").read_text())
    root = Path(args.root or meta["root"])
    image_dir = root / saved["split"]
    # Same ground truth as evaluate.py used (corrected boxes / dropped images, if any).
    records = load_records(root, saved["split"], corrections=meta.get("corrections"), decisions=meta.get("decisions"))
    if [d["file_name"] for d in saved["detections"]] != [r["file_name"] for r in records]:
        raise SystemExit("the detections do not match the annotation file of this split")

    out, rows = analyse(records, saved["detections"])
    out_dir = args.detections.parent / f"analysis_{args.detections.stem.removeprefix('eval_')}"
    out_dir.mkdir(exist_ok=True)
    text = format_report(out)
    (out_dir / "report.txt").write_text(text + "\n")
    (out_dir / "report.json").write_text(json.dumps(out, indent=2))
    print(text)

    for source in ("A", "B"):
        false_alarms = sorted(
            ((s, x, box, lab) for x in rows if x["record"]["source"] == source
             for box, s, lab in zip(x["boxes"], x["scores"], x["labels"]) if lab != "tp"),
            key=lambda item: -item[0],
        )[: args.n_tiles]
        sheet([tile(image_dir, x["record"], (x["boxes"], x["scores"]), box, f"FP {lab} score {s:.2f}")
               for s, x, box, lab in false_alarms]).save(
            out_dir / f"top_false_alarms_{source}.png")

    b_rows = [x for x in rows if x["record"]["source"] == "B"]

    missed = [(x, g, iou, lab) for x in b_rows
              for g, iou, lab in zip(x["record"]["boxes"], x["court_best_iou"], x["court_labels"]) if lab != "touched"]
    missed.sort(key=lambda item: item[2])  # worst first
    sheet([tile(image_dir, x["record"], (x["boxes"], x["scores"]), g,
                f"{lab}, best IoU {iou:.2f}, side {np.sqrt((g[2]-g[0])*(g[3]-g[1])):.0f}px", focus_color=CYAN)
           for x, g, iou, lab in missed[: args.n_tiles]]).save(out_dir / "missed_courts.png")
    print(f"\nwritten to {out_dir} ({min(len(missed), args.n_tiles)}/{len(missed)} "
          f"courts not touched shown)")


if __name__ == "__main__":
    main(sys.argv[1:])
