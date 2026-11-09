"""Evaluate an Ultralytics YOLO model with OUR evaluator (same images, ground truth and metric as evaluate.py).

    python scripts/predict_yolo.py --weights runs/yolo/v8n/weights/best.pt \
        --compare runs/backbone_layer3_lr3e-4/eval_valid_best_corrected.pt

Whole images at NATIVE resolution: each image is padded right / bottom (grey 114, YOLO's letterbox
colour) to a multiple of 32 and at least 512 px, and `imgsz` is set to that size, so Ultralytics
neither shrinks the 2048 px scenes nor enlarges the 256 px tiles (checked on every image).
Same output thresholds as our model (RoIConfig): score >= 0.05, at most 100 detections; the NMS
IoU is YOLO's own default (0.7; ours is 0.5), as part of the baseline "as shipped".
Writes next to the weights eval_<split>_<weights name>_corrected.{json,pt}, readable by
evaluate.py --compare, analyze_errors.py and the review pages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from courtdet.config import load_config
from courtdet.data.corrections import load_records
from courtdet.engine.evaluate import match_split, paired_bootstrap, report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import format_table  # noqa: E402  (scripts/evaluate.py)

PAD_VALUE = 114


def padded_size(height: int, width: int, min_size: int = 512, stride: int = 32) -> tuple[int, int]:
    """Smallest (H, W) >= (height, width) and >= min_size, multiples of `stride`."""
    return tuple(math.ceil(max(v, min_size) / stride) * stride for v in (height, width))


def predict_image(model, path: Path, device: str, conf: float, max_det: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    H, W = padded_size(h, w)
    canvas = np.full((H, W, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:h, :w] = rgb
    # numpy inputs are read as BGR by Ultralytics (OpenCV convention)
    result = model.predict(canvas[:, :, ::-1].copy(), imgsz=[H, W], conf=conf, max_det=max_det, device=device,
                           verbose=False)[0]
    if tuple(result.orig_shape) != (H, W):
        raise RuntimeError(f"{path.name}: unexpected shape {result.orig_shape}")
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    scores = result.boxes.conf.cpu().numpy().astype(np.float64)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)  # the padding is not part of the image
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    keep = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0)
    order = np.argsort(-scores[keep], kind="mergesort")
    return boxes[keep][order], scores[keep][order]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--config", default="configs/corrected.yaml", help="ground truth (corrections, decisions)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", type=Path, default=None, help="detections .pt of another model, same split")
    args = parser.parse_args(argv)

    from ultralytics import YOLO  # optional dependency: pip install -e ".[yolo]"

    d = load_config(args.config).data
    root = Path(d.root)
    records = load_records(root, args.split, corrections=d.corrections, decisions=d.decisions)
    model = YOLO(str(args.weights))
    t0 = time.time()
    detections = []
    for i, r in enumerate(records):
        boxes, scores = predict_image(model, root / args.split / r["file_name"], args.device, args.conf, args.max_det)
        detections.append({"file_name": r["file_name"], "boxes": boxes, "scores": scores})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)} images", flush=True)
    seconds = time.time() - t0
    print(f"inference: {len(records)} images in {seconds:.1f}s on {args.device}", flush=True)

    results = match_split(detections, records)
    metrics = report(results, records, n_boot=args.n_boot, seed=args.seed)
    print(format_table(metrics))
    stem = args.weights.parent / f"eval_{args.split}_{args.weights.stem}{'_corrected' if d.corrections else ''}"
    payload = {"checkpoint": str(args.weights), "model": "ultralytics", "split": args.split, "root": str(root),
               "corrections": d.corrections, "decisions": d.decisions, "conf": args.conf, "max_det": args.max_det,
               "n_boot": args.n_boot, "seed": args.seed, "inference_seconds": round(seconds, 1), "metrics": metrics}
    if args.compare is not None:
        other = torch.load(args.compare, weights_only=False)
        if other["split"] != args.split:
            raise SystemExit(f"--compare holds the {other['split']!r} split, not {args.split!r}")
        diff = paired_bootstrap(results, match_split(other["detections"], records), n_boot=args.n_boot, seed=args.seed)
        payload["compare"] = {"other": str(args.compare), "this_minus_other": diff}
        print(f"this - other: AP@0.5 {diff['ap50']:+.3f} [{diff['ap50_lo']:+.3f} {diff['ap50_hi']:+.3f}]"
              f"   AP@[.5:.95] {diff['ap']:+.3f} [{diff['ap_lo']:+.3f} {diff['ap_hi']:+.3f}]")
    Path(f"{stem}.json").write_text(json.dumps(payload, indent=2))
    torch.save({"checkpoint": str(args.weights), "split": args.split, "detections": detections}, f"{stem}.pt")
    print(f"written: {stem}.json, {stem}.pt")


if __name__ == "__main__":
    main(sys.argv[1:])
