"""Run a checkpoint on arbitrary images (no ground truth, no metric) and draw the detections.

    python scripts/predict.py --checkpoint runs/augment_layer3_e30_lr3e-4/best.pt --images a.png b.webp
    python scripts/predict.py --checkpoint runs/a/best.pt --images photos/ --threshold 0.5 --out-dir out/

For each image, prints every detection with a score >= --min-score and writes <out-dir>/<name>_pred.png with the
boxes whose score >= --threshold drawn in red. The whole image is processed in one pass, as in scripts/evaluate.py.
The threshold only affects the drawing: it is NOT a tuned operating point (the AP evaluation uses every detection).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from courtdet.engine.evaluate import load_image, predict_image
from courtdet.engine.train import select_device
from evaluate import load_model  # scripts/evaluate.py (the script directory is on sys.path)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def collect_images(paths: list[Path]) -> list[Path]:
    """Files as given; a directory is expanded to the images it contains (sorted, not recursive)."""
    images = []
    for p in paths:
        if p.is_dir():
            images += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.is_file():
            images.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    return images


def draw_detections(path: Path, boxes, scores, threshold: float) -> Image.Image:
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x1, y1, x2, y2 = (float(v) for v in box)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1 + 3, y1 + 3), f"{score:.2f}", fill=(255, 0, 0))
    return img


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--images", required=True, nargs="+", type=Path, help="image files and/or directories")
    parser.add_argument("--out-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--threshold", type=float, default=0.3, help="minimum score for a box to be drawn")
    parser.add_argument("--min-score", type=float, default=0.05, help="minimum score for a detection to be printed")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    images = collect_images(args.images)
    if not images:
        raise SystemExit("no image found")
    device = select_device(args.device)
    model, config = load_model(args.checkpoint, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in images:
        image = load_image(path)
        boxes, scores = predict_image(model, image, device, min_size=config.data.crop_size)
        order = scores.argsort()[::-1]
        print(f"== {path}  ({image.shape[2]}x{image.shape[1]} px)")
        for i in order:
            if scores[i] >= args.min_score:
                print(f"  score {scores[i]:.3f}  box {[round(float(v), 1) for v in boxes[i]]}")
        if not (scores >= args.min_score).any():
            print(f"  no detection with a score >= {args.min_score}")
        out = args.out_dir / f"{path.stem}_pred.png"
        draw_detections(path, boxes, scores, args.threshold).save(out)
        print(f"  drawn (score >= {args.threshold}): {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
