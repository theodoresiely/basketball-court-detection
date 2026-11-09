"""Train the Ultralytics YOLO baseline on the export of scripts/export_yolo.py.

    python scripts/train_yolo.py --data data/yolo/v2_corrected/data.yaml --name v8n_e30 --epochs 30

Ultralytics defaults everywhere (optimizer "auto", online augmentations: mosaic, HSV, scale,
translation, horizontal flip), except: imgsz 512 (= our crop size, the export is at native
resolution), batch 16 (the notebook's), and the number of epochs, set so that the run takes about
as long as our own training on the same machine. Ultralytics keeps as best.pt
the epoch with the best mAP@[.5:.95] on its validation crops (same criterion as our own best.pt).
Outputs in runs/yolo/<name>/ (weights/best.pt, results.csv, args.yaml).

--no-augment turns off every online augmentation active by default (mosaic, HSV, translation,
scale, horizontal flip; the others are already 0), like our own training, which only draws a new
random crop each epoch. It isolates how much of YOLO's lead comes from augmentations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default="runs/yolo/yolov8n.pt", help="pretrained weights (downloaded if absent)")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-augment", action="store_true", help="no online augmentation (see above)")
    args = parser.parse_args(argv)

    from ultralytics import YOLO  # optional dependency: pip install -e ".[yolo]"
    from ultralytics.utils import SETTINGS

    SETTINGS.update(sync=False)  # no anonymous usage statistics sent to Ultralytics
    no_aug = dict(mosaic=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0, fliplr=0.0)
    model = YOLO(args.model)
    print(f"parameters: {sum(p.numel() for p in model.model.parameters())}", flush=True)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=512, batch=args.batch, device=args.device,
                seed=args.seed, project=str(Path("runs/yolo").resolve()), name=args.name, exist_ok=False,
                plots=False, **(no_aug if args.no_augment else {}))


if __name__ == "__main__":
    main(sys.argv[1:])
