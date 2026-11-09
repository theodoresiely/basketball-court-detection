"""Export train / valid to the YOLO format for the Ultralytics baseline (see courtdet.data.yolo_export).

    python scripts/export_yolo.py --config configs/corrected.yaml --out data/yolo/v2_corrected --views 4

Uses the records of OUR training (corrections, decisions and crop rule of the config), so that
YOLO sees the same boxes. Writes <out>/{train,valid}/{images,labels} and <out>/data.yaml.
The valid crops only serve Ultralytics to pick its best.pt; the comparison itself runs on the
whole valid images with our evaluator (scripts/predict_yolo.py). The test split is never exported.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from courtdet.config import load_config
from courtdet.data.corrections import load_records
from courtdet.data.yolo_export import export_split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/corrected.yaml")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--views", type=int, default=4, help="crops per image (A tiles are repeated)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    d = load_config(args.config).data
    if args.out.exists():
        shutil.rmtree(args.out)
    for k, split in enumerate(("train", "valid")):
        records = load_records(d.root, split, corrections=d.corrections, decisions=d.decisions,
                               source_a_fraction=d.source_a_fraction if split == "train" else 1.0, seed=d.subset_seed)
        counts = export_split(records, Path(d.root) / split, args.out / split, views=args.views,
                              crop_size=d.crop_size, tol=d.tol, p_pos=d.p_pos, max_tries=d.max_tries, seed=args.seed + k)
        print(f"{split}: {len(records)} records -> {counts}", flush=True)
    (args.out / "data.yaml").write_text(
        f"# written by scripts/export_yolo.py --config {args.config} --views {args.views} --seed {args.seed}\n"
        f"path: {args.out.resolve()}\ntrain: train/images\nval: valid/images\nnames:\n  0: court\n")
    print(f"written: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main(sys.argv[1:])
