"""Evaluate a checkpoint: AP@0.5 and AP@[.5:.95] on whole images, with bootstrap intervals.

    python scripts/evaluate.py --checkpoint runs/baseline/best.pt                 # valid split
    python scripts/evaluate.py --checkpoint runs/a/best.pt --compare runs/b/eval_valid_best.pt

Writes next to the checkpoint:
    eval_<split>_<checkpoint name>.json   metrics per subset (all, A, B, B_1-3, B_4+)
    eval_<split>_<checkpoint name>.pt     raw detections (to compare models later without re-running)
With corrected ground truth (--corrections, by default the one of the checkpoint config), the files
are named eval_<split>_<checkpoint name>_corrected.* so that the Roboflow-GT results are kept.

Use `valid` for every choice (lr, checkpoint, thresholds). Run `test` only once, at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from courtdet.config import Config, from_dict
from courtdet.data.corrections import load_records
from courtdet.engine.evaluate import match_split, paired_bootstrap, predict_split, report
from courtdet.engine.train import select_device
from courtdet.models.faster_rcnn import FasterRCNN


def load_model(checkpoint: Path, device: torch.device) -> tuple[FasterRCNN, Config]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = from_dict(Config, ckpt["config"])
    config.model.pretrained_backbone = False  # the weights come from the checkpoint: no download
    model = FasterRCNN(config.model)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), config


def format_table(metrics: dict) -> str:
    lines = [f"{'subset':<7} {'images':>6} {'courts':>6}   {'AP@0.5 [95% CI]':<22} {'AP@[.5:.95] [95% CI]':<22}"]
    for name, m in metrics.items():
        if "ap50" in m:
            ap50 = f"{m['ap50']:.3f} [{m['ap50_lo']:.3f}-{m['ap50_hi']:.3f}]"
            ap = f"{m['ap']:.3f} [{m['ap_lo']:.3f}-{m['ap_hi']:.3f}]"
        else:
            ap50 = ap = "n/a (no court)"
        lines.append(f"{name:<7} {m['n_images']:>6} {m['n_gt']:>6}   {ap50:<22} {ap:<22}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--root", default=None, help="dataset root (default: the one in the checkpoint config)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--corrections", default=None,
                        help="corrected boxes (step 2 file); default: data.corrections of the checkpoint config")
    parser.add_argument("--decisions", default=None,
                        help="step 1 file, drops uncorrected multiple/unsure A images; default: from the config")
    parser.add_argument("--compare", type=Path, default=None, help="detections .pt of another model, same split")
    args = parser.parse_args(argv)

    device = select_device(args.device)
    model, config = load_model(args.checkpoint, device)
    root = Path(args.root or config.data.root)
    corrections = args.corrections or config.data.corrections
    decisions = args.decisions or config.data.decisions
    records = load_records(root, args.split, corrections=corrections, decisions=decisions)
    print(f"ground truth: {'corrected (' + corrections + ')' if corrections else 'Roboflow export'}"
          f"{', decisions ' + decisions if decisions else ''}; {len(records)} images", flush=True)

    t0 = time.time()
    detections = predict_split(model, root / args.split, records, device, min_size=config.data.crop_size)
    seconds = time.time() - t0
    print(f"inference: {len(records)} images in {seconds:.1f}s on {device}", flush=True)

    results = match_split(detections, records)
    metrics = report(results, records, n_boot=args.n_boot, seed=args.seed)
    print(format_table(metrics))

    stem = args.checkpoint.parent / f"eval_{args.split}_{args.checkpoint.stem}{'_corrected' if corrections else ''}"
    payload = {"checkpoint": str(args.checkpoint), "split": args.split, "root": str(root),
               "corrections": corrections, "decisions": decisions,
               "n_boot": args.n_boot, "seed": args.seed, "inference_seconds": round(seconds, 1),
               "metrics": metrics}
    if args.compare is not None:
        other = torch.load(args.compare, weights_only=False)
        if other["split"] != args.split:
            raise SystemExit(f"--compare holds the {other['split']!r} split, not {args.split!r}")
        diff = paired_bootstrap(results, match_split(other["detections"], records), n_boot=args.n_boot, seed=args.seed)
        payload["compare"] = {"other": str(args.compare), "this_minus_other": diff}
        print(f"this - other: AP@0.5 {diff['ap50']:+.3f} [{diff['ap50_lo']:+.3f} {diff['ap50_hi']:+.3f}]"
              f"   AP@[.5:.95] {diff['ap']:+.3f} [{diff['ap_lo']:+.3f} {diff['ap_hi']:+.3f}]")

    Path(f"{stem}.json").write_text(json.dumps(payload, indent=2))
    torch.save({"checkpoint": str(args.checkpoint), "split": args.split, "detections": detections}, f"{stem}.pt")
    print(f"written: {stem}.json, {stem}.pt")


if __name__ == "__main__":
    main(sys.argv[1:])
