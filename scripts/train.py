"""Train the Faster R-CNN.

    python scripts/train.py --config configs/default.yaml --out runs/baseline
    python scripts/train.py --out runs/lr3e-4 --set train.lr=3e-4 --set train.epochs=1 --max-steps 200
    python scripts/train.py --out runs/baseline --resume

--set takes dotted keys of the YAML file; values are parsed as YAML (3e-4, true, [1, 2], null).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from courtdet.config import load_config
from courtdet.engine.train import fit


def parse_set(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="YAML file (default: %(default)s)")
    parser.add_argument("--out", required=True, help="run folder (config.yaml, history.csv, last.pt, best.pt)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.lr=3e-4")
    parser.add_argument("--resume", action="store_true", help="continue from OUT/last.pt with OUT/config.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="batches per epoch (smoke tests, lr comparisons)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    # On resume, the run's own config.yaml is the reference, not --config.
    config_path = out / "config.yaml" if args.resume else args.config
    config = load_config(config_path, parse_set(args.set))
    print(f"config: {config_path}" + (f", overrides {args.set}" if args.set else ""), flush=True)
    fit(config, out, resume=args.resume, max_steps=args.max_steps, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main(sys.argv[1:])
