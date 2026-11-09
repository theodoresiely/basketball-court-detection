"""Training loop: epochs, validation loss, logging, checkpoints, resume.

Every run writes to its own folder:
    config.yaml   the exact configuration of the run
    history.csv   one line per epoch: the 4 loss terms + total, train and val, lr, grad norm, time
    last.pt       state at the end of the last epoch (to resume an interrupted run)
    best.pt       state of the epoch with the best validation score, `train.select_metric`
                  (default: AP@[.5:.95] on the whole validation images)
    best_loss.pt  state of the epoch with the lowest validation total loss (kept for comparison)

The validation loss is NOT the final metric: near the optimum it disagrees with the mAP.
It monitors training and drives the lr scheduler (ReduceLROnPlateau).
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from courtdet.config import Config, save_config, to_dict
from courtdet.data.augment import make_train_transform
from courtdet.data.corrections import load_records
from courtdet.data.dataset import CourtDataset, collate_fn
from courtdet.engine.evaluate import average_precision, match_split, predict_split, summarize
from courtdet.models.faster_rcnn import FasterRCNN

LOSS_KEYS = ("rpn_cls", "rpn_reg", "roi_cls", "roi_reg", "total")


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """cfg: TrainConfig. Frozen parameters are left out (they would get no gradient anyway)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r} (expected 'adam' or 'sgd')")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience
    )


def _to_device(images: Tensor, targets: list[dict], device: torch.device) -> tuple[Tensor, list[Tensor]]:
    return images.to(device), [t["boxes"].to(device) for t in targets]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
    max_steps: int | None = None,
    log_every: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, float]:
    """One pass over `loader` (or `max_steps` batches). Returns the mean of each loss term
    and of the gradient norm (computed before clipping)."""
    model.train()
    sums = dict.fromkeys(LOSS_KEYS + ("grad_norm",), 0.0)
    steps = 0
    for images, targets, _ in loader:
        if max_steps is not None and steps >= max_steps:
            break
        images, gt_boxes = _to_device(images, targets, device)
        losses = model(images, gt_boxes)
        optimizer.zero_grad()
        losses["total"].backward()
        params = [p for p in model.parameters() if p.grad is not None]
        # max_norm = inf: only measures the norm. Measuring it on every step makes a
        # gradient explosion visible in history.csv before deciding to clip.
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip_norm or math.inf)
        optimizer.step()

        for k in LOSS_KEYS:
            sums[k] += losses[k].item()
        sums["grad_norm"] += grad_norm.item()
        steps += 1
        if log_every and steps % log_every == 0:
            log(f"  iter {steps}/{len(loader)}  " + "  ".join(f"{k} {losses[k].item():.4f}" for k in LOSS_KEYS))
    return {k: v / max(1, steps) for k, v in sums.items()}


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Mean loss terms over `loader`, with a FIXED seed.

    Crops, anchor sampling and RoI sampling are random: without a fixed seed, the
    validation loss would move between epochs even for an unchanged model. fork_rng
    restores the training random state afterwards, so validating does not change the
    course of training. DataLoader workers take their seeds from the main generator
    when the iterator is created, so seeding before the loop also fixes them.
    """
    model.eval()
    sums = dict.fromkeys(LOSS_KEYS, 0.0)
    steps = 0
    accel = [] if device.type == "cpu" else [device]
    with torch.random.fork_rng(devices=accel, device_type=device.type if accel else "cuda"):
        torch.manual_seed(seed)
        for images, targets, _ in loader:
            if max_steps is not None and steps >= max_steps:
                break
            losses = model(*_to_device(images, targets, device))
            for k in LOSS_KEYS:
                sums[k] += losses[k].item()
            steps += 1
    return {k: v / max(1, steps) for k, v in sums.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, history: list[dict], config: Config) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,  # number of completed epochs
            "history": history,
            "config": to_dict(config),
            "rng_cpu": torch.get_rng_state(),  # so that a resumed run draws the same crops
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu") -> tuple[int, list[dict]]:
    """Restore the states in place. Returns (completed epochs, history)."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    # map_location moved every tensor to the device, the CPU RNG state included: it must come back to the CPU.
    torch.set_rng_state(ckpt["rng_cpu"].cpu())
    return ckpt["epoch"], ckpt["history"]


def write_history(path, history: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        # Union of the columns in order of appearance: a run resumed with a newer version may log more.
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in history for k in row)))
        writer.writeheader()
        writer.writerows(history)


def make_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    d = config.data
    crop = dict(crop_size=d.crop_size, tol=d.tol, p_pos=d.p_pos, max_tries=d.max_tries)
    review = dict(corrections=d.corrections, decisions=d.decisions)
    train_records = load_records(d.root, d.train_split, **review, source_a_fraction=d.source_a_fraction,
                                 seed=d.subset_seed)
    augment = make_train_transform(d.aug_dihedral, d.aug_brightness, d.aug_contrast, d.aug_saturation, d.aug_gray)
    train_ds = CourtDataset(d.root, d.train_split, **crop, records=train_records, transforms=augment)
    val_ds = CourtDataset(d.root, d.val_split, **crop, records=load_records(d.root, d.val_split, **review))
    common = dict(batch_size=d.batch_size, num_workers=d.num_workers, collate_fn=collate_fn,
                  persistent_workers=False)
    return DataLoader(train_ds, shuffle=True, **common), DataLoader(val_ds, shuffle=False, **common)


SELECT_METRICS = ("ap", "ap50", "loss")


def evaluate_map(model: FasterRCNN, dataset: CourtDataset, device: torch.device, min_size: int) -> dict[str, float]:
    """AP@0.5 and AP@[.5:.95] of `model` on the WHOLE images of `dataset` (no crops, no randomness).

    Same inference and matching as scripts/evaluate.py, without the bootstrap. NaN without any court.
    """
    detections = predict_split(model, dataset.image_dir, dataset.records, device, min_size=min_size, log=lambda s: None)
    return summarize(average_precision(match_split(detections, dataset.records)))


def selection_score(row: dict, metric: str) -> float:
    """Higher is better: AP as is, minus the loss. NaN when the metric was not computed this epoch."""
    return -row["val_total"] if metric == "loss" else row[f"val_{metric}"]


def fit(
    config: Config,
    out_dir: str | Path,
    resume: bool = False,
    max_steps: int | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Train `config` into `out_dir`. resume=True continues from out_dir/last.pt.

    max_steps limits the batches per epoch (train and val): smoke tests, lr comparisons.
    """
    out_dir = Path(out_dir)
    if not resume and (out_dir / "history.csv").exists():
        raise FileExistsError(f"{out_dir} already holds a run: choose another folder or resume")
    out_dir.mkdir(parents=True, exist_ok=True)

    t = config.train
    torch.manual_seed(t.seed)
    device = select_device(t.device)
    model = FasterRCNN(config.model).to(device)
    optimizer = build_optimizer(model, t)
    scheduler = build_scheduler(optimizer, t)
    train_loader, val_loader = make_loaders(config)

    start_epoch, history = 0, []
    if resume:
        start_epoch, history = load_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, map_location=device)
        log(f"resumed from {out_dir / 'last.pt'} after epoch {start_epoch}")
    else:
        save_config(config, out_dir / "config.yaml")
    if t.select_metric not in SELECT_METRICS:
        raise ValueError(f"train.select_metric must be one of {SELECT_METRICS}, got {t.select_metric!r}")
    if t.select_metric != "loss" and t.eval_every <= 0:
        raise ValueError(f"train.select_metric={t.select_metric!r} needs train.eval_every >= 1")
    # Best scores so far (higher is better), recomputed from the history when resuming.
    best_loss = max((-h["val_total"] for h in history), default=-math.inf)
    best = max((x for h in history if not math.isnan(x := selection_score(h, t.select_metric))), default=-math.inf)
    log(f"device {device}, {len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")

    for epoch in range(start_epoch, t.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train = train_one_epoch(model, train_loader, optimizer, device, t.grad_clip_norm, max_steps, t.log_every, log)
        val = evaluate_loss(model, val_loader, device, t.val_seed, max_steps)
        scheduler.step(val["total"])

        row = {"epoch": epoch + 1, "lr": lr, "time_s": round(time.time() - t0, 1)}
        row |= {f"train_{k}": v for k, v in train.items()}
        row |= {f"val_{k}": v for k, v in val.items()}
        ap = {"ap50": math.nan, "ap": math.nan}  # every row has the same columns (CSV)
        if t.eval_every > 0 and (epoch + 1) % t.eval_every == 0:
            t1 = time.time()
            ap = evaluate_map(model, val_loader.dataset, device, config.data.crop_size)
            row["eval_time_s"] = round(time.time() - t1, 1)
        row.setdefault("eval_time_s", math.nan)
        row |= {f"val_{k}": v for k, v in ap.items()}
        row["time_s"] = round(time.time() - t0, 1)
        history.append(row)
        write_history(out_dir / "history.csv", history)
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch + 1, history, config)
        marks = []
        if -val["total"] > best_loss:
            best_loss = -val["total"]
            save_checkpoint(out_dir / "best_loss.pt", model, optimizer, scheduler, epoch + 1, history, config)
            marks.append("best loss")
        score = selection_score(row, t.select_metric)
        if not math.isnan(score) and score > best:
            best = score
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch + 1, history, config)
            marks.append(f"best {t.select_metric}")
        log(f"epoch {epoch + 1}/{t.epochs}  train {train['total']:.4f}  val {val['total']:.4f}  "
            f"AP50 {row['val_ap50']:.3f}  AP {row['val_ap']:.3f}  lr {lr:.2e}  {row['time_s']}s"
            + (f"  ({', '.join(marks)})" if marks else ""))
    return history
