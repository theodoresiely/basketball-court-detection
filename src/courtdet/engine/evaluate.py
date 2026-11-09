"""Detection metrics: COCO-style average precision, bootstrap confidence intervals,
and whole-image inference.

AP, step by step (one class, so mAP = AP):
  1. per image, detections sorted by score (at most `max_dets`); each one is matched to
     the not-yet-matched ground truth with the highest IoU, if that IoU >= t -> true
     positive (TP), otherwise false positive (FP). A second box on an already found
     court is therefore a FP. Matching is image-local, so it is done once per image.
  2. all detections of all images sorted by score; cumulative TP and FP give, at each
     rank, precision = TP / (TP + FP) and recall = TP / number of ground truths;
  3. precision is replaced by its decreasing envelope (best precision at recall >= r),
     read at 101 recall levels 0, 0.01, ..., 1 (0 beyond the highest recall reached);
     AP = mean of these 101 values.
AP@0.5 uses t = 0.5; AP@[.5:.95] averages t = 0.50, 0.55, ..., 0.95.

These rules were read in pycocotools 2.0.11 (cocoeval.py: evaluateImg, accumulate,
Params.setDetParams) and the result is checked against pycocotools in the tests.

Bootstrap: images are resampled WITH replacement (an image is the independent unit:
its courts and its false alarms go together), the AP is recomputed on each resample, and
the 2.5 / 97.5 percentiles give a 95 % interval. It measures the uncertainty due to the
finite evaluation set, not the training randomness (another seed = another model).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

from courtdet.data.dataset import IMAGENET_MEAN, IMAGENET_STD

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # = pycocotools Params.iouThrs
RECALL_LEVELS = np.linspace(0.0, 1.0, 101)  # = pycocotools Params.recThrs
MAX_DETS = 100  # = the maxDets of pycocotools' summary AP


# ---------------------------------------------------------------- AP


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix [N, M] of (x1, y1, x2, y2) boxes, no +1 (same convention as torchvision.ops)."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    w = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(min=0)
    h = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(min=0)
    inter = w * h
    return inter / (area_a[:, None] + area_b[None, :] - inter)


@dataclass
class ImageResult:
    """Detections of one image, already matched: everything the AP needs, per image."""

    scores: np.ndarray  # [D] sorted by decreasing score, D <= max_dets
    tp: np.ndarray  # [T, D] bool: is detection d a TP at IoU threshold t?
    n_gt: int


def match_image(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    max_dets: int = MAX_DETS,
) -> ImageResult:
    """Greedy matching of one image's detections to its ground truth (COCO rule)."""
    det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(det_scores, dtype=np.float64), kind="mergesort")[:max_dets]
    det_boxes, scores = det_boxes[order], np.asarray(det_scores, dtype=np.float64)[order]

    n_t, n_d, n_g = len(iou_thresholds), len(det_boxes), len(gt_boxes)
    tp = np.zeros((n_t, n_d), dtype=bool)
    if n_d and n_g:
        ious = box_iou_np(det_boxes, gt_boxes)
        for ti, t in enumerate(iou_thresholds):
            threshold = min(t, 1 - 1e-10)  # as pycocotools
            matched = np.zeros(n_g, dtype=bool)
            for d in range(n_d):
                candidates = np.where(matched, -np.inf, ious[d])
                # highest IoU; on exact ties pycocotools keeps the LAST ground truth
                g = n_g - 1 - int(np.argmax(candidates[::-1]))
                if candidates[g] >= threshold:
                    matched[g] = True
                    tp[ti, d] = True
    return ImageResult(scores=scores, tp=tp, n_gt=n_g)


def average_precision(results: Sequence[ImageResult]) -> np.ndarray:
    """AP at each IoU threshold, [T]. NaN if there is no ground truth at all (undefined)."""
    n_gt = sum(r.n_gt for r in results)
    n_t = results[0].tp.shape[0] if results else len(IOU_THRESHOLDS)
    if n_gt == 0:
        return np.full(n_t, np.nan)
    scores = np.concatenate([r.scores for r in results])
    tp = np.concatenate([r.tp for r in results], axis=1)
    order = np.argsort(-scores, kind="mergesort")  # stable, as pycocotools
    tp = tp[:, order]

    tp_sum = np.cumsum(tp, axis=1, dtype=np.float64)
    fp_sum = np.cumsum(~tp, axis=1, dtype=np.float64)
    ap = np.zeros(n_t)
    for t in range(n_t):
        recall = tp_sum[t] / n_gt
        precision = tp_sum[t] / (tp_sum[t] + fp_sum[t] + np.spacing(1))
        precision = np.maximum.accumulate(precision[::-1])[::-1]  # decreasing envelope
        idx = np.searchsorted(recall, RECALL_LEVELS, side="left")
        sampled = np.zeros(len(RECALL_LEVELS))
        reached = idx < len(precision)  # recall levels never reached keep precision 0
        sampled[reached] = precision[idx[reached]]
        ap[t] = sampled.mean()
    return ap


def summarize(ap: np.ndarray) -> dict[str, float]:
    """{"ap50": AP at IoU 0.5, "ap": mean over 0.5:0.95} from the [T] vector of IOU_THRESHOLDS."""
    return {"ap50": float(ap[0]), "ap": float(ap.mean())}


# ---------------------------------------------------------------- bootstrap


def bootstrap(
    results: Sequence[ImageResult], n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> dict[str, float]:
    """Point estimates and percentile intervals of AP@0.5 and AP@[.5:.95], resampling images.

    Resamples without any ground truth (AP undefined) are dropped; their number is returned.
    """
    point = summarize(average_precision(results))
    rng = np.random.default_rng(seed)
    n = len(results)
    draws = {"ap50": [], "ap": []}
    for _ in range(n_boot):
        sample = [results[i] for i in rng.integers(0, n, n)]
        s = summarize(average_precision(sample))
        if not np.isnan(s["ap50"]):
            for k in draws:
                draws[k].append(s[k])
    alpha = (1 - level) / 2
    out = dict(point)
    for k, values in draws.items():
        lo, hi = np.quantile(values, [alpha, 1 - alpha]) if values else (np.nan, np.nan)
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    out["n_boot_valid"] = len(draws["ap50"])
    return out


def paired_bootstrap(
    results_a: Sequence[ImageResult],
    results_b: Sequence[ImageResult],
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Difference AP(a) - AP(b) on the SAME images, with a paired bootstrap interval.

    Both lists must describe the same images in the same order. Each resample uses the same
    image indices for both models: the hard images are shared, so the difference is far less
    noisy than either AP. If the interval excludes 0, the gap is significant on this image set.
    """
    if len(results_a) != len(results_b) or any(a.n_gt != b.n_gt for a, b in zip(results_a, results_b)):
        raise ValueError("the two result lists must cover the same images in the same order")
    diff = lambda ra, rb: {k: summarize(average_precision(ra))[k] - summarize(average_precision(rb))[k]
                           for k in ("ap50", "ap")}
    out = diff(results_a, results_b)
    rng = np.random.default_rng(seed)
    n = len(results_a)
    draws = {"ap50": [], "ap": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = diff([results_a[i] for i in idx], [results_b[i] for i in idx])
        if not np.isnan(d["ap50"]):
            for k in draws:
                draws[k].append(d[k])
    alpha = (1 - level) / 2
    for k, values in draws.items():
        out[f"{k}_lo"], out[f"{k}_hi"] = (float(q) for q in np.quantile(values, [alpha, 1 - alpha]))
    out["n_boot_valid"] = len(draws["ap50"])
    return out


# ---------------------------------------------------------------- subsets


def subsets(records: Sequence[dict]) -> dict[str, list[int]]:
    """Image indices of each reported subset.

    all; A; B; B_1-3 / B_4+: B images with 1-3 / >= 4 courts (criterion of 28/10 for the
    crop-rule bias). B images without any court only enter "all" and "B" (alone they have
    no ground truth, so no AP).
    """
    out = {"all": list(range(len(records))), "A": [], "B": [], "B_1-3": [], "B_4+": []}
    for i, r in enumerate(records):
        out[r["source"]].append(i)
        n = len(r["boxes"])
        if r["source"] == "B" and 1 <= n <= 3:
            out["B_1-3"].append(i)
        elif r["source"] == "B" and n >= 4:
            out["B_4+"].append(i)
    return out


def report(results: Sequence[ImageResult], records: Sequence[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Metrics with bootstrap intervals for every subset, plus their sizes."""
    out = {}
    for name, idx in subsets(records).items():
        part = [results[i] for i in idx]
        entry = {"n_images": len(part), "n_gt": int(sum(r.n_gt for r in part))}
        if entry["n_gt"]:
            entry |= bootstrap(part, n_boot=n_boot, seed=seed)
        out[name] = entry
    return out


# ---------------------------------------------------------------- inference


def load_image(path: str | Path) -> Tensor:
    """Whole image as float [3, H, W] in [0, 1]."""
    with Image.open(path) as img:
        return pil_to_tensor(img.convert("RGB")).float() / 255.0


@torch.no_grad()
def predict_image(
    model, image: Tensor, device: torch.device, min_size: int = 512, size_multiple: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Detections on a WHOLE image in one pass: (boxes [D, 4] in image pixels, scores [D]).

    Same preprocessing as training: ImageNet normalisation, zero padding (= mean colour)
    on the right and bottom. Padding up to `min_size` (the training crop size) gives a
    256 x 256 source A tile the same 512 x 512 input as in training; padding to a multiple
    of the stride makes the feature map cover the image exactly. Test proposal counts are
    scaled by the padded area / min_size^2, so that a big image gets as many proposals per
    unit of area as a training-size crop. Detections are finally clipped to the REAL image:
    nothing is counted in the padding.
    """
    _, h, w = image.shape
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    ph = max(min_size, -(-h // size_multiple) * size_multiple)
    pw = max(min_size, -(-w // size_multiple) * size_multiple)
    x = torch.nn.functional.pad((image - mean) / std, (0, pw - w, 0, ph - h), value=0.0)

    model.eval()
    det = model.detect(x[None].to(device), proposal_scale=max(1.0, ph * pw / min_size**2))[0]
    boxes, scores = det["boxes"].cpu().double().numpy(), det["scores"].cpu().double().numpy()
    boxes = boxes.clip(0, [w, h, w, h])
    keep = ((boxes[:, 2] - boxes[:, 0]) >= 1e-2) & ((boxes[:, 3] - boxes[:, 1]) >= 1e-2)
    return boxes[keep], scores[keep]


def predict_split(
    model,
    image_dir: str | Path,
    records: Sequence[dict],
    device: torch.device,
    min_size: int = 512,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Detections for every record: list of {"file_name", "boxes", "scores"} (numpy)."""
    out = []
    for i, r in enumerate(records):
        image = load_image(Path(image_dir) / r["file_name"])
        if image.shape[1:] != (r["height"], r["width"]):
            raise ValueError(f"{r['file_name']}: size {tuple(image.shape[1:])} differs from the annotation file")
        boxes, scores = predict_image(model, image, device, min_size=min_size)
        out.append({"file_name": r["file_name"], "boxes": boxes, "scores": scores})
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(records)} images")
    return out


def match_split(detections: Sequence[dict], records: Sequence[dict]) -> list[ImageResult]:
    """ImageResult per record; `detections` must be in the same order as `records`."""
    if [d["file_name"] for d in detections] != [r["file_name"] for r in records]:
        raise ValueError("detections and records do not describe the same images in the same order")
    return [match_image(d["boxes"], d["scores"], r["boxes"]) for d, r in zip(detections, records)]
