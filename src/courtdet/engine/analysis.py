"""Error analysis of a detector: WHY the AP is what it is.

An AP mixes three kinds of failure, which call for different fixes:
  - false alarms (a "court" where there is none);
  - missed courts (no detection on them, at any score);
  - badly placed boxes (the court is seen, but IoU < 0.5).
The functions below separate them. Everything works at IoU 0.5 (the AP@0.5 rule) on
detections that were already computed and saved by scripts/evaluate.py.

Definitions (thresholds are parameters, defaults below):
  - a detection is a TP if the greedy COCO matching of evaluate.match_image pairs it with
    a court; otherwise it is a FP, labelled by its best IoU with ANY court of the image:
        IoU <  0.1          background     (nothing there)
        0.1 <= IoU < 0.5    localization   (near a court, badly placed)
        IoU >= 0.5          duplicate      (on a court already taken by a better-scored box)
  - a court is, from its best IoU with ANY detection (of any score):
        IoU >= 0.5          touched
        0.1 <= IoU < 0.5    localization
        IoU <  0.1          missed
    "touched" is slightly more generous than "matched": a box can overlap two courts but
    is matched to only one. The matched fraction is reported too (max recall).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from courtdet.engine.evaluate import ImageResult, average_precision, box_iou_np, match_image

MATCH_IOU = 0.5
BACKGROUND_IOU = 0.1
SIZE_EDGES = (32.0, 64.0, 128.0)  # sqrt(area) in native pixels -> <32, 32-64, 64-128, >=128
SIZE_LABELS = ("<32", "32-64", "64-128", ">=128")
RATIO_EDGES = (2 / 3, 3 / 2)  # width / height -> tall (<0.67), square-ish, wide (>1.5)
RATIO_LABELS = ("tall", "square", "wide")


def image_type(file_name: str) -> str:
    """Finer grouping than the A / B source: A, P####, map tile, screenshot, other."""
    if file_name.startswith("basketball_court"):
        return "A"
    if re.match(r"P\d{4}_png", file_name):
        return "P####"
    if re.match(r"\d+-\d+-\d+_jpg", file_name):
        return "tile"
    if file_name.startswith("Screenshot"):
        return "screenshot"
    return "other"


def size_bins(boxes: np.ndarray, edges: Sequence[float] = SIZE_EDGES) -> np.ndarray:
    """Index of the size bin of each box, from sqrt(area)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    side = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    return np.searchsorted(np.asarray(edges), side, side="right")


def ratio_bins(boxes: np.ndarray, edges: Sequence[float] = RATIO_EDGES) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    ratio = (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1])
    return np.searchsorted(np.asarray(edges), ratio, side="right")


def classify_detections(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    gt_boxes: np.ndarray,
    match_iou: float = MATCH_IOU,
    background_iou: float = BACKGROUND_IOU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label every kept detection of one image.

    Returns (boxes, scores, labels), sorted by decreasing score and truncated exactly as
    match_image does, labels in {"tp", "duplicate", "localization", "background"}.
    """
    det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    det_scores = np.asarray(det_scores, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    result = match_image(det_boxes, det_scores, gt_boxes, iou_thresholds=[match_iou])
    order = np.argsort(-det_scores, kind="mergesort")[: len(result.scores)]  # same order as match_image
    boxes = det_boxes[order]

    best = box_iou_np(boxes, gt_boxes).max(axis=1) if len(gt_boxes) and len(boxes) else np.zeros(len(boxes))
    labels = np.where(best < background_iou, "background", np.where(best < match_iou, "localization", "duplicate"))
    labels = np.where(result.tp[0], "tp", labels)
    return boxes, result.scores, labels


def classify_courts(
    det_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    match_iou: float = MATCH_IOU,
    background_iou: float = BACKGROUND_IOU,
) -> tuple[np.ndarray, np.ndarray]:
    """(best IoU with any detection [M], label in {"touched", "localization", "missed"}) per court."""
    det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    best = box_iou_np(gt_boxes, det_boxes).max(axis=1) if len(det_boxes) and len(gt_boxes) else np.zeros(len(gt_boxes))
    labels = np.where(best >= match_iou, "touched", np.where(best >= background_iou, "localization", "missed"))
    return best, labels


def ap50_without(classified: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], n_gts: Sequence[int],
                 remove: Sequence[str] = ()) -> float:
    """AP@0.5 after deleting the detections whose label is in `remove` (an "oracle" fix).

    Deleting UNMATCHED detections does not change which detections are matched (they took no
    court), so the TP flags of the remaining ones are unchanged: the gain is exactly the AP lost
    to that kind of false alarm.
    """
    results = []
    for (_, scores, labels), n_gt in zip(classified, n_gts):
        keep = ~np.isin(labels, list(remove))
        results.append(ImageResult(scores=scores[keep], tp=(labels[keep] == "tp")[None, :], n_gt=n_gt))
    return float(average_precision(results)[0])


def max_recall(classified: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], n_gts: Sequence[int]) -> float:
    """Recall reached when ALL kept detections are accepted (matched courts / courts)."""
    n = sum(n_gts)
    return sum(int((labels == "tp").sum()) for _, _, labels in classified) / n if n else float("nan")


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    return [float(v) for v in np.quantile(values, qs)] if len(values) else []
