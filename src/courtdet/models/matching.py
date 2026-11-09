"""IoU-based matching of boxes to ground truth, and sampling of training boxes.

Faster R-CNN needs the same two operations at both stages:

    stage        boxes to label   high_thresh  low_thresh  low-quality matches
    RPN          anchors          0.7          0.3         yes
    RoI head     proposals        0.5          0.5         no   (standard values)

1. `match_boxes` gives every box a label (positive / negative / ignored) and the
   index of the ground-truth box it is matched to.
2. `sample_indices` keeps a small, balanced subset of the labelled boxes. Training
   on all of them would be dominated by negatives: an image holds a few courts
   and tens of thousands of background anchors.

Boxes are (x1, y1, x2, y2), see the convention in anchors.py.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import box_iou

POSITIVE = 1  # "there is a court"
NEGATIVE = 0  # background
IGNORE = -1  # ambiguous box: contributes nothing to the loss


def match_boxes(
    boxes: Tensor,
    gt_boxes: Tensor,
    high_thresh: float,
    low_thresh: float,
    allow_low_quality_matches: bool = False,
) -> tuple[Tensor, Tensor]:
    """Label each box from its best IoU with the ground-truth boxes.

    Rules, with best_iou = max over ground-truth boxes of IoU(box, gt):
        best_iou >= high_thresh              -> POSITIVE
        best_iou <  low_thresh               -> NEGATIVE
        low_thresh <= best_iou < high_thresh -> IGNORE (empty if low == high)
    If `allow_low_quality_matches`, additionally, for each ground-truth box, the
    box(es) with the highest IoU become POSITIVE, even below `high_thresh`, as
    long as that IoU is > 0. This guarantees every court has at least one
    positive anchor. The `> 0` guard matters: a ground truth overlapping no
    anchor at all must not turn every anchor positive.

    Args:
        boxes: [N, 4] anchors or proposals.
        gt_boxes: [M, 4] ground-truth boxes of ONE image (M may be 0).

    Returns:
        labels: [N] long tensor with values POSITIVE / NEGATIVE / IGNORE.
        matched_idx: [N] long tensor, index in `gt_boxes` of the best-overlapping
            ground truth. Only meaningful where labels == POSITIVE.
            Known subtlety (shared with torchvision): for a box forced positive
            by the low-quality rule, this is the ground truth with which it
            overlaps most, which is not necessarily the one that forced it.
    """
    assert low_thresh <= high_thresh
    n = boxes.shape[0]
    if gt_boxes.shape[0] == 0:  # image without any court: everything is background
        zeros = torch.zeros(n, dtype=torch.long, device=boxes.device)
        return zeros, zeros.clone()

    iou = box_iou(boxes, gt_boxes)  # [N, M]
    best_iou, matched_idx = iou.max(dim=1)  # best ground truth for each box

    labels = torch.full((n,), IGNORE, dtype=torch.long, device=boxes.device)
    labels[best_iou < low_thresh] = NEGATIVE
    labels[best_iou >= high_thresh] = POSITIVE

    if allow_low_quality_matches:
        best_per_gt = iou.max(dim=0).values  # [M] best IoU reached by each ground truth
        is_best_for_some_gt = (iou == best_per_gt[None, :]) & (best_per_gt[None, :] > 0)
        labels[is_best_for_some_gt.any(dim=1)] = POSITIVE

    return labels, matched_idx


def sample_indices(
    labels: Tensor, num_samples: int, positive_fraction: float
) -> tuple[Tensor, Tensor]:
    """Pick at most `num_samples` boxes of ONE image, at most `positive_fraction` positive.

    Negatives fill the remaining budget. If there are fewer positives (or negatives)
    than requested, we simply take all there are. IGNORE boxes are never sampled.

    Must be called once per image: sampling over a whole batch at once lets an
    image with many positives take the budget of the others.

    Returns:
        (positive_indices, negative_indices), 1-D long tensors of indices into `labels`.
    """
    pos = (labels == POSITIVE).nonzero().flatten()
    neg = (labels == NEGATIVE).nonzero().flatten()

    num_pos = min(int(num_samples * positive_fraction), pos.numel())
    num_neg = min(num_samples - num_pos, neg.numel())

    pos = pos[torch.randperm(pos.numel(), device=pos.device)[:num_pos]]
    neg = neg[torch.randperm(neg.numel(), device=neg.device)[:num_neg]]
    return pos, neg
