"""Region Proposal Network: prediction head, training loss and proposal generation.

The RPN looks at the backbone feature map and, for each of the k anchors centred on
each cell, predicts:
  - an objectness logit: "is there a court here?"
  - 4 deltas (tx, ty, tw, th): how to move/resize the anchor (see box_coder.py).

The IoU with the ground truth is NOT used by the head (there is no ground truth at
inference time). It is only used by the loss, to build the targets (matching.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import nms

from courtdet.models.box_coder import clip_boxes, decode, encode
from courtdet.models.matching import match_boxes, sample_indices


def flatten_predictions(x: Tensor, per_anchor: int) -> Tensor:
    """[B, k * per_anchor, H, W] -> [B, H * W * k, per_anchor].

    The output order is (row i, column j, anchor a), i.e. flat index (i * W + j) * k + a,
    exactly the order of `anchors.grid_anchors`. If the two orders disagreed, nothing
    would crash and every prediction would be compared to the wrong anchor.
    """
    b, c, h, w = x.shape
    assert c % per_anchor == 0
    k = c // per_anchor
    return x.permute(0, 2, 3, 1).reshape(b, h * w * k, per_anchor)


class RPNHead(nn.Module):
    """3x3 conv shared by all anchors, then two 1x1 convs (objectness and deltas).

    The 3x3 conv lets each anchor "see" its neighbourhood (a court is recognised by
    its surroundings and its lines, not by a single feature vector). The 1x1 heads
    are then applied independently at every position.

    Objectness uses ONE logit per anchor (sigmoid + binary cross-entropy). The
    notebook used two outputs + softmax; with two classes it is the same model
    (softmax over 2 logits = sigmoid of their difference) with half the parameters.
    """

    def __init__(self, in_channels: int, num_anchors: int, mid_channels: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, num_anchors, kernel_size=1)
        self.bbox_deltas = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)
        # Small weights, zero biases: at the start every anchor has p(court) ~ 0.5 and
        # deltas ~ 0, so the first losses and gradients are moderate and uniform.
        for layer in (self.conv, self.cls_logits, self.bbox_deltas):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """features: [B, C, H, W] -> objectness [B, H*W*k], deltas [B, H*W*k, 4]."""
        x = F.relu(self.conv(features))
        objectness = flatten_predictions(self.cls_logits(x), 1).squeeze(-1)
        deltas = flatten_predictions(self.bbox_deltas(x), 4)
        return objectness, deltas


def rpn_loss(
    objectness: Tensor,
    deltas: Tensor,
    anchors: Tensor,
    gt_boxes_list: list[Tensor],
    pos_iou_thresh: float = 0.7,
    neg_iou_thresh: float = 0.3,
    num_samples: int = 256,
    positive_fraction: float = 0.5,
    smooth_l1_beta: float = 1.0 / 9.0,
) -> tuple[Tensor, Tensor]:
    """Classification and regression losses of the RPN, returned SEPARATELY.

    The weight between the two (lambda) is applied by the caller, so that it is a
    config parameter, not a constant buried here.

    For each image: match anchors to ground truth, sample <= `num_samples` anchors
    (<= `positive_fraction` positive), then
      classification: binary cross-entropy on the sampled anchors (target 1 / 0),
      regression:     Smooth L1 between predicted and target deltas, positives only,
                      target = encode(anchor, matched ground truth).

    Normalisation: both terms are divided by the total number of sampled anchors of
    the batch, so the value does not depend on the batch size nor on how many
    positives a batch happens to contain (the notebook divided a batch-wide sum by
    a constant 256, which made the loss scale with the batch size).

    Args:
        objectness: [B, N] logits. deltas: [B, N, 4]. anchors: [N, 4].
        gt_boxes_list: B tensors [M_i, 4] (M_i may be 0).
        smooth_l1_beta: width of the quadratic zone of Smooth L1. Deltas are small
            numbers (typically < 0.3), so with beta = 1 the loss is almost always
            purely quadratic and loses its robustness to outliers. 1/9 is torchvision's
            value for the RPN (checked in torchvision 0.29.1, models/detection/rpn.py).
    """
    sampled_logits, sampled_targets = [], []
    pred_deltas, target_deltas = [], []

    for obj_i, deltas_i, gt in zip(objectness, deltas, gt_boxes_list):
        labels, matched_idx = match_boxes(
            anchors, gt, pos_iou_thresh, neg_iou_thresh, allow_low_quality_matches=True
        )
        pos, neg = sample_indices(labels, num_samples, positive_fraction)

        sampled_logits.append(obj_i[torch.cat([pos, neg])])
        sampled_targets.append(
            torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(obj_i)
        )
        pred_deltas.append(deltas_i[pos])
        target_deltas.append(encode(anchors[pos], gt[matched_idx[pos]]))

    logits = torch.cat(sampled_logits)
    cls_loss = F.binary_cross_entropy_with_logits(logits, torch.cat(sampled_targets))
    reg_loss = F.smooth_l1_loss(
        torch.cat(pred_deltas),
        torch.cat(target_deltas),
        beta=smooth_l1_beta,
        reduction="sum",
    ) / max(1, logits.numel())
    return cls_loss, reg_loss


@torch.no_grad()
def generate_proposals(
    objectness: Tensor,
    deltas: Tensor,
    anchors: Tensor,
    image_size: tuple[int, int],
    pre_nms_top_n: int = 2000,
    post_nms_top_n: int = 1000,
    nms_thresh: float = 0.7,
    min_size: float = 1e-3,
) -> tuple[list[Tensor], list[Tensor]]:
    """Turn the RPN outputs into a short list of candidate boxes per image.

    Used at inference AND during training (the RoI head is trained on these boxes).
    Per image:
      1. keep the `pre_nms_top_n` anchors with the highest objectness (NMS cost grows
         quickly with the number of boxes: with layer3 (stride 16) and 18 anchors per cell,
         a 512 crop has 18,432 anchors and a 2048 x 2048 image 294,912);
      2. decode their deltas into boxes;
      3. clip the boxes to the image, drop the ones that became (almost) empty;
      4. NMS: repeatedly keep the best-scoring box and discard the boxes whose IoU with
         it exceeds `nms_thresh` (the anchors around one court all predict nearly the
         same box; we want one proposal per court, not hundreds);
      5. keep the `post_nms_top_n` best.

    Runs under no_grad: proposals are CONSTANTS for the next stage. The RPN is trained
    by its own loss only; the RoI head's targets, computed from the proposals, must not
    send gradients back into the RPN (the notebook did not detach them).

    Args:
        objectness: [B, N] logits. deltas: [B, N, 4]. anchors: [N, 4].
        image_size: (height, width).

    Returns:
        (boxes, scores): lists of B tensors, [P_i, 4] and [P_i], sorted by decreasing
        score. `scores` are probabilities (sigmoid of the logits).
    """
    all_boxes, all_scores = [], []
    for obj_i, deltas_i in zip(objectness, deltas):
        top_logits, top_idx = obj_i.topk(min(pre_nms_top_n, obj_i.numel()))
        boxes = clip_boxes(decode(anchors[top_idx], deltas_i[top_idx]), image_size)
        scores = top_logits.sigmoid()

        big_enough = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
        boxes, scores = boxes[big_enough], scores[big_enough]

        keep = nms(boxes, scores, nms_thresh)[:post_nms_top_n]  # indices, best score first
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
    return all_boxes, all_scores
