"""RoI head (second stage of Faster R-CNN): classify and refine the RPN proposals.

For each proposal, `roi_align` cuts the matching region out of the feature map and
resamples it to a fixed roi_output_size x roi_output_size grid, whatever the size of the
box. Two fully connected layers then answer two questions:
  - classification: background or court? 2 logits, softmax;
  - regression: 4 deltas that move/resize the PROPOSAL (same box coder as the RPN).

Training (`select_training_samples` then `roi_loss`):
  1. ground-truth boxes are appended to the proposals of their image, so that the head
     sees positives even when the RPN is still bad (start of training);
  2. every proposal is labelled by its best IoU with the ground truth: >= pos_iou_thresh
     -> court, below -> background. Low and high thresholds are equal, so there is no
     "ignore" zone (see matching.match_boxes);
  3. `num_samples` RoIs per image are sampled, at most `positive_fraction` positive.

Defaults marked "torchvision" were read in the installed torchvision 0.29.1 source
(models/detection/faster_rcnn.py, roi_heads.py), not quoted from memory.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import nms, roi_align

from courtdet.models.box_coder import clip_boxes, decode, encode
from courtdet.models.matching import POSITIVE, match_boxes, sample_indices

# torchvision (roi_heads.py): regression targets of the RoI head are multiplied by these.
# Proposals are already close to the ground truth, so the raw deltas are small; the
# weights bring them closer to magnitude 1.
DEFAULT_BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)

COURT = 1  # class index of a court in the 2 logits; 0 is background


class RoIHead(nn.Module):
    """roi_align -> flatten -> FC -> ReLU -> FC -> ReLU -> (2 class logits, 4 deltas).

    The deltas are class-agnostic (one set of 4, not one per class): with a single
    object class this is the same as torchvision's per-class layout.

    Args:
        in_channels: channels of the feature map (128 after ResNet-18 layer2, 256 after layer3).
        roi_output_size: side of the roi_align output grid. 7 is the Fast R-CNN
            convention, not a demonstrated optimum (to compare with 5 on the mAP).
        spatial_scale: feature map size / image size = 1 / stride.
        sampling_ratio: roi_align takes exactly sampling_ratio x sampling_ratio bilinear
            samples per output bin, then averages them. 2 = torchvision's Faster R-CNN
            value. (<= 0 would pick ceil(roi_size / roi_output_size) per RoI: variable cost.)
        aligned: True shifts box coordinates by -0.5 cell, so that a continuous
            coordinate x (pixels, no +1 convention) lands at the right place relative to
            the cell centres. False (legacy, which torchvision's Faster R-CNN uses
            implicitly) is off by half a cell (4 px at stride 8, 8 px at stride 16) and forces every RoI to
            be at least 1 cell wide, which distorts small courts.
        hidden_dim: width of the two FC layers.
    """

    def __init__(
        self,
        in_channels: int = 128,
        roi_output_size: int = 7,
        spatial_scale: float = 1.0 / 8,
        sampling_ratio: int = 2,
        aligned: bool = True,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

        self.fc6 = nn.Linear(in_channels * roi_output_size**2, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, 2)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        # Neutral start, as for the RPN: p(court) ~ 0.5 and deltas ~ 0. The std values
        # (0.01 / 0.001) are the Detectron convention, quoted from memory (not verified).
        # fc6 / fc7 keep PyTorch's default initialisation.
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def pool(self, features: Tensor, rois: list[Tensor]) -> Tensor:
        """features [B, C, H, W], B lists of [K_i, 4] boxes (image pixels) -> [sum K_i, C, s, s]."""
        return roi_align(
            features,
            rois,
            output_size=self.roi_output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )

    def forward(self, features: Tensor, rois: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Returns class logits [K, 2] and deltas [K, 4], K = sum of the K_i, in image order."""
        x = self.pool(features, rois).flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return self.cls_score(x), self.bbox_pred(x)


@torch.no_grad()
def select_training_samples(
    proposals: list[Tensor],
    gt_boxes_list: list[Tensor],
    pos_iou_thresh: float = 0.5,
    num_samples: int = 128,
    positive_fraction: float = 0.25,
    box_weights: Sequence[float] = DEFAULT_BOX_WEIGHTS,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Build the RoIs to train on, their labels and their regression targets, per image.

    Everything here is a constant for the loss (no_grad, proposals detached): the RPN
    is trained by its own loss only.

    Args:
        proposals: B tensors [N_i, 4] (from rpn.generate_proposals).
        gt_boxes_list: B tensors [M_i, 4] (M_i may be 0), e.g. [t["boxes"] for t in targets].
        pos_iou_thresh: IoU >= this -> court, < this -> background. 0.5 = torchvision
            (box_fg_iou_thresh = box_bg_iou_thresh = 0.5); the notebook used 0.4.
        num_samples: RoIs per image. 128 = the notebook / Fast R-CNN paper value;
            torchvision's default is 512.
        positive_fraction: at most this fraction of positives (0.25 = torchvision).
            If there are not enough positives, negatives fill the budget.
        box_weights: weights of the box coder (torchvision: 10, 10, 5, 5).

    Returns:
        rois: B tensors [K_i, 4], positives first.
        labels: B long tensors [K_i], COURT (1) or background (0).
        regression_targets: B tensors [K_i, 4]; zero rows for negatives (unused).
    """
    all_rois, all_labels, all_targets = [], [], []
    for props, gt in zip(proposals, gt_boxes_list):
        gt = gt.to(props)
        candidates = torch.cat([props.detach(), gt])
        # high == low: no IGNORE zone; no low-quality matches: the appended ground
        # truths already guarantee one positive (IoU = 1) per court.
        labels, matched_idx = match_boxes(
            candidates, gt, pos_iou_thresh, pos_iou_thresh, allow_low_quality_matches=False
        )
        pos, neg = sample_indices(labels, num_samples, positive_fraction)
        assert (labels[pos] == POSITIVE).all()

        targets = torch.zeros(len(pos) + len(neg), 4, dtype=props.dtype, device=props.device)
        if len(pos) > 0:
            targets[: len(pos)] = encode(candidates[pos], gt[matched_idx[pos]], box_weights)

        all_rois.append(candidates[torch.cat([pos, neg])])
        all_labels.append(
            torch.cat([torch.full((len(pos),), COURT), torch.zeros(len(neg))]).long().to(props.device)
        )
        all_targets.append(targets)
    return all_rois, all_labels, all_targets


def roi_loss(
    class_logits: Tensor,
    box_deltas: Tensor,
    labels: list[Tensor],
    regression_targets: list[Tensor],
    smooth_l1_beta: float = 1.0 / 9.0,
) -> tuple[Tensor, Tensor]:
    """Classification and regression losses of the RoI head, returned SEPARATELY.

    As for rpn_loss, the regression weight (`reg_weight`, default 1.0) is applied by
    the caller, so that it lives in the config.

      classification: cross-entropy on all sampled RoIs;
      regression:     Smooth L1 on the positive RoIs only (a background RoI has no box
                      to regress to);
    both divided by K, the total number of sampled RoIs of the batch. Same formula as
    torchvision's fastrcnn_loss (checked by a test), including beta = 1/9.

    Args:
        class_logits: [K, 2]. box_deltas: [K, 4]. labels, regression_targets: per-image
            lists from select_training_samples, in the same order as the RoIs.
    """
    labels_all = torch.cat(labels)
    targets_all = torch.cat(regression_targets)
    k = max(1, labels_all.numel())  # no division by zero on an empty batch

    cls_loss = F.cross_entropy(class_logits, labels_all, reduction="sum") / k
    pos = labels_all == COURT
    reg_loss = F.smooth_l1_loss(
        box_deltas[pos], targets_all[pos], beta=smooth_l1_beta, reduction="sum"
    ) / k
    return cls_loss, reg_loss


@torch.no_grad()
def postprocess_detections(
    class_logits: Tensor,
    box_deltas: Tensor,
    rois: list[Tensor],
    image_size: tuple[int, int],
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_img: int = 100,
    box_weights: Sequence[float] = DEFAULT_BOX_WEIGHTS,
    min_size: float = 1e-2,
) -> tuple[list[Tensor], list[Tensor]]:
    """Turn the head outputs into final detections, per image.

    Per image: score = softmax probability of the court class; decode the deltas on
    the RoIs and clip to the image; keep score > score_thresh; drop boxes thinner than
    `min_size`; NMS; keep the `detections_per_img` best. Defaults = torchvision
    (box_score_thresh 0.05, box_nms_thresh 0.5, box_detections_per_img 100,
    remove_small_boxes(min_size=1e-2)).

    Args:
        class_logits: [K, 2], box_deltas: [K, 4], rois: B tensors [K_i, 4] that produced
            them (sum K_i = K). image_size: (height, width).

    Returns:
        (boxes, scores): lists of B tensors [D_i, 4] and [D_i], by decreasing score.
    """
    counts = [len(r) for r in rois]
    scores_all = F.softmax(class_logits, dim=-1)[:, COURT]
    all_boxes, all_scores = [], []
    for r, deltas, scores in zip(rois, box_deltas.split(counts), scores_all.split(counts)):
        boxes = clip_boxes(decode(r, deltas, box_weights), image_size)

        keep = scores > score_thresh
        keep &= ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
        boxes, scores = boxes[keep], scores[keep]

        keep = nms(boxes, scores, nms_thresh)[:detections_per_img]  # best score first
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
    return all_boxes, all_scores
