"""Faster R-CNN: one module that owns every brick (backbone, RPN, RoI head).

Why a single module: the optimizer is built from `model.parameters()`, so it cannot
miss a sub-network (the notebook's optimizer did not contain the RoI head); one
`state_dict` checkpoints everything; `.train()` / `.eval()` reach every brick.

    images [B, 3, S, S]
      -> backbone              features [B, C, S/s, S/s]   (C, s = 128, 8 by default; backbone.py)
      -> RPN head              objectness [B, N], deltas [B, N, 4]  (N = cells x anchors)
      -> generate_proposals    <= post_nms_top_n boxes per image   (constants, no_grad)
      -> RoI head              class logits, deltas per RoI

API (differs from torchvision on purpose):
  - `model(images, gt_boxes)` ALWAYS returns the losses, whatever the mode, so that
    a validation loss can be computed under `model.eval()` + `torch.no_grad()`.
    The mode only selects the number of proposals (train / test values).
  - `model.detect(images)` ALWAYS returns detections, with the test values.
  In this network the mode has no other effect: BatchNorm is always frozen (see
  backbone.py) and there is no dropout.

Known limitation (hypothesis, not measured): source A tiles are 256 x 256 padded to
512; proposals and detections are clipped to the padded crop, not to the real image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from courtdet.models.anchors import generate_base_anchors, grid_anchors
from courtdet.models.backbone import ResNet18Backbone
from courtdet.models.roi_head import (
    DEFAULT_BOX_WEIGHTS,
    RoIHead,
    postprocess_detections,
    roi_loss,
    select_training_samples,
)
from courtdet.models.rpn import RPNHead, generate_proposals, rpn_loss


@dataclass
class AnchorConfig:
    # 6 sizes spaced by <= x1.5 (measured anchor recall: 90 % / 76 % at IoU >= 0.7
    # for sources A / B, vs 45 % / 41 % with 32/64/128) x 3 ratios = 18 anchors per cell.
    sizes: tuple[float, ...] = (32.0, 48.0, 64.0, 96.0, 128.0, 192.0)
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass
class RPNConfig:
    # Matching / sampling / loss: values of rpn_loss, = torchvision (faster_rcnn.py:183-187).
    pos_iou_thresh: float = 0.7
    neg_iou_thresh: float = 0.3
    num_samples: int = 256
    positive_fraction: float = 0.5
    smooth_l1_beta: float = 1.0 / 9.0
    reg_weight: float = 1.0  # lambda; the notebook used 10 (to compare on the mAP)
    # Proposals: torchvision (faster_rcnn.py:179-183). More proposals in training give
    # more varied negatives to sample from; 1000 is enough (and cheaper) at test time.
    nms_thresh: float = 0.7
    pre_nms_top_n_train: int = 2000
    post_nms_top_n_train: int = 2000
    pre_nms_top_n_test: int = 1000
    post_nms_top_n_test: int = 1000
    mid_channels: int = 256


@dataclass
class RoIConfig:
    # Defaults of roi_head.py; see its docstrings for the sources.
    roi_output_size: int = 7
    sampling_ratio: int = 2
    aligned: bool = True
    hidden_dim: int = 1024
    pos_iou_thresh: float = 0.5
    num_samples: int = 128
    positive_fraction: float = 0.25
    box_weights: tuple[float, float, float, float] = DEFAULT_BOX_WEIGHTS
    smooth_l1_beta: float = 1.0 / 9.0
    reg_weight: float = 1.0
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100


@dataclass
class FasterRCNNConfig:
    pretrained_backbone: bool = True
    trainable_layers: tuple[str, ...] = ("layer2",)
    backbone_features: str = "layer2"  # "layer2" | "layer3" | "fused" (backbone.py)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    rpn: RPNConfig = field(default_factory=RPNConfig)
    roi: RoIConfig = field(default_factory=RoIConfig)


class FasterRCNN(nn.Module):
    def __init__(self, config: FasterRCNNConfig | None = None):
        super().__init__()
        self.config = config = config or FasterRCNNConfig()
        self.backbone = ResNet18Backbone(config.pretrained_backbone, config.trainable_layers,
                                         config.backbone_features)
        # Base anchors are a buffer: they follow .to(device) and are part of the state_dict,
        # so a checkpoint cannot be reloaded with a different anchor set by mistake.
        self.register_buffer(
            "base_anchors", generate_base_anchors(config.anchors.sizes, config.anchors.aspect_ratios)
        )
        self.rpn_head = RPNHead(
            self.backbone.out_channels, num_anchors=len(self.base_anchors), mid_channels=config.rpn.mid_channels
        )
        self.roi_head = RoIHead(
            in_channels=self.backbone.out_channels,
            roi_output_size=config.roi.roi_output_size,
            spatial_scale=1.0 / self.backbone.stride,
            sampling_ratio=config.roi.sampling_ratio,
            aligned=config.roi.aligned,
            hidden_dim=config.roi.hidden_dim,
        )

    def _rpn(self, images: Tensor, training: bool, proposal_scale: float = 1.0):
        """Shared first stage: features, RPN outputs, anchors and proposals.

        proposal_scale multiplies the test-time proposal counts (see `detect`).
        """
        features = self.backbone(images)
        objectness, deltas = self.rpn_head(features)
        anchors = grid_anchors(tuple(features.shape[-2:]), self.backbone.stride, self.base_anchors)
        c = self.config.rpn
        proposals, _ = generate_proposals(
            objectness,
            deltas,
            anchors,
            tuple(images.shape[-2:]),
            pre_nms_top_n=c.pre_nms_top_n_train if training else round(c.pre_nms_top_n_test * proposal_scale),
            post_nms_top_n=c.post_nms_top_n_train if training else round(c.post_nms_top_n_test * proposal_scale),
            nms_thresh=c.nms_thresh,
        )
        return features, objectness, deltas, anchors, proposals

    def forward(self, images: Tensor, gt_boxes: list[Tensor]) -> dict[str, Tensor]:
        """Losses of both stages.

        Args:
            images: [B, 3, S, S]. gt_boxes: B tensors [M_i, 4] (M_i may be 0), i.e.
                [t["boxes"] for t in targets] from the dataset's collate_fn.

        Returns:
            {"rpn_cls", "rpn_reg", "roi_cls", "roi_reg"} (unweighted, for logging) and
            "total" = rpn_cls + rpn.reg_weight * rpn_reg + roi_cls + roi.reg_weight * roi_reg.
        """
        cr, ci = self.config.rpn, self.config.roi
        features, objectness, deltas, anchors, proposals = self._rpn(images, self.training)

        rpn_cls, rpn_reg = rpn_loss(
            objectness, deltas, anchors, gt_boxes,
            pos_iou_thresh=cr.pos_iou_thresh, neg_iou_thresh=cr.neg_iou_thresh,
            num_samples=cr.num_samples, positive_fraction=cr.positive_fraction,
            smooth_l1_beta=cr.smooth_l1_beta,
        )
        rois, labels, targets = select_training_samples(
            proposals, gt_boxes,
            pos_iou_thresh=ci.pos_iou_thresh, num_samples=ci.num_samples,
            positive_fraction=ci.positive_fraction, box_weights=ci.box_weights,
        )
        class_logits, box_deltas = self.roi_head(features, rois)
        roi_cls, roi_reg = roi_loss(class_logits, box_deltas, labels, targets, smooth_l1_beta=ci.smooth_l1_beta)

        total = rpn_cls + cr.reg_weight * rpn_reg + roi_cls + ci.reg_weight * roi_reg
        return {"rpn_cls": rpn_cls, "rpn_reg": rpn_reg, "roi_cls": roi_cls, "roi_reg": roi_reg, "total": total}

    @torch.no_grad()
    def detect(self, images: Tensor, proposal_scale: float = 1.0) -> list[dict[str, Tensor]]:
        """Detections per image: {"boxes": [D, 4], "scores": [D]}, by decreasing score.

        proposal_scale multiplies the test proposal counts. The defaults (1000) are meant
        for a 512 x 512 crop; for a whole 2048 x 2048 image (16x the area), pass 16 to keep
        the same number of proposals per unit of area.
        """
        ci = self.config.roi
        features, _, _, _, proposals = self._rpn(images, training=False, proposal_scale=proposal_scale)
        class_logits, box_deltas = self.roi_head(features, proposals)
        boxes, scores = postprocess_detections(
            class_logits, box_deltas, proposals, tuple(images.shape[-2:]),
            score_thresh=ci.score_thresh, nms_thresh=ci.nms_thresh,
            detections_per_img=ci.detections_per_img, box_weights=ci.box_weights,
        )
        return [{"boxes": b, "scores": s} for b, s in zip(boxes, scores)]
