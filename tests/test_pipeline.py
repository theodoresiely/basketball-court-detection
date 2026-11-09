"""Wiring tests: several bricks together, on random weights and random images."""

import torch

from courtdet.models.anchors import generate_base_anchors, grid_anchors
from courtdet.models.backbone import ResNet18Backbone
from courtdet.models.rpn import RPNHead, generate_proposals, rpn_loss

SIZES, RATIOS = (32.0, 64.0, 128.0), (0.5, 1.0, 2.0)


def build():
    torch.manual_seed(0)
    backbone = ResNet18Backbone(pretrained=False)
    head = RPNHead(backbone.out_channels, num_anchors=len(SIZES) * len(RATIOS))
    anchors = grid_anchors((64, 64), backbone.stride, generate_base_anchors(SIZES, RATIOS))
    return backbone, head, anchors


def test_image_to_proposals():
    backbone, head, anchors = build()
    assert anchors.shape[0] == 64 * 64 * 9 == 36864

    objectness, deltas = head(backbone(torch.randn(2, 3, 512, 512)))
    # one prediction per anchor: this is where the anchor / head ordering contract matters
    assert objectness.shape == (2, 36864)
    assert deltas.shape == (2, 36864, 4)

    boxes, scores = generate_proposals(objectness, deltas, anchors, (512, 512))
    for b, s in zip(boxes, scores):
        assert 0 < len(b) <= 1000
        assert b.min() >= 0 and b.max() <= 512  # clipped to the image
        assert ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).all()  # no empty box
        assert (s[:-1] >= s[1:]).all()  # sorted by decreasing score


def test_loss_gradients_reach_the_head_and_layer2_only():
    # The kind of check that would have caught the notebook's "RoI head never trained" problem:
    # every module that is supposed to learn must receive a gradient.
    backbone, head, anchors = build()
    gts = [torch.tensor([[100.0, 120.0, 220.0, 260.0]]), torch.zeros((0, 4))]  # one court, one empty image
    objectness, deltas = head(backbone(torch.randn(2, 3, 512, 512)))
    cls_loss, reg_loss = rpn_loss(objectness, deltas, anchors, gts)
    (cls_loss + reg_loss).backward()

    assert head.conv.weight.grad.abs().sum() > 0
    assert head.cls_logits.weight.grad.abs().sum() > 0
    assert head.bbox_deltas.weight.grad.abs().sum() > 0
    assert backbone.layer2[0].conv1.weight.grad.abs().sum() > 0
    assert backbone.layer1[0].conv1.weight.grad is None  # frozen
