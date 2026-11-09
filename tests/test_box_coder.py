import math

import pytest
import torch

from courtdet.models.box_coder import clip_boxes, decode, encode


def random_boxes(n, seed, max_xy=512.0, min_size=4.0, max_size=200.0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * max_xy
    wh = min_size + torch.rand(n, 2, generator=g) * (max_size - min_size)
    return torch.cat([xy, xy + wh], dim=1)


def test_hand_computed_example():
    # anchor: centre (5, 5), size 10x10 ; target: centre (10, 10), size 10x20
    anchor = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    target = torch.tensor([[5.0, 0.0, 15.0, 20.0]])
    deltas = encode(anchor, target)
    expected = torch.tensor([[0.5, 0.5, 0.0, math.log(2.0)]])
    assert torch.allclose(deltas, expected)


def test_identity():
    boxes = random_boxes(100, seed=0)
    assert torch.allclose(encode(boxes, boxes), torch.zeros(100, 4), atol=1e-6)
    assert torch.allclose(decode(boxes, torch.zeros(100, 4)), boxes, atol=1e-4)


@pytest.mark.parametrize("weights", [(1.0, 1.0, 1.0, 1.0), (10.0, 10.0, 5.0, 5.0)])
def test_decode_inverts_encode(weights):
    anchors = random_boxes(1000, seed=1)
    gt = random_boxes(1000, seed=2)
    deltas = encode(anchors, gt, weights)
    assert torch.allclose(decode(anchors, deltas, weights), gt, atol=1e-3)


def test_weights_only_rescale_the_targets():
    anchors = random_boxes(50, seed=3)
    gt = random_boxes(50, seed=4)
    d1 = encode(anchors, gt)
    d2 = encode(anchors, gt, weights=(10.0, 10.0, 5.0, 5.0))
    assert torch.allclose(d2, d1 * torch.tensor([10.0, 10.0, 5.0, 5.0]), atol=1e-5)


def test_decode_is_finite_for_huge_predictions():
    anchors = random_boxes(10, seed=5)
    deltas = torch.full((10, 4), 1e4)  # what an untrained network may output
    assert torch.isfinite(decode(anchors, deltas)).all()


def test_matches_torchvision_box_coder():
    # torchvision's BoxCoder lives in a private module: skip rather than fail if it moves.
    tv_utils = pytest.importorskip("torchvision.models.detection._utils")
    coder = tv_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
    anchors = random_boxes(500, seed=6)
    gt = random_boxes(500, seed=7)
    # torchvision naming: encode_single(reference_boxes=gt, proposals=anchors)
    ours = encode(anchors, gt, weights=(10.0, 10.0, 5.0, 5.0))
    assert torch.allclose(ours, coder.encode_single(gt, anchors), atol=1e-4)
    assert torch.allclose(
        decode(anchors, ours, weights=(10.0, 10.0, 5.0, 5.0)),
        coder.decode_single(ours, anchors),
        atol=1e-3,
    )


def test_clip_boxes_to_image():
    boxes = torch.tensor([[-5.0, -5.0, 10.0, 10.0], [90.0, 90.0, 150.0, 120.0]])
    clipped = clip_boxes(boxes, image_size=(100, 200))  # (height, width)
    # x is limited to [0, 200], y to [0, 100]
    assert torch.equal(clipped, torch.tensor([[0.0, 0.0, 10.0, 10.0], [90.0, 90.0, 150.0, 100.0]]))
