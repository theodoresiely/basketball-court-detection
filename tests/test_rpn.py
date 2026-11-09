import math

import pytest
import torch

from courtdet.models.anchors import generate_base_anchors, grid_anchors
from courtdet.models.box_coder import encode
from courtdet.models.matching import match_boxes
from courtdet.models.rpn import RPNHead, flatten_predictions, generate_proposals, rpn_loss


def t(rows):
    return torch.tensor(rows, dtype=torch.float32)


GT = [[0.0, 0.0, 10.0, 10.0]]

# With this GT (IoU thresholds 0.7 / 0.3) the labels are:
# positive, positive (IoU 0.8), negative, negative, ignored (IoU 0.5).
ANCHORS = t([[0, 0, 10, 10], [0, 0, 10, 8], [20, 20, 30, 30], [40, 40, 50, 50], [0, 0, 10, 5]])


# ---------------------------------------------------------------- head

def test_head_output_shapes():
    head = RPNHead(in_channels=128, num_anchors=9)
    objectness, deltas = head(torch.randn(2, 128, 8, 6))
    assert objectness.shape == (2, 8 * 6 * 9)
    assert deltas.shape == (2, 8 * 6 * 9, 4)


def test_initial_predictions_are_neutral():
    head = RPNHead(128, 9)
    objectness, deltas = head(torch.randn(1, 128, 8, 8))
    assert objectness.abs().max() < 0.5  # p(court) close to 0.5
    assert deltas.abs().max() < 0.5  # anchors barely moved


def test_flatten_matches_grid_anchor_order():
    b, k, h, w, c = 2, 3, 4, 5, 4  # h != w on purpose
    x = torch.randn(b, k * c, h, w)
    flat = flatten_predictions(x, per_anchor=c)
    assert flat.shape == (b, h * w * k, c)
    for i in range(h):
        for j in range(w):
            for a in range(k):
                assert torch.equal(flat[:, (i * w + j) * k + a, :], x[:, a * c:(a + 1) * c, i, j])


# ---------------------------------------------------------------- loss

def test_classification_loss_hand_computed():
    objectness = torch.tensor([[2.0, 0.5, -1.0, 0.3, 0.0]])  # last anchor is ignored
    deltas = torch.zeros(1, 5, 4)
    cls_loss, _ = rpn_loss(objectness, deltas, ANCHORS, [t(GT)])
    # Binary cross-entropy with logit z: log(1 + e^-z) if the target is 1, log(1 + e^z) if 0.
    # Sampled: 2 positives (z = 2, 0.5) and 2 negatives (z = -1, 0.3), averaged.
    expected = (
        math.log1p(math.exp(-2.0)) + math.log1p(math.exp(-0.5))
        + math.log1p(math.exp(-1.0)) + math.log1p(math.exp(0.3))
    ) / 4  # about 0.4422
    assert cls_loss.item() == pytest.approx(expected, abs=1e-5)


def test_regression_loss_hand_computed():
    objectness = torch.zeros(1, 5)
    deltas = torch.zeros(1, 5, 4)  # the network predicts "do not move"
    _, reg_loss = rpn_loss(objectness, deltas, ANCHORS, [t(GT)], smooth_l1_beta=1.0)
    # Anchor 0 equals the GT: target deltas are 0 -> no loss.
    # Anchor 1 is [0,0,10,8]: tx = tw = 0, ty = (5 - 4) / 8 = 0.125, th = log(10 / 8).
    # Smooth L1 with beta = 1 on |x| < 1 is 0.5 x^2. Divided by the 4 sampled anchors.
    expected = (0.5 * 0.125 ** 2 + 0.5 * math.log(1.25) ** 2) / 4  # about 0.00818
    assert reg_loss.item() == pytest.approx(expected, abs=1e-6)


def test_no_ground_truth_gives_background_loss_only():
    cls_loss, reg_loss = rpn_loss(
        torch.zeros(1, 5), torch.randn(1, 5, 4), ANCHORS, [torch.zeros((0, 4))]
    )
    assert cls_loss.item() == pytest.approx(math.log(2), abs=1e-6)  # BCE(logit 0, target 0)
    assert reg_loss.item() == 0.0


def test_loss_does_not_depend_on_batch_size():
    g = torch.Generator().manual_seed(0)
    objectness = torch.randn(1, 5, generator=g)
    deltas = torch.randn(1, 5, 4, generator=g)
    single = rpn_loss(objectness, deltas, ANCHORS, [t(GT)])
    # Same image twice. Few anchors (< 256), so all are sampled: no randomness involved.
    double = rpn_loss(objectness.repeat(2, 1), deltas.repeat(2, 1, 1), ANCHORS, [t(GT), t(GT)])
    assert torch.allclose(torch.stack(single), torch.stack(double))


def test_perfect_predictions_give_near_zero_loss():
    base = generate_base_anchors((32.0, 64.0, 128.0), (0.5, 1.0, 2.0))
    anchors = grid_anchors((16, 16), stride=8, base_anchors=base)  # 128x128 image
    gt = t([[40, 30, 90, 80]])

    labels, matched = match_boxes(anchors, gt, 0.7, 0.3, allow_low_quality_matches=True)
    assert (labels == 1).any()  # sanity check of the setup

    objectness = torch.where(labels == 1, 20.0, -20.0)[None]  # confident and right
    deltas = encode(anchors, gt[matched])[None]  # exactly the targets
    cls_loss, reg_loss = rpn_loss(objectness, deltas, anchors, [gt])
    assert cls_loss.item() < 1e-6
    assert reg_loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- proposals

IMAGE_SIZE = (100, 100)  # (height, width)


def test_proposals_apply_nms_and_are_sorted_by_score():
    anchors = t([[0, 0, 10, 10], [0, 0, 10, 9], [50, 50, 60, 60]])
    objectness = torch.tensor([[3.0, 2.0, 1.0]])
    boxes, scores = generate_proposals(objectness, torch.zeros(1, 3, 4), anchors, IMAGE_SIZE, nms_thresh=0.7)
    # anchor 1 overlaps anchor 0 with IoU 0.9 (90 / 100) and scores lower: suppressed
    assert torch.allclose(boxes[0], anchors[[0, 2]], atol=1e-4)
    assert torch.allclose(scores[0], torch.sigmoid(torch.tensor([3.0, 1.0])))


def test_proposals_apply_deltas_to_the_right_anchors():
    anchors = t([[0, 0, 10, 10], [50, 50, 60, 60]])
    objectness = torch.tensor([[2.0, 1.0]])
    deltas = torch.tensor([[[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    boxes, _ = generate_proposals(objectness, deltas, anchors, IMAGE_SIZE)
    # tx = 0.5 shifts the first anchor to the right by 0.5 * width = 5 px
    assert torch.allclose(boxes[0], t([[5, 0, 15, 10], [50, 50, 60, 60]]), atol=1e-4)


def test_proposals_are_clipped_and_empty_boxes_removed():
    anchors = t([[-5, -5, 10, 10], [150, 150, 160, 160]])
    objectness = torch.tensor([[2.0, 1.0]])
    boxes, scores = generate_proposals(objectness, torch.zeros(1, 2, 4), anchors, IMAGE_SIZE)
    # the second anchor is entirely outside: clipping squeezes it to zero size, it is dropped
    assert torch.allclose(boxes[0], t([[0, 0, 10, 10]]), atol=1e-4)
    assert scores[0].shape == (1,)


def test_top_k_before_and_after_nms():
    n = 20
    anchors = t([[20.0 * i, 0, 20.0 * i + 10, 10] for i in range(n)])  # disjoint boxes
    objectness = torch.arange(n, dtype=torch.float32)[None]  # anchor 19 has the best score
    boxes, _ = generate_proposals(
        objectness, torch.zeros(1, n, 4), anchors, (100, 500), pre_nms_top_n=10, post_nms_top_n=5
    )
    # before NMS: anchors 19..10 remain; after: the 5 best of them, best first
    assert torch.allclose(boxes[0], anchors[[19, 18, 17, 16, 15]], atol=1e-4)


def test_proposals_are_constants_for_the_next_stage():
    anchors = t([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]])
    objectness = torch.randn(1, 3, requires_grad=True)
    deltas = torch.randn(1, 3, 4, requires_grad=True)
    boxes, scores = generate_proposals(objectness, deltas, anchors, IMAGE_SIZE)
    assert not boxes[0].requires_grad and not scores[0].requires_grad


def test_one_result_per_image_in_the_batch():
    anchors = t([[0, 0, 10, 10], [50, 50, 60, 60]])
    objectness = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    boxes, scores = generate_proposals(objectness, torch.zeros(2, 2, 4), anchors, IMAGE_SIZE)
    assert len(boxes) == len(scores) == 2
    assert torch.allclose(boxes[0], anchors[[0, 1]], atol=1e-4)
    assert torch.allclose(boxes[1], anchors[[1, 0]], atol=1e-4)  # image 1 prefers anchor 1
