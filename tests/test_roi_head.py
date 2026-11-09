import math

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import roi_align

from courtdet.data.dataset import CourtDataset, collate_fn
from courtdet.models.anchors import generate_base_anchors, grid_anchors
from courtdet.models.backbone import ResNet18Backbone
from courtdet.models.box_coder import encode
from courtdet.models.roi_head import (
    DEFAULT_BOX_WEIGHTS,
    RoIHead,
    postprocess_detections,
    roi_loss,
    select_training_samples,
)
from courtdet.models.rpn import RPNHead, generate_proposals
from test_dataset import MINI, write_coco  # pytest puts tests/ on sys.path


def t(rows):
    return torch.tensor(rows, dtype=torch.float32)


def rows_equal_to(rois, box):
    """Indices of the rows of `rois` equal to `box`."""
    return (rois == t(box)).all(dim=1).nonzero().flatten()


GT = [[0.0, 0.0, 10.0, 10.0]]
# IoU with GT:      1.0 (pos)          0.6 (pos)       0.5 (pos, >=)    0.45 (neg at 0.5)   0.0 (neg)
PROPOSALS = t([[0, 0, 10, 10], [0, 0, 10, 6], [0, 0, 10, 5], [0, 0, 10, 4.5], [20, 20, 30, 30]])


# ---------------------------------------------------------------- targets

def test_labels_and_targets_on_toy_case():
    rois, labels, targets = select_training_samples([PROPOSALS], [t(GT)], num_samples=100, positive_fraction=1.0)
    rois, labels, targets = rois[0], labels[0], targets[0]
    # 5 proposals + 1 appended GT, all labelled (high == low threshold: no ignored box), all sampled.
    assert len(rois) == 6

    expected = {(0, 0, 10, 10): 1, (0, 0, 10, 6): 1, (0, 0, 10, 5): 1, (0, 0, 10, 4.5): 0, (20, 20, 30, 30): 0}
    for box, label in expected.items():
        idx = rows_equal_to(rois, box)
        assert len(idx) >= 1 and (labels[idx] == label).all(), box
    assert len(rows_equal_to(rois, (0, 0, 10, 10))) == 2  # proposal + appended GT

    # Hand-computed target of [0, 0, 10, 6] -> [0, 0, 10, 10], weights (10, 10, 5, 5):
    # same cx and width; cy 3 -> 5 with h = 6; h 6 -> 10.
    idx = rows_equal_to(rois, (0, 0, 10, 6))[0]
    assert torch.allclose(targets[idx], torch.tensor([0.0, 10 * 2 / 6, 0.0, 5 * math.log(10 / 6)]))
    assert (targets[labels == 0] == 0).all()  # negatives: no regression target
    assert (labels[: int(labels.sum())] == 1).all()  # positives come first


def test_notebook_threshold_0_4_turns_the_0_45_proposal_positive():
    rois, labels, _ = select_training_samples(
        [PROPOSALS], [t(GT)], pos_iou_thresh=0.4, num_samples=100, positive_fraction=1.0
    )
    assert labels[0][rows_equal_to(rois[0], (0, 0, 10, 4.5))].item() == 1


def test_appended_ground_truths_are_positive_with_zero_deltas():
    gts = t([[100, 100, 180, 160], [300, 50, 330, 120]])
    far = t([[0, 0, 20, 20]])  # no proposal overlaps the courts: only the GTs can be positive
    rois, labels, targets = select_training_samples([far], [gts])
    assert labels[0].sum() == 2
    for gt in gts.tolist():
        idx = rows_equal_to(rois[0], gt)
        assert len(idx) == 1
        assert labels[0][idx].item() == 1
        assert torch.equal(targets[0][idx], torch.zeros(1, 4))


def test_at_most_25_percent_positives_per_image():
    torch.manual_seed(0)
    gt = t([[100, 100, 200, 200]])
    near = gt + torch.randint(-5, 6, (100, 4)).float()  # 100 proposals with IoU > 0.5
    far = torch.rand(300, 2) * 250 + 250  # 300 proposals in [250, 550] x [250, 550]: IoU 0
    far = torch.cat([far, far + 30], dim=1)
    rois, labels, _ = select_training_samples([torch.cat([near, far]), gt + 2], [gt, gt])

    # image 1: plenty of both -> exactly 32 / 96
    assert len(rois[0]) == 128 and labels[0].sum() == 32
    # image 2: 2 positives only (the proposal and the GT) and no negative at all
    assert len(rois[1]) == 2 and labels[1].sum() == 2


def test_image_without_ground_truth():
    torch.manual_seed(0)
    props = torch.rand(50, 2) * 400
    props = torch.cat([props, props + 40], dim=1)
    rois, labels, targets = select_training_samples([props], [torch.zeros(0, 4)])
    assert len(rois[0]) == 50  # all negatives, all fit in the 128 budget
    assert (labels[0] == 0).all() and (targets[0] == 0).all()

    logits, deltas = torch.randn(50, 2), torch.randn(50, 4)
    cls_loss, reg_loss = roi_loss(logits, deltas, labels, targets)
    assert torch.isfinite(cls_loss) and reg_loss.item() == 0.0


def test_empty_batch_gives_zero_loss_not_nan():
    # no proposal and no ground truth: zero RoI. Must not crash nor divide by zero.
    rois, labels, targets = select_training_samples([torch.zeros(0, 4)], [torch.zeros(0, 4)])
    head = RoIHead()
    logits, deltas = head(torch.randn(1, 128, 16, 16), rois)
    assert logits.shape == (0, 2) and deltas.shape == (0, 4)
    cls_loss, reg_loss = roi_loss(logits, deltas, labels, targets)
    assert cls_loss.item() == 0.0 and reg_loss.item() == 0.0


# ---------------------------------------------------------------- architecture

@pytest.mark.parametrize("size", [7, 5])
def test_output_shapes(size):
    head = RoIHead(in_channels=128, roi_output_size=size)
    assert head.fc6.in_features == 128 * size**2
    features = torch.randn(2, 128, 64, 64)
    rois = [t([[0, 0, 64, 64], [10, 20, 200, 120], [300, 300, 330, 320]]), t([[50, 50, 450, 450], [0, 0, 512, 512]])]
    assert head.pool(features, rois).shape == (5, 128, size, size)
    logits, deltas = head(features, rois)
    assert logits.shape == (5, 2) and deltas.shape == (5, 4)


def test_initial_predictions_are_neutral():
    torch.manual_seed(0)
    head = RoIHead()
    logits, deltas = head(torch.randn(1, 128, 64, 64), [t([[0, 0, 100, 100], [200, 200, 300, 260]])])
    assert (logits.softmax(-1)[:, 1] - 0.5).abs().max() < 0.05
    assert deltas.abs().max() < 0.05


@pytest.mark.parametrize("aligned, expected", [(True, 11.5), (False, 12.0)])
def test_aligned_flag_on_a_ramp(aligned, expected):
    # Feature cell j holds the value j, and covers image pixels [8j, 8j + 8).
    # The box [64, 128] covers exactly cells 8..15, whose mean value is 11.5.
    # aligned=True gets it; the legacy mode reads half a cell (4 px) too far right.
    ramp = torch.arange(64.0).expand(1, 1, 64, 64).contiguous()
    head = RoIHead(in_channels=1, aligned=aligned)
    pooled = head.pool(ramp, [t([[64, 64, 128, 128]])])
    assert pooled.mean().item() == pytest.approx(expected, abs=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_roi_align_on_mps_matches_cpu():
    torch.manual_seed(0)
    features = torch.randn(2, 128, 64, 64)
    rois = [t([[3.3, 7.1, 120.4, 90.2]]), t([[200, 10, 260, 300], [0, 0, 512, 512]])]
    head = RoIHead()
    cpu = head.pool(features, rois)
    mps = head.pool(features.to("mps"), [r.to("mps") for r in rois]).cpu()
    assert torch.allclose(cpu, mps, atol=1e-4)


# ---------------------------------------------------------------- loss and gradients

def test_loss_matches_torchvision_fastrcnn_loss():
    torch.manual_seed(0)
    k = 40
    logits = torch.randn(k, 2)
    deltas = torch.randn(k, 4)
    labels = [torch.randint(0, 2, (25,)), torch.randint(0, 2, (15,))]
    targets = [torch.randn(25, 4), torch.randn(15, 4)]

    ours = roi_loss(logits, deltas, labels, targets)
    # torchvision predicts 4 deltas PER CLASS and reads the slice of the true class:
    # put ours in the court slice, garbage in the background slice (never read).
    per_class = torch.cat([torch.randn(k, 4) * 100, deltas], dim=1)
    theirs = fastrcnn_loss(logits, per_class, labels, targets)
    assert torch.allclose(ours[0], theirs[0]) and torch.allclose(ours[1], theirs[1])


def test_gradients_reach_the_head_but_not_the_proposals():
    # The check that would have caught the notebook's untrained RoI head.
    torch.manual_seed(0)
    head = RoIHead()
    features = torch.randn(1, 128, 64, 64, requires_grad=True)
    leaf = torch.tensor([[95.0, 100, 210, 190], [300, 300, 400, 380]], requires_grad=True)
    proposals = [leaf * 1.0]  # NOT detached, as if it came out of the RPN graph
    rois, labels, targets = select_training_samples(proposals, [t([[100, 100, 200, 200]])])
    logits, deltas = head(features, rois)
    cls_loss, reg_loss = roi_loss(logits, deltas, labels, targets)
    (cls_loss + reg_loss).backward()

    for layer in (head.fc6, head.fc7, head.cls_score, head.bbox_pred):
        assert layer.weight.grad.abs().sum() > 0
    assert features.grad.abs().sum() > 0  # flows back to the backbone through roi_align
    assert leaf.grad is None  # proposals and their targets are constants


# ---------------------------------------------------------------- inference

def test_postprocess_on_toy_case():
    def logit(p):
        return [0.0, math.log(p / (1 - p))]  # softmax([0, z])[1] = sigmoid(z) = p

    target = t([[8, 12, 56, 48]])
    rois = t([
        [10, 10, 50, 50],      # score 0.9, moved onto `target`
        [12, 10, 52, 50],      # score 0.8, moved onto almost the same box -> removed by NMS
        [100, 100, 140, 140],  # score 0.03 < 0.05 -> removed
        [200, 200, 240, 240],  # score 0.6, kept as is
        [480, 480, 520, 520],  # score 0.7, sticks out -> clipped to 512
        [600, 600, 640, 640],  # score 0.95, outside the image -> empty after clipping -> removed
    ])
    deltas = torch.zeros(6, 4)
    deltas[0] = encode(rois[0], target[0], DEFAULT_BOX_WEIGHTS)
    deltas[1] = encode(rois[1], t([9, 12, 57, 48]), DEFAULT_BOX_WEIGHTS)
    logits = t([logit(p) for p in (0.9, 0.8, 0.03, 0.6, 0.7, 0.95)])

    boxes, scores = postprocess_detections(logits, deltas, [rois], (512, 512))
    assert torch.allclose(scores[0], t([0.9, 0.7, 0.6]))
    assert torch.allclose(boxes[0], t([[8, 12, 56, 48], [480, 480, 512, 512], [200, 200, 240, 240]]), atol=1e-4)

    boxes, scores = postprocess_detections(logits, deltas, [rois], (512, 512), detections_per_img=2)
    assert len(boxes[0]) == 2


def test_postprocess_splits_per_image():
    rois = [t([[0, 0, 50, 50]]), t([[0, 0, 50, 50], [100, 100, 150, 150]])]
    logits = t([[0, 5], [0, 5], [5, 0]])  # third RoI: background
    boxes, scores = postprocess_detections(logits, torch.zeros(3, 4), rois, (512, 512))
    assert [len(b) for b in boxes] == [1, 1]


# ---------------------------------------------------------------- contract with the other bricks

def test_contract_with_dataset_rpn_and_proposals(tmp_path):
    write_coco(tmp_path, "train", MINI)
    torch.manual_seed(0)
    images, targets, _ = next(iter(DataLoader(CourtDataset(tmp_path, "train"), batch_size=3, collate_fn=collate_fn)))

    backbone = ResNet18Backbone(pretrained=False)
    sizes, ratios = (32.0, 64.0, 128.0), (0.5, 1.0, 2.0)
    rpn = RPNHead(backbone.out_channels, num_anchors=len(sizes) * len(ratios))
    head = RoIHead(in_channels=backbone.out_channels, spatial_scale=1.0 / backbone.stride)

    features = backbone(images)
    anchors = grid_anchors(tuple(features.shape[-2:]), backbone.stride, generate_base_anchors(sizes, ratios))
    proposals, _ = generate_proposals(*rpn(features), anchors, tuple(images.shape[-2:]))

    gt_boxes = [tg["boxes"] for tg in targets]  # the collate format: list of [N_i, 4]
    rois, labels, reg_targets = select_training_samples(proposals, gt_boxes)
    for r, g, lab in zip(rois, gt_boxes, labels):
        assert len(r) == 128  # ~1000 proposals: the budget is always filled
        assert lab.sum() >= len(g)  # at least the appended ground truths

    logits, deltas = head(features, rois)
    cls_loss, reg_loss = roi_loss(logits, deltas, labels, reg_targets)
    assert torch.isfinite(cls_loss) and torch.isfinite(reg_loss)
    (cls_loss + reg_loss).backward()
    assert backbone.layer2[0].conv1.weight.grad.abs().sum() > 0

    head.eval()
    with torch.no_grad():
        boxes, scores = postprocess_detections(*head(features, proposals), proposals, tuple(images.shape[-2:]))
    assert len(boxes) == 3
    for b in boxes:
        assert len(b) <= 100 and (b >= 0).all() and (b <= 512).all()
