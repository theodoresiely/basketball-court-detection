import torch

from courtdet.models.matching import (
    IGNORE,
    NEGATIVE,
    POSITIVE,
    match_boxes,
    sample_indices,
)

# One 10x10 ground truth (area 100) used by most tests.
GT = [[0.0, 0.0, 10.0, 10.0]]


def t(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_thresholds_on_hand_computed_ious():
    # IoU with GT = intersection / union:
    #   same box -> 1.0 ; 10x5 inside it -> 50/100 = 0.5 ; disjoint -> 0 ;
    #   10x8 inside it -> 0.8 ; 10x2 inside it -> 0.2
    anchors = t([[0, 0, 10, 10], [0, 0, 10, 5], [20, 20, 30, 30], [0, 0, 10, 8], [0, 0, 10, 2]])
    labels, matched = match_boxes(anchors, t(GT), high_thresh=0.7, low_thresh=0.3)
    assert labels.tolist() == [POSITIVE, IGNORE, NEGATIVE, POSITIVE, NEGATIVE]
    assert matched[labels == POSITIVE].tolist() == [0, 0]


def test_best_anchor_of_each_gt_is_forced_positive():
    anchors = t([[0, 0, 10, 5], [20, 20, 30, 30]])  # best IoU is 0.5 < 0.7
    without, _ = match_boxes(anchors, t(GT), 0.7, 0.3, allow_low_quality_matches=False)
    with_rule, _ = match_boxes(anchors, t(GT), 0.7, 0.3, allow_low_quality_matches=True)
    assert without.tolist() == [IGNORE, NEGATIVE]  # this court would be invisible to training
    assert with_rule.tolist() == [POSITIVE, NEGATIVE]


def test_gt_overlapping_no_anchor_creates_no_positive():
    # The notebook turned every anchor positive here (all IoU = 0 = the max).
    anchors = t([[20, 20, 30, 30], [40, 40, 50, 50]])
    labels, _ = match_boxes(anchors, t(GT), 0.7, 0.3, allow_low_quality_matches=True)
    assert labels.tolist() == [NEGATIVE, NEGATIVE]


def test_image_without_ground_truth_is_all_background():
    anchors = t([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    labels, _ = match_boxes(anchors, torch.zeros((0, 4)), 0.7, 0.3, allow_low_quality_matches=True)
    assert labels.tolist() == [NEGATIVE] * 3


def test_matched_index_points_to_the_right_ground_truth():
    gts = t([[0, 0, 10, 10], [100, 100, 120, 120]])
    anchors = t([[100, 100, 120, 120], [0, 0, 10, 10]])
    labels, matched = match_boxes(anchors, gts, 0.7, 0.3)
    assert labels.tolist() == [POSITIVE, POSITIVE]
    assert matched.tolist() == [1, 0]


def test_roi_head_thresholds_leave_nothing_ignored():
    # high == low == 0.5: an IoU of exactly 0.5 is positive, nothing falls in between.
    anchors = t([[0, 0, 10, 5], [0, 0, 10, 4]])  # IoU 0.5 and 0.4
    labels, _ = match_boxes(anchors, t(GT), 0.5, 0.5)
    assert labels.tolist() == [POSITIVE, NEGATIVE]


def test_sampling_balances_positives_and_negatives():
    labels = torch.cat([torch.ones(300), torch.zeros(300)]).long()
    pos, neg = sample_indices(labels, num_samples=256, positive_fraction=0.5)
    assert (len(pos), len(neg)) == (128, 128)


def test_sampling_fills_with_negatives_when_positives_are_rare():
    labels = torch.cat([torch.ones(5), torch.zeros(1000)]).long()
    pos, neg = sample_indices(labels, 256, 0.5)
    assert (len(pos), len(neg)) == (5, 251)


def test_sampling_is_limited_by_what_exists():
    labels = torch.cat([torch.ones(10), torch.zeros(3)]).long()
    pos, neg = sample_indices(labels, 256, 0.5)
    assert (len(pos), len(neg)) == (10, 3)


def test_sampling_returns_correct_unique_indices_and_skips_ignored():
    labels = torch.cat([torch.full((50,), IGNORE), torch.ones(20), torch.zeros(100)]).long()
    pos, neg = sample_indices(labels, 64, 0.25)
    assert (labels[pos] == POSITIVE).all() and (labels[neg] == NEGATIVE).all()
    both = torch.cat([pos, neg])
    assert both.unique().numel() == both.numel()  # no duplicates
    assert len(pos) == 16 and len(neg) == 48
