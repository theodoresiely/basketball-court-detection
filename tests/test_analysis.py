import numpy as np
import pytest

from courtdet.engine.analysis import (
    ap50_without,
    classify_courts,
    classify_detections,
    image_type,
    max_recall,
    quantiles,
    ratio_bins,
    size_bins,
)
from courtdet.engine.evaluate import average_precision, match_image


def b(rows):
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


GT = b([[0, 0, 100, 100], [300, 300, 340, 340]])
# score: 0.9 on court 0 (TP), 0.8 shifted (IoU 0.33 with court 0: localization), 0.7 on court 0 again
# (duplicate), 0.6 far away (background), 0.5 on court 1 (TP)
DETS = b([[0, 0, 100, 100], [50, 0, 150, 100], [5, 0, 105, 100], [600, 600, 650, 650], [300, 300, 340, 340]])
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5])


def test_image_types():
    assert image_type("basketball_court_065_jpg.rf.x.jpg") == "A"
    assert image_type("P0850_png.rf.abc.jpg") == "P####"
    assert image_type("38123-49870-16_jpg.rf.abc.jpg") == "tile"
    assert image_type("Screenshot-2025-08-15-at-10_png.rf.abc.jpg") == "screenshot"
    assert image_type("3f2a9c1e-uuid.rf.abc.jpg") == "other"


def test_size_and_ratio_bins():
    boxes = b([[0, 0, 20, 20], [0, 0, 32, 32], [0, 0, 50, 50], [0, 0, 64, 64], [0, 0, 200, 200]])
    assert size_bins(boxes).tolist() == [0, 1, 1, 2, 3]  # a 32 px box falls in "32-64"
    assert ratio_bins(b([[0, 0, 10, 30], [0, 0, 30, 30], [0, 0, 30, 10]])).tolist() == [0, 1, 2]


def test_classify_detections_on_hand_built_case():
    boxes, scores, labels = classify_detections(DETS, SCORES, GT)
    assert scores.tolist() == SCORES.tolist()
    assert labels.tolist() == ["tp", "localization", "duplicate", "background", "tp"]


def test_classification_follows_score_order_and_max_dets():
    shuffled = [4, 2, 0, 3, 1]
    _, scores, labels = classify_detections(DETS[shuffled], SCORES[shuffled], GT)
    assert scores.tolist() == SCORES.tolist()
    assert labels.tolist() == ["tp", "localization", "duplicate", "background", "tp"]
    many = np.repeat(b([[600, 600, 650, 650]]), 150, axis=0)
    _, scores, labels = classify_detections(many, np.linspace(0.9, 0.1, 150), GT)
    assert len(labels) == 100 and set(labels) == {"background"}


def test_image_without_courts():
    _, _, labels = classify_detections(DETS, SCORES, b([]))
    assert set(labels) == {"background"}
    _, labels = classify_courts(DETS, b([]))
    assert len(labels) == 0


def test_classify_courts():
    gt = b([[0, 0, 100, 100], [300, 300, 340, 340], [700, 700, 760, 760]])
    dets = b([[0, 0, 100, 90], [310, 300, 350, 340]])  # IoU 0.9 ; 30*40 / (2*1600 - 1200) = 0.6 ; nothing
    best, labels = classify_courts(dets, gt)
    assert np.allclose(best, [0.9, 0.6, 0.0])
    assert labels.tolist() == ["touched", "touched", "missed"]
    _, labels = classify_courts(b([[0, 0, 100, 30]]), gt[:1])  # IoU 0.3
    assert labels.tolist() == ["localization"]


def test_oracle_removal_and_max_recall():
    classified = [classify_detections(DETS, SCORES, GT)]
    n_gts = [2]
    ap = ap50_without(classified, n_gts)
    assert ap == pytest.approx(average_precision([match_image(DETS, SCORES, GT)])[0])
    # without any false alarm: TP, TP -> precision 1 up to recall 1 -> AP = 1
    assert ap50_without(classified, n_gts, remove=("localization", "duplicate", "background")) == pytest.approx(1.0)
    # removing only the background FP (rank 4) leaves precision 2/4 at the 2nd TP:
    # precision 1, 1/2, 1/3, 2/4 ; recall 1/2, 1/2, 1/2, 1 ; envelope 1, 1/2, 1/2, 1/2
    # levels 0..0.5 (51) -> 1 ; 0.51..1 (50) -> 1/2  => AP = (51 + 25) / 101
    assert ap50_without(classified, n_gts, remove=("background",)) == pytest.approx((51 + 25) / 101)
    assert max_recall(classified, n_gts) == 1.0
    assert max_recall([classify_detections(DETS[:4], SCORES[:4], GT)], n_gts) == 0.5


def test_quantiles():
    assert quantiles([]) == []
    assert quantiles(np.arange(11), (0.5,)) == [5.0]
