import numpy as np
import pytest

from courtdet.data.crops import (
    admissible_origin_range,
    clip_boxes_to_crop,
    crop_window,
    sample_crop,
    visible_fraction,
)

S = 512


# --- visible_fraction -------------------------------------------------------------------

def test_visible_fraction_hand_computed():
    crop = [0, 0, 20, 20]
    assert visible_fraction([5, 5, 15, 15], crop) == 1.0  # inside
    assert visible_fraction([30, 0, 40, 10], crop) == 0.0  # disjoint
    assert visible_fraction([20, 0, 30, 10], crop) == 0.0  # touches the edge: zero-area overlap
    assert visible_fraction([10, 0, 30, 10], crop) == 0.5  # 10x10 visible out of 20x10
    assert visible_fraction([10, 10, 30, 30], crop) == 0.25  # corner: 10x10 out of 20x20


def test_visible_fraction_is_vectorised():
    f = visible_fraction([[5, 5, 15, 15], [10, 0, 30, 10], [30, 0, 40, 10]], [0, 0, 20, 20])
    assert f.tolist() == [1.0, 0.5, 0.0]
    assert visible_fraction(np.zeros((0, 4)), [0, 0, 20, 20]).shape == (0,)


def test_visible_fraction_rejects_empty_boxes():
    with pytest.raises(ValueError):
        visible_fraction([5, 5, 5, 15], [0, 0, 20, 20])


# --- admissible_origin_range ------------------------------------------------------------
# Per axis: [ceil(x2 - S), floor(x1)] & [0, max(W - S, 0)].  image_size is (height, width).

def test_origin_range_box_far_from_borders():
    # x: [max(150 - 512, 0), min(100, 2048 - 512)] = [0, 100] ; y: [0, min(200, 1024 - 512)] = [0, 200]
    assert admissible_origin_range([100, 200, 150, 260], S, (1024, 2048)) == ((0, 100), (0, 200))


def test_origin_range_box_near_bottom_right():
    # x: [2000 - 512, min(1900, 1536)] = [1488, 1536] ; y: [1000 - 512, min(900, 512)] = [488, 512]
    assert admissible_origin_range([1900, 900, 2000, 1000], S, (1024, 2048)) == ((1488, 1536), (488, 512))


def test_origin_range_rounds_inwards():
    # x: [ceil(521.8 - 512) = 10, floor(10.3) = 10]: one integer origin left
    assert admissible_origin_range([10.3, 0, 521.8, 10], S, (2048, 2048))[0] == (10, 10)
    # exactly S wide but not on an integer: [ceil(10.3) = 11, floor(10.3) = 10] is empty
    assert admissible_origin_range([10.3, 0, 522.3, 10], S, (2048, 2048)) is None


def test_origin_range_box_larger_than_crop():
    assert admissible_origin_range([0, 0, 600, 10], S, (2048, 2048)) is None


def test_origin_range_image_smaller_than_crop():
    # source A tile: only origin 0 on both axes (the Dataset pads right / bottom)
    assert admissible_origin_range([50, 60, 100, 120], S, (256, 256)) == ((0, 0), (0, 0))


# --- clip_boxes_to_crop -----------------------------------------------------------------

def test_clip_boxes_to_crop_hand_computed():
    out = clip_boxes_to_crop([[100, 100, 200, 200], [500, 500, 700, 600]], (50, 80), S)
    # shift by (-50, -80), then clip to [0, 512]
    assert out.tolist() == [[50, 20, 150, 120], [450, 420, 512, 512]]


# --- sample_crop: targeted cases --------------------------------------------------------

def test_small_image_gives_the_whole_image_whatever_p_pos():
    boxes = [[30, 40, 140, 150]]
    for p_pos in (0.0, 1.0):
        crop = sample_crop(boxes, (256, 256), p_pos=p_pos, rng=np.random.default_rng(0))
        assert crop.mode == "whole" and crop.origin == (0, 0)
        assert crop.boxes.tolist() == boxes  # unchanged: padding goes right / bottom


def test_image_without_courts_gives_negative_crops_only():
    rng = np.random.default_rng(0)
    for _ in range(50):
        crop = sample_crop(np.zeros((0, 4)), (2048, 2048), p_pos=1.0, rng=rng)
        assert crop.mode == "negative" and crop.boxes.shape == (0, 4)


def test_p_pos_selects_positive_or_negative():
    boxes = [[1000, 1000, 1100, 1100]]
    rng = np.random.default_rng(0)
    assert {sample_crop(boxes, (2048, 2048), p_pos=1.0, rng=rng).mode for _ in range(50)} == {"union"}
    assert {sample_crop(boxes, (2048, 2048), p_pos=0.0, rng=rng).mode for _ in range(50)} == {"negative"}


def test_unavoidable_ambiguity_is_rejected_and_tol_controls_it():
    # Image 512 high (y origin fixed at 0), 700 wide, two 200-px courts at x in [0, 200] and [400, 600].
    # Union = 600 px > 512.  Containing A forces x0 = 0: B visible 112/200 = 0.56.
    # Containing B forces x0 in [88, 188]: A visible (200 - x0)/200 in [0.06, 0.56].
    # Any crop (x0 <= 188) covers [188, 512], which overlaps both: no negative crop either.
    boxes = [[0, 100, 200, 300], [400, 100, 600, 300]]
    rng = np.random.default_rng(0)
    for p_pos in (0.0, 1.0):
        for _ in range(20):
            assert sample_crop(boxes, (512, 700), tol=0.9, p_pos=p_pos, rng=rng) is None
    # With tol = 0.5, containing A is valid (B kept at f = 0.56, clipped to the crop).
    crops = [sample_crop(boxes, (512, 700), tol=0.5, p_pos=1.0, rng=rng) for _ in range(50)]
    assert any(c is not None and c.pivot == 0 and len(c.indices) == 2 for c in crops)


def test_same_seed_same_crops():
    boxes = [[100, 100, 200, 200], [1500, 1500, 1600, 1600]]
    a = [sample_crop(boxes, (2048, 2048), rng=np.random.default_rng(3)) for _ in range(5)]
    b = [sample_crop(boxes, (2048, 2048), rng=np.random.default_rng(3)) for _ in range(5)]
    assert [c.origin for c in a] == [c.origin for c in b]


def test_invalid_parameters_raise():
    rng = np.random.default_rng(0)
    for kwargs in ({"tol": 0.0}, {"tol": 1.5}, {"p_pos": -0.1}, {"max_tries": 0}, {"crop_size": 0}):
        with pytest.raises(ValueError):
            sample_crop([[0, 0, 10, 10]], (2048, 2048), rng=rng, **kwargs)


# --- sample_crop: property test on synthetic annotations --------------------------------

def random_annotation(rng):
    """Synthetic image: a size among those seen in the data, 0 to 8 courts (some clustered)."""
    height, width = [(256, 256), (2048, 2048), (1024, 2048), (700, 900), (300, 1500)][rng.integers(5)]
    n = int(rng.integers(0, 9))
    sizes = rng.uniform(10, 300, size=(n, 2))
    sizes = np.minimum(sizes, [width - 1, height - 1])
    if rng.random() < 0.5:  # clustered courts: makes the "union" case frequent
        centre = rng.uniform([0, 0], [width, height])
        x1y1 = centre + rng.normal(0, 150, size=(n, 2)) - sizes / 2
    else:  # spread out: makes the "pivot" case and rejections frequent
        x1y1 = rng.uniform(0, 1, size=(n, 2)) * ([width, height] - sizes)
    x1y1 = np.clip(x1y1, 0, [width, height] - sizes)
    return np.concatenate([x1y1, x1y1 + sizes], axis=1), (height, width)


def test_sampled_crops_satisfy_the_visibility_rule():
    rng = np.random.default_rng(12345)
    modes = {"whole": 0, "union": 0, "pivot": 0, "negative": 0, None: 0}
    for _ in range(600):
        boxes, (height, width) = random_annotation(rng)
        tol = float(rng.choice([0.7, 0.9, 1.0]))
        for _ in range(10):
            crop = sample_crop(boxes, (height, width), tol=tol, p_pos=0.6, max_tries=10, rng=rng)
            if crop is None:
                modes[None] += 1
                continue
            modes[crop.mode] += 1
            x0, y0 = crop.origin
            assert isinstance(x0, int) and isinstance(y0, int)
            assert 0 <= x0 <= max(width - S, 0) and 0 <= y0 <= max(height - S, 0)

            # (a) every output box lies in [0, S] and is not empty
            assert crop.boxes.shape == (len(crop.indices), 4)
            assert (crop.boxes >= 0).all() and (crop.boxes <= S).all()
            assert ((crop.boxes[:, 2] > crop.boxes[:, 0]) & (crop.boxes[:, 3] > crop.boxes[:, 1])).all()

            f = visible_fraction(boxes, crop_window(crop.origin, S))
            # (b) no kept court below tol, and (c) no court in the ambiguous zone 0 < f < tol
            assert (f[crop.indices] >= tol).all()
            assert not ((f > 0) & (f < tol)).any()
            # kept = exactly the courts present in the crop, with the right coordinates
            assert crop.indices.tolist() == np.flatnonzero(f > 0).tolist()
            assert np.allclose(crop.boxes, clip_boxes_to_crop(boxes[crop.indices], crop.origin, S))

            # (d) the court drawn in case 2 is entirely inside the crop
            if crop.mode == "pivot":
                assert f[crop.pivot] == 1.0
            if crop.mode == "union":
                assert (f == 1.0).all()
            if crop.mode == "negative":
                assert len(crop.indices) == 0

    # the property test is only meaningful if every branch was exercised
    assert all(count > 0 for count in modes.values()), modes
