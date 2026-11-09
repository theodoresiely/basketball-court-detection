"""Random fixed-size crops at native resolution, with a "no ambiguous court" rule.

Why crops: images are kept at native resolution (resizing large scenes to 512 shrinks the
courts below the anchor sizes), so a large scene is seen through a
random crop_size x crop_size window.

Why the visibility rule: for a court cut by the crop border, let

    f = area(box & crop) / area(box)        (visible fraction, in [0, 1])

A crop is valid iff every court has f == 0 (absent) or f >= tol (kept, box clipped to the
crop). A court with 0 < f < tol would be a contradictory label: kept, the network must call
half a court a court; dropped, its visible pixels become "background". Such crops are
rejected and re-drawn.

Conventions (same as the rest of the package):
- boxes are (x1, y1, x2, y2), continuous, width = x2 - x1, positive area;
- `image_size` is (height, width), as in `models.box_coder.clip_boxes`;
- a crop with origin (x0, y0) is the window [x0, x0 + S] x [y0, y0 + S] in image pixels.
  Origins are integers because the crop is a pixel slice.

Images smaller than the crop (source A tiles, 256x256): on that axis the only origin is 0 and
the window extends past the right / bottom edge; the Dataset pads there. Boxes therefore keep
their coordinates.

Everything here is pure geometry (numpy only, no image), so it can be tested exhaustively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Range = tuple[tuple[int, int], tuple[int, int]]  # ((x_lo, x_hi), (y_lo, y_hi)), inclusive


@dataclass(frozen=True)
class Crop:
    origin: tuple[int, int]  # (x0, y0): top-left corner of the crop, image pixels
    boxes: np.ndarray  # [K, 4] kept boxes in crop coordinates, clipped to [0, crop_size]
    indices: np.ndarray  # [K] index of each kept box in the input boxes
    mode: str  # "whole" | "union" | "pivot" | "negative" (which rule produced the crop)
    pivot: int | None = None  # "pivot" mode: index of the box drawn to be fully contained


def _as_boxes(boxes) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def crop_window(origin: tuple[int, int], crop_size: int) -> np.ndarray:
    """The crop as a box (x1, y1, x2, y2) in image coordinates."""
    x0, y0 = origin
    return np.array([x0, y0, x0 + crop_size, y0 + crop_size], dtype=np.float64)


def visible_fraction(boxes, crop) -> np.ndarray | float:
    """area(box & crop) / area(box), for one box [4] (returns a float) or many [N, 4] (returns [N])."""
    single = np.ndim(boxes) == 1
    b = _as_boxes(boxes)
    c = np.asarray(crop, dtype=np.float64)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    if (area <= 0).any():
        raise ValueError("boxes must have a positive area")
    iw = (np.minimum(b[:, 2], c[2]) - np.maximum(b[:, 0], c[0])).clip(min=0)
    ih = (np.minimum(b[:, 3], c[3]) - np.maximum(b[:, 1], c[1])).clip(min=0)
    f = iw * ih / area
    return float(f[0]) if single else f


def admissible_origin_range(box, crop_size: int, image_size: tuple[int, int]) -> Range | None:
    """Integer origins whose crop contains `box` entirely and stays in the image.

    Per axis: [ceil(x2 - S), floor(x1)] & [0, max(W - S, 0)]. The `max(., 0)` handles images
    smaller than the crop (only origin 0). Returns None if the range is empty on either axis
    (box larger than the crop, or rounding of fractional coordinates leaves no integer).
    """
    height, width = image_size
    x1, y1, x2, y2 = (float(v) for v in box)
    out = []
    for lo_box, hi_box, length in ((x1, x2, width), (y1, y2, height)):
        lo = math.ceil(max(hi_box - crop_size, 0.0))
        hi = math.floor(min(lo_box, max(length - crop_size, 0)))
        if lo > hi:
            return None
        out.append((lo, hi))
    return out[0], out[1]


def clip_boxes_to_crop(boxes, origin: tuple[int, int], crop_size: int) -> np.ndarray:
    """Shift boxes into crop coordinates (origin -> (0, 0)) and clip them to [0, crop_size]."""
    b = _as_boxes(boxes) - np.array([origin[0], origin[1], origin[0], origin[1]], dtype=np.float64)
    return b.clip(0, crop_size)


def _draw_origin(ranges: Range, rng: np.random.Generator) -> tuple[int, int]:
    (x_lo, x_hi), (y_lo, y_hi) = ranges
    return int(rng.integers(x_lo, x_hi + 1)), int(rng.integers(y_lo, y_hi + 1))


def _try_crop(boxes, origin, crop_size, tol, mode, pivot=None) -> Crop | None:
    """Apply the visibility rule to one candidate origin: a Crop if valid, else None."""
    f = visible_fraction(boxes, crop_window(origin, crop_size))
    if ((f > 0) & (f < tol)).any():
        return None
    keep = np.flatnonzero(f >= tol)
    return Crop(origin, clip_boxes_to_crop(boxes[keep], origin, crop_size), keep, mode, pivot)


def sample_positive_crop(
    boxes, image_size: tuple[int, int], *, crop_size: int, tol: float, max_tries: int, rng: np.random.Generator
) -> Crop | None:
    """A valid crop containing at least one court entirely, or None.

    Case 1 ("union"): if the bounding box of all courts fits in a crop, draw a crop containing
    all of them (always valid, no retry needed).
    Case 2 ("pivot"): otherwise draw one court that can fit, draw a crop containing it entirely,
    check the others with the visibility rule; retry at most `max_tries` times.
    """
    boxes = _as_boxes(boxes)
    if len(boxes) == 0:
        return None
    union = [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()]
    union_range = admissible_origin_range(union, crop_size, image_size)
    if union_range is not None:
        return _try_crop(boxes, _draw_origin(union_range, rng), crop_size, tol, "union")

    ranges = [admissible_origin_range(b, crop_size, image_size) for b in boxes]
    candidates = [i for i, r in enumerate(ranges) if r is not None]
    if not candidates:  # every court is larger than the crop
        return None
    for _ in range(max_tries):
        i = candidates[int(rng.integers(len(candidates)))]
        crop = _try_crop(boxes, _draw_origin(ranges[i], rng), crop_size, tol, "pivot", pivot=i)
        if crop is not None:
            return crop
    return None


def sample_negative_crop(
    boxes, image_size: tuple[int, int], *, crop_size: int, max_tries: int, rng: np.random.Generator
) -> Crop | None:
    """A crop touching no court at all (every f == 0), uniform over the image, or None."""
    boxes = _as_boxes(boxes)
    height, width = image_size
    full_range = ((0, max(width - crop_size, 0)), (0, max(height - crop_size, 0)))
    for _ in range(max_tries):
        origin = _draw_origin(full_range, rng)
        f = visible_fraction(boxes, crop_window(origin, crop_size))
        if not (f > 0).any():
            return Crop(origin, np.zeros((0, 4)), np.zeros(0, dtype=np.int64), "negative")
    return None


def sample_crop(
    boxes,
    image_size: tuple[int, int],
    *,
    crop_size: int = 512,
    tol: float = 0.85,
    p_pos: float = 1.0,
    max_tries: int = 20,
    rng: np.random.Generator,
) -> Crop | None:
    """Draw one crop of an image following the visibility rule; None if rejected.

    - Image no larger than the crop on both axes: only one crop exists, the whole image
      (padded right / bottom by the Dataset), mode "whole"; `p_pos` does not apply.
    - Otherwise, an image with courts gives a positive crop with probability `p_pos`
      (see `sample_positive_crop`), else a negative one; an image without courts always
      gives a negative crop.

    Args:
        boxes: [N, 4] courts in image pixels (N may be 0).
        image_size: (height, width).
        crop_size: side S of the square crop.
        tol: minimum visible fraction for a cut court to be kept.
        p_pos: probability of a positive crop for an image with courts.
        max_tries: re-draws allowed before giving up (case 2 and negatives only).
        rng: numpy Generator, required so that crops are reproducible.
    """
    if crop_size <= 0 or max_tries < 1:
        raise ValueError("crop_size must be > 0 and max_tries >= 1")
    if not 0 < tol <= 1 or not 0 <= p_pos <= 1:
        raise ValueError("expected 0 < tol <= 1 and 0 <= p_pos <= 1")
    boxes = _as_boxes(boxes)
    height, width = image_size
    if height <= crop_size and width <= crop_size:
        return _try_crop(boxes, (0, 0), crop_size, tol, "whole")
    if len(boxes) > 0 and rng.random() < p_pos:
        return sample_positive_crop(boxes, image_size, crop_size=crop_size, tol=tol, max_tries=max_tries, rng=rng)
    return sample_negative_crop(boxes, image_size, crop_size=crop_size, max_tries=max_tries, rng=rng)
