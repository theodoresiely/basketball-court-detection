"""Box coder: conversion between boxes and regression targets (deltas).

The network never predicts box coordinates directly. It predicts how to move
and rescale a reference box (an anchor for the RPN, a proposal for the RoI head):

    tx = wx * (cx_t - cx_r) / w_r        ty = wy * (cy_t - cy_r) / h_r
    tw = ww * log(w_t / w_r)             th = wh * log(h_t / h_r)

where r is the reference box and t the target box. Why this parametrisation:
- dividing the shift by the reference size makes the target scale-invariant
  (a 10 px shift means a lot for a 32 px anchor, little for a 128 px one);
- the log makes the size change additive and symmetric (x2 and /2 give +log2 and
  -log2) and guarantees a positive width after decoding (exp > 0).

The optional weights (wx, wy, ww, wh) rescale the targets so that their typical
magnitude is close to 1; torchvision uses (1, 1, 1, 1) for the RPN and
(10, 10, 5, 5) for the RoI head.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

# Upper bound on the predicted log-scale before exp(): avoids overflow when an
# untrained network outputs huge values. Same value as torchvision.
DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


def _center_size(boxes: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    cx = boxes[..., 0] + 0.5 * w
    cy = boxes[..., 1] + 0.5 * h
    return cx, cy, w, h


def encode(
    reference: Tensor,
    target: Tensor,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> Tensor:
    """Deltas that transform `reference` boxes into `target` boxes.

    Args:
        reference: [..., 4] anchors or proposals (x1, y1, x2, y2), positive size.
        target: [..., 4] ground-truth boxes matched one-to-one with `reference`,
            positive size (a zero-width box gives log(0) = -inf).

    Returns:
        [..., 4] tensor of (tx, ty, tw, th).
    """
    cx_r, cy_r, w_r, h_r = _center_size(reference)
    cx_t, cy_t, w_t, h_t = _center_size(target)
    wx, wy, ww, wh = weights
    return torch.stack(
        [
            wx * (cx_t - cx_r) / w_r,
            wy * (cy_t - cy_r) / h_r,
            ww * torch.log(w_t / w_r),
            wh * torch.log(h_t / h_r),
        ],
        dim=-1,
    )


def decode(
    reference: Tensor,
    deltas: Tensor,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    scale_clamp: float = DEFAULT_SCALE_CLAMP,
) -> Tensor:
    """Inverse of `encode`: apply predicted deltas to reference boxes.

    Returns:
        [..., 4] boxes (x1, y1, x2, y2). Not clipped to the image: clipping is
        the caller's job because it needs the image size.
    """
    cx_r, cy_r, w_r, h_r = _center_size(reference)
    wx, wy, ww, wh = weights
    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = (deltas[..., 2] / ww).clamp(max=scale_clamp)
    dh = (deltas[..., 3] / wh).clamp(max=scale_clamp)

    cx = cx_r + dx * w_r
    cy = cy_r + dy * h_r
    w = w_r * torch.exp(dw)
    h = h_r * torch.exp(dh)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def clip_boxes(boxes: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Clamp boxes (x1, y1, x2, y2) to the image. `image_size` is (height, width).

    Boxes lying entirely outside the image end up with zero width or height:
    callers should drop them.
    """
    height, width = image_size
    x1 = boxes[..., 0].clamp(0, width)
    y1 = boxes[..., 1].clamp(0, height)
    x2 = boxes[..., 2].clamp(0, width)
    y2 = boxes[..., 3].clamp(0, height)
    return torch.stack([x1, y1, x2, y2], dim=-1)
