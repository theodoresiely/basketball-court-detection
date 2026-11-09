"""Anchor generation for the Region Proposal Network (RPN).

Coordinate convention used everywhere in this package
-----------------------------------------------------
Boxes are (x1, y1, x2, y2) in *continuous* pixel coordinates:
width = x2 - x1, height = y2 - y1 (no "+1").
This is the convention of torchvision.ops (box_iou, nms, roi_align), which we
rely on. The original notebook mixed the legacy "+1" convention (inherited from
old Caffe code) with torchvision ops; we drop it.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def generate_base_anchors(
    sizes: Sequence[float],
    aspect_ratios: Sequence[float],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the k = len(sizes) * len(aspect_ratios) reference anchors centred on (0, 0).

    Args:
        sizes: square-root of the anchor area, in pixels (e.g. (32, 64, 128)).
        aspect_ratios: width / height ratios (e.g. (0.5, 1.0, 2.0)).

    Each anchor keeps the area size**2 whatever its ratio r:
        w = size * sqrt(r),  h = size / sqrt(r)   ->  w * h = size**2,  w / h = r.

    Order: sizes in the outer loop, ratios in the inner loop, i.e.
    anchor index a = i_size * len(aspect_ratios) + i_ratio.

    Returns:
        Tensor of shape [k, 4] in (x1, y1, x2, y2) format.
    """
    sizes_t = torch.as_tensor(sizes, dtype=dtype)
    ratios_t = torch.as_tensor(aspect_ratios, dtype=dtype)
    s, r = torch.meshgrid(sizes_t, ratios_t, indexing="ij")  # both [n_sizes, n_ratios]
    w = (s * r.sqrt()).reshape(-1)
    h = (s / r.sqrt()).reshape(-1)
    return torch.stack([-w / 2, -h / 2, w / 2, h / 2], dim=1)


def grid_anchors(feature_size: tuple[int, int], stride: int, base_anchors: Tensor) -> Tensor:
    """Tile the k base anchors on every cell of an (H, W) feature map.

    Feature cell (i, j) "sees" the image pixels [j*stride, (j+1)*stride) horizontally
    and [i*stride, (i+1)*stride) vertically, so its centre in image coordinates is
    ((j + 0.5) * stride, (i + 0.5) * stride).

    Returns:
        Tensor of shape [H * W * k, 4], ordered by (row i, column j, anchor a):
            index = (i * W + j) * k + a
        This ordering MUST match the way the RPN head flattens its outputs,
        `x.permute(0, 2, 3, 1).reshape(B, H * W * k, C)`. If the two disagree,
        nothing crashes, but every prediction is compared to the wrong anchor.
    """
    H, W = feature_size
    device, dtype = base_anchors.device, base_anchors.dtype
    cy = (torch.arange(H, device=device, dtype=dtype) + 0.5) * stride
    cx = (torch.arange(W, device=device, dtype=dtype) + 0.5) * stride
    cy, cx = torch.meshgrid(cy, cx, indexing="ij")  # both [H, W]
    shifts = torch.stack([cx, cy, cx, cy], dim=-1).reshape(-1, 1, 4)  # [H*W, 1, 4]
    return (shifts + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)  # [H*W*k, 4]
