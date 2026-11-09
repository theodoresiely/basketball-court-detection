"""Online augmentations of the training crops.

Why: the YOLO baseline loses 0.124 AP@[.5:.95] without its online augmentations, more than its
lead over our model. Our training had none.

Each transform has the `CourtDataset` signature `f(image [3, h, w] in [0, 1], target) -> (image, target)`
and is applied to the real pixels of the crop, before normalisation and padding (h, w <= crop size).
Randomness comes from torch's global RNG, seeded per DataLoader worker (see dataset.py).

- `random_dihedral`: one of the 8 symmetries of the square (4 rotations by 90 degrees x optional
  flip). A satellite image has no "up": the 8 versions are equally plausible scenes. An axis-aligned
  box stays axis-aligned, so its new coordinates are exact (no box enlargement, unlike a free rotation).
- `ColorJitter`: brightness, contrast and saturation factors drawn uniformly in [1 - d, 1 + d],
  then, with probability p_gray, conversion to grey levels (part of the small missed courts are on
  grey-level images).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

Transform = Callable[[Tensor, dict], tuple[Tensor, dict]]

GRAY_WEIGHTS = (0.299, 0.587, 0.114)  # ITU-R 601 luma, the usual RGB -> grey conversion


def rot90(image: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate by 90 degrees counter-clockwise. A point (x, y) of a w-wide image goes to (y, w - x)."""
    w = image.shape[-1]
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.rot90(image, 1, dims=(-2, -1)), torch.stack([y1, w - x2, y2, w - x1], dim=-1)


def hflip(image: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Mirror left <-> right. A point (x, y) goes to (w - x, y)."""
    w = image.shape[-1]
    x1, y1, x2, y2 = boxes.unbind(-1)
    return image.flip(-1), torch.stack([w - x2, y1, w - x1, y2], dim=-1)


def random_dihedral(image: Tensor, target: dict) -> tuple[Tensor, dict]:
    boxes = target["boxes"]
    for _ in range(int(torch.randint(0, 4, ()))):
        image, boxes = rot90(image, boxes)
    if torch.rand(()) < 0.5:
        image, boxes = hflip(image, boxes)
    return image, {**target, "boxes": boxes}


def to_gray(image: Tensor) -> Tensor:
    """[3, h, w] -> [1, h, w] grey levels."""
    return (image * torch.tensor(GRAY_WEIGHTS).view(3, 1, 1)).sum(0, keepdim=True)


class ColorJitter:
    """Classes rather than closures: DataLoader workers are spawned processes on macOS, and the
    transform reaches them through pickle, which cannot serialise a function defined inside another."""

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0, p_gray: float = 0.0):
        for name, value in (("brightness", brightness), ("contrast", contrast), ("saturation", saturation)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if not 0.0 <= p_gray <= 1.0:
            raise ValueError(f"p_gray must be in [0, 1], got {p_gray}")
        self.brightness, self.contrast, self.saturation, self.p_gray = brightness, contrast, saturation, p_gray

    @staticmethod
    def factor(d: float) -> float:
        return float(1.0 + d * (2 * torch.rand(()) - 1))

    def __call__(self, image: Tensor, target: dict) -> tuple[Tensor, dict]:
        image = image * self.factor(self.brightness)
        mean = to_gray(image).mean()
        image = mean + (image - mean) * self.factor(self.contrast)  # stretch around the mean grey level
        gray = to_gray(image)
        image = gray + (image - gray) * self.factor(self.saturation)  # 0 = grey, 1 = unchanged
        if torch.rand(()) < self.p_gray:
            image = to_gray(image).expand(3, -1, -1)
        return image.clamp(0.0, 1.0), target


class Compose:
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, image: Tensor, target: dict) -> tuple[Tensor, dict]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def make_train_transform(dihedral: bool = False, brightness: float = 0.0, contrast: float = 0.0,
                         saturation: float = 0.0, p_gray: float = 0.0) -> Transform | None:
    """Training transform from the config values; None when everything is off (old behaviour)."""
    transforms: list[Transform] = []
    if dihedral:
        transforms.append(random_dihedral)
    if brightness or contrast or saturation or p_gray:
        transforms.append(ColorJitter(brightness, contrast, saturation, p_gray))
    return Compose(transforms) if transforms else None  # None: the Dataset draws no random number
