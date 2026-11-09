"""COCO dataset of basketball courts, served as random fixed-size crops at native resolution.

One class: label 1 = court (0 = background, implicit). COCO categories 1 (source A tiles) and
2 (source B scenes) are both courts; category 0 is an empty Roboflow parent category.

Per sample:
    image  float32 [3, S, S], ImageNet-normalised, zero-padded right / bottom when the crop
           extends past the image (0 after normalisation = the ImageNet mean colour);
    target {"boxes": float32 [N, 4] (x1, y1, x2, y2) in crop pixels, "labels": int64 [N]};
    meta   dict (image_id, file_name, source, crop origin and mode, rejected indices, ...).

Pipeline order in __getitem__:
    sample_crop (geometry) -> PIL crop inside the image -> tensor in [0, 1]
    -> transforms(image, target) -> normalise -> pad to S x S.
Transforms therefore see real pixels only, in [0, 1] (where colour jitter makes sense).

Randomness: a numpy Generator stored on the dataset would be copied identically into every
DataLoader worker (same crops in every worker). Instead each call draws a seed from torch's
global RNG, which the DataLoader seeds differently per worker; torch.manual_seed makes it
reproducible.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from courtdet.data.crops import sample_crop

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COURT_LABEL = 1

Transform = Callable[[Tensor, dict], tuple[Tensor, dict]]


def source_of(file_name: str) -> str:
    """"A" for the 256x256 `basketball_court_*` tiles (COCO category 1), "B" for everything else
    (large scenes, category 2). Checked on the v2 export: consistent with the categories on all splits."""
    return "A" if file_name.startswith("basketball_court") else "B"


def load_coco_records(
    annotation_file: str | Path,
    court_category_ids: tuple[int, ...] = (1, 2),
    ignored_category_ids: tuple[int, ...] = (0,),
) -> list[dict]:
    """One record per image: file_name, image_id, height, width, source, boxes [N, 4] (xyxy, image pixels).

    Boxes are clipped to the image: a few exported boxes overshoot the border by a fraction of a
    pixel, which would make them impossible to contain in any integer crop.
    """
    coco = json.loads(Path(annotation_file).read_text())
    boxes_per_image: dict[int, list] = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        cat = ann["category_id"]
        if cat in ignored_category_ids:
            continue
        if cat not in court_category_ids:
            raise ValueError(f"unexpected category_id {cat} in annotation {ann['id']}")
        x, y, w, h = ann["bbox"]  # COCO: top-left corner + size
        boxes_per_image[ann["image_id"]].append([x, y, x + w, y + h])

    records = []
    for img in coco["images"]:
        height, width = img["height"], img["width"]
        boxes = np.asarray(boxes_per_image[img["id"]], dtype=np.float64).reshape(-1, 4)
        boxes = boxes.clip(0, [width, height, width, height])
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        records.append(
            {
                "file_name": img["file_name"],
                "image_id": img["id"],
                "height": height,
                "width": width,
                "source": source_of(img["file_name"]),
                "boxes": boxes,
            }
        )
    return records


class CourtDataset(Dataset):
    """Random crops of `root/split/` (Roboflow COCO export: images + `_annotations.coco.json`).

    Args:
        root: folder containing the split folders.
        split: "train", "valid" or "test".
        crop_size, tol, p_pos, max_tries: crop rule, see `crops.sample_crop`.
        max_redraws: rejected crops tolerated for one __getitem__ before raising.
        transforms: optional `f(image [3, h, w] in [0, 1], target) -> (image, target)`,
            applied before normalisation and padding. None = no augmentation.
        court_category_ids, ignored_category_ids: COCO category mapping (see `load_coco_records`).
        records: records to use instead of reading the annotation file (e.g. after
            `corrections.select_records`); the images are still read from root/split.

    When `sample_crop` rejects an image (no crop without an ambiguous court), another index is
    drawn uniformly. Rejections are counted per index in `rejection_counts` (reliable only with
    num_workers=0: each worker owns a copy of the dataset) and also returned in each sample's
    meta["rejected_indices"], which survives the workers.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        crop_size: int = 512,
        tol: float = 0.85,
        p_pos: float = 1.0,
        max_tries: int = 20,
        max_redraws: int = 100,
        transforms: Transform | None = None,
        court_category_ids: tuple[int, ...] = (1, 2),
        ignored_category_ids: tuple[int, ...] = (0,),
        records: list[dict] | None = None,
    ):
        self.image_dir = Path(root) / split
        if records is None:
            records = load_coco_records(
                self.image_dir / "_annotations.coco.json", court_category_ids, ignored_category_ids
            )
        self.records = records
        self.crop_kwargs = {"crop_size": crop_size, "tol": tol, "p_pos": p_pos, "max_tries": max_tries}
        self.crop_size = crop_size
        self.max_redraws = max_redraws
        self.transforms = transforms
        self.rejection_counts: Counter[int] = Counter()
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def n_rejections(self) -> int:
        return sum(self.rejection_counts.values())

    def __getitem__(self, index: int) -> tuple[Tensor, dict, dict]:
        rejected = []
        for _ in range(self.max_redraws + 1):
            record = self.records[index]
            rng = np.random.default_rng(int(torch.randint(0, 2**62, ()).item()))
            crop = sample_crop(record["boxes"], (record["height"], record["width"]), rng=rng, **self.crop_kwargs)
            if crop is not None:
                break
            self.rejection_counts[index] += 1
            rejected.append(index)
            index = int(torch.randint(0, len(self), ()).item())
        else:
            raise RuntimeError(f"{self.max_redraws + 1} crops rejected in a row; last indices {rejected[-5:]}")

        image = self._load_crop(record, crop.origin)
        target = {
            "boxes": torch.as_tensor(crop.boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.full((len(crop.boxes),), COURT_LABEL, dtype=torch.int64),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        image = (image - self.mean) / self.std
        _, h, w = image.shape
        if h > self.crop_size or w > self.crop_size:
            raise ValueError(f"image {h}x{w} larger than the crop size after transforms")
        image = F.pad(image, (0, self.crop_size - w, 0, self.crop_size - h), value=0.0)

        meta = {
            "index": index,
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "source": record["source"],
            "image_size": (record["height"], record["width"]),
            "crop_origin": crop.origin,
            "crop_mode": crop.mode,
            "rejected_indices": rejected,
        }
        return image, target, meta

    def _load_crop(self, record: dict, origin: tuple[int, int]) -> Tensor:
        """Crop of the image as float [3, h, w] in [0, 1]; h, w <= crop_size (no padding here)."""
        with Image.open(self.image_dir / record["file_name"]) as img:
            if img.size != (record["width"], record["height"]):  # PIL size is (width, height)
                raise ValueError(f"{record['file_name']}: size {img.size} differs from the annotation file")
            x0, y0 = origin
            box = (x0, y0, min(x0 + self.crop_size, record["width"]), min(y0 + self.crop_size, record["height"]))
            pixels = img.crop(box).convert("RGB")
        return pil_to_tensor(pixels).float() / 255.0


def collate_fn(batch: list[tuple[Tensor, dict, dict]]) -> tuple[Tensor, list[dict], list[dict]]:
    """Stack the images (all S x S); keep targets and metas as lists (variable number of boxes)."""
    images, targets, metas = zip(*batch)
    return torch.stack(images), list(targets), list(metas)
