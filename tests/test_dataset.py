"""CourtDataset on a tiny synthetic COCO export written to tmp_path (plus one test on the real data, if present)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from courtdet.data.dataset import IMAGENET_MEAN, IMAGENET_STD, CourtDataset, collate_fn

S = 512
CATEGORIES = [
    {"id": 0, "name": "basket-courts", "supercategory": "none"},
    {"id": 1, "name": "basketball", "supercategory": "basket-courts"},
    {"id": 2, "name": "basketball-court", "supercategory": "basket-courts"},
]


def gradient_image(height, width):
    """R = x mod 256, G = y mod 256: the pixel values tell where a crop was taken."""
    x = np.arange(width) % 256
    y = np.arange(height) % 256
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = x[None, :]
    rgb[..., 1] = y[:, None]
    rgb[..., 2] = 128
    return Image.fromarray(rgb)


def write_coco(root: Path, split: str, images: list[tuple[str, int, int, list]]):
    """images: (file_name, height, width, [(category_id, [x, y, w, h]), ...]). PNG = lossless."""
    folder = root / split
    folder.mkdir(parents=True)
    coco = {"categories": CATEGORIES, "images": [], "annotations": []}
    for image_id, (name, height, width, anns) in enumerate(images):
        gradient_image(height, width).save(folder / name, format="PNG")
        coco["images"].append({"id": image_id, "file_name": name, "height": height, "width": width})
        for cat, bbox in anns:
            coco["annotations"].append(
                {"id": len(coco["annotations"]), "image_id": image_id, "category_id": cat, "bbox": bbox}
            )
    (folder / "_annotations.coco.json").write_text(json.dumps(coco))


MINI = [
    ("basketball_court_1_jpg.rf.a.png", 256, 256, [(1, [30, 40, 80, 50])]),  # 0: source A
    ("basketball_court_2_jpg.rf.b.png", 256, 256, []),  # 1: source A, no court
    ("P0001_png.rf.c.png", 900, 1200, [(2, [100, 100, 60, 60]), (2, [200, 150, 40, 80]), (0, [500, 500, 10, 10])]),  # 2
    ("P0002_png.rf.d.png", 700, 800, []),  # 3: source B, true negative
    ("Screenshot-1_png.rf.e.png", 600, 700, [(2, [650, 500, 50.3, 100.2])]),  # 4: box overshoots by 0.3 / 0.2 px
]
# Two 200-px courts that no 512 crop can separate (see test_crops.py): always rejected.
AMBIGUOUS = ("P0003_png.rf.f.png", 512, 700, [(2, [0, 100, 200, 200]), (2, [400, 100, 200, 200])])


@pytest.fixture
def mini_root(tmp_path):
    write_coco(tmp_path, "train", MINI)
    return tmp_path


def unnormalise(image):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((image * std + mean) * 255).round()


def test_category_mapping_and_sources(mini_root):
    ds = CourtDataset(mini_root, "train")
    assert [len(r["boxes"]) for r in ds.records] == [1, 0, 2, 0, 1]  # category 0 ignored
    assert [r["source"] for r in ds.records] == ["A", "A", "B", "B", "B"]
    # COCO [x, y, w, h] -> [x1, y1, x2, y2]
    assert ds.records[2]["boxes"].tolist() == [[100, 100, 160, 160], [200, 150, 240, 230]]
    torch.manual_seed(0)
    for i in range(len(ds)):
        _, target, _ = ds[i]
        assert (target["labels"] == 1).all()


def test_boxes_overshooting_the_image_are_clipped(mini_root):
    ds = CourtDataset(mini_root, "train")
    assert ds.records[4]["boxes"].tolist() == [[650, 500, 700, 600]]


def test_unexpected_category_raises(tmp_path):
    write_coco(tmp_path, "train", [("P0009_png.png", 600, 600, [(7, [0, 0, 10, 10])])])
    with pytest.raises(ValueError):
        CourtDataset(tmp_path, "train")


def test_output_shapes_and_dtypes(mini_root):
    ds = CourtDataset(mini_root, "train")  # transforms=None by default
    torch.manual_seed(0)
    for i in range(len(ds)):
        image, target, meta = ds[i]
        n = len(ds.records[meta["index"]]["boxes"])
        assert image.shape == (3, S, S) and image.dtype == torch.float32
        assert target["boxes"].dtype == torch.float32 and target["labels"].dtype == torch.int64
        assert target["boxes"].shape[1] == 4 and len(target["labels"]) == len(target["boxes"]) <= n


def test_images_without_courts(mini_root):
    ds = CourtDataset(mini_root, "train")
    torch.manual_seed(0)
    for i in (1, 3):
        _, target, meta = ds[i]
        assert target["boxes"].shape == (0, 4) and target["labels"].shape == (0,)
    assert meta["crop_mode"] == "negative"  # image 3 is larger than the crop


def test_small_image_is_padded_right_and_bottom_with_the_mean(mini_root):
    image, target, meta = CourtDataset(mini_root, "train")[0]
    assert meta["crop_origin"] == (0, 0) and meta["crop_mode"] == "whole"
    assert target["boxes"].tolist() == [[30, 40, 110, 90]]  # unchanged
    assert (image[:, 256:, :] == 0).all() and (image[:, :, 256:] == 0).all()  # 0 = mean after normalisation
    pixels = unnormalise(image[:, :256, :256])
    assert (pixels[0] == torch.arange(256)[None, :]).all()  # R = x
    assert (pixels[1] == torch.arange(256)[:, None]).all()  # G = y


def test_pixels_and_boxes_come_from_the_same_crop(mini_root):
    ds = CourtDataset(mini_root, "train")
    torch.manual_seed(0)
    origins = set()
    for _ in range(20):
        image, target, meta = ds[2]
        x0, y0 = meta["crop_origin"]
        origins.add((x0, y0))
        pixels = unnormalise(image)
        assert (pixels[0] == (x0 + torch.arange(S))[None, :] % 256).all()
        assert (pixels[1] == (y0 + torch.arange(S))[:, None] % 256).all()
        # both courts fit in one crop ("union" mode): boxes are the originals shifted by the origin
        expected = torch.tensor([[100, 100, 160, 160], [200, 150, 240, 230]]) - torch.tensor([x0, y0, x0, y0])
        assert torch.equal(target["boxes"], expected.float())
    assert len(origins) > 1  # crops are actually random


def test_transforms_see_the_unpadded_crop_in_0_1(mini_root):
    seen = {}

    def hflip(image, target):
        seen["shape"], seen["range"] = tuple(image.shape), (image.min().item(), image.max().item())
        width = image.shape[-1]
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image.flip(-1), {**target, "boxes": boxes}

    _, target, _ = CourtDataset(mini_root, "train", transforms=hflip)[0]
    assert seen["shape"] == (3, 256, 256)
    assert 0 <= seen["range"][0] and seen["range"][1] <= 1
    assert target["boxes"].tolist() == [[146, 40, 226, 90]]  # x1' = 256 - x2, x2' = 256 - x1


def test_rejected_image_is_replaced_and_counted(tmp_path):
    write_coco(tmp_path, "train", [AMBIGUOUS, MINI[3]])
    ds = CourtDataset(tmp_path, "train")
    torch.manual_seed(0)
    _, _, meta = ds[0]
    assert meta["index"] == 1 and meta["file_name"] == MINI[3][0]
    assert meta["rejected_indices"][0] == 0 and set(meta["rejected_indices"]) == {0}
    assert ds.rejection_counts[0] == len(meta["rejected_indices"]) == ds.n_rejections


def test_endless_rejection_raises(tmp_path):
    write_coco(tmp_path, "train", [AMBIGUOUS])
    with pytest.raises(RuntimeError):
        CourtDataset(tmp_path, "train", max_redraws=3)[0]


def test_same_torch_seed_same_crops(mini_root):
    ds = CourtDataset(mini_root, "train")
    runs = []
    for _ in range(2):
        torch.manual_seed(42)
        runs.append([ds[i][2]["crop_origin"] for i in range(len(ds))])
    assert runs[0] == runs[1]


def test_collate_and_dataloader(mini_root):
    ds = CourtDataset(mini_root, "train")
    images, targets, metas = next(iter(DataLoader(ds, batch_size=5, collate_fn=collate_fn)))
    assert images.shape == (5, 3, S, S)
    assert len(targets) == len(metas) == 5
    assert [len(t["boxes"]) for t in targets][:2] == [1, 0]


def test_dataloader_with_workers(mini_root):
    # The dataset must survive pickling into worker processes (spawn start method on macOS).
    loader = DataLoader(CourtDataset(mini_root, "train"), batch_size=2, num_workers=2, collate_fn=collate_fn)
    assert sum(len(images) for images, _, _ in loader) == len(MINI)


# --- real data ------------------------------------------------------------------------

REAL_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "basket_dataset.coco"


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="real dataset not present")
def test_real_dataset():
    # expected counts measured on the v2 COCO export
    expected = {"train": 2132, "valid": 303, "test": 152}
    for split, n_images in expected.items():
        ds = CourtDataset(REAL_ROOT, split)
        assert len(ds) == n_images
        for r in ds.records:  # header only: cheap; catches EXIF-rotated or mis-sized files
            with Image.open(ds.image_dir / r["file_name"]) as img:
                assert img.size == (r["width"], r["height"])

    ds = CourtDataset(REAL_ROOT, "train")
    assert sum(len(r["boxes"]) for r in ds.records) == 996 + 879  # categories 1 + 2
    assert sum(r["source"] == "A" for r in ds.records) == 980
    assert sum(r["source"] == "B" for r in ds.records) == 1152

    torch.manual_seed(0)
    for i in torch.randperm(len(ds))[:20].tolist():
        image, target, _ = ds[i]
        assert image.shape == (3, S, S)
        boxes = target["boxes"]
        assert (boxes >= 0).all() and (boxes <= S).all() and (target["labels"] == 1).all()
