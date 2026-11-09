import pickle

import pytest
import torch

from courtdet.data.augment import (
    ColorJitter,
    hflip,
    make_train_transform,
    random_dihedral,
    rot90,
)


def painted(h=40, w=64, box=(5, 10, 20, 16)):
    """Image whose only non-zero pixels are exactly the integer box (x1, y1, x2, y2)."""
    image = torch.zeros(3, h, w)
    x1, y1, x2, y2 = box
    image[:, y1:y2, x1:x2] = 1.0
    return image, torch.tensor([box], dtype=torch.float32)


def mask_box(image):
    """Box (x1, y1, x2, y2) enclosing the non-zero pixels (continuous convention, no +1)."""
    ys, xs = torch.nonzero(image[0] > 0.5, as_tuple=True)
    return torch.tensor([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=torch.float32)


@pytest.mark.parametrize("op", [rot90, hflip])
def test_box_follows_the_pixels(op):
    image, boxes = painted()  # not square: catches a mix-up between h and w
    out, out_boxes = op(image, boxes)
    assert torch.equal(out_boxes, mask_box(out))


def test_rot90_swaps_height_and_width():
    image, boxes = painted(h=40, w=64)
    out, _ = rot90(image, boxes)
    assert out.shape == (3, 64, 40)


def test_random_dihedral_reaches_the_8_symmetries_with_exact_boxes():
    image, boxes = painted()
    torch.manual_seed(0)
    seen = set()
    for _ in range(200):
        out, target = random_dihedral(image, {"boxes": boxes, "labels": torch.ones(1, dtype=torch.int64)})
        assert torch.equal(target["boxes"], mask_box(out))
        assert target["labels"].tolist() == [1]
        seen.add(tuple(target["boxes"][0].tolist()))
    assert len(seen) == 8  # the box is not symmetric: 8 different positions


def test_random_dihedral_without_boxes():
    out, target = random_dihedral(torch.rand(3, 30, 50), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0)})
    assert target["boxes"].shape == (0, 4)
    assert out.shape in {(3, 30, 50), (3, 50, 30)}


def test_color_jitter_zero_is_identity():
    image = torch.rand(3, 20, 30)
    out, _ = ColorJitter()(image, {})
    assert torch.allclose(out, image, atol=1e-6)


def test_color_jitter_keeps_range_and_boxes():
    torch.manual_seed(0)
    image, boxes = painted()
    image = image * 0.5 + torch.rand(3, 40, 64) * 0.5
    t = ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9)
    for _ in range(20):
        out, target = t(image, {"boxes": boxes})
        assert out.shape == image.shape and out.min() >= 0 and out.max() <= 1
        assert torch.equal(target["boxes"], boxes)


def test_color_jitter_gray():
    out, _ = ColorJitter(p_gray=1.0)(torch.rand(3, 10, 10), {})
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])


def test_color_jitter_rejects_bad_values():
    with pytest.raises(ValueError):
        ColorJitter(brightness=1.0)
    with pytest.raises(ValueError):
        ColorJitter(p_gray=1.5)


def test_make_train_transform_off_is_none():
    assert make_train_transform() is None
    assert make_train_transform(dihedral=True) is not None
    assert make_train_transform(p_gray=0.1) is not None


def test_transform_survives_pickle():
    """DataLoader workers (spawn) receive the dataset and its transform through pickle."""
    t = pickle.loads(pickle.dumps(make_train_transform(True, 0.3, 0.3, 0.5, 0.1)))
    image, boxes = painted()
    out, target = t(image, {"boxes": boxes})
    assert target["boxes"].shape == (1, 4)
