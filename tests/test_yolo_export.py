import numpy as np
from PIL import Image

from courtdet.data.yolo_export import PAD_VALUE, crop_canvas, export_split, yolo_lines


def test_yolo_lines_are_normalised_centre_size():
    assert yolo_lines([[0, 0, 512, 256], [100, 200, 150, 300]], 512) == [
        "0 0.500000 0.250000 1.000000 0.500000",
        "0 0.244141 0.488281 0.097656 0.195312",
    ]
    assert yolo_lines(np.zeros((0, 4)), 512) == []


def test_crop_canvas_keeps_native_pixels_and_pads_outside():
    image = Image.new("RGB", (256, 256), (10, 20, 30))
    canvas = crop_canvas(image, (0, 0), 512)
    assert canvas.size == (512, 512)
    assert canvas.getpixel((255, 255)) == (10, 20, 30)  # not resized
    assert canvas.getpixel((256, 0)) == (PAD_VALUE,) * 3


def _round_trip(path, size=512):
    lines = path.read_text().split("\n") if path.read_text() else []
    out = []
    for line in lines:
        c, cx, cy, w, h = line.split()
        assert c == "0"
        cx, cy, w, h = (float(v) * size for v in (cx, cy, w, h))
        out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(out).reshape(-1, 4)


def test_export_split_tile_and_scene(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (256, 256)).save(src / "basketball_court_1.jpg")
    Image.new("RGB", (1024, 1024)).save(src / "scene.jpg")
    records = [
        {"file_name": "basketball_court_1.jpg", "height": 256, "width": 256, "source": "A",
         "boxes": np.array([[10.0, 20.0, 110.0, 220.0]])},
        {"file_name": "scene.jpg", "height": 1024, "width": 1024, "source": "B",
         "boxes": np.array([[600.0, 600.0, 700.0, 680.0]])},
    ]
    counts = export_split(records, src, tmp_path / "out", views=3, seed=0)
    assert counts == {"images": 6, "boxes": 6, "skipped": 0}
    # source A: same box in every view (whole tile, top-left of the canvas, native scale)
    for k in range(3):
        np.testing.assert_allclose(_round_trip(tmp_path / "out" / "labels" / f"basketball_court_1_v{k}.txt"),
                                   [[10, 20, 110, 220]], atol=1e-3)
        with Image.open(tmp_path / "out" / "images" / f"scene_v{k}.jpg") as im:
            assert im.size == (512, 512)
        b = _round_trip(tmp_path / "out" / "labels" / f"scene_v{k}.txt")
        np.testing.assert_allclose(b[:, 2] - b[:, 0], [100], atol=1e-3)  # p_pos = 1: the court is in the crop, whole
        np.testing.assert_allclose(b[:, 3] - b[:, 1], [80], atol=1e-3)
