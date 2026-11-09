"""Export of our training data to the YOLO format, for the Ultralytics baseline.

Goal: YOLO and our Faster R-CNN must see the SAME pixels and the SAME boxes. Ultralytics resizes
every image so that its long side equals `imgsz`: a 2048 px source B scene at imgsz=512 would be
shrunk 4x (a 76 px court -> 19 px) and a 256 px source A tile enlarged 2x. So we export crops at
native resolution, drawn with our own sampler (`crops.sample_crop`, same `tol` / `p_pos`):
    - source A tile (256 x 256): placed top-left on a 512 x 512 canvas, as our Dataset pads it;
    - source B scene: `views` independent random 512 crops (positive when the image has courts).
Each image gets `views` exported crops (A tiles are simply repeated): the A / B mix stays the one
of our training, where every image gives one crop per epoch.

YOLO label file, one line per box: "0 cx cy w h", normalised by the canvas size (class 0 = court).
The padding colour is YOLO's own letterbox grey (114); our Dataset pads with the ImageNet mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from courtdet.data.crops import sample_crop

PAD_VALUE = 114


def yolo_lines(boxes, size: int) -> list[str]:
    """Boxes (x1, y1, x2, y2) in canvas pixels -> YOLO lines "0 cx cy w h", normalised by `size`."""
    out = []
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        cx, cy, w, h = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size, (x2 - x1) / size, (y2 - y1) / size
        out.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def crop_canvas(image: Image.Image, origin: tuple[int, int], size: int) -> Image.Image:
    """The size x size window at `origin`; the part outside the image is filled with PAD_VALUE."""
    x0, y0 = origin
    canvas = Image.new("RGB", (size, size), (PAD_VALUE,) * 3)
    canvas.paste(image.crop((x0, y0, min(x0 + size, image.width), min(y0 + size, image.height))), (0, 0))
    return canvas


def export_split(records: Sequence[dict], image_dir: str | Path, out_dir: str | Path, *, views: int,
                 crop_size: int = 512, tol: float = 0.85, p_pos: float = 1.0, max_tries: int = 20,
                 seed: int = 0) -> dict[str, int]:
    """Write out_dir/images/<stem>_v<k>.jpg and out_dir/labels/<stem>_v<k>.txt for every record.

    A view whose crop is rejected (`sample_crop` returns None) is drawn again, up to 10 times,
    then skipped. Returns counts: {"images", "boxes", "skipped"}.
    """
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts = {"images": 0, "boxes": 0, "skipped": 0}
    for r in records:
        with Image.open(image_dir / r["file_name"]) as img:
            image = img.convert("RGB")
        if (image.height, image.width) != (r["height"], r["width"]):
            raise ValueError(f"{r['file_name']}: size {image.size} differs from the annotation file")
        stem = Path(r["file_name"]).stem
        for k in range(views):
            for _ in range(10):
                crop = sample_crop(r["boxes"], (r["height"], r["width"]), crop_size=crop_size, tol=tol,
                                   p_pos=p_pos, max_tries=max_tries, rng=rng)
                if crop is not None:
                    break
            else:
                counts["skipped"] += 1
                continue
            name = f"{stem}_v{k}"
            crop_canvas(image, crop.origin, crop_size).save(out_dir / "images" / f"{name}.jpg", quality=95)
            # an empty label file = a background image for YOLO
            (out_dir / "labels" / f"{name}.txt").write_text("\n".join(yolo_lines(crop.boxes, crop_size)))
            counts["images"] += 1
            counts["boxes"] += len(crop.boxes)
    return counts
