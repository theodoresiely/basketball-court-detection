"""Contact sheets for visual inspection (Pillow only: matplotlib is not a dependency).

Colour code, the same on every sheet:
    green  thin-medium   annotated courts (ground truth)
    red    thin + score  every detection of the model in the window
    yellow thick         the DETECTION the tile is about (e.g. a false alarm, a proposed split)
    cyan   thick         the ANNOTATED court the tile is about (e.g. a missed court, a box to relabel)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

GREEN, RED, YELLOW, CYAN = (40, 220, 60), (240, 40, 40), (255, 230, 0), (0, 230, 255)


def tile(
    image_dir: Path,
    record: dict,
    detections: tuple[np.ndarray, np.ndarray],
    focus: np.ndarray,
    caption: str,
    focus_color: tuple[int, int, int] = YELLOW,
    extra: Sequence[tuple[np.ndarray, tuple[int, int, int]]] = (),
    size: int = 256,
    context: int = 160,
) -> Image.Image:
    """Square window around `focus` (+ `context` px), resized to `size`, with a 2-line caption.

    detections: (boxes [D, 4], scores [D]) of the image, drawn in red with their score.
    extra: more (box, colour) pairs drawn thick, on top (e.g. the proposed split boxes).
    """
    cx, cy = (focus[0] + focus[2]) / 2, (focus[1] + focus[3]) / 2
    half = max(focus[2] - focus[0], focus[3] - focus[1]) / 2 + context
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(np.ceil(min(record["width"], cx + half))), int(np.ceil(min(record["height"], cy + half)))
    with Image.open(Path(image_dir) / record["file_name"]) as img:
        window = img.convert("RGB").crop((x0, y0, x1, y1))
    scale = size / max(window.size)
    window = window.resize((max(1, round(window.width * scale)), max(1, round(window.height * scale))))
    canvas = Image.new("RGB", (size, size + 28), (30, 30, 30))
    canvas.paste(window, (0, 0))
    draw = ImageDraw.Draw(canvas)

    def rect(box, color, width):
        draw.rectangle([(box[0] - x0) * scale, (box[1] - y0) * scale, (box[2] - x0) * scale, (box[3] - y0) * scale],
                       outline=color, width=width)

    for g in record["boxes"]:
        rect(g, GREEN, 2)
    for box, score in zip(*detections):
        rect(box, RED, 1)
        draw.text(((box[0] - x0) * scale + 2, (box[1] - y0) * scale + 1), f"{score:.2f}", fill=RED)
    for box, color in extra:
        rect(box, color, 3)
    rect(focus, focus_color, 3)
    draw.text((3, size + 2), caption[:42], fill=(230, 230, 230))
    draw.text((3, size + 14), record["file_name"][:42], fill=(160, 160, 160))
    return canvas


def sheet(tiles: Sequence[Image.Image], columns: int = 4) -> Image.Image:
    """Tiles side by side, `columns` per row, 4 px black gaps."""
    if not tiles:
        return Image.new("RGB", (64, 64), (30, 30, 30))
    w, h = tiles[0].size
    rows = -(-len(tiles) // columns)
    out = Image.new("RGB", (columns * w + (columns - 1) * 4, rows * h + (rows - 1) * 4), (0, 0, 0))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % columns) * (w + 4), (i // columns) * (h + 4)))
    return out
