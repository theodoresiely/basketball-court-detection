import math

import torch

from courtdet.models.anchors import generate_base_anchors, grid_anchors

SIZES = (32.0, 64.0, 128.0)
RATIOS = (0.5, 1.0, 2.0)


def widths_heights(boxes):
    return boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]


def test_base_anchors_count_area_and_ratio():
    base = generate_base_anchors(SIZES, RATIOS)
    assert base.shape == (9, 4)

    w, h = widths_heights(base)
    # order: sizes outer loop, ratios inner loop
    expected_area = torch.tensor([s * s for s in SIZES for _ in RATIOS])
    expected_ratio = torch.tensor([r for _ in SIZES for r in RATIOS])
    assert torch.allclose(w * h, expected_area)
    assert torch.allclose(w / h, expected_ratio)


def test_base_anchors_are_centred_on_origin():
    base = generate_base_anchors(SIZES, RATIOS)
    cx = (base[:, 0] + base[:, 2]) / 2
    cy = (base[:, 1] + base[:, 3]) / 2
    assert torch.allclose(cx, torch.zeros(9))
    assert torch.allclose(cy, torch.zeros(9))


def test_grid_anchor_count_for_project_config():
    # 512x512 image, stride 8 -> 64x64 feature map, 9 anchors per cell
    base = generate_base_anchors(SIZES, RATIOS)
    anchors = grid_anchors((64, 64), stride=8, base_anchors=base)
    assert anchors.shape == (64 * 64 * 9, 4)


def test_grid_anchor_centres_are_cell_centres():
    stride = 8
    base = generate_base_anchors((32.0,), (1.0,))  # k = 1 to read the grid easily
    anchors = grid_anchors((2, 3), stride=stride, base_anchors=base)
    centres = torch.stack(
        [(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2], dim=1
    )
    expected = torch.tensor(
        [[4.0, 4.0], [12.0, 4.0], [20.0, 4.0],      # row 0: x moves first
         [4.0, 12.0], [12.0, 12.0], [20.0, 12.0]]   # row 1
    )
    assert torch.allclose(centres, expected)


def test_grid_order_matches_rpn_head_flattening():
    """The contract between anchors and RPN outputs.

    We build, with explicit loops, a fake RPN regression map of shape [1, k*4, H, W]
    whose channels a*4:(a+1)*4 at cell (i, j) contain "anchor a of cell (i, j)".
    Flattening it exactly like the RPN head does must give grid_anchors' output.
    H != W on purpose, so that swapping rows and columns cannot go unnoticed.
    """
    H, W, stride = 3, 5, 8
    base = generate_base_anchors(SIZES, RATIOS)
    k = base.shape[0]

    fake_head_output = torch.zeros(1, k * 4, H, W)
    for i in range(H):
        for j in range(W):
            cx, cy = (j + 0.5) * stride, (i + 0.5) * stride
            for a in range(k):
                fake_head_output[0, a * 4:(a + 1) * 4, i, j] = base[a] + torch.tensor([cx, cy, cx, cy])

    flattened = fake_head_output.permute(0, 2, 3, 1).reshape(1, H * W * k, 4)[0]
    assert torch.allclose(flattened, grid_anchors((H, W), stride, base))
