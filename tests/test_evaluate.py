import contextlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from courtdet.config import load_config
from courtdet.data.dataset import load_coco_records
from courtdet.engine.evaluate import (
    ImageResult,
    average_precision,
    bootstrap,
    match_image,
    match_split,
    paired_bootstrap,
    predict_image,
    predict_split,
    report,
    subsets,
    summarize,
)
from courtdet.engine.train import fit
from courtdet.models.faster_rcnn import FasterRCNN, FasterRCNNConfig
from test_dataset import MINI, write_coco  # pytest puts tests/ on sys.path

_spec = importlib.util.spec_from_file_location("eval_script", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py")
eval_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_script)


def ap_of(images):
    """images: list of (det_boxes, det_scores, gt_boxes) -> summarize(AP)."""
    return summarize(average_precision([match_image(*im) for im in images]))


# ---------------------------------------------------------------- pycocotools reference


def coco_reference(images):
    """(AP@[.5:.95], AP@0.5) computed by pycocotools on the same images."""
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "court"}]}
    dets = []
    for image_id, (boxes, scores, gts) in enumerate(images, start=1):
        gt["images"].append({"id": image_id, "width": 4096, "height": 4096})
        for x1, y1, x2, y2 in np.asarray(gts).reshape(-1, 4):
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": image_id, "category_id": 1,
                                      "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        for (x1, y1, x2, y2), s in zip(np.asarray(boxes).reshape(-1, 4), scores):
            dets.append({"image_id": image_id, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        evaluator = COCOeval(coco, coco.loadRes(dets), "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator.stats[0], evaluator.stats[1]


def random_images(seed, n_images=25):
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n_images):
        n_gt = rng.integers(0, 5)
        xy = rng.uniform(0, 900, (n_gt, 2))
        gts = np.concatenate([xy, xy + rng.uniform(20, 150, (n_gt, 2))], axis=1)
        jittered = gts + rng.normal(0, 12, gts.shape)  # IoUs spread over [0, 1]
        jittered[:, 2:] = np.maximum(jittered[:, 2:], jittered[:, :2] + 1)
        has_dupe = rng.random(n_gt) < 0.3  # second detection on the same court
        dupes = gts[has_dupe] + rng.normal(0, 4, (int(has_dupe.sum()), 4))
        noise_xy = rng.uniform(0, 900, (rng.integers(0, 6), 2))
        noise = np.concatenate([noise_xy, noise_xy + rng.uniform(20, 150, noise_xy.shape)], axis=1)
        boxes = np.concatenate([jittered, gts[: max(0, n_gt - 2)] + 3, noise, dupes])
        if i == 0:  # more than max_dets detections in one image
            boxes = np.concatenate([boxes, np.tile([[1000.0, 1000, 1050, 1050]], (120, 1))])
        images.append((boxes, rng.random(len(boxes)), gts))
    return images


@pytest.mark.parametrize("seed", range(5))
def test_matches_pycocotools_on_random_sets(seed):
    images = random_images(seed)
    ours = ap_of(images)
    ap, ap50 = coco_reference(images)
    assert ours["ap"] == pytest.approx(ap, abs=1e-9)
    assert ours["ap50"] == pytest.approx(ap50, abs=1e-9)


# ---------------------------------------------------------------- hand-computed cases

G1, G2, G3, G4 = [0, 0, 100, 100], [200, 200, 260, 260], [0, 0, 80, 40], [300, 300, 340, 380]
# Scores 0.9 .. 0.5 -> TP (G1), TP (G3), FP (background), TP (G2), FP (duplicate of G1).
TOY = [
    (np.array([[0, 0, 100, 100], [200, 200, 260, 260], [2, 0, 100, 100]], float), np.array([0.9, 0.6, 0.5]), np.array([G1, G2], float)),
    (np.array([[0, 0, 80, 40], [500, 500, 550, 550]], float), np.array([0.8, 0.7]), np.array([G3, G4], float)),
]


def test_hand_computed_ap():
    # precision 1, 1, 2/3, 3/4, 3/5 ; recall 1/4, 1/2, 1/2, 3/4, 3/4 ; envelope 1, 1, 3/4, 3/4, 3/4.
    # 101 recall levels: 0..0.25 (26 levels) -> 1 ; 0.26..0.50 (25) -> 1 ; 0.51..0.75 (25) -> 0.75 ;
    # 0.76..1 (25) never reached -> 0.  AP@0.5 = (26 + 25 + 25 * 0.75) / 101.
    assert ap_of(TOY)["ap50"] == pytest.approx((26 + 25 + 25 * 0.75) / 101, abs=1e-12)
    assert ap_of(TOY)["ap50"] == pytest.approx(coco_reference(TOY)[1], abs=1e-12)


def test_duplicate_is_a_false_positive():
    result = match_image(TOY[0][0], TOY[0][1], TOY[0][2])
    assert result.tp[0].tolist() == [True, True, False]  # sorted by score: G1, G2, duplicate of G1


def test_greedy_matching_takes_the_best_unmatched_ground_truth():
    gts = np.array([[0, 0, 100, 100], [20, 0, 120, 100]], float)
    det = np.array([[15, 0, 115, 100], [-30, 0, 70, 100]], float)
    # det 0: IoU 85/115 = 0.74 with gt 0, 95/105 = 0.90 with gt 1 -> both above 0.5, best = gt 1
    # det 1: IoU 70/130 = 0.54 with gt 0, 50/150 = 0.33 with gt 1 -> can only match gt 0
    result = match_image(det, np.array([0.9, 0.8]), gts, iou_thresholds=[0.5])
    assert result.tp.tolist() == [[True, True]]
    # with "first ground truth above the threshold", det 0 would take gt 0 and det 1 would be a FP
    assert ap_of([(det, np.array([0.9, 0.8]), gts)])["ap50"] == pytest.approx(coco_reference([(det, [0.9, 0.8], gts)])[1])


def test_perfect_empty_and_undefined():
    perfect = [(np.array([G1, G2], float), np.array([0.9, 0.8]), np.array([G1, G2], float))]
    assert ap_of(perfect) == {"ap50": pytest.approx(1.0), "ap": pytest.approx(1.0)}
    nothing = [(np.zeros((0, 4)), np.zeros(0), np.array([G1], float))]
    assert ap_of(nothing) == {"ap50": 0.0, "ap": 0.0}
    no_gt = [(np.array([G1], float), np.array([0.9]), np.zeros((0, 4)))]
    assert np.isnan(ap_of(no_gt)["ap50"])


def test_localisation_precision_lowers_ap_5095_not_ap50():
    loose = [(np.array([[0, 0, 100, 120]], float), np.array([0.9]), np.array([G1], float))]  # IoU 100/120 = 0.83
    assert ap_of(loose)["ap50"] == pytest.approx(1.0)
    assert ap_of(loose)["ap"] == pytest.approx(0.7)  # TP at t = 0.50 .. 0.80 (7 of 10 thresholds)


# ---------------------------------------------------------------- bootstrap


def test_resampled_image_counts_twice_like_two_identical_images():
    # The bootstrap reuses per-image matches: drawing an image twice must equal evaluating it twice.
    r = [match_image(*im) for im in TOY]
    ours = summarize(average_precision([r[0], r[0], r[1]]))
    ap, ap50 = coco_reference([TOY[0], TOY[0], TOY[1]])
    assert ours["ap50"] == pytest.approx(ap50, abs=1e-12) and ours["ap"] == pytest.approx(ap, abs=1e-12)


def test_bootstrap_interval_is_reproducible_and_brackets_the_estimate():
    results = [match_image(*im) for im in random_images(0, n_images=60)]
    a = bootstrap(results, n_boot=200, seed=3)
    assert a == bootstrap(results, n_boot=200, seed=3)
    assert a["ap50_lo"] <= a["ap50"] <= a["ap50_hi"] and a["ap_lo"] <= a["ap"] <= a["ap_hi"]
    assert a["ap50_hi"] - a["ap50_lo"] > 0


def test_bootstrap_of_a_single_image_has_no_spread():
    a = bootstrap([match_image(*TOY[0])], n_boot=50)
    assert a["ap50_lo"] == a["ap50"] == a["ap50_hi"]


def test_paired_bootstrap():
    images = random_images(1, n_images=60)
    results = [match_image(*im) for im in images]
    same = paired_bootstrap(results, results, n_boot=100)
    assert same["ap50"] == same["ap50_lo"] == same["ap50_hi"] == 0.0

    perfect = [match_image(gts, np.ones(len(gts)), gts) for _, _, gts in images]
    better = paired_bootstrap(perfect, results, n_boot=200)
    assert better["ap50_lo"] > 0  # clearly better on the same images

    with pytest.raises(ValueError):
        paired_bootstrap(results, results[:-1])


# ---------------------------------------------------------------- subsets and report


def test_subsets_and_report():
    records = [
        {"source": "A", "boxes": np.array([G1], float)},
        {"source": "B", "boxes": np.zeros((0, 4))},
        {"source": "B", "boxes": np.array([G1, G2], float)},
        {"source": "B", "boxes": np.array([G1, G2, G3, G4, [600, 600, 700, 700]], float)},
    ]
    s = subsets(records)
    assert s == {"all": [0, 1, 2, 3], "A": [0], "B": [1, 2, 3], "B_1-3": [2], "B_4+": [3]}
    results = [match_image(r["boxes"], np.ones(len(r["boxes"])), r["boxes"]) for r in records]
    rep = report(results, records, n_boot=20)
    assert rep["B"]["n_images"] == 3 and rep["B"]["n_gt"] == 7
    assert rep["all"]["ap50"] == pytest.approx(1.0)


# ---------------------------------------------------------------- inference


class StubModel:
    """Records what predict_image feeds to the model; returns fixed boxes in padded coordinates."""

    def __init__(self, boxes):
        self.boxes = torch.tensor(boxes, dtype=torch.float32)
        self.calls = []

    def eval(self):
        return self

    def detect(self, x, proposal_scale=1.0):
        self.calls.append((x.clone(), proposal_scale))
        return [{"boxes": self.boxes, "scores": torch.linspace(0.9, 0.5, len(self.boxes))}]


def test_small_image_is_padded_to_512_and_detections_clipped_to_the_real_image():
    image = torch.rand(3, 256, 256)
    stub = StubModel([[10, 10, 60, 60], [200, 200, 300, 300], [300, 300, 400, 400]])
    boxes, scores = predict_image(stub, image, torch.device("cpu"))
    x, scale = stub.calls[0]
    assert x.shape == (1, 3, 512, 512) and scale == 1.0
    assert (x[0, :, 256:, :] == 0).all() and (x[0, :, :, 256:] == 0).all()  # zero padding = mean colour
    # box 2 clipped to the 256 image; box 3 lies entirely in the padding -> dropped
    assert np.allclose(boxes, [[10, 10, 60, 60], [200, 200, 256, 256]]) and len(scores) == 2


def test_big_image_gets_proportionally_more_proposals():
    stub = StubModel([[0, 0, 10, 10]])
    predict_image(stub, torch.rand(3, 1000, 1203), torch.device("cpu"))
    x, scale = stub.calls[0]
    assert x.shape == (1, 3, 1000, 1208)  # padded to a multiple of the stride (8)
    assert scale == pytest.approx(1000 * 1208 / 512**2)


def test_end_to_end_with_a_real_model_and_the_cli(tmp_path):
    root = tmp_path / "data"
    write_coco(root, "train", MINI)
    write_coco(root, "valid", MINI)
    records = load_coco_records(root / "valid" / "_annotations.coco.json")

    torch.manual_seed(0)
    model = FasterRCNN(FasterRCNNConfig(pretrained_backbone=False))
    detections = predict_split(model, root / "valid", records, torch.device("cpu"), log=lambda s: None)
    results = match_split(detections, records)
    assert len(results) == len(records)
    for d, r in zip(detections, records):
        assert (d["boxes"] >= 0).all() and (d["boxes"][:, [0, 2]] <= r["width"]).all() and (d["boxes"][:, [1, 3]] <= r["height"]).all()
    with pytest.raises(ValueError):
        match_split(detections[::-1], records)

    config = load_config(None, {"model.pretrained_backbone": False, "data.root": str(root), "data.crop_size": 256,
                                "data.batch_size": 2, "data.num_workers": 0, "train.epochs": 1,
                                "train.device": "cpu", "train.log_every": 0})
    fit(config, tmp_path / "run", max_steps=1, log=lambda s: None)
    eval_script.main(["--checkpoint", str(tmp_path / "run" / "best.pt"), "--device", "cpu", "--n-boot", "20"])
    out = json.loads((tmp_path / "run" / "eval_valid_best.json").read_text())
    assert set(out["metrics"]) == {"all", "A", "B", "B_1-3", "B_4+"}
    assert out["metrics"]["all"]["n_images"] == len(MINI)
    eval_script.main(["--checkpoint", str(tmp_path / "run" / "best.pt"), "--device", "cpu", "--n-boot", "20",
                      "--compare", str(tmp_path / "run" / "eval_valid_best.pt")])
    out = json.loads((tmp_path / "run" / "eval_valid_best.json").read_text())
    assert out["compare"]["this_minus_other"]["ap50"] == 0.0  # same model on both sides
