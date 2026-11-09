import csv
import math

import pytest
import torch

from courtdet.config import load_config
from courtdet.engine.train import (
    LOSS_KEYS,
    build_optimizer,
    build_scheduler,
    evaluate_loss,
    evaluate_map,
    fit,
    make_loaders,
)
from courtdet.models.faster_rcnn import FasterRCNN
from test_dataset import MINI, write_coco  # pytest puts tests/ on sys.path

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("train_script", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    write_coco(root, "train", MINI)
    write_coco(root, "valid", MINI[:3])
    return root


def small_config(data_root, **overrides):
    values = {
        "model.pretrained_backbone": False,
        "data.root": str(data_root),
        "data.crop_size": 256,
        "data.batch_size": 2,
        "data.num_workers": 0,
        "train.epochs": 2,
        "train.device": "cpu",
        "train.log_every": 0,
    }
    values.update(overrides)
    return load_config(None, values)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_fit_writes_a_complete_run(data_root, tmp_path):
    config = small_config(data_root)
    history = fit(config, tmp_path / "run", log=lambda s: None)
    assert len(history) == 2
    for row in history:
        for split in ("train", "val"):
            for k in LOSS_KEYS:
                assert math.isfinite(row[f"{split}_{k}"])
        assert row["train_grad_norm"] > 0
        assert 0.0 <= row["val_ap"] <= row["val_ap50"] <= 1.0  # evaluated every epoch by default
    run = tmp_path / "run"
    assert {p.name for p in run.iterdir()} == {"config.yaml", "history.csv", "last.pt", "best.pt", "best_loss.pt"}
    assert load_config(run / "config.yaml") == config
    assert len(read_csv(run / "history.csv")) == 2

    # best.pt = highest validation AP@[.5:.95] (first epoch on a tie), best_loss.pt = lowest validation loss
    assert torch.load(run / "best.pt")["epoch"] == max(history, key=lambda h: h["val_ap"])["epoch"]
    assert torch.load(run / "best_loss.pt")["epoch"] == min(history, key=lambda h: h["val_total"])["epoch"]


def test_selection_on_the_loss_without_map(data_root, tmp_path):
    config = small_config(data_root, **{"train.eval_every": 0, "train.select_metric": "loss"})
    history = fit(config, tmp_path / "run", log=lambda s: None)
    assert all(math.isnan(h["val_ap"]) and math.isnan(h["val_ap50"]) for h in history)
    best_epoch = min(history, key=lambda h: h["val_total"])["epoch"]
    assert torch.load(tmp_path / "run" / "best.pt")["epoch"] == best_epoch


def test_eval_every_and_invalid_selection(data_root, tmp_path):
    history = fit(small_config(data_root, **{"train.eval_every": 2}), tmp_path / "run", log=lambda s: None)
    assert math.isnan(history[0]["val_ap"]) and not math.isnan(history[1]["val_ap"])  # only epoch 2 evaluated
    assert torch.load(tmp_path / "run" / "best.pt")["epoch"] == 2
    for bad in ({"train.select_metric": "f1"}, {"train.eval_every": 0}):  # ap needs an evaluation
        with pytest.raises(ValueError):
            fit(small_config(data_root, **bad), tmp_path / f"bad{len(bad)}{list(bad)[0]}", log=lambda s: None)


class OracleModel:
    """Stands for FasterRCNN in evaluate_map: returns fixed boxes, moved right by `shift` pixels."""

    def __init__(self, boxes, shift=0.0):
        self.boxes = torch.as_tensor(boxes, dtype=torch.float32)
        self.shift = torch.tensor([shift, 0.0, shift, 0.0])

    def eval(self):
        return self

    def detect(self, images, proposal_scale=1.0):
        return [{"boxes": self.boxes + self.shift, "scores": torch.ones(len(self.boxes))}]


def test_evaluate_map_wiring(data_root):
    # One image, its own boxes as detections: AP 1 at every IoU threshold; shifted boxes: AP@0.5 only.
    config = small_config(data_root)
    _, val_loader = make_loaders(config)
    ds = val_loader.dataset
    ds.records = [r for r in ds.records if len(r["boxes"])][:1]
    gt = ds.records[0]["boxes"]
    exact = evaluate_map(OracleModel(gt), ds, torch.device("cpu"), 512)
    assert exact == pytest.approx({"ap50": 1.0, "ap": 1.0})
    # moved by 10 % of the narrowest width: IoU >= 0.9 / 1.1 = 0.82 for every box (< 1 at thresholds 0.85-0.95)
    shifted = evaluate_map(OracleModel(gt, shift=0.1 * float(min(gt[:, 2] - gt[:, 0]))), ds, torch.device("cpu"), 512)
    assert shifted["ap50"] == pytest.approx(1.0) and shifted["ap"] < 0.9


def test_validation_loss_is_deterministic_and_leaves_the_rng_alone(data_root):
    config = small_config(data_root)
    _, val_loader = make_loaders(config)
    model = FasterRCNN(config.model)
    state = torch.get_rng_state()
    a = evaluate_loss(model, val_loader, torch.device("cpu"), seed=7)
    assert torch.equal(torch.get_rng_state(), state)  # training randomness untouched
    torch.rand(1000)  # a training epoch consumes randomness between two validations
    b = evaluate_loss(model, val_loader, torch.device("cpu"), seed=7)
    assert a == b
    c = evaluate_loss(model, val_loader, torch.device("cpu"), seed=8)
    assert c != a  # the seed does matter: crops / samplings are random


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_validation_loss_is_deterministic_on_mps(data_root):
    config = small_config(data_root)
    _, val_loader = make_loaders(config)
    model = FasterRCNN(config.model).to("mps")
    device = torch.device("mps")
    a = evaluate_loss(model, val_loader, device, seed=7)
    torch.rand(1000), torch.rand(1000, device=device)  # randomness consumed by a training epoch
    assert evaluate_loss(model, val_loader, device, seed=7) == a


def test_resume_gives_the_same_run_as_no_interruption(data_root, tmp_path):
    full = fit(small_config(data_root), tmp_path / "full", log=lambda s: None)

    fit(small_config(data_root, **{"train.epochs": 1}), tmp_path / "cut", log=lambda s: None)
    resumed = fit(small_config(data_root), tmp_path / "cut", resume=True, log=lambda s: None)

    timings = ("time_s", "eval_time_s")  # wall-clock durations differ between two runs
    assert [{k: v for k, v in h.items() if k not in timings} for h in full] == \
           [{k: v for k, v in h.items() if k not in timings} for h in resumed]
    a = torch.load(tmp_path / "full" / "last.pt")["model"]
    b = torch.load(tmp_path / "cut" / "last.pt")["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_refuses_to_overwrite_a_run(data_root, tmp_path):
    config = small_config(data_root, **{"train.epochs": 1})
    fit(config, tmp_path / "run", max_steps=1, log=lambda s: None)
    with pytest.raises(FileExistsError):
        fit(config, tmp_path / "run", log=lambda s: None)


def test_scheduler_divides_lr_on_plateau():
    config = load_config(None, {"model.pretrained_backbone": False, "train.plateau_patience": 1})
    model = FasterRCNN(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_scheduler(optimizer, config.train)
    for loss in (1.0, 1.0, 1.0):  # epoch 1 sets the best, 2 = 1st bad epoch, 3 = 2nd > patience
        scheduler.step(loss)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_optimizer_holds_exactly_the_trainable_parameters():
    config = load_config(None, {"model.pretrained_backbone": False})
    model = FasterRCNN(config.model)
    optimizer = build_optimizer(model, config.train)
    in_optimizer = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert in_optimizer == {id(p) for p in model.parameters() if p.requires_grad}
    assert {id(p) for p in model.roi_head.parameters()} <= in_optimizer
    config.train.optimizer = "rmsprop"
    with pytest.raises(ValueError):
        build_optimizer(model, config.train)


def test_cli_smoke(data_root, tmp_path, capsys):
    out = tmp_path / "cli"
    args = ["--out", str(out), "--max-steps", "1",
            "--set", f"data.root={data_root}", "--set", "model.pretrained_backbone=false",
            "--set", "data.crop_size=256", "--set", "data.batch_size=2", "--set", "data.num_workers=0",
            "--set", "train.epochs=1", "--set", "train.device=cpu"]
    train_script.main(args)
    assert "epoch 1/1" in capsys.readouterr().out
    assert (out / "best.pt").exists()
    with pytest.raises(KeyError):
        train_script.main(["--out", str(tmp_path / "typo"), "--set", "train.lrr=1"])


def test_cli_parses_scientific_notation():
    assert train_script.parse_set(["train.lr=1e-4"]) == {"train.lr": "1e-4"}  # still a string here...
    assert load_config(None, train_script.parse_set(["train.lr=1e-4"])).train.lr == 1e-4  # ...float after checking


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_load_checkpoint_onto_mps_restores_the_cpu_rng(tmp_path):
    # Bug found on a real --resume: map_location="mps" also moved the saved CPU RNG state to MPS,
    # and torch.set_rng_state only accepts a CPU ByteTensor.
    from courtdet.engine.train import load_checkpoint
    torch.manual_seed(3)
    state = torch.get_rng_state()
    torch.save({"model": {}, "rng_cpu": state, "epoch": 2, "history": []}, tmp_path / "last.pt")
    torch.rand(5)  # move the RNG on
    epoch, _ = load_checkpoint(tmp_path / "last.pt", torch.nn.Module(), map_location=torch.device("mps"))
    assert epoch == 2 and torch.equal(torch.get_rng_state(), state)
