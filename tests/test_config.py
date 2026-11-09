from pathlib import Path

import pytest

from courtdet.config import Config, from_dict, load_config, save_config, to_dict

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_equals_the_dataclass_defaults():
    assert to_dict(load_config(ROOT / "configs" / "default.yaml")) == to_dict(Config())


def test_missing_keys_take_the_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 0.0003\nmodel:\n  roi:\n    reg_weight: 10\n")
    config = load_config(path)
    assert config.train.lr == 3e-4 and config.model.roi.reg_weight == 10
    assert config.train.epochs == Config().train.epochs
    assert config.model.rpn == Config().model.rpn


@pytest.mark.parametrize("text", ["trian:\n  lr: 0.1\n", "model:\n  roi:\n    reg_wieght: 10\n"])
def test_unknown_keys_raise(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    with pytest.raises(KeyError, match="unknown key"):
        load_config(path)


def test_section_must_be_a_mapping(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train: 3\n")
    with pytest.raises(TypeError):
        load_config(path)


def test_yaml_lists_become_tuples(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  anchors:\n    sizes: [32, 64]\n")
    assert load_config(path).model.anchors.sizes == (32, 64)


def test_round_trip_and_overrides(tmp_path):
    config = load_config(None, {"train.lr": 1e-4, "data.batch_size": 4, "model.anchors.aspect_ratios": [1.0]})
    assert config.train.lr == 1e-4 and config.data.batch_size == 4 and config.model.anchors.aspect_ratios == (1.0,)
    save_config(config, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == config


def test_scientific_notation_without_dot_is_a_float(tmp_path):
    # PyYAML (YAML 1.1) reads `1e-4` as the string "1e-4": this crashed the first lr test (29/10).
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 1e-4\n")
    assert load_config(path).train.lr == 1e-4
    assert load_config(None, {"train.lr": "3e-4"}).train.lr == 3e-4


def test_ints_are_accepted_for_floats():
    config = load_config(None, {"train.lr": 1, "model.anchors.sizes": [32, 64]})
    assert isinstance(config.train.lr, float)
    assert all(isinstance(s, float) for s in config.model.anchors.sizes)


@pytest.mark.parametrize(
    "key, value",
    [("train.lr", "fast"), ("train.epochs", 2.5), ("train.epochs", True), ("train.epochs", "20"),
     ("model.roi.aligned", 1), ("train.optimizer", 3), ("model.anchors.sizes", 32)],
)
def test_wrong_types_raise(key, value):
    with pytest.raises(TypeError):
        load_config(None, {key: value})


def test_optional_float_accepts_none_and_numbers():
    assert load_config(None, {"train.grad_clip_norm": None}).train.grad_clip_norm is None
    assert load_config(None, {"train.grad_clip_norm": "1e1"}).train.grad_clip_norm == 10.0


def test_retired_keys_load_only_when_off():
    # configs saved in older checkpoints still hold the keys of hard negative mining (removed from the code)
    off = {"hard_negatives": None, "hard_negatives_review": None, "hard_min_score": 0.5, "p_hard": 0.0}
    config = from_dict(Config, {"data": {**off, "tol": 0.85}})
    assert config.data.tol == 0.85 and not hasattr(config.data, "p_hard")
    with pytest.raises(ValueError, match="removed"):
        from_dict(Config, {"data": {**off, "p_hard": 0.5}})
    with pytest.raises(ValueError, match="removed"):
        from_dict(Config, {"data": {"hard_negatives": "mined.json"}})
