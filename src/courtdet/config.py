"""Run configuration: dataclasses <-> YAML.

A YAML file has three sections, one per dataclass:

    model: -> FasterRCNNConfig (models/faster_rcnn.py)
    data:  -> DataConfig
    train: -> TrainConfig

Missing keys take the dataclass defaults. Unknown keys RAISE: a typo such as
`reg_wieght: 10` must not silently run with the default lambda.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from courtdet.models.faster_rcnn import FasterRCNNConfig


@dataclass
class DataConfig:
    root: str = "dataset/basket_dataset.coco"
    train_split: str = "train"
    val_split: str = "valid"
    # Crop rule: defaults of CourtDataset.
    crop_size: int = 512
    tol: float = 0.85
    p_pos: float = 1.0
    max_tries: int = 20
    batch_size: int = 8  # notebook value
    num_workers: int = 2  # not tuned: to be chosen from a timing measurement
    # Human review of source A, None = Roboflow boxes as exported.
    corrections: str | None = None  # step 2 file: corrected boxes / excluded images
    decisions: str | None = None  # step 1 file: drops "multiple"/"unsure" A images left uncorrected
    source_a_fraction: float = 1.0  # train only: fraction of source A originals kept (learning curve)
    subset_seed: int = 0  # seed of that random fraction
    # Online augmentations of the training crops (data/augment.py), train only. Off = no augmentation.
    aug_dihedral: bool = False  # random rotation by k x 90 degrees + optional flip
    aug_brightness: float = 0.0  # factors drawn in [1 - d, 1 + d]
    aug_contrast: float = 0.0
    aug_saturation: float = 0.0
    aug_gray: float = 0.0  # probability of a grey-level crop


@dataclass
class TrainConfig:
    epochs: int = 20  # notebook value
    optimizer: str = "adam"  # "adam" or "sgd"
    lr: float = 1e-3  # notebook value (Adam); to be checked by a short lr comparison
    momentum: float = 0.9  # sgd only
    weight_decay: float = 0.0
    # ReduceLROnPlateau on the validation total loss: lr *= factor after `patience`
    # epochs without improvement. Notebook values.
    plateau_factor: float = 0.1
    plateau_patience: int = 3
    grad_clip_norm: float | None = None  # None = no clipping (the norm is logged anyway)
    seed: int = 0
    val_seed: int = 1234  # fixed seed of every validation pass: same crops and samplings each epoch
    device: str = "auto"  # "auto" = cuda, else mps, else cpu
    log_every: int = 20  # iterations between two progress lines (0 = silent)
    # Checkpoint selection: the validation loss and the detection metric disagree near the
    # optimum, so best.pt is chosen on the validation mAP (whole images, as scripts/evaluate.py).
    eval_every: int = 1  # epochs between two validation mAP passes (0 = never; ~35 s on MPS for valid)
    select_metric: str = "ap"  # best.pt criterion: "ap" (AP@[.5:.95]), "ap50" or "loss" (val total loss)


@dataclass
class Config:
    model: FasterRCNNConfig = field(default_factory=FasterRCNNConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Keys of features removed from the code, still present in the config saved inside older checkpoints.
# {dataclass name: {key: value meaning "feature off", or ANY if the key only matters when the feature is on}}.
# Such a key is dropped on load when it is off, so old checkpoints still load; a config that turns the
# feature ON raises, since the code to run it is gone (git history: hard negative mining, not kept).
ANY = object()
RETIRED_KEYS = {
    "DataConfig": {"hard_negatives": None, "hard_negatives_review": None, "hard_min_score": ANY, "p_hard": 0.0},
}


def _drop_retired(cls, values: dict[str, Any]) -> dict[str, Any]:
    retired = RETIRED_KEYS.get(cls.__name__, {})
    for key in set(values) & set(retired):
        off = retired[key]
        if off is not ANY and values[key] != off:
            raise ValueError(f"{cls.__name__}.{key} = {values[key]!r}: this feature was removed from the code "
                             f"(only {off!r} is accepted, to read older checkpoints)")
    return {k: v for k, v in values.items() if k not in retired}


def from_dict(cls, values: dict[str, Any] | None):
    """Build dataclass `cls` from a (possibly partial) nested dict. Unknown keys raise."""
    values = values or {}
    if not isinstance(values, dict):
        raise TypeError(f"{cls.__name__}: expected a mapping, got {values!r}")
    values = _drop_retired(cls, values)
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}; valid keys: {sorted(names)}")

    kwargs = {}
    for name, value in values.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = from_dict(hint, value)
        else:
            kwargs[name] = _check_type(hint, value, f"{cls.__name__}.{name}")
    return cls(**kwargs)


def _check_type(hint, value, where: str):
    """Return `value` converted to `hint`, or raise TypeError.

    Conversions allowed: int -> float, list -> tuple (YAML has no tuples), and str -> float
    because PyYAML follows YAML 1.1, where `1e-4` (no dot) is read as the STRING "1e-4"
    (only `1.0e-4` is a float). Anything else that does not match raises.
    """
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):  # e.g. float | None
        args = typing.get_args(hint)
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _check_type(inner, value, where)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        inner = typing.get_args(hint)[0]  # tuple[X, ...] or tuple[X, X, X, X]
        return tuple(_check_type(inner, v, where) for v in value)
    if hint is float:
        if isinstance(value, float):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise TypeError(f"{where}: expected a number, got {value!r}")
    if hint in (int, bool, str):
        # bool is a subclass of int: `epochs: true` must not pass as 1
        if type(value) is not hint:
            raise TypeError(f"{where}: expected {hint.__name__}, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported type {hint}")


def to_dict(config) -> dict[str, Any]:
    """Nested plain dict (tuples -> lists), safe for YAML and for torch.save(weights_only)."""

    def plain(x):
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        return x

    return plain(dataclasses.asdict(config))


def set_value(values: dict[str, Any], dotted_key: str, value: Any) -> None:
    """values["train"]["lr"] = value for dotted_key "train.lr" (used by --set on the CLI)."""
    *parents, last = dotted_key.split(".")
    for key in parents:
        values = values.setdefault(key, {})
    values[last] = value


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Config from a YAML file (None = all defaults), then dotted-key overrides."""
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    for key, value in (overrides or {}).items():
        set_value(values, key, value)
    return from_dict(Config, values)


def save_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(to_dict(config), sort_keys=False))
