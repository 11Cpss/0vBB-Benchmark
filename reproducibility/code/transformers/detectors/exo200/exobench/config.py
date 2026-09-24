"""Single-source configuration for EXO-200 binary classification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


DEFAULT_DATA_ROOT = Path("data/EXO-200")
DATASET_NAME = "EXO-200"
TASK = "classification"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / TASK

INPUT_FIELD = "Waveforms"
LABEL_FIELD = "Charge_cluster_number"
EVENT_FIELD = "event_number"
GROUP_ATTRIBUTE = "run_number"
POSITION_FIELD = "Charge_Clusters_Pos"
INPUT_SHAPE = (226, 300)
INPUT_DTYPE = "int16"

NUM_CLASSES = 2
CLASS_NAMES = ["signal", "background"]
SIGNAL_LABEL = 0
BACKGROUND_LABEL = 1
SIGNAL_NCCL = 1

# MJD has 16 official train shards and 6 official test shards of equal size.
# EXO-200 has no official split, so this reproduces MJD's held-out share while
# keeping EXO runs indivisible. Validation remains 10% of the non-test pool.
MJD_TEST_FRACTION = 6.0 / 22.0


@dataclass(frozen=True)
class DataConfig:
    """EXO-200 schema, preprocessing, and grouped-split settings."""

    data_root: Path = DEFAULT_DATA_ROOT
    validation_fraction: float = 0.10
    test_fraction: float = MJD_TEST_FRACTION
    baseline_samples: int = 200
    classification_amplitude_normalization: bool = True
    seed: int = 42
    input_field: str = INPUT_FIELD
    label_field: str = LABEL_FIELD
    event_field: str = EVENT_FIELD
    group_attribute: str = GROUP_ATTRIBUTE
    position_field: str = POSITION_FIELD
    input_shape: tuple[int, int] = INPUT_SHAPE
    input_dtype: str = INPUT_DTYPE
    num_classes: int = NUM_CLASSES
    class_names: tuple[str, str] = tuple(CLASS_NAMES)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "class_names", tuple(self.class_names))
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if not 0.0 < float(self.test_fraction) < 1.0:
            raise ValueError("test_fraction must be between zero and one")
        if int(self.baseline_samples) <= 0:
            raise ValueError("baseline_samples must be positive")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if len(self.input_shape) != 2 or any(int(value) <= 0 for value in self.input_shape):
            raise ValueError("input_shape must contain positive channel and sample counts")
        if int(self.num_classes) != NUM_CLASSES:
            raise ValueError("EXO-200 classification requires num_classes=2")
        if list(self.class_names) != CLASS_NAMES:
            raise ValueError(
                'EXO-200 class_names must be ["signal", "background"] in that order'
            )
        fixed_fields = {
            "input_field": INPUT_FIELD,
            "label_field": LABEL_FIELD,
            "event_field": EVENT_FIELD,
            "group_attribute": GROUP_ATTRIBUTE,
            "position_field": POSITION_FIELD,
            "input_dtype": INPUT_DTYPE,
        }
        for name, expected in fixed_fields.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must remain {expected!r} for EXO-200")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["input_shape"] = list(self.input_shape)
        payload["class_names"] = list(self.class_names)
        return payload


@dataclass(frozen=True)
class ModelConfig:
    """The MJD CNN-001 parameters with the EXO input channel count."""

    input_channels: int = INPUT_SHAPE[0]
    base_channels: int = 16
    num_classes: int = NUM_CLASSES

    def __post_init__(self) -> None:
        if int(self.input_channels) <= 0:
            raise ValueError("input_channels must be positive")
        if int(self.base_channels) <= 0:
            raise ValueError("base_channels must be positive")
        if int(self.num_classes) != NUM_CLASSES:
            raise ValueError("the binary classifier requires num_classes=2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """MJD CNN-001 training defaults, unchanged for EXO-200."""

    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    early_stopping_min_delta: float = 0.0
    deterministic: bool = False
    use_amp: bool = False
    amp_precision: Literal["auto", "float16", "bfloat16"] = "auto"

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "early_stopping_patience"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        for name in ("learning_rate", "gradient_clip_norm"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if (
            not math.isfinite(float(self.early_stopping_min_delta))
            or float(self.early_stopping_min_delta) < 0.0
        ):
            raise ValueError("early_stopping_min_delta must be finite and non-negative")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if self.amp_precision not in {"auto", "float16", "bfloat16"}:
            raise ValueError(
                "amp_precision must be 'auto', 'float16', or 'bfloat16'"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "BACKGROUND_LABEL",
    "CLASS_NAMES",
    "DATASET_NAME",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DataConfig",
    "ModelConfig",
    "NUM_CLASSES",
    "PROJECT_ROOT",
    "SIGNAL_LABEL",
    "SIGNAL_NCCL",
    "TASK",
    "TrainingConfig",
]
