"""Explicit configuration for the paired MJD tasks."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


DEFAULT_DATA_ROOT = Path(
    os.environ.get("MJD_BENCH_DATA", "data/MJD")
).expanduser()


@dataclass(frozen=True)
class DataConfig:
    """MJD file selection and waveform preprocessing settings."""

    data_root: Path = DEFAULT_DATA_ROOT
    validation_fraction: float = 0.10
    baseline_samples: int = 200
    classification_amplitude_normalization: bool = True
    regression_waveform_scale: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if int(self.baseline_samples) <= 0:
            raise ValueError("baseline_samples must be positive")
        if float(self.regression_waveform_scale) <= 0.0:
            raise ValueError("regression_waveform_scale must be positive")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    """Training defaults shared by classification and regression."""

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
