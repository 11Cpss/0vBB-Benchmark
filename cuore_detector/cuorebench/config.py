"""Configuration and target scaling for the CUORE Δt regression task."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


# ``evaluate_regression`` -> ``metrics.make_fixed_energy_bins`` rejects any
# finite target outside [0, 3000] keV, i.e. [0, 3] with energy_unit="MeV".
# The scaled Δt target must stay inside this band.
CANONICAL_SCALED_MAX: float = 3.0


@dataclass(frozen=True)
class CuoreDataConfig:
    """Configuration for loading and splitting the CUORE regression data."""

    data_dir: str
    train_file: str = "cuoreTraining.h5"
    test_file: str = "cuoreTest.h5"
    waveform_length: int = 10_000

    val_fraction: float = 0.15
    split_seed: int = 42

    # Target scaling: y = Δt_ms / dt_scale. dt_scale=3000 keeps the observed
    # Δt (<= ~6813 ms) inside [0, 3) with headroom.
    dt_scale: float = 3000.0
    missing_dt_fill: float = 0.0

    max_events: int | None = None
    norm_eps: float = 1e-8
    clip_normalized: bool = True

    # sample_weight = base_weight
    #   * (tailelevated_weight if tailElevated else 1)   [train/val only]
    #   * (zero_dt_weight if single-pulse else 1)        [train/val only]
    # All 1.0 by default -> unweighted run. tailelevated_weight == 0.0 drops
    # those rows from train/val (never from test).
    base_weight: float = 1.0
    tailelevated_weight: float = 1.0
    zero_dt_weight: float = 1.0
    weight_normalization: str = "none"  # "none" | "mean_one"

    def __post_init__(self) -> None:
        if (
            isinstance(self.waveform_length, bool)
            or not isinstance(self.waveform_length, int)
            or self.waveform_length <= 0
        ):
            raise ValueError("waveform_length must be a positive integer")

        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")

        if (
            isinstance(self.split_seed, bool)
            or not isinstance(self.split_seed, int)
            or self.split_seed < 0
        ):
            raise ValueError("split_seed must be a non-negative integer")

        if not math.isfinite(self.dt_scale) or self.dt_scale <= 0.0:
            raise ValueError("dt_scale must be finite and positive")

        if not math.isfinite(self.missing_dt_fill) or self.missing_dt_fill < 0.0:
            raise ValueError("missing_dt_fill must be finite and non-negative")

        if self.max_events is not None and (
            isinstance(self.max_events, bool)
            or not isinstance(self.max_events, int)
            or self.max_events <= 0
        ):
            raise ValueError("max_events must be a positive integer or None")

        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be finite and positive")

        for name in ("base_weight", "tailelevated_weight", "zero_dt_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        if self.weight_normalization not in {"none", "mean_one"}:
            raise ValueError("weight_normalization must be 'none' or 'mean_one'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scale_target(dt_ms: np.ndarray, config: CuoreDataConfig) -> np.ndarray:
    """Δt in milliseconds -> scaled regression target in ``[0, 3)``."""

    return np.asarray(dt_ms, dtype=np.float64) / float(config.dt_scale)


def unscale_target(scaled: np.ndarray, config: CuoreDataConfig) -> np.ndarray:
    """Scaled regression target -> Δt in milliseconds."""

    return np.asarray(scaled, dtype=np.float64) * float(config.dt_scale)


__all__ = [
    "CANONICAL_SCALED_MAX",
    "CuoreDataConfig",
    "scale_target",
    "unscale_target",
]
