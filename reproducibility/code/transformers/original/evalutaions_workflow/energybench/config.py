"""Small, explicit configuration objects for the simple EnergyBench workflow."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


CANONICAL_ENERGY_MIN_KEV = 0.0
CANONICAL_ENERGY_MAX_KEV = 3000.0
CANONICAL_ENERGY_BIN_WIDTH_KEV = 5.0
CANONICAL_ENERGY_BIN_COUNT = 600


@dataclass
class TrainingConfig:
    """Standard training settings shared by classification and regression.

    The numerical defaults are the collaboration standard and should normally
    be used unchanged so experiments remain directly comparable.
    """

    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    seed: int = 42
    deterministic: bool = False
    use_amp: bool = True
    amp_precision: Literal["auto", "float16", "bfloat16"] = "auto"
    optimizer: Literal["adamw"] = "adamw"
    scheduler: Literal["cosine"] = "cosine"
    classification_loss: Literal["bce_with_logits"] = "bce_with_logits"
    regression_loss: Literal["mse"] = "mse"
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        _positive_integer(self.batch_size, "batch_size")
        _positive_integer(self.epochs, "epochs")
        _positive_float(self.learning_rate, "learning_rate")
        _nonnegative_float(self.weight_decay, "weight_decay")
        _positive_float(self.gradient_clip_norm, "gradient_clip_norm")
        _positive_integer(
            self.early_stopping_patience, "early_stopping_patience"
        )
        _nonnegative_float(
            self.early_stopping_min_delta, "early_stopping_min_delta"
        )
        _nonnegative_integer(self.seed, "seed")
        _nonnegative_integer(self.num_workers, "num_workers")
        if self.amp_precision not in {"auto", "float16", "bfloat16"}:
            raise ValueError(
                "amp_precision must be 'auto', 'float16', or 'bfloat16'"
            )
        if self.optimizer != "adamw":
            raise ValueError("the standard optimizer is 'adamw'")
        if self.scheduler != "cosine":
            raise ValueError("the standard scheduler is 'cosine'")
        if self.classification_loss != "bce_with_logits":
            raise ValueError(
                "the standard classification loss is 'bce_with_logits'"
            )
        if self.regression_loss != "mse":
            raise ValueError("the standard regression loss is 'mse'")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class ProjectionConfig:
    """Definition of the three ``XY/XZ/YZ`` detector projections.

    Coordinates and ``bin_size`` are in millimetres.  ``input_scale`` is
    applied after projection and, for classification, after event-energy
    normalization.
    """

    grid_size: int = 128
    bin_size: float = 30.0
    origin: tuple[float, float, float] = (-1920.0, -1920.0, -120.0)
    normalize_energy: bool = True
    input_scale: float = 100.0
    representation: Literal["energy", "binary_occupancy"] = "energy"

    def __post_init__(self) -> None:
        _positive_integer(self.grid_size, "grid_size")
        _positive_float(self.bin_size, "bin_size")
        if len(self.origin) != 3 or any(
            not math.isfinite(float(value)) for value in self.origin
        ):
            raise ValueError("origin must contain three finite coordinates")
        _positive_float(self.input_scale, "input_scale")
        if self.representation not in {"energy", "binary_occupancy"}:
            raise ValueError(
                "representation must be 'energy' or 'binary_occupancy'"
            )
        if self.representation == "binary_occupancy" and self.normalize_energy:
            raise ValueError(
                "binary occupancy cannot use event-energy normalization"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable projection dictionary."""

        payload = asdict(self)
        payload["origin"] = list(self.origin)
        return payload


@dataclass(frozen=True)
class EvaluationConfig:
    """EnergyBench protocol settings used by both evaluation entry points."""

    energy_bin_width_kev: float = CANONICAL_ENERGY_BIN_WIDTH_KEV
    energy_grid_min_kev: float = CANONICAL_ENERGY_MIN_KEV
    energy_grid_max_kev: float = CANONICAL_ENERGY_MAX_KEV
    energy_grid_bin_count: int = CANONICAL_ENERGY_BIN_COUNT
    energy_unit: Literal["MeV"] = "MeV"
    matching_target: Literal["overlap", "uniform"] = "overlap"
    min_per_class: int = 20
    min_valid_bins: int = 2
    support_trim_quantile: float = 0.005
    energy_roi: tuple[float, float] | None = None
    min_coverage: float = 0.5
    target_tpr: float = 0.90
    score_bins: int = 20
    min_per_bin: int = 20
    distance_correlation_max_samples: int = 1200
    performance_bins: int = 10
    fractional_energy_floor: float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        _canonical_value(
            self.energy_bin_width_kev,
            CANONICAL_ENERGY_BIN_WIDTH_KEV,
            "energy_bin_width_kev",
        )
        _canonical_value(
            self.energy_grid_min_kev,
            CANONICAL_ENERGY_MIN_KEV,
            "energy_grid_min_kev",
        )
        _canonical_value(
            self.energy_grid_max_kev,
            CANONICAL_ENERGY_MAX_KEV,
            "energy_grid_max_kev",
        )
        if (
            isinstance(self.energy_grid_bin_count, bool)
            or int(self.energy_grid_bin_count) != self.energy_grid_bin_count
            or int(self.energy_grid_bin_count) != CANONICAL_ENERGY_BIN_COUNT
        ):
            raise ValueError(
                "energy_grid_bin_count is fixed at 600 by the "
                "EnergyBench protocol"
            )
        if self.energy_unit != "MeV":
            raise ValueError("the NEXT workflow uses physical energy in MeV")
        if self.matching_target not in {"overlap", "uniform"}:
            raise ValueError("matching_target must be 'overlap' or 'uniform'")
        _positive_integer(self.min_per_class, "min_per_class")
        _positive_integer(self.min_valid_bins, "min_valid_bins")
        _probability(
            self.support_trim_quantile,
            "support_trim_quantile",
            upper=0.5,
            include_upper=False,
            include_lower=True,
        )
        if self.energy_roi is not None:
            if (
                len(self.energy_roi) != 2
                or not all(math.isfinite(float(x)) for x in self.energy_roi)
                or not float(self.energy_roi[1]) > float(self.energy_roi[0])
            ):
                raise ValueError("energy_roi must be (low, high) with high > low")
        _probability(
            self.min_coverage,
            "min_coverage",
            include_lower=True,
            include_upper=True,
        )
        _probability(self.target_tpr, "target_tpr")
        _positive_integer(self.score_bins, "score_bins")
        _positive_integer(self.min_per_bin, "min_per_bin")
        _positive_integer(
            self.distance_correlation_max_samples,
            "distance_correlation_max_samples",
        )
        _positive_integer(self.performance_bins, "performance_bins")
        if self.fractional_energy_floor is not None:
            _positive_float(
                self.fractional_energy_floor, "fractional_energy_floor"
            )
        _nonnegative_integer(self.seed, "seed")

    @property
    def energy_bin_width_mev(self) -> float:
        """The configured energy-bin width converted from keV to MeV."""

        return float(self.energy_bin_width_kev) / 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable protocol dictionary."""

        payload = asdict(self)
        if self.energy_roi is not None:
            payload["energy_roi"] = list(self.energy_roi)
        payload["energy_bin_width_mev"] = self.energy_bin_width_mev
        return payload


def _canonical_value(value: Any, expected: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number != expected:
        raise ValueError(
            f"{name} is fixed at {expected:g} keV by the EnergyBench protocol"
        )
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _probability(
    value: Any,
    name: str,
    *,
    upper: float = 1.0,
    include_lower: bool = False,
    include_upper: bool = False,
) -> float:
    number = float(value)
    lower_ok = number >= 0.0 if include_lower else number > 0.0
    upper_ok = number <= upper if include_upper else number < upper
    if not math.isfinite(number) or not lower_ok or not upper_ok:
        left = "[" if include_lower else "("
        right = "]" if include_upper else ")"
        raise ValueError(f"{name} must lie in {left}0, {upper}{right}")
    return number


__all__ = [
    "CANONICAL_ENERGY_BIN_COUNT",
    "CANONICAL_ENERGY_BIN_WIDTH_KEV",
    "CANONICAL_ENERGY_MAX_KEV",
    "CANONICAL_ENERGY_MIN_KEV",
    "EvaluationConfig",
    "ProjectionConfig",
    "TrainingConfig",
]
