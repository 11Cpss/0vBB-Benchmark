"""Data and optimization settings for published NEXT Transformers."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

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
