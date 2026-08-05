"""Configuration loading and validation for alternative NEXT classifiers.

The architecture directories intentionally contain small YAML files.  This
module owns all defaults and validation so a training entry point only needs
to call :func:`load_training_config` and pass the resulting mapping to the
shared runner.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/home/klz/Data/zeronu_benchmark/NEXT")
INPUT_KINDS = {
    "projection2d",
    "multiscale2d",
    "dense3d",
    "points",
    "graph",
    "hybrid",
    "sequence",
    "topology",
    "sparse3d",
}


def default_training_config() -> Dict[str, Any]:
    """Return a fresh, fully expanded default training configuration."""

    return {
        "data": {
            "root": str(DEFAULT_DATA_ROOT),
            "max_files_per_class": 100,
            "split_seed": 42,
            "split_fractions": [0.8, 0.1, 0.1],
            "num_workers": 0,
            "balance_training_classes": True,
            "event_shuffle_buffer_size": 0,
        },
        "representation": {
            "projection_grid_size": 128,
            "projection_bin_size": 30.0,
            "projection_origin": [-1920.0, -1920.0, -120.0],
            "projection_input_scale": 100.0,
            "fine_grid_size": 128,
            "fine_bin_size": 15.0,
            "point_bin_size": 15.0,
            "coordinate_scale": 1000.0,
            "max_points": 512,
            "dense_grid_size": 96,
            "dense_bin_size": 15.0,
            "center_projection": False,
        },
        "model": {},
        "training": {
            "batch_size": 8,
            "epochs": 50,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 12,
            "early_stopping_min_delta": 0.0,
            "seed": 42,
            "deterministic": False,
            "use_amp": True,
            "amp_precision": "auto",
        },
        "output": {
            "checkpoint_dir": str(PROJECT_ROOT / "02_models" / "checkpoints"),
            "log_dir": str(PROJECT_ROOT / "03_training_runs" / "logs"),
            "plot_dir": str(
                PROJECT_ROOT / "03_training_runs" / "history_plots"
            ),
            "allow_overwrite": False,
        },
    }


def _merge_mapping(
    destination: MutableMapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    for key, value in source.items():
        if (
            isinstance(value, Mapping)
            and isinstance(destination.get(key), MutableMapping)
        ):
            _merge_mapping(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)


def _require_mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a YAML mapping" % name)
    return dict(value)


def _positive_int(value: Any, name: str, allow_none: bool = False) -> Optional[int]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError("%s must be a positive integer" % name)
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a positive integer" % name) from exc
    if converted != value or converted <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return converted


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError("%s must be a non-negative integer" % name)
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a non-negative integer" % name) from exc
    if converted != value or converted < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return converted


def _positive_float(value: Any, name: str, allow_zero: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if not math.isfinite(converted) or (
        converted < 0.0 if allow_zero else converted <= 0.0
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError("%s must be finite and %s" % (name, qualifier))
    return converted


def _validate_fractions(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("data.split_fractions must contain three numbers")
    fractions = [float(item) for item in value]
    if len(fractions) != 3 or any(
        not math.isfinite(item) or item <= 0.0 for item in fractions
    ):
        raise ValueError("data.split_fractions must contain three positive numbers")
    if abs(sum(fractions) - 1.0) > 1.0e-9:
        raise ValueError("data.split_fractions must sum to 1")
    return fractions


def validate_training_config(
    config: Mapping[str, Any],
    architecture_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and normalize a merged configuration mapping."""

    result = copy.deepcopy(dict(config))
    for section in ("data", "representation", "model", "training", "output"):
        result[section] = _require_mapping(result.get(section), section)

    configured_architecture = result.get("architecture_id")
    if architecture_id is not None:
        if configured_architecture not in (None, architecture_id):
            raise ValueError(
                "configuration architecture_id %r does not match runner %r"
                % (configured_architecture, architecture_id)
            )
        result["architecture_id"] = str(architecture_id)

    data = result["data"]
    root = Path(str(data["root"])).expanduser()
    data["root"] = str(root)
    maximum = data.get("max_files_per_class")
    if maximum in (None, 0, "0"):
        data["max_files_per_class"] = None
    else:
        data["max_files_per_class"] = _positive_int(
            maximum, "data.max_files_per_class"
        )
    data["split_seed"] = int(data["split_seed"])
    data["split_fractions"] = _validate_fractions(data["split_fractions"])
    data["num_workers"] = _nonnegative_int(
        data["num_workers"], "data.num_workers"
    )
    data["event_shuffle_buffer_size"] = _nonnegative_int(
        data["event_shuffle_buffer_size"],
        "data.event_shuffle_buffer_size",
    )
    data["balance_training_classes"] = bool(
        data["balance_training_classes"]
    )

    representation = result["representation"]
    for field in ("projection_grid_size", "fine_grid_size", "dense_grid_size"):
        representation[field] = _positive_int(
            representation[field], "representation.%s" % field
        )
    for field in (
        "projection_bin_size",
        "projection_input_scale",
        "fine_bin_size",
        "point_bin_size",
        "coordinate_scale",
        "dense_bin_size",
    ):
        representation[field] = _positive_float(
            representation[field], "representation.%s" % field
        )
    origin = representation["projection_origin"]
    if not isinstance(origin, Sequence) or len(origin) != 3:
        raise ValueError("representation.projection_origin must contain x, y, z")
    converted_origin = [float(item) for item in origin]
    if any(not math.isfinite(item) for item in converted_origin):
        raise ValueError("representation.projection_origin must be finite")
    representation["projection_origin"] = converted_origin
    maximum_points = representation.get("max_points")
    representation["max_points"] = (
        None
        if maximum_points in (None, 0, "0")
        else _positive_int(maximum_points, "representation.max_points")
    )
    representation["center_projection"] = bool(
        representation.get("center_projection", False)
    )

    training = result["training"]
    for field in ("batch_size", "epochs", "early_stopping_patience"):
        training[field] = _positive_int(
            training[field], "training.%s" % field
        )
    training["seed"] = int(training["seed"])
    training["learning_rate"] = _positive_float(
        training["learning_rate"], "training.learning_rate"
    )
    training["weight_decay"] = _positive_float(
        training["weight_decay"], "training.weight_decay", allow_zero=True
    )
    training["gradient_clip_norm"] = _positive_float(
        training["gradient_clip_norm"], "training.gradient_clip_norm"
    )
    training["early_stopping_min_delta"] = _positive_float(
        training["early_stopping_min_delta"],
        "training.early_stopping_min_delta",
        allow_zero=True,
    )
    training["deterministic"] = bool(training["deterministic"])
    training["use_amp"] = bool(training["use_amp"])
    precision = str(training["amp_precision"]).strip().lower()
    aliases = {
        "auto": "auto",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
    }
    if precision not in aliases:
        raise ValueError(
            "training.amp_precision must be auto, bfloat16/bf16, or float16/fp16"
        )
    training["amp_precision"] = aliases[precision]

    output = result["output"]
    for field in ("checkpoint_dir", "log_dir", "plot_dir"):
        output[field] = str(Path(str(output[field])).expanduser())
    output["allow_overwrite"] = bool(output["allow_overwrite"])
    output["campaign_layout"] = bool(output.get("campaign_layout", False))
    return result


def load_training_config(
    path: str | Path,
    architecture_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one YAML file, apply defaults, and validate the result."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError("training configuration does not exist: %s" % path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    overrides = _require_mapping(loaded, str(config_path))
    merged = default_training_config()
    _merge_mapping(merged, overrides)
    merged["config_path"] = str(config_path)
    return validate_training_config(merged, architecture_id=architecture_id)


__all__ = [
    "DEFAULT_DATA_ROOT",
    "INPUT_KINDS",
    "PROJECT_ROOT",
    "default_training_config",
    "load_training_config",
    "validate_training_config",
]
