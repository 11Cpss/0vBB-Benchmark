#!/usr/bin/env python3
"""Run cnn_003_residual_spatial energy regression with Simple EnergyBench."""

from __future__ import annotations

import sys
from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ARCHITECTURES_ROOT = SCRIPT_DIR.parent
if str(ARCHITECTURES_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHITECTURES_ROOT))

from workflow_runner import main_for_architecture

from next_cnn.model import ResidualSpatialEnergyRegressor as EnergyRegressor


def initialize_regression_output(model: EnergyRegressor) -> None:
    """Zero the learned residual so the physical projection baseline is exact."""

    linear_layers = [
        module for module in model.regressor.modules() if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        raise ValueError("energy regressor has no linear output layer")
    torch.nn.init.zeros_(linear_layers[-1].weight)
    if linear_layers[-1].bias is not None:
        torch.nn.init.zeros_(linear_layers[-1].bias)


def validation_improved(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
) -> bool:
    """Select lower validation RMSE, using MAE as the exact-tie breaker."""

    candidate_key = (
        float(candidate["energy_rmse_mev"]),
        float(candidate["energy_mae_mev"]),
    )
    incumbent_key = (
        float(incumbent["energy_rmse_mev"]),
        float(incumbent["energy_mae_mev"]),
    )
    return candidate_key < incumbent_key


def early_stopping_improved(current: float, best: float, min_delta: float) -> bool:
    """Return true only for an improvement strictly larger than ``min_delta``."""

    return float(current) < float(best) - float(min_delta)


def normalize_regression_loss(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {"mse": "mse", "mean_squared_error": "mse", "smooth_l1": "smooth_l1", "huber": "smooth_l1"}
    if normalized not in aliases:
        raise ValueError("regression loss must be mse or smooth_l1")
    return aliases[normalized]


def complete_regression_metrics(
    target: Any,
    prediction: Any,
    *,
    energy_std: float,
    objective_name: str,
    smooth_l1_beta: float,
) -> dict[str, float]:
    """Compute the legacy compact regression diagnostics used by old callers."""

    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    estimate = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("target and prediction must be non-empty vectors of equal size")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise ValueError("target and prediction must be finite")
    beta = float(smooth_l1_beta)
    if not beta > 0.0:
        raise ValueError("smooth_l1_beta must be positive")
    residual = estimate - truth
    absolute = np.abs(residual)
    squared = np.square(residual)
    smooth = np.where(
        absolute < beta,
        0.5 * squared / beta,
        absolute - 0.5 * beta,
    )
    objective = normalize_regression_loss(objective_name)
    mse = float(np.mean(squared))
    smooth_loss = float(np.mean(smooth))
    truth_centered = truth - float(np.mean(truth))
    total_variance = float(np.sum(np.square(truth_centered)))
    r2 = float("nan") if total_variance == 0.0 else 1.0 - float(np.sum(squared)) / total_variance
    if truth.size < 2 or float(np.std(truth)) == 0.0 or float(np.std(estimate)) == 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(truth, estimate)[0, 1])
    return {
        "loss": mse if objective == "mse" else smooth_loss,
        "mse_loss": mse,
        "smooth_l1_loss": smooth_loss,
        "standardized_rmse": float(np.sqrt(mse) / float(energy_std)),
        "energy_rmse_mev": float(np.sqrt(mse)),
        "energy_mae_mev": float(np.mean(absolute)),
        "energy_bias_mev": float(np.mean(residual)),
        "energy_r2": r2,
        "energy_pearson_r": pearson,
        "energy_prediction_std_mev": float(np.std(estimate)),
        "energy_target_std_mev": float(np.std(truth)),
    }


def validate_checkpoint(checkpoint: Any, path: Path) -> Any:
    """Validate the stable minimum of legacy format-v2 regression checkpoints."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint root must be a mapping: {path}")
    if int(checkpoint.get("format_version", -1)) != 2:
        raise ValueError(f"unsupported checkpoint format in {path}")
    if checkpoint.get("task") != "energy_regression":
        raise ValueError(f"checkpoint task must be energy_regression: {path}")
    if checkpoint.get("model_name") not in {
        "ResidualSpatialNextCNN",
        "ResidualSpatialEnergyRegressor",
    }:
        raise ValueError(f"unsupported regression model in {path}")
    if not isinstance(checkpoint.get("model_config"), Mapping):
        raise ValueError(f"checkpoint model_config must be a mapping: {path}")
    return checkpoint


def checkpoint_payload(
    model: EnergyRegressor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    projection: Any,
    data_selection: Mapping[str, Any],
    model_suffix: str,
    training_file_count: int,
    validation_file_count: int,
    training_event_count: int,
    validation_event_count: int,
    use_amp: bool,
    amp_precision: str,
    objective_name: str,
    smooth_l1_beta: float,
    event_shuffle_buffer_size: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    early_stopping_state: Mapping[str, Any],
    baselines: Mapping[str, Any],
    energy_mean: float,
    energy_std: float,
    completed_epochs: int,
) -> dict[str, Any]:
    """Build the legacy format-v2 evidence payload retained for compatibility."""

    objective = normalize_regression_loss(objective_name)
    projection_dict = projection.to_dict() if hasattr(projection, "to_dict") else dict(projection)
    return {
        "format_version": 2,
        "task": "energy_regression",
        "model_name": "ResidualSpatialEnergyRegressor",
        "model_suffix": str(model_suffix),
        "epoch": int(epoch),
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": list(history),
        "projection_config": projection_dict,
        "input_feature_config": {
            "uses_deposited_energy_amplitude": projection_dict.get("representation") == "energy",
            "normalizes_event_energy": bool(projection_dict.get("normalize_energy", False)),
        },
        "energy_target_config": {
            "kind": "summed_voxel_deposited_energy",
            "unit": "MeV",
            "normalizer": {
                "transform": "standardize",
                "mean": float(energy_mean),
                "std": float(energy_std),
                "fit_split": "train",
            },
        },
        "objective": {
            "name": objective,
            "smooth_l1_beta": float(smooth_l1_beta),
        },
        "selection": {"metric": "validation_energy_rmse_mev", "tie_breaker": "validation_energy_mae_mev"},
        "training_config": {
            "data": {"event_shuffle_buffer_size": int(event_shuffle_buffer_size)},
            "use_amp": bool(use_amp),
            "amp_precision": str(amp_precision),
            "completed_epochs": int(completed_epochs),
        },
        "early_stopping": {
            "patience": int(early_stopping_patience),
            "min_delta": float(early_stopping_min_delta),
            **dict(early_stopping_state),
        },
        "data_selection": dict(data_selection),
        "counts": {
            "training_files": int(training_file_count),
            "validation_files": int(validation_file_count),
            "training_events": int(training_event_count),
            "validation_events": int(validation_event_count),
        },
        "baselines": dict(baselines),
    }


def validation_acceptance(
    metrics: Mapping[str, Any], baselines: Mapping[str, Any]
) -> dict[str, Any]:
    """Check improvement over constant/geometry baselines and basic bias sanity."""

    rmse = float(metrics["energy_rmse_mev"])
    mae = float(metrics["energy_mae_mev"])
    bias = abs(float(metrics["energy_bias_mev"]))
    r2 = float(metrics["energy_r2"])
    constant = baselines["constant_train_mean"]["validation"]
    geometry = baselines["geometry_linear"]["validation"]
    checks = {
        "beats_constant_rmse": rmse < float(constant["energy_rmse_mev"]),
        "beats_constant_mae": mae < float(constant["energy_mae_mev"]),
        "beats_geometry_rmse": rmse < float(geometry["energy_rmse_mev"]),
        "positive_r2": r2 > 0.0,
        "small_bias": bias < rmse,
    }
    return {"passed": all(checks.values()), "checks": checks}


if __name__ == "__main__":
    raise SystemExit(
        main_for_architecture("cnn_003_residual_spatial", task="regression")
    )
