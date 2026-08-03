#!/usr/bin/env python3
"""
train_energy_regression.py — CNN-003 residual-spatial energy regression

This program predicts summed voxel-deposited energy from binary XY/XZ/YZ voxel
occupancy only. Deposited-energy amplitudes are excluded from the model input;
they are used solely as the supervised regression target.

Run
---
    cd /home/wenyu/summer
    source .venv/bin/activate
    python 01_code/architectures/cnn_003_residual_spatial/train_energy_regression.py

Quick GPU smoke test
--------------------
    python 01_code/architectures/cnn_003_residual_spatial/train_energy_regression.py --smoke

Configuration values can be edited below or overridden with ``NEXT_*``
environment variables.  CUDA is mandatory: this script never falls back to
CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# Must be set before torch is imported for deterministic CUDA matrix products.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    """Find the repository root without depending on script nesting depth."""

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "next_cnn"
        ).is_dir():
            return candidate
    raise RuntimeError("cannot locate project root from %s" % start)


PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
CODE_SRC = PROJECT_ROOT / "01_code" / "src"
PACKAGE_SRC = PROJECT_ROOT / "src"
sys.dont_write_bytecode = True
for source_path in (CODE_SRC, PACKAGE_SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit(
        "PyTorch is missing. Activate the project environment first:\n"
        "  source .venv/bin/activate"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "tqdm is missing from .venv. Reinstall the project with:\n"
        "  python -m pip install -e ."
    ) from exc

import project_paths as paths
from next_cnn.data import (
    NextIterableDataset as NEXTDataset,
    ProjectionConfig,
    SourceFile,
    dataset_inventory,
    discover_source_files,
    iter_file_events,
)
from next_cnn.model import ResidualSpatialNextCNN as EnergyRegressor


# =============================================================================
# Configuration
# =============================================================================


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


MODEL_SUFFIX = os.environ.get(
    "NEXT_MODEL_SUFFIX",
    "_cnn_003_residual_spatial_energy_regression",
)
BASE_DATA = os.environ.get(
    "NEXT_DATA_ROOT", "/home/klz/Data/zeronu_benchmark/NEXT"
)

BATCH_SIZE = _env_int("NEXT_BATCH_SIZE", 16)
NUM_EPOCHS = _env_int("NEXT_NUM_EPOCHS", 50)
MAX_FILES_PER_CLASS = _env_int("NEXT_MAX_FILES_PER_CLASS", 100)
NUM_WORKERS = _env_int("NEXT_NUM_WORKERS", 0)

LEARNING_RATE = _env_float("NEXT_LEARNING_RATE", 1.0e-3)
WEIGHT_DECAY = _env_float("NEXT_WEIGHT_DECAY", 1.0e-3)
GRAD_CLIP = _env_float("NEXT_GRAD_CLIP", 1.0)
BASE_CHANNELS = _env_int("NEXT_BASE_CHANNELS", 4)
POOLED_SIZE = _env_int("NEXT_POOLED_SIZE", 1)
HEAD_FEATURES = _env_int("NEXT_HEAD_FEATURES", 32)
SMOOTH_L1_BETA = _env_float("NEXT_SMOOTH_L1_BETA", 1.0)
REGRESSION_LOSS = os.environ.get(
    "NEXT_REGRESSION_LOSS", "mse"
).strip().lower()
EVENT_SHUFFLE_BUFFER_SIZE = _env_int(
    "NEXT_EVENT_SHUFFLE_BUFFER_SIZE", 512
)
EARLY_STOPPING_PATIENCE = _env_int(
    "NEXT_EARLY_STOPPING_PATIENCE", 12
)
EARLY_STOPPING_MIN_DELTA = _env_float(
    "NEXT_EARLY_STOPPING_MIN_DELTA", 1.0e-6
)

SEED = _env_int("NEXT_SEED", 42)
DETERMINISTIC = _env_bool("NEXT_DETERMINISTIC", True)
USE_AMP = _env_bool("NEXT_USE_AMP", True)
AMP_PRECISION = os.environ.get("NEXT_AMP_PRECISION", "auto").strip().lower()
MAX_TRAIN_BATCHES = _env_int("NEXT_MAX_TRAIN_BATCHES", 0)
MAX_VALIDATION_BATCHES = _env_int("NEXT_MAX_VALIDATION_BATCHES", 0)

SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
# Classification labels do not define the standalone regression objective.
# Use every selected training event so the fitted target normalizer and the
# effective regression training distribution contain exactly the same events.
BALANCE_TRAINING_CLASSES = False

GRID_SIZE = 128
BIN_SIZE = 30.0
GRID_ORIGIN = (-1920.0, -1920.0, -120.0)
# Binary occupancy depends only on voxel coordinates.  It deliberately removes
# deposited-energy amplitudes, including the event-total shortcut.
NORMALIZE_EVENT_ENERGY = False
INPUT_SCALE = 1.0
INPUT_REPRESENTATION = "binary_occupancy"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the CNN-003 topology-only NEXT energy regressor on CUDA"
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one epoch, one HDF5 per class/split, at most two batches",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="load the best checkpoint and run validation only",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="deliberately replace best/last files for the requested model suffix",
    )
    return parser


def checkpoint_pair(model_suffix: str) -> Tuple[Path, Path]:
    """Return best/last checkpoint paths for one regression run."""

    return (
        paths.checkpoint_path("NEXTCNN%s_best.pt" % model_suffix),
        paths.checkpoint_path("NEXTCNN%s_last.pt" % model_suffix),
    )


def next_available_model_suffix(requested_suffix: str) -> str:
    """Preserve existing checkpoints by choosing _run2, _run3, ... ."""

    candidate = requested_suffix
    run_number = 1
    while True:
        best_checkpoint, last_checkpoint = checkpoint_pair(candidate)
        if not best_checkpoint.exists() and not last_checkpoint.exists():
            return candidate
        run_number += 1
        candidate = "%s_run%d" % (requested_suffix, run_number)


def require_cuda_device() -> torch.device:
    """Return cuda:0 or fail; CPU fallback is intentionally forbidden."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This program is GPU-only and will not use CPU."
        )
    device = torch.device("cuda:0")
    try:
        probe = torch.zeros(1, device=device)
        del probe
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            "CUDA device 0 is present but unusable; refusing CPU fallback"
        ) from exc
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_amp_precision(
    use_amp: bool = USE_AMP,
    requested_precision: str = AMP_PRECISION,
) -> Tuple[torch.dtype, str]:
    """Choose BF16 on modern CUDA devices, with scaled FP16 as fallback."""

    if not use_amp:
        return torch.float32, "disabled"

    aliases = {
        "auto": "auto",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
    }
    try:
        requested = aliases[requested_precision.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "NEXT_AMP_PRECISION must be auto, bfloat16/bf16, or float16/fp16"
        ) from exc

    if requested == "auto":
        requested = (
            "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        )
    if requested == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bfloat16 AMP was requested, but CUDA device 0 does not support it"
            )
        return torch.bfloat16, requested
    return torch.float16, requested


# =============================================================================
# Dataset / target normalization
# =============================================================================


def load_data(
    batch_size: int,
    max_files_per_class: Optional[int],
    projection: ProjectionConfig,
    device: torch.device,
    data_root: str = BASE_DATA,
    split_seed: int = SPLIT_SEED,
    split_fractions: Sequence[float] = SPLIT_FRACTIONS,
    dataset_seed: int = SEED,
    num_workers: int = NUM_WORKERS,
    event_shuffle_buffer_size: int = EVENT_SHUFFLE_BUFFER_SIZE,
) -> Tuple[Any, Any, NEXTDataset, List[SourceFile], List[SourceFile]]:
    common = {
        "root": data_root,
        "split_seed": int(split_seed),
        "split_fractions": split_fractions,
        "max_files_per_class": max_files_per_class,
    }
    train_files = discover_source_files(split="train", **common)
    validation_files = discover_source_files(split="validation", **common)

    train_dataset = NEXTDataset(
        train_files,
        projection=projection,
        shuffle_files=True,
        balance_classes=BALANCE_TRAINING_CLASSES,
        seed=dataset_seed,
        event_shuffle_buffer_size=event_shuffle_buffer_size,
    )
    validation_dataset = NEXTDataset(
        validation_files,
        projection=projection,
        shuffle_files=False,
        balance_classes=False,
        seed=dataset_seed,
    )
    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_options)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        **loader_options,
    )
    return (
        train_loader,
        validation_loader,
        train_dataset,
        train_files,
        validation_files,
    )


def fit_energy_normalizer(
    train_files: Sequence[SourceFile],
) -> Tuple[float, float, int]:
    """Fit population mean/std from training targets only using Welford."""

    count = 0
    mean = 0.0
    sum_squared_deviation = 0.0
    progress = tqdm(
        train_files,
        desc="Energy target statistics",
        unit="file",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for source in progress:
        for event in iter_file_events(source):
            value = float(event.energy_sum)
            count += 1
            delta = value - mean
            mean += delta / count
            sum_squared_deviation += delta * (value - mean)

    if count < 2:
        raise RuntimeError(
            "at least two training events are required to standardize energy"
        )
    standard_deviation = float(
        np.sqrt(sum_squared_deviation / count)
    )
    if not np.isfinite(mean) or not np.isfinite(standard_deviation):
        raise FloatingPointError("training energy statistics are non-finite")
    if standard_deviation <= 0.0:
        raise ValueError("training energy standard deviation must be positive")
    return float(mean), standard_deviation, count


# =============================================================================
# Metrics / validation / plots
# =============================================================================


def normalize_regression_loss(name: str) -> str:
    aliases = {
        "mse": "mse",
        "mean_squared_error": "mse",
        "smooth_l1": "smooth_l1",
        "smoothl1": "smooth_l1",
        "huber": "smooth_l1",
    }
    try:
        return aliases[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "NEXT_REGRESSION_LOSS must be mse or smooth_l1"
        ) from exc


def make_regression_criterion(
    name: str,
    smooth_l1_beta: float,
    device: torch.device,
) -> Tuple[Any, str]:
    normalized = normalize_regression_loss(name)
    if normalized == "mse":
        return nn.MSELoss().to(device), normalized
    if smooth_l1_beta <= 0.0:
        raise ValueError("Smooth-L1 beta must be positive")
    return nn.SmoothL1Loss(beta=smooth_l1_beta).to(device), normalized


def standardized_objective_loss(
    targets_mev: np.ndarray,
    predictions_mev: np.ndarray,
    energy_std: float,
    objective_name: str,
    smooth_l1_beta: float,
) -> float:
    residual = (
        np.asarray(predictions_mev, dtype=np.float64)
        - np.asarray(targets_mev, dtype=np.float64)
    ) / float(energy_std)
    normalized = normalize_regression_loss(objective_name)
    if normalized == "mse":
        return float(np.mean(np.square(residual)))
    absolute = np.abs(residual)
    values = np.where(
        absolute < smooth_l1_beta,
        0.5 * np.square(residual) / smooth_l1_beta,
        absolute - 0.5 * smooth_l1_beta,
    )
    return float(np.mean(values))


def standardized_smooth_l1_loss(
    targets_mev: np.ndarray,
    predictions_mev: np.ndarray,
    energy_std: float,
    smooth_l1_beta: float,
) -> float:
    return standardized_objective_loss(
        targets_mev,
        predictions_mev,
        energy_std,
        "smooth_l1",
        smooth_l1_beta,
    )


def regression_metrics(
    targets_mev: np.ndarray,
    predictions_mev: np.ndarray,
) -> Dict[str, Optional[float]]:
    targets = np.asarray(targets_mev, dtype=np.float64)
    predictions = np.asarray(predictions_mev, dtype=np.float64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("regression arrays must be aligned one-dimensional data")
    if len(targets) == 0:
        raise ValueError("regression metrics need at least one event")
    if np.any(~np.isfinite(targets)) or np.any(~np.isfinite(predictions)):
        raise FloatingPointError("regression metrics received non-finite values")

    residual = predictions - targets
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    bias = float(np.mean(residual))
    denominator = float(np.sum(np.square(targets - np.mean(targets))))
    r2 = None
    if denominator > 0.0:
        r2 = 1.0 - float(np.sum(np.square(residual))) / denominator
    prediction_std = float(np.std(predictions))
    pearson_r = None
    target_std = float(np.std(targets))
    if target_std > 0.0 and prediction_std > 0.0:
        pearson_r = float(np.corrcoef(targets, predictions)[0, 1])
    return {
        "energy_mae_mev": mae,
        "energy_rmse_mev": rmse,
        "energy_bias_mev": bias,
        "energy_r2": r2,
        "energy_prediction_std_mev": prediction_std,
        "energy_pearson_r": pearson_r,
    }


def complete_regression_metrics(
    targets_mev: np.ndarray,
    predictions_mev: np.ndarray,
    energy_std: float,
    objective_name: str,
    smooth_l1_beta: float,
) -> Dict[str, Any]:
    return {
        "loss": standardized_objective_loss(
            targets_mev,
            predictions_mev,
            energy_std,
            objective_name,
            smooth_l1_beta,
        ),
        "smooth_l1_loss": standardized_smooth_l1_loss(
            targets_mev,
            predictions_mev,
            energy_std,
            smooth_l1_beta,
        ),
        **regression_metrics(targets_mev, predictions_mev),
        "events": int(len(targets_mev)),
    }


def initialize_regression_output(model: EnergyRegressor) -> None:
    """Make an untrained regression model predict the train target mean."""

    output = model.head[-1]
    if not isinstance(output, nn.Linear) or output.out_features != 1:
        raise TypeError("ResidualSpatialNextCNN must end in a scalar Linear layer")
    nn.init.zeros_(output.weight)
    if output.bias is not None:
        nn.init.zeros_(output.bias)


def validation_selection_key(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    """Prefer lower validation RMSE, then lower validation MAE."""

    return (
        float(metrics["energy_rmse_mev"]),
        float(metrics["energy_mae_mev"]),
    )


def validation_improved(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> bool:
    return validation_selection_key(candidate) < validation_selection_key(
        incumbent
    )


def early_stopping_improved(
    candidate_rmse: float,
    reference_rmse: float,
    min_delta: float,
) -> bool:
    return float(candidate_rmse) < float(reference_rmse) - float(min_delta)


def validation_acceptance(
    metrics: Mapping[str, Any],
    baselines: Mapping[str, Any],
    bias_limit_mev: float = 1.0e-3,
) -> Dict[str, Any]:
    constant = baselines["constant_train_mean"]["validation"]
    geometry = baselines["geometry_linear"]["validation"]
    r2 = metrics.get("energy_r2")
    checks = {
        "rmse_below_constant": bool(
            metrics["energy_rmse_mev"] < constant["energy_rmse_mev"]
        ),
        "rmse_below_geometry_linear": bool(
            metrics["energy_rmse_mev"] < geometry["energy_rmse_mev"]
        ),
        "positive_r2": bool(r2 is not None and float(r2) > 0.0),
        "mae_not_worse_than_constant": bool(
            metrics["energy_mae_mev"] <= constant["energy_mae_mev"]
        ),
        "absolute_bias_below_limit": bool(
            abs(float(metrics["energy_bias_mev"])) < bias_limit_mev
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "bias_limit_mev": float(bias_limit_mev),
        "checks": checks,
        "conclusion": (
            "compact residual-spatial CNN beats both topology baselines"
            if all(checks.values())
            else (
                "pure-topology deep model has no demonstrated advantage "
                "over the geometry-linear baseline"
            )
        ),
    }


def run_epoch(
    loader: Any,
    model: EnergyRegressor,
    criterion: Any,
    device: torch.device,
    energy_mean: float,
    energy_std: float,
    description: str,
    optimizer: Optional[Any] = None,
    scaler: Optional[Any] = None,
    amp_dtype: torch.dtype = torch.float16,
    amp_enabled: bool = USE_AMP,
    max_batches: int = 0,
    objective_name: str = REGRESSION_LOSS,
    smooth_l1_beta: float = SMOOTH_L1_BETA,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_absolute_error = 0.0
    total_events = 0
    coverage_sum = 0.0
    amp_skipped_steps = 0
    consecutive_amp_skips = 0
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    progress = tqdm(loader, desc=description, dynamic_ncols=True, file=sys.stdout)
    for batch_index, batch in enumerate(progress):
        if max_batches and batch_index >= max_batches:
            break
        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        targets_mev = batch["energy_target"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        standardized_targets = (targets_mev - energy_mean) / energy_std
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                standardized_predictions = model(images)
                if standardized_predictions.shape != standardized_targets.shape:
                    raise ValueError(
                        "model output and energy target must both have shape (batch,)"
                    )
                loss = criterion(
                    standardized_predictions,
                    standardized_targets,
                )
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise FloatingPointError(
                    "%s produced a non-finite loss at batch %d"
                    % (description, batch_index)
                )
            if training:
                if scaler is None:
                    raise RuntimeError("training requires a GradScaler")
                scale_before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_values_are_finite = all(
                    bool(torch.isfinite(parameter.grad).all().detach().cpu())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP
                )
                gradient_norm_is_finite = bool(
                    torch.isfinite(gradient_norm).detach().cpu()
                )
                if (
                    not gradient_values_are_finite
                    or not gradient_norm_is_finite
                ) and not scaler.is_enabled():
                    raise FloatingPointError(
                        "%s produced non-finite gradients at batch %d without "
                        "dynamic loss scaling"
                        % (description, batch_index)
                    )
                if (
                    not gradient_norm_is_finite
                    and gradient_values_are_finite
                ):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "%s produced a non-finite total gradient norm at batch "
                        "%d even though each gradient tensor was finite; the "
                        "optimizer was not stepped"
                        % (description, batch_index)
                    )
                scaler.step(optimizer)
                scaler.update()
                step_was_skipped = (
                    scaler.is_enabled()
                    and float(scaler.get_scale()) < scale_before
                )
                if step_was_skipped:
                    amp_skipped_steps += 1
                    consecutive_amp_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    if consecutive_amp_skips >= 8:
                        raise FloatingPointError(
                            "%s had %d consecutive AMP-overflow batches; "
                            "stopping because this is no longer an isolated "
                            "dynamic-scaling event"
                            % (description, consecutive_amp_skips)
                        )
                else:
                    consecutive_amp_skips = 0
                    if not gradient_values_are_finite:
                        raise FloatingPointError(
                            "%s produced non-finite gradients at batch %d, "
                            "but GradScaler did not report a skipped step"
                            % (description, batch_index)
                        )

        predictions_mev = (
            energy_mean + energy_std * standardized_predictions.detach().float()
        )
        size = int(targets_mev.shape[0])
        total_events += size
        total_loss += float(loss.detach().cpu()) * size
        total_absolute_error += float(
            torch.sum(torch.abs(predictions_mev - targets_mev)).detach().cpu()
        )
        all_targets.append(targets_mev.detach().cpu().numpy().astype(np.float64))
        all_predictions.append(
            predictions_mev.cpu().numpy().astype(np.float64)
        )
        coverage_sum += float(
            torch.sum(batch["projection_coverage"]).detach().cpu()
        )
        progress.set_postfix(
            loss="%.4f" % (total_loss / total_events),
            mae_mev="%.5f" % (total_absolute_error / total_events),
            events=total_events,
            amp_skip=amp_skipped_steps,
        )

    if total_events == 0:
        raise RuntimeError("data loader produced no events")
    targets_array = np.concatenate(all_targets)
    predictions_array = np.concatenate(all_predictions)
    metrics: Dict[str, Any] = {
        **complete_regression_metrics(
            targets_array,
            predictions_array,
            energy_std,
            objective_name,
            smooth_l1_beta,
        ),
        # Preserve the exact loss value produced by the configured PyTorch
        # criterion, including its autocast arithmetic.
        "loss": total_loss / total_events,
        "mean_projection_coverage": coverage_sum / total_events,
        "amp_skipped_steps": amp_skipped_steps,
    }
    return metrics, targets_array, predictions_array


def collect_geometry_baseline_data(
    loader: Any,
    device: torch.device,
    description: str,
    max_batches: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    progress = tqdm(loader, desc=description, dynamic_ncols=True, file=sys.stdout)
    with torch.inference_mode():
        for batch_index, batch in enumerate(progress):
            if max_batches and batch_index >= max_batches:
                break
            images = batch["image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            features = EnergyRegressor.global_geometry_features(images)
            feature_batches.append(
                features.cpu().numpy().astype(np.float64)
            )
            target_batches.append(
                batch["energy_target"].numpy().astype(np.float64)
            )
    if not feature_batches:
        raise RuntimeError("geometry baseline data loader produced no events")
    return np.concatenate(feature_batches), np.concatenate(target_batches)


def fit_reference_baselines(
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    energy_mean: float,
    energy_std: float,
    objective_name: str,
    smooth_l1_beta: float,
    max_train_batches: int = 0,
    max_validation_batches: int = 0,
) -> Dict[str, Any]:
    """Fit train-only constant and 15-feature linear reference models."""

    train_features, train_targets = collect_geometry_baseline_data(
        train_loader,
        device,
        "Geometry baseline train data",
        max_train_batches,
    )
    validation_features, validation_targets = collect_geometry_baseline_data(
        validation_loader,
        device,
        "Geometry baseline validation data",
        max_validation_batches,
    )

    feature_mean = np.mean(train_features, axis=0)
    feature_std = np.std(train_features, axis=0)
    feature_std[feature_std == 0.0] = 1.0
    train_design = np.column_stack(
        (
            np.ones(len(train_features)),
            (train_features - feature_mean) / feature_std,
        )
    )
    standardized_train_targets = (
        train_targets - energy_mean
    ) / energy_std
    coefficients = np.linalg.lstsq(
        train_design,
        standardized_train_targets,
        rcond=None,
    )[0]

    def geometry_predictions(features: np.ndarray) -> np.ndarray:
        design = np.column_stack(
            (
                np.ones(len(features)),
                (features - feature_mean) / feature_std,
            )
        )
        return energy_mean + energy_std * (design @ coefficients)

    train_constant = np.full_like(train_targets, energy_mean)
    validation_constant = np.full_like(validation_targets, energy_mean)
    train_geometry = geometry_predictions(train_features)
    validation_geometry = geometry_predictions(validation_features)
    return {
        "constant_train_mean": {
            "prediction_mev": float(energy_mean),
            "train": complete_regression_metrics(
                train_targets,
                train_constant,
                energy_std,
                objective_name,
                smooth_l1_beta,
            ),
            "validation": complete_regression_metrics(
                validation_targets,
                validation_constant,
                energy_std,
                objective_name,
                smooth_l1_beta,
            ),
        },
        "geometry_linear": {
            "kind": "ordinary_least_squares",
            "features": (
                "per-view log mass, x/y centroid, and x/y variance"
            ),
            "fit_split": "train",
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "coefficients_standardized_target": coefficients.tolist(),
            "train": complete_regression_metrics(
                train_targets,
                train_geometry,
                energy_std,
                objective_name,
                smooth_l1_beta,
            ),
            "validation": complete_regression_metrics(
                validation_targets,
                validation_geometry,
                energy_std,
                objective_name,
                smooth_l1_beta,
            ),
        },
    }


def save_history_plot(
    history: Sequence[Mapping[str, Any]],
    model_suffix: str,
    run_id: str,
) -> None:
    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record["train"]["loss"]) for record in history]
    validation_loss = [
        float(record["validation"]["loss"]) for record in history
    ]
    train_mae = [
        float(record["train"]["energy_mae_mev"]) for record in history
    ]
    validation_mae = [
        float(record["validation"]["energy_mae_mev"])
        for record in history
    ]
    train_rmse = [
        float(record["train"]["energy_rmse_mev"]) for record in history
    ]
    validation_rmse = [
        float(record["validation"]["energy_rmse_mev"])
        for record in history
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="online train")
    axes[0].plot(epochs, validation_loss, marker="o", label="validation")
    axes[0].set_xlabel("Epoch")
    objective_name = str(history[0].get("objective", "smooth_l1"))
    axes[0].set_ylabel("Standardized %s loss" % objective_name.upper())
    axes[0].legend()
    axes[1].plot(
        epochs,
        train_mae,
        marker="o",
        label="online train MAE",
    )
    axes[1].plot(
        epochs,
        validation_mae,
        marker="o",
        label="validation MAE",
    )
    axes[1].plot(
        epochs,
        train_rmse,
        marker="s",
        linestyle="--",
        label="online train RMSE",
    )
    axes[1].plot(
        epochs,
        validation_rmse,
        marker="s",
        linestyle="--",
        label="validation RMSE",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Energy error (MeV)")
    axes[1].legend()
    figure.suptitle("CNN-003 energy regression%s" % model_suffix)
    figure.tight_layout()
    figure.savefig(paths.history_plot_path(model_suffix, run_id), dpi=180)
    plt.close(figure)


def save_regression_plot(
    targets_mev: np.ndarray,
    predictions_mev: np.ndarray,
    model_suffix: str,
    run_id: str,
) -> None:
    targets = np.asarray(targets_mev, dtype=np.float64)
    predictions = np.asarray(predictions_mev, dtype=np.float64)
    if len(targets) == 0 or predictions.shape != targets.shape:
        return
    valid = np.isfinite(targets) & np.isfinite(predictions)
    targets = targets[valid]
    predictions = predictions[valid]
    if len(targets) == 0:
        return

    max_scatter_events = 20_000
    if len(targets) > max_scatter_events:
        generator = np.random.default_rng(SEED)
        indices = np.sort(
            generator.choice(
                len(targets),
                size=max_scatter_events,
                replace=False,
            )
        )
    else:
        indices = np.arange(len(targets))
    residual = predictions - targets
    metrics = regression_metrics(targets, predictions)
    low = float(min(np.min(targets), np.min(predictions)))
    high = float(max(np.max(targets), np.max(predictions)))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(
        targets[indices],
        predictions[indices],
        s=4,
        alpha=0.25,
        rasterized=True,
    )
    axes[0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[0].set_xlabel("Target energy (MeV)")
    axes[0].set_ylabel("Predicted energy (MeV)")
    axes[0].set_title(
        "MAE=%.5f MeV, RMSE=%.5f MeV"
        % (
            metrics["energy_mae_mev"],
            metrics["energy_rmse_mev"],
        )
    )
    axes[1].hist(residual, bins=60, histtype="stepfilled", alpha=0.55)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Prediction - target (MeV)")
    axes[1].set_ylabel("Events")
    axes[1].set_title(
        "Bias=%.5f MeV, R2=%s"
        % (
            metrics["energy_bias_mev"],
            "NA"
            if metrics["energy_r2"] is None
            else "%.4f" % metrics["energy_r2"],
        )
    )
    figure.suptitle("CNN-003 validation energy regression")
    figure.tight_layout()
    figure.savefig(paths.regression_plot_path(model_suffix, run_id), dpi=180)
    plt.close(figure)


def append_validation_log(
    path: Path,
    run_id: str,
    record: Mapping[str, Any],
    learning_rate: float,
    improved: bool,
    baselines: Mapping[str, Any],
    patience_counter: int,
) -> None:
    constant_validation = baselines["constant_train_mean"]["validation"]
    geometry_validation = baselines["geometry_linear"]["validation"]
    row = {
        "run_id": run_id,
        "epoch": int(record["epoch"]),
        "objective": str(record["objective"]),
        "train_loss": float(record["train"]["loss"]),
        "train_smooth_l1_loss": float(
            record["train"]["smooth_l1_loss"]
        ),
        "train_energy_mae_mev": float(
            record["train"]["energy_mae_mev"]
        ),
        "train_energy_rmse_mev": float(
            record["train"]["energy_rmse_mev"]
        ),
        "train_energy_bias_mev": float(
            record["train"]["energy_bias_mev"]
        ),
        "train_energy_r2": record["train"]["energy_r2"],
        "train_energy_prediction_std_mev": float(
            record["train"]["energy_prediction_std_mev"]
        ),
        "train_energy_pearson_r": record["train"]["energy_pearson_r"],
        "train_amp_skipped_steps": int(
            record["train"]["amp_skipped_steps"]
        ),
        "validation_loss": float(record["validation"]["loss"]),
        "validation_smooth_l1_loss": float(
            record["validation"]["smooth_l1_loss"]
        ),
        "validation_energy_mae_mev": float(
            record["validation"]["energy_mae_mev"]
        ),
        "validation_energy_rmse_mev": float(
            record["validation"]["energy_rmse_mev"]
        ),
        "validation_energy_bias_mev": float(
            record["validation"]["energy_bias_mev"]
        ),
        "validation_energy_r2": record["validation"]["energy_r2"],
        "validation_energy_prediction_std_mev": float(
            record["validation"]["energy_prediction_std_mev"]
        ),
        "validation_energy_pearson_r": record["validation"][
            "energy_pearson_r"
        ],
        "projection_coverage": float(
            record["validation"]["mean_projection_coverage"]
        ),
        "constant_validation_mae_mev": float(
            constant_validation["energy_mae_mev"]
        ),
        "constant_validation_rmse_mev": float(
            constant_validation["energy_rmse_mev"]
        ),
        "constant_validation_r2": constant_validation["energy_r2"],
        "geometry_validation_mae_mev": float(
            geometry_validation["energy_mae_mev"]
        ),
        "geometry_validation_rmse_mev": float(
            geometry_validation["energy_rmse_mev"]
        ),
        "geometry_validation_r2": geometry_validation["energy_r2"],
        "beats_constant_rmse": int(
            record["validation"]["energy_rmse_mev"]
            < constant_validation["energy_rmse_mev"]
        ),
        "beats_geometry_rmse": int(
            record["validation"]["energy_rmse_mev"]
            < geometry_validation["energy_rmse_mev"]
        ),
        "learning_rate": float(learning_rate),
        "early_stopping_patience_counter": int(patience_counter),
        "improved": int(improved),
    }
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# Checkpoint provenance
# =============================================================================


def checkpoint_payload(
    model: EnergyRegressor,
    optimizer: Any,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    projection: ProjectionConfig,
    data_selection: Mapping[str, Any],
    model_suffix: str,
    max_files_per_class: Optional[int],
    actual_epochs: int,
    max_train_batches: int,
    max_validation_batches: int,
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
    normalizer_event_count: int,
) -> Dict[str, Any]:
    energy_target_config = {
        "kind": "summed_voxel_deposited_energy",
        "unit": "MeV",
        "source": "/MC/hits/table values_block_1 energy column",
        "derivation": "float64 sum of voxel energies grouped by event_id",
        "normalizer": {
            "transform": "standardize",
            "mean": float(energy_mean),
            "std": float(energy_std),
            "fit_split": "train",
            "event_count": int(normalizer_event_count),
        },
    }
    return {
        "format_version": 2,
        "task": "energy_regression",
        "model_name": "ResidualSpatialNextCNN",
        "model_suffix": model_suffix,
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "energy_target_config": energy_target_config,
        "projection_config": projection.to_dict(),
        "input_feature_config": {
            "kind": "binary_voxel_occupancy",
            "source": "voxel coordinates only",
            "uses_deposited_energy_amplitude": False,
            "pixel_values": [0.0, 1.0],
        },
        "split_config": {
            "seed": SPLIT_SEED,
            "fractions": list(SPLIT_FRACTIONS),
        },
        "epoch": int(epoch),
        "objective": {
            "name": str(objective_name),
            "target_space": "standardized_energy",
            "smooth_l1_beta": float(smooth_l1_beta),
        },
        "selection": {
            "metric": "validation_energy_rmse_mev",
            "direction": "minimize",
            "tie_break": "validation_energy_mae_mev",
        },
        "early_stopping": {
            "metric": "validation_energy_rmse_mev",
            "direction": "minimize",
            "patience": int(early_stopping_patience),
            "min_delta_mev": float(early_stopping_min_delta),
            **dict(early_stopping_state),
        },
        "baselines": dict(baselines),
        "training_config": {
            "entrypoint": (
                "01_code/architectures/cnn_003_residual_spatial/"
                "train_energy_regression.py"
            ),
            "data": {
                "root": BASE_DATA,
                "max_files_per_class": max_files_per_class,
                "balance_training_classes": BALANCE_TRAINING_CLASSES,
                "event_shuffle_buffer_size": int(
                    event_shuffle_buffer_size
                ),
            },
            "projection": projection.to_dict(),
            "model": model.config_dict(),
            "target": energy_target_config,
            "training": {
                "batch_size": BATCH_SIZE,
                "epochs": actual_epochs,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": GRAD_CLIP,
                "regression_loss": str(objective_name),
                "smooth_l1_beta": float(smooth_l1_beta),
                "early_stopping_patience": int(
                    early_stopping_patience
                ),
                "early_stopping_min_delta_mev": float(
                    early_stopping_min_delta
                ),
                "num_workers": NUM_WORKERS,
                "max_train_batches": max_train_batches,
                "max_validation_batches": max_validation_batches,
                "seed": SEED,
                "device": "cuda:0",
                "use_amp": use_amp,
                "amp_precision": amp_precision,
                "deterministic": DETERMINISTIC,
            },
        },
        "data_selection": dict(data_selection),
        "history": list(history),
    }


def validate_checkpoint(
    loaded: Any,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    if not isinstance(loaded, dict) or loaded.get("format_version") != 2:
        raise ValueError("unsupported checkpoint format: %s" % checkpoint_path)
    if loaded.get("task") != "energy_regression":
        raise ValueError(
            "checkpoint is not an energy-regression run: %s" % checkpoint_path
        )
    if loaded.get("model_name") != "ResidualSpatialNextCNN":
        raise ValueError(
            "checkpoint model is not CNN-003 topology-only regression: %s"
            % checkpoint_path
        )
    model_config = loaded.get("model_config")
    required_model_config = {
        "base_channels",
        "pooled_size",
        "head_features",
    }
    if not isinstance(model_config, Mapping) or not required_model_config.issubset(
        model_config
    ):
        raise ValueError("checkpoint model configuration is incomplete")
    normalizer = loaded.get("energy_target_config", {}).get("normalizer", {})
    required = {"mean", "std", "fit_split", "transform"}
    if not isinstance(normalizer, Mapping) or not required.issubset(normalizer):
        raise ValueError("checkpoint energy normalizer is incomplete")
    if normalizer["fit_split"] != "train":
        raise ValueError("checkpoint energy normalizer was not fit on train")
    if normalizer["transform"] != "standardize":
        raise ValueError("unsupported checkpoint energy transform")
    if not float(normalizer["std"]) > 0.0:
        raise ValueError("checkpoint energy standard deviation must be positive")
    projection = ProjectionConfig.from_dict(loaded.get("projection_config", {}))
    if projection.representation != "binary_occupancy":
        raise ValueError("checkpoint input is not binary occupancy")
    objective = loaded.get("objective")
    if objective is not None:
        if not isinstance(objective, Mapping) or "name" not in objective:
            raise ValueError("checkpoint objective configuration is incomplete")
        normalize_regression_loss(str(objective["name"]))
    return loaded


# =============================================================================
# Training
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    device = require_cuda_device()
    seed_everything(SEED)
    paths.ensure_layout()

    data_root = BASE_DATA
    smooth_l1_beta = SMOOTH_L1_BETA
    if smooth_l1_beta <= 0.0:
        raise ValueError("NEXT_SMOOTH_L1_BETA must be positive")
    objective_name = normalize_regression_loss(REGRESSION_LOSS)
    event_shuffle_buffer_size = EVENT_SHUFFLE_BUFFER_SIZE
    if event_shuffle_buffer_size < 0:
        raise ValueError(
            "NEXT_EVENT_SHUFFLE_BUFFER_SIZE must be non-negative"
        )
    early_stopping_patience = EARLY_STOPPING_PATIENCE
    if early_stopping_patience < 1:
        raise ValueError(
            "NEXT_EARLY_STOPPING_PATIENCE must be a positive integer"
        )
    early_stopping_min_delta = EARLY_STOPPING_MIN_DELTA
    if not np.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0:
        raise ValueError(
            "NEXT_EARLY_STOPPING_MIN_DELTA must be finite and non-negative"
        )

    use_amp = USE_AMP
    requested_amp_precision = AMP_PRECISION

    requested_model_suffix = MODEL_SUFFIX + ("_smoke" if args.smoke else "")
    model_suffix = requested_model_suffix
    if not args.full_validation and not args.allow_existing:
        model_suffix = next_available_model_suffix(requested_model_suffix)
        if model_suffix != requested_model_suffix:
            print(
                "Checkpoint suffix %s already exists; preserving it and "
                "starting this run as %s."
                % (requested_model_suffix, model_suffix),
                flush=True,
            )
    run_id = os.environ.get("NEXT_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
    max_files = 1 if args.smoke else MAX_FILES_PER_CLASS
    if max_files <= 0:
        max_files = None
    epochs = 1 if args.smoke else NUM_EPOCHS
    max_train_batches = 2 if args.smoke else MAX_TRAIN_BATCHES
    max_validation_batches = 2 if args.smoke else MAX_VALIDATION_BATCHES
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS

    best_checkpoint, last_checkpoint = checkpoint_pair(model_suffix)
    validation_log = paths.validation_log_path(model_suffix, run_id)
    history_json = paths.history_json_path(model_suffix, run_id)

    validation_checkpoint: Optional[Dict[str, Any]] = None
    split_seed = SPLIT_SEED
    split_fractions: Sequence[float] = SPLIT_FRACTIONS
    model_configuration: Optional[Dict[str, Any]] = None
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None
    normalizer_event_count = 0
    if args.full_validation:
        if not best_checkpoint.is_file():
            raise FileNotFoundError(
                "best checkpoint not found: %s" % best_checkpoint
            )
        loaded = torch.load(
            best_checkpoint,
            map_location=device,
            weights_only=False,
        )
        validation_checkpoint = validate_checkpoint(loaded, best_checkpoint)
        projection = ProjectionConfig.from_dict(
            validation_checkpoint["projection_config"]
        )
        model_configuration = dict(validation_checkpoint["model_config"])
        split_seed = int(validation_checkpoint["split_config"]["seed"])
        split_fractions = list(
            validation_checkpoint["split_config"]["fractions"]
        )
        saved_data_config = validation_checkpoint.get(
            "training_config", {}
        ).get("data", {})
        data_root = str(saved_data_config.get("root", data_root))
        max_files = saved_data_config.get("max_files_per_class", max_files)
        if max_files is not None and int(max_files) <= 0:
            max_files = None
        saved_training_config = validation_checkpoint.get(
            "training_config", {}
        ).get("training", {})
        objective_config = validation_checkpoint.get("objective")
        if isinstance(objective_config, Mapping):
            objective_name = normalize_regression_loss(
                str(objective_config.get("name", "smooth_l1"))
            )
        else:
            # Format-v2 checkpoints created before objective metadata always
            # used standardized Smooth-L1.
            objective_name = normalize_regression_loss(
                str(
                    saved_training_config.get(
                        "regression_loss", "smooth_l1"
                    )
                )
            )
        smooth_l1_beta = float(
            saved_training_config.get("smooth_l1_beta", smooth_l1_beta)
        )
        if smooth_l1_beta <= 0.0:
            raise ValueError(
                "checkpoint Smooth-L1 beta must be positive"
            )
        batch_size = int(saved_training_config.get("batch_size", batch_size))
        num_workers = int(
            saved_training_config.get("num_workers", num_workers)
        )
        event_shuffle_buffer_size = int(
            saved_data_config.get("event_shuffle_buffer_size", 0)
        )
        max_validation_batches = int(
            saved_training_config.get(
                "max_validation_batches",
                max_validation_batches,
            )
        )
        use_amp = bool(saved_training_config.get("use_amp", USE_AMP))
        requested_amp_precision = str(
            saved_training_config.get(
                "amp_precision",
                "float16" if use_amp else "disabled",
            )
        )
        normalizer = validation_checkpoint["energy_target_config"][
            "normalizer"
        ]
        energy_mean = float(normalizer["mean"])
        energy_std = float(normalizer["std"])
        normalizer_event_count = int(normalizer.get("event_count", 0))
    else:
        projection = ProjectionConfig(
            grid_size=GRID_SIZE,
            bin_size=BIN_SIZE,
            origin=GRID_ORIGIN,
            normalize_energy=NORMALIZE_EVENT_ENERGY,
            input_scale=INPUT_SCALE,
            representation=INPUT_REPRESENTATION,
        )

    amp_dtype, amp_precision = resolve_amp_precision(
        use_amp,
        requested_amp_precision,
    )

    (
        train_loader,
        validation_loader,
        train_dataset,
        train_files,
        validation_files,
    ) = load_data(
        batch_size,
        max_files,
        projection,
        device,
        data_root=data_root,
        split_seed=split_seed,
        split_fractions=split_fractions,
        dataset_seed=SEED,
        num_workers=num_workers,
        event_shuffle_buffer_size=event_shuffle_buffer_size,
    )

    if validation_checkpoint is not None:
        expected_inventory = validation_checkpoint["data_selection"].get(
            "inventory"
        )
        actual_inventory = dataset_inventory(data_root)
        if actual_inventory != expected_inventory:
            raise ValueError(
                "NEXT data inventory differs from the checkpoint; "
                "refusing validation"
            )
        expected_groups = set(
            validation_checkpoint["data_selection"].get(
                "validation_groups", []
            )
        )
        actual_groups = {source.group_id for source in validation_files}
        if actual_groups != expected_groups:
            raise ValueError(
                "validation file selection differs from the checkpoint; "
                "refusing validation"
            )
    else:
        energy_mean, energy_std, normalizer_event_count = (
            fit_energy_normalizer(train_files)
        )

    if energy_mean is None or energy_std is None:
        raise RuntimeError("energy target normalizer was not resolved")

    print("Using GPU : %s" % torch.cuda.get_device_name(0))
    print("PyTorch   : %s (CUDA %s)" % (torch.__version__, torch.version.cuda))
    print("Data      : %s" % data_root)
    print("Model     : CNN-003 residual-spatial energy regression%s" % model_suffix)
    print(
        "Input     : (3, %d, %d), %s"
        % (
            projection.grid_size,
            projection.grid_size,
            projection.representation,
        )
    )
    print("Batch     : %d" % batch_size)
    print("Workers   : %d" % num_workers)
    print(
        "Shuffle   : %d-event deterministic training buffer"
        % event_shuffle_buffer_size
    )
    print("Epochs    : %d" % epochs)
    print(
        "AMP       : %s"
        % ("False" if not use_amp else "True (%s)" % amp_precision)
    )
    print(
        "Files     : train=%d validation=%d%s"
        % (
            len(train_files),
            len(validation_files),
            " (smoke)" if args.smoke else "",
        )
    )
    print(
        "Target    : mean=%.9f MeV std=%.9f MeV (train events=%d)"
        % (energy_mean, energy_std, normalizer_event_count)
    )
    print("Objective : %s (standardized energy)" % objective_name.upper())
    print("Metric    : SmoothL1 beta=%.6g" % smooth_l1_beta)

    if model_configuration is None:
        model_configuration = {
            "base_channels": BASE_CHANNELS,
            "pooled_size": POOLED_SIZE,
            "head_features": HEAD_FEATURES,
        }
    if projection.representation != "binary_occupancy":
        raise ValueError("CNN-003 topology-only regression requires binary occupancy")
    if projection.normalize_energy or not np.isclose(projection.input_scale, 1.0):
        raise ValueError("binary occupancy must be unnormalized with input_scale=1")
    model = EnergyRegressor(**model_configuration).to(device)
    if not args.full_validation:
        initialize_regression_output(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        "Parameters: %s total, %s trainable"
        % (
            format(parameter_count, ","),
            format(trainable_parameter_count, ","),
        )
    )

    criterion, objective_name = make_regression_criterion(
        objective_name,
        smooth_l1_beta,
        device,
    )
    if args.full_validation:
        if validation_checkpoint is None:
            raise RuntimeError("validation checkpoint was not loaded")
        model.load_state_dict(
            validation_checkpoint["model_state_dict"],
            strict=True,
        )
        metrics, targets, predictions = run_epoch(
            validation_loader,
            model,
            criterion,
            device,
            energy_mean,
            energy_std,
            description="Validation",
            amp_dtype=amp_dtype,
            amp_enabled=use_amp,
            max_batches=max_validation_batches,
            objective_name=objective_name,
            smooth_l1_beta=smooth_l1_beta,
        )
        save_regression_plot(
            targets,
            predictions,
            model_suffix,
            run_id,
        )
        print("Validation: %s" % json.dumps(metrics, ensure_ascii=False))
        return 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=LEARNING_RATE * 0.05,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )

    inventory = dataset_inventory(data_root)
    data_selection = {
        "inventory": inventory,
        "training_groups": sorted(
            source.group_id for source in train_files
        ),
        "validation_groups": sorted(
            source.group_id for source in validation_files
        ),
    }

    history: List[Dict[str, Any]] = []

    baselines = fit_reference_baselines(
        train_loader,
        validation_loader,
        device,
        energy_mean,
        energy_std,
        objective_name,
        smooth_l1_beta,
        max_train_batches,
        max_validation_batches,
    )
    constant_validation = baselines["constant_train_mean"]["validation"]
    geometry_validation = baselines["geometry_linear"]["validation"]
    print(
        "Constant baseline: MAE=%.8g MeV RMSE=%.8g MeV R2=%s"
        % (
            constant_validation["energy_mae_mev"],
            constant_validation["energy_rmse_mev"],
            "NA"
            if constant_validation["energy_r2"] is None
            else "%.9f" % constant_validation["energy_r2"],
        )
    )
    print(
        "Geometry baseline: MAE=%.8g MeV RMSE=%.8g MeV R2=%s"
        % (
            geometry_validation["energy_mae_mev"],
            geometry_validation["energy_rmse_mev"],
            "NA"
            if geometry_validation["energy_r2"] is None
            else "%.9f" % geometry_validation["energy_r2"],
        )
    )

    # The zero-initialized scalar output makes this an exact train-mean
    # baseline instead of a random-network checkpoint.
    initial_validation_metrics, _, _ = run_epoch(
        validation_loader,
        model,
        criterion,
        device,
        energy_mean,
        energy_std,
        description="Initial validation (epoch 0)",
        amp_dtype=amp_dtype,
        amp_enabled=use_amp,
        max_batches=max_validation_batches,
        objective_name=objective_name,
        smooth_l1_beta=smooth_l1_beta,
    )
    best_validation_metrics = dict(initial_validation_metrics)
    patience_counter = 0
    significant_reference_rmse = float(
        initial_validation_metrics["energy_rmse_mev"]
    )
    early_stopping_state = {
        "patience_counter": patience_counter,
        "reference_rmse_mev": significant_reference_rmse,
        "stopped_early": False,
    }
    initial_payload = checkpoint_payload(
        model,
        optimizer,
        0,
        history,
        projection,
        data_selection,
        model_suffix,
        max_files,
        epochs,
        max_train_batches,
        max_validation_batches,
        use_amp,
        amp_precision,
        objective_name,
        smooth_l1_beta,
        event_shuffle_buffer_size,
        early_stopping_patience,
        early_stopping_min_delta,
        early_stopping_state,
        baselines,
        energy_mean,
        energy_std,
        normalizer_event_count,
    )
    initial_payload["initial_validation_metrics"] = dict(
        initial_validation_metrics
    )
    torch.save(initial_payload, best_checkpoint)
    print(
        "Initial validation: loss=%.8g MAE=%.8g MeV RMSE=%.8g MeV R2=%s"
        % (
            initial_validation_metrics["loss"],
            initial_validation_metrics["energy_mae_mev"],
            initial_validation_metrics["energy_rmse_mev"],
            "NA"
            if initial_validation_metrics["energy_r2"] is None
            else "%.9f" % initial_validation_metrics["energy_r2"],
        )
    )
    print("Training starts. Validation runs after every epoch.\n")

    for epoch in range(1, epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_dataset.set_epoch(epoch)
        train_metrics, _, _ = run_epoch(
            train_loader,
            model,
            criterion,
            device,
            energy_mean,
            energy_std,
            description="Epoch %d/%d" % (epoch, epochs),
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            amp_enabled=use_amp,
            max_batches=max_train_batches,
            objective_name=objective_name,
            smooth_l1_beta=smooth_l1_beta,
        )
        validation_metrics, validation_targets, validation_predictions = (
            run_epoch(
                validation_loader,
                model,
                criterion,
                device,
                energy_mean,
                energy_std,
                description="Validation %d/%d" % (epoch, epochs),
                amp_dtype=amp_dtype,
                amp_enabled=use_amp,
                max_batches=max_validation_batches,
                objective_name=objective_name,
                smooth_l1_beta=smooth_l1_beta,
            )
        )
        scheduler.step()

        improved = validation_improved(
            validation_metrics,
            best_validation_metrics,
        )
        validation_rmse = float(validation_metrics["energy_rmse_mev"])
        if early_stopping_improved(
            validation_rmse,
            significant_reference_rmse,
            early_stopping_min_delta,
        ):
            significant_reference_rmse = validation_rmse
            patience_counter = 0
        else:
            patience_counter += 1
        stop_now = patience_counter >= early_stopping_patience
        early_stopping_state = {
            "patience_counter": patience_counter,
            "reference_rmse_mev": significant_reference_rmse,
            "stopped_early": stop_now,
        }
        record: Dict[str, Any] = {
            "epoch": epoch,
            "objective": objective_name,
            "train": train_metrics,
            "validation": validation_metrics,
            "baseline_comparison": {
                "constant_validation_rmse_mev": constant_validation[
                    "energy_rmse_mev"
                ],
                "geometry_validation_rmse_mev": geometry_validation[
                    "energy_rmse_mev"
                ],
                "beats_constant_rmse": bool(
                    validation_rmse
                    < constant_validation["energy_rmse_mev"]
                ),
                "beats_geometry_rmse": bool(
                    validation_rmse
                    < geometry_validation["energy_rmse_mev"]
                ),
            },
            "early_stopping": dict(early_stopping_state),
        }
        history.append(record)
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch,
            history,
            projection,
            data_selection,
            model_suffix,
            max_files,
            epochs,
            max_train_batches,
            max_validation_batches,
            use_amp,
            amp_precision,
            objective_name,
            smooth_l1_beta,
            event_shuffle_buffer_size,
            early_stopping_patience,
            early_stopping_min_delta,
            early_stopping_state,
            baselines,
            energy_mean,
            energy_std,
            normalizer_event_count,
        )
        payload["initial_validation_metrics"] = dict(
            initial_validation_metrics
        )
        torch.save(payload, last_checkpoint)

        if improved:
            best_validation_metrics = dict(validation_metrics)
            torch.save(payload, best_checkpoint)

        append_validation_log(
            validation_log,
            run_id,
            record,
            learning_rate,
            improved,
            baselines,
            patience_counter,
        )
        with history_json.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        save_history_plot(history, model_suffix, run_id)

        print(
            "[Epoch %d/%d] train loss=%.4f MAE=%.5f MeV "
            "RMSE=%.5f MeV amp_skip=%d | validation loss=%.4f "
            "MAE=%.5f MeV RMSE=%.5f MeV R2=%s | lr=%.2e%s"
            % (
                epoch,
                epochs,
                train_metrics["loss"],
                train_metrics["energy_mae_mev"],
                train_metrics["energy_rmse_mev"],
                train_metrics["amp_skipped_steps"],
                validation_metrics["loss"],
                validation_metrics["energy_mae_mev"],
                validation_metrics["energy_rmse_mev"],
                "NA"
                if validation_metrics["energy_r2"] is None
                else "%.4f" % validation_metrics["energy_r2"],
                learning_rate,
                "  *** new best" if improved else "",
            ),
            flush=True,
        )
        if stop_now:
            print(
                "Early stopping: validation RMSE did not improve by more "
                "than %.3g MeV for %d epochs."
                % (early_stopping_min_delta, early_stopping_patience),
                flush=True,
            )
            break

    best_payload = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )
    validate_checkpoint(best_payload, best_checkpoint)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    train_dataset.set_epoch(int(best_payload["epoch"]))
    frozen_train_metrics, _, _ = run_epoch(
        train_loader,
        model,
        criterion,
        device,
        energy_mean,
        energy_std,
        description="Frozen best train audit",
        amp_dtype=amp_dtype,
        amp_enabled=use_amp,
        max_batches=max_train_batches,
        objective_name=objective_name,
        smooth_l1_beta=smooth_l1_beta,
    )
    final_metrics, final_targets, final_predictions = run_epoch(
        validation_loader,
        model,
        criterion,
        device,
        energy_mean,
        energy_std,
        description="Final validation",
        amp_dtype=amp_dtype,
        amp_enabled=use_amp,
        max_batches=max_validation_batches,
        objective_name=objective_name,
        smooth_l1_beta=smooth_l1_beta,
    )
    acceptance = validation_acceptance(final_metrics, baselines)
    best_payload["frozen_train_metrics"] = dict(frozen_train_metrics)
    best_payload["final_validation_metrics"] = dict(final_metrics)
    best_payload["validation_acceptance"] = dict(acceptance)
    torch.save(best_payload, best_checkpoint)
    save_regression_plot(
        final_targets,
        final_predictions,
        model_suffix,
        run_id,
    )

    print("\nDone.")
    print("Best checkpoint : %s" % best_checkpoint)
    print("Last checkpoint : %s" % last_checkpoint)
    print("Validation log  : %s" % validation_log)
    print(
        "Frozen train    : %s"
        % json.dumps(frozen_train_metrics, ensure_ascii=False)
    )
    print("Final metrics   : %s" % json.dumps(final_metrics, ensure_ascii=False))
    print("Acceptance      : %s" % json.dumps(acceptance, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
