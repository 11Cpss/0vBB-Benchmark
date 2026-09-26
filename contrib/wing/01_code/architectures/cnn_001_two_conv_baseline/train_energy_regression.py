#!/usr/bin/env python3
"""
train_energy_regression.py — CNN-001 deposited-energy regression baseline

This program trains the same two-convolution, MaxPool-based topology used by
``train_classification.py``.  Its one scalar output represents standardized
summed voxel-deposited energy rather than a class logit.

Run
---
    cd /home/wenyu/summer
    source .venv/bin/activate
    python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py

Quick GPU smoke test
--------------------
    python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py --smoke

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
from next_cnn.model import SimpleNextEnergyRegressor as EnergyRegressor


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
    "_cnn_001_two_conv_baseline_energy_regression",
)
BASE_DATA = os.environ.get(
    "NEXT_DATA_ROOT", "/home/klz/Data/zeronu_benchmark/NEXT"
)

BATCH_SIZE = _env_int("NEXT_BATCH_SIZE", 16)
NUM_EPOCHS = _env_int("NEXT_NUM_EPOCHS", 50)
MAX_FILES_PER_CLASS = _env_int("NEXT_MAX_FILES_PER_CLASS", 100)
NUM_WORKERS = _env_int("NEXT_NUM_WORKERS", 0)

LEARNING_RATE = _env_float("NEXT_LEARNING_RATE", 1.0e-3)
WEIGHT_DECAY = _env_float("NEXT_WEIGHT_DECAY", 1.0e-4)
GRAD_CLIP = _env_float("NEXT_GRAD_CLIP", 1.0)
BASE_CHANNELS = _env_int("NEXT_BASE_CHANNELS", 8)
SMOOTH_L1_BETA = _env_float("NEXT_SMOOTH_L1_BETA", 1.0)

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
# Regression must retain absolute energy amplitude.  A scale of 40 makes one
# raw-energy view sum to roughly 100 for a 2.5 MeV event, similar to the
# classification program's normalized input magnitude.
NORMALIZE_EVENT_ENERGY = False
INPUT_SCALE = 40.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the CNN-001 NEXT summed-energy regressor on CUDA"
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
    return {
        "energy_mae_mev": mae,
        "energy_rmse_mev": rmse,
        "energy_bias_mev": bias,
        "energy_r2": r2,
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
        "loss": total_loss / total_events,
        **regression_metrics(targets_array, predictions_array),
        "events": total_events,
        "mean_projection_coverage": coverage_sum / total_events,
        "amp_skipped_steps": amp_skipped_steps,
    }
    return metrics, targets_array, predictions_array


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
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, validation_loss, marker="o", label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Standardized Smooth-L1 loss")
    axes[0].legend()
    axes[1].plot(epochs, train_mae, marker="o", label="train MAE")
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
        label="train RMSE",
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
    figure.suptitle("CNN-001 energy regression%s" % model_suffix)
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
    figure.suptitle("CNN-001 validation energy regression")
    figure.tight_layout()
    figure.savefig(paths.regression_plot_path(model_suffix, run_id), dpi=180)
    plt.close(figure)


def append_validation_log(
    path: Path,
    run_id: str,
    record: Mapping[str, Any],
    learning_rate: float,
    improved: bool,
) -> None:
    row = {
        "run_id": run_id,
        "epoch": int(record["epoch"]),
        "train_loss": float(record["train"]["loss"]),
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
        "train_amp_skipped_steps": int(
            record["train"]["amp_skipped_steps"]
        ),
        "validation_loss": float(record["validation"]["loss"]),
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
        "projection_coverage": float(
            record["validation"]["mean_projection_coverage"]
        ),
        "learning_rate": float(learning_rate),
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
    smooth_l1_beta: float,
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
        "model_name": "SimpleNextEnergyRegressor",
        "model_suffix": model_suffix,
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "energy_target_config": energy_target_config,
        "projection_config": projection.to_dict(),
        "split_config": {
            "seed": SPLIT_SEED,
            "fractions": list(SPLIT_FRACTIONS),
        },
        "epoch": int(epoch),
        "selection": {
            "metric": "validation_loss",
            "direction": "minimize",
        },
        "training_config": {
            "entrypoint": (
                "01_code/architectures/cnn_001_two_conv_baseline/"
                "train_energy_regression.py"
            ),
            "data": {
                "root": BASE_DATA,
                "max_files_per_class": max_files_per_class,
                "balance_training_classes": BALANCE_TRAINING_CLASSES,
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
                "smooth_l1_beta": float(smooth_l1_beta),
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
    if loaded.get("model_name") != "SimpleNextEnergyRegressor":
        raise ValueError(
            "checkpoint model is not CNN-001 regression: %s" % checkpoint_path
        )
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
    model_channels = BASE_CHANNELS
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
        model_channels = int(
            validation_checkpoint["model_config"]["base_channels"]
        )
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
    print("Model     : CNN-001 energy regression%s" % model_suffix)
    print(
        "Input     : (3, %d, %d), raw energy x %.1f"
        % (projection.grid_size, projection.grid_size, projection.input_scale)
    )
    print("Batch     : %d" % batch_size)
    print("Workers   : %d" % num_workers)
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
    print("Loss      : SmoothL1 beta=%.6g (standardized energy)" % smooth_l1_beta)

    model = EnergyRegressor(base_channels=model_channels).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    expected_parameter_count = (
        18 * model_channels * model_channels + 32 * model_channels + 1
    )
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            "CNN-001 parameter count changed: expected %d, found %d"
            % (expected_parameter_count, parameter_count)
        )
    print(
        "Parameters: %s total, %s trainable"
        % (
            format(parameter_count, ","),
            format(trainable_parameter_count, ","),
        )
    )

    criterion = nn.SmoothL1Loss(beta=smooth_l1_beta).to(device)
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
    best_validation_loss = float("inf")
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
            )
        )
        scheduler.step()

        record: Dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
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
            smooth_l1_beta,
            energy_mean,
            energy_std,
            normalizer_event_count,
        )
        torch.save(payload, last_checkpoint)

        validation_loss = float(validation_metrics["loss"])
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            torch.save(payload, best_checkpoint)

        append_validation_log(
            validation_log,
            run_id,
            record,
            learning_rate,
            improved,
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

    best_payload = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )
    validate_checkpoint(best_payload, best_checkpoint)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
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
    )
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
    print("Final metrics   : %s" % json.dumps(final_metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
