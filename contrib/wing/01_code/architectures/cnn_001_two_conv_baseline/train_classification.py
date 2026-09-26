#!/usr/bin/env python3
"""
train_classification.py — CNN-001 two-convolution classification baseline

A deliberately small baseline for 0nu-beta-beta versus Bi-214 classification.
Each event is converted to three fixed detector-coordinate energy projections
(XY, XZ and YZ), then classified by two Conv2d layers.

This file follows the layout of the training programs in ``wing_directory``:
configuration, validation, and training remain visible in one directly
runnable file.  The validated dataset and model are imported from
``src/next_cnn`` so training and scoring use exactly the same implementation.

Run
---
    cd /home/wenyu/summer
    source .venv/bin/activate
    python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py

Quick GPU smoke test
--------------------
    python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py --smoke

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
)
from next_cnn.model import SimpleNextCNN as NEXTCNN


# =============================================================================
# Configuration — edit here, like the programs in wing_directory
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
    "_cnn_001_two_conv_baseline_classification",
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

SEED = _env_int("NEXT_SEED", 42)
DETERMINISTIC = _env_bool("NEXT_DETERMINISTIC", True)
USE_AMP = _env_bool("NEXT_USE_AMP", True)
AMP_PRECISION = os.environ.get("NEXT_AMP_PRECISION", "auto").strip().lower()
MAX_TRAIN_BATCHES = _env_int("NEXT_MAX_TRAIN_BATCHES", 0)
MAX_VALIDATION_BATCHES = _env_int("NEXT_MAX_VALIDATION_BATCHES", 0)

SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
BALANCE_TRAINING_CLASSES = True

GRID_SIZE = 128
BIN_SIZE = 30.0
GRID_ORIGIN = (-1920.0, -1920.0, -120.0)
NORMALIZE_EVENT_ENERGY = True
INPUT_SCALE = 100.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the CNN-001 NEXT binary classifier on CUDA"
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
    """Return best/last checkpoint paths for one model suffix."""

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
# Dataset / DataLoader
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


# =============================================================================
# Metrics / validation / plots
# =============================================================================


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    n_positive = int(np.sum(positive))
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=float)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + 1 + stop)
        start = stop
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    statistic = float(np.sum(original_ranks[positive]))
    return (
        statistic - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def run_epoch(
    loader: Any,
    model: NEXTCNN,
    criterion: Any,
    device: torch.device,
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
    total_correct = 0
    total_events = 0
    coverage_sum = 0.0
    amp_skipped_steps = 0
    consecutive_amp_skips = 0
    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    progress = tqdm(loader, desc=description, dynamic_ncols=True, file=sys.stdout)
    for batch_index, batch in enumerate(progress):
        if max_batches and batch_index >= max_batches:
            break
        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        labels = batch["label"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
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
                # GradScaler recorded inf/NaN during unscale_().  Its step()
                # intentionally skips the optimizer update in this case, and
                # update() lowers the scale for the next batch.  Raising here
                # would defeat AMP's normal dynamic-loss-scaling recovery.
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

        size = int(labels.shape[0])
        total_events += size
        total_loss += float(loss.detach().cpu()) * size
        total_correct += int(
            torch.sum((logits >= 0) == (labels >= 0.5)).detach().cpu()
        )
        labels_array = labels.detach().cpu().numpy().astype(np.int64)
        scores_array = logits.detach().float().cpu().numpy()
        all_labels.append(labels_array)
        all_scores.append(scores_array)
        coverage_sum += float(
            torch.sum(batch["projection_coverage"]).detach().cpu()
        )
        progress.set_postfix(
            loss="%.4f" % (total_loss / total_events),
            events=total_events,
            amp_skip=amp_skipped_steps,
        )

    if total_events == 0:
        raise RuntimeError("data loader produced no events")
    labels_array = np.concatenate(all_labels)
    scores_array = np.concatenate(all_scores)
    metrics = {
        "loss": total_loss / total_events,
        "accuracy": total_correct / total_events,
        "auc": binary_auc(labels_array, scores_array),
        "events": total_events,
        "mean_projection_coverage": coverage_sum / total_events,
        "amp_skipped_steps": amp_skipped_steps,
    }
    return metrics, labels_array, scores_array


def save_history_plot(
    history: Sequence[Mapping[str, Any]],
    model_suffix: str,
    run_id: str,
) -> None:
    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record["train"]["loss"]) for record in history]
    validation_loss = [float(record["validation"]["loss"]) for record in history]
    train_auc = [
        np.nan if record["train"]["auc"] is None else float(record["train"]["auc"])
        for record in history
    ]
    validation_auc = [
        np.nan
        if record["validation"]["auc"] is None
        else float(record["validation"]["auc"])
        for record in history
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, validation_loss, marker="o", label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[1].plot(epochs, train_auc, marker="o", label="train")
    axes[1].plot(epochs, validation_auc, marker="o", label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Inclusive AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    figure.suptitle("NEXT CNN%s" % model_suffix)
    figure.tight_layout()
    figure.savefig(paths.history_plot_path(model_suffix, run_id), dpi=180)
    plt.close(figure)


def save_score_plot(
    labels: np.ndarray,
    scores: np.ndarray,
    model_suffix: str,
    run_id: str,
) -> None:
    signal = scores[labels == 1]
    background = scores[labels == 0]
    if len(signal) == 0 or len(background) == 0:
        return
    low = float(min(np.min(signal), np.min(background)))
    high = float(max(np.max(signal), np.max(background)))
    if not np.isfinite(low + high) or high <= low:
        return
    bins = np.linspace(low, high, 60)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(signal, bins=bins, density=True, alpha=0.45, label="0nubb")
    axis.hist(
        background,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        label="Bi214",
    )
    auc = binary_auc(labels, scores)
    axis.set_xlabel("NEXT CNN logit")
    axis.set_ylabel("Density")
    axis.set_title("Validation score, AUC=%s" % (
        "NA" if auc is None else "%.4f" % auc
    ))
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths.score_plot_path(model_suffix, run_id), dpi=180)
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
        "train_accuracy": float(record["train"]["accuracy"]),
        "train_auc": record["train"]["auc"],
        "train_amp_skipped_steps": int(record["train"]["amp_skipped_steps"]),
        "validation_loss": float(record["validation"]["loss"]),
        "validation_accuracy": float(record["validation"]["accuracy"]),
        "validation_auc": record["validation"]["auc"],
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
    model: NEXTCNN,
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
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "task": "binary_classification",
        "model_name": "SimpleNextCNN",
        "model_suffix": model_suffix,
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "projection_config": projection.to_dict(),
        "split_config": {
            "seed": SPLIT_SEED,
            "fractions": list(SPLIT_FRACTIONS),
        },
        "epoch": int(epoch),
        "training_config": {
            "entrypoint": (
                "01_code/architectures/cnn_001_two_conv_baseline/"
                "train_classification.py"
            ),
            "data": {
                "root": BASE_DATA,
                "max_files_per_class": max_files_per_class,
                "balance_training_classes": BALANCE_TRAINING_CLASSES,
            },
            "projection": projection.to_dict(),
            "model": model.config_dict(),
            "training": {
                "batch_size": BATCH_SIZE,
                "epochs": actual_epochs,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": GRAD_CLIP,
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


def validate_classification_checkpoint(
    loaded: Any,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Reject a checkpoint from another task or architecture."""

    if not isinstance(loaded, dict) or loaded.get("format_version") != 2:
        raise ValueError("unsupported checkpoint format: %s" % checkpoint_path)
    if loaded.get("model_name", "SimpleNextCNN") != "SimpleNextCNN":
        raise ValueError(
            "checkpoint model is not CNN-001 classification: %s"
            % checkpoint_path
        )
    task = loaded.get("task")
    if task not in {None, "binary_classification"}:
        raise ValueError(
            "checkpoint task is not binary classification: %s"
            % checkpoint_path
        )
    model_config = loaded.get("model_config")
    if not isinstance(model_config, Mapping) or "base_channels" not in model_config:
        raise ValueError("checkpoint model configuration is incomplete")
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
    if args.full_validation:
        if not best_checkpoint.is_file():
            raise FileNotFoundError("best checkpoint not found: %s" % best_checkpoint)
        loaded = torch.load(
            best_checkpoint,
            map_location=device,
            weights_only=False,
        )
        validation_checkpoint = validate_classification_checkpoint(
            loaded,
            best_checkpoint,
        )
        projection = ProjectionConfig.from_dict(loaded["projection_config"])
        model_channels = int(loaded["model_config"]["base_channels"])
        split_seed = int(loaded["split_config"]["seed"])
        split_fractions = list(loaded["split_config"]["fractions"])
        saved_data_config = loaded.get("training_config", {}).get("data", {})
        data_root = str(saved_data_config.get("root", data_root))
        max_files = saved_data_config.get("max_files_per_class", max_files)
        if max_files is not None and int(max_files) <= 0:
            max_files = None
        saved_training_config = loaded.get("training_config", {}).get(
            "training", {}
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
        # Format-v2 checkpoints created before amp_precision was recorded used
        # fixed FP16 autocast, so missing precision has an unambiguous meaning.
        requested_amp_precision = str(
            saved_training_config.get(
                "amp_precision",
                "float16" if use_amp else "disabled",
            )
        )
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
        expected_inventory = validation_checkpoint["data_selection"].get("inventory")
        actual_inventory = dataset_inventory(data_root)
        if actual_inventory != expected_inventory:
            raise ValueError(
                "NEXT data inventory differs from the checkpoint; "
                "refusing validation"
            )
        expected_groups = set(
            validation_checkpoint["data_selection"].get("validation_groups", [])
        )
        actual_groups = {source.group_id for source in validation_files}
        if actual_groups != expected_groups:
            raise ValueError(
                "validation file selection differs from the checkpoint; "
                "refusing validation"
            )

    print("Using GPU : %s" % torch.cuda.get_device_name(0))
    print("PyTorch   : %s (CUDA %s)" % (torch.__version__, torch.version.cuda))
    print("Data      : %s" % data_root)
    print("Model     : CNN-001 classification%s" % model_suffix)
    print(
        "Input     : (3, %d, %d)" % (projection.grid_size, projection.grid_size)
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

    model = NEXTCNN(base_channels=model_channels).to(device)
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

    criterion = nn.BCEWithLogitsLoss().to(device)
    if args.full_validation:
        if validation_checkpoint is None:
            raise RuntimeError("validation checkpoint was not loaded")
        model.load_state_dict(
            validation_checkpoint["model_state_dict"],
            strict=True,
        )
        metrics, labels, scores = run_epoch(
            validation_loader,
            model,
            criterion,
            device,
            description="Validation",
            amp_dtype=amp_dtype,
            amp_enabled=use_amp,
            max_batches=max_validation_batches,
        )
        save_score_plot(labels, scores, model_suffix, run_id)
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
    # BF16 has FP32-like exponent range and does not need loss scaling.  FP16
    # retains GradScaler so isolated overflows skip one update and lower scale.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )

    inventory = dataset_inventory(data_root)
    data_selection = {
        "inventory": inventory,
        "training_groups": sorted(source.group_id for source in train_files),
        "validation_groups": sorted(source.group_id for source in validation_files),
    }

    history: List[Dict[str, Any]] = []
    best_auc = -float("inf")
    print("Training starts. Validation runs after every epoch.\n")

    for epoch in range(1, epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_dataset.set_epoch(epoch)
        train_metrics, _, _ = run_epoch(
            train_loader,
            model,
            criterion,
            device,
            description="Epoch %d/%d" % (epoch, epochs),
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            amp_enabled=use_amp,
            max_batches=max_train_batches,
        )
        validation_metrics, validation_labels, validation_scores = run_epoch(
            validation_loader,
            model,
            criterion,
            device,
            description="Validation %d/%d" % (epoch, epochs),
            amp_dtype=amp_dtype,
            amp_enabled=use_amp,
            max_batches=max_validation_batches,
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
        )
        torch.save(payload, last_checkpoint)

        validation_auc = validation_metrics["auc"]
        selection_score = (
            -float(validation_metrics["loss"])
            if validation_auc is None
            else float(validation_auc)
        )
        improved = selection_score > best_auc
        if improved:
            best_auc = selection_score
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
            "[Epoch %d/%d] train loss=%.4f auc=%s amp_skip=%d | "
            "validation loss=%.4f auc=%s | lr=%.2e%s"
            % (
                epoch,
                epochs,
                train_metrics["loss"],
                "NA" if train_metrics["auc"] is None else "%.4f" % train_metrics["auc"],
                train_metrics["amp_skipped_steps"],
                validation_metrics["loss"],
                "NA"
                if validation_metrics["auc"] is None
                else "%.4f" % validation_metrics["auc"],
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
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    final_metrics, final_labels, final_scores = run_epoch(
        validation_loader,
        model,
        criterion,
        device,
        description="Final validation",
        amp_dtype=amp_dtype,
        amp_enabled=use_amp,
        max_batches=max_validation_batches,
    )
    save_score_plot(final_labels, final_scores, model_suffix, run_id)

    print("\nDone.")
    print("Best checkpoint : %s" % best_checkpoint)
    print("Last checkpoint : %s" % last_checkpoint)
    print("Validation log  : %s" % validation_log)
    print("Final metrics   : %s" % json.dumps(final_metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
