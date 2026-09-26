"""Unified CUDA training runner for every alternative NEXT architecture."""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from next_cnn.data import SourceFile

from .checkpoint import build_v3_checkpoint, save_v3_checkpoint
from .config import load_training_config
from .data import RepresentationConfig, build_training_loaders
from .metrics import (
    classification_metrics,
    save_history_plot,
    write_history_csv,
    write_history_json,
)
from .registry import build_model, get_model_spec


def require_cuda_device() -> torch.device:
    """Return CUDA device zero or fail without a CPU fallback."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Alternative architecture training is GPU-only."
        )
    device = torch.device("cuda:0")
    try:
        probe = torch.zeros(1, device=device)
        del probe
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            "CUDA device 0 is visible but unusable; refusing a CPU fallback"
        ) from exc
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(bool(deterministic))
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_amp(
    use_amp: bool,
    requested_precision: str,
) -> Tuple[torch.dtype, str]:
    """Resolve auto/BF16/FP16 after a real CUDA device has been selected."""

    if not use_amp:
        return torch.float32, "disabled"
    requested = str(requested_precision).strip().lower()
    if requested == "auto":
        requested = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if requested == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bfloat16 AMP was requested but CUDA device 0 does not support it"
            )
        return torch.bfloat16, requested
    if requested == "float16":
        return torch.float16, requested
    raise ValueError("AMP precision must be auto, bfloat16, or float16")


def _make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Move tensor values while preserving string metadata lists."""

    return {
        key: (
            value.to(device=device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _run_epoch(
    *,
    loader: Any,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    description: str,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    gradient_clip_norm: float = 1.0,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    total_events = 0
    coverage_sum = 0.0
    coverage_events = 0

    # A tmux queue pipes stdout through ``tee``. Disable the per-batch carriage
    # return stream there so the persistent queue log stays line-oriented;
    # direct interactive runs still receive the full progress bar.
    progress = tqdm(
        loader,
        desc=description,
        dynamic_ncols=True,
        file=sys.stdout,
        disable=not sys.stdout.isatty(),
    )
    for batch_index, host_batch in enumerate(progress):
        batch = move_batch_to_device(host_batch, device)
        labels = batch["label"]
        if labels.ndim != 1:
            raise ValueError("batch label must have shape (batch,)")
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits = model(batch)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("model(batch) must return a torch.Tensor")
                if logits.shape != labels.shape:
                    raise ValueError(
                        "model logits must have shape %s, received %s"
                        % (tuple(labels.shape), tuple(logits.shape))
                    )
                loss = criterion(logits, labels)
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise FloatingPointError(
                    "%s produced non-finite loss at batch %d"
                    % (description, batch_index)
                )
            if not bool(torch.isfinite(logits).all().detach().cpu()):
                raise FloatingPointError(
                    "%s produced non-finite logits at batch %d"
                    % (description, batch_index)
                )
            if training:
                if scaler is None:
                    raise RuntimeError("training requires a GradScaler")
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(gradient_clip_norm)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(gradient_clip_norm),
                        error_if_nonfinite=True,
                    )
                    optimizer.step()

        size = int(labels.shape[0])
        total_events += size
        total_loss += float(loss.detach().cpu()) * size
        all_labels.append(labels.detach().cpu().numpy().astype(np.int64))
        all_logits.append(logits.detach().float().cpu().numpy())
        coverage = host_batch.get("representation_coverage")
        if isinstance(coverage, torch.Tensor):
            coverage_sum += float(coverage.sum().item())
            coverage_events += int(coverage.numel())
        progress.set_postfix(
            loss="%.4f" % (total_loss / total_events),
            events=total_events,
        )

    if total_events == 0:
        raise RuntimeError("%s data loader produced no events" % description)
    result = classification_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_logits),
        total_loss,
    )
    result["mean_representation_coverage"] = (
        None if coverage_events == 0 else coverage_sum / coverage_events
    )
    return result


def _source_records(files: Sequence[SourceFile]) -> List[Dict[str, Any]]:
    return [
        {
            "relative_path": source.relative_path,
            "group_id": source.group_id,
            "label": int(source.label),
            "category": source.category,
            "split": source.split,
        }
        for source in files
    ]


def _artifact_paths(
    architecture_id: str,
    output_config: Mapping[str, Any],
) -> Dict[str, Path]:
    checkpoint_dir = Path(output_config["checkpoint_dir"])
    log_dir = Path(output_config["log_dir"])
    plot_dir = Path(output_config["plot_dir"])
    if bool(output_config.get("campaign_layout", False)):
        return {
            "best": checkpoint_dir / "best.pt",
            "last": checkpoint_dir / "last.pt",
            "csv": log_dir / "epochs.csv",
            "json": log_dir / "history.json",
            "plot": plot_dir / "history.png",
            "summary": log_dir / "run_summary.json",
        }
    basename = "NEXTALT_%s_classification" % architecture_id
    return {
        "best": checkpoint_dir / (basename + "_best.pt"),
        "last": checkpoint_dir / (basename + "_last.pt"),
        "csv": log_dir / (basename + "_epochs.csv"),
        "json": log_dir / (basename + "_history.json"),
        "plot": plot_dir / (basename + "_history.png"),
        "summary": log_dir / (basename + "_run_summary.json"),
    }


def _claim_artifact_paths(
    paths: Mapping[str, Path],
    allow_overwrite: bool,
) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not allow_overwrite:
        raise FileExistsError(
            "refusing to overwrite existing training artifacts:\n  %s"
            % "\n  ".join(str(path) for path in existing)
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def main_for_architecture(
    architecture_id: str,
    config_path: str | Path,
) -> int:
    """Train one registered architecture from its YAML configuration.

    Architecture entry points pass a fixed identifier and their adjacent YAML
    path.  There is intentionally no smoke mode and no CPU fallback: invoking
    an entry point starts the configured formal CUDA training run.
    """

    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    config = load_training_config(config_path, architecture_id=architecture_id)
    spec = get_model_spec(architecture_id)
    output = config["output"]
    artifacts = _artifact_paths(architecture_id, output)
    allow_overwrite = bool(output["allow_overwrite"])
    _claim_artifact_paths(artifacts, allow_overwrite)

    device = require_cuda_device()
    training = config["training"]
    seed_everything(int(training["seed"]), bool(training["deterministic"]))
    amp_dtype, amp_precision = resolve_amp(
        bool(training["use_amp"]), str(training["amp_precision"])
    )
    amp_enabled = amp_precision != "disabled"
    scaler = _make_grad_scaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )

    model = build_model(architecture_id, config["model"]).to(device)
    actual_model_config: Mapping[str, Any] = config["model"]
    if hasattr(model, "config_dict"):
        candidate = model.config_dict()
        if not isinstance(candidate, Mapping):
            raise TypeError("model.config_dict() must return a mapping")
        actual_model_config = dict(candidate)
    train_loader, validation_loader, train_dataset, train_files, validation_files = (
        build_training_loaders(config, spec.input_kind, device_type=device.type)
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["epochs"]),
    )

    representation = RepresentationConfig.from_mapping(
        config["representation"]
    ).to_dict()
    split_config = {
        "seed": int(config["data"]["split_seed"]),
        "fractions": list(config["data"]["split_fractions"]),
    }
    data_selection = {
        # This training-only campaign deliberately records only the selected
        # train/validation sources.  It does not stat or fingerprint files in
        # the reserved split.
        "inventory": {
            "scope": "selected-train-validation-files-only",
            "training_file_count": len(train_files),
            "validation_file_count": len(validation_files),
        },
        "training_groups": sorted(source.group_id for source in train_files),
        "validation_groups": sorted(
            source.group_id for source in validation_files
        ),
        "training_files": _source_records(train_files),
        "validation_files": _source_records(validation_files),
    }
    checkpoint_training_config = {
        "config_path": config["config_path"],
        "data": copy.deepcopy(config["data"]),
        "training": copy.deepcopy(config["training"]),
        "amp_precision_resolved": amp_precision,
        "device": "cuda:0",
    }

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "Training %s (%s, %s parameters) on cuda:0"
        % (architecture_id, spec.model_name, format(parameter_count, ",")),
        flush=True,
    )
    print("Best checkpoint: %s" % artifacts["best"], flush=True)

    history: List[Dict[str, Any]] = []
    best_auc = -float("inf")
    epochs_without_improvement = 0
    patience = int(training["early_stopping_patience"])
    min_delta = float(training["early_stopping_min_delta"])

    for epoch in range(1, int(training["epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = _run_epoch(
            loader=train_loader,
            model=model,
            criterion=criterion,
            device=device,
            description="train %d" % epoch,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        validation_metrics = _run_epoch(
            loader=validation_loader,
            model=model,
            criterion=criterion,
            device=device,
            description="validation %d" % epoch,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        validation_auc = validation_metrics["auc"]
        if validation_auc is None:
            raise RuntimeError(
                "validation split contains only one class; AUC cannot select a checkpoint"
            )
        improved = float(validation_auc) > best_auc + min_delta
        if improved:
            best_auc = float(validation_auc)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
            "improved": improved,
        }
        history.append(record)

        payload = build_v3_checkpoint(
            architecture_id=architecture_id,
            model_name=spec.model_name,
            input_kind=spec.input_kind,
            model_config=actual_model_config,
            model=model,
            representation_config=representation,
            split_config=split_config,
            data_selection=data_selection,
            training_config=checkpoint_training_config,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            validation_metrics=validation_metrics,
        )
        save_v3_checkpoint(payload, artifacts["last"], overwrite=True)
        if improved:
            save_v3_checkpoint(payload, artifacts["best"], overwrite=True)
        write_history_csv(history, artifacts["csv"])
        write_history_json(history, artifacts["json"])
        save_history_plot(history, artifacts["plot"], architecture_id)
        print(
            "epoch=%d train_auc=%s validation_auc=%.6f best_auc=%.6f"
            % (
                epoch,
                (
                    "NA"
                    if train_metrics["auc"] is None
                    else "%.6f" % float(train_metrics["auc"])
                ),
                float(validation_auc),
                best_auc,
            ),
            flush=True,
        )
        if epochs_without_improvement >= patience:
            print(
                "Early stopping after %d epochs without validation AUC improvement."
                % epochs_without_improvement,
                flush=True,
            )
            break

    best_record = max(
        history,
        key=lambda item: float(item["validation"]["auc"]),
    )
    completed_at = datetime.now(timezone.utc)
    gpu_name = torch.cuda.get_device_name(device)
    summary = {
        "status": "DONE",
        "architecture_id": architecture_id,
        "model_name": spec.model_name,
        "backend": "pytorch",
        "parameter_count": int(parameter_count),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": float(time.monotonic() - started_clock),
        "epochs_completed": int(len(history)),
        "configured_epochs": int(training["epochs"]),
        "best_epoch": int(best_record["epoch"]),
        "best_validation_auc": float(best_record["validation"]["auc"]),
        "best_validation_loss": float(best_record["validation"]["loss"]),
        "early_stopped": bool(len(history) < int(training["epochs"])),
        "attempt": int(os.environ.get("NEXT_CAMPAIGN_ATTEMPT", "1")),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": "cuda:0",
            "gpu": gpu_name,
            "amp_precision": amp_precision,
        },
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    temporary_summary = artifacts["summary"].with_suffix(".json.tmp")
    with temporary_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_summary, artifacts["summary"])
    print("Training complete. Best validation AUC: %.6f" % best_auc, flush=True)
    return 0


__all__ = [
    "main_for_architecture",
    "move_batch_to_device",
    "require_cuda_device",
    "resolve_amp",
    "seed_everything",
]
