"""Small task-aware training and evaluation loops for MJD."""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainingConfig


Task = Literal["classification", "regression"]


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic))
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _device(name: str) -> torch.device:
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if selected == "auto":
        selected = "cpu"
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _amp(
    device: torch.device,
    use_amp: bool,
    precision: str,
) -> tuple[bool, torch.dtype]:
    if not use_amp or device.type != "cuda":
        return False, torch.float32
    selected = precision
    if selected == "auto":
        selected = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if selected == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA device does not support bfloat16 AMP")
        return True, torch.bfloat16
    if selected == "float16":
        return True, torch.float16
    raise ValueError("amp_precision must be 'auto', 'float16', or 'bfloat16'")


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    positive = int(target.sum())
    negative = int((~target).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    positive_rank_sum = float(ranks[target].sum())
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (
        positive * negative
    )


def _metrics(task: Task, target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if task == "classification":
        target = np.asarray(target).reshape(-1)
        prediction = np.asarray(prediction).reshape(-1)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(prediction, -40.0, 40.0)))
        binary = probabilities >= 0.5
        auc = _binary_auc(target, prediction)
        accuracy = float(np.mean(binary == target))
        return {
            "auc": auc,
            "macro_auc": auc,
            "accuracy": accuracy,
            "clean_accuracy": accuracy,
            "clean_fraction": float(np.mean(target == 1)),
            "events": int(target.size),
        }
    residual = prediction - target
    return {
        "mae_kev": float(np.mean(np.abs(residual))),
        "rmse_kev": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_kev": float(np.mean(residual)),
        "events": int(target.size),
    }


def _epoch(
    model: nn.Module,
    loader: Any,
    *,
    task: Task,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: Any | None = None,
    metadata_out: dict[str, list[np.ndarray]] | None = None,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in loader:
        inputs = batch["inputs"].to(device=device, non_blocking=device.type == "cuda")
        target = batch["clean"] if task == "classification" else batch["energy"]
        target = target.to(device=device, dtype=torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast:
            prediction = model(inputs)
            if task == "classification":
                prediction = prediction.reshape(-1)
                target = target.reshape(-1)
                if prediction.shape != target.shape:
                    raise ValueError("classifier must return [B] or [B, 1] raw logits")
                loss = F.binary_cross_entropy_with_logits(prediction, target)
            else:
                prediction = prediction.reshape(-1)
                target = target.reshape(-1)
                if prediction.shape != target.shape:
                    raise ValueError("regressor must return [B] or [B, 1]")
                loss = F.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite training loss")
        if training:
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        if metadata_out is not None:
            for key in metadata_out:
                value = batch[key]
                array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                if array.ndim != 1 or len(array) != target.numel():
                    raise ValueError(f"{key} is not aligned with the inference batch")
                metadata_out[key].append(array.copy())
        targets.append(target.detach().cpu().numpy())
        predictions.append(prediction.detach().float().cpu().numpy())
    if not losses:
        raise RuntimeError("data loader produced no batches")
    target_array = np.concatenate(targets, axis=0)
    prediction_array = np.concatenate(predictions, axis=0)
    return (
        float(np.mean(losses)),
        _metrics(task, target_array, prediction_array),
        target_array,
        prediction_array,
    )


def train_model(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    *,
    task: Task,
    config: TrainingConfig | None = None,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Train one task and restore the best validation checkpoint."""

    selected = config or TrainingConfig()
    device = _device(selected.device)
    amp_enabled, amp_dtype = _amp(
        device, selected.use_amp, selected.amp_precision
    )
    set_seed(selected.seed, selected.deterministic)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=selected.learning_rate,
        weight_decay=selected.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=selected.epochs
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    history: list[dict[str, Any]] = []
    best = -math.inf if task == "classification" else math.inf
    stale_epochs = 0
    scaler: Any | None = None
    if amp_enabled and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(1, selected.epochs + 1):
        train_loss, train_metrics, _, _ = _epoch(
            model,
            train_loader,
            task=task,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=selected.gradient_clip_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        validation_loss, validation_metrics, _, _ = _epoch(
            model,
            validation_loader,
            task=task,
            device=device,
            optimizer=None,
            gradient_clip_norm=selected.gradient_clip_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        if task == "classification":
            score = float(validation_metrics["macro_auc"])
            # A tiny smoke validation subset can contain only one class for
            # every PSD label. In that case AUC is undefined, so validation
            # loss provides a deterministic checkpoint fallback.
            if not math.isfinite(score):
                score = -validation_loss
        else:
            score = float(validation_metrics["rmse_kev"])
        improved = (
            score > best + selected.early_stopping_min_delta
            if task == "classification"
            else score < best - selected.early_stopping_min_delta
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        history.append(row)
        metric_name = "macro_auc" if task == "classification" else "rmse_kev"
        print(
            f"Epoch {epoch:03d}/{selected.epochs:03d} | "
            f"train loss {train_loss:.6g} | "
            f"validation loss {validation_loss:.6g} | "
            f"validation {metric_name} {validation_metrics[metric_name]:.6g}",
            flush=True,
        )
        if improved:
            best = score
            stale_epochs = 0
            torch.save(
                {
                    "task": task,
                    "epoch": epoch,
                    "score": score,
                    "model_state_dict": model.state_dict(),
                    "training_config": selected.to_dict(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        (output / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=True), encoding="utf-8"
        )
        if stale_epochs >= selected.early_stopping_patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return history


def evaluate_model(
    model: nn.Module,
    loader: Any,
    *,
    task: Task,
    device: str = "auto",
    output_dir: str | Path,
    use_amp: bool = False,
    amp_precision: str = "auto",
) -> dict[str, Any]:
    """Evaluate a restored model and save auditable metrics and predictions."""

    selected_device = _device(device)
    amp_enabled, amp_dtype = _amp(selected_device, use_amp, amp_precision)
    model.to(selected_device)
    metadata = {key: [] for key in ('event_id', 'source_file', 'row', 'energy_keV')}
    loss, metrics, targets, predictions = _epoch(
        model,
        loader,
        task=task,
        device=selected_device,
        optimizer=None,
        gradient_clip_norm=1.0,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        metadata_out=metadata,
    )
    columns = {key: np.concatenate(values) for key, values in metadata.items()}
    if np.unique(columns["event_id"]).size != len(targets):
        raise ValueError("Exported physical event IDs must be unique")
    if columns["energy_keV"].dtype != np.float64:
        raise TypeError("Physical evaluation energy must remain float64")
    metrics["loss"] = loss
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    np.savez_compressed(output / "predictions.npz", target=targets, prediction=predictions,
                        label=targets, score=predictions, **columns)
    return metrics


__all__ = ["evaluate_model", "set_seed", "train_model"]
