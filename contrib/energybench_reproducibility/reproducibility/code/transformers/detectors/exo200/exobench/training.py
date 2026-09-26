"""MJD-equivalent training and evaluation for EXO-200 classification."""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import BACKGROUND_LABEL, CLASS_NAMES, SIGNAL_LABEL, TASK, TrainingConfig


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


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(prediction, -40.0, 40.0)))
    binary = probabilities >= 0.5
    auc = _binary_auc(target, prediction)
    return {
        "auc": auc,
        "macro_auc": auc,
        "positive_class": CLASS_NAMES[BACKGROUND_LABEL],
        "class_names": CLASS_NAMES,
        "accuracy": float(np.mean(binary == target)),
        "signal_count": int(np.sum(target == SIGNAL_LABEL)),
        "background_count": int(np.sum(target == BACKGROUND_LABEL)),
        "signal_fraction": float(np.mean(target == SIGNAL_LABEL)),
        "background_fraction": float(np.mean(target == BACKGROUND_LABEL)),
        "events": int(target.size),
    }


def _epoch(
    model: nn.Module,
    loader: Any,
    *,
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
        inputs = batch["inputs"].to(
            device=device, non_blocking=device.type == "cuda"
        )
        target = batch["label"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        ).reshape(-1)
        if not bool(torch.isfinite(target).all().item()) or not bool(
            ((target == SIGNAL_LABEL) | (target == BACKGROUND_LABEL)).all().item()
        ):
            raise ValueError("classification labels must be finite 0/1 values")
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast:
            prediction = model(inputs).reshape(-1)
            if prediction.shape != target.shape:
                raise ValueError("classifier must return [B] or [B, 1] raw logits")
            if not bool(torch.isfinite(prediction).all().item()):
                raise FloatingPointError("classifier produced non-finite logits")
            loss = F.binary_cross_entropy_with_logits(prediction, target)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite classification loss")
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
        _metrics(target_array, prediction_array),
        target_array,
        prediction_array,
    )


def train_model(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    *,
    config: TrainingConfig | None = None,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Train with MJD's optimizer/scheduler and restore the best checkpoint."""

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
    best = -math.inf
    stale_epochs = 0
    start_epoch = 1
    scaler: Any | None = None
    if amp_enabled and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    if resume_from is not None:
        resume_path = Path(resume_from)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_task = checkpoint.get("task")
        if checkpoint_task != TASK:
            raise ValueError(
                f"cannot resume classification from a {checkpoint_task!r} checkpoint"
            )
        resume_epoch = int(checkpoint["epoch"])
        if resume_epoch >= selected.epochs:
            raise ValueError(
                f"checkpoint epoch {resume_epoch} has already reached the configured "
                f"{selected.epochs} epochs"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        best = float(checkpoint["score"])
        start_epoch = resume_epoch + 1

        history_path = output / "history.json"
        if history_path.exists():
            loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
            history = [
                row for row in loaded_history if int(row.get("epoch", 0)) <= resume_epoch
            ]

        optimizer_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if optimizer_state is not None and scheduler_state is not None:
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            resume_mode = "full optimizer/scheduler state"
        else:
            scheduler.step(resume_epoch)
            resume_mode = "model weights and scheduled learning rate (fresh optimizer)"
        print(
            f"Resuming from {resume_path} after epoch {resume_epoch:03d} "
            f"using {resume_mode}",
            flush=True,
        )

    for epoch in range(start_epoch, selected.epochs + 1):
        train_loss, train_metrics, _, _ = _epoch(
            model,
            train_loader,
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
            device=device,
            optimizer=None,
            gradient_clip_norm=selected.gradient_clip_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        score = float(validation_metrics["macro_auc"])
        if not math.isfinite(score):
            score = -validation_loss
        improved = score > best + selected.early_stopping_min_delta
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{selected.epochs:03d} | "
            f"train loss {train_loss:.6g} | "
            f"validation loss {validation_loss:.6g} | "
            f"validation macro_auc {validation_metrics['macro_auc']:.6g}",
            flush=True,
        )
        if improved:
            best = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if improved:
            torch.save(
                {
                    "task": TASK,
                    "epoch": epoch,
                    "score": score,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": (
                        scaler.state_dict() if scaler is not None else None
                    ),
                    "training_config": selected.to_dict(),
                },
                checkpoint_path,
            )
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
    device: str = "auto",
    output_dir: str | Path,
    use_amp: bool = False,
    amp_precision: str = "auto",
) -> dict[str, Any]:
    """Evaluate the restored classifier and save MJD-format artifacts."""

    selected_device = _device(device)
    amp_enabled, amp_dtype = _amp(selected_device, use_amp, amp_precision)
    model.to(selected_device)
    metadata = {key: [] for key in ('event_id', 'source_file', 'row', 'energy_keV', 'run_number', 'event_number')}
    loss, metrics, targets, predictions = _epoch(
        model,
        loader,
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
    np.savez_compressed(
        output / "predictions.npz", target=targets, prediction=predictions,
        label=targets, score=predictions, **columns
    )
    return metrics


__all__ = ["evaluate_model", "set_seed", "train_model"]
