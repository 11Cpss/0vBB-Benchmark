"""A small, standard PyTorch training loop for Simple EnergyBench."""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.nn import functional as F

from .config import TrainingConfig


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch, and every available CUDA device."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic))
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _resolve_device(requested: Any) -> torch.device:
    name = str(requested).strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or a CUDA device")
    return device


def _resolve_amp(
    device: torch.device,
    use_amp: bool,
    requested_precision: str,
) -> tuple[bool, torch.dtype, str]:
    if not use_amp or device.type != "cuda":
        return False, torch.float32, "disabled"
    precision = str(requested_precision).strip().lower()
    precision = {"bf16": "bfloat16", "fp16": "float16"}.get(
        precision, precision
    )
    if precision == "auto":
        precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if precision == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA device does not support bfloat16 AMP")
        return True, torch.bfloat16, precision
    if precision == "float16":
        return True, torch.float16, precision
    raise ValueError("amp_precision must be auto, bfloat16, or float16")


def _weighted_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return binary AUC with sample weights and half credit for ties."""

    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    weights = weights[order]
    positive_total = float(weights[labels == 1].sum())
    negative_total = float(weights[labels == 0].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        return float("nan")

    _, starts = np.unique(scores, return_index=True)
    stops = np.append(starts[1:], scores.size)
    negative_before = 0.0
    concordant = 0.0
    for start, stop in zip(starts, stops):
        group_labels = labels[start:stop]
        group_weights = weights[start:stop]
        positive_weight = float(group_weights[group_labels == 1].sum())
        negative_weight = float(group_weights[group_labels == 0].sum())
        concordant += positive_weight * (negative_before + 0.5 * negative_weight)
        negative_before += negative_weight
    return concordant / (positive_total * negative_total)


def _batch_vector(value: Any, name: str, device: torch.device) -> torch.Tensor:
    value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    value = value.to(device=device, non_blocking=device.type == "cuda")
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1:
        raise ValueError(f"batch {name!r} must have shape [B] or [B, 1]")
    return value


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensor leaves while preserving input containers."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _infer_batch_size(inputs: Any, target: torch.Tensor) -> int:
    """Infer the event count using the same contract as evaluation inference."""

    if isinstance(inputs, torch.Tensor):
        if inputs.ndim == 0:
            raise ValueError("batch 'inputs' tensor must include a batch dimension")
        batch_size = int(inputs.shape[0])
    else:
        batch_size = int(target.shape[0])
    if int(target.shape[0]) != batch_size:
        raise ValueError(
            f"batch target contains {target.shape[0]} values for batch size "
            f"{batch_size}"
        )
    return batch_size


def _predict(model: nn.Module, inputs: Any, batch_size: int) -> torch.Tensor:
    prediction = model(inputs)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("model(inputs) must return a torch.Tensor")
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if prediction.ndim != 1 or prediction.shape[0] != batch_size:
        raise ValueError(
            "model output must have shape [B] or [B, 1]; "
            f"received {tuple(prediction.shape)} for B={batch_size}"
        )
    if not prediction.is_floating_point():
        raise TypeError("model output must be floating point")
    if not bool(torch.isfinite(prediction).all().item()):
        raise FloatingPointError("model produced non-finite predictions")
    return prediction


def _run_epoch(
    model: nn.Module,
    loader: Any,
    *,
    task: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[Any],
    gradient_clip_norm: float,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    weight_sum = 0.0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for batch_index, batch in enumerate(loader):
        inputs = _move_to_device(batch["inputs"], device)
        target_name = "label" if task == "classification" else "energy"
        target = _batch_vector(batch[target_name], target_name, device)
        batch_size = _infer_batch_size(inputs, target)
        if not bool(torch.isfinite(target).all().item()):
            raise FloatingPointError(f"batch {target_name!r} contains non-finite values")
        if task == "classification" and not bool(
            torch.logical_or(target == 0, target == 1).all().item()
        ):
            raise ValueError("classification labels must be binary (0 or 1)")

        sample_weight = (
            _batch_vector(batch["sample_weight"], "sample_weight", device)
            if "sample_weight" in batch
            else torch.ones(batch_size, device=device)
        ).float()
        if int(sample_weight.shape[0]) != batch_size:
            raise ValueError(
                "batch 'sample_weight' must contain "
                f"{batch_size} values; received {sample_weight.shape[0]}"
            )
        if (
            not bool(torch.isfinite(sample_weight).all().item())
            or bool((sample_weight < 0).any().item())
            or float(sample_weight.sum().item()) <= 0.0
        ):
            raise ValueError("sample weights must be finite, non-negative, and sum to > 0")

        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast:
            prediction = _predict(model, inputs, batch_size)
            numeric_target = target.to(prediction.dtype)
            per_event_loss = (
                F.binary_cross_entropy_with_logits(
                    prediction, numeric_target, reduction="none"
                )
                if task == "classification"
                else F.mse_loss(prediction, numeric_target, reduction="none")
            )
            loss = (
                per_event_loss * sample_weight.to(per_event_loss.dtype)
            ).sum() / sample_weight.sum()
        if not bool(torch.isfinite(loss).item()):
            phase = "training" if training else "validation"
            raise FloatingPointError(f"non-finite {phase} loss in batch {batch_index}")

        if training:
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError("model produced non-finite gradients")
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

        host_weights = sample_weight.detach().cpu().numpy().astype(np.float64)
        host_losses = per_event_loss.detach().float().cpu().numpy()
        loss_sum += float(np.sum(host_losses * host_weights))
        weight_sum += float(host_weights.sum())
        predictions.append(prediction.detach().float().cpu().numpy())
        targets.append(target.detach().float().cpu().numpy())
        weights.append(host_weights)

    if not predictions:
        phase = "training" if training else "validation"
        raise RuntimeError(f"{phase} data loader produced no events")
    prediction_array = np.concatenate(predictions).astype(np.float64, copy=False)
    target_array = np.concatenate(targets).astype(np.float64, copy=False)
    weight_array = np.concatenate(weights)
    result: dict[str, float | int] = {
        "loss": loss_sum / weight_sum,
        "events": int(target_array.size),
    }
    if task == "classification":
        result["auc"] = _weighted_auc(
            target_array.astype(np.int64), prediction_array, weight_array
        )
    else:
        squared_error = np.square(prediction_array - target_array)
        result["rmse"] = float(
            np.sqrt(np.sum(weight_array * squared_error) / weight_array.sum())
        )
    return result


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _model_metadata(model: nn.Module) -> dict[str, Any]:
    """Capture optional reconstruction metadata without requiring a registry."""

    metadata: dict[str, Any] = {
        "module": model.__class__.__module__,
        "class_name": model.__class__.__qualname__,
    }
    config_method = getattr(model, "config_dict", None)
    if callable(config_method):
        config = config_method()
        if not isinstance(config, Mapping):
            raise TypeError("model.config_dict() must return a mapping")
        metadata["config"] = dict(config)
    architecture = getattr(model, "architecture_metadata", None)
    if architecture is not None:
        if not isinstance(architecture, Mapping):
            raise TypeError("model.architecture_metadata must be a mapping")
        metadata["architecture"] = dict(architecture)
    return metadata


def _plot_history(
    path: Path,
    history: list[dict[str, Any]],
    task: str,
) -> None:
    epoch = [row["epoch"] for row in history]
    metric = "auc" if task == "classification" else "rmse"
    metric_label = "AUC" if task == "classification" else "RMSE (MeV)"
    figure = Figure(figsize=(10, 4), constrained_layout=True)
    loss_axis, metric_axis = figure.subplots(1, 2)
    loss_axis.plot(epoch, [row["train_loss"] for row in history], label="Train")
    loss_axis.plot(epoch, [row["val_loss"] for row in history], label="Validation")
    loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    metric_axis.plot(
        epoch, [row[f"train_{metric}"] for row in history], label="Train"
    )
    metric_axis.plot(
        epoch, [row[f"val_{metric}"] for row in history], label="Validation"
    )
    metric_axis.set(title=metric_label, xlabel="Epoch", ylabel=metric_label)
    metric_axis.grid(alpha=0.25)
    metric_axis.legend()
    figure.savefig(path, dpi=150)


def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: TrainingConfig | Mapping[str, Any],
    task: str,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train a binary classifier or physical-energy regressor.

    Loaders must yield mappings containing ``inputs`` and either ``label`` or
    ``energy``. The model receives only ``inputs`` and must return raw logits
    or energy predictions with shape ``[B]`` or ``[B, 1]``. The best model is
    selected using validation AUC or RMSE and restored before this returns.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    task = str(task).strip().lower()
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")
    if isinstance(config, Mapping):
        config = TrainingConfig(**dict(config))
    elif not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig or a mapping of its fields")

    device = _resolve_device(config.device)
    amp_enabled, amp_dtype, amp_precision = _resolve_amp(
        device, config.use_amp, config.amp_precision
    )
    output_path = Path(output_dir).expanduser()
    artifacts = {
        "best_model": output_path / "best_model.pt",
        "last_model": output_path / "last_model.pt",
        "history": output_path / "history.json",
        "history_plot": output_path / "training_history.png",
    }
    existing = [path for path in artifacts.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "training outputs already exist: " + ", ".join(map(str, existing))
        )
    output_path.mkdir(parents=True, exist_ok=True)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    set_seed(config.seed, config.deterministic)
    model.to(device)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    scaler: Optional[Any] = None
    if amp_enabled and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    config_snapshot = asdict(config)
    model_metadata = _model_metadata(model)
    metric_name = "auc" if task == "classification" else "rmse"
    maximize = task == "classification"
    best_metric = -math.inf if maximize else math.inf
    best_epoch = 0
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        train_dataset = getattr(train_loader, "dataset", None)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch - 1)
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        epoch_options = {
            "task": task,
            "device": device,
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype,
            "gradient_clip_norm": config.gradient_clip_norm,
        }
        train_summary = _run_epoch(
            model, train_loader, optimizer=optimizer, scaler=scaler, **epoch_options
        )
        val_summary = _run_epoch(
            model, val_loader, optimizer=None, scaler=None, **epoch_options
        )
        current_metric = float(val_summary[metric_name])
        if not math.isfinite(current_metric):
            if task == "classification":
                raise ValueError(
                    "validation AUC needs positive and negative events with "
                    "non-zero sample weight"
                )
            raise FloatingPointError("validation RMSE is non-finite")

        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate_now,
                "train_loss": float(train_summary["loss"]),
                "val_loss": float(val_summary["loss"]),
                f"train_{metric_name}": float(train_summary[metric_name]),
                f"val_{metric_name}": current_metric,
                "train_events": int(train_summary["events"]),
                "val_events": int(val_summary["events"]),
            }
        )
        improved = (
            current_metric > best_metric + config.early_stopping_min_delta
            if maximize
            else current_metric < best_metric - config.early_stopping_min_delta
        )
        checkpoint = {
            "task": task,
            "epoch": epoch,
            "config": config_snapshot,
            "model": model_metadata,
            "validation": val_summary,
        }
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
            torch.save(
                {**checkpoint, "model_state_dict": best_state},
                artifacts["best_model"],
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {**checkpoint, "model_state_dict": _cpu_state_dict(model)},
            artifacts["last_model"],
        )
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train loss {train_summary['loss']:.6g} | "
            f"val loss {val_summary['loss']:.6g} | "
            f"val {metric_name} {current_metric:.6g}"
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            stopped_early = True
            print(f"Early stopping; restoring best model from epoch {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("training finished without a valid best model")
    model.load_state_dict(best_state)
    result = {
        "task": task,
        "device": str(device),
        "amp_precision": amp_precision,
        "loss_function": "bce_with_logits" if task == "classification" else "mse",
        "model": model_metadata,
        "selection_metric": f"val_{metric_name}",
        "best_metric": float(best_metric),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "history": history,
        "artifact_paths": {name: str(path) for name, path in artifacts.items()},
    }
    artifacts["history"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_history(artifacts["history_plot"], history, task)
    return result


__all__ = ["set_seed", "train_model"]
