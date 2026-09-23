"""Task-aware training and evaluation loops for SuperNEMO."""

from __future__ import annotations

import json
import math
import os
import random
import inspect
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import DEFAULT_SEED, TrainingConfig
from .evaluation import (
    assert_same_provenance,
    classification_bundle,
    evaluate_classification_bundle,
    file_sha256,
    matched_validation_auc,
    save_validation_bundle,
)


Task = Literal["classification", "energy"]


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch, and every available CUDA device."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic))
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _task(value: str) -> Task:
    selected = str(value).strip().lower()
    if selected not in {"classification", "energy"}:
        raise ValueError("task must be 'classification' or 'energy'")
    return cast(Task, selected)


def _device(name: str) -> torch.device:
    selected = str(name).strip().lower()
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
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
    selected = str(precision).strip().lower()
    if selected == "auto":
        selected = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if selected == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA device does not support bfloat16 AMP")
        return True, torch.bfloat16
    if selected == "float16":
        return True, torch.float16
    raise ValueError("amp_precision must be 'auto', 'float16', or 'bfloat16'")


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensor leaves without flattening input containers."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _target_vector(value: Any) -> torch.Tensor:
    """Validate and return a CPU target vector before any device transfer."""

    target = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if target.dtype != torch.float32:
        raise TypeError("batch 'target' must have dtype float32")
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1:
        raise ValueError("batch 'target' must have shape [B] or [B, 1]")
    if target.numel() == 0:
        raise ValueError("batch 'target' must not be empty")
    if target.device.type != "cpu":
        target = target.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(target).all().item()):
        raise FloatingPointError("batch 'target' contains non-finite values")
    return target


def _energy_vector(value: Any, expected: int) -> np.ndarray:
    """Return audit energy as float64 without down-casting the E1+E2 sum."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != expected:
        raise ValueError(
            f"batch 'energy' must contain {expected} values; received {array.size}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("batch 'energy' must contain finite positive E1+E2 values")
    return array


def _prediction_vector(output: Any, batch_size: int) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("model(inputs) must return a torch.Tensor")
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or int(output.shape[0]) != batch_size:
        raise ValueError(
            "model output must have shape [B] or [B, 1]; "
            f"received {tuple(output.shape)} for B={batch_size}"
        )
    if not output.is_floating_point():
        raise TypeError("model output must be floating point")
    return output


def _input_tensor_summary(inputs: Mapping[str, Any]) -> str:
    parts = []
    for name, value in sorted(inputs.items()):
        if isinstance(value, torch.Tensor):
            parts.append(f"{name}={tuple(value.shape)}/{value.dtype}")
        else:
            parts.append(f"{name}={type(value).__name__}")
    return ",".join(parts)


def _metadata_vector(value: Any, name: str, expected: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    if array.size != expected:
        raise ValueError(
            f"batch {name!r} must contain {expected} values; received {array.size}"
        )
    items = array.tolist()
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f"batch {name!r} must contain strings")
    return np.asarray(items, dtype=np.str_)


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    """Return unweighted binary AUC with half credit for tied scores."""

    labels = np.asarray(target).reshape(-1).astype(bool)
    scores = np.asarray(score, dtype=np.float64).reshape(-1)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (
        positive * negative
    )


def _classification_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("target and prediction must be aligned non-empty vectors")
    binary_prediction = prediction >= 0.0
    return {
        "auc": _binary_auc(target, prediction),
        "accuracy": float(np.mean(binary_prediction == target.astype(bool))),
        "signal_fraction": float(np.mean(target == 1.0)),
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
    collect_outputs: bool = False,
    collect_energy: bool = False,
) -> tuple[
    float,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    training = optimizer is not None
    model.train(training)
    loss_sum_device = torch.zeros((), dtype=torch.float64, device=device)
    event_count = 0
    try:
        expected_events = int(len(loader.dataset))
    except (AttributeError, TypeError) as error:
        raise TypeError("data loader must expose a sized dataset") from error
    if expected_events <= 0:
        raise ValueError("data-loader dataset must contain at least one event")
    selected_task_is_classification = task == "classification"
    retain_numeric = selected_task_is_classification or bool(collect_outputs)
    target_storage = (
        torch.empty(expected_events, dtype=torch.float32, device=device)
        if retain_numeric
        else None
    )
    prediction_storage = (
        torch.empty(expected_events, dtype=torch.float32, device=device)
        if retain_numeric
        else None
    )
    event_id_array = (
        np.empty(expected_events, dtype=object)
        if collect_outputs
        else np.empty(0, dtype=np.str_)
    )
    category_array = (
        np.empty(expected_events, dtype=object)
        if collect_outputs
        else np.empty(0, dtype=np.str_)
    )
    retain_energy = bool(collect_outputs or collect_energy)
    energy_array = (
        np.empty(expected_events, dtype=np.float64)
        if retain_energy
        else np.empty(0, dtype=np.float64)
    )
    group_id_array = (
        np.empty(expected_events, dtype=object)
        if collect_outputs
        else np.empty(0, dtype=np.str_)
    )
    split_array = (
        np.empty(expected_events, dtype=object)
        if collect_outputs
        else np.empty(0, dtype=np.str_)
    )
    residual_sum_device = torch.zeros((), dtype=torch.float64, device=device)
    absolute_residual_sum_device = torch.zeros(
        (), dtype=torch.float64, device=device
    )
    squared_residual_sum_device = torch.zeros(
        (), dtype=torch.float64, device=device
    )
    coverage_presence: bool | None = None
    coverage_sum = 0.0
    coverage_minimum = math.inf
    truncated_event_count = 0

    for batch_index, batch in enumerate(loader):
        if not isinstance(batch, Mapping):
            raise TypeError("each data-loader batch must be a mapping")
        required = {"inputs", "target", "event_id", "category"}
        if retain_energy:
            required.add("energy")
        if collect_outputs:
            required.update(("group_id", "split"))
        missing = required - set(batch)
        if missing:
            raise KeyError(f"batch is missing required fields: {sorted(missing)}")
        if not isinstance(batch["inputs"], Mapping):
            raise TypeError("batch 'inputs' must be a nested tensor mapping")

        target_cpu = _target_vector(batch["target"])
        batch_size = int(target_cpu.shape[0])
        batch_has_coverage = "token_coverage" in batch
        if coverage_presence is None:
            coverage_presence = batch_has_coverage
        elif coverage_presence != batch_has_coverage:
            raise ValueError(
                "token_coverage must be present for every data-loader batch or none"
            )
        if batch_has_coverage:
            coverage = batch["token_coverage"]
            if not isinstance(coverage, torch.Tensor):
                raise TypeError("batch 'token_coverage' must be a tensor")
            if coverage.dtype != torch.float32:
                raise TypeError("batch 'token_coverage' must have dtype float32")
            if coverage.ndim != 1 or int(coverage.shape[0]) != batch_size:
                raise ValueError("batch 'token_coverage' must have shape [B]")
            coverage_cpu = coverage.detach().to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(coverage_cpu).all().item()):
                raise FloatingPointError("batch 'token_coverage' contains non-finite values")
            if not bool(
                torch.logical_and(coverage_cpu > 0.0, coverage_cpu <= 1.0)
                .all()
                .item()
            ):
                raise ValueError("batch 'token_coverage' values must be in (0, 1]")
            coverage_sum += float(coverage_cpu.sum().item())
            coverage_minimum = min(
                coverage_minimum,
                float(coverage_cpu.min().item()),
            )
            truncated_event_count += int((coverage_cpu < 1.0).sum().item())
        event_id = _metadata_vector(batch["event_id"], "event_id", batch_size)
        category = _metadata_vector(batch["category"], "category", batch_size)
        energy = (
            _energy_vector(batch["energy"], batch_size)
            if retain_energy
            else np.empty(0, dtype=np.float64)
        )
        group_id = (
            _metadata_vector(batch["group_id"], "group_id", batch_size)
            if collect_outputs
            else np.empty(0, dtype=np.str_)
        )
        split = (
            _metadata_vector(batch["split"], "split", batch_size)
            if collect_outputs
            else np.empty(0, dtype=np.str_)
        )
        if task == "classification" and not bool(
            torch.logical_or(target_cpu == 0.0, target_cpu == 1.0).all().item()
        ):
            raise ValueError("classification targets must be binary 0/1 values")
        inputs = _move_to_device(batch["inputs"], device)
        target = target_cpu.to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        gradient_context = torch.enable_grad() if training else torch.inference_mode()
        with gradient_context, autocast:
            # Only the input representation is visible to the model. Event IDs,
            # categories, and targets remain outside the model call.
            prediction = _prediction_vector(model(inputs), batch_size)
            loss_target = target.to(dtype=prediction.dtype)
            loss = (
                F.binary_cross_entropy_with_logits(prediction, loss_target)
                if task == "classification"
                else F.mse_loss(prediction, loss_target)
            )
            finite_checks = {
                "prediction": torch.isfinite(prediction).all(),
                "loss": torch.isfinite(loss),
            }

        if training:
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            finite_checks["gradient"] = torch.isfinite(gradient_norm)

        # One device-to-host synchronization checks every finite predicate in
        # this batch.  For training it occurs before optimizer.step(), so an
        # invalid prediction, loss, or gradient can never update the model.
        finite_values = (
            torch.stack(tuple(finite_checks.values()))
            .detach()
            .to(device="cpu")
            .tolist()
        )
        finite_status = {
            name: bool(value) for name, value in zip(finite_checks, finite_values)
        }
        if not all(finite_status.values()):
            failures = [name for name, valid in finite_status.items() if not valid]
            category_counts = {
                str(name): int(np.count_nonzero(category == name))
                for name in np.unique(category)
            }
            raise FloatingPointError(
                f"non-finite {','.join(failures)}; "
                f"phase={'training' if training else 'evaluation'}; "
                f"batch={batch_index}; event_ids={event_id[:4].tolist()}; "
                f"categories={category_counts}; "
                f"target_range=({float(target_cpu.min().item())},"
                f"{float(target_cpu.max().item())}); "
                f"inputs={_input_tensor_summary(inputs)}; "
                f"amp_enabled={amp_enabled}; amp_dtype={amp_dtype}; "
                f"finite_checks={finite_status}"
            )

        if training:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

        loss_sum_device.add_(
            loss.detach().to(dtype=torch.float64) * float(batch_size)
        )
        batch_stop = event_count + batch_size
        if batch_stop > expected_events:
            raise ValueError(
                "data loader produced more events than its dataset length: "
                f"expected {expected_events}"
            )
        if retain_numeric:
            if target_storage is None or prediction_storage is None:
                raise RuntimeError("numeric output storage was not initialized")
            target_storage[event_count:batch_stop].copy_(target.detach())
            prediction_storage[event_count:batch_stop].copy_(
                prediction.detach().to(dtype=torch.float32)
            )
        if collect_outputs:
            event_id_array[event_count:batch_stop] = event_id
            category_array[event_count:batch_stop] = category
            group_id_array[event_count:batch_stop] = group_id
            split_array[event_count:batch_stop] = split
        if retain_energy:
            energy_array[event_count:batch_stop] = energy
        if not selected_task_is_classification:
            residual = prediction.detach().to(dtype=torch.float64) - target.detach().to(
                dtype=torch.float64
            )
            residual_sum_device.add_(torch.sum(residual))
            absolute_residual_sum_device.add_(torch.sum(torch.abs(residual)))
            squared_residual_sum_device.add_(torch.sum(torch.square(residual)))
        event_count += batch_size

    if event_count == 0:
        raise RuntimeError("data loader produced no events")
    if event_count != expected_events:
        raise ValueError(
            "data loader event count disagrees with its dataset length: "
            f"expected {expected_events}, received {event_count}"
        )
    loss_sum = float(loss_sum_device.detach().cpu().item())
    if retain_numeric:
        if target_storage is None or prediction_storage is None:
            raise RuntimeError("numeric output storage was not initialized")
        target_array = target_storage.detach().cpu().numpy()
        prediction_array = prediction_storage.detach().cpu().numpy()
    else:
        target_array = np.empty(0, dtype=np.float32)
        prediction_array = np.empty(0, dtype=np.float32)
    if selected_task_is_classification:
        metrics = _classification_metrics(target_array, prediction_array)
    else:
        residual_sum, absolute_residual_sum, squared_residual_sum = (
            torch.stack(
                (
                    residual_sum_device,
                    absolute_residual_sum_device,
                    squared_residual_sum_device,
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        metrics = {
            "mse_kev2": squared_residual_sum / event_count,
            "mae_kev": absolute_residual_sum / event_count,
            "rmse_kev": math.sqrt(squared_residual_sum / event_count),
            "bias_kev": residual_sum / event_count,
            "events": event_count,
        }
    if coverage_presence:
        metrics.update(
            {
                "token_coverage_mean": coverage_sum / event_count,
                "token_coverage_minimum": coverage_minimum,
                "token_truncated_events": truncated_event_count,
                "token_truncated_event_fraction": (
                    truncated_event_count / event_count
                ),
            }
        )
    if collect_outputs:
        event_id_array = np.asarray(event_id_array, dtype=np.str_)
        category_array = np.asarray(category_array, dtype=np.str_)
        group_id_array = np.asarray(group_id_array, dtype=np.str_)
        split_array = np.asarray(split_array, dtype=np.str_)
        if np.unique(event_id_array).size != event_id_array.size:
            raise ValueError("event_id values must be unique within a data-loader pass")
    return (
        loss_sum / event_count,
        metrics,
        target_array,
        prediction_array,
        event_id_array,
        category_array,
        energy_array,
        group_id_array,
        split_array,
    )


def _set_loader_epoch(loader: Any, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    set_epoch = getattr(dataset, "set_epoch", None)
    if set_epoch is not None:
        if not callable(set_epoch):
            raise TypeError("dataset.set_epoch must be callable")
        set_epoch(epoch)


def _validate_config(config: TrainingConfig) -> TrainingConfig:
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    return config


def _evaluation_protocol_identity(provenance: Mapping[str, Any]) -> tuple[str, str]:
    protocol = provenance.get("evaluation_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("dataset provenance has no evaluation protocol identity")
    manifest_digest = protocol.get("manifest_sha256")
    evaluator_digest = protocol.get("evaluator_code_sha256")
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        raise ValueError("dataset provenance has no valid evaluation manifest SHA256")
    if not isinstance(evaluator_digest, str) or len(evaluator_digest) != 64:
        raise ValueError("dataset provenance has no valid evaluator code SHA256")
    return manifest_digest, evaluator_digest


def _model_source_identity(model: nn.Module) -> dict[str, str]:
    source = inspect.getsourcefile(model.__class__)
    if source is None:
        raise ValueError("cannot identify the model source file")
    path = Path(source).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _training_source_identity() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint has no valid RNG state")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if required - set(state):
        raise ValueError("resume checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_state = (
        state["torch_cpu"]
        .detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        # last.pt is loaded with map_location=device. On a CUDA resume this
        # places the saved RNG tensors on CUDA too, but PyTorch's RNG restore
        # API specifically requires CPU ByteTensors.
        cuda_states = [
            cuda_state.detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
            for cuda_state in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _validate_checkpoint_identity(
    checkpoint: Any,
    *,
    model: nn.Module,
    task: Task,
    config: TrainingConfig,
    model_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    context: str,
    expected_kind: str,
) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid {context}")
    expected = {
        "task": task,
        "architecture_id": getattr(model, "architecture_id", None),
        "model_name": getattr(model, "model_name", model.__class__.__name__),
        "input_kind": getattr(model, "input_kind", None),
        "model_config": dict(model_config),
        "model_source": _model_source_identity(model),
        "training_source": _training_source_identity(),
        "training_config": config.to_dict(),
    }
    changed = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if changed:
        raise ValueError(
            f"{context} does not match this training request; changed: "
            + ", ".join(changed)
        )
    expected_metric = (
        "energy_matched_auc" if task == "classification" else "rmse_kev"
    )
    if int(checkpoint.get("schema_version", -1)) != 2:
        raise ValueError(f"{context} must use checkpoint schema version 2")
    if checkpoint.get("kind") != expected_kind:
        raise ValueError(f"{context} kind must be {expected_kind!r}")
    if checkpoint.get("selection_split") != "validation":
        raise ValueError(f"{context} selection split must be validation")
    if checkpoint.get("selection_metric") != expected_metric:
        raise ValueError(f"{context} selection metric must be {expected_metric}")
    epoch = checkpoint.get("epoch")
    score = checkpoint.get("score")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError(f"{context} epoch must be a positive integer")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError(f"{context} selection score must be finite")
    checkpoint_history = checkpoint.get("history")
    if not isinstance(checkpoint_history, list) or len(checkpoint_history) != epoch:
        raise ValueError(f"{context} history must contain exactly {epoch} epochs")
    best_score = checkpoint.get("best_score")
    if (
        isinstance(best_score, bool)
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
    ):
        raise ValueError(f"{context} best score must be finite")
    assert_same_provenance(
        checkpoint.get("dataset_provenance"), provenance, context=context
    )
    return checkpoint


def _checkpoint_payload(
    *,
    kind: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None,
    task: Task,
    epoch: int,
    score: float,
    best: float,
    stale_epochs: int,
    history: list[dict[str, Any]],
    config: TrainingConfig,
    model_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    model_source: Mapping[str, str],
    training_source: Mapping[str, str],
    best_epoch: int | None = None,
    best_model_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "kind": kind,
        "task": task,
        "architecture_id": getattr(model, "architecture_id", None),
        "model_name": getattr(model, "model_name", model.__class__.__name__),
        "input_kind": getattr(model, "input_kind", None),
        "model_config": dict(model_config),
        "model_source": dict(model_source),
        "training_source": dict(training_source),
        "epoch": int(epoch),
        "score": float(score),
        "best_score": float(best),
        "selection_split": "validation",
        "selection_metric": (
            "energy_matched_auc" if task == "classification" else "rmse_kev"
        ),
        "stale_epochs": int(stale_epochs),
        "history": list(history),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "training_config": config.to_dict(),
        "dataset_provenance": dict(provenance),
        "rng_state": _rng_state(),
    }
    if kind == "last":
        if best_epoch is None or best_model_state_dict is None:
            raise ValueError("last checkpoint requires the recoverable best-model snapshot")
        payload["best_epoch"] = int(best_epoch)
        payload["best_model_state_dict"] = dict(best_model_state_dict)
    return payload


def _frozen_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a small best-model snapshot to CPU for crash-safe resume."""

    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    task: Task,
    config: TrainingConfig,
    output_dir: str | Path,
    *,
    model_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Train one binary-classification or energy-regression model."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    selected_task = _task(task)
    selected = _validate_config(config)
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
    scaler: Any | None = None
    if amp_enabled and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    last_checkpoint_path = output / "last.pt"
    history_path = output / "history.json"
    history: list[dict[str, Any]] = []
    best = -math.inf if selected_task == "classification" else math.inf
    stale_epochs = 0
    start_epoch = 1
    best_epoch = 0
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    model_source = _model_source_identity(model)
    training_source = _training_source_identity()
    evaluation_protocol_identity = (
        _evaluation_protocol_identity(provenance)
        if selected_task == "classification"
        else None
    )

    if resume:
        if not last_checkpoint_path.is_file():
            if checkpoint_path.exists() or history_path.exists():
                raise ValueError(
                    "training artifacts exist but last.pt is missing; refusing an "
                    "ambiguous resume"
                )
            print("No committed epoch found; restarting from epoch 1.", flush=True)
        else:
            last_checkpoint = torch.load(
                last_checkpoint_path, map_location=device, weights_only=False
            )
            _validate_checkpoint_identity(
                last_checkpoint,
                model=model,
                task=selected_task,
                config=selected,
                model_config=model_config,
                provenance=provenance,
                context="last checkpoint",
                expected_kind="last",
            )
            model.load_state_dict(last_checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(last_checkpoint["scheduler_state_dict"])
            recorded_scaler = last_checkpoint.get("scaler_state_dict")
            if scaler is None and recorded_scaler is not None:
                raise ValueError(
                    "resume checkpoint used a gradient scaler but this run does not"
                )
            if scaler is not None:
                if recorded_scaler is None:
                    raise ValueError(
                        "resume checkpoint is missing the gradient scaler state"
                    )
                scaler.load_state_dict(recorded_scaler)
            completed_epoch = int(last_checkpoint.get("epoch", -1))
            history = last_checkpoint["history"]
            if history_path.is_file():
                disk_history = json.loads(history_path.read_text(encoding="utf-8"))
                if disk_history != history:
                    # last.pt is the atomic commit record; repair a history file
                    # left one write behind/ahead by process interruption.
                    _atomic_json(history_path, history)
            else:
                _atomic_json(history_path, history)
            best = float(last_checkpoint["best_score"])
            best_epoch = int(last_checkpoint.get("best_epoch", 0))
            recorded_best_state = last_checkpoint.get("best_model_state_dict")
            if best_epoch < 1 or not isinstance(recorded_best_state, Mapping):
                raise ValueError("last checkpoint has no recoverable best-model snapshot")
            best_model_state_dict = {
                str(name): value.detach().to(device="cpu").clone()
                for name, value in recorded_best_state.items()
                if isinstance(value, torch.Tensor)
            }
            if set(best_model_state_dict) != set(model.state_dict()):
                raise ValueError("last checkpoint best-model snapshot is incomplete")
            recovered_best = dict(last_checkpoint)
            recovered_best.update(
                {
                    "kind": "best",
                    "epoch": best_epoch,
                    "score": best,
                    "stale_epochs": 0,
                    "history": history[:best_epoch],
                    "model_state_dict": best_model_state_dict,
                }
            )
            recovered_best.pop("best_epoch", None)
            recovered_best.pop("best_model_state_dict", None)
            _atomic_torch_save(recovered_best, checkpoint_path)
            stale_epochs = int(last_checkpoint["stale_epochs"])
            start_epoch = completed_epoch + 1
            if (
                start_epoch > selected.epochs
                or stale_epochs >= selected.early_stopping_patience
            ):
                raise RuntimeError("training run is already complete; nothing to resume")
            _restore_rng_state(last_checkpoint.get("rng_state"))
    elif any(path.exists() for path in (checkpoint_path, last_checkpoint_path, history_path)):
        raise FileExistsError(
            f"refusing to overwrite an existing training run in {output}; use --resume"
        )

    for epoch in range(start_epoch, selected.epochs + 1):
        _set_loader_epoch(train_loader, epoch)
        _set_loader_epoch(val_loader, epoch)
        train_loss, train_metrics, _, _, _, _, _, _, _ = _epoch(
            model,
            train_loader,
            task=selected_task,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=selected.gradient_clip_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_loss, val_metrics, val_target, val_prediction, _, _, val_energy, _, _ = _epoch(
            model,
            val_loader,
            task=selected_task,
            device=device,
            optimizer=None,
            gradient_clip_norm=selected.gradient_clip_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            collect_energy=selected_task == "classification",
        )
        if selected_task == "classification":
            score = matched_validation_auc(
                val_target,
                val_prediction,
                val_energy,
                seed=selected.seed,
                expected_manifest_sha256=evaluation_protocol_identity[0],
                expected_evaluator_sha256=evaluation_protocol_identity[1],
            )
            val_metrics["energy_matched_auc"] = score
            improved = score > best + selected.early_stopping_min_delta
            metric_name = "energy_matched_auc"
        else:
            score = float(val_metrics["rmse_kev"])
            improved = score < best - selected.early_stopping_min_delta
            metric_name = "rmse_kev"

        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
            }
        )
        print(
            f"Epoch {epoch:03d}/{selected.epochs:03d} | "
            f"train loss {train_loss:.6g} | validation loss {val_loss:.6g} | "
            f"validation {metric_name} {val_metrics[metric_name]:.6g}",
            flush=True,
        )

        if improved:
            best = score
            best_epoch = epoch
            best_model_state_dict = _frozen_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        best_payload = (
            _checkpoint_payload(
                kind="best",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                task=selected_task,
                epoch=epoch,
                score=score,
                best=best,
                stale_epochs=stale_epochs,
                history=history,
                config=selected,
                model_config=model_config,
                provenance=provenance,
                model_source=model_source,
                training_source=training_source,
            )
            if improved
            else None
        )
        if best_model_state_dict is None or best_epoch < 1:
            raise RuntimeError("training epoch did not establish a best model")
        _atomic_torch_save(
            _checkpoint_payload(
                kind="last",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                task=selected_task,
                epoch=epoch,
                score=score,
                best=best,
                stale_epochs=stale_epochs,
                history=history,
                config=selected,
                model_config=model_config,
                provenance=provenance,
                model_source=model_source,
                training_source=training_source,
                best_epoch=best_epoch,
                best_model_state_dict=best_model_state_dict,
            ),
            last_checkpoint_path,
        )
        _atomic_json(history_path, history)
        if best_payload is not None:
            _atomic_torch_save(best_payload, checkpoint_path)
        if stale_epochs >= selected.early_stopping_patience:
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("training finished without a best checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _validate_checkpoint_identity(
        checkpoint,
        model=model,
        task=selected_task,
        config=selected,
        model_config=model_config,
        provenance=provenance,
        context="best checkpoint",
        expected_kind="best",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return history


def evaluate_model(
    model: nn.Module,
    loader: Any,
    task: Task,
    config: TrainingConfig,
    output_dir: str | Path,
    split_name: str,
    *,
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    model_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate an already-restored model and save aligned split artifacts."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    selected_task = _task(task)
    selected = _validate_config(config)
    split = str(split_name).strip()
    if not split or Path(split).name != split or split in {".", ".."}:
        raise ValueError("split_name must be a non-empty file-name component")
    device = _device(selected.device)
    amp_enabled, amp_dtype = _amp(
        device, selected.use_amp, selected.amp_precision
    )
    model.to(device)
    (
        loss,
        metrics,
        target,
        prediction,
        event_id,
        category,
        energy,
        group_id,
        split_column,
    ) = _epoch(
        model,
        loader,
        task=selected_task,
        device=device,
        optimizer=None,
        gradient_clip_norm=selected.gradient_clip_norm,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        collect_outputs=True,
    )
    metrics["loss"] = loss

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_splits = np.unique(split_column).tolist()
    if expected_splits != [split]:
        raise ValueError(
            f"data-loader split column {expected_splits} does not match {split!r}"
        )
    assert_same_provenance(
        checkpoint.get("dataset_provenance"),
        provenance,
        context="evaluation checkpoint",
    )
    if checkpoint.get("model_config") != dict(model_config):
        raise ValueError("evaluation checkpoint model configuration changed")
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    checkpoint_file_sha256 = checkpoint.get("_checkpoint_file_sha256")
    if not isinstance(checkpoint_file_sha256, str) or len(checkpoint_file_sha256) != 64:
        raise ValueError("evaluation checkpoint has no byte-stable SHA256")
    model_source = inspect.getsourcefile(model.__class__)
    metadata = {
        "schema_version": 1,
        "framework": "pytorch",
        "task": selected_task,
        "architecture_id": getattr(model, "architecture_id", None),
        "model_name": getattr(model, "model_name", model.__class__.__name__),
        "input_kind": getattr(model, "input_kind", None),
        "model_config": dict(model_config),
        "model_source": (
            None
            if model_source is None
            else {
                "path": str(Path(model_source).resolve()),
                "sha256": file_sha256(model_source),
            }
        ),
        "checkpoint": {
            "path": str(resolved_checkpoint),
            "sha256": checkpoint_file_sha256,
            "hash_source": "same_bytes_used_for_model_state",
            "epoch": int(checkpoint["epoch"]),
            "selection_split": checkpoint.get("selection_split"),
            "selection_metric": checkpoint.get("selection_metric"),
            "selection_score": float(checkpoint["score"]),
        },
        "dataset_provenance": dict(provenance),
        "classification": {
            "label_mapping": {"2nu": 1, "Bi214": 0},
            "positive_category": "2nu",
            "positive_label": 1,
            "score_space": "logit",
            "score_direction": "higher",
        },
        "energy": {
            "definition": "E1 + E2",
            "unit": "keV",
            "clipped": False,
        },
        "grouping": {
            "group_id": "event_id",
            "higher_level_correlation_id_available": False,
            "bootstrap": 0,
        },
        "native_metrics": dict(metrics),
    }

    if selected_task == "classification":
        bundle = classification_bundle(
            event_id=event_id,
            label=target,
            category=category,
            score=prediction,
            energy_condition=energy,
            group_id=group_id,
            split=split_column,
            metadata=metadata,
        )
        if split == "test":
            return evaluate_classification_bundle(
                bundle,
                architecture_id=str(getattr(model, "architecture_id", "unknown")),
                provenance=provenance,
                output_dir=output,
            )
        if split != "validation":
            raise ValueError("classification evaluation supports validation or test")
        metrics["energy_matched_auc"] = matched_validation_auc(
            target,
            prediction,
            energy,
            seed=selected.seed,
            expected_manifest_sha256=_evaluation_protocol_identity(provenance)[0],
            expected_evaluator_sha256=_evaluation_protocol_identity(provenance)[1],
        )
        validation_path = output / "validation_evaluation"
        legacy_paths = (
            output / "validation_metrics.json",
            output / "validation_predictions.npz",
        )
        if validation_path.exists() or any(path.exists() for path in legacy_paths):
            raise FileExistsError(
                f"refusing to overwrite validation artifacts in {output}"
            )
        staging = Path(tempfile.mkdtemp(prefix=".validation-", dir=output))
        try:
            save_validation_bundle(bundle, output_dir=staging)
            _atomic_json(staging / "validation_metrics.json", metrics)
            os.rename(staging, validation_path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return metrics

    # Keep the existing energy-regression workflow usable without broadening
    # this Transformer's classification-only benchmark.
    metrics_path = output / f"{split}_metrics.json"
    prediction_path = output / f"{split}_predictions.npz"
    if metrics_path.exists() or prediction_path.exists():
        raise FileExistsError(f"refusing to overwrite {split} artifacts in {output}")
    temporary = prediction_path.with_name(
        f".{prediction_path.stem}.{os.getpid()}.tmp.npz"
    )
    try:
        np.savez_compressed(
            temporary,
            event_id=event_id,
            category=category,
            energy_target=energy,
            energy_pred=np.asarray(prediction, dtype=np.float64),
            sample_weight=np.ones(len(event_id), dtype=np.float32),
            group_id=group_id,
            split=split_column,
            __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary, prediction_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _atomic_json(metrics_path, metrics)
    return metrics


__all__ = ["evaluate_model", "set_seed", "train_model"]
