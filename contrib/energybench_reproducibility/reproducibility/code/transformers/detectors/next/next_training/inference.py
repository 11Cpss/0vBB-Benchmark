"""NEXT native prediction export; no EnergyBench metric calculation."""
from __future__ import annotations
from typing import Any, Iterable, Mapping, MutableMapping
import numpy as np
import torch

def _resolve_device(device: str | torch.device | None) -> torch.device:
    requested = "auto" if device is None else str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved

def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value

def _prediction_vector(output: Any, expected: int) -> np.ndarray:
    if not isinstance(output, torch.Tensor):
        raise TypeError("model(inputs) must return a torch.Tensor")
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or output.shape[0] != expected:
        raise ValueError(
            "model output must have shape [batch] or [batch, 1]; "
            f"received {tuple(output.shape)} for batch size {expected}"
        )
    return output.detach().float().cpu().numpy()

def _numpy_1d(value: Any, name: str, expected: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array).reshape(-1)
    if array.size != expected:
        raise ValueError(f"batch {name!r} must contain {expected} values")
    return array

def _strings(value: Any, expected: int, default: str) -> np.ndarray:
    if value is None:
        return np.full(expected, default, dtype=str)
    values = _numpy_1d(value, "metadata", expected)
    return values.astype(str)

def _run_inference(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
    require_labels: bool,
) -> dict[str, np.ndarray]:
    was_training = model.training
    model.to(device)
    model.eval()
    columns: MutableMapping[str, list[np.ndarray]] = {
        "prediction": [],
        "energy": [],
        "sample_weight": [],
        "event_id": [],
        "category": [],
        "group_id": [],
        "split": [],
        "projection_coverage": [],
    }
    if require_labels:
        columns["label"] = []
    event_offset = 0
    try:
        with torch.inference_mode():
            for batch in dataloader:
                if "inputs" not in batch or "energy" not in batch:
                    raise KeyError("each batch must contain 'inputs' and 'energy'")
                inputs = _move_to_device(batch["inputs"], device)
                if isinstance(inputs, torch.Tensor):
                    batch_size = int(inputs.shape[0])
                else:
                    energy_probe = _numpy_1d(
                        batch["energy"], "energy", len(batch["energy"])
                    )
                    batch_size = int(energy_probe.size)
                prediction = _prediction_vector(model(inputs), batch_size)
                columns["prediction"].append(prediction)
                columns["energy"].append(
                    _numpy_1d(batch["energy"], "energy", batch_size).astype(
                        np.float64
                    )
                )
                weight = batch.get(
                    "sample_weight", np.ones(batch_size, dtype=np.float32)
                )
                columns["sample_weight"].append(
                    _numpy_1d(weight, "sample_weight", batch_size).astype(
                        np.float64
                    )
                )
                generated_ids = np.asarray(
                    [f"event-{event_offset + index}" for index in range(batch_size)]
                )
                columns["event_id"].append(
                    generated_ids
                    if batch.get("event_id") is None
                    else _strings(batch.get("event_id"), batch_size, "")
                )
                columns["category"].append(
                    _strings(batch.get("category"), batch_size, "")
                )
                columns["group_id"].append(
                    _strings(batch.get("group_id"), batch_size, "")
                )
                columns["split"].append(
                    _strings(batch.get("split"), batch_size, "test")
                )
                coverage = batch.get(
                    "projection_coverage", np.ones(batch_size, dtype=np.float32)
                )
                columns["projection_coverage"].append(
                    _numpy_1d(
                        coverage, "projection_coverage", batch_size
                    ).astype(np.float32)
                )
                if require_labels:
                    if "label" not in batch:
                        raise KeyError("classification batches must contain 'label'")
                    raw_labels = _numpy_1d(batch["label"], "label", batch_size)
                    if not np.issubdtype(raw_labels.dtype, np.number):
                        raise ValueError(
                            "classification labels must be numeric 0/1 values"
                        )
                    numeric_labels = raw_labels.astype(np.float64)
                    if not np.all(np.isfinite(numeric_labels)) or not np.all(
                        np.isin(numeric_labels, (0.0, 1.0))
                    ):
                        raise ValueError(
                            "classification labels must be numeric 0/1 values"
                        )
                    columns["label"].append(numeric_labels.astype(np.int64))
                event_offset += batch_size
    finally:
        model.train(was_training)
    if not columns["prediction"]:
        raise RuntimeError("dataloader produced no events")
    result = {name: np.concatenate(parts) for name, parts in columns.items()}
    if np.unique(result["event_id"]).size != result["event_id"].size:
        raise ValueError("event_id values must be unique within an evaluation")
    return result
