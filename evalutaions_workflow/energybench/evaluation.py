"""One-call inference, EnergyBench metrics, plots, and artifact writing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import numpy as np
import torch

from .config import EvaluationConfig
from .metrics import (
    evaluate_energy_dependence,
    evaluate_energy_matched_classification,
    evaluate_regression_metrics,
)
from .plotting import (
    plot_energy_histograms,
    plot_energy_matched_roc,
    plot_energy_regression,
    plot_score_energy_dependence,
)


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


def _normalize_evaluation_config(
    config: EvaluationConfig | Mapping[str, Any] | None,
) -> EvaluationConfig:
    if config is None:
        return EvaluationConfig()
    if isinstance(config, EvaluationConfig):
        return config
    if isinstance(config, Mapping):
        return EvaluationConfig(**dict(config))
    raise TypeError("config must be an EvaluationConfig, a flat mapping, or None")


def _config_dict(config: EvaluationConfig) -> dict[str, Any]:
    return config.to_dict()


_LARGE_ARRAY_KEYS = {
    "fpr",
    "tpr",
    "thresholds",
    "matched_mask",
    "matched_weights",
}


def _json_ready(value: Any, *, omit_large_arrays: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, omit_large_arrays=omit_large_arrays)
            for key, item in value.items()
            if not (omit_large_arrays and key in _LARGE_ARRAY_KEYS)
        }
    if isinstance(value, np.ndarray):
        return [_json_ready(item, omit_large_arrays=omit_large_arrays) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, omit_large_arrays=omit_large_arrays) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _claim_output(directory: Path, names: Iterable[str], overwrite: bool) -> None:
    existing = [directory / name for name in names if (directory / name).exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"evaluation artifacts already exist: {paths}")
    directory.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_summary_csv(path: Path, values: Mapping[str, Any]) -> None:
    row = {
        key: value
        for key, value in values.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _matching_threshold(classification: Mapping[str, Any]) -> float | None:
    matched = classification.get("matched")
    if not isinstance(matched, Mapping):
        return None
    point = matched.get("operating_point")
    if not isinstance(point, Mapping):
        return None
    threshold = point.get("threshold")
    return None if threshold is None else float(threshold)


def evaluate_classification(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    device: str | torch.device | None = "auto",
    output_dir: str | Path = "results/classification",
    config: EvaluationConfig | Mapping[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run binary inference, EnergyBench classification metrics, and plots."""

    evaluation_config = _normalize_evaluation_config(config)
    destination = Path(output_dir).expanduser().resolve()
    artifacts = [
        "metrics.json",
        "results.csv",
        "predictions.npz",
        "energy_matched_roc.png",
        "score_energy_dependence.png",
    ]
    _claim_output(destination, artifacts, overwrite)
    arrays = _run_inference(model, dataloader, _resolve_device(device), True)
    labels = arrays["label"]
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("classification labels must be numeric 0/1 values")
    classification = evaluate_energy_matched_classification(
        labels,
        arrays["prediction"],
        arrays["energy"],
        arrays["sample_weight"],
        evaluation_config,
    )
    dependence = evaluate_energy_dependence(
        labels,
        arrays["prediction"],
        arrays["energy"],
        arrays["sample_weight"],
        arrays["category"],
        _matching_threshold(classification),
        evaluation_config,
    )
    plot_energy_matched_roc(
        classification, destination / "energy_matched_roc.png", model.__class__.__name__
    )
    plot_score_energy_dependence(
        dependence,
        destination / "score_energy_dependence.png",
        _config_dict(evaluation_config).get("energy_unit", "MeV"),
    )
    np.savez_compressed(
        destination / "predictions.npz",
        score=arrays["prediction"].astype(np.float32),
        label=labels.astype(np.int8),
        energy=arrays["energy"],
        sample_weight=arrays["sample_weight"].astype(np.float32),
        event_id=arrays["event_id"],
        category=arrays["category"],
        group_id=arrays["group_id"],
        split=arrays["split"],
        projection_coverage=arrays["projection_coverage"],
    )
    inclusive_auc = classification["inclusive"]["auc"]
    metrics = {
        "task": "classification",
        "n_events": int(labels.size),
        "auc": None if inclusive_auc is None else float(inclusive_auc),
        "matched_auc": classification.get("matched_auc"),
        "matched_auc_status": classification.get("matched_auc_status", classification.get("status")),
        "common_support_auc": classification.get("common_support_auc"),
        "shortcut_gap": classification.get("shortcut_gap"),
        "energy_independence_score": dependence.get("overall_energy_independence_score"),
        "worst_energy_independence_score": dependence.get(
            "worst_group_energy_independence_score"
        ),
        "protocol": _config_dict(evaluation_config),
        "classification": classification,
        "energy_dependence": dependence,
        "artifacts": {
            "metrics": "metrics.json",
            "summary": "results.csv",
            "predictions": "predictions.npz",
            "roc": "energy_matched_roc.png",
            "dependence": "score_energy_dependence.png",
        },
    }
    ready = _json_ready(metrics, omit_large_arrays=True)
    _write_json(destination / "metrics.json", ready)
    _write_summary_csv(
        destination / "results.csv",
        {key: ready.get(key) for key in (
            "task",
            "n_events",
            "auc",
            "matched_auc",
            "matched_auc_status",
            "common_support_auc",
            "shortcut_gap",
            "energy_independence_score",
            "worst_energy_independence_score",
        )},
    )
    return ready


def evaluate_regression(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    device: str | torch.device | None = "auto",
    output_dir: str | Path = "results/regression",
    config: EvaluationConfig | Mapping[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run scalar-energy inference, ERS-v1 metrics, and standard plots."""

    evaluation_config = _normalize_evaluation_config(config)
    destination = Path(output_dir).expanduser().resolve()
    artifacts = [
        "metrics.json",
        "results.csv",
        "predictions.npz",
        "energy_regression.png",
        "energy_histograms.png",
    ]
    _claim_output(destination, artifacts, overwrite)
    arrays = _run_inference(model, dataloader, _resolve_device(device), False)
    regression = evaluate_regression_metrics(
        arrays["energy"], arrays["prediction"], arrays["sample_weight"], evaluation_config
    )
    if "ers" not in regression and "energy_regression_score" in regression:
        regression["ers"] = regression["energy_regression_score"]
    if "histogram_similarity" not in regression and "hist_similarity" in regression:
        regression["histogram_similarity"] = regression["hist_similarity"]
    plot_energy_regression(
        arrays["energy"],
        arrays["prediction"],
        regression,
        destination / "energy_regression.png",
        _config_dict(evaluation_config).get("energy_unit", "MeV"),
        int(_config_dict(evaluation_config).get("seed", 42)),
    )
    plot_energy_histograms(
        regression,
        destination / "energy_histograms.png",
        _config_dict(evaluation_config).get("energy_unit", "MeV"),
    )
    np.savez_compressed(
        destination / "predictions.npz",
        energy_true=arrays["energy"],
        energy_pred=arrays["prediction"].astype(np.float64),
        sample_weight=arrays["sample_weight"].astype(np.float32),
        event_id=arrays["event_id"],
        category=arrays["category"],
        group_id=arrays["group_id"],
        split=arrays["split"],
        projection_coverage=arrays["projection_coverage"],
    )
    scalar_names = (
        "ers",
        "event_score",
        "histogram_similarity",
        "histogram_overlap",
        "jsd_bits",
        "wasserstein_1",
        "mae",
        "rmse",
        "bias",
        "r2",
        "mae_skill",
        "fractional_bias",
        "fractional_resolution_68",
        "balanced_fractional_mae",
        "finite_fraction",
    )
    metrics: dict[str, Any] = {
        "task": "regression",
        "n_events": int(arrays["energy"].size),
        **{name: regression.get(name) for name in scalar_names},
        "protocol": _config_dict(evaluation_config),
        "energy_regression": regression,
        "artifacts": {
            "metrics": "metrics.json",
            "summary": "results.csv",
            "predictions": "predictions.npz",
            "response": "energy_regression.png",
            "histograms": "energy_histograms.png",
        },
    }
    ready = _json_ready(metrics, omit_large_arrays=True)
    _write_json(destination / "metrics.json", ready)
    _write_summary_csv(
        destination / "results.csv",
        {key: ready.get(key) for key in ("task", "n_events", *scalar_names)},
    )
    return ready


__all__ = ["evaluate_classification", "evaluate_regression"]
