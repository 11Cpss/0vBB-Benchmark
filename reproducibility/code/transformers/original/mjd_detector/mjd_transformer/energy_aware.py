"""Post-training EnergyBench evaluation for completed MJD classifiers.

The original MJD training evaluator stores classification scores and labels but
not event energy.  These helpers restore a saved best checkpoint and perform
one additional held-out-test inference pass so energy-aware metrics can be
computed without retraining.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .model import MJDTransformer
from .tokenization import TokenizationConfig


DEFAULT_ENERGYBENCH_SETTINGS: dict[str, Any] = {
    "energy_bin_width_kev": 5.0,
    "energy_unit": "MeV",
    "matching_target": "overlap",
    "min_per_class": 20,
    "min_valid_bins": 2,
    "support_trim_quantile": 0.005,
    "energy_roi": None,
    "min_coverage": 0.5,
    "target_tpr": 0.90,
    "score_bins": 20,
    "min_per_bin": 20,
    "distance_correlation_max_samples": 1200,
    "seed": 42,
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _build_model(run_config: Mapping[str, Any]) -> MJDTransformer:
    representation = run_config["representation"]
    tokenization = TokenizationConfig(**representation["tokenization"])
    return MJDTransformer(
        task="classification",
        tokenization_config=tokenization,
        position_encoding=representation["position_encoding"],
        d_model=int(representation["d_model"]),
        nhead=int(representation["nhead"]),
        num_layers=int(representation["num_layers"]),
        dim_feedforward=int(representation["dim_feedforward"]),
        dropout=float(representation["dropout"]),
        num_frequencies=int(representation["num_frequencies"]),
    )


def collect_test_predictions(
    run_dir: str | Path,
    test_loader: Iterable[Mapping[str, Any]],
    *,
    device: str | torch.device = "auto",
) -> dict[str, np.ndarray]:
    """Restore one best checkpoint and collect scores, labels, and energy."""

    root = Path(run_dir)
    requested = str(device)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    selected_device = torch.device(requested)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if run_config.get("task") != "classification":
        raise ValueError("energy-aware classification requires a classification run")
    checkpoint = torch.load(
        root / "best.pt", map_location="cpu", weights_only=False
    )
    model = _build_model(run_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device).eval()

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    energies_kev: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in test_loader:
            inputs = batch["inputs"].to(
                device=selected_device,
                non_blocking=selected_device.type == "cuda",
            )
            scores.append(model(inputs).detach().float().cpu().numpy())
            labels.append(batch["clean"].detach().cpu().numpy())
            energies_kev.append(batch["energy"].detach().cpu().numpy())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not scores:
        raise RuntimeError("test loader produced no events")
    return {
        "score": np.concatenate(scores).reshape(-1).astype(np.float64),
        "label": np.concatenate(labels).reshape(-1).astype(np.int8),
        "energy_kev": np.concatenate(energies_kev).reshape(-1).astype(np.float64),
    }


def _configure_observed_grid(
    energy_mev: np.ndarray,
    energybench_metrics: Any,
) -> tuple[float, int]:
    """Extend the zero-anchored 5 keV grid through observed MJD support.

    The current NEXT package fixes its endpoint at 3000 keV, while the older
    MJD EnergyBench artifacts extend through the observed 3035 keV endpoint.
    This changes constants only in the current Python process and does not edit
    or weaken the shared NEXT implementation.
    """

    if not np.all(np.isfinite(energy_mev)):
        raise ValueError("MJD test energies must be finite")
    width_kev = 5.0
    observed_max_kev = float(np.max(energy_mev)) * 1000.0
    grid_max_kev = max(
        3000.0,
        math.ceil((observed_max_kev - 1.0e-10) / width_kev) * width_kev,
    )
    grid_count = int(round(grid_max_kev / width_kev))
    energybench_metrics.CANONICAL_ENERGY_MAX_KEV = grid_max_kev
    energybench_metrics.CANONICAL_ENERGY_BIN_COUNT = grid_count
    return grid_max_kev, grid_count


def compute_energy_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute EnergyBench classification metrics from event-aligned arrays."""

    import simple_energybench.metrics as energybench_metrics
    from simple_energybench.metrics import (
        evaluate_energy_dependence,
        evaluate_energy_matched_classification,
    )

    score = np.asarray(arrays["score"], dtype=np.float64).reshape(-1)
    label = np.asarray(arrays["label"], dtype=np.int8).reshape(-1)
    energy_kev = np.asarray(arrays["energy_kev"], dtype=np.float64).reshape(-1)
    if not (len(score) == len(label) == len(energy_kev)):
        raise ValueError("score, label, and energy must be event aligned")
    if not np.all(np.isin(label, (0, 1))):
        raise ValueError("classification labels must contain only 0 and 1")

    # MJD stores energy_label in keV; EnergyBench's public unit is MeV.
    energy_mev = energy_kev / 1000.0
    grid_max_kev, grid_count = _configure_observed_grid(
        energy_mev, energybench_metrics
    )
    selected_settings = {
        **DEFAULT_ENERGYBENCH_SETTINGS,
        **({} if settings is None else dict(settings)),
        "energy_grid_min_kev": 0.0,
        "energy_grid_max_kev": grid_max_kev,
        "energy_grid_bin_count": grid_count,
    }
    weights = np.ones(len(label), dtype=np.float64)
    classification = evaluate_energy_matched_classification(
        label, score, energy_mev, weights, selected_settings
    )
    matched = classification.get("matched") or {}
    threshold = (matched.get("operating_point") or {}).get("threshold")
    dependence = evaluate_energy_dependence(
        label,
        score,
        energy_mev,
        weights,
        None,
        threshold,
        selected_settings,
    )
    return {
        "task": "classification",
        "n_events": int(len(label)),
        "auc": classification["inclusive"]["auc"],
        "matched_auc": classification.get("matched_auc"),
        "matched_auc_status": classification.get("matched_auc_status"),
        "common_support_auc": classification.get("common_support_auc"),
        "shortcut_gap": classification.get("shortcut_gap"),
        "energy_independence_score": dependence.get(
            "overall_energy_independence_score"
        ),
        "worst_energy_independence_score": dependence.get(
            "worst_group_energy_independence_score"
        ),
        "protocol": selected_settings,
        "classification": classification,
        "energy_dependence": dependence,
    }


def evaluate_run(
    run_dir: str | Path,
    test_loader: Iterable[Mapping[str, Any]],
    *,
    expected_auc: float | None = None,
    device: str | torch.device = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate one run and persist reusable metrics and prediction arrays."""

    root = Path(run_dir)
    destination = root / "energybench_classification"
    metrics_path = destination / "metrics.json"
    if metrics_path.is_file() and not overwrite:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    arrays = collect_test_predictions(root, test_loader, device=device)
    metrics = compute_energy_metrics(arrays)
    metrics["run_id"] = root.name
    metrics["evaluation_seconds"] = time.perf_counter() - started
    if expected_auc is not None and abs(
        float(metrics["auc"]) - float(expected_auc)
    ) > 1.0e-5:
        raise RuntimeError(
            f"inclusive AUC mismatch: {metrics['auc']} versus {expected_auc}"
        )

    ready = _json_ready(metrics)
    metrics_path.write_text(
        json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        destination / "predictions.npz",
        score=arrays["score"].astype(np.float32),
        label=arrays["label"],
        energy_kev=arrays["energy_kev"],
        energy_mev=arrays["energy_kev"] / 1000.0,
    )
    return ready


__all__ = [
    "DEFAULT_ENERGYBENCH_SETTINGS",
    "collect_test_predictions",
    "compute_energy_metrics",
    "evaluate_run",
]
