"""Millisecond-unit reporting layered on ``evaluate_regression``.

``evaluate_regression`` was written for the NEXT/MJD *energy* regression task.
It is numerically task-agnostic, but every name and label it emits is hard-coded
to energy vocabulary, so for the Δt task its outputs are correct numbers under
misleading names:

- the target arrives under the batch key ``"energy"`` (the only regression
  target key ``train_model`` reads);
- ``energy_regression.png`` axes read "True/Predicted energy [MeV]" and
  ``energy_histograms.png`` reads "Energy [MeV]";
- ``predictions.npz`` stores ``energy_true`` / ``energy_pred``;
- ``metrics.json`` / ``results.csv`` report ``rmse`` / ``mae`` / ``bias`` in
  *scaled* units (Δt / dt_scale), not milliseconds.

The ``/ dt_scale`` scaling itself is not physics: ``EvaluationConfig`` is locked
to ``energy_unit="MeV"`` and ``make_fixed_energy_bins`` rejects any target
outside [0, 3] MeV, so dt_scale is simply the divisor that makes Δt fit that
grid.

This wrapper runs ``evaluate_regression`` unchanged, then de-scales the saved
predictions back to milliseconds and writes:

- ``cuore_regression_ms.json`` -- weighted RMSE / MAE / bias / R^2 in ms, plus a
  pile-up (Δt > 0) vs clean (Δt = 0) breakdown;
- ``cuore_dt_scatter.png`` -- predicted vs true Δt with honest axis labels.

EnergyBench's own output files are left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from simple_energybench import evaluate_regression

from .config import CuoreDataConfig, unscale_target


# Metrics EnergyBench reports that are meaningless for this target, because
# ~51% of the Δt values are exactly 0 and these are all fraction-based.
DEGENERATE_METRICS: dict[str, str] = {
    "ers": "0.0 by construction: ~51% of targets are exactly 0",
    "fractional_resolution_68": "diverges: |pred - true| / true with true == 0",
    "clean_ms.r2": "NaN by construction: ss_tot == 0 when every true value is 0",
}

SELECTION_METRICS: dict[str, str] = {
    "primary": "pileup_ms.rmse_ms",
    "secondary": "overall_ms.r2",
}


def _weighted_stats(
    y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    nan = float("nan")
    if y_true.size == 0 or float(weight.sum()) <= 0.0:
        return {
            "rmse_ms": nan,
            "mae_ms": nan,
            "bias_ms": nan,
            "r2": nan,
            "median_abs_error_ms": nan,
            "n_events": int(y_true.size),
            "weight_sum": float(weight.sum()) if weight.size else 0.0,
        }

    residual = y_pred - y_true
    weight_sum = float(weight.sum())
    mean_true = float(np.sum(weight * y_true) / weight_sum)
    ss_res = float(np.sum(weight * residual**2))
    ss_tot = float(np.sum(weight * (y_true - mean_true) ** 2))

    return {
        "rmse_ms": float(np.sqrt(ss_res / weight_sum)),
        "mae_ms": float(np.sum(weight * np.abs(residual)) / weight_sum),
        "bias_ms": float(np.sum(weight * residual) / weight_sum),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else nan,
        "median_abs_error_ms": float(np.median(np.abs(residual))),
        "n_events": int(y_true.size),
        "weight_sum": weight_sum,
    }


def _plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_pileup: np.ndarray,
    path: Path,
    model: Any,
) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6.0, 6.0))
    axis = figure.add_subplot(1, 1, 1)

    if np.any(~is_pileup):
        axis.scatter(
            y_true[~is_pileup],
            y_pred[~is_pileup],
            s=8,
            alpha=0.4,
            label="clean (Δt = 0)",
        )
    if np.any(is_pileup):
        axis.scatter(
            y_true[is_pileup],
            y_pred[is_pileup],
            s=8,
            alpha=0.4,
            label="pile-up",
        )

    if y_true.size:
        low = float(min(y_true.min(), y_pred.min(), 0.0))
        high = float(max(y_true.max(), y_pred.max(), 1.0))
    else:
        low, high = 0.0, 1.0
    axis.plot([low, high], [low, high], color="black", linewidth=1.0)

    axis.set_xlabel("pulseFinder Δt(pulse 1 -> pulse 2), true [ms]")
    axis.set_ylabel("predicted Δt [ms]")
    axis.set_title(
        f"{model.__class__.__name__} -- heuristic pulseFinder labels, "
        f"N = {int(y_true.size)}"
    )
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def evaluate_cuore_regression(
    model: Any,
    test_loader: Any,
    *,
    output_dir: str | Path,
    data_config: CuoreDataConfig,
    evaluation_config: Any = None,
    device: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run ``evaluate_regression`` then write the ms-unit CUORE report."""

    destination = Path(output_dir).expanduser().resolve()
    scaled = evaluate_regression(
        model,
        test_loader,
        device=device,
        output_dir=destination,
        config=evaluation_config,
        overwrite=overwrite,
    )

    predictions = np.load(destination / "predictions.npz", allow_pickle=True)
    y_true = unscale_target(predictions["energy_true"], data_config)
    y_pred = unscale_target(predictions["energy_pred"], data_config)
    weight = np.asarray(predictions["sample_weight"], dtype=np.float64)
    category = np.asarray(predictions["category"]).astype(str)
    is_pileup = category == "pileup"

    overall = _weighted_stats(y_true, y_pred, weight)
    abs_error = np.abs(y_pred - y_true)
    overall["frac_within_100ms"] = (
        float(np.mean(abs_error <= 100.0)) if abs_error.size else float("nan")
    )
    overall["frac_within_250ms"] = (
        float(np.mean(abs_error <= 250.0)) if abs_error.size else float("nan")
    )

    report = {
        "dt_scale": float(data_config.dt_scale),
        "n_events": int(y_true.size),
        "scaled_rmse": scaled.get("rmse"),
        "overall_ms": overall,
        "overall_ms_unweighted": _weighted_stats(
            y_true, y_pred, np.ones_like(weight)
        ),
        "pileup_ms": _weighted_stats(
            y_true[is_pileup], y_pred[is_pileup], weight[is_pileup]
        ),
        "clean_ms": _weighted_stats(
            y_true[~is_pileup], y_pred[~is_pileup], weight[~is_pileup]
        ),
        "selection_metrics": SELECTION_METRICS,
        "degenerate_metrics": DEGENERATE_METRICS,
        "energybench_degenerate_values": {
            "ers": scaled.get("ers"),
            "fractional_resolution_68": scaled.get("fractional_resolution_68"),
        },
    }

    (destination / "cuore_regression_ms.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_scatter(
        y_true, y_pred, is_pileup, destination / "cuore_dt_scatter.png", model
    )

    return {"scaled": scaled, "ms": report}


__all__ = [
    "DEGENERATE_METRICS",
    "SELECTION_METRICS",
    "evaluate_cuore_regression",
]
