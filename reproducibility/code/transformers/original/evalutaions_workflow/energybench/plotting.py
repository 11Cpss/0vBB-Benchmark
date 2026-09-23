"""Small, headless plotting helpers for the standalone evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_energy_matched_roc(
    result: Mapping[str, Any], output_path: str | Path, title: str = "Classifier"
) -> Path:
    """Plot inclusive and energy-matched ROC curves."""

    plt = _pyplot()
    inclusive = result["inclusive"]
    matched = result.get("matched")
    figure, axis = plt.subplots(figsize=(7.4, 6.4))
    axis.plot(
        inclusive["fpr"],
        inclusive["tpr"],
        linestyle="--",
        linewidth=1.6,
        color="tab:blue",
        alpha=0.75,
        label=f"Inclusive AUC = {inclusive['auc']:.4f}",
    )
    if matched is not None and matched.get("auc") is not None:
        formal_auc = result.get("matched_auc")
        status = result.get("matched_auc_status", result.get("status", "ok"))
        label_auc = matched["auc"] if formal_auc is None else formal_auc
        suffix = "" if status == "ok" else f" ({status})"
        axis.plot(
            matched["fpr"],
            matched["tpr"],
            linewidth=2.2,
            color="tab:orange",
            label=f"Energy-matched AUC = {label_auc:.4f}{suffix}",
        )
    axis.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False-positive rate / background acceptance",
        ylabel="True-positive rate / signal efficiency",
        title=f"{title} — inclusive and energy-matched ROC",
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="lower right")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_score_energy_dependence(
    result: Mapping[str, Any],
    output_path: str | Path,
    energy_unit: str = "MeV",
) -> Path:
    """Plot class-conditional mean score and cut acceptance versus energy."""

    groups = [
        (name, values)
        for name, values in result.get("groups", {}).items()
        if values.get("status") == "ok"
    ]
    if not groups:
        plt = _pyplot()
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        axis.text(
            0.5,
            0.5,
            "Energy-dependence curves are unavailable.\n"
            "No class group passed the configured sparse-bin gates.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return destination
    plt = _pyplot()
    has_acceptance = any("acceptance_by_energy_bin" in values for _, values in groups)
    columns = 2 if has_acceptance else 1
    figure, axes = plt.subplots(
        1, columns, figsize=(6.5 * columns, 4.8), squeeze=False
    )
    palette = plt.get_cmap("tab10")(np.linspace(0, 1, len(groups)))
    for color, (name, values) in zip(palette, groups):
        centers = np.asarray(values["energy_bin_centers"], dtype=float)
        means = np.asarray(values["score_mean_by_energy_bin"], dtype=float)
        independence = values.get("energy_independence_score")
        label = name if independence is None else f"{name} (independence={independence:.3f})"
        axes[0, 0].plot(centers, means, marker="o", linewidth=1.4, color=color, label=label)
        if has_acceptance and "acceptance_by_energy_bin" in values:
            axes[0, 1].plot(
                centers,
                values["acceptance_by_energy_bin"],
                marker="o",
                linewidth=1.4,
                color=color,
                label=name,
            )
    axes[0, 0].set(
        xlabel=f"Energy [{energy_unit}]",
        ylabel="Mean classifier logit",
        title="Class-conditional score dependence",
    )
    axes[0, 0].legend(fontsize=8)
    if has_acceptance:
        axes[0, 1].set(
            xlabel=f"Energy [{energy_unit}]",
            ylabel="Acceptance",
            title="Cut acceptance versus energy",
            ylim=(-0.03, 1.03),
        )
        axes[0, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_energy_regression(
    truth: np.ndarray,
    prediction: np.ndarray,
    result: Mapping[str, Any],
    output_path: str | Path,
    energy_unit: str = "MeV",
    seed: int = 42,
) -> Path:
    """Plot predicted energy and fixed-width residual response."""

    valid = np.isfinite(truth) & np.isfinite(prediction)
    x = np.asarray(truth, dtype=float)[valid]
    y = np.asarray(prediction, dtype=float)[valid]
    if x.size == 0:
        plt = _pyplot()
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        axis.text(
            0.5,
            0.5,
            "No finite regression predictions were produced.\n"
            "ERS-v1 is 0 and event-level error metrics are unavailable.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return destination
    if x.size > 30_000:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(x.size, 30_000, replace=False))
        plot_x, plot_y = x[selected], y[selected]
    else:
        plot_x, plot_y = x, y
    plot_residual = plot_y - plot_x
    figure_values = result.get("diagnostic_bins", result)
    centers = np.asarray(figure_values.get("energy_bin_centers", []), dtype=float)
    medians = np.asarray(
        figure_values.get("median_residual_by_energy_bin", figure_values.get("bias_by_energy_bin", [])),
        dtype=float,
    )
    widths = np.asarray(
        figure_values.get("resolution68_by_energy_bin", np.full_like(medians, np.nan)),
        dtype=float,
    )

    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    axes[0].hexbin(plot_x, plot_y, gridsize=65, bins="log", mincnt=1, cmap="viridis")
    lower = float(min(np.min(plot_x), np.min(plot_y)))
    upper = float(max(np.max(plot_x), np.max(plot_y)))
    axes[0].plot([lower, upper], [lower, upper], color="red", linestyle="--", linewidth=1.3)
    axes[0].set(
        xlabel=f"True energy [{energy_unit}]",
        ylabel=f"Predicted energy [{energy_unit}]",
        title=(
            "Prediction response"
            if result.get("rmse") is None
            else f"Prediction response (RMSE={result['rmse']:.4g} {energy_unit})"
        ),
    )
    axes[1].scatter(
        plot_x,
        plot_residual,
        s=3,
        alpha=0.08,
        color="tab:blue",
        rasterized=True,
    )
    if centers.size and medians.size == centers.size:
        axes[1].plot(centers, medians, color="tab:orange", marker="o", label="Median residual")
        if widths.size == centers.size:
            axes[1].fill_between(
                centers,
                medians - widths,
                medians + widths,
                color="tab:orange",
                alpha=0.22,
                label="Median ± 68% half-width",
            )
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set(
        xlabel=f"True energy [{energy_unit}]",
        ylabel=f"Prediction − truth [{energy_unit}]",
        title="Residual versus energy (5 keV bins)",
    )
    if centers.size:
        axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_energy_histograms(
    result: Mapping[str, Any],
    output_path: str | Path,
    energy_unit: str = "MeV",
) -> Path:
    """Plot truth/prediction probability masses and their ratio."""

    edges = np.asarray(result["histogram_edges"], dtype=float)
    truth_mass = np.asarray(
        result.get("truth_histogram", result.get("true_histogram_probability")), dtype=float
    )
    pred_mass = np.asarray(
        result.get("prediction_histogram", result.get("pred_histogram_probability")), dtype=float
    )
    # Metric arrays may include underflow/overflow; plots show visible bins only.
    if truth_mass.size == edges.size + 1:
        truth_visible = truth_mass[1:-1]
        pred_visible = pred_mass[1:-1]
    else:
        truth_visible = truth_mass[: edges.size - 1]
        pred_visible = pred_mass[: edges.size - 1]
    centers = (edges[:-1] + edges[1:]) / 2
    ratio = np.divide(
        pred_visible,
        truth_visible,
        out=np.full_like(pred_visible, np.nan),
        where=truth_visible > 0,
    )
    plt = _pyplot()
    figure, axes = plt.subplots(
        2, 1, figsize=(8.0, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    axes[0].stairs(truth_visible, edges, linewidth=1.8, label="Truth")
    axes[0].stairs(pred_visible, edges, linewidth=1.8, label="Prediction")
    axes[0].set(
        ylabel="Probability mass per 5 keV bin",
        title=(
            f"Energy spectra — ERS={result['ers']:.4f}, "
            f"histogram similarity={result['histogram_similarity']:.4f}"
        ),
    )
    axes[0].legend()
    axes[1].plot(centers, ratio, marker="o", markersize=3, linewidth=1.2)
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    finite_ratio = ratio[np.isfinite(ratio)]
    if finite_ratio.size:
        axes[1].set_ylim(0, max(1.25, min(5.0, float(np.quantile(finite_ratio, 0.95)) * 1.2)))
    axes[1].set(xlabel=f"Energy [{energy_unit}]", ylabel="Prediction / truth")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


__all__ = [
    "plot_energy_histograms",
    "plot_energy_matched_roc",
    "plot_energy_regression",
    "plot_score_energy_dependence",
]
