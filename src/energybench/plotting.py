"""Headless publication-style plots for EnergyBench outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires matplotlib; install the project dependencies"
        ) from exc
    return plt


def plot_energy_matched_rocs(
    pair_records: Sequence[Mapping[str, Any]],
    output_path: Path,
    model_name: str,
) -> None:
    """Plot inclusive (faint dashed) and matched (solid) ROC per pair."""

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(pair_records), 1)))
    unavailable = []
    for index, record in enumerate(pair_records):
        result = record["result"]
        name = str(record["name"])
        inclusive = result.inclusive
        if inclusive.auc is not None and len(inclusive.fpr):
            axis.plot(
                inclusive.fpr,
                inclusive.tpr,
                color=colors[index],
                linestyle="--",
                linewidth=1.0,
                alpha=0.35,
            )
        if result.matched is None or result.matched.auc is None:
            unavailable.append("%s: %s" % (name, result.status))
            continue
        ci_text = ""
        if result.matched_auc_ci is not None and result.matched_auc_ci.lower is not None:
            ci_text = " [%0.3f, %0.3f]" % (
                result.matched_auc_ci.lower,
                result.matched_auc_ci.upper,
            )
        axis.plot(
            result.matched.fpr,
            result.matched.tpr,
            color=colors[index],
            linewidth=2.0,
            label="%s  matched AUC=%0.4f%s" % (
                name,
                result.matched.auc,
                ci_text,
            ),
        )
    axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle=":", linewidth=1)
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="FPR / background acceptance",
        ylabel="TPR / signal efficiency",
        title="%s — energy-matched ROC" % model_name,
    )
    axis.grid(alpha=0.22)
    if any(
        record["result"].matched is not None for record in pair_records
    ):
        axis.legend(fontsize=7.5, loc="lower right", framealpha=0.92)
    axis.text(
        0.02,
        0.98,
        "solid: energy matched; dashed: inclusive",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        color="dimgray",
    )
    if unavailable:
        display = unavailable[:4]
        if len(unavailable) > 4:
            display.append("… and %d more" % (len(unavailable) - 4))
        axis.text(
            0.02,
            0.02,
            "NA pairs:\n" + "\n".join(display),
            transform=axis.transAxes,
            va="bottom",
            fontsize=7,
            color="firebrick",
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_score_energy_dependence(
    dependence_result: Mapping[str, Any],
    output_path: Path,
    energy_unit: str,
    score_space: str,
) -> None:
    """Plot mean score and threshold acceptance against energy per group."""

    plt = _pyplot()
    groups = [
        (name, values)
        for name, values in dependence_result.get("groups", {}).items()
        if values.get("status") == "ok"
    ]
    if not groups:
        return
    has_acceptance = any("acceptance_by_energy_bin" in values for _, values in groups)
    columns = 2 if has_acceptance else 1
    figure, axes_raw = plt.subplots(
        1, columns, figsize=(7.0 * columns, 5.0), squeeze=False
    )
    axes = axes_raw[0]
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(groups)))
    for index, (name, values) in enumerate(groups):
        centers = np.asarray(values["energy_bin_centers"], dtype=float)
        score_mean = np.asarray(values["score_mean_by_energy_bin"], dtype=float)
        label = (
            "%s (dCor=%0.3f, independence=%0.3f)"
            % (
                name,
                values.get("distance_correlation", float("nan")),
                values["energy_independence_score"],
            )
        )
        axes[0].plot(
            centers,
            score_mean,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=colors[index],
            label=label,
        )
        if has_acceptance and "acceptance_by_energy_bin" in values:
            axes[1].plot(
                centers,
                np.asarray(values["acceptance_by_energy_bin"], dtype=float),
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                color=colors[index],
                label=name,
            )
    axes[0].set(
        xlabel="Energy [%s]" % energy_unit,
        ylabel="Mean classifier score (%s)" % score_space,
        title="Class-conditional score dependence",
    )
    if has_acceptance:
        axes[1].set(
            xlabel="Energy [%s]" % energy_unit,
            ylabel="Acceptance at reported threshold",
            title="Cut efficiency / sculpting diagnostic",
            ylim=(-0.03, 1.03),
        )
        axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=7.5)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=170, bbox_inches="tight")
    plt.close(figure)


def _weighted_subsample(
    size: int, limit: int, seed: int = 42
) -> np.ndarray:
    if size <= limit:
        return np.arange(size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=limit, replace=False))


def plot_energy_regression(
    energy_true: Any,
    energy_pred: Any,
    sample_weight: Any,
    result: Any,
    output_path: Path,
    energy_unit: str,
    model_name: str,
) -> None:
    """True-vs-predicted and residual response plot."""

    plt = _pyplot()
    truth = np.asarray(energy_true, dtype=float)
    prediction = np.asarray(energy_pred, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    finite = (
        np.isfinite(truth)
        & np.isfinite(prediction)
        & np.isfinite(weight)
        & (weight > 0)
    )
    truth, prediction, weight = truth[finite], prediction[finite], weight[finite]
    if len(truth) == 0:
        return
    indices = _weighted_subsample(len(truth), 30000)
    residual = prediction - truth

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].hexbin(
        truth[indices],
        prediction[indices],
        gridsize=60,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    low = float(min(np.min(truth), np.min(prediction)))
    high = float(max(np.max(truth), np.max(prediction)))
    axes[0].plot([low, high], [low, high], "r--", linewidth=1.3, label="ideal")
    axes[0].set(
        xlabel="True energy [%s]" % energy_unit,
        ylabel="Predicted energy [%s]" % energy_unit,
        title="Event-wise response",
    )
    axes[0].legend()

    centers = np.asarray(getattr(result, "energy_bin_centers", []), dtype=float)
    bias = np.asarray(getattr(result, "bias_by_energy_bin", []), dtype=float)
    resolution = np.asarray(
        getattr(result, "resolution68_by_energy_bin", []), dtype=float
    )
    axes[1].scatter(
        truth[indices],
        residual[indices],
        s=4,
        alpha=0.08,
        color="#4c78a8",
        rasterized=True,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    if centers.size and bias.size == centers.size:
        axes[1].plot(
            centers,
            bias,
            color="#d62728",
            marker="o",
            linewidth=1.8,
            label="median residual",
        )
        if resolution.size == centers.size:
            axes[1].fill_between(
                centers,
                bias - resolution,
                bias + resolution,
                color="#d62728",
                alpha=0.18,
                label="±68% half-width",
            )
    axes[1].set(
        xlabel="True energy [%s]" % energy_unit,
        ylabel="Predicted − true [%s]" % energy_unit,
        title="Residual vs true energy",
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    score = getattr(result, "energy_regression_score", None)
    score_text = "NA" if score is None else "%0.4f" % score
    figure.suptitle("%s — energy regression (ERS-v1=%s)" % (model_name, score_text))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_energy_histograms(
    result: Any,
    output_path: Path,
    energy_unit: str,
    model_name: str,
) -> None:
    """Overlay normalized true/predicted energy histograms and their ratio."""

    plt = _pyplot()
    edges = np.asarray(getattr(result, "histogram_edges", []), dtype=float)
    true_probability = np.asarray(
        getattr(result, "true_histogram_probability", []), dtype=float
    )
    pred_probability = np.asarray(
        getattr(result, "pred_histogram_probability", []), dtype=float
    )
    if edges.size < 2 or true_probability.size == 0:
        return
    # Histogram arrays include explicit underflow and overflow.  The visible
    # bins are indices 1..N; the two tail masses are annotated separately.
    visible_true = true_probability[1:-1]
    visible_pred = pred_probability[1:-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    axes[0].stairs(visible_true, edges, color="black", linewidth=1.8, label="true")
    axes[0].stairs(
        visible_pred, edges, color="#d62728", linewidth=1.8, label="predicted"
    )
    axes[0].set_ylabel("Probability per bin")
    axes[0].set_title(
        "%s — true/predicted energy distribution" % model_name
    )
    axes[0].legend()
    similarity = getattr(result, "histogram_similarity", None)
    overlap = getattr(result, "histogram_overlap", None)
    axes[0].text(
        0.02,
        0.96,
        "JS similarity=%s\nhist overlap=%s\nunder/overflow true=(%0.3g, %0.3g)\n"
        "under/overflow pred=(%0.3g, %0.3g)"
        % (
            "NA" if similarity is None else "%0.4f" % similarity,
            "NA" if overlap is None else "%0.4f" % overlap,
            true_probability[0],
            true_probability[-1],
            pred_probability[0],
            pred_probability[-1],
        ),
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    ratio = np.divide(
        visible_pred,
        visible_true,
        out=np.full_like(visible_pred, np.nan),
        where=visible_true > 0,
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].plot(centers, ratio, color="#d62728", marker="o", markersize=2.5)
    axes[1].set(
        xlabel="Energy [%s]" % energy_unit,
        ylabel="Pred / true",
        ylim=(0.0, min(5.0, max(2.0, float(np.nanpercentile(ratio, 95)) * 1.2)))
        if np.any(np.isfinite(ratio))
        else (0.0, 2.0),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(figure)
