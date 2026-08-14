"""Dependency-light EnergyBench metrics used by the public evaluators.

The functions in this module deliberately operate on aligned one-dimensional
arrays.  They preserve the legacy EnergyBench definitions while using the
canonical 0--3000 keV grid with 5 keV bins for every energy-dependent
diagnostic.  Returned edges are the smallest contiguous slice of that global
grid covering the observed range.  ERS-v1 keeps its weighted-quantile
performance bins because equal population bins are part of that score's
definition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .config import (
    CANONICAL_ENERGY_BIN_COUNT,
    CANONICAL_ENERGY_BIN_WIDTH_KEV,
    CANONICAL_ENERGY_MAX_KEV,
    CANONICAL_ENERGY_MIN_KEV,
)


ERS_NAME = "ERS-v1"
ERS_DEFINITION = (
    "finite_fraction * sqrt(event_score * hist_similarity), where "
    "event_score=max(0, 1-balanced_fractional_mae) and "
    "hist_similarity=1-sqrt(JSD_base2(true_hist, predicted_hist))"
)


def _as_1d(values: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received {array.shape}")
    return array


def _aligned(*named_arrays: tuple[str, np.ndarray]) -> int:
    lengths = {name: len(array) for name, array in named_arrays}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"input arrays must have equal lengths: {lengths}")
    size = next(iter(lengths.values()), 0)
    if size == 0:
        raise ValueError("input arrays must not be empty")
    return int(size)


def _weights(values: Any | None, size: int) -> np.ndarray:
    if values is None:
        return np.ones(size, dtype=float)
    weight = _as_1d(values, "weights", float)
    if len(weight) != size:
        raise ValueError(f"weights has length {len(weight)}, expected {size}")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weights must contain only finite values")
    if np.any(weight < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.any(weight > 0.0):
        raise ValueError("weights must have positive total weight")
    return weight


def _config(config: Any, name: str, default: Any) -> Any:
    """Read a setting from the public flat configuration interface."""

    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def make_fixed_energy_bins(
    values: Any,
    width_kev: float = 5.0,
    energy_unit: str = "MeV",
) -> np.ndarray:
    """Return the canonical bins covering the finite observed energy range.

    The global EnergyBench grid has 600 bins from 0 through 3000 keV, anchored
    at zero with an exact 5 keV width.  This function returns only the
    contiguous slice needed for ``values``.  Bins are left-closed/right-open;
    an observation exactly on an internal edge belongs to the bin starting at
    that edge.  The global 3000 keV endpoint is included in the final bin.
    """

    energy = _as_1d(values, "values", float)
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        raise ValueError("values must contain at least one finite energy")
    width_kev = float(width_kev)
    if (
        not math.isfinite(width_kev)
        or width_kev != CANONICAL_ENERGY_BIN_WIDTH_KEV
    ):
        raise ValueError(
            "width_kev is fixed at 5 keV by the EnergyBench protocol"
        )
    normalized_unit = (
        str(energy_unit)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
    )
    kev_per_unit = {
        "ev": 1.0e-3,
        "kev": 1.0,
        "mev": 1.0e3,
        "gev": 1.0e6,
    }
    if normalized_unit not in kev_per_unit:
        raise ValueError("energy_unit must be one of eV, keV, MeV, or GeV")
    scale = kev_per_unit[normalized_unit]
    width = CANONICAL_ENERGY_BIN_WIDTH_KEV / scale
    grid_low = CANONICAL_ENERGY_MIN_KEV / scale
    grid_high = CANONICAL_ENERGY_MAX_KEV / scale
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low < grid_low or high > grid_high:
        observed_low_kev = low * scale
        observed_high_kev = high * scale
        raise ValueError(
            "finite energy values must lie within the canonical EnergyBench "
            f"range [0, 3000] keV (inclusive); observed "
            f"[{observed_low_kev:.17g}, {observed_high_kev:.17g}] keV"
        )

    global_edges = grid_low + width * np.arange(
        CANONICAL_ENERGY_BIN_COUNT + 1, dtype=float
    )
    # Assign the declared endpoints exactly rather than relying on a floating
    # multiplication at the ends of the global grid.
    global_edges[0] = grid_low
    global_edges[-1] = grid_high
    low_index = int(np.searchsorted(global_edges, low, side="right") - 1)
    high_index = int(np.searchsorted(global_edges, high, side="right") - 1)
    low_index = min(max(low_index, 0), CANONICAL_ENERGY_BIN_COUNT - 1)
    high_index = min(max(high_index, 0), CANONICAL_ENERGY_BIN_COUNT - 1)
    return global_edges[low_index : high_index + 2].copy()


def _fixed_grid_protocol(
    edges: np.ndarray, width_kev: float, energy_unit: str
) -> dict[str, Any]:
    """Describe both the global grid and its dataset-selected slice."""

    return {
        "energy_bin_width_kev": float(width_kev),
        "energy_unit": str(energy_unit),
        "energy_grid_min_kev": CANONICAL_ENERGY_MIN_KEV,
        "energy_grid_max_kev": CANONICAL_ENERGY_MAX_KEV,
        "energy_grid_bin_count": CANONICAL_ENERGY_BIN_COUNT,
        "selected_energy_bin_count": max(0, int(len(edges)) - 1),
    }


def weighted_quantile(
    values: Any,
    quantiles: Any,
    weights: Any | None = None,
) -> np.ndarray:
    """Compute center-of-weight empirical quantiles with linear interpolation."""

    x = _as_1d(values, "values", float)
    q = np.asarray(quantiles, dtype=float)
    if not np.all(np.isfinite(q)) or np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantiles must be finite and lie in [0, 1]")
    if weights is None:
        weight = np.ones(len(x), dtype=float)
    else:
        weight = _as_1d(weights, "weights", float)
        if len(weight) != len(x):
            raise ValueError("values and weights must have equal lengths")
        if np.any(np.isfinite(weight) & (weight < 0.0)):
            raise ValueError("weights must be non-negative")
    valid = np.isfinite(x) & np.isfinite(weight) & (weight > 0.0)
    x, weight = x[valid], weight[valid]
    if x.size == 0:
        return np.full(q.shape, np.nan, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x, weight = x[order], weight[order]
    positions = (np.cumsum(weight) - 0.5 * weight) / np.sum(weight)
    return np.asarray(np.interp(q, positions, x, left=x[0], right=x[-1]))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0.0 else float("nan")


def _effective_sample_size(weights: np.ndarray) -> float:
    weight = np.asarray(weights, dtype=float)
    weight = weight[np.isfinite(weight) & (weight > 0.0)]
    denominator = float(np.sum(weight * weight))
    return float(np.sum(weight) ** 2 / denominator) if denominator > 0.0 else 0.0


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def weighted_roc_curve(
    labels: Any,
    scores: Any,
    weights: Any | None = None,
    *,
    positive_label: Any = 1,
    target_tpr: float = 0.90,
    sample_weight: Any | None = None,
) -> dict[str, Any]:
    """Compute a weighted empirical binary ROC with exact score-tie handling."""

    label = _as_1d(labels, "labels")
    score = _as_1d(scores, "scores", float)
    size = _aligned(("labels", label), ("scores", score))
    if weights is not None and sample_weight is not None:
        raise ValueError("pass either weights or sample_weight, not both")
    weight = _weights(sample_weight if sample_weight is not None else weights, size)
    target_tpr = float(target_tpr)
    if not 0.0 <= target_tpr <= 1.0:
        raise ValueError("target_tpr must lie in [0, 1]")
    valid = np.isfinite(score) & (weight > 0.0)
    positive = label[valid] == positive_label
    score = score[valid]
    weight = weight[valid]
    n_signal = int(np.sum(positive))
    n_background = int(np.sum(~positive))
    signal_weight = float(np.sum(weight[positive]))
    background_weight = float(np.sum(weight[~positive]))
    empty = np.asarray([], dtype=float)
    if n_signal == 0 or n_background == 0 or signal_weight <= 0.0 or background_weight <= 0.0:
        return {
            "status": "not_estimable_missing_class",
            "auc": None,
            "fpr": empty,
            "tpr": empty,
            "thresholds": empty,
            "operating_point": None,
            "n_signal": n_signal,
            "n_background": n_background,
            "signal_weight": signal_weight,
            "background_weight": background_weight,
        }

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_positive = positive[order]
    sorted_weight = weight[order]
    fpr = [0.0]
    tpr = [0.0]
    thresholds = [float("inf")]
    true_mass = false_mass = 0.0
    start = 0
    while start < len(sorted_score):
        stop = start + 1
        while stop < len(sorted_score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        tied_positive = sorted_positive[start:stop]
        tied_weight = sorted_weight[start:stop]
        true_mass += float(np.sum(tied_weight[tied_positive]))
        false_mass += float(np.sum(tied_weight[~tied_positive]))
        tpr.append(true_mass / signal_weight)
        fpr.append(false_mass / background_weight)
        thresholds.append(float(sorted_score[start]))
        start = stop
    fpr_array = np.asarray(fpr, dtype=float)
    tpr_array = np.asarray(tpr, dtype=float)
    threshold_array = np.asarray(thresholds, dtype=float)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0 compatibility.
        trapezoid = np.trapz
    auc = float(np.clip(trapezoid(tpr_array, fpr_array), 0.0, 1.0))
    reached = np.flatnonzero(tpr_array >= target_tpr)
    operating_point = None
    if reached.size:
        index = int(reached[0])
        operating_point = {
            "target_tpr": target_tpr,
            "tpr": float(tpr_array[index]),
            "fpr": float(fpr_array[index]),
            "threshold": float(threshold_array[index]),
            "background_rejection_fraction": float(1.0 - fpr_array[index]),
        }
    return {
        "status": "ok",
        "auc": auc,
        "fpr": fpr_array,
        "tpr": tpr_array,
        "thresholds": threshold_array,
        "operating_point": operating_point,
        "n_signal": n_signal,
        "n_background": n_background,
        "signal_weight": signal_weight,
        "background_weight": background_weight,
    }


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = len(edges) - 2
    return indices.astype(int)


def _bin_mask(values: np.ndarray, edges: np.ndarray, index: int) -> np.ndarray:
    if index == len(edges) - 2:
        return (values >= edges[index]) & (values <= edges[index + 1])
    return (values >= edges[index]) & (values < edges[index + 1])


def _weighted_ks(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: np.ndarray,
    second_weight: np.ndarray,
) -> float:
    first_order = np.argsort(first, kind="mergesort")
    second_order = np.argsort(second, kind="mergesort")
    first, first_weight = first[first_order], first_weight[first_order]
    second, second_weight = second[second_order], second_weight[second_order]
    first_weight = first_weight / np.sum(first_weight)
    second_weight = second_weight / np.sum(second_weight)
    points = np.unique(np.concatenate((first, second)))
    first_cdf = np.r_[0.0, np.cumsum(first_weight)]
    second_cdf = np.r_[0.0, np.cumsum(second_weight)]
    first_index = np.searchsorted(first, points, side="right")
    second_index = np.searchsorted(second, points, side="right")
    return float(np.max(np.abs(first_cdf[first_index] - second_cdf[second_index])))


def _wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: np.ndarray,
    second_weight: np.ndarray,
) -> float:
    first_valid = np.isfinite(first) & np.isfinite(first_weight) & (first_weight > 0.0)
    second_valid = np.isfinite(second) & np.isfinite(second_weight) & (second_weight > 0.0)
    first, first_weight = first[first_valid], first_weight[first_valid]
    second, second_weight = second[second_valid], second_weight[second_valid]
    if first.size == 0 or second.size == 0:
        return float("nan")
    first_order, second_order = np.argsort(first), np.argsort(second)
    first, first_weight = first[first_order], first_weight[first_order]
    second, second_weight = second[second_order], second_weight[second_order]
    first_weight = first_weight / np.sum(first_weight)
    second_weight = second_weight / np.sum(second_weight)
    points = np.sort(np.concatenate((first, second)))
    if points.size < 2:
        return 0.0
    intervals = np.diff(points)
    first_cdf = np.r_[0.0, np.cumsum(first_weight)]
    second_cdf = np.r_[0.0, np.cumsum(second_weight)]
    first_index = np.searchsorted(first, points[:-1], side="right")
    second_index = np.searchsorted(second, points[:-1], side="right")
    return float(np.sum(np.abs(first_cdf[first_index] - second_cdf[second_index]) * intervals))


def _classification_coverage(
    positive: np.ndarray,
    inclusive: np.ndarray,
    with_energy: np.ndarray,
    common: np.ndarray,
    matched: np.ndarray,
    weight: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, mask in (
        ("total", inclusive),
        ("with_energy", with_energy),
        ("common_support", common),
        ("matched", matched),
    ):
        result[f"signal_count_{name}"] = int(np.sum(mask & positive))
        result[f"background_count_{name}"] = int(np.sum(mask & ~positive))
    for class_name, class_mask in (("signal", positive), ("background", ~positive)):
        count_denominator = int(np.sum(with_energy & class_mask))
        weight_denominator = float(np.sum(weight[with_energy & class_mask]))
        for scope, mask in (("common_support", common), ("matched", matched)):
            result[f"{class_name}_{scope}_count_fraction"] = _safe_fraction(
                int(np.sum(mask & class_mask)), count_denominator
            )
            result[f"{class_name}_{scope}_weight_fraction"] = _safe_fraction(
                float(np.sum(weight[mask & class_mask])), weight_denominator
            )
    return result


def evaluate_energy_matched_classification(
    labels: Any,
    scores: Any,
    energies: Any,
    weights: Any | None,
    config: Any,
) -> dict[str, Any]:
    """Evaluate inclusive, common-support, and fixed-bin energy-matched ROC."""

    label = _as_1d(labels, "labels")
    score = _as_1d(scores, "scores", float)
    energy = _as_1d(energies, "energies", float)
    size = _aligned(("labels", label), ("scores", score), ("energies", energy))
    weight = _weights(weights, size)
    valid_labels = label[np.isfinite(score) & (weight > 0.0)]
    if not np.all(np.isin(valid_labels, (0, 1))):
        raise ValueError("classification labels must contain only 0 and 1")

    width_kev = float(_config(config, "energy_bin_width_kev", 5.0))
    energy_unit = str(_config(config, "energy_unit", "MeV"))
    finite_observed_energy = energy[np.isfinite(energy)]
    if finite_observed_energy.size:
        # Validate the complete observed energy column before common-support
        # trimming so an excluded outlier cannot hide a protocol violation.
        make_fixed_energy_bins(finite_observed_energy, width_kev, energy_unit)
    min_per_class = int(_config(config, "min_per_class", 20))
    min_valid_bins = int(_config(config, "min_valid_bins", 2))
    support_trim = float(_config(config, "support_trim_quantile", 0.005))
    min_coverage = float(_config(config, "min_coverage", 0.5))
    target_tpr = float(_config(config, "target_tpr", 0.90))
    target_name = str(_config(config, "matching_target", "overlap")).lower()
    roi = _config(config, "energy_roi", None)
    if min_per_class < 1 or min_valid_bins < 1:
        raise ValueError("min_per_class and min_valid_bins must be positive")
    if not 0.0 <= support_trim < 0.5:
        raise ValueError("support_trim_quantile must lie in [0, 0.5)")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must lie in [0, 1]")
    if target_name not in {"overlap", "uniform", "legacy_uniform"}:
        raise ValueError("matching_target must be 'overlap' or 'uniform'")
    if roi is not None:
        if len(roi) != 2 or not np.all(np.isfinite(roi)) or float(roi[1]) <= float(roi[0]):
            raise ValueError("energy_roi must be finite [low, high] with high > low")

    inclusive_mask = np.isfinite(score) & (weight > 0.0)
    positive = label == 1
    if not np.any(inclusive_mask & positive) or not np.any(inclusive_mask & ~positive):
        raise ValueError("classification requires finite scores from both labels")
    inclusive = weighted_roc_curve(label, score, weight, target_tpr=target_tpr)
    with_energy = inclusive_mask & np.isfinite(energy)
    empty_mask = np.zeros(size, dtype=bool)
    empty_edges = np.asarray([], dtype=float)

    def unavailable(status: str, reason: str, common: np.ndarray = empty_mask,
                    support: list[float] | None = None, edges: np.ndarray = empty_edges,
                    bins: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        coverage = _classification_coverage(
            positive, inclusive_mask, with_energy, common, empty_mask, weight
        )
        return {
            "status": status,
            "reason": reason,
            "positive_label": 1,
            "negative_label": 0,
            "matched_auc_status": status,
            "matched_auc_reason": reason,
            "inclusive_auc": inclusive["auc"],
            "inclusive": inclusive,
            "common_support_auc": None,
            "inclusive_common_support_auc": None,
            "common_support_roc": None,
            "diagnostic_matched_auc": None,
            "matched_auc": None,
            "matched": None,
            "shortcut_gap": None,
            "common_support": support,
            "bin_edges": edges,
            "n_bins_actual": max(0, len(edges) - 1),
            "bins": [] if bins is None else bins,
            "valid_bins": 0 if bins is None else sum(bool(row["valid"]) for row in bins),
            "min_per_class": min_per_class,
            "target_method": "uniform" if target_name != "overlap" else "overlap",
            "coverage": coverage,
            "effective_sample_sizes": {
                "signal_inclusive": _effective_sample_size(weight[inclusive_mask & positive]),
                "background_inclusive": _effective_sample_size(weight[inclusive_mask & ~positive]),
                "signal_matched": 0.0,
                "background_matched": 0.0,
            },
            "balance": None,
            "protocol": {
                **_fixed_grid_protocol(edges, width_kev, energy_unit),
                "support_trim_quantile": support_trim,
                "min_coverage": min_coverage,
                "min_valid_bins": min_valid_bins,
            },
        }

    if (
        int(np.sum(with_energy & positive)) < min_per_class
        or int(np.sum(with_energy & ~positive)) < min_per_class
    ):
        return unavailable(
            "not_evaluable_insufficient_statistics",
            "fewer than min_per_class finite-energy events in at least one class",
        )

    signal_energy = energy[with_energy & positive]
    background_energy = energy[with_energy & ~positive]
    signal_weight = weight[with_energy & positive]
    background_weight = weight[with_energy & ~positive]
    if support_trim:
        signal_range = weighted_quantile(
            signal_energy, [support_trim, 1.0 - support_trim], signal_weight
        )
        background_range = weighted_quantile(
            background_energy, [support_trim, 1.0 - support_trim], background_weight
        )
        support_low = max(float(signal_range[0]), float(background_range[0]))
        support_high = min(float(signal_range[1]), float(background_range[1]))
    else:
        support_low = max(float(np.min(signal_energy)), float(np.min(background_energy)))
        support_high = min(float(np.max(signal_energy)), float(np.max(background_energy)))
    if roi is not None:
        support_low = max(support_low, float(roi[0]))
        support_high = min(support_high, float(roi[1]))
    if support_high < support_low:
        return unavailable(
            "not_evaluable_no_common_support",
            "signal and background energy ranges do not overlap",
        )

    common = with_energy & (energy >= support_low) & (energy <= support_high)
    support = [support_low, support_high]
    common_roc = weighted_roc_curve(
        label[common], score[common], weight[common], target_tpr=target_tpr
    )
    edges = make_fixed_energy_bins(
        np.asarray([support_low, support_high]), width_kev, energy_unit
    )
    n_bins = len(edges) - 1
    common_indices = np.flatnonzero(common)
    common_bin = _bin_indices(energy[common], edges)
    in_grid = (common_bin >= 0) & (common_bin < n_bins)
    if not np.all(in_grid):
        common_indices = common_indices[in_grid]
        common_bin = common_bin[in_grid]
    bin_for_event = np.full(size, -1, dtype=int)
    bin_for_event[common_indices] = common_bin
    signal_counts = np.bincount(
        common_bin[positive[common_indices]], minlength=n_bins
    ).astype(int)
    background_counts = np.bincount(
        common_bin[~positive[common_indices]], minlength=n_bins
    ).astype(int)
    signal_masses = np.bincount(
        common_bin[positive[common_indices]],
        weights=weight[common_indices[positive[common_indices]]],
        minlength=n_bins,
    )
    background_masses = np.bincount(
        common_bin[~positive[common_indices]],
        weights=weight[common_indices[~positive[common_indices]]],
        minlength=n_bins,
    )
    valid_bins = (
        (signal_counts >= min_per_class)
        & (background_counts >= min_per_class)
        & (signal_masses > 0.0)
        & (background_masses > 0.0)
    )
    signal_total = float(np.sum(signal_masses))
    background_total = float(np.sum(background_masses))
    signal_fraction = signal_masses / signal_total if signal_total else np.zeros(n_bins)
    background_fraction = (
        background_masses / background_total if background_total else np.zeros(n_bins)
    )
    base_rows: list[dict[str, Any]] = []
    for index in range(n_bins):
        base_rows.append(
            {
                "index": index,
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "valid": bool(valid_bins[index]),
                "signal_count": int(signal_counts[index]),
                "background_count": int(background_counts[index]),
                "signal_base_mass": float(signal_masses[index]),
                "background_base_mass": float(background_masses[index]),
                "signal_base_fraction": float(signal_fraction[index]),
                "background_base_fraction": float(background_fraction[index]),
                "target_mass": 0.0,
                "signal_matched_mass": 0.0,
                "background_matched_mass": 0.0,
            }
        )
    if not np.any(valid_bins):
        result = unavailable(
            "not_evaluable_insufficient_statistics",
            "no 5 keV bin contains min_per_class events from both classes",
            common,
            support,
            edges,
            base_rows,
        )
        result["common_support_auc"] = common_roc["auc"]
        result["inclusive_common_support_auc"] = common_roc["auc"]
        result["common_support_roc"] = common_roc
        return result

    target_mass = (
        np.minimum(signal_fraction, background_fraction)
        if target_name == "overlap"
        else np.ones(n_bins, dtype=float)
    )
    target_mass[~valid_bins] = 0.0
    target_mass /= np.sum(target_mass)
    matched_weight = np.zeros(size, dtype=float)
    for index in np.flatnonzero(valid_bins):
        signal_selection = common & positive & (bin_for_event == index)
        background_selection = common & ~positive & (bin_for_event == index)
        matched_weight[signal_selection] = (
            weight[signal_selection] * target_mass[index] / signal_masses[index]
        )
        matched_weight[background_selection] = (
            weight[background_selection] * target_mass[index] / background_masses[index]
        )
    matched_mask = matched_weight > 0.0
    matched = weighted_roc_curve(
        label[matched_mask], score[matched_mask], matched_weight[matched_mask], target_tpr=target_tpr
    )
    signal_matched_mass = np.bincount(
        bin_for_event[matched_mask & positive],
        weights=matched_weight[matched_mask & positive],
        minlength=n_bins,
    )
    background_matched_mass = np.bincount(
        bin_for_event[matched_mask & ~positive],
        weights=matched_weight[matched_mask & ~positive],
        minlength=n_bins,
    )
    for index, row in enumerate(base_rows):
        row["target_mass"] = float(target_mass[index])
        row["signal_matched_mass"] = float(signal_matched_mass[index])
        row["background_matched_mass"] = float(background_matched_mass[index])

    coverage = _classification_coverage(
        positive, inclusive_mask, with_energy, common, matched_mask, weight
    )
    coverage_minimum = min(
        coverage["signal_matched_weight_fraction"],
        coverage["background_matched_weight_fraction"],
    )
    valid_count = int(np.sum(valid_bins))
    exact_single_energy = support_low == support_high
    diagnostic_auc = matched["auc"]
    if coverage_minimum < min_coverage:
        formal_auc = None
        metric_status = "not_evaluable_low_coverage"
        reason = f"matched weight coverage {coverage_minimum:.3f} is below {min_coverage:.3f}"
    elif valid_count < min_valid_bins and not exact_single_energy:
        formal_auc = None
        metric_status = "not_evaluable_too_few_valid_bins"
        reason = f"{valid_count} valid bins, need {min_valid_bins}"
    else:
        formal_auc = diagnostic_auc
        metric_status = "ok"
        reason = None
    shortcut_gap = (
        None
        if common_roc["auc"] is None or diagnostic_auc is None
        else float(common_roc["auc"] - diagnostic_auc)
    )
    before_tv = 0.5 * float(np.sum(np.abs(signal_fraction - background_fraction)))
    after_tv = 0.5 * float(
        np.sum(np.abs(signal_matched_mass - background_matched_mass))
    )
    balance = {
        "wasserstein_before": _wasserstein(
            energy[common & positive], energy[common & ~positive],
            weight[common & positive], weight[common & ~positive],
        ),
        "wasserstein_after": _wasserstein(
            energy[matched_mask & positive], energy[matched_mask & ~positive],
            matched_weight[matched_mask & positive], matched_weight[matched_mask & ~positive],
        ),
        "ks_before": _weighted_ks(
            energy[common & positive], energy[common & ~positive],
            weight[common & positive], weight[common & ~positive],
        ),
        "ks_after": _weighted_ks(
            energy[matched_mask & positive], energy[matched_mask & ~positive],
            matched_weight[matched_mask & positive], matched_weight[matched_mask & ~positive],
        ),
        "bin_total_variation_before": before_tv,
        "bin_total_variation_after": after_tv,
        "max_bin_mass_difference_after": float(
            np.max(np.abs(signal_matched_mass - background_matched_mass))
        ),
    }
    return {
        "status": metric_status,
        "reason": reason,
        "positive_label": 1,
        "negative_label": 0,
        "matched_auc_status": metric_status,
        "matched_auc_reason": reason,
        "inclusive_auc": inclusive["auc"],
        "inclusive": inclusive,
        "common_support_auc": common_roc["auc"],
        "inclusive_common_support_auc": common_roc["auc"],
        "common_support_roc": common_roc,
        "diagnostic_matched_auc": diagnostic_auc,
        "matched_auc": formal_auc,
        "matched": matched,
        "shortcut_gap": shortcut_gap,
        "common_support": support,
        "bin_edges": edges,
        "n_bins_actual": n_bins,
        "bins": base_rows,
        "valid_bins": valid_count,
        "min_per_class": min_per_class,
        "target_method": "uniform" if target_name != "overlap" else "overlap",
        "coverage": coverage,
        "effective_sample_sizes": {
            "signal_inclusive": _effective_sample_size(weight[inclusive_mask & positive]),
            "background_inclusive": _effective_sample_size(weight[inclusive_mask & ~positive]),
            "signal_matched": _effective_sample_size(matched_weight[matched_mask & positive]),
            "background_matched": _effective_sample_size(matched_weight[matched_mask & ~positive]),
        },
        "balance": balance,
        "protocol": {
            **_fixed_grid_protocol(edges, width_kev, energy_unit),
            "support_trim_quantile": support_trim,
            "min_coverage": min_coverage,
            "min_valid_bins": min_valid_bins,
        },
    }


def _weighted_correlation(
    first: np.ndarray, second: np.ndarray, weights: np.ndarray
) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & np.isfinite(weights) & (weights > 0.0)
    if int(np.sum(valid)) < 2:
        return float("nan")
    first, second, weights = first[valid], second[valid], weights[valid]
    weights = weights / np.sum(weights)
    first_delta = first - np.sum(weights * first)
    second_delta = second - np.sum(weights * second)
    denominator = math.sqrt(
        float(np.sum(weights * first_delta**2) * np.sum(weights * second_delta**2))
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights * first_delta * second_delta) / denominator)


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * (start + 1 + stop)
        start = stop
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def _weighted_spearman(
    first: np.ndarray, second: np.ndarray, weights: np.ndarray
) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & np.isfinite(weights) & (weights > 0.0)
    if int(np.sum(valid)) < 2:
        return float("nan")
    return _weighted_correlation(
        _midranks(first[valid]), _midranks(second[valid]), weights[valid]
    )


def _distance_correlation(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
    max_samples: int,
    seed: int,
) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & np.isfinite(weights) & (weights > 0.0)
    first, second, weights = first[valid], second[valid], weights[valid]
    if len(first) < 4:
        return float("nan")
    if len(first) > max_samples:
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            len(first), size=max_samples, replace=False, p=weights / np.sum(weights)
        )
        first, second = first[selected], second[selected]
    first_distance = np.abs(first[:, None] - first[None, :])
    second_distance = np.abs(second[:, None] - second[None, :])
    first_centered = (
        first_distance
        - np.mean(first_distance, axis=0)[None, :]
        - np.mean(first_distance, axis=1)[:, None]
        + np.mean(first_distance)
    )
    second_centered = (
        second_distance
        - np.mean(second_distance, axis=0)[None, :]
        - np.mean(second_distance, axis=1)[:, None]
        + np.mean(second_distance)
    )
    covariance_squared = max(float(np.mean(first_centered * second_centered)), 0.0)
    first_variance_squared = max(float(np.mean(first_centered**2)), 0.0)
    second_variance_squared = max(float(np.mean(second_centered**2)), 0.0)
    denominator = math.sqrt(first_variance_squared * second_variance_squared)
    if denominator <= 0.0:
        return 0.0
    return float(math.sqrt(max(covariance_squared / denominator, 0.0)))


def _probability_histogram(
    values: np.ndarray, edges: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    counts = counts.astype(float)
    total = float(np.sum(counts))
    return counts / total if total > 0.0 else np.zeros(len(edges) - 1, dtype=float)


def _jensen_shannon_nats(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first / np.sum(first) if np.sum(first) > 0.0 else first
    second = second / np.sum(second) if np.sum(second) > 0.0 else second
    middle = 0.5 * (first + second)
    value = 0.0
    positive = first > 0.0
    if np.any(positive):
        value += 0.5 * float(np.sum(first[positive] * np.log(first[positive] / middle[positive])))
    positive = second > 0.0
    if np.any(positive):
        value += 0.5 * float(np.sum(second[positive] * np.log(second[positive] / middle[positive])))
    return max(0.0, value)


def _score_edges(score: np.ndarray, weight: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(
        weighted_quantile(score, np.linspace(0.0, 1.0, n_bins + 1), weight)
    ).astype(float)
    if len(edges) < 2:
        center = float(score[0])
        span = max(abs(center) * 1e-9, 1e-12)
        return np.asarray([center - span, center + span], dtype=float)
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _dependence_group(
    score: np.ndarray,
    energy: np.ndarray,
    weight: np.ndarray,
    energy_edges: np.ndarray,
    threshold: float | None,
    score_bins: int,
    min_per_bin: int,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    valid = np.isfinite(score) & np.isfinite(energy) & np.isfinite(weight) & (weight > 0.0)
    score, energy, weight = score[valid], energy[valid], weight[valid]
    if len(score) < min_per_bin:
        return {
            "status": "not_estimable_insufficient_events",
            "n_valid": int(len(score)),
            "reason": f"need at least {min_per_bin} finite events",
        }
    indices = _bin_indices(energy, energy_edges)
    n_energy_bins = len(energy_edges) - 1
    counts_all = np.bincount(indices, minlength=n_energy_bins).astype(int)
    valid_bins = counts_all >= min_per_bin
    if not np.any(valid_bins):
        return {
            "status": "not_estimable_sparse_energy_bins",
            "n_valid": int(len(score)),
            "reason": f"no fixed energy bin has at least {min_per_bin} events",
            "energy_bin_edges": energy_edges,
            "events_by_fixed_energy_bin": counts_all,
            "energy_bin_valid_mask": valid_bins,
        }
    retained = valid_bins[indices]
    score_edges = _score_edges(score[retained], weight[retained], score_bins)
    pooled = _probability_histogram(score[retained], score_edges, weight[retained])
    centers: list[float] = []
    means: list[float] = []
    acceptances: list[float] = []
    local_weights: list[float] = []
    jsd_values: list[float] = []
    counts: list[int] = []
    kept_indices: list[int] = []
    for index in np.flatnonzero(valid_bins):
        selected = indices == index
        local_weight = weight[selected]
        centers.append(_weighted_mean(energy[selected], local_weight))
        means.append(_weighted_mean(score[selected], local_weight))
        counts.append(int(np.sum(selected)))
        kept_indices.append(int(index))
        local_weights.append(float(np.sum(local_weight)))
        distribution = _probability_histogram(score[selected], score_edges, local_weight)
        jsd_values.append(_jensen_shannon_nats(distribution, pooled))
        if threshold is not None and math.isfinite(float(threshold)):
            acceptances.append(
                _weighted_mean((score[selected] >= threshold).astype(float), local_weight)
            )
    bin_weight = np.asarray(local_weights, dtype=float)
    bin_probability = bin_weight / np.sum(bin_weight)
    jsd = float(np.sum(bin_probability * np.asarray(jsd_values)))
    independence = float(np.clip(1.0 - math.sqrt(jsd / math.log(2.0)), 0.0, 1.0))
    result: dict[str, Any] = {
        "status": "ok",
        "n_valid": int(len(score)),
        "n_retained": int(np.sum(retained)),
        "pearson_abs": abs(_weighted_correlation(score, energy, weight)),
        "spearman_abs": abs(_weighted_spearman(score, energy, weight)),
        "distance_correlation": _distance_correlation(
            score, energy, weight, max_samples, seed
        ),
        "distance_correlation_max_samples": max_samples,
        "conditional_score_jsd_nats": jsd,
        "energy_independence_score": independence,
        "energy_bin_edges": energy_edges,
        "energy_bin_indices": np.asarray(kept_indices, dtype=int),
        "energy_bin_centers": np.asarray(centers, dtype=float),
        "score_mean_by_energy_bin": np.asarray(means, dtype=float),
        "events_by_energy_bin": np.asarray(counts, dtype=int),
        "events_by_fixed_energy_bin": counts_all,
        "energy_bin_valid_mask": valid_bins,
        "weight_fraction_by_energy_bin": bin_probability,
        "score_histogram_edges": score_edges,
    }
    if threshold is not None and math.isfinite(float(threshold)):
        acceptance = np.asarray(acceptances, dtype=float)
        global_acceptance = _weighted_mean((score >= threshold).astype(float), weight)
        difference = acceptance - global_acceptance
        rms = float(np.sqrt(np.sum(bin_probability * difference**2)))
        result.update(
            {
                "threshold": float(threshold),
                "global_acceptance": global_acceptance,
                "acceptance_by_energy_bin": acceptance,
                "acceptance_rms": rms,
                "acceptance_flatness_rms": rms,
                "acceptance_max_abs_deviation": float(np.max(np.abs(difference))),
            }
        )
        selected = score >= threshold
        if np.any(selected):
            before = _probability_histogram(energy, energy_edges, weight)
            after = _probability_histogram(energy[selected], energy_edges, weight[selected])
            sculpting_jsd = _jensen_shannon_nats(before, after)
            result["energy_sculpting_jsd_nats"] = sculpting_jsd
            result["energy_sculpting_distance"] = float(
                math.sqrt(sculpting_jsd / math.log(2.0))
            )
        else:
            result["energy_sculpting_jsd_nats"] = None
            result["energy_sculpting_distance"] = None
    return result


def evaluate_energy_dependence(
    labels: Any,
    scores: Any,
    energies: Any,
    weights: Any | None,
    categories: Any | None,
    threshold: float | None,
    config: Any,
) -> dict[str, Any]:
    """Evaluate class-conditional score/energy dependence on fixed 5 keV bins."""

    label = _as_1d(labels, "labels")
    score = _as_1d(scores, "scores", float)
    energy = _as_1d(energies, "energies", float)
    size = _aligned(("labels", label), ("scores", score), ("energies", energy))
    weight = _weights(weights, size)
    if categories is None:
        group = np.where(label == 1, "signal", "background")
    else:
        category = _as_1d(categories, "categories")
        if len(category) != size:
            raise ValueError("categories must be event-aligned")
        group_text = category.astype(str)
        # Object dtype avoids truncating fallback names when the input contains
        # only short/empty fixed-width strings.
        group = group_text.astype(object)
        # Empty strings are common when custom loaders omit category metadata.
        empty = np.char.strip(group_text) == ""
        group[empty] = np.where(label[empty] == 1, "signal", "background")
    width_kev = float(_config(config, "energy_bin_width_kev", 5.0))
    energy_unit = str(_config(config, "energy_unit", "MeV"))
    finite_observed_energy = energy[np.isfinite(energy)]
    observed_edges = (
        make_fixed_energy_bins(finite_observed_energy, width_kev, energy_unit)
        if finite_observed_energy.size
        else np.asarray([], dtype=float)
    )
    finite = np.isfinite(score) & np.isfinite(energy) & (weight > 0.0)
    if not np.any(finite):
        return {
            "status": "not_estimable",
            "reason": "no finite positive-weight score/energy pairs",
            "overall_energy_independence_score": None,
            "worst_group_energy_independence_score": None,
            "groups": {},
            "protocol": {
                **_fixed_grid_protocol(observed_edges, width_kev, energy_unit),
                "min_per_bin": int(_config(config, "min_per_bin", 20)),
                "score_bins": int(_config(config, "score_bins", 20)),
            },
        }
    score_bins = int(_config(config, "score_bins", 20))
    min_per_bin = int(_config(config, "min_per_bin", 20))
    max_samples = int(_config(config, "distance_correlation_max_samples", 1200))
    seed = int(_config(config, "seed", 42))
    if score_bins < 2 or min_per_bin < 1 or max_samples < 4:
        raise ValueError("score_bins >= 2, min_per_bin >= 1, and max_samples >= 4 are required")
    energy_edges = observed_edges
    groups: dict[str, Any] = {}
    independence_scores: list[float] = []
    group_weights: list[float] = []
    for offset, name in enumerate(np.unique(group)):
        selected = group == name
        metrics = _dependence_group(
            score[selected],
            energy[selected],
            weight[selected],
            energy_edges,
            threshold,
            score_bins,
            min_per_bin,
            max_samples,
            seed + offset * 1009,
        )
        groups[str(name)] = metrics
        if metrics.get("status") == "ok":
            independence_scores.append(float(metrics["energy_independence_score"]))
            valid_group = selected & finite
            group_weights.append(float(np.sum(weight[valid_group])))
    if independence_scores:
        overall = float(np.average(independence_scores, weights=group_weights))
        worst = float(np.min(independence_scores))
        status = "ok"
        reason = None
    else:
        overall = worst = None
        status = "not_estimable"
        reason = "no class-conditional group met the sparse-bin requirements"
    return {
        "status": status,
        "reason": reason,
        "definition": (
            "1 - sqrt(mean_energy_bin_JSD(score_distribution, pooled) / ln(2)); "
            "higher is more class-conditionally energy-independent"
        ),
        "overall_energy_independence_score": overall,
        "worst_group_energy_independence_score": worst,
        "energy_bin_edges": energy_edges,
        "groups": groups,
        "protocol": {
            **_fixed_grid_protocol(energy_edges, width_kev, energy_unit),
            "min_per_bin": min_per_bin,
            "score_bins": score_bins,
        },
    }


def _histogram_with_flow(
    values: np.ndarray, weights: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    counts = np.zeros(len(edges) + 1, dtype=float)
    underflow = values < edges[0]
    overflow = values > edges[-1]
    interior = ~(underflow | overflow)
    counts[0] = float(np.sum(weights[underflow]))
    if np.any(interior):
        inside, _ = np.histogram(values[interior], bins=edges, weights=weights[interior])
        counts[1:-1] = inside.astype(float)
    counts[-1] = float(np.sum(weights[overflow]))
    total = float(np.sum(counts))
    return counts / total if total > 0.0 else counts


def _jensen_shannon_bits(first: np.ndarray, second: np.ndarray) -> float | None:
    first_total, second_total = float(np.sum(first)), float(np.sum(second))
    if first_total <= 0.0 or second_total <= 0.0:
        return None
    first, second = first / first_total, second / second_total
    middle = 0.5 * (first + second)
    value = 0.0
    positive = first > 0.0
    if np.any(positive):
        value += 0.5 * float(np.sum(first[positive] * np.log2(first[positive] / middle[positive])))
    positive = second > 0.0
    if np.any(positive):
        value += 0.5 * float(np.sum(second[positive] * np.log2(second[positive] / middle[positive])))
    return float(np.clip(value, 0.0, 1.0))


def _fractional_floor(truth: np.ndarray, weight: np.ndarray, requested: Any) -> float:
    if requested is not None:
        floor = float(requested)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("fractional_energy_floor must be finite and > 0")
        return floor
    positive = (np.abs(truth) > 0.0) & (weight > 0.0)
    typical = (
        float(weighted_quantile(np.abs(truth[positive]), [0.5], weight[positive])[0])
        if np.any(positive)
        else 0.0
    )
    return max(1e-12, 1e-6 * typical)


def _performance_edges(
    truth: np.ndarray,
    weight: np.ndarray,
    n_bins: int,
    width_kev: float,
    energy_unit: str,
) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("performance_bins must be positive")
    edges = np.unique(
        weighted_quantile(truth, np.linspace(0.0, 1.0, n_bins + 1), weight)
    ).astype(float)
    if len(edges) < 2:
        return make_fixed_energy_bins(truth, width_kev, energy_unit)
    return edges


def _central_resolution(values: np.ndarray, weights: np.ndarray) -> float:
    low, high = weighted_quantile(values, [0.16, 0.84], weights)
    return float(0.5 * (high - low))


def _regression_diagnostics(
    truth: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    edges: np.ndarray,
    floor: float,
    min_per_bin: int,
) -> dict[str, np.ndarray]:
    valid = (weight > 0.0) & np.isfinite(prediction)
    true = truth[valid]
    predicted = prediction[valid]
    local_weight_all = weight[valid]
    residual = predicted - true
    denominator = np.maximum(np.abs(true), floor)
    fractional_residual = residual / denominator
    response = 1.0 + fractional_residual
    n_bins = len(edges) - 1
    centers = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    bin_weights = np.zeros(n_bins, dtype=float)
    response_by_bin = np.full(n_bins, np.nan, dtype=float)
    bias_by_bin = np.full(n_bins, np.nan, dtype=float)
    resolution_by_bin = np.full(n_bins, np.nan, dtype=float)
    fractional_bias_by_bin = np.full(n_bins, np.nan, dtype=float)
    fractional_resolution_by_bin = np.full(n_bins, np.nan, dtype=float)
    mae_by_bin = np.full(n_bins, np.nan, dtype=float)
    rmse_by_bin = np.full(n_bins, np.nan, dtype=float)
    valid_bin = np.zeros(n_bins, dtype=bool)
    for index in range(n_bins):
        selected = _bin_mask(true, edges, index)
        count = int(np.sum(selected))
        counts[index] = count
        if count:
            bin_weights[index] = float(np.sum(local_weight_all[selected]))
        if count < min_per_bin or bin_weights[index] <= 0.0:
            continue
        valid_bin[index] = True
        local_weight = local_weight_all[selected]
        centers[index] = _weighted_mean(true[selected], local_weight)
        response_by_bin[index] = float(weighted_quantile(response[selected], [0.5], local_weight)[0])
        bias_by_bin[index] = float(weighted_quantile(residual[selected], [0.5], local_weight)[0])
        resolution_by_bin[index] = _central_resolution(residual[selected], local_weight)
        fractional_bias_by_bin[index] = float(
            weighted_quantile(fractional_residual[selected], [0.5], local_weight)[0]
        )
        fractional_resolution_by_bin[index] = _central_resolution(
            fractional_residual[selected], local_weight
        )
        mae_by_bin[index] = _weighted_mean(np.abs(residual[selected]), local_weight)
        rmse_by_bin[index] = math.sqrt(
            max(0.0, _weighted_mean(residual[selected] ** 2, local_weight))
        )
    return {
        "energy_bin_edges": edges,
        "energy_bin_centers": centers,
        "energy_bin_counts": counts,
        "energy_bin_weights": bin_weights,
        "energy_bin_valid_mask": valid_bin,
        "response_by_energy_bin": response_by_bin,
        "bias_by_energy_bin": bias_by_bin,
        "median_residual_by_energy_bin": bias_by_bin.copy(),
        "resolution68_by_energy_bin": resolution_by_bin,
        "fractional_bias_by_energy_bin": fractional_bias_by_bin,
        "fractional_resolution_68_by_energy_bin": fractional_resolution_by_bin,
        "mae_by_energy_bin": mae_by_bin,
        "rmse_by_energy_bin": rmse_by_bin,
    }


def evaluate_regression_metrics(
    truth: Any,
    prediction: Any,
    weights: Any | None,
    config: Any,
) -> dict[str, Any]:
    """Evaluate scalar energy regression, including the versioned ERS-v1 score."""

    true = _as_1d(truth, "truth", float)
    predicted = _as_1d(prediction, "prediction", float)
    size = _aligned(("truth", true), ("prediction", predicted))
    weight = _weights(weights, size)
    if not np.all(np.isfinite(true)):
        raise ValueError("truth must contain only finite values")
    width_kev = float(_config(config, "energy_bin_width_kev", 5.0))
    energy_unit = str(_config(config, "energy_unit", "MeV"))
    performance_bins = int(_config(config, "performance_bins", 10))
    min_per_bin = int(_config(config, "min_per_bin", 20))
    floor = _fractional_floor(
        true, weight, _config(config, "fractional_energy_floor", None)
    )
    positive_weight = weight > 0.0
    fixed_edges = make_fixed_energy_bins(true, width_kev, energy_unit)
    performance_edges = _performance_edges(
        true[positive_weight], weight[positive_weight], performance_bins, width_kev, energy_unit
    )
    valid = positive_weight & np.isfinite(predicted)
    total_weight = float(np.sum(weight[positive_weight]))
    finite_weight = float(np.sum(weight[valid]))
    finite_fraction = _safe_fraction(finite_weight, total_weight)
    truth_histogram = _histogram_with_flow(
        true[positive_weight], weight[positive_weight], fixed_edges
    )
    prediction_histogram = (
        _histogram_with_flow(predicted[valid], weight[valid], fixed_edges)
        if np.any(valid)
        else np.zeros(len(fixed_edges) + 1, dtype=float)
    )
    jsd_bits = _jensen_shannon_bits(truth_histogram, prediction_histogram)
    if jsd_bits is None:
        js_distance = None
        hist_similarity = 0.0
        histogram_overlap = None
    else:
        js_distance = float(math.sqrt(jsd_bits))
        hist_similarity = float(np.clip(1.0 - js_distance, 0.0, 1.0))
        histogram_overlap = float(np.sum(np.minimum(truth_histogram, prediction_histogram)))
    diagnostics = _regression_diagnostics(
        true, predicted, weight, fixed_edges, floor, min_per_bin
    )
    base: dict[str, Any] = {
        "score_name": ERS_NAME,
        "score_definition": ERS_DEFINITION,
        "n_total": size,
        "n_valid": int(np.sum(valid)),
        "finite_fraction": finite_fraction,
        "hist_similarity": hist_similarity,
        "histogram_similarity": hist_similarity,
        "jsd_bits": jsd_bits,
        "js_distance": js_distance,
        "histogram_overlap": histogram_overlap,
        "histogram_edges": fixed_edges,
        "truth_histogram": truth_histogram,
        "prediction_histogram": prediction_histogram,
        "true_histogram_probability": truth_histogram,
        "pred_histogram_probability": prediction_histogram,
        "performance_bin_edges": performance_edges,
        "fractional_energy_floor": floor,
        "diagnostic_bins": diagnostics,
        **diagnostics,
        "protocol": {
            **_fixed_grid_protocol(fixed_edges, width_kev, energy_unit),
            "histogram_bins": "fixed_width",
            "diagnostic_bins": "fixed_width",
            "ers_event_bins": "weighted_truth_quantiles",
            "ers_performance_bins_requested": performance_bins,
            "diagnostic_min_per_bin": min_per_bin,
        },
    }
    if not np.any(valid):
        return {
            **base,
            "status": "no_finite_predictions",
            "ers": 0.0,
            "energy_regression_score": 0.0,
            "event_score": 0.0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "r2": None,
            "mae_skill": None,
            "fractional_bias": None,
            "fractional_resolution_68": None,
            "balanced_fractional_mae": None,
            "wasserstein_1": None,
        }

    true_valid = true[valid]
    predicted_valid = predicted[valid]
    weight_valid = weight[valid]
    residual = predicted_valid - true_valid
    absolute_residual = np.abs(residual)
    denominator = np.maximum(np.abs(true_valid), floor)
    fractional_residual = residual / denominator
    absolute_fractional_residual = np.abs(fractional_residual)
    mae = _weighted_mean(absolute_residual, weight_valid)
    rmse = math.sqrt(max(0.0, _weighted_mean(residual**2, weight_valid)))
    bias = _weighted_mean(residual, weight_valid)
    truth_mean = _weighted_mean(true_valid, weight_valid)
    r2_denominator = float(np.sum(weight_valid * (true_valid - truth_mean) ** 2))
    r2_tolerance = (
        np.finfo(float).eps
        * float(np.sum(weight_valid))
        * max(1.0, truth_mean * truth_mean)
    )
    r2 = (
        None
        if r2_denominator <= r2_tolerance
        else float(1.0 - np.sum(weight_valid * residual**2) / r2_denominator)
    )
    baseline = float(weighted_quantile(true_valid, [0.5], weight_valid)[0])
    baseline_mae = _weighted_mean(np.abs(true_valid - baseline), weight_valid)
    mae_skill = (
        None
        if baseline_mae <= np.finfo(float).eps
        else float(1.0 - mae / baseline_mae)
    )
    fractional_bias = float(
        weighted_quantile(fractional_residual, [0.5], weight_valid)[0]
    )
    fractional_resolution = _central_resolution(fractional_residual, weight_valid)
    fractional_mae_by_performance_bin: list[float] = []
    performance_counts: list[int] = []
    for index in range(len(performance_edges) - 1):
        selected = _bin_mask(true_valid, performance_edges, index)
        performance_counts.append(int(np.sum(selected)))
        if np.any(selected):
            fractional_mae_by_performance_bin.append(
                _weighted_mean(absolute_fractional_residual[selected], weight_valid[selected])
            )
    balanced_fractional_mae = (
        float(np.mean(fractional_mae_by_performance_bin))
        if fractional_mae_by_performance_bin
        else None
    )
    event_score = (
        float(np.clip(1.0 - balanced_fractional_mae, 0.0, 1.0))
        if balanced_fractional_mae is not None
        else 0.0
    )
    ers = float(finite_fraction * math.sqrt(max(0.0, event_score * hist_similarity)))
    wasserstein = _wasserstein(
        true[positive_weight], predicted[valid], weight[positive_weight], weight[valid]
    )
    return {
        **base,
        "status": "ok",
        "ers": ers,
        "energy_regression_score": ers,
        "event_score": event_score,
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "r2": r2,
        "mae_skill": mae_skill,
        "fractional_bias": fractional_bias,
        "fractional_resolution_68": fractional_resolution,
        "balanced_fractional_mae": balanced_fractional_mae,
        "fractional_mae_by_performance_bin": np.asarray(
            fractional_mae_by_performance_bin, dtype=float
        ),
        "performance_bin_counts": np.asarray(performance_counts, dtype=int),
        "wasserstein_1": None if not math.isfinite(wasserstein) else float(wasserstein),
    }


__all__ = [
    "evaluate_energy_dependence",
    "evaluate_energy_matched_classification",
    "evaluate_regression_metrics",
    "make_fixed_energy_bins",
    "weighted_quantile",
    "weighted_roc_curve",
]
