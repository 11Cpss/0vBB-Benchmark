"""Pure-NumPy binary ROC and energy-matched ROC evaluation.

The public entry point, :func:`evaluate_energy_matched_roc`, evaluates one
signal/background pair at a time.  It deliberately keeps the ordinary
(``inclusive``) ROC separate from the energy-matched ROC:

* if the two classes have no common energy support, or
* if no energy bin meets the fixed ``min_per_class`` requirement,

then ``matched`` and ``matched_auc`` are ``None``.  The inclusive ROC remains
available, but it is never substituted for the missing matched result.

The default target distribution is the empirical overlap distribution.  In
energy bin ``k`` its mass is

``t_k proportional to min(p_signal,k, p_background,k)``,

where the two ``p`` arrays are class-conditional bin masses after applying the
optional base sample weights.  ``target="legacy_uniform"`` instead gives every
valid bin equal target mass, matching the historical Wing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import (
    effective_sample_size,
    one_dimensional_wasserstein,
    percentile_interval,
    weighted_quantile,
)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _float_array(values: np.ndarray) -> List[Optional[float]]:
    return [_optional_float(value) for value in np.asarray(values).ravel()]


def _label_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class OperatingPoint:
    """First empirical ROC point whose TPR reaches ``target_tpr``."""

    target_tpr: float
    tpr: float
    fpr: float
    threshold: float
    background_rejection_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_tpr": float(self.target_tpr),
            "tpr": float(self.tpr),
            "fpr": float(self.fpr),
            "threshold": _optional_float(self.threshold),
            "background_rejection_fraction": float(
                self.background_rejection_fraction
            ),
        }


@dataclass(frozen=True)
class RocResult:
    """A weighted empirical ROC curve."""

    auc: Optional[float]
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    operating_point: Optional[OperatingPoint]
    n_signal: int
    n_background: int
    signal_weight: float
    background_weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auc": _optional_float(self.auc),
            "fpr": _float_array(self.fpr),
            "tpr": _float_array(self.tpr),
            "thresholds": _float_array(self.thresholds),
            "operating_point": (
                None
                if self.operating_point is None
                else self.operating_point.to_dict()
            ),
            "n_signal": int(self.n_signal),
            "n_background": int(self.n_background),
            "signal_weight": float(self.signal_weight),
            "background_weight": float(self.background_weight),
        }


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile bootstrap interval and bookkeeping."""

    estimate: Optional[float]
    mean: Optional[float]
    standard_deviation: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    confidence_level: float
    n_requested: int
    n_successful: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": _optional_float(self.estimate),
            "mean": _optional_float(self.mean),
            "standard_deviation": _optional_float(self.standard_deviation),
            "lower": _optional_float(self.lower),
            "upper": _optional_float(self.upper),
            "confidence_level": float(self.confidence_level),
            "n_requested": int(self.n_requested),
            "n_successful": int(self.n_successful),
        }


@dataclass(frozen=True)
class CoverageDiagnostics:
    """Event-count and base-weight coverage for the matched estimand."""

    signal_count_total: int
    background_count_total: int
    signal_count_with_energy: int
    background_count_with_energy: int
    signal_count_common_support: int
    background_count_common_support: int
    signal_count_matched: int
    background_count_matched: int
    signal_common_support_count_fraction: float
    background_common_support_count_fraction: float
    signal_matched_count_fraction: float
    background_matched_count_fraction: float
    signal_common_support_weight_fraction: float
    background_common_support_weight_fraction: float
    signal_matched_weight_fraction: float
    background_matched_weight_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: (
                int(value)
                if name.endswith(
                    (
                        "_total",
                        "_with_energy",
                        "_common_support",
                        "_matched",
                    )
                )
                else float(value)
            )
            for name, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class EffectiveSampleSizes:
    signal_inclusive: float
    background_inclusive: float
    signal_matched: float
    background_matched: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "signal_inclusive": float(self.signal_inclusive),
            "background_inclusive": float(self.background_inclusive),
            "signal_matched": float(self.signal_matched),
            "background_matched": float(self.background_matched),
        }


@dataclass(frozen=True)
class BinBalance:
    """Per-bin counts, base masses, and final matched masses."""

    index: int
    low: float
    high: float
    valid: bool
    signal_count: int
    background_count: int
    signal_base_mass: float
    background_base_mass: float
    signal_base_fraction: float
    background_base_fraction: float
    target_mass: float
    signal_matched_mass: float
    background_matched_mass: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "low": float(self.low),
            "high": float(self.high),
            "valid": bool(self.valid),
            "signal_count": int(self.signal_count),
            "background_count": int(self.background_count),
            "signal_base_mass": float(self.signal_base_mass),
            "background_base_mass": float(self.background_base_mass),
            "signal_base_fraction": float(self.signal_base_fraction),
            "background_base_fraction": float(self.background_base_fraction),
            "target_mass": float(self.target_mass),
            "signal_matched_mass": float(self.signal_matched_mass),
            "background_matched_mass": float(self.background_matched_mass),
        }


@dataclass(frozen=True)
class BalanceDiagnostics:
    """Energy-distribution discrepancy before and after matching."""

    wasserstein_before: float
    wasserstein_after: float
    ks_before: float
    ks_after: float
    bin_total_variation_before: float
    bin_total_variation_after: float
    max_bin_mass_difference_after: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "wasserstein_before": float(self.wasserstein_before),
            "wasserstein_after": float(self.wasserstein_after),
            "ks_before": float(self.ks_before),
            "ks_after": float(self.ks_after),
            "bin_total_variation_before": float(
                self.bin_total_variation_before
            ),
            "bin_total_variation_after": float(self.bin_total_variation_after),
            "max_bin_mass_difference_after": float(
                self.max_bin_mass_difference_after
            ),
        }


@dataclass(frozen=True)
class EnergyMatchedRocResult:
    """Complete result for one binary signal/background pair."""

    status: str
    reason: Optional[str]
    positive_label: Any
    negative_label: Any
    target_method: str
    n_bins_requested: int
    n_bins_actual: int
    min_per_class: int
    common_support: Optional[Tuple[float, float]]
    bin_edges: np.ndarray
    bins: Tuple[BinBalance, ...]
    coverage: CoverageDiagnostics
    effective_sample_sizes: EffectiveSampleSizes
    balance: Optional[BalanceDiagnostics]
    inclusive: RocResult
    matched: Optional[RocResult]
    inclusive_auc_ci: Optional[BootstrapInterval]
    matched_auc_ci: Optional[BootstrapInterval]
    matched_mask: np.ndarray
    matched_weights: np.ndarray

    @property
    def inclusive_auc(self) -> Optional[float]:
        return self.inclusive.auc

    @property
    def matched_auc(self) -> Optional[float]:
        return None if self.matched is None else self.matched.auc

    def to_dict(self, include_event_weights: bool = False) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "reason": self.reason,
            "positive_label": _label_value(self.positive_label),
            "negative_label": _label_value(self.negative_label),
            "target_method": self.target_method,
            "n_bins_requested": int(self.n_bins_requested),
            "n_bins_actual": int(self.n_bins_actual),
            "min_per_class": int(self.min_per_class),
            "common_support": (
                None
                if self.common_support is None
                else [float(self.common_support[0]), float(self.common_support[1])]
            ),
            "bin_edges": _float_array(self.bin_edges),
            "bins": [item.to_dict() for item in self.bins],
            "coverage": self.coverage.to_dict(),
            "effective_sample_sizes": self.effective_sample_sizes.to_dict(),
            "balance": None if self.balance is None else self.balance.to_dict(),
            "inclusive": self.inclusive.to_dict(),
            "matched": None if self.matched is None else self.matched.to_dict(),
            "inclusive_auc": _optional_float(self.inclusive_auc),
            "matched_auc": _optional_float(self.matched_auc),
            "inclusive_auc_ci": (
                None
                if self.inclusive_auc_ci is None
                else self.inclusive_auc_ci.to_dict()
            ),
            "matched_auc_ci": (
                None
                if self.matched_auc_ci is None
                else self.matched_auc_ci.to_dict()
            ),
        }
        if include_event_weights:
            payload["matched_mask"] = self.matched_mask.astype(bool).tolist()
            payload["matched_weights"] = _float_array(self.matched_weights)
        return payload


def _one_dimensional(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            "%s must be one-dimensional; received shape %s"
            % (name, array.shape)
        )
    return array


def _validate_equal_lengths(arrays: Sequence[Tuple[str, np.ndarray]]) -> int:
    lengths = {name: len(values) for name, values in arrays}
    if len(set(lengths.values())) != 1:
        raise ValueError("all input arrays must have equal length: %s" % lengths)
    size = next(iter(lengths.values())) if lengths else 0
    if size == 0:
        raise ValueError("input arrays must not be empty")
    return int(size)


def _validate_weights(sample_weight: Optional[Any], size: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(size, dtype=float)
    weights = _one_dimensional(
        "sample_weight", np.asarray(sample_weight, dtype=float)
    )
    _validate_equal_lengths(
        (("sample_weight", weights), ("reference", np.empty(size)))
    )
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(weights < 0):
        raise ValueError("sample_weight must be non-negative")
    return weights


def _weighted_ks(
    first: np.ndarray,
    second: np.ndarray,
    first_weights: np.ndarray,
    second_weights: np.ndarray,
) -> float:
    first_order = np.argsort(first, kind="mergesort")
    second_order = np.argsort(second, kind="mergesort")
    first = first[first_order]
    second = second[second_order]
    first_weights = first_weights[first_order]
    second_weights = second_weights[second_order]
    first_weights = first_weights / np.sum(first_weights)
    second_weights = second_weights / np.sum(second_weights)
    points = np.unique(np.concatenate((first, second)))
    first_cumulative = np.r_[0.0, np.cumsum(first_weights)]
    second_cumulative = np.r_[0.0, np.cumsum(second_weights)]
    first_indices = np.searchsorted(first, points, side="right")
    second_indices = np.searchsorted(second, points, side="right")
    return float(
        np.max(
            np.abs(
                first_cumulative[first_indices]
                - second_cumulative[second_indices]
            )
        )
    )


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _roc_from_binary(
    positive: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
    target_tpr: float,
) -> RocResult:
    mask = np.isfinite(score) & np.isfinite(weights) & (weights > 0)
    positive = np.asarray(positive, dtype=bool)[mask]
    score = np.asarray(score, dtype=float)[mask]
    weights = np.asarray(weights, dtype=float)[mask]
    signal_weight = float(np.sum(weights[positive]))
    background_weight = float(np.sum(weights[~positive]))
    n_signal = int(np.sum(positive))
    n_background = int(np.sum(~positive))
    if (
        n_signal == 0
        or n_background == 0
        or signal_weight <= 0
        or background_weight <= 0
    ):
        return RocResult(
            auc=None,
            fpr=np.array([], dtype=float),
            tpr=np.array([], dtype=float),
            thresholds=np.array([], dtype=float),
            operating_point=None,
            n_signal=n_signal,
            n_background=n_background,
            signal_weight=signal_weight,
            background_weight=background_weight,
        )

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_positive = positive[order]
    sorted_weight = weights[order]
    fpr = [0.0]
    tpr = [0.0]
    thresholds = [float("inf")]
    true_mass = 0.0
    false_mass = 0.0
    start = 0
    while start < len(sorted_score):
        stop = start + 1
        while (
            stop < len(sorted_score)
            and sorted_score[stop] == sorted_score[start]
        ):
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
    # ``np.trapz`` was removed in NumPy 2.4; ``trapezoid`` is available in
    # modern NumPy while the fallback keeps the project compatible with its
    # Python 3.8 / NumPy 1.x baseline.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    auc = float(trapezoid(tpr_array, fpr_array))
    auc = float(np.clip(auc, 0.0, 1.0))
    reached = np.flatnonzero(tpr_array >= target_tpr)
    point = None
    if reached.size:
        index = int(reached[0])
        point = OperatingPoint(
            target_tpr=float(target_tpr),
            tpr=float(tpr_array[index]),
            fpr=float(fpr_array[index]),
            threshold=float(threshold_array[index]),
            background_rejection_fraction=float(1.0 - fpr_array[index]),
        )
    return RocResult(
        auc=auc,
        fpr=fpr_array,
        tpr=tpr_array,
        thresholds=threshold_array,
        operating_point=point,
        n_signal=n_signal,
        n_background=n_background,
        signal_weight=signal_weight,
        background_weight=background_weight,
    )


def weighted_roc_curve(
    y_true: Any,
    score: Any,
    sample_weight: Optional[Any] = None,
    positive_label: Any = 1,
    target_tpr: float = 0.90,
) -> RocResult:
    """Compute a weighted binary ROC with exact tie handling."""

    labels = _one_dimensional("y_true", y_true)
    scores = _one_dimensional("score", np.asarray(score, dtype=float))
    size = _validate_equal_lengths((("y_true", labels), ("score", scores)))
    weights = _validate_weights(sample_weight, size)
    if not 0.0 <= float(target_tpr) <= 1.0:
        raise ValueError("target_tpr must lie in [0, 1]")
    valid = np.isfinite(scores) & (weights > 0)
    positive = labels == positive_label
    if not np.any(valid & positive) or not np.any(valid & ~positive):
        return _roc_from_binary(
            positive[valid], scores[valid], weights[valid], float(target_tpr)
        )
    return _roc_from_binary(
        positive[valid], scores[valid], weights[valid], float(target_tpr)
    )


def _coverage(
    positive: np.ndarray,
    inclusive_mask: np.ndarray,
    energy_mask: np.ndarray,
    common_mask: np.ndarray,
    matched_mask: np.ndarray,
    base_weights: np.ndarray,
) -> CoverageDiagnostics:
    signal_total = int(np.sum(inclusive_mask & positive))
    background_total = int(np.sum(inclusive_mask & ~positive))
    signal_energy = int(np.sum(energy_mask & positive))
    background_energy = int(np.sum(energy_mask & ~positive))
    signal_common = int(np.sum(common_mask & positive))
    background_common = int(np.sum(common_mask & ~positive))
    signal_matched = int(np.sum(matched_mask & positive))
    background_matched = int(np.sum(matched_mask & ~positive))

    signal_energy_mass = float(np.sum(base_weights[energy_mask & positive]))
    background_energy_mass = float(
        np.sum(base_weights[energy_mask & ~positive])
    )
    signal_common_mass = float(np.sum(base_weights[common_mask & positive]))
    background_common_mass = float(
        np.sum(base_weights[common_mask & ~positive])
    )
    signal_matched_mass = float(
        np.sum(base_weights[matched_mask & positive])
    )
    background_matched_mass = float(
        np.sum(base_weights[matched_mask & ~positive])
    )
    return CoverageDiagnostics(
        signal_count_total=signal_total,
        background_count_total=background_total,
        signal_count_with_energy=signal_energy,
        background_count_with_energy=background_energy,
        signal_count_common_support=signal_common,
        background_count_common_support=background_common,
        signal_count_matched=signal_matched,
        background_count_matched=background_matched,
        signal_common_support_count_fraction=_safe_fraction(
            signal_common, signal_energy
        ),
        background_common_support_count_fraction=_safe_fraction(
            background_common, background_energy
        ),
        signal_matched_count_fraction=_safe_fraction(
            signal_matched, signal_energy
        ),
        background_matched_count_fraction=_safe_fraction(
            background_matched, background_energy
        ),
        signal_common_support_weight_fraction=_safe_fraction(
            signal_common_mass, signal_energy_mass
        ),
        background_common_support_weight_fraction=_safe_fraction(
            background_common_mass, background_energy_mass
        ),
        signal_matched_weight_fraction=_safe_fraction(
            signal_matched_mass, signal_energy_mass
        ),
        background_matched_weight_fraction=_safe_fraction(
            background_matched_mass, background_energy_mass
        ),
    )


def _actual_edges(
    energy: np.ndarray,
    positive: np.ndarray,
    common_mask: np.ndarray,
    base_weights: np.ndarray,
    n_bins: int,
    low: float,
    high: float,
) -> np.ndarray:
    if low == high:
        return np.asarray([low, high], dtype=float)
    unique_energy = np.unique(energy[common_mask])
    if unique_energy.size <= n_bins:
        # Preserve exact discrete energy levels.  Pooled quantiles can collapse
        # several highly populated levels into one coarse bin, leaving an
        # avoidable energy shortcut inside that bin.
        midpoints = unique_energy[:-1] + 0.5 * np.diff(unique_energy)
        return np.r_[low, midpoints, high].astype(float)
    signal_mass = float(np.sum(base_weights[common_mask & positive]))
    background_mass = float(np.sum(base_weights[common_mask & ~positive]))
    edge_weights = np.zeros(len(energy), dtype=float)
    edge_weights[common_mask & positive] = (
        0.5 * base_weights[common_mask & positive] / signal_mass
    )
    edge_weights[common_mask & ~positive] = (
        0.5 * base_weights[common_mask & ~positive] / background_mass
    )
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = weighted_quantile(
        energy[common_mask], quantiles, edge_weights[common_mask]
    )
    edges = np.unique(np.asarray(edges, dtype=float))
    if edges.size < 2:
        return np.asarray([low, high], dtype=float)
    edges[0] = low
    edges[-1] = high
    return edges


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=int)
    return np.searchsorted(edges[1:-1], values, side="right").astype(int)


def _empty_bin_rows(
    edges: np.ndarray,
    signal_counts: np.ndarray,
    background_counts: np.ndarray,
    signal_masses: np.ndarray,
    background_masses: np.ndarray,
    valid_bins: np.ndarray,
) -> Tuple[BinBalance, ...]:
    signal_total = float(np.sum(signal_masses))
    background_total = float(np.sum(background_masses))
    rows = []
    for index in range(len(signal_counts)):
        rows.append(
            BinBalance(
                index=index,
                low=float(edges[index]),
                high=float(edges[index + 1]),
                valid=bool(valid_bins[index]),
                signal_count=int(signal_counts[index]),
                background_count=int(background_counts[index]),
                signal_base_mass=float(signal_masses[index]),
                background_base_mass=float(background_masses[index]),
                signal_base_fraction=_safe_fraction(
                    signal_masses[index], signal_total
                ),
                background_base_fraction=_safe_fraction(
                    background_masses[index], background_total
                ),
                target_mass=0.0,
                signal_matched_mass=0.0,
                background_matched_mass=0.0,
            )
        )
    return tuple(rows)


def _unavailable_result(
    *,
    status: str,
    reason: str,
    positive_label: Any,
    negative_label: Any,
    target_method: str,
    n_bins: int,
    min_per_class: int,
    common_support: Optional[Tuple[float, float]],
    bin_edges: np.ndarray,
    bins: Tuple[BinBalance, ...],
    positive: np.ndarray,
    inclusive_mask: np.ndarray,
    energy_mask: np.ndarray,
    common_mask: np.ndarray,
    base_weights: np.ndarray,
    inclusive: RocResult,
) -> EnergyMatchedRocResult:
    zeros = np.zeros(len(positive), dtype=float)
    matched_mask = np.zeros(len(positive), dtype=bool)
    return EnergyMatchedRocResult(
        status=status,
        reason=reason,
        positive_label=positive_label,
        negative_label=negative_label,
        target_method=target_method,
        n_bins_requested=n_bins,
        n_bins_actual=max(len(bin_edges) - 1, 0),
        min_per_class=min_per_class,
        common_support=common_support,
        bin_edges=bin_edges,
        bins=bins,
        coverage=_coverage(
            positive,
            inclusive_mask,
            energy_mask,
            common_mask,
            matched_mask,
            base_weights,
        ),
        effective_sample_sizes=EffectiveSampleSizes(
            signal_inclusive=effective_sample_size(
                base_weights[inclusive_mask & positive]
            ),
            background_inclusive=effective_sample_size(
                base_weights[inclusive_mask & ~positive]
            ),
            signal_matched=0.0,
            background_matched=0.0,
        ),
        balance=None,
        inclusive=inclusive,
        matched=None,
        inclusive_auc_ci=None,
        matched_auc_ci=None,
        matched_mask=matched_mask,
        matched_weights=zeros,
    )


def _evaluate_core(
    y_true: np.ndarray,
    score: np.ndarray,
    energy: np.ndarray,
    base_weights: np.ndarray,
    positive_label: Any,
    n_bins: int,
    min_per_class: int,
    target_method: str,
    target_tpr: float,
    support_trim_quantile: float,
    energy_roi: Optional[Tuple[float, float]],
) -> EnergyMatchedRocResult:
    inclusive_mask = np.isfinite(score) & (base_weights > 0)
    positive = y_true == positive_label
    if not np.any(inclusive_mask & positive):
        raise ValueError("positive_label has no finite, positive-weight scores")
    if not np.any(inclusive_mask & ~positive):
        raise ValueError("background class has no finite, positive-weight scores")
    negative_values = np.unique(y_true[inclusive_mask & ~positive])
    if len(negative_values) != 1:
        raise ValueError(
            "one pair must contain exactly two labels; found %d background labels"
            % len(negative_values)
        )
    negative_label = _label_value(negative_values[0])
    inclusive = _roc_from_binary(
        positive[inclusive_mask],
        score[inclusive_mask],
        base_weights[inclusive_mask],
        target_tpr,
    )

    energy_mask = inclusive_mask & np.isfinite(energy)
    empty = np.array([], dtype=float)
    if (
        np.sum(energy_mask & positive) < min_per_class
        or np.sum(energy_mask & ~positive) < min_per_class
    ):
        return _unavailable_result(
            status="insufficient_statistics",
            reason=(
                "fewer than min_per_class finite-energy events in at least "
                "one class"
            ),
            positive_label=positive_label,
            negative_label=negative_label,
            target_method=target_method,
            n_bins=n_bins,
            min_per_class=min_per_class,
            common_support=None,
            bin_edges=empty,
            bins=tuple(),
            positive=positive,
            inclusive_mask=inclusive_mask,
            energy_mask=energy_mask,
            common_mask=np.zeros(len(positive), dtype=bool),
            base_weights=base_weights,
            inclusive=inclusive,
        )

    signal_energy = energy[energy_mask & positive]
    background_energy = energy[energy_mask & ~positive]
    signal_energy_weights = base_weights[energy_mask & positive]
    background_energy_weights = base_weights[energy_mask & ~positive]
    if support_trim_quantile > 0:
        signal_range = weighted_quantile(
            signal_energy,
            [support_trim_quantile, 1.0 - support_trim_quantile],
            signal_energy_weights,
        )
        background_range = weighted_quantile(
            background_energy,
            [support_trim_quantile, 1.0 - support_trim_quantile],
            background_energy_weights,
        )
        support_low = max(float(signal_range[0]), float(background_range[0]))
        support_high = min(float(signal_range[1]), float(background_range[1]))
    else:
        support_low = max(
            float(np.min(signal_energy)), float(np.min(background_energy))
        )
        support_high = min(
            float(np.max(signal_energy)), float(np.max(background_energy))
        )
    if energy_roi is not None:
        support_low = max(support_low, float(energy_roi[0]))
        support_high = min(support_high, float(energy_roi[1]))
    if support_high < support_low:
        return _unavailable_result(
            status="no_common_support",
            reason="signal and background energy ranges do not overlap",
            positive_label=positive_label,
            negative_label=negative_label,
            target_method=target_method,
            n_bins=n_bins,
            min_per_class=min_per_class,
            common_support=None,
            bin_edges=empty,
            bins=tuple(),
            positive=positive,
            inclusive_mask=inclusive_mask,
            energy_mask=energy_mask,
            common_mask=np.zeros(len(positive), dtype=bool),
            base_weights=base_weights,
            inclusive=inclusive,
        )

    common_mask = (
        energy_mask & (energy >= support_low) & (energy <= support_high)
    )
    common_support = (support_low, support_high)
    edges = _actual_edges(
        energy,
        positive,
        common_mask,
        base_weights,
        n_bins,
        support_low,
        support_high,
    )
    n_actual = max(len(edges) - 1, 1)
    common_indices = np.flatnonzero(common_mask)
    common_bins = _bin_indices(energy[common_mask], edges)
    signal_counts = np.bincount(
        common_bins[positive[common_mask]], minlength=n_actual
    )
    background_counts = np.bincount(
        common_bins[~positive[common_mask]], minlength=n_actual
    )
    signal_masses = np.bincount(
        common_bins[positive[common_mask]],
        weights=base_weights[common_mask & positive],
        minlength=n_actual,
    )
    background_masses = np.bincount(
        common_bins[~positive[common_mask]],
        weights=base_weights[common_mask & ~positive],
        minlength=n_actual,
    )
    valid_bins = (
        (signal_counts >= min_per_class)
        & (background_counts >= min_per_class)
        & (signal_masses > 0)
        & (background_masses > 0)
    )
    if not np.any(valid_bins):
        return _unavailable_result(
            status="insufficient_statistics",
            reason=(
                "no energy bin contains min_per_class events from both classes"
            ),
            positive_label=positive_label,
            negative_label=negative_label,
            target_method=target_method,
            n_bins=n_bins,
            min_per_class=min_per_class,
            common_support=common_support,
            bin_edges=edges,
            bins=_empty_bin_rows(
                edges,
                signal_counts,
                background_counts,
                signal_masses,
                background_masses,
                valid_bins,
            ),
            positive=positive,
            inclusive_mask=inclusive_mask,
            energy_mask=energy_mask,
            common_mask=common_mask,
            base_weights=base_weights,
            inclusive=inclusive,
        )

    signal_fractions = signal_masses / np.sum(signal_masses)
    background_fractions = background_masses / np.sum(background_masses)
    if target_method == "overlap":
        target_mass = np.minimum(signal_fractions, background_fractions)
        target_mass[~valid_bins] = 0.0
    else:
        target_mass = valid_bins.astype(float)
    target_sum = float(np.sum(target_mass))
    if target_sum <= 0:
        return _unavailable_result(
            status="insufficient_statistics",
            reason="valid bins have zero overlap target mass",
            positive_label=positive_label,
            negative_label=negative_label,
            target_method=target_method,
            n_bins=n_bins,
            min_per_class=min_per_class,
            common_support=common_support,
            bin_edges=edges,
            bins=_empty_bin_rows(
                edges,
                signal_counts,
                background_counts,
                signal_masses,
                background_masses,
                valid_bins,
            ),
            positive=positive,
            inclusive_mask=inclusive_mask,
            energy_mask=energy_mask,
            common_mask=common_mask,
            base_weights=base_weights,
            inclusive=inclusive,
        )
    target_mass = target_mass / target_sum

    matched_weights = np.zeros(len(score), dtype=float)
    bin_for_event = np.full(len(score), -1, dtype=int)
    bin_for_event[common_indices] = common_bins
    for index in np.flatnonzero(valid_bins):
        signal_bin = common_mask & positive & (bin_for_event == index)
        background_bin = common_mask & ~positive & (bin_for_event == index)
        matched_weights[signal_bin] = (
            base_weights[signal_bin] * target_mass[index] / signal_masses[index]
        )
        matched_weights[background_bin] = (
            base_weights[background_bin]
            * target_mass[index]
            / background_masses[index]
        )
    matched_mask = matched_weights > 0
    matched = _roc_from_binary(
        positive[matched_mask],
        score[matched_mask],
        matched_weights[matched_mask],
        target_tpr,
    )

    signal_matched_mass = np.bincount(
        bin_for_event[matched_mask & positive],
        weights=matched_weights[matched_mask & positive],
        minlength=n_actual,
    )
    background_matched_mass = np.bincount(
        bin_for_event[matched_mask & ~positive],
        weights=matched_weights[matched_mask & ~positive],
        minlength=n_actual,
    )
    bins: List[BinBalance] = []
    for index in range(n_actual):
        bins.append(
            BinBalance(
                index=index,
                low=float(edges[index]),
                high=float(edges[index + 1]),
                valid=bool(valid_bins[index]),
                signal_count=int(signal_counts[index]),
                background_count=int(background_counts[index]),
                signal_base_mass=float(signal_masses[index]),
                background_base_mass=float(background_masses[index]),
                signal_base_fraction=float(signal_fractions[index]),
                background_base_fraction=float(background_fractions[index]),
                target_mass=float(target_mass[index]),
                signal_matched_mass=float(signal_matched_mass[index]),
                background_matched_mass=float(background_matched_mass[index]),
            )
        )

    before_tv = 0.5 * float(
        np.sum(np.abs(signal_fractions - background_fractions))
    )
    after_tv = 0.5 * float(
        np.sum(np.abs(signal_matched_mass - background_matched_mass))
    )
    balance = BalanceDiagnostics(
        wasserstein_before=one_dimensional_wasserstein(
            energy[common_mask & positive],
            energy[common_mask & ~positive],
            base_weights[common_mask & positive],
            base_weights[common_mask & ~positive],
        ),
        wasserstein_after=one_dimensional_wasserstein(
            energy[matched_mask & positive],
            energy[matched_mask & ~positive],
            matched_weights[matched_mask & positive],
            matched_weights[matched_mask & ~positive],
        ),
        ks_before=_weighted_ks(
            energy[common_mask & positive],
            energy[common_mask & ~positive],
            base_weights[common_mask & positive],
            base_weights[common_mask & ~positive],
        ),
        ks_after=_weighted_ks(
            energy[matched_mask & positive],
            energy[matched_mask & ~positive],
            matched_weights[matched_mask & positive],
            matched_weights[matched_mask & ~positive],
        ),
        bin_total_variation_before=before_tv,
        bin_total_variation_after=after_tv,
        max_bin_mass_difference_after=float(
            np.max(np.abs(signal_matched_mass - background_matched_mass))
        ),
    )
    return EnergyMatchedRocResult(
        status="ok",
        reason=None,
        positive_label=_label_value(positive_label),
        negative_label=negative_label,
        target_method=target_method,
        n_bins_requested=n_bins,
        n_bins_actual=n_actual,
        min_per_class=min_per_class,
        common_support=common_support,
        bin_edges=edges,
        bins=tuple(bins),
        coverage=_coverage(
            positive,
            inclusive_mask,
            energy_mask,
            common_mask,
            matched_mask,
            base_weights,
        ),
        effective_sample_sizes=EffectiveSampleSizes(
            signal_inclusive=effective_sample_size(
                base_weights[inclusive_mask & positive]
            ),
            background_inclusive=effective_sample_size(
                base_weights[inclusive_mask & ~positive]
            ),
            signal_matched=effective_sample_size(
                matched_weights[matched_mask & positive]
            ),
            background_matched=effective_sample_size(
                matched_weights[matched_mask & ~positive]
            ),
        ),
        balance=balance,
        inclusive=inclusive,
        matched=matched,
        inclusive_auc_ci=None,
        matched_auc_ci=None,
        matched_mask=matched_mask,
        matched_weights=matched_weights,
    )


def _bootstrap_interval(
    estimate: Optional[float],
    values: Sequence[float],
    n_requested: int,
    confidence_level: float,
) -> BootstrapInterval:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lower, upper = percentile_interval(finite, confidence_level)
        mean = float(np.mean(finite))
        standard_deviation = float(np.std(finite))
    else:
        lower = upper = mean = standard_deviation = None
    return BootstrapInterval(
        estimate=_optional_float(estimate),
        mean=_optional_float(mean),
        standard_deviation=_optional_float(standard_deviation),
        lower=_optional_float(lower),
        upper=_optional_float(upper),
        confidence_level=confidence_level,
        n_requested=n_requested,
        n_successful=int(finite.size),
    )


def evaluate_energy_matched_roc(
    y_true: Any,
    score: Any,
    energy: Any,
    sample_weight: Optional[Any] = None,
    *,
    positive_label: Any = 1,
    n_bins: int = 10,
    min_per_class: int = 5,
    target: str = "overlap",
    target_tpr: float = 0.90,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 0,
    support_trim_quantile: float = 0.0,
    energy_roi: Optional[Tuple[float, float]] = None,
) -> EnergyMatchedRocResult:
    """Evaluate inclusive and energy-matched ROC for one binary pair.

    Parameters
    ----------
    y_true, score, energy:
        Strictly one-dimensional, equal-length event arrays.
    sample_weight:
        Optional non-negative base event weights.  Matching multiplies these
        weights rather than replacing them.
    positive_label:
        Label treated as signal.  Exactly one other label must be present.
    n_bins:
        Requested number of class-balanced pooled energy-quantile bins.
        Repeated quantile edges may reduce the actual number of bins.
    min_per_class:
        Fixed minimum positive-weight event count in every retained bin.  It
        is never relaxed automatically.
    target:
        ``"overlap"`` (default) for
        ``t_k ∝ min(p_signal,k, p_background,k)`` or
        ``"legacy_uniform"`` for equal mass in every valid bin.
    n_bootstrap:
        Number of class-stratified bootstrap samples.  Common support, bin
        edges, valid bins, and matching weights are recomputed inside every
        replicate.
    support_trim_quantile:
        Optional symmetric weighted-quantile trimming applied independently
        to each class before intersecting their energy support.  For example,
        ``0.005`` uses each class's central 99% range and is less sensitive to
        a single extreme event than sample minima/maxima.
    energy_roi:
        Optional predeclared physical ``(low, high)`` interval.  It intersects
        the empirical common support and is held fixed across bootstrap
        replicates.
    """

    labels = _one_dimensional("y_true", y_true)
    scores = _one_dimensional("score", np.asarray(score, dtype=float))
    energies = _one_dimensional("energy", np.asarray(energy, dtype=float))
    size = _validate_equal_lengths(
        (("y_true", labels), ("score", scores), ("energy", energies))
    )
    weights = _validate_weights(sample_weight, size)
    if int(n_bins) != n_bins or int(n_bins) < 1:
        raise ValueError("n_bins must be a positive integer")
    if int(min_per_class) != min_per_class or int(min_per_class) < 1:
        raise ValueError("min_per_class must be a positive integer")
    target_method = str(target).strip().lower()
    if target_method not in {"overlap", "legacy_uniform"}:
        raise ValueError("target must be 'overlap' or 'legacy_uniform'")
    if not 0.0 <= float(target_tpr) <= 1.0:
        raise ValueError("target_tpr must lie in [0, 1]")
    if int(n_bootstrap) != n_bootstrap or int(n_bootstrap) < 0:
        raise ValueError("n_bootstrap must be a non-negative integer")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if not 0.0 <= float(support_trim_quantile) < 0.5:
        raise ValueError("support_trim_quantile must lie in [0, 0.5)")
    resolved_roi = None
    if energy_roi is not None:
        if len(energy_roi) != 2:
            raise ValueError("energy_roi must contain exactly (low, high)")
        resolved_roi = (float(energy_roi[0]), float(energy_roi[1]))
        if (
            not np.all(np.isfinite(resolved_roi))
            or resolved_roi[1] <= resolved_roi[0]
        ):
            raise ValueError("energy_roi must be finite with high > low")

    result = _evaluate_core(
        labels,
        scores,
        energies,
        weights,
        positive_label,
        int(n_bins),
        int(min_per_class),
        target_method,
        float(target_tpr),
        float(support_trim_quantile),
        resolved_roi,
    )
    if int(n_bootstrap) == 0:
        return result

    positive = labels == positive_label
    resample_mask = np.isfinite(scores) & (weights > 0)
    signal_indices = np.flatnonzero(resample_mask & positive)
    background_indices = np.flatnonzero(resample_mask & ~positive)
    rng = np.random.default_rng(random_state)
    inclusive_aucs: List[float] = []
    matched_aucs: List[float] = []
    for _ in range(int(n_bootstrap)):
        signal_sample = rng.choice(
            signal_indices, size=len(signal_indices), replace=True
        )
        background_sample = rng.choice(
            background_indices, size=len(background_indices), replace=True
        )
        indices = np.concatenate((signal_sample, background_sample))
        replicate = _evaluate_core(
            labels[indices],
            scores[indices],
            energies[indices],
            weights[indices],
            positive_label,
            int(n_bins),
            int(min_per_class),
            target_method,
            float(target_tpr),
            float(support_trim_quantile),
            resolved_roi,
        )
        if replicate.inclusive_auc is not None:
            inclusive_aucs.append(float(replicate.inclusive_auc))
        if replicate.matched_auc is not None:
            matched_aucs.append(float(replicate.matched_auc))

    return replace(
        result,
        inclusive_auc_ci=_bootstrap_interval(
            result.inclusive_auc,
            inclusive_aucs,
            int(n_bootstrap),
            float(confidence_level),
        ),
        matched_auc_ci=_bootstrap_interval(
            result.matched_auc,
            matched_aucs,
            int(n_bootstrap),
            float(confidence_level),
        ),
    )


__all__ = [
    "BalanceDiagnostics",
    "BinBalance",
    "BootstrapInterval",
    "CoverageDiagnostics",
    "EffectiveSampleSizes",
    "EnergyMatchedRocResult",
    "OperatingPoint",
    "RocResult",
    "evaluate_energy_matched_roc",
    "weighted_roc_curve",
]
