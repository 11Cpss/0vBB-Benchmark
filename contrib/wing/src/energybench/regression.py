"""Energy-regression metrics and the versioned ERS-v1 score.

The module intentionally consumes arrays rather than model objects.  A model
only needs to export one predicted energy per event.

ERS-v1 combines two complementary requirements:

``event_score = max(0, 1 - balanced_fractional_mae)``
    Measures event-by-event agreement, giving equal importance to populated
    true-energy bins.

``hist_similarity = 1 - sqrt(JSD_2(p_true, p_pred))``
    Measures agreement of the weighted true and predicted energy spectra on
    shared, truth-derived histogram edges.  The first and last histogram
    entries are explicit underflow and overflow bins.

``ERS-v1 = finite_fraction * sqrt(event_score * hist_similarity)``
    A shuffled prediction can preserve the complete energy histogram, but it
    cannot receive a high score because the eventwise factor is also required.

All calculations in this file use NumPy only.  ``sample_weight`` must be
finite and non-negative.  Truth values must be finite.  Non-finite predictions
are retained as failed predictions: they reduce ``finite_fraction`` instead of
being silently discarded.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .utils import json_ready, one_dimensional_wasserstein, weighted_quantile


SCORE_NAME = "ERS-v1"
SCORE_DEFINITION = (
    "finite_fraction * sqrt(event_score * hist_similarity), where "
    "event_score=max(0, 1-balanced_fractional_mae) and "
    "hist_similarity=1-sqrt(JSD_base2(true_hist, predicted_hist))"
)


@dataclass
class EnergyRegressionResult:
    """Complete scalar and plot-ready energy-regression evaluation."""

    status: str
    score_name: str
    score_definition: str
    n_total: int
    n_valid: int
    finite_fraction: float

    energy_regression_score: float
    event_score: float
    hist_similarity: float

    mae: Optional[float]
    rmse: Optional[float]
    bias: Optional[float]
    r2: Optional[float]
    mae_skill: Optional[float]
    fractional_bias: Optional[float]
    fractional_resolution_68: Optional[float]
    balanced_fractional_mae: Optional[float]
    fractional_energy_floor: float

    jsd_bits: Optional[float]
    js_distance: Optional[float]
    histogram_overlap: Optional[float]
    wasserstein_1: Optional[float]

    histogram_edges: np.ndarray
    truth_histogram: np.ndarray
    prediction_histogram: np.ndarray

    energy_bin_edges: np.ndarray
    energy_bin_centers: np.ndarray
    energy_bin_counts: np.ndarray
    energy_bin_weights: np.ndarray
    response_by_energy_bin: np.ndarray
    bias_by_energy_bin: np.ndarray
    resolution68_by_energy_bin: np.ndarray
    fractional_bias_by_energy_bin: np.ndarray
    fractional_resolution_68_by_energy_bin: np.ndarray
    mae_by_energy_bin: np.ndarray
    rmse_by_energy_bin: np.ndarray

    bootstrap: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Short alias for ``energy_regression_score``."""

        return self.energy_regression_score

    @property
    def histogram_similarity(self) -> float:
        """Compatibility alias used by plotting/reporting consumers."""

        return self.hist_similarity

    @property
    def true_histogram_probability(self) -> np.ndarray:
        """Truth probability masses, including underflow and overflow."""

        return self.truth_histogram

    @property
    def pred_histogram_probability(self) -> np.ndarray:
        """Prediction probability masses, including underflow and overflow."""

        return self.prediction_histogram

    def to_dict(self) -> Dict[str, Any]:
        """Return a strict JSON-compatible representation."""

        payload = asdict(self)
        # Keep concise internal names while making the serialized artifact
        # self-explanatory and directly consumable by the plotting module.
        payload.update(
            {
                "histogram_similarity": self.histogram_similarity,
                "true_histogram_probability": self.truth_histogram,
                "pred_histogram_probability": self.prediction_histogram,
            }
        )
        return json_ready(payload)


def _as_float_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(
            "%s must be one-dimensional; received shape %s"
            % (name, array.shape)
        )
    return array


def _validate_inputs(
    truth: Any,
    prediction: Any,
    sample_weight: Optional[Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = _as_float_1d(truth, "truth")
    predicted = _as_float_1d(prediction, "prediction")
    if len(true) != len(predicted):
        raise ValueError(
            "truth and prediction lengths differ: %d != %d"
            % (len(true), len(predicted))
        )
    if len(true) == 0:
        raise ValueError("truth and prediction must contain at least one event")
    if not np.all(np.isfinite(true)):
        bad = int(np.sum(~np.isfinite(true)))
        raise ValueError("truth contains %d non-finite value(s)" % bad)

    if sample_weight is None:
        weight = np.ones(len(true), dtype=float)
    else:
        weight = _as_float_1d(sample_weight, "sample_weight")
        if len(weight) != len(true):
            raise ValueError(
                "sample_weight length differs from truth: %d != %d"
                % (len(weight), len(true))
            )
        if not np.all(np.isfinite(weight)):
            bad = int(np.sum(~np.isfinite(weight)))
            raise ValueError(
                "sample_weight contains %d non-finite value(s)" % bad
            )
        if np.any(weight < 0.0):
            raise ValueError("sample_weight must be non-negative")
        if not np.any(weight > 0.0):
            raise ValueError("sample_weight must have positive total weight")
    return true, predicted, weight


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return float(weighted_quantile(values, np.asarray([0.5]), weights)[0])


def _weighted_central_resolution(
    values: np.ndarray, weights: np.ndarray
) -> float:
    quantiles = weighted_quantile(
        values, np.asarray([0.16, 0.84]), weights
    )
    return float(0.5 * (quantiles[1] - quantiles[0]))


def _derive_fractional_floor(
    truth: np.ndarray,
    weights: np.ndarray,
    requested_floor: Optional[float],
) -> float:
    if requested_floor is not None:
        floor = float(requested_floor)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("fractional_energy_floor must be finite and > 0")
        return floor

    positive = (np.abs(truth) > 0.0) & (weights > 0.0)
    if np.any(positive):
        typical = float(
            weighted_quantile(
                np.abs(truth[positive]),
                np.asarray([0.5]),
                weights[positive],
            )[0]
        )
    else:
        typical = 0.0
    return max(1e-12, 1e-6 * typical)


def _constant_span(center: float) -> float:
    return max(abs(center), 1.0) * 1e-6


def _derive_histogram_edges(
    truth: np.ndarray,
    weights: np.ndarray,
    n_histogram_bins: int,
) -> np.ndarray:
    if int(n_histogram_bins) != n_histogram_bins or n_histogram_bins < 1:
        raise ValueError("n_histogram_bins must be a positive integer")
    selected = weights > 0.0
    low = float(np.min(truth[selected]))
    high = float(np.max(truth[selected]))
    if not high > low:
        half_span = 0.5 * _constant_span(low)
        low, high = low - half_span, high + half_span
    return np.linspace(low, high, int(n_histogram_bins) + 1, dtype=float)


def _validate_edges(edges: Any, name: str) -> np.ndarray:
    array = _as_float_1d(edges, name)
    if len(array) < 2:
        raise ValueError("%s must contain at least two values" % name)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s must contain only finite values" % name)
    if not np.all(np.diff(array) > 0.0):
        raise ValueError("%s must be strictly increasing" % name)
    return array


def _derive_energy_bin_edges(
    truth: np.ndarray,
    weights: np.ndarray,
    n_energy_bins: int,
) -> np.ndarray:
    if int(n_energy_bins) != n_energy_bins or n_energy_bins < 1:
        raise ValueError("n_energy_bins must be a positive integer")
    quantiles = weighted_quantile(
        truth,
        np.linspace(0.0, 1.0, int(n_energy_bins) + 1),
        weights,
    )
    edges = np.unique(np.asarray(quantiles, dtype=float))
    if len(edges) < 2:
        center = float(truth[np.flatnonzero(weights > 0.0)[0]])
        half_span = 0.5 * _constant_span(center)
        edges = np.asarray(
            [center - half_span, center + half_span], dtype=float
        )
    return edges


def _histogram_with_flow(
    values: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Return probability masses with explicit underflow and overflow."""

    counts = np.zeros(len(edges) + 1, dtype=float)
    underflow = values < edges[0]
    overflow = values > edges[-1]
    interior = ~(underflow | overflow)
    counts[0] = float(np.sum(weights[underflow]))
    if np.any(interior):
        inside_counts, _ = np.histogram(
            values[interior], bins=edges, weights=weights[interior]
        )
        counts[1:-1] = inside_counts.astype(float)
    counts[-1] = float(np.sum(weights[overflow]))
    total = float(np.sum(counts))
    if total > 0.0:
        counts /= total
    return counts


def _jensen_shannon_bits(
    first: np.ndarray, second: np.ndarray
) -> Optional[float]:
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    p_total = float(np.sum(p))
    q_total = float(np.sum(q))
    if p_total <= 0.0 or q_total <= 0.0:
        return None
    p, q = p / p_total, q / q_total
    middle = 0.5 * (p + q)
    value = 0.0
    positive = p > 0.0
    if np.any(positive):
        value += 0.5 * float(
            np.sum(
                p[positive]
                * np.log2(p[positive] / middle[positive])
            )
        )
    positive = q > 0.0
    if np.any(positive):
        value += 0.5 * float(
            np.sum(
                q[positive]
                * np.log2(q[positive] / middle[positive])
            )
        )
    return float(np.clip(value, 0.0, 1.0))


def _bin_selection(
    values: np.ndarray, edges: np.ndarray, index: int
) -> np.ndarray:
    if index == len(edges) - 2:
        return (values >= edges[index]) & (values <= edges[index + 1])
    return (values >= edges[index]) & (values < edges[index + 1])


def _empty_plot_arrays(
    energy_bin_edges: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    n_bins = len(energy_bin_edges) - 1
    centers = 0.5 * (energy_bin_edges[:-1] + energy_bin_edges[1:])
    nan_values = np.full(n_bins, np.nan, dtype=float)
    return (
        centers,
        np.zeros(n_bins, dtype=int),
        np.zeros(n_bins, dtype=float),
        nan_values.copy(),
        nan_values.copy(),
        nan_values.copy(),
        nan_values.copy(),
        nan_values.copy(),
        nan_values.copy(),
        nan_values.copy(),
    )


def _compute_core(
    truth: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    histogram_edges: np.ndarray,
    energy_bin_edges: np.ndarray,
    fractional_floor: float,
) -> EnergyRegressionResult:
    positive_weight = weight > 0.0
    valid = positive_weight & np.isfinite(prediction)
    total_weight = float(np.sum(weight[positive_weight]))
    valid_weight = float(np.sum(weight[valid]))
    finite_fraction = (
        valid_weight / total_weight if total_weight > 0.0 else 0.0
    )
    n_valid = int(np.sum(valid))

    truth_histogram = _histogram_with_flow(
        truth[positive_weight],
        weight[positive_weight],
        histogram_edges,
    )
    if n_valid:
        prediction_histogram = _histogram_with_flow(
            prediction[valid], weight[valid], histogram_edges
        )
    else:
        prediction_histogram = np.zeros(
            len(histogram_edges) + 1, dtype=float
        )

    jsd_bits = _jensen_shannon_bits(
        truth_histogram, prediction_histogram
    )
    if jsd_bits is None:
        js_distance = None
        hist_similarity = 0.0
        histogram_overlap = None
    else:
        js_distance = float(math.sqrt(jsd_bits))
        hist_similarity = float(np.clip(1.0 - js_distance, 0.0, 1.0))
        histogram_overlap = float(
            np.sum(np.minimum(truth_histogram, prediction_histogram))
        )

    if not n_valid:
        (
            energy_centers,
            energy_counts,
            energy_weights,
            response_by_bin,
            bias_by_bin,
            resolution_by_bin,
            fractional_bias_by_bin,
            fractional_resolution_by_bin,
            mae_by_bin,
            rmse_by_bin,
        ) = _empty_plot_arrays(energy_bin_edges)
        return EnergyRegressionResult(
            status="no_finite_predictions",
            score_name=SCORE_NAME,
            score_definition=SCORE_DEFINITION,
            n_total=int(len(truth)),
            n_valid=0,
            finite_fraction=0.0,
            energy_regression_score=0.0,
            event_score=0.0,
            hist_similarity=hist_similarity,
            mae=None,
            rmse=None,
            bias=None,
            r2=None,
            mae_skill=None,
            fractional_bias=None,
            fractional_resolution_68=None,
            balanced_fractional_mae=None,
            fractional_energy_floor=float(fractional_floor),
            jsd_bits=jsd_bits,
            js_distance=js_distance,
            histogram_overlap=histogram_overlap,
            wasserstein_1=None,
            histogram_edges=histogram_edges.copy(),
            truth_histogram=truth_histogram,
            prediction_histogram=prediction_histogram,
            energy_bin_edges=energy_bin_edges.copy(),
            energy_bin_centers=energy_centers,
            energy_bin_counts=energy_counts,
            energy_bin_weights=energy_weights,
            response_by_energy_bin=response_by_bin,
            bias_by_energy_bin=bias_by_bin,
            resolution68_by_energy_bin=resolution_by_bin,
            fractional_bias_by_energy_bin=fractional_bias_by_bin,
            fractional_resolution_68_by_energy_bin=(
                fractional_resolution_by_bin
            ),
            mae_by_energy_bin=mae_by_bin,
            rmse_by_energy_bin=rmse_by_bin,
        )

    true_valid = truth[valid]
    predicted_valid = prediction[valid]
    weight_valid = weight[valid]
    residual = predicted_valid - true_valid
    absolute_residual = np.abs(residual)
    denominator = np.maximum(np.abs(true_valid), fractional_floor)
    fractional_residual = residual / denominator
    absolute_fractional_residual = np.abs(fractional_residual)
    response = 1.0 + fractional_residual

    mae = _weighted_mean(absolute_residual, weight_valid)
    rmse = math.sqrt(
        max(0.0, _weighted_mean(residual * residual, weight_valid))
    )
    bias = _weighted_mean(residual, weight_valid)

    truth_mean = _weighted_mean(true_valid, weight_valid)
    denominator_r2 = float(
        np.sum(weight_valid * (true_valid - truth_mean) ** 2)
    )
    r2_tolerance = (
        np.finfo(float).eps
        * float(np.sum(weight_valid))
        * max(1.0, truth_mean * truth_mean)
    )
    if denominator_r2 <= r2_tolerance:
        r2 = None
    else:
        r2 = float(
            1.0
            - np.sum(weight_valid * residual * residual) / denominator_r2
        )

    baseline = _weighted_median(true_valid, weight_valid)
    baseline_mae = _weighted_mean(
        np.abs(true_valid - baseline), weight_valid
    )
    mae_skill = (
        None
        if baseline_mae <= np.finfo(float).eps
        else float(1.0 - mae / baseline_mae)
    )

    fractional_bias = _weighted_median(
        fractional_residual, weight_valid
    )
    fractional_resolution = _weighted_central_resolution(
        fractional_residual, weight_valid
    )

    n_bins = len(energy_bin_edges) - 1
    energy_centers = np.full(n_bins, np.nan, dtype=float)
    energy_counts = np.zeros(n_bins, dtype=int)
    energy_weights = np.zeros(n_bins, dtype=float)
    response_by_bin = np.full(n_bins, np.nan, dtype=float)
    bias_by_bin = np.full(n_bins, np.nan, dtype=float)
    resolution_by_bin = np.full(n_bins, np.nan, dtype=float)
    fractional_bias_by_bin = np.full(n_bins, np.nan, dtype=float)
    fractional_resolution_by_bin = np.full(
        n_bins, np.nan, dtype=float
    )
    mae_by_bin = np.full(n_bins, np.nan, dtype=float)
    rmse_by_bin = np.full(n_bins, np.nan, dtype=float)
    fractional_mae_by_bin = []

    for index in range(n_bins):
        selected = _bin_selection(
            true_valid, energy_bin_edges, index
        )
        if not np.any(selected):
            continue
        local_weight = weight_valid[selected]
        energy_counts[index] = int(np.sum(selected))
        energy_weights[index] = float(np.sum(local_weight))
        energy_centers[index] = _weighted_mean(
            true_valid[selected], local_weight
        )
        response_by_bin[index] = _weighted_median(
            response[selected], local_weight
        )
        bias_by_bin[index] = _weighted_median(
            residual[selected], local_weight
        )
        resolution_by_bin[index] = _weighted_central_resolution(
            residual[selected], local_weight
        )
        fractional_bias_by_bin[index] = _weighted_median(
            fractional_residual[selected], local_weight
        )
        fractional_resolution_by_bin[index] = (
            _weighted_central_resolution(
                fractional_residual[selected], local_weight
            )
        )
        mae_by_bin[index] = _weighted_mean(
            absolute_residual[selected], local_weight
        )
        rmse_by_bin[index] = math.sqrt(
            max(
                0.0,
                _weighted_mean(
                    residual[selected] * residual[selected],
                    local_weight,
                ),
            )
        )
        fractional_mae_by_bin.append(
            _weighted_mean(
                absolute_fractional_residual[selected], local_weight
            )
        )

    balanced_fractional_mae = (
        float(np.mean(fractional_mae_by_bin))
        if fractional_mae_by_bin
        else None
    )
    event_score = (
        float(np.clip(1.0 - balanced_fractional_mae, 0.0, 1.0))
        if balanced_fractional_mae is not None
        else 0.0
    )
    score = float(
        finite_fraction
        * math.sqrt(max(0.0, event_score * hist_similarity))
    )

    wasserstein = one_dimensional_wasserstein(
        truth[positive_weight],
        prediction[valid],
        weight[positive_weight],
        weight[valid],
    )
    if not math.isfinite(wasserstein):
        wasserstein_value = None
    else:
        wasserstein_value = float(wasserstein)

    return EnergyRegressionResult(
        status="ok",
        score_name=SCORE_NAME,
        score_definition=SCORE_DEFINITION,
        n_total=int(len(truth)),
        n_valid=n_valid,
        finite_fraction=float(finite_fraction),
        energy_regression_score=score,
        event_score=event_score,
        hist_similarity=hist_similarity,
        mae=float(mae),
        rmse=float(rmse),
        bias=float(bias),
        r2=r2,
        mae_skill=mae_skill,
        fractional_bias=float(fractional_bias),
        fractional_resolution_68=float(fractional_resolution),
        balanced_fractional_mae=balanced_fractional_mae,
        fractional_energy_floor=float(fractional_floor),
        jsd_bits=jsd_bits,
        js_distance=js_distance,
        histogram_overlap=histogram_overlap,
        wasserstein_1=wasserstein_value,
        histogram_edges=histogram_edges.copy(),
        truth_histogram=truth_histogram,
        prediction_histogram=prediction_histogram,
        energy_bin_edges=energy_bin_edges.copy(),
        energy_bin_centers=energy_centers,
        energy_bin_counts=energy_counts,
        energy_bin_weights=energy_weights,
        response_by_energy_bin=response_by_bin,
        bias_by_energy_bin=bias_by_bin,
        resolution68_by_energy_bin=resolution_by_bin,
        fractional_bias_by_energy_bin=fractional_bias_by_bin,
        fractional_resolution_68_by_energy_bin=(
            fractional_resolution_by_bin
        ),
        mae_by_energy_bin=mae_by_bin,
        rmse_by_energy_bin=rmse_by_bin,
    )


_BOOTSTRAP_FIELDS: Sequence[str] = (
    "energy_regression_score",
    "event_score",
    "hist_similarity",
    "mae",
    "rmse",
    "bias",
    "r2",
    "mae_skill",
    "fractional_bias",
    "fractional_resolution_68",
    "balanced_fractional_mae",
    "jsd_bits",
    "js_distance",
    "histogram_overlap",
    "wasserstein_1",
)


def _bootstrap_intervals(
    truth: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    histogram_edges: np.ndarray,
    energy_bin_edges: np.ndarray,
    fractional_floor: float,
    estimates: EnergyRegressionResult,
    n_bootstrap: int,
    confidence: float,
    random_state: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    if n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(random_state)
    samples: Dict[str, list] = {
        name: [] for name in _BOOTSTRAP_FIELDS
    }
    n_events = len(truth)
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, n_events, size=n_events)
        replicate = _compute_core(
            truth[indices],
            prediction[indices],
            weight[indices],
            histogram_edges,
            energy_bin_edges,
            fractional_floor,
        )
        for name in _BOOTSTRAP_FIELDS:
            value = getattr(replicate, name)
            if value is not None and math.isfinite(float(value)):
                samples[name].append(float(value))

    alpha = 0.5 * (1.0 - confidence)
    output: Dict[str, Dict[str, Any]] = {}
    for name in _BOOTSTRAP_FIELDS:
        values = np.asarray(samples[name], dtype=float)
        estimate = getattr(estimates, name)
        if values.size == 0:
            output[name] = {
                "estimate": estimate,
                "mean": None,
                "lower": None,
                "upper": None,
                "standard_error": None,
                "standard_deviation": None,
                "confidence": float(confidence),
                "confidence_level": float(confidence),
                "n_requested": int(n_bootstrap),
                "n_successful": 0,
                "n_bootstrap_valid": 0,
            }
            continue
        low, high = np.quantile(values, [alpha, 1.0 - alpha])
        standard_deviation = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )
        output[name] = {
            "estimate": estimate,
            "mean": float(np.mean(values)),
            "lower": float(low),
            "upper": float(high),
            "standard_error": standard_deviation,
            "standard_deviation": standard_deviation,
            "confidence": float(confidence),
            "confidence_level": float(confidence),
            "n_requested": int(n_bootstrap),
            "n_successful": int(values.size),
            "n_bootstrap_valid": int(values.size),
        }
    # Serialized reports and the plotting layer use the longer name.
    output["histogram_similarity"] = dict(output["hist_similarity"])
    return output


def evaluate_energy_regression(
    truth: Any,
    prediction: Any,
    sample_weight: Optional[Any] = None,
    *,
    n_histogram_bins: int = 50,
    histogram_edges: Optional[Any] = None,
    n_energy_bins: int = 10,
    energy_bin_edges: Optional[Any] = None,
    fractional_energy_floor: Optional[float] = None,
    n_bootstrap: int = 0,
    confidence: float = 0.95,
    random_state: Optional[int] = 42,
) -> EnergyRegressionResult:
    """Evaluate scalar energy predictions.

    Parameters
    ----------
    truth, prediction:
        Aligned one-dimensional arrays.  Truth must be finite.  Non-finite
        predictions are treated as failures and reduce ``finite_fraction``.
    sample_weight:
        Optional aligned, finite, non-negative event weights.
    n_histogram_bins, histogram_edges:
        Shared histogram definition.  If explicit edges are omitted, equal
        width edges are derived only from the positive-weight truth range.
        Histograms contain two extra entries: underflow first and overflow
        last.
    n_energy_bins, energy_bin_edges:
        Quantile bins used for the energy-balanced event term and plot-ready
        response curves.  Explicit edges make comparisons independently
        reproducible.
    fractional_energy_floor:
        Positive lower bound in fractional residual denominators.  The
        default is one millionth of the weighted median non-zero |truth|.
    n_bootstrap:
        Number of paired event-bootstrap replicates.  Every replicate
        recomputes the complete pointwise and histogram metrics while keeping
        the benchmark's shared edges fixed.
    """

    true, predicted, weight = _validate_inputs(
        truth, prediction, sample_weight
    )
    if int(n_bootstrap) != n_bootstrap or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a non-negative integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    if histogram_edges is None:
        shared_histogram_edges = _derive_histogram_edges(
            true, weight, n_histogram_bins
        )
    else:
        shared_histogram_edges = _validate_edges(
            histogram_edges, "histogram_edges"
        )
    if energy_bin_edges is None:
        shared_energy_edges = _derive_energy_bin_edges(
            true, weight, n_energy_bins
        )
    else:
        shared_energy_edges = _validate_edges(
            energy_bin_edges, "energy_bin_edges"
        )
        selected_truth = true[weight > 0.0]
        if (
            np.min(selected_truth) < shared_energy_edges[0]
            or np.max(selected_truth) > shared_energy_edges[-1]
        ):
            raise ValueError(
                "energy_bin_edges must cover every positive-weight truth value"
            )
    floor = _derive_fractional_floor(
        true, weight, fractional_energy_floor
    )

    result = _compute_core(
        true,
        predicted,
        weight,
        shared_histogram_edges,
        shared_energy_edges,
        floor,
    )
    result.bootstrap = _bootstrap_intervals(
        true,
        predicted,
        weight,
        shared_histogram_edges,
        shared_energy_edges,
        floor,
        result,
        int(n_bootstrap),
        float(confidence),
        random_state,
    )
    return result


__all__ = [
    "EnergyRegressionResult",
    "SCORE_DEFINITION",
    "SCORE_NAME",
    "evaluate_energy_regression",
]
