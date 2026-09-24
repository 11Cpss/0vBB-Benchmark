"""Class-conditional classifier score/energy dependence diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .utils import (
    json_ready,
    weighted_correlation,
    weighted_mean,
    weighted_quantile,
    weighted_spearman,
)


def _probability_histogram(
    values: np.ndarray, edges: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    counts = counts.astype(float)
    total = float(np.sum(counts))
    if total <= 0:
        return np.zeros(len(edges) - 1, dtype=float)
    return counts / total


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    middle = 0.5 * (p + q)
    positive_p = p > 0
    positive_q = q > 0
    value = 0.0
    if np.any(positive_p):
        value += 0.5 * float(
            np.sum(p[positive_p] * np.log(p[positive_p] / middle[positive_p]))
        )
    if np.any(positive_q):
        value += 0.5 * float(
            np.sum(q[positive_q] * np.log(q[positive_q] / middle[positive_q]))
        )
    return max(0.0, value)


def _strict_unique_edges(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(edges, dtype=float))
    if unique.size >= 2:
        unique[0] = np.nextafter(unique[0], -np.inf)
        unique[-1] = np.nextafter(unique[-1], np.inf)
        return unique
    center = float(values[0]) if len(values) else 0.0
    span = max(abs(center) * 1e-9, 1e-12)
    return np.asarray([center - span, center + span], dtype=float)


def distance_correlation(
    first: Any,
    second: Any,
    sample_weight: Optional[Any] = None,
    max_samples: int = 1200,
    seed: int = 42,
) -> float:
    """Approximate distance correlation using a deterministic weighted subset.

    The exact estimator needs O(n²) memory.  For large evaluation sets we draw
    at most ``max_samples`` events without replacement, with probabilities
    proportional to the input event weights, and record that cap in the report.
    """

    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if sample_weight is None:
        weight = np.ones(len(x), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    x, y, weight = x[mask], y[mask], weight[mask]
    if len(x) < 4:
        return float("nan")
    if len(x) > max_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            len(x),
            size=int(max_samples),
            replace=False,
            p=weight / np.sum(weight),
        )
        x, y = x[chosen], y[chosen]
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a_centered = (
        a - np.mean(a, axis=0)[None, :] - np.mean(a, axis=1)[:, None] + np.mean(a)
    )
    b_centered = (
        b - np.mean(b, axis=0)[None, :] - np.mean(b, axis=1)[:, None] + np.mean(b)
    )
    dcov_squared = max(float(np.mean(a_centered * b_centered)), 0.0)
    dvar_x_squared = max(float(np.mean(a_centered * a_centered)), 0.0)
    dvar_y_squared = max(float(np.mean(b_centered * b_centered)), 0.0)
    denom = math.sqrt(dvar_x_squared * dvar_y_squared)
    if denom <= 0:
        return 0.0
    dcor_squared = max(dcov_squared / denom, 0.0)
    return float(math.sqrt(dcor_squared))


def evaluate_group_dependence(
    score: Any,
    energy: Any,
    sample_weight: Optional[Any] = None,
    n_energy_bins: int = 8,
    n_score_bins: int = 20,
    threshold: Optional[float] = None,
    min_per_bin: int = 20,
    distance_correlation_max_samples: int = 1200,
    seed: int = 42,
) -> Dict[str, Any]:
    score = np.asarray(score, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if score.ndim != 1 or energy.ndim != 1 or len(score) != len(energy):
        raise ValueError("score and energy must be aligned 1D arrays")
    if sample_weight is None:
        weight = np.ones(len(score), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.ndim != 1 or len(weight) != len(score):
            raise ValueError("sample_weight must be event-aligned")
    finite = (
        np.isfinite(score)
        & np.isfinite(energy)
        & np.isfinite(weight)
        & (weight > 0)
    )
    score, energy, weight = score[finite], energy[finite], weight[finite]
    if len(score) < max(4, 2 * min_per_bin):
        return {
            "status": "not_estimable_insufficient_events",
            "n_valid": int(len(score)),
            "reason": "need at least %d finite events" % max(4, 2 * min_per_bin),
        }

    maximum_energy_bins = max(2, min(int(n_energy_bins), len(score) // min_per_bin))
    energy_edges = _strict_unique_edges(
        weighted_quantile(
            energy, np.linspace(0.0, 1.0, maximum_energy_bins + 1), weight
        ),
        energy,
    )
    score_edges = _strict_unique_edges(
        weighted_quantile(
            score, np.linspace(0.0, 1.0, int(n_score_bins) + 1), weight
        ),
        score,
    )
    pooled = _probability_histogram(score, score_edges, weight)

    energy_centers = []
    score_means = []
    acceptances = []
    bin_weight = []
    conditional_jsd = []
    counts = []
    for index in range(len(energy_edges) - 1):
        in_bin = (energy > energy_edges[index]) & (energy <= energy_edges[index + 1])
        count = int(np.sum(in_bin))
        if count < 2:
            continue
        local_weight = float(np.sum(weight[in_bin]))
        local_distribution = _probability_histogram(
            score[in_bin], score_edges, weight[in_bin]
        )
        energy_centers.append(weighted_mean(energy[in_bin], weight[in_bin]))
        score_means.append(weighted_mean(score[in_bin], weight[in_bin]))
        bin_weight.append(local_weight)
        counts.append(count)
        conditional_jsd.append(_jensen_shannon(local_distribution, pooled))
        if threshold is not None and np.isfinite(threshold):
            acceptances.append(
                weighted_mean((score[in_bin] >= threshold).astype(float), weight[in_bin])
            )

    bin_weight_array = np.asarray(bin_weight, dtype=float)
    bin_probability = bin_weight_array / np.sum(bin_weight_array)
    jsd = float(
        np.sum(bin_probability * np.asarray(conditional_jsd, dtype=float))
    )
    independence_score = float(
        np.clip(1.0 - math.sqrt(jsd / math.log(2.0)), 0.0, 1.0)
    )
    result = {
        "status": "ok",
        "n_valid": int(len(score)),
        "pearson_abs": abs(weighted_correlation(score, energy, weight)),
        "spearman_abs": abs(weighted_spearman(score, energy, weight)),
        "distance_correlation": distance_correlation(
            score,
            energy,
            weight,
            max_samples=distance_correlation_max_samples,
            seed=seed,
        ),
        "distance_correlation_max_samples": int(
            distance_correlation_max_samples
        ),
        "conditional_score_jsd_nats": jsd,
        "energy_independence_score": independence_score,
        "energy_bin_edges": energy_edges,
        "energy_bin_centers": np.asarray(energy_centers),
        "score_mean_by_energy_bin": np.asarray(score_means),
        "events_by_energy_bin": np.asarray(counts, dtype=int),
        "weight_fraction_by_energy_bin": bin_probability,
        "score_histogram_edges": score_edges,
    }
    if threshold is not None and np.isfinite(threshold) and acceptances:
        acceptance_array = np.asarray(acceptances, dtype=float)
        global_acceptance = weighted_mean(
            (score >= threshold).astype(float), weight
        )
        difference = acceptance_array - global_acceptance
        result.update(
            {
                "threshold": float(threshold),
                "global_acceptance": global_acceptance,
                "acceptance_by_energy_bin": acceptance_array,
                "acceptance_rms": float(
                    np.sqrt(np.sum(bin_probability * difference * difference))
                ),
                "acceptance_max_abs_deviation": float(np.max(np.abs(difference))),
            }
        )
        selected = score >= threshold
        if np.any(selected):
            energy_before = _probability_histogram(energy, energy_edges, weight)
            energy_after = _probability_histogram(
                energy[selected], energy_edges, weight[selected]
            )
            sculpt_jsd = _jensen_shannon(energy_before, energy_after)
            result["energy_sculpting_jsd_nats"] = sculpt_jsd
            result["energy_sculpting_distance"] = float(
                math.sqrt(sculpt_jsd / math.log(2.0))
            )
        else:
            result["energy_sculpting_jsd_nats"] = None
            result["energy_sculpting_distance"] = None
    return result


def evaluate_dependence(
    score: Any,
    energy: Any,
    label: Any,
    category: Optional[Any] = None,
    sample_weight: Optional[Any] = None,
    n_energy_bins: int = 8,
    n_score_bins: int = 20,
    threshold: Optional[float] = None,
    min_per_bin: int = 20,
    distance_correlation_max_samples: int = 1200,
    seed: int = 42,
) -> Dict[str, Any]:
    score_array = np.asarray(score)
    energy_array = np.asarray(energy)
    label_array = np.asarray(label)
    if not (
        score_array.ndim == energy_array.ndim == label_array.ndim == 1
        and len(score_array) == len(energy_array) == len(label_array)
    ):
        raise ValueError("score, energy, and label must be aligned 1D arrays")
    if sample_weight is None:
        weight = np.ones(len(score_array), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
    if category is None:
        group = np.asarray(
            ["signal" if bool(value) else "background" for value in label_array],
            dtype=str,
        )
    else:
        group = np.asarray(category).astype(str)
        if group.ndim != 1 or len(group) != len(score_array):
            raise ValueError("category must be event-aligned")

    groups = {}
    scores = []
    group_weights = []
    for name in np.unique(group):
        selected = group == name
        metrics = evaluate_group_dependence(
            score_array[selected],
            energy_array[selected],
            weight[selected],
            n_energy_bins=n_energy_bins,
            n_score_bins=n_score_bins,
            threshold=threshold,
            min_per_bin=min_per_bin,
            distance_correlation_max_samples=distance_correlation_max_samples,
            seed=seed,
        )
        groups[str(name)] = metrics
        if metrics.get("status") == "ok":
            scores.append(float(metrics["energy_independence_score"]))
            group_weights.append(float(np.sum(weight[selected])))
    if scores:
        overall = float(np.average(scores, weights=group_weights))
        worst = float(np.min(scores))
        status = "ok"
    else:
        overall, worst, status = float("nan"), float("nan"), "not_estimable"
    return {
        "status": status,
        "definition": (
            "1 - sqrt(mean_energy_bin_JSD(score_distribution, pooled)"
            " / ln(2)); higher is more class-conditional energy-independent"
        ),
        "overall_energy_independence_score": overall,
        "worst_group_energy_independence_score": worst,
        "groups": groups,
    }
