"""Numerical helpers used by the published SuperNEMO Transformer profile."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import numpy as np

def json_ready(value: Any) -> Any:
    """Recursively convert NumPy/path objects to strict JSON-compatible data."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def weighted_quantile(
    values: Any, quantiles: Any, weights: Optional[Any] = None
) -> np.ndarray:
    """Weighted empirical quantiles with linear interpolation.

    The interpolation positions are the centers of each observation's weight.
    Scaling all weights by a constant therefore leaves the answer unchanged.
    """

    x = np.asarray(values, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    if weights is None:
        w = np.ones(len(x), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[mask], w[mask]
    if x.size == 0:
        return np.full(q.shape, np.nan, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    centers = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(q, centers, x, left=x[0], right=x[-1])


def weighted_mean(values: Any, weights: Optional[Any] = None) -> float:
    x = np.asarray(values, dtype=float)
    if weights is None:
        mask = np.isfinite(x)
        return float(np.mean(x[mask])) if np.any(mask) else float("nan")
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


def weighted_variance(values: Any, weights: Optional[Any] = None) -> float:
    x = np.asarray(values, dtype=float)
    if weights is None:
        mask = np.isfinite(x)
        return float(np.var(x[mask])) if np.any(mask) else float("nan")
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    x, w = x[mask], w[mask]
    mean = np.sum(x * w) / np.sum(w)
    return float(np.sum(w * (x - mean) ** 2) / np.sum(w))


def weighted_correlation(
    first: Any, second: Any, weights: Optional[Any] = None
) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if weights is None:
        w = np.ones(len(x), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if np.sum(mask) < 2:
        return float("nan")
    x, y, w = x[mask], y[mask], w[mask]
    w = w / np.sum(w)
    dx = x - np.sum(w * x)
    dy = y - np.sum(w * y)
    denom = math.sqrt(float(np.sum(w * dx * dx) * np.sum(w * dy * dy)))
    if denom <= 0:
        return 0.0
    return float(np.sum(w * dx * dy) / denom)


def midranks(values: Any) -> np.ndarray:
    """Return one-based average ranks, including exact-tie handling."""

    x = np.asarray(values)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks_sorted = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks_sorted[start:stop] = 0.5 * (start + 1 + stop)
        start = stop
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def weighted_spearman(
    first: Any, second: Any, weights: Optional[Any] = None
) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if weights is None:
        w = np.ones(len(x), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if np.sum(mask) < 2:
        return float("nan")
    return weighted_correlation(midranks(x[mask]), midranks(y[mask]), w[mask])


def effective_sample_size(weights: Any) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    denom = float(np.sum(w * w))
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0


def one_dimensional_wasserstein(
    first: Any,
    second: Any,
    first_weights: Optional[Any] = None,
    second_weights: Optional[Any] = None,
) -> float:
    """Weighted 1-Wasserstein distance on the real line.

    This is the integral of the absolute difference between the two empirical
    CDFs and is equivalent to ``scipy.stats.wasserstein_distance`` in 1D.
    """

    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    wx = np.ones(len(x), dtype=float) if first_weights is None else np.asarray(
        first_weights, dtype=float
    )
    wy = np.ones(len(y), dtype=float) if second_weights is None else np.asarray(
        second_weights, dtype=float
    )
    mx = np.isfinite(x) & np.isfinite(wx) & (wx > 0)
    my = np.isfinite(y) & np.isfinite(wy) & (wy > 0)
    x, wx, y, wy = x[mx], wx[mx], y[my], wy[my]
    if x.size == 0 or y.size == 0:
        return float("nan")
    ox, oy = np.argsort(x), np.argsort(y)
    x, wx, y, wy = x[ox], wx[ox], y[oy], wy[oy]
    wx, wy = wx / np.sum(wx), wy / np.sum(wy)
    all_values = np.sort(np.concatenate([x, y]))
    if all_values.size < 2:
        return 0.0
    deltas = np.diff(all_values)
    x_indices = np.searchsorted(x, all_values[:-1], side="right")
    y_indices = np.searchsorted(y, all_values[:-1], side="right")
    x_cum = np.r_[0.0, np.cumsum(wx)]
    y_cum = np.r_[0.0, np.cumsum(wy)]
    return float(np.sum(np.abs(x_cum[x_indices] - y_cum[y_indices]) * deltas))


def percentile_interval(
    samples: Sequence[float], confidence: float = 0.95
) -> Tuple[float, float]:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)
