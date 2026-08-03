"""Small dependency-light numerical and serialization utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def finite_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            "%s must be one-dimensional; received shape %s" % (name, array.shape)
        )
    return array


def slugify(value: Any, fallback: str = "item") -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            json_ready(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_output_directory(
    path: Path, allow_existing: bool, known_artifacts: Sequence[str]
) -> None:
    """Reject unsafe output-directory reuse without changing the filesystem."""

    if path.exists() and not path.is_dir():
        raise FileExistsError("output path is not a directory: %s" % path)
    if not path.exists():
        return
    children = list(path.iterdir())
    if children and not allow_existing:
        raise FileExistsError(
            "output directory is non-empty: %s; choose another directory or "
            "pass --allow-existing" % path
        )
    if allow_existing:
        known = set(known_artifacts)
        unknown = sorted(child.name for child in children if child.name not in known)
        if unknown:
            raise FileExistsError(
                "output directory contains unknown files that will not be removed: %s"
                % ", ".join(unknown)
            )


def prepare_output_directory(
    path: Path, allow_existing: bool, known_artifacts: Sequence[str]
) -> None:
    """Create a clean output directory, removing only explicitly known artifacts."""

    validate_output_directory(path, allow_existing, known_artifacts)
    if path.exists() and allow_existing:
        for child in list(path.iterdir()):
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(str(child))
    path.mkdir(parents=True, exist_ok=True)


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


def array_digest(hasher: "hashlib._Hash", name: str, values: Any) -> None:
    array = np.asarray(values)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(str(array.dtype).encode("ascii"))
    if array.dtype.kind in {"U", "S", "O"}:
        for item in array.astype(str).ravel():
            encoded = item.encode("utf-8", errors="surrogatepass")
            hasher.update(len(encoded).to_bytes(8, "little"))
            hasher.update(encoded)
    else:
        contiguous = np.ascontiguousarray(array)
        hasher.update(contiguous.view(np.uint8).tobytes())


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": sys.platform,
    }
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except Exception:
        versions["matplotlib"] = "not-installed"
    return versions
