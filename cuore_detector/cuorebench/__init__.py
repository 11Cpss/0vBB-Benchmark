"""Shared data and evaluation workflow for the CUORE Δt regression benchmark."""

from __future__ import annotations

from .config import (
    CANONICAL_SCALED_MAX,
    CuoreDataConfig,
    scale_target,
    unscale_target,
)
from .data import CuoreRegressionDataset, prepare_cuore_regression_data
from .evaluation import (
    DEGENERATE_METRICS,
    SELECTION_METRICS,
    evaluate_cuore_regression,
)

__all__ = [
    "CANONICAL_SCALED_MAX",
    "DEGENERATE_METRICS",
    "SELECTION_METRICS",
    "CuoreDataConfig",
    "CuoreRegressionDataset",
    "evaluate_cuore_regression",
    "prepare_cuore_regression_data",
    "scale_target",
    "unscale_target",
]
