"""Stable public entry point for the standalone Simple EnergyBench workflow.

The implementation remains in the adjacent ``energybench`` source directory.
Extending this package's module search path lets imports such as
``simple_energybench.metrics`` reuse those files without exposing the generic
``energybench`` package name to collaborators or copying the implementation.
"""

from pathlib import Path as _Path


_IMPLEMENTATION_DIRECTORY = _Path(__file__).resolve().parent.parent / "energybench"
if not _IMPLEMENTATION_DIRECTORY.is_dir():
    raise ImportError(
        "the bundled Simple EnergyBench implementation directory is missing: "
        f"{_IMPLEMENTATION_DIRECTORY}"
    )
__path__.append(str(_IMPLEMENTATION_DIRECTORY))

from .config import EvaluationConfig, ProjectionConfig, TrainingConfig
from .data import PreparedData, prepare_dataset
from .evaluation import evaluate_classification, evaluate_regression
from .models import SimpleClassifier, SimpleRegressor
from .training import set_seed, train_model

__all__ = [
    "EvaluationConfig",
    "PreparedData",
    "ProjectionConfig",
    "SimpleClassifier",
    "SimpleRegressor",
    "TrainingConfig",
    "evaluate_classification",
    "evaluate_regression",
    "prepare_dataset",
    "set_seed",
    "train_model",
]

__version__ = "1.0.0"
