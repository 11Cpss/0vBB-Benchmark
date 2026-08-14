"""Minimal public API for the standalone Simple EnergyBench workflow."""

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
