#!/usr/bin/env python3
"""Shared output paths for the organized NEXT CNN layout."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "02_models" / "checkpoints"
TRAIN_LOG_DIR = PROJECT_ROOT / "03_training_runs" / "logs"
TRAIN_HISTORY_DIR = PROJECT_ROOT / "03_training_runs" / "history_plots"
EVALUATION_DIR = PROJECT_ROOT / "04_evaluations"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_layout():
    for path in (
        CHECKPOINT_DIR,
        TRAIN_LOG_DIR,
        TRAIN_HISTORY_DIR,
        EVALUATION_DIR,
    ):
        ensure_dir(path)


def checkpoint_path(name):
    return ensure_dir(CHECKPOINT_DIR) / name


def validation_log_path(model_suffix, run_id):
    filename = "%s_validation_metrics_%s.csv" % (
        model_suffix.strip("_"),
        run_id,
    )
    return ensure_dir(TRAIN_LOG_DIR) / filename


def history_json_path(model_suffix, run_id):
    filename = "%s_history_%s.json" % (model_suffix.strip("_"), run_id)
    return ensure_dir(TRAIN_LOG_DIR) / filename


def history_plot_path(model_suffix, run_id):
    filename = "%s_history_%s.png" % (model_suffix.strip("_"), run_id)
    return ensure_dir(TRAIN_HISTORY_DIR) / filename


def score_plot_path(model_suffix, run_id):
    filename = "%s_score_%s.png" % (model_suffix.strip("_"), run_id)
    return ensure_dir(TRAIN_HISTORY_DIR) / filename


def regression_plot_path(model_suffix, run_id):
    filename = "%s_energy_%s.png" % (model_suffix.strip("_"), run_id)
    return ensure_dir(TRAIN_HISTORY_DIR) / filename


def model_evaluation_dir(model_suffix):
    return ensure_dir(EVALUATION_DIR / model_suffix.strip("_"))


__all__ = [
    "CHECKPOINT_DIR",
    "EVALUATION_DIR",
    "PROJECT_ROOT",
    "TRAIN_HISTORY_DIR",
    "TRAIN_LOG_DIR",
    "checkpoint_path",
    "ensure_dir",
    "ensure_layout",
    "history_json_path",
    "history_plot_path",
    "model_evaluation_dir",
    "regression_plot_path",
    "score_plot_path",
    "validation_log_path",
]
