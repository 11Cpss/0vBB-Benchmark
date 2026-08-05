"""Small dependency-light metrics and plots for binary classification."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Compute ROC AUC with average ranks for tied scores.

    ``None`` is returned when only one class is present, which keeps the
    training loop explicit about an invalid validation split.
    """

    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(label_array) != len(score_array):
        raise ValueError("labels and scores must have equal length")
    if len(label_array) == 0:
        raise ValueError("labels and scores cannot be empty")
    if np.any(~np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    if np.any((label_array != 0) & (label_array != 1)):
        raise ValueError("labels must contain only 0 and 1")
    positive = label_array == 1
    n_positive = int(np.sum(positive))
    n_negative = int(len(label_array) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None

    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.arange(1, len(score_array) + 1, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        stop = start + 1
        while stop < len(sorted_scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + 1 + stop)
        start = stop
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum = float(np.sum(original_ranks[positive]))
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    total_loss: float,
) -> Dict[str, Any]:
    """Summarize one epoch from concatenated labels and logits."""

    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    logit_array = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(label_array) == 0 or len(label_array) != len(logit_array):
        raise ValueError("one non-empty logit is required for every label")
    return {
        "loss": float(total_loss) / len(label_array),
        "accuracy": float(np.mean((logit_array >= 0.0) == (label_array == 1))),
        "auc": binary_auc(label_array, logit_array),
        "events": int(len(label_array)),
    }


def write_history_csv(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> None:
    """Atomically rewrite the compact epoch CSV."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_auc",
        "validation_loss",
        "validation_accuracy",
        "validation_auc",
        "improved",
    ]
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(
                {
                    "epoch": int(record["epoch"]),
                    "learning_rate": float(record["learning_rate"]),
                    "train_loss": float(record["train"]["loss"]),
                    "train_accuracy": float(record["train"]["accuracy"]),
                    "train_auc": record["train"]["auc"],
                    "validation_loss": float(record["validation"]["loss"]),
                    "validation_accuracy": float(
                        record["validation"]["accuracy"]
                    ),
                    "validation_auc": record["validation"]["auc"],
                    "improved": int(bool(record["improved"])),
                }
            )
    os.replace(temporary, destination)


def write_history_json(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> None:
    """Atomically rewrite the lossless JSON history."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(list(history), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, destination)


def save_history_plot(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
    title: str,
) -> None:
    """Save loss and inclusive-AUC curves for a completed or partial run."""

    if not history:
        raise ValueError("history cannot be empty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record["train"]["loss"]) for record in history]
    validation_loss = [
        float(record["validation"]["loss"]) for record in history
    ]
    train_auc = [
        np.nan if record["train"]["auc"] is None else record["train"]["auc"]
        for record in history
    ]
    validation_auc = [
        np.nan
        if record["validation"]["auc"] is None
        else record["validation"]["auc"]
        for record in history
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, validation_loss, marker="o", label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[1].plot(epochs, train_auc, marker="o", label="train")
    axes[1].plot(epochs, validation_auc, marker="o", label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Inclusive AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    figure.suptitle(str(title))
    figure.tight_layout()
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    os.replace(temporary, destination)


__all__ = [
    "binary_auc",
    "classification_metrics",
    "save_history_plot",
    "write_history_csv",
    "write_history_json",
]
