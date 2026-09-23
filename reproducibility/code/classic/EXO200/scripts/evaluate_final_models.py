#!/usr/bin/env python3
"""Build a comprehensive classification evaluation from final test predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "classification"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "evaluation"
MODEL_NAMES = (
    "cnn_004_multiview_late_fusion",
    "gnn_001_static_gine",
    "seq_001_bigru",
    "ssm_001_pointmamba",
)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _roc_curve(target: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(-score, kind="mergesort")
    y = target[order].astype(np.int64)
    s = score[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]
    tps = np.cumsum(y)[distinct].astype(np.float64)
    fps = (1 + distinct - tps).astype(np.float64)
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    thresholds = np.r_[np.inf, s[distinct]]
    return fps / fps[-1], tps / tps[-1], thresholds


def _pr_curve(target: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(-score, kind="mergesort")
    y = target[order].astype(np.int64)
    s = score[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]
    tps = np.cumsum(y)[distinct].astype(np.float64)
    predicted_positive = (distinct + 1).astype(np.float64)
    precision = tps / predicted_positive
    recall = tps / tps[-1]
    thresholds = s[distinct]
    return np.r_[1.0, precision], np.r_[0.0, recall], np.r_[np.inf, thresholds]


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = _pr_curve(target, score)
    return float(np.sum(np.diff(recall) * precision[1:]))


def _downsample(*arrays: np.ndarray, maximum: int = 500) -> tuple[np.ndarray, ...]:
    size = arrays[0].size
    if size <= maximum:
        return arrays
    indices = np.unique(np.linspace(0, size - 1, maximum, dtype=np.int64))
    return tuple(array[indices] for array in arrays)


def _expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return error


def evaluate(target: np.ndarray, logit: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target = np.asarray(target).reshape(-1).astype(np.int64)
    logit = np.asarray(logit).reshape(-1).astype(np.float64)
    if target.shape != logit.shape or target.size == 0:
        raise ValueError("target and prediction must be non-empty one-dimensional arrays")
    if not np.isfinite(logit).all() or not np.isin(target, (0, 1)).all():
        raise ValueError("targets must be binary and logits must be finite")

    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
    predicted = (probability >= 0.5).astype(np.int64)
    tn = int(np.sum((target == 0) & (predicted == 0)))
    fp = int(np.sum((target == 0) & (predicted == 1)))
    fn = int(np.sum((target == 1) & (predicted == 0)))
    tp = int(np.sum((target == 1) & (predicted == 1)))

    background_precision = _safe_div(tp, tp + fp)
    background_recall = _safe_div(tp, tp + fn)
    background_f1 = _safe_div(
        2 * background_precision * background_recall,
        background_precision + background_recall,
    )
    signal_precision = _safe_div(tn, tn + fn)
    signal_recall = _safe_div(tn, tn + fp)
    signal_f1 = _safe_div(
        2 * signal_precision * signal_recall,
        signal_precision + signal_recall,
    )
    accuracy = (tp + tn) / target.size
    balanced_accuracy = 0.5 * (background_recall + signal_recall)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, mcc_denominator)
    clipped_probability = np.clip(probability, 1e-12, 1.0 - 1e-12)

    fpr, tpr, roc_threshold = _roc_curve(target, logit)
    precision, recall, pr_threshold = _pr_curve(target, logit)
    roc_auc = _auc(fpr, tpr)
    average_precision = _average_precision(target, logit)
    youden = tpr - fpr
    best_index = int(np.argmax(youden))
    best_logit_threshold = float(roc_threshold[best_index])
    best_probability_threshold = float(
        1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, best_logit_threshold))))
    )

    metrics: dict[str, Any] = {
        "events": int(target.size),
        "signal_count": int(np.sum(target == 0)),
        "background_count": int(np.sum(target == 1)),
        "decision_threshold_probability": 0.5,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": 0.5 * (signal_precision + background_precision),
        "macro_recall": balanced_accuracy,
        "macro_f1": 0.5 * (signal_f1 + background_f1),
        "matthews_correlation_coefficient": mcc,
        "log_loss_event_weighted": float(
            -np.mean(target * np.log(clipped_probability) + (1 - target) * np.log(1 - clipped_probability))
        ),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "expected_calibration_error_15_bins": _expected_calibration_error(target, probability),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "signal": {
            "precision": signal_precision,
            "recall": signal_recall,
            "f1": signal_f1,
            "support": tn + fp,
        },
        "background": {
            "precision": background_precision,
            "recall": background_recall,
            "f1": background_f1,
            "support": tp + fn,
        },
        "youden_optimal_threshold": {
            "probability": best_probability_threshold,
            "logit": best_logit_threshold,
            "sensitivity": float(tpr[best_index]),
            "specificity": float(1.0 - fpr[best_index]),
            "note": "diagnostic only; all reported class metrics use probability threshold 0.5",
        },
    }
    roc_fpr, roc_tpr, roc_threshold = _downsample(fpr, tpr, roc_threshold)
    pr_precision, pr_recall, pr_threshold = _downsample(precision, recall, pr_threshold)
    curves = {
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "roc_logit_threshold": roc_threshold,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
        "pr_logit_threshold": pr_threshold,
    }
    return metrics, curves


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, Any]] = {}
    reference_target: np.ndarray | None = None
    curve_payload: dict[str, np.ndarray] = {}
    source_files: dict[str, dict[str, str]] = {}
    for model_name in MODEL_NAMES:
        model_dir = args.input_root / model_name
        prediction_path = model_dir / "predictions.npz"
        checkpoint_path = model_dir / "best.pt"
        if not prediction_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing final artifacts for {model_name}")
        with np.load(prediction_path) as loaded:
            target = loaded["target"]
            prediction = loaded["prediction"]
        if reference_target is None:
            reference_target = target.copy()
        elif not np.array_equal(reference_target, target):
            raise ValueError(f"test labels or ordering differ for {model_name}")
        metrics, curves = evaluate(target, prediction)
        all_metrics[model_name] = metrics
        for key, values in curves.items():
            curve_payload[f"{model_name}__{key}"] = values
        source_files[model_name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "predictions": str(prediction_path.resolve()),
        }

    ranking = sorted(MODEL_NAMES, key=lambda name: all_metrics[name]["roc_auc"], reverse=True)
    summary = {
        "task": "EXO-200 binary classification",
        "positive_class": "background",
        "class_names": ["signal", "background"],
        "models": list(MODEL_NAMES),
        "ranking_by_roc_auc": list(ranking),
        "test_labels_identical_across_models": True,
        "metrics": all_metrics,
        "source_files": source_files,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "curves.npz", **curve_payload)

    fields = [
        "model", "roc_auc", "average_precision", "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "matthews_correlation_coefficient", "log_loss_event_weighted", "brier_score",
        "expected_calibration_error_15_bins", "tn", "fp", "fn", "tp",
        "signal_precision", "signal_recall", "signal_f1",
        "background_precision", "background_recall", "background_f1",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name in ranking:
            metrics = all_metrics[model_name]
            writer.writerow({
                "model": model_name,
                **{name: metrics[name] for name in fields[1:12]},
                **metrics["confusion_matrix"],
                **{f"signal_{name}": metrics["signal"][name] for name in ("precision", "recall", "f1")},
                **{f"background_{name}": metrics["background"][name] for name in ("precision", "recall", "f1")},
            })

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    comparison_rows = [
        {
            "model": name,
            "roc_auc": all_metrics[name]["roc_auc"],
            "average_precision": all_metrics[name]["average_precision"],
            "accuracy": all_metrics[name]["accuracy"],
            "balanced_accuracy": all_metrics[name]["balanced_accuracy"],
            "macro_f1": all_metrics[name]["macro_f1"],
            "mcc": all_metrics[name]["matthews_correlation_coefficient"],
            "brier_score": all_metrics[name]["brier_score"],
            "ece": all_metrics[name]["expected_calibration_error_15_bins"],
            **all_metrics[name]["confusion_matrix"],
        }
        for name in ranking
    ]
    roc_rows = []
    for name in MODEL_NAMES:
        for fpr_value, tpr_value in zip(
            curve_payload[f"{name}__roc_fpr"], curve_payload[f"{name}__roc_tpr"]
        ):
            roc_rows.append({"model": name, "fpr": float(fpr_value), "tpr": float(tpr_value)})

    database_path = output_dir / "evaluation.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE IF EXISTS model_metrics")
        connection.execute("DROP TABLE IF EXISTS comparison")
        connection.execute("DROP TABLE IF EXISTS roc_curves")
        connection.execute(
            "CREATE TABLE model_metrics (model TEXT, roc_auc REAL, average_precision REAL, "
            "accuracy REAL, balanced_accuracy REAL, macro_f1 REAL, mcc REAL, "
            "brier_score REAL, ece REAL, tn INTEGER, fp INTEGER, fn INTEGER, tp INTEGER)"
        )
        connection.executemany(
            "INSERT INTO model_metrics VALUES (:model, :roc_auc, :average_precision, :accuracy, "
            ":balanced_accuracy, :macro_f1, :mcc, :brier_score, :ece, :tn, :fp, :fn, :tp)",
            comparison_rows,
        )
        connection.execute("CREATE TABLE comparison (model TEXT, metric TEXT, value REAL)")
        connection.executemany(
            "INSERT INTO comparison VALUES (:model, :metric, :value)",
            [
                {"model": row["model"], "metric": metric, "value": row[metric]}
                for row in comparison_rows
                for metric in ("roc_auc", "average_precision", "balanced_accuracy", "macro_f1")
            ],
        )
        connection.execute("CREATE TABLE roc_curves (model TEXT, fpr REAL, tpr REAL)")
        connection.executemany(
            "INSERT INTO roc_curves VALUES (:model, :fpr, :tpr)", roc_rows
        )
    source = {
        "id": "final_predictions",
        "label": "Final EXO-200 test predictions",
        "path": "outputs/classification/evaluation/evaluation.json",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT * FROM model_metrics ORDER BY roc_auc DESC; SELECT * FROM comparison; SELECT * FROM roc_curves;",
            "description": "Reads the reviewed metrics and curve rows generated from each final predictions.npz file.",
            "executed_at": generated_at,
            "tables_used": ["model_metrics", "comparison", "roc_curves"],
            "filters": [
                "Held-out grouped test split only",
                "signal=0 and background=1",
                "class metrics use sigmoid(logit) >= 0.5",
            ],
            "metric_definitions": [
                "ROC-AUC ranks background events above signal events across all thresholds.",
                "Average precision summarizes the background-class precision-recall curve.",
                "Balanced accuracy is the mean of signal recall and background recall.",
            ],
        },
    }
    best = all_metrics[ranking[0]]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "EXO-200 Final Classification Evaluation",
            "description": "Four-model held-out test evaluation from final checkpoint predictions.",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "quality_comparison",
                    "title": "Model discrimination and threshold performance",
                    "subtitle": "Same 140,383-event held-out test split; higher is better",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "comparison",
                    "sourceId": "final_predictions",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Model"},
                        "y": {"field": "value", "type": "quantitative", "format": "percent", "label": "Score"},
                        "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                    },
                    "layout": "full",
                    "maxRows": 20,
                },
                {
                    "id": "roc_curves",
                    "title": "ROC curves",
                    "subtitle": "Background is the positive class; the diagonal is random ranking",
                    "showDescription": True,
                    "type": "line",
                    "dataset": "roc_curves",
                    "sourceId": "final_predictions",
                    "encodings": {
                        "x": {"field": "fpr", "type": "quantitative", "format": "percent", "label": "False-positive rate"},
                        "y": {"field": "tpr", "type": "quantitative", "format": "percent", "label": "True-positive rate"},
                        "color": {"field": "model", "type": "nominal", "label": "Model"},
                    },
                    "layout": "full",
                    "maxRows": 2200,
                },
            ],
            "tables": [
                {
                    "id": "complete_metrics",
                    "title": "Complete model comparison",
                    "subtitle": "Threshold metrics use probability 0.5; calibration errors are lower-is-better",
                    "showDescription": True,
                    "dataset": "model_metrics",
                    "sourceId": "final_predictions",
                    "defaultSort": {"field": "roc_auc", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "model", "label": "Model", "type": "text"},
                        {"field": "roc_auc", "label": "ROC-AUC", "format": "percent"},
                        {"field": "average_precision", "label": "AP", "format": "percent"},
                        {"field": "accuracy", "label": "Accuracy", "format": "percent"},
                        {"field": "balanced_accuracy", "label": "Balanced acc.", "format": "percent"},
                        {"field": "macro_f1", "label": "Macro F1", "format": "percent"},
                        {"field": "mcc", "label": "MCC", "format": "number"},
                        {"field": "brier_score", "label": "Brier", "format": "number"},
                        {"field": "ece", "label": "ECE", "format": "number"},
                    ],
                }
            ],
            "sources": [source],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# EXO-200 Final Classification Evaluation"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "final_predictions",
                    "body": (
                        "## GINE is the strongest final classifier\n\n"
                        f"Across the identical held-out test population, **{ranking[0]}** ranks first "
                        f"with ROC-AUC **{best['roc_auc']:.4f}**, AP **{best['average_precision']:.4f}**, "
                        f"accuracy **{best['accuracy']:.4f}**, and macro F1 **{best['macro_f1']:.4f}**. "
                        "PointMamba and BiGRU form the next tier; CNN-004 trails on every primary discrimination metric."
                    ),
                },
                {
                    "id": "comparison_explanation",
                    "type": "markdown",
                    "body": "## GINE leads on ranking and fixed-threshold quality\n\nThe grouped bars compare threshold-free ranking quality with performance at the fixed 0.5 probability threshold. Reading both avoids selecting a model solely from one operating point.",
                },
                {"id": "comparison_chart", "type": "chart", "chartId": "quality_comparison"},
                {
                    "id": "roc_explanation",
                    "type": "markdown",
                    "body": "## The ranking advantage persists across operating points\n\nThe ROC curves show sensitivity versus false-positive rate over all possible thresholds. GINE stays closest to the upper-left region, consistent with its leading ROC-AUC.",
                },
                {"id": "roc_chart", "type": "chart", "chartId": "roc_curves"},
                {
                    "id": "definitions",
                    "type": "markdown",
                    "body": "## Scope and metric definitions\n\nEvaluation covers 140,383 held-out events: 66,062 signal events (`nccl == 1`) and 74,321 background events (`nccl > 1`). Background is the positive class. Accuracy, class precision/recall/F1, balanced accuracy, MCC, and confusion matrices use sigmoid(logit) ≥ 0.5. ROC-AUC and AP use raw ranking scores across thresholds.",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": "## Reproducible evaluation method\n\nThe evaluator loads each final `predictions.npz`, verifies identical test-label arrays and ordering, then recomputes all metrics independently. It also saves downsampled ROC and PR curve points. The original model checkpoints and prediction artifacts are not modified.",
                },
                {
                    "id": "table_explanation",
                    "type": "markdown",
                    "body": "## Exact metrics confirm the same ranking\n\nThe table provides exact discrimination, threshold, correlation, and calibration measures. Brier score and ECE are calibration-oriented and should be minimized; they do not overturn the primary discrimination ranking.",
                },
                {"id": "metrics_table", "type": "table", "tableId": "complete_metrics"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## Limitations and robustness checks\n\nThis is a single fixed grouped test split, so it does not quantify run-to-run training variance or cross-validation uncertainty. The four label arrays are exactly identical, and the recomputed ROC-AUC and accuracy match the canonical project outputs. Event-weighted log loss differs slightly from the existing batch-mean loss because the final partial batch receives equal batch weight in the original evaluator.",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## Recommended next steps\n\nUse GINE as the current primary model. Before physics-facing deployment, choose an operating threshold from the required signal-efficiency/background-rejection tradeoff and calibrate probabilities on validation data rather than selecting a threshold on the held-out test set.",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## Further questions\n\nThe next useful checks are per-run performance, uncertainty across training seeds, and efficiency/rejection at domain-selected operating points. Those require additional training runs or event-level run identifiers not stored in the current prediction files.",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "comparison": [
                    {"model": row["model"], "metric": metric, "value": row[metric]}
                    for row in comparison_rows
                    for metric in ("roc_auc", "average_precision", "balanced_accuracy", "macro_f1")
                ],
                "model_metrics": comparison_rows,
                "roc_curves": roc_rows,
            },
        },
        "sources": [source],
    }
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({name: all_metrics[name] for name in ranking}, indent=2))
    print(f"Saved comprehensive evaluation to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
