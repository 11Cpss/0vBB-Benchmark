#!/usr/bin/env python3
"""Combine all requested final MJD evaluation CSV rows."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DESTINATION = OUTPUT_ROOT / "all_model_results.csv"
ARCHITECTURES = (
    "cnn_004_multiview_late_fusion",
    "gnn_001_static_gine",
    "seq_001_bigru",
    "ssm_001_pointmamba",
)
FAMILIES = {"cnn": "CNN", "gnn": "Graph Neural Network", "seq": "Sequence Model", "ssm": "State Space Model"}
RESULT_COLUMNS = [
    "task", "n_events", "auc", "matched_auc", "matched_auc_status",
    "common_support_auc", "shortcut_gap", "energy_independence_score",
    "worst_energy_independence_score", "ers", "event_score",
    "histogram_similarity", "histogram_overlap", "jsd_bits", "wasserstein_1",
    "mae", "rmse", "bias", "r2", "mae_skill", "fractional_bias",
    "fractional_resolution_68", "balanced_fractional_mae", "finite_fraction",
]
IDENTITY_COLUMNS = [
    "evaluation_status", "dataset", "job_id", "model_category", "architecture_id",
    "model_name", "backend", "trainable_parameters", "best_epoch", "early_stop",
    "epochs_completed",
]
PATH_COLUMNS = ["result_csv", "metrics_json", "predictions_npz", "plot_1", "plot_2"]


def _csv_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def _training_summary(model_dir: Path) -> tuple[Any, Any, int | str]:
    path = model_dir / "history.json"
    if not path.is_file():
        return "", "", ""
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        return "", "", ""
    checkpoint_path = model_dir / "best.pt"
    best_epoch: Any = ""
    if checkpoint_path.is_file():
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        best_epoch = checkpoint.get("epoch", "")
    run_config_path = model_dir / "run_config.json"
    early_stop: Any = ""
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        early_stop = run_config.get("training", {}).get("epochs", "")
    return best_epoch, early_stop, len(history)


def _model_metadata(architecture: str, task: str) -> tuple[str, int]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    module = importlib.import_module(f"architectures.{architecture}.model")
    model = module.build_model(task)
    return model.__class__.__name__, sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for task in ("classification", "regression"):
        for architecture in ARCHITECTURES:
            model_dir = OUTPUT_ROOT / task / architecture
            if task == "classification":
                result_dir = model_dir / "energybench_classification" / "clean"
                result_path = result_dir / "results.csv"
                metrics_path = result_dir / "metrics.json"
                plot_names = ("energy_matched_roc.png", "score_energy_dependence.png")
            else:
                result_dir = model_dir
                result_path = result_dir / "results.csv"
                metrics_path = result_dir / "energybench_metrics.json"
                plot_names = ("energy_regression.png", "energy_histograms.png")
            result = _csv_row(result_path) if result_path.is_file() else {}
            best_epoch, early_stop, epochs = _training_summary(model_dir)
            model_name, parameters = _model_metadata(architecture, task)
            row: dict[str, Any] = {
                "evaluation_status": "DONE" if result else "UNAVAILABLE",
                "dataset": "MJD",
                "job_id": f"{architecture}:{task}",
                "model_category": FAMILIES[architecture.split("_", 1)[0]],
                "architecture_id": architecture,
                "model_name": model_name,
                "backend": "torch",
                "trainable_parameters": parameters,
                "best_epoch": best_epoch,
                "early_stop": early_stop,
                "epochs_completed": epochs,
                **{key: result.get(key, "") for key in RESULT_COLUMNS},
                "result_csv": str(result_path.resolve()) if result_path.is_file() else "",
                "metrics_json": str(metrics_path.resolve()) if metrics_path.is_file() else "",
                "predictions_npz": str((model_dir / "predictions.npz").resolve()),
                "plot_1": str((result_dir / plot_names[0]).resolve()) if (result_dir / plot_names[0]).is_file() else "",
                "plot_2": str((result_dir / plot_names[1]).resolve()) if (result_dir / plot_names[1]).is_file() else "",
            }
            row["task"] = task
            rows.append(row)
    columns = IDENTITY_COLUMNS + RESULT_COLUMNS + PATH_COLUMNS
    with DESTINATION.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(DESTINATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
