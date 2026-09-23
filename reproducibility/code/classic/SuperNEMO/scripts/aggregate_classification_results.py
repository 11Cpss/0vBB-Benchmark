#!/usr/bin/env python3
"""Validate and combine the ten active SuperNEMO classification results."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DESTINATION = OUTPUT_ROOT / "all_model_results.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supernemobench.config import ARCHITECTURES  # noqa: E402
from supernemobench.models import build_model  # noqa: E402


FAMILIES = {
    "cnn": "CNN",
    "gnn": "Graph Neural Network",
    "seq": "Sequence Model",
    "ssm": "State Space Model",
    "transformer": "Transformer",
}

TRAINING_STATUS = {
    "seq_001_bigru": (
        "PARTIAL_CHECKPOINT",
        "Training stopped after epoch 17 without a completion marker; evaluated best epoch 17.",
    ),
    "ssm_001_pointmamba": (
        "PARTIAL_CHECKPOINT",
        "Training stopped after epoch 4 when epoch 5 produced non-finite values; evaluated best epoch 4.",
    ),
}

IDENTITY_COLUMNS = [
    "evaluation_status",
    "training_status",
    "training_note",
    "job_id",
    "model_category",
    "architecture_id",
    "model_name",
    "input_kind",
    "backend",
    "trainable_parameters",
    "best_epoch",
    "training_end_epoch",
    "configured_max_epochs",
    "early_stopped",
    "native_test_loss",
    "native_test_accuracy",
    "native_test_auc",
    "common_support_auc",
    "shortcut_gap",
    "overflow_events_above_3000_kev",
    "overflow_bin_policy",
]

PATH_COLUMNS = [
    "result_csv",
    "metrics_json",
    "predictions_npz",
    "plot_1",
    "plot_2",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _single_csv_row(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or ())
    if len(rows) != 1:
        raise ValueError(f"expected exactly one result row in {path}; found {len(rows)}")
    return columns, rows[0]


def _training_metadata(model_dir: Path, model_id: str) -> dict[str, Any]:
    history = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"missing training history for {model_id}")
    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    run_config = _json(model_dir / "run_config.json")
    best_epoch = int(checkpoint["epoch"])
    end_epoch = int(history[-1]["epoch"])
    maximum = int(run_config["training"]["epochs"])
    patience = int(run_config["training"]["early_stopping_patience"])
    status, note = TRAINING_STATUS.get(model_id, ("COMPLETE", ""))
    early_stopped: bool | str
    if status != "COMPLETE":
        early_stopped = ""
    else:
        early_stopped = end_epoch < maximum and end_epoch - best_epoch >= patience
    return {
        "training_status": status,
        "training_note": note,
        "best_epoch": best_epoch,
        "training_end_epoch": end_epoch,
        "configured_max_epochs": maximum,
        "early_stopped": early_stopped,
    }


def _overflow_count(predictions_path: Path) -> int:
    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {"event_id", "label", "category", "score", "energy_condition"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{predictions_path} is missing {sorted(missing)}")
        energy = np.asarray(archive["energy_condition"], dtype=np.float64)
        category = np.asarray(archive["category"]).astype(str)
        event_id = np.asarray(archive["event_id"]).astype(str)
        if energy.ndim != 1 or category.shape != energy.shape or event_id.shape != energy.shape:
            raise ValueError(f"unaligned prediction columns in {predictions_path}")
        if np.unique(event_id).size != event_id.size:
            raise ValueError(f"duplicate event IDs in {predictions_path}")
        overflow = energy > 3000.0
        if np.any(category[overflow] != "Bi214"):
            raise ValueError(f">3000 keV events are not all Bi214 in {predictions_path}")
        return int(overflow.sum())


def main() -> int:
    model_ids = list(ARCHITECTURES)
    if len(model_ids) != 10 or len(set(model_ids)) != 10:
        raise ValueError(f"active classification registry must contain 10 unique IDs; found {len(model_ids)}")

    rows: list[dict[str, Any]] = []
    canonical_result_columns: list[str] | None = None
    expected_protocol: str | None = None
    expected_events: str | None = None
    for model_id in model_ids:
        architecture = ARCHITECTURES[model_id]
        if "classification" not in architecture.tasks:
            raise ValueError(f"active model does not support classification: {model_id}")
        model_dir = OUTPUT_ROOT / "classification" / model_id
        evaluation_dir = model_dir / "test_evaluation"
        energybench_dir = evaluation_dir / "energybench"
        result_path = energybench_dir / "results.csv"
        metrics_path = energybench_dir / ".energybench" / "metrics.json"
        predictions_path = evaluation_dir / "test_predictions.npz"
        native_metrics_path = evaluation_dir / "test_metrics.json"
        for path in (result_path, metrics_path, predictions_path, native_metrics_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing final evaluation artifact: {path}")

        result_columns, result = _single_csv_row(result_path)
        if canonical_result_columns is None:
            canonical_result_columns = result_columns
        elif result_columns != canonical_result_columns:
            raise ValueError(f"EnergyBench result schema differs for {model_id}")
        if result.get("classification_status") != "ok" or result.get("matched_auc_status") != "ok":
            raise ValueError(f"classification evaluation is not OK for {model_id}")
        if result.get("strict") != "True" or result.get("warning_count") != "0" or result.get("error_count") != "0":
            raise ValueError(f"strict quality gate failed for {model_id}")
        if not result.get("model_id", "").startswith(f"{model_id}@"):
            raise ValueError(f"result model ID does not match directory for {model_id}")
        checkpoint_digest = _sha256(model_dir / "best.pt")
        if result["model_id"].split("@", 1)[1] != checkpoint_digest[:12]:
            raise ValueError(f"result checkpoint hash does not match best.pt for {model_id}")
        if result.get("input_sha256") != _sha256(predictions_path):
            raise ValueError(f"result input hash does not match predictions for {model_id}")
        if expected_protocol is None:
            expected_protocol = result["protocol_fingerprint"]
        elif result["protocol_fingerprint"] != expected_protocol:
            raise ValueError(f"protocol fingerprint differs for {model_id}")
        if expected_events is None:
            expected_events = result["n_events"]
        elif result["n_events"] != expected_events:
            raise ValueError(f"test event count differs for {model_id}")

        report = _json(metrics_path)
        pair = report["classification"]["pairs"]
        if not isinstance(pair, list) or len(pair) != 1:
            raise ValueError(f"expected one classification pair for {model_id}")
        native = _json(native_metrics_path)
        model = build_model(model_id, "classification")
        family_key = model_id.split("_", 1)[0]
        row: dict[str, Any] = {
            "evaluation_status": "DONE",
            **_training_metadata(model_dir, model_id),
            "job_id": f"{model_id}:classification",
            "model_category": FAMILIES[family_key],
            "architecture_id": model_id,
            "model_name": architecture.model_name,
            "input_kind": architecture.input_kind,
            "backend": "torch",
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "native_test_loss": native.get("loss", ""),
            "native_test_accuracy": native.get("accuracy", ""),
            "native_test_auc": native.get("auc", ""),
            "common_support_auc": pair[0].get("inclusive_common_support_auc", ""),
            "shortcut_gap": pair[0].get("shortcut_gap", ""),
            "overflow_events_above_3000_kev": _overflow_count(predictions_path),
            "overflow_bin_policy": "single highest-energy bin",
            **result,
            "result_csv": str(result_path.resolve()),
            "metrics_json": str(metrics_path.resolve()),
            "predictions_npz": str(predictions_path.resolve()),
            "plot_1": str((energybench_dir / "energy_matched_roc.png").resolve()),
            "plot_2": str((energybench_dir / "score_energy_dependence.png").resolve()),
        }
        rows.append(row)

    if len(rows) != 10 or len({row["architecture_id"] for row in rows}) != 10:
        raise ValueError("combined output is not exactly 10 unique classification models")
    columns = IDENTITY_COLUMNS + list(canonical_result_columns or ()) + PATH_COLUMNS
    temporary = DESTINATION.with_name(f".{DESTINATION.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, DESTINATION)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"{DESTINATION}: rows={len(rows)}, events_per_model={expected_events}, protocol={expected_protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
