#!/usr/bin/env python3
"""Train topology boosted trees on NEXT train/validation data only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from next_alt.config import load_training_config
from next_alt.data import build_training_loaders
from next_alt.metrics import save_history_plot
from next_alt.models.classic_topology import (
    TopologyBoostedTreeClassifier,
    TopologyFeatureExtractor,
)


ARCHITECTURE_ID = "classic_001_topology_xgboost"


def _atomic_json(payload: Any, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)


def _write_history_csv(history: Iterable[Mapping[str, Any]], destination: Path) -> None:
    fields = (
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_auc",
        "validation_loss",
        "validation_accuracy",
        "validation_auc",
        "improved",
    )
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in history:
            writer.writerow(
                {
                    "epoch": record["epoch"],
                    "learning_rate": record["learning_rate"],
                    "train_loss": record["train"]["loss"],
                    "train_accuracy": record["train"]["accuracy"],
                    "train_auc": record["train"]["auc"],
                    "validation_loss": record["validation"]["loss"],
                    "validation_accuracy": record["validation"]["accuracy"],
                    "validation_auc": record["validation"]["auc"],
                    "improved": int(bool(record["improved"])),
                }
            )
    os.replace(temporary, destination)


def _collect(loader: Any, extractor: TopologyFeatureExtractor) -> Tuple[np.ndarray, np.ndarray]:
    matrices = []
    labels = []
    for batch in loader:
        matrices.append(
            extractor.extract_batch(batch["coords"], batch["features"], batch["mask"])
        )
        labels.append(batch["label"].numpy().astype(np.int64, copy=False))
    if not matrices:
        raise RuntimeError("training loader produced no events")
    return np.concatenate(matrices), np.concatenate(labels)


def _source_records(files: Any) -> list[Dict[str, Any]]:
    return [
        {
            "relative_path": source.relative_path,
            "group_id": source.group_id,
            "label": int(source.label),
            "category": source.category,
            "split": source.split,
        }
        for source in files
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=SCRIPT_DIR / "config.yaml")
    args = parser.parse_args()
    started = time.monotonic()
    config = load_training_config(args.config, architecture_id=ARCHITECTURE_ID)
    model_config = dict(config["model"])
    extractor_config = dict(model_config.pop("extractor"))
    backend = str(model_config.pop("backend", "auto"))
    estimator_config = dict(model_config.pop("estimator"))
    if model_config:
        raise ValueError("unknown classic model keys: %s" % ", ".join(sorted(model_config)))

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    log_dir = Path(config["output"]["log_dir"])
    plot_dir = Path(config["output"]["plot_dir"])
    artifacts = {
        "best": checkpoint_dir / "best.json",
        "last": checkpoint_dir / "last.json",
        "csv": log_dir / "epochs.csv",
        "history": log_dir / "history.json",
        "plot": plot_dir / "history.png",
        "summary": log_dir / "run_summary.json",
    }
    existing = [path for path in artifacts.values() if path.exists()]
    if existing and not bool(config["output"]["allow_overwrite"]):
        raise FileExistsError(
            "refusing to overwrite existing training artifacts:\n  %s"
            % "\n  ".join(str(path) for path in existing)
        )
    for path in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    extractor = TopologyFeatureExtractor(**extractor_config)
    train_loader, validation_loader, train_dataset, train_files, validation_files = (
        build_training_loaders(config, "points", device_type="cpu")
    )
    train_dataset.set_epoch(1)
    train_matrix, train_labels = _collect(train_loader, extractor)
    validation_matrix, validation_labels = _collect(validation_loader, extractor)
    classifier = TopologyBoostedTreeClassifier(backend=backend, **estimator_config)
    print(
        "Training %s with backend=%s on %d/%d train/validation events"
        % (ARCHITECTURE_ID, classifier.backend, len(train_labels), len(validation_labels)),
        flush=True,
    )
    classifier.fit(
        train_matrix,
        train_labels,
        validation_data=(validation_matrix, validation_labels),
    )
    history = classifier.history_
    if not history or classifier.best_iteration_ < 0:
        raise RuntimeError("boosted-tree training did not produce a best validation round")
    best_epoch = classifier.best_iteration_ + 1
    best_record = history[classifier.best_iteration_]
    best_state = classifier.checkpoint_state(tree_limit=best_epoch)
    last_state = classifier.checkpoint_state(tree_limit=len(history))
    common_checkpoint = {
        "format_version": 1,
        "architecture_id": ARCHITECTURE_ID,
        "model_name": "TopologyBoostedTreeClassifier",
        "input_kind": "topology",
        "task": "classification",
        "label_semantics": {"0nubb": 1, "Bi214": 0},
        "backend": classifier.backend,
        "feature_extractor": extractor.config_dict(),
        "split_config": {
            "seed": int(config["data"]["split_seed"]),
            "fractions": list(config["data"]["split_fractions"]),
        },
        "data_selection": {
            "training_files": _source_records(train_files),
            "validation_files": _source_records(validation_files),
        },
        "training_config": {
            "config_path": config["config_path"],
            "data": config["data"],
            "model": config["model"],
            "training": config["training"],
        },
    }
    _atomic_json(
        {**common_checkpoint, "checkpoint_role": "best", "epoch": best_epoch, "model": best_state},
        artifacts["best"],
    )
    _atomic_json(
        {
            **common_checkpoint,
            "checkpoint_role": "last",
            "epoch": len(history),
            "model": last_state,
        },
        artifacts["last"],
    )
    _write_history_csv(history, artifacts["csv"])
    _atomic_json(history, artifacts["history"])
    save_history_plot(history, artifacts["plot"], ARCHITECTURE_ID)

    duration = time.monotonic() - started
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "gpu": "not used (XGBoost hist on CPU)",
        "xgboost": None,
    }
    if classifier.backend == "xgboost":
        import xgboost

        environment["xgboost"] = xgboost.__version__
    node_count = best_state.get("tree_node_count")
    summary = {
        "status": "DONE",
        "architecture_id": ARCHITECTURE_ID,
        "backend": classifier.backend,
        "parameter_count": None,
        "tree_node_count": node_count,
        "tree_count": best_state.get("tree_count", best_state.get("tree_limit")),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_auc": float(best_record["validation"]["auc"]),
        "best_validation_loss": float(best_record["validation"]["loss"]),
        "duration_seconds": duration,
        "early_stopped": len(history) < int(estimator_config["n_estimators"]),
        "attempt": int(os.environ.get("NEXT_CAMPAIGN_ATTEMPT", "1")),
        "environment": environment,
        "artifacts": {name: str(path) for name, path in artifacts.items() if name != "summary"},
    }
    _atomic_json(summary, artifacts["summary"])
    print(
        "complete backend=%s best_epoch=%d best_validation_auc=%.6f duration=%.1fs"
        % (classifier.backend, best_epoch, summary["best_validation_auc"], duration),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
