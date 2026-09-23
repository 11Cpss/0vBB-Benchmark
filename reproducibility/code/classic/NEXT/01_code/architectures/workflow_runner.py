#!/usr/bin/env python3
"""Run one model-zoo architecture with the standalone EnergyBench workflow.

This module is the bridge between the repository's heterogeneous model zoo and
``evalutaions_workflow``.  It deliberately keeps the standalone workflow's
event-count split, training defaults, evaluation metrics, and artifact layout
while allowing a custom ``inputs`` mapping for point, graph, sequence, 3-D,
topology, sparse, and hybrid models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch import nn


ARCHITECTURES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ARCHITECTURES_ROOT.parents[1]
WORKFLOW_ROOT = PROJECT_ROOT / "evalutaions_workflow"
for candidate in (WORKFLOW_ROOT, PROJECT_ROOT / "src", ARCHITECTURES_ROOT):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from simple_energybench import (  # noqa: E402
    EvaluationConfig,
    TrainingConfig,
    evaluate_classification,
    evaluate_regression,
    prepare_dataset,
    set_seed,
    train_model,
)

from next_alt.models.classic_topology import (  # noqa: E402
    TOPOLOGY_FEATURE_NAMES,
    TopologyBoostedTreeClassifier,
    TopologyFeatureExtractor,
)

from workflow_data import build_architecture_loaders  # noqa: E402
from workflow_models import (  # noqa: E402
    MicrobatchModel,
    build_classifier,
    build_regressor,
    get_spec,
)


DEFAULT_DATA_ROOT = Path("/home/klz/Data/zeronu_benchmark/NEXT")
TASKS = ("classification", "regression")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    """Atomically write a JSON run record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_architecture_config(architecture_id: str) -> dict[str, Any]:
    """Load model/representation settings without inheriting old training knobs."""

    path = ARCHITECTURES_ROOT / architecture_id / "config.yaml"
    if not path.is_file():
        return {
            "architecture_id": architecture_id,
            "data": {"root": str(DEFAULT_DATA_ROOT)},
            "representation": {},
            "model": {},
            "config_path": None,
        }
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"configuration root must be a mapping: {path}")
    result = dict(payload)
    declared = result.get("architecture_id")
    if declared != architecture_id:
        raise ValueError(
            f"{path} declares architecture_id={declared!r}, expected {architecture_id!r}"
        )
    result["data"] = dict(result.get("data") or {})
    result["representation"] = dict(result.get("representation") or {})
    result["model"] = dict(result.get("model") or {})
    result["config_path"] = str(path)
    return result


def make_training_config(
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    patience: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
) -> TrainingConfig:
    """Return standard settings, applying only explicit command-line overrides."""

    values = asdict(TrainingConfig())
    overrides = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "early_stopping_patience": patience,
        "device": device,
        "num_workers": num_workers,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return TrainingConfig(**values)


def _prepared_loaders(
    architecture_id: str,
    prepared: Any,
    data_root: Path,
    architecture_config: Mapping[str, Any],
    training_config: TrainingConfig,
) -> tuple[Any, Any, Any]:
    spec = get_spec(architecture_id)
    if spec.input_kind == "projection_tensor":
        return (
            prepared.train_loader,
            prepared.validation_loader,
            prepared.test_loader,
        )
    return build_architecture_loaders(
        prepared,
        data_root=data_root,
        input_kind=spec.input_kind,
        representation=architecture_config.get("representation", {}),
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        shuffle_buffer_size=512,
    )


def _build_torch_model(
    architecture_id: str,
    task: str,
    model_config: Mapping[str, Any],
    microbatch_size: int | None,
) -> nn.Module:
    model = (
        build_classifier(architecture_id, model_config)
        if task == "classification"
        else build_regressor(architecture_id, model_config)
    )
    spec = get_spec(architecture_id)
    chunk_size = spec.microbatch_size if microbatch_size is None else microbatch_size
    return MicrobatchModel(
        model,
        microbatch_size=chunk_size,
        architecture_id=architecture_id,
        task=task,
    )


def _classic_collect(
    loader: Iterable[Mapping[str, Any]], extractor: TopologyFeatureExtractor
) -> tuple[np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        inputs = batch["inputs"]
        matrices.append(
            extractor.extract_batch(
                inputs["coords"], inputs["features"], inputs["mask"]
            )
        )
        raw_labels = batch["label"]
        if isinstance(raw_labels, torch.Tensor):
            raw_labels = raw_labels.detach().cpu().numpy()
        labels.append(np.asarray(raw_labels, dtype=np.int64).reshape(-1))
    if not matrices:
        raise RuntimeError("classic training loader produced no events")
    return np.concatenate(matrices), np.concatenate(labels)


def _classic_logits(classifier: TopologyBoostedTreeClassifier, matrix: np.ndarray) -> np.ndarray:
    if classifier.model is None:
        raise RuntimeError("classic classifier has not been fitted")
    if classifier.backend == "numpy_hist_gbdt":
        return np.asarray(classifier.model.predict_logits(matrix), dtype=np.float64)

    import xgboost as xgb

    probabilities = classifier.model.predict(
        xgb.DMatrix(
            np.asarray(matrix, dtype=np.float32),
            feature_names=list(TOPOLOGY_FEATURE_NAMES),
        ),
        iteration_range=(0, classifier.best_iteration_ + 1),
    )
    epsilon = np.finfo(np.float32).eps
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(probabilities) - np.log1p(-probabilities)


class ClassicInferenceModule(nn.Module):
    """Expose a fitted boosted tree through the EnergyBench model contract."""

    def __init__(
        self,
        classifier: TopologyBoostedTreeClassifier,
        extractor: TopologyFeatureExtractor,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.extractor = extractor

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        matrix = self.extractor.extract_batch(
            inputs["coords"], inputs["features"], inputs["mask"]
        )
        logits = _classic_logits(self.classifier, matrix)
        return torch.from_numpy(logits.astype(np.float32, copy=False))


def _classic_training_plot(history: list[Mapping[str, Any]], destination: Path) -> None:
    from next_alt.metrics import save_history_plot

    save_history_plot(history, destination, "classic_001_topology_xgboost")


def _run_classic(
    *,
    architecture_config: Mapping[str, Any],
    train_loader: Any,
    validation_loader: Any,
    test_loader: Any,
    training_dir: Path,
    evaluation_dir: Path,
    evaluation_config: EvaluationConfig,
    estimator_rounds: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_config = dict(architecture_config.get("model", {}))
    backend = str(model_config.pop("backend", "auto"))
    extractor_config = dict(model_config.pop("extractor", {}))
    estimator_config = dict(model_config.pop("estimator", {}))
    if model_config:
        raise ValueError(f"unknown classic model keys: {sorted(model_config)}")
    if estimator_rounds is not None:
        estimator_config["n_estimators"] = int(estimator_rounds)
        estimator_config["early_stopping_rounds"] = min(
            int(estimator_config.get("early_stopping_rounds", 12)),
            int(estimator_rounds),
        )

    artifacts = {
        "best_model": training_dir / "best_model.pt",
        "last_model": training_dir / "last_model.pt",
        "history": training_dir / "history.json",
        "history_plot": training_dir / "training_history.png",
    }
    existing = [path for path in artifacts.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "classic training outputs already exist: " + ", ".join(map(str, existing))
        )
    training_dir.mkdir(parents=True, exist_ok=True)
    extractor = TopologyFeatureExtractor(**extractor_config)
    train_matrix, train_labels = _classic_collect(train_loader, extractor)
    validation_matrix, validation_labels = _classic_collect(
        validation_loader, extractor
    )
    classifier = TopologyBoostedTreeClassifier(backend=backend, **estimator_config)
    classifier.fit(
        train_matrix,
        train_labels,
        validation_data=(validation_matrix, validation_labels),
    )
    if not classifier.history_ or classifier.best_iteration_ < 0:
        raise RuntimeError("boosted-tree training did not select a valid round")
    best_epoch = classifier.best_iteration_ + 1
    best_record = classifier.history_[classifier.best_iteration_]
    common = {
        "task": "classification",
        "architecture_id": "classic_001_topology_xgboost",
        "model_name": "TopologyBoostedTreeClassifier",
        "backend": classifier.backend,
        "feature_extractor": extractor.config_dict(),
    }
    torch.save(
        {
            **common,
            "epoch": best_epoch,
            "model": classifier.checkpoint_state(tree_limit=best_epoch),
        },
        artifacts["best_model"],
    )
    torch.save(
        {
            **common,
            "epoch": len(classifier.history_),
            "model": classifier.checkpoint_state(tree_limit=len(classifier.history_)),
        },
        artifacts["last_model"],
    )
    history_result = {
        "task": "classification",
        "backend": classifier.backend,
        "selection_metric": "val_auc",
        "best_metric": float(best_record["validation"]["auc"]),
        "best_epoch": best_epoch,
        "epochs_completed": len(classifier.history_),
        "stopped_early": len(classifier.history_)
        < int(estimator_config.get("n_estimators", 500)),
        "history": classifier.history_,
        "artifact_paths": {key: str(value) for key, value in artifacts.items()},
    }
    write_json(artifacts["history"], history_result)
    _classic_training_plot(classifier.history_, artifacts["history_plot"])

    inference_model = ClassicInferenceModule(classifier, extractor)
    metrics = evaluate_classification(
        inference_model,
        test_loader,
        device="cpu",
        output_dir=evaluation_dir,
        config=evaluation_config,
    )
    return history_result, metrics


def run_architecture(
    architecture_id: str,
    *,
    task: str = "classification",
    output_dir: str | Path,
    data_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    max_files_per_class: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    patience: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    microbatch_size: int | None = None,
    estimator_rounds: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Prepare data, train one model, evaluate its best weights, and record a run."""

    task = str(task).strip().lower()
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    spec = get_spec(architecture_id)
    if task == "regression" and not spec.supports_regression:
        raise ValueError(f"{architecture_id} does not define a regression model")
    if task == "regression" and spec.backend == "xgboost":
        raise ValueError("the classic boosted-tree architecture is classification-only")

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"run output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started_clock = time.monotonic()
    architecture_config = load_architecture_config(architecture_id)
    root = Path(
        data_root
        or architecture_config.get("data", {}).get("root")
        or DEFAULT_DATA_ROOT
    ).expanduser().resolve()
    training_config = make_training_config(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
        num_workers=num_workers,
    )
    evaluation_config = EvaluationConfig()
    set_seed(training_config.seed, training_config.deterministic)
    split_path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else destination.parent / "event_split.json"
    )
    mode = "classification" if task == "classification" else "regression_energy"
    prepared = prepare_dataset(
        root,
        batch_size=training_config.batch_size,
        mode=mode,
        seed=training_config.seed,
        num_workers=training_config.num_workers,
        manifest_path=split_path,
        max_files_per_class=max_files_per_class,
        shuffle_buffer_size=512,
        verbose=verbose,
    )
    train_loader, validation_loader, test_loader = _prepared_loaders(
        architecture_id,
        prepared,
        root,
        architecture_config,
        training_config,
    )
    training_dir = destination / "training"
    evaluation_dir = destination / "evaluation"
    if spec.backend == "xgboost":
        history, metrics = _run_classic(
            architecture_config=architecture_config,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            training_dir=training_dir,
            evaluation_dir=evaluation_dir,
            evaluation_config=evaluation_config,
            estimator_rounds=(
                training_config.epochs
                if estimator_rounds is None
                else estimator_rounds
            ),
        )
        parameter_count = None
        model_config = architecture_config.get("model", {})
    else:
        model_config = dict(architecture_config.get("model", {}))
        model = _build_torch_model(
            architecture_id,
            task,
            model_config,
            microbatch_size,
        )
        actual_model_config = model.config_dict()
        if actual_model_config:
            model_config = actual_model_config
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        history = train_model(
            model,
            train_loader,
            validation_loader,
            config=training_config,
            task=task,
            output_dir=training_dir,
        )
        if task == "classification":
            metrics = evaluate_classification(
                model,
                test_loader,
                device=training_config.device,
                output_dir=evaluation_dir,
                config=evaluation_config,
            )
        else:
            metrics = evaluate_regression(
                model,
                test_loader,
                device=training_config.device,
                output_dir=evaluation_dir,
                config=evaluation_config,
            )

    completed_at = utc_now()
    result = {
        "status": "DONE",
        "architecture_id": architecture_id,
        "model_name": spec.model_name,
        "backend": spec.backend,
        "input_kind": spec.input_kind,
        "task": task,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": time.monotonic() - started_clock,
        "data_root": str(root),
        "event_split_manifest": str(split_path),
        "max_files_per_class": max_files_per_class,
        "data_counts": prepared.counts,
        "training_config": asdict(training_config),
        "evaluation_config": evaluation_config.to_dict(),
        "representation_config": architecture_config.get("representation", {}),
        "model_config": model_config,
        "parameter_count": parameter_count,
        "best_epoch": history.get("best_epoch"),
        "best_validation_metric": history.get("best_metric"),
        "epochs_completed": history.get("epochs_completed"),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "task",
                "n_events",
                "auc",
                "matched_auc",
                "matched_auc_status",
                "energy_independence_score",
                "ers",
                "mae",
                "rmse",
                "status",
            )
            if key in metrics
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_requested": training_config.device,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "artifacts": {
            "training": str(training_dir),
            "evaluation": str(evaluation_dir),
            "run_summary": str(destination / "run_summary.json"),
        },
    }
    write_json(destination / "run_summary.json", result)
    return result


def _positive_int(value: str) -> int:
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return converted


def main_for_architecture(
    architecture_id: str,
    task: str = "classification",
    argv: list[str] | None = None,
) -> int:
    """CLI used by every small per-architecture entry point."""

    parser = argparse.ArgumentParser(
        description=f"Train and evaluate {architecture_id} with Simple EnergyBench."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-files-per-class", type=_positive_int)
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=_positive_int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--microbatch-size", type=_positive_int)
    parser.add_argument("--estimator-rounds", type=_positive_int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_architecture(
        architecture_id,
        task=task,
        output_dir=args.output_dir,
        data_root=args.data,
        manifest_path=args.manifest,
        max_files_per_class=args.max_files_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
        num_workers=args.num_workers,
        microbatch_size=args.microbatch_size,
        estimator_rounds=args.estimator_rounds,
        verbose=not args.quiet,
    )
    metric = result["metrics"]
    print(
        f"DONE {architecture_id} task={task} "
        f"best={result['best_validation_metric']} metrics={metric}",
        flush=True,
    )
    return 0


__all__ = [
    "DEFAULT_DATA_ROOT",
    "load_architecture_config",
    "main_for_architecture",
    "make_training_config",
    "run_architecture",
]
