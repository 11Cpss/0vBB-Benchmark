"""Unified train, validation, and test entry point for every registered model."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import (
    ARCHITECTURES,
    CLASSIFICATION_LABELS,
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_ROOT,
    ENERGY_TARGET,
    ENERGY_UNIT,
    PROJECT_ROOT,
    TASKS,
    TRANSFORMER_ARCHITECTURE_IDS,
    DataConfig,
)
from .data import prepare_dataset
from .evaluation import (
    EVALUATION_MANIFEST,
    assert_same_provenance,
    dataset_provenance,
    file_sha256,
)
from .models import build_model, get_model_spec, registered_models
from .training import evaluate_model, set_seed, train_model


MODES = ("train", "validate", "test")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _project_output(task: str, model_id: str) -> Path:
    output = (DEFAULT_OUTPUT_ROOT / task / model_id).resolve()
    try:
        output.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"configured output escapes the writable project: {output}") from error
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the strict SuperNEMO classification or energy workflow."
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", required=True, choices=registered_models())
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Read-only directory containing the four canonical SuperNEMO HDF5 files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Wing's authoritative split manifest (or a synthetic test manifest).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for validate/test (default: outputs/<task>/<model>/best.pt).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Execution device override (for example cpu or cuda); training settings are unchanged.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume --mode train from the atomically committed last.pt checkpoint.",
    )
    return parser


@contextmanager
def _run_lock(output_dir: Path, *, exclusive: bool) -> Any:
    """Lock one run across checkpoint updates or complete evaluation reads."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".train.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            action = "training or evaluation" if exclusive else "training"
            raise RuntimeError(f"another {action} process owns {output_dir}") from error
        if exclusive:
            handle.seek(0)
            handle.truncate()
            handle.write(f"training_pid={os.getpid()}\n")
            handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _training_lock(output_dir: Path) -> Any:
    return _run_lock(output_dir, exclusive=True)


def _evaluation_lock(output_dir: Path) -> Any:
    return _run_lock(output_dir, exclusive=False)


def _load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    task: str,
    model_id: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid SuperNEMO checkpoint: {checkpoint_path}")
    if checkpoint.get("task") != task:
        raise ValueError(
            f"checkpoint task {checkpoint.get('task')!r} does not match requested {task!r}"
        )
    checkpoint_model = checkpoint.get("architecture_id")
    if checkpoint_model is not None and checkpoint_model != model_id:
        raise ValueError(
            f"checkpoint architecture {checkpoint_model!r} does not match {model_id!r}"
        )
    strict_checkpoint = model_id in TRANSFORMER_ARCHITECTURE_IDS
    if strict_checkpoint != (
        getattr(model, "tokenization_config", None) is not None
    ):
        raise RuntimeError("Transformer registry and model tokenization metadata disagree")
    if strict_checkpoint:
        if checkpoint.get("kind") != "best" or int(checkpoint.get("schema_version", -1)) != 2:
            raise ValueError("Transformer evaluation requires a schema-v2 best checkpoint")
        if checkpoint.get("architecture_id") != model_id:
            raise ValueError("checkpoint architecture must exactly match the Transformer ID")
        if checkpoint.get("model_name") != getattr(model, "model_name", None):
            raise ValueError("checkpoint model name does not match the registry")
        if checkpoint.get("input_kind") != getattr(model, "input_kind", None):
            raise ValueError("checkpoint input kind does not match the registry")
        if checkpoint.get("model_config") != model_config:
            raise ValueError("checkpoint model configuration does not match the registry")
        source = inspect.getsourcefile(model.__class__)
        if source is None:
            raise ValueError("cannot identify current model source")
        expected_source = {
            "path": str(Path(source).resolve()),
            "sha256": file_sha256(source),
        }
        if checkpoint.get("model_source") != expected_source:
            raise ValueError("checkpoint model source fingerprint does not match")
        training_source = inspect.getsourcefile(train_model)
        if training_source is None:
            raise ValueError("cannot identify current training source")
        expected_training_source = {
            "path": str(Path(training_source).resolve()),
            "sha256": file_sha256(training_source),
        }
        if checkpoint.get("training_source") != expected_training_source:
            raise ValueError("checkpoint training source fingerprint does not match")
        if checkpoint.get("selection_split") != "validation":
            raise ValueError("checkpoint selection split must be validation")
        if checkpoint.get("selection_metric") != "energy_matched_auc":
            raise ValueError("checkpoint selection metric must be energy_matched_auc")
        epoch = checkpoint.get("epoch")
        score = checkpoint.get("score")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("checkpoint epoch must be a positive integer")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("checkpoint selection score must be finite")
        history = checkpoint.get("history")
        if not isinstance(history, list) or len(history) != epoch:
            raise ValueError("checkpoint history must align exactly with its epoch")
        recorded_training = dict(checkpoint.get("training_config", {}))
        expected_training = dict(training_config)
        # Device placement is an execution override, not a model-selection
        # hyperparameter; allow evaluation on CPU or another CUDA device.
        recorded_training.pop("device", None)
        expected_training.pop("device", None)
        if recorded_training != expected_training:
            raise ValueError("checkpoint training configuration does not match the registry")
        assert_same_provenance(
            checkpoint.get("dataset_provenance"),
            provenance,
            context="checkpoint",
        )
    else:
        # Checkpoints written by runs already active when provenance support was
        # added lack these fields. Preserve their usability while marking the
        # inferred metadata; all new checkpoints are schema v2.
        checkpoint = dict(checkpoint)
        if "dataset_provenance" in checkpoint:
            assert_same_provenance(
                checkpoint["dataset_provenance"], provenance, context="checkpoint"
            )
        else:
            checkpoint["dataset_provenance"] = provenance
            checkpoint["checkpoint_provenance_status"] = "inferred_legacy"
        checkpoint.setdefault("model_config", model_config)
        checkpoint.setdefault("selection_split", "validation")
        checkpoint.setdefault(
            "selection_metric", "auc" if task == "classification" else "rmse_kev"
        )
    checkpoint["_checkpoint_file_sha256"] = hashlib.sha256(
        checkpoint_bytes
    ).hexdigest()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    architecture = ARCHITECTURES[args.model]
    if args.task not in architecture.tasks:
        raise ValueError(
            f"{args.model!r} supports only: {', '.join(architecture.tasks)}"
        )
    if args.resume and args.mode != "train":
        raise ValueError("--resume is only valid with --mode train")
    reference_spec = get_model_spec(args.model)
    if reference_spec.input_kind != architecture.input_kind:
        raise RuntimeError("model registry input kind changed after configuration validation")
    training_config = architecture.training.with_device(args.device)
    data_config = DataConfig(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        seed=training_config.seed,
    )
    set_seed(training_config.seed, training_config.deterministic)
    data = prepare_dataset(
        task=args.task,
        input_kind=architecture.input_kind,
        data_config=data_config,
        tokenization_config=architecture.tokenization,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
    )
    model = build_model(args.model, args.task)
    output_dir = _project_output(args.task, args.model)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else output_dir / "best.pt"
    )
    representation_config = (
        None
        if architecture.tokenization is None
        else architecture.tokenization.to_dict()
    )
    provenance = dataset_provenance(
        manifest_path=data.manifest_path,
        data_root=data_config.data_root,
        task=args.task,
        counts=data.counts,
        data_config=data_config.to_dict(),
        input_kind=architecture.input_kind,
        representation_config=representation_config,
    )

    if args.mode == "train":
        if args.checkpoint is not None:
            raise ValueError("--checkpoint is only valid for validate or test")
        if args.resume:
            if not (output_dir / "run_config.json").is_file():
                raise FileNotFoundError(
                    f"cannot resume without {output_dir / 'run_config.json'}"
                )
        elif output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty training directory: {output_dir}"
            )
        run_config = {
            "dataset": "SuperNEMO",
            "task": args.task,
            "architecture_id": architecture.architecture_id,
            "model_name": architecture.model_name,
            "input_kind": architecture.input_kind,
            "model": dict(architecture.model),
            "tokenization": representation_config,
            "training": training_config.to_dict(),
            "supported_tasks": list(architecture.tasks),
            "data": data_config.to_dict(),
            "manifest_path": str(data.manifest_path),
            "counts": data.counts,
            "dataset_provenance": provenance,
            "evaluation_manifest": str(EVALUATION_MANIFEST),
            "input_representation": (
                "tracker_tokens"
                if architecture.tokenization is not None
                else "topology_only"
            ),
            "classification_label_mapping": (
                dict(CLASSIFICATION_LABELS) if args.task == "classification" else None
            ),
            "energy_target": (
                {"definition": ENERGY_TARGET, "unit": ENERGY_UNIT}
                if args.task == "energy"
                else None
            ),
        }
        with _training_lock(output_dir):
            run_config_path = output_dir / "run_config.json"
            if args.resume:
                recorded_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                if recorded_config != run_config:
                    raise ValueError("run_config.json changed; refusing an inexact resume")
            else:
                _write_json(run_config_path, run_config)
            train_model(
                model,
                data.train_loader,
                data.validation_loader,
                task=args.task,
                config=training_config,
                output_dir=output_dir,
                model_config=dict(architecture.model),
                provenance=provenance,
                resume=args.resume,
            )
            last_checkpoint_path = output_dir / "last.pt"
            best_checkpoint_path = output_dir / "best.pt"
            last_checkpoint = torch.load(
                last_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            completed_epoch = int(last_checkpoint.get("epoch", 0))
            stale_epochs = int(last_checkpoint.get("stale_epochs", 0))
            reached_epoch_limit = completed_epoch >= training_config.epochs
            reached_early_stopping = (
                stale_epochs >= training_config.early_stopping_patience
            )
            if not (reached_epoch_limit or reached_early_stopping):
                raise RuntimeError(
                    "train_model returned without reaching the epoch limit or "
                    "early-stopping condition"
                )
            _write_json(
                output_dir / "training_complete.json",
                {
                    "schema_version": 1,
                    "architecture_id": architecture.architecture_id,
                    "completed_epoch": completed_epoch,
                    "best_epoch": int(last_checkpoint["best_epoch"]),
                    "stale_epochs": stale_epochs,
                    "stop_reason": (
                        "epoch_limit"
                        if reached_epoch_limit
                        else "early_stopping"
                    ),
                    "last_checkpoint_sha256": file_sha256(last_checkpoint_path),
                    "best_checkpoint_sha256": file_sha256(best_checkpoint_path),
                },
            )
        print(f"Saved training outputs to {output_dir}")
        return 0

    with _evaluation_lock(output_dir):
        checkpoint = _load_checkpoint(
            model,
            checkpoint_path,
            task=args.task,
            model_id=args.model,
            model_config=dict(architecture.model),
            training_config=architecture.training.to_dict(),
            provenance=provenance,
        )
        if args.mode == "validate":
            loader = data.validation_loader
            split_name = "validation"
        else:
            loader = data.test_loader
            split_name = "test"
        metrics = evaluate_model(
            model,
            loader,
            task=args.task,
            config=training_config,
            output_dir=output_dir,
            split_name=split_name,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            model_config=dict(architecture.model),
            provenance=provenance,
        )
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Saved {split_name} outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
