"""Command-line orchestration shared by all paired architecture entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from torch import nn

from .config import DEFAULT_DATA_ROOT, DataConfig, TrainingConfig
from .data import prepare_dataset
from .training import evaluate_model, set_seed, train_model


Task = Literal["classification", "regression"]


def main_for_architecture(
    architecture_name: str,
    *,
    task: Task,
    model_factory: Callable[..., nn.Module],
    architecture_config: Mapping[str, Any] | None = None,
) -> int:
    selected_architecture_config = dict(architecture_config or {})
    model_config = dict(selected_architecture_config.get("model", {}))
    representation_config = dict(
        selected_architecture_config.get("representation", {})
    )
    classification_config = dict(
        selected_architecture_config.get("classification", {})
    )
    training_defaults = dict(selected_architecture_config.get("training", {}))
    known_training_keys = {
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "seed",
        "deterministic",
        "use_amp",
        "amp_precision",
        "num_workers",
        "device",
    }
    unknown_training_keys = set(training_defaults) - known_training_keys
    if unknown_training_keys:
        raise ValueError(
            "unsupported architecture training settings: "
            + ", ".join(sorted(unknown_training_keys))
        )
    parser = argparse.ArgumentParser(
        description=f"Train {architecture_name} for MJD {task}."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=training_defaults.get("batch_size", 64)
    )
    parser.add_argument(
        "--epochs", type=int, default=training_defaults.get("epochs", 50)
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=training_defaults.get("learning_rate", 5.0e-4),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=training_defaults.get("weight_decay", 1.0e-4),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=training_defaults.get("gradient_clip_norm", 1.0),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=training_defaults.get("early_stopping_patience", 5),
    )
    parser.add_argument(
        "--num-workers", type=int, default=training_defaults.get("num_workers", 0)
    )
    parser.add_argument("--device", default=training_defaults.get("device", "auto"))
    parser.add_argument("--seed", type=int, default=training_defaults.get("seed", 42))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from a checkpoint (older weight-only checkpoints are supported).",
    )
    if not architecture_config:
        parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--baseline-samples", type=int, default=200)
    parser.add_argument("--regression-waveform-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-classification-amplitude-normalization",
        action="store_true",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use at most 512 events per split and one epoch.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (
        project_root / "outputs" / task / architecture_name
    )
    data_config = DataConfig(
        data_root=args.data_root,
        baseline_samples=args.baseline_samples,
        classification_amplitude_normalization=(
            not args.no_classification_amplitude_normalization
        ),
        regression_waveform_scale=args.regression_waveform_scale,
        seed=args.seed,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=training_defaults.get(
            "early_stopping_min_delta", 0.0
        ),
        deterministic=training_defaults.get("deterministic", False),
        use_amp=training_defaults.get("use_amp", False) and not args.no_amp,
        amp_precision=training_defaults.get("amp_precision", "auto"),
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    set_seed(args.seed, training_config.deterministic)
    data = prepare_dataset(
        task=task,
        data_config=data_config,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        limit_per_split=512 if args.smoke else None,
    )
    if architecture_config:
        model = model_factory(task)
    else:
        model = model_factory(task, args.base_channels)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "architecture": architecture_name,
        "task": task,
        "data": data_config.to_dict(),
        "training": training_config.to_dict(),
        "counts": data.counts,
        "smoke": args.smoke,
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
    }
    if task == "classification":
        run_config["classification_target"] = "clean_vs_non_clean"
        if classification_config:
            run_config["classification_auxiliary"] = classification_config
    if architecture_config:
        run_config["representation"] = representation_config
        run_config["model"] = model_config
    else:
        run_config["base_channels"] = args.base_channels
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    train_model(
        model,
        data.train_loader,
        data.validation_loader,
        task=task,
        config=training_config,
        output_dir=output_dir,
        resume_from=args.resume_from,
    )
    metrics = evaluate_model(
        model,
        data.test_loader,
        task=task,
        device=training_config.device,
        output_dir=output_dir,
        use_amp=training_config.use_amp,
        amp_precision=training_config.amp_precision,
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Saved {task} outputs to {output_dir}")
    return 0


__all__ = ["main_for_architecture"]
