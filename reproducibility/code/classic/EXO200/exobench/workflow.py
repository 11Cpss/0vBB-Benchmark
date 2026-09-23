"""Single CLI workflow matching MJD's train/validate/test orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from torch import nn

from .config import (
    BACKGROUND_LABEL,
    CLASS_NAMES,
    DATASET_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    PROJECT_ROOT,
    SIGNAL_LABEL,
    SIGNAL_NCCL,
    TASK,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from .data import prepare_dataset
from .training import evaluate_model, set_seed, train_model


def main_for_architecture(
    architecture_name: str,
    *,
    model_factory: Callable[..., nn.Module],
    architecture_config: Mapping[str, Any] | None = None,
) -> int:
    uses_architecture_config = architecture_config is not None
    selected_architecture_config = dict(architecture_config or {})
    architecture_model_config = dict(
        selected_architecture_config.get("model", {})
    )
    representation_config = dict(
        selected_architecture_config.get("representation", {})
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
    defaults = TrainingConfig()
    data_defaults = DataConfig()
    parser = argparse.ArgumentParser(
        description=f"Train {architecture_name} for EXO-200 binary classification."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=training_defaults.get("batch_size", defaults.batch_size),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=training_defaults.get("epochs", defaults.epochs),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=training_defaults.get("learning_rate", defaults.learning_rate),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=training_defaults.get("weight_decay", defaults.weight_decay),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=training_defaults.get(
            "gradient_clip_norm", defaults.gradient_clip_norm
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=training_defaults.get(
            "early_stopping_patience", defaults.early_stopping_patience
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=training_defaults.get("num_workers", defaults.num_workers),
    )
    parser.add_argument(
        "--device",
        default=training_defaults.get("device", defaults.device),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=training_defaults.get("seed", defaults.seed),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    if not uses_architecture_config:
        parser.add_argument(
            "--base-channels", type=int, default=ModelConfig().base_channels
        )
    parser.add_argument(
        "--baseline-samples", type=int, default=data_defaults.baseline_samples
    )
    parser.add_argument(
        "--no-classification-amplitude-normalization",
        action="store_true",
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir or (DEFAULT_OUTPUT_ROOT / architecture_name)
    ).expanduser().resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(
            f"output_dir must stay inside the writable project root {PROJECT_ROOT}"
        )
    data_config = DataConfig(
        data_root=args.data_root,
        baseline_samples=args.baseline_samples,
        classification_amplitude_normalization=(
            not args.no_classification_amplitude_normalization
        ),
        seed=args.seed,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=training_defaults.get(
            "early_stopping_min_delta", defaults.early_stopping_min_delta
        ),
        deterministic=training_defaults.get(
            "deterministic", defaults.deterministic
        ),
        use_amp=(
            training_defaults.get("use_amp", defaults.use_amp)
            and not args.no_amp
        ),
        amp_precision=training_defaults.get(
            "amp_precision", defaults.amp_precision
        ),
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    set_seed(args.seed, training_config.deterministic)
    data = prepare_dataset(
        data_config=data_config,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
    )
    # Require both fixed classes before model creation, forward, optimizer
    # construction, or output mutation.
    data.require_two_classes()

    if uses_architecture_config:
        model = model_factory()
        serialized_model_config = architecture_model_config
    else:
        model_config = ModelConfig(
            input_channels=data_config.input_shape[0],
            base_channels=args.base_channels,
            num_classes=data_config.num_classes,
        )
        model = model_factory(**model_config.to_dict())
        serialized_model_config = model_config.to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": DATASET_NAME,
        "architecture": architecture_name,
        "task": TASK,
        "data": data_config.to_dict(),
        "model": serialized_model_config,
        "training": training_config.to_dict(),
        "counts": data.counts,
        "class_counts": data.class_counts,
        "split_runs": data.runs,
        "split_overlap_counts": data.overlap_counts,
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
        "classification_target": {
            "source_field": data_config.label_field,
            "signal": {
                "condition": f"nccl == {SIGNAL_NCCL}",
                "label": SIGNAL_LABEL,
                "name": CLASS_NAMES[SIGNAL_LABEL],
            },
            "background": {
                "condition": f"nccl > {SIGNAL_NCCL}",
                "label": BACKGROUND_LABEL,
                "name": CLASS_NAMES[BACKGROUND_LABEL],
            },
            "positive_class": CLASS_NAMES[BACKGROUND_LABEL],
        },
    }
    if uses_architecture_config:
        run_config["representation"] = representation_config
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    train_model(
        model,
        data.train_loader,
        data.validation_loader,
        config=training_config,
        output_dir=output_dir,
        resume_from=args.resume_from,
    )
    metrics = evaluate_model(
        model,
        data.test_loader,
        device=training_config.device,
        output_dir=output_dir,
        use_amp=training_config.use_amp,
        amp_precision=training_config.amp_precision,
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Saved classification outputs to {output_dir}")
    return 0


__all__ = ["main_for_architecture"]
