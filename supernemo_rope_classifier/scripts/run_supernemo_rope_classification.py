"""One cell of the SuperNEMO RoPE classification benchmark (2nubb vs Bi214).

Trains a ``NEXTTransformerClassifier`` on SuperNEMO tracker-hit tokens with the
shared, unchanged EnergyBench ``train_model`` / ``evaluate_classification``,
then writes the run tree::

    <output_root>/<run_id>/
    ├── run_config.json
    ├── training/     best_model.pt, last_model.pt, history.json, training_history.png
    ├── evaluation/   metrics.json, results.csv, predictions.npz, *.png
    └── run_summary.json

The default position encoding is ``rope``. ``coordinate_mlp`` and
``fourier_xyz`` run through the same code path (same tokens, split, trainer,
and evaluator) for like-for-like controls.

All settings come from ``SUPERNEMO_ROPE_*`` environment variables; see the
README for the list.

Example (CPU smoke on a tiny slice of the real data)::

    SUPERNEMO_ROPE_TOKENIZATION=voxel SUPERNEMO_ROPE_MAX_TRAIN_EVENTS=4000 \\
    SUPERNEMO_ROPE_MAX_VALIDATION_EVENTS=1000 SUPERNEMO_ROPE_MAX_TEST_EVENTS=2000 \\
    SUPERNEMO_ROPE_EPOCHS=1 SUPERNEMO_ROPE_DEVICE=cpu \\
    python supernemo_rope_classifier/scripts/run_supernemo_rope_classification.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path


def _find_project_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (
            (candidate / "next_detector" / "next_transformer").is_dir()
            and (candidate / "evalutaions_workflow" / "simple_energybench").is_dir()
            and (candidate / "supernemo_rope_classifier" / "supernemorope_bench").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "could not locate the project root (needs next_detector/, "
        "evalutaions_workflow/ and supernemo_rope_classifier/)"
    )


PROJECT_ROOT = _find_project_root()
for _source_root in (
    PROJECT_ROOT / "evalutaions_workflow",
    PROJECT_ROOT / "next_detector",
    PROJECT_ROOT / "supernemo_rope_classifier",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

import torch  # noqa: E402
from simple_energybench import (  # noqa: E402
    EvaluationConfig,
    TrainingConfig,
    evaluate_classification,
    set_seed,
    train_model,
)

from supernemo_rope_transformer import (  # noqa: E402
    DEFAULT_SUPERNEMO_ROPE_BASE,
    POSITION_ENCODINGS,
    SuperNEMOTrackerTokenizationConfig,
    build_supernemo_transformer,
    rope_frequency_table,
)
from simple_energybench.config import CANONICAL_ENERGY_MAX_KEV  # noqa: E402
from supernemorope_bench import (  # noqa: E402
    DataConfig,
    EnergyWindowLoader,
    prepare_dataset,
)
from supernemorope_bench.config import KEV_PER_MEV  # noqa: E402


TOKENIZATIONS = ("sampled_hits", "voxel", "summary_features")

# Published SuperNEMO Transformer tokenization settings (max_tokens 128 for
# sampled-hit and voxel tokens, 16 for summary tokens, 60 mm voxels).
PUBLISHED_TOKENIZATION = {
    "sampled_hits": {"max_tokens": 128},
    "voxel": {"max_tokens": 128, "voxel_size_mm": 60.0},
    "summary_features": {"max_tokens": 16},
}

# Published SuperNEMO training settings.
PUBLISHED_EARLY_STOPPING_PATIENCE = 12


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value is None or not value.strip() else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, not {value!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    data_dir = _env_str("SUPERNEMO_ROPE_DATA_DIR", "/vast/pvenkata/vis224/SuperNEMO")
    output_root = (
        Path(
            _env_str(
                "SUPERNEMO_ROPE_OUTPUT_ROOT",
                str(
                    PROJECT_ROOT
                    / "supernemo_rope_classifier"
                    / "results"
                    / "supernemo_rope_classification_v1"
                ),
            )
        )
        .expanduser()
        .resolve()
    )
    tokenization = _env_str("SUPERNEMO_ROPE_TOKENIZATION", "voxel")
    position_encoding = _env_str("SUPERNEMO_ROPE_POSITION_ENCODING", "rope")
    rope_base = _env_float("SUPERNEMO_ROPE_BASE", DEFAULT_SUPERNEMO_ROPE_BASE)
    run_suffix = _env_str("SUPERNEMO_ROPE_RUN_SUFFIX", "")
    run_id = _env_str(
        "SUPERNEMO_ROPE_RUN_ID",
        f"classification__{tokenization}__{position_encoding}{run_suffix}",
    )
    seed = _env_int("SUPERNEMO_ROPE_SEED", 42)
    epochs = _env_int("SUPERNEMO_ROPE_EPOCHS", 50)
    batch_size = _env_int("SUPERNEMO_ROPE_BATCH_SIZE", 64)
    patience = _env_int("SUPERNEMO_ROPE_PATIENCE", PUBLISHED_EARLY_STOPPING_PATIENCE)
    device = _env_str("SUPERNEMO_ROPE_DEVICE", "auto")
    num_workers = _env_int("SUPERNEMO_ROPE_NUM_WORKERS", 0)
    use_amp = _env_flag("SUPERNEMO_ROPE_USE_AMP", True)
    overwrite = _env_flag("SUPERNEMO_ROPE_OVERWRITE", False)

    if tokenization not in TOKENIZATIONS:
        raise ValueError(
            f"SUPERNEMO_ROPE_TOKENIZATION must be one of {TOKENIZATIONS}, got {tokenization!r}"
        )
    if position_encoding not in POSITION_ENCODINGS:
        raise ValueError(
            "SUPERNEMO_ROPE_POSITION_ENCODING must be one of "
            f"{POSITION_ENCODINGS}, got {position_encoding!r}"
        )

    data_config = DataConfig(
        data_root=data_dir,
        seed=seed,
        max_train_events=_env_optional_int("SUPERNEMO_ROPE_MAX_TRAIN_EVENTS"),
        max_validation_events=_env_optional_int("SUPERNEMO_ROPE_MAX_VALIDATION_EVENTS"),
        max_test_events=_env_optional_int("SUPERNEMO_ROPE_MAX_TEST_EVENTS"),
    )
    tokenization_config = SuperNEMOTrackerTokenizationConfig(
        tokenization=tokenization,
        seed=seed,
        **PUBLISHED_TOKENIZATION[tokenization],
    )
    training_config = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=5.0e-4,
        weight_decay=1.0e-4,
        gradient_clip_norm=1.0,
        early_stopping_patience=patience,
        seed=seed,
        use_amp=use_amp,
        device=device,
        num_workers=num_workers,
    )
    evaluation_config = EvaluationConfig(seed=seed)

    run_dir = output_root / run_id
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "run_summary.json"

    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Data:          {data_dir}")
    print(f"Run id:        {run_id}")
    print(f"Run dir:       {run_dir}")
    print(
        f"Tokenization:  {tokenization} (max_tokens={tokenization_config.max_tokens}, "
        f"feature_dim={tokenization_config.feature_dim})"
    )
    print(f"Position:      {position_encoding} (rope_base={rope_base})")
    print(f"Split capped:  {not data_config.is_published_split}")

    if config_path.is_file() and summary_path.is_file() and not overwrite:
        print(f"Run already complete: {summary_path}. Set SUPERNEMO_ROPE_OVERWRITE=1 to redo.")
        return 0
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"partial output exists for {run_id}: {run_dir}. "
            "Archive it or set SUPERNEMO_ROPE_OVERWRITE=1."
        )

    index_start = time.perf_counter()
    prepared = prepare_dataset(
        data_config=data_config,
        tokenization_config=tokenization_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    index_seconds = time.perf_counter() - index_start
    print(f"Counts:        {prepared.counts}  (indexed in {index_seconds:.1f}s)")

    # Seed again right before construction so initial weights do not depend on
    # how much RNG the indexing consumed.
    set_seed(training_config.seed, training_config.deterministic)
    model = build_supernemo_transformer(
        tokenization_config,
        position_encoding,
        rope_base=rope_base,
    )
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_config = model.config_dict()
    frequency_table = (
        rope_frequency_table(
            head_dim=model_config["d_model"] // model_config["nhead"],
            rope_base=rope_base,
        )
        if position_encoding == "rope"
        else None
    )
    print(f"Parameters:    {parameter_count:,}")

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_path,
        {
            "run_id": run_id,
            "task": "classification: 2nubb (1) vs Bi214 (0)",
            "tokenization": tokenization_config.to_dict(),
            "position_encoding": position_encoding,
            "rope_base": rope_base,
            "rope_frequency_table": frequency_table,
            "model": model_config,
            "parameter_count": int(parameter_count),
            "training_config": training_config.to_dict(),
            "evaluation_config": evaluation_config.to_dict(),
            "data_config": data_config.to_dict(),
            "counts": prepared.counts,
        },
    )

    training_start = time.perf_counter()
    result = train_model(
        model,
        prepared.train_loader,
        prepared.validation_loader,
        training_config,
        "classification",
        run_dir / "training",
        overwrite=overwrite,
    )
    training_seconds = time.perf_counter() - training_start

    # EnergyBench raises for events outside its fixed 0-3 MeV grid, and 0.013% of
    # Bi214 events exceed it; evaluate the events inside the grid and record
    # exactly how many were set aside.
    evaluation_loader = EnergyWindowLoader(
        prepared.test_loader,
        max_energy_mev=CANONICAL_ENERGY_MAX_KEV / KEV_PER_MEV,
    )
    evaluation_start = time.perf_counter()
    metrics = evaluate_classification(
        model,
        evaluation_loader,
        device=training_config.device,
        output_dir=run_dir / "evaluation",
        config=evaluation_config,
        overwrite=overwrite,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    above_grid = dict(sorted(evaluation_loader.dropped.items()))

    epochs_completed = int(result["epochs_completed"])
    summary = {
        "run_id": run_id,
        "task": "classification",
        "tokenization": tokenization,
        "position_encoding": position_encoding,
        "rope_base": rope_base if position_encoding == "rope" else None,
        "seed": seed,
        "max_tokens": tokenization_config.max_tokens,
        "feature_dim": int(model.feature_dim),
        "parameter_count": int(parameter_count),
        "n_train": sum(prepared.counts["train"].values()),
        "n_val": sum(prepared.counts["validation"].values()),
        "n_test": sum(prepared.counts["test"].values()),
        "published_split": data_config.is_published_split,
        "epochs_completed": epochs_completed,
        "best_epoch": int(result["best_epoch"]),
        "stopped_early": bool(result["stopped_early"]),
        "best_val_auc": float(result["best_metric"]),
        "test_matched_auc": metrics.get("matched_auc"),
        "test_matched_auc_status": metrics.get("matched_auc_status"),
        "test_inclusive_auc": metrics.get("auc"),
        "test_common_support_auc": metrics.get("common_support_auc"),
        "test_shortcut_gap": metrics.get("shortcut_gap"),
        "energy_independence_score": metrics.get("energy_independence_score"),
        "worst_energy_independence_score": metrics.get("worst_energy_independence_score"),
        "test_events": metrics.get("n_events"),
        "test_events_above_energy_grid": sum(above_grid.values()),
        "test_events_above_energy_grid_by_category": above_grid,
        "energy_grid_max_mev": CANONICAL_ENERGY_MAX_KEV / KEV_PER_MEV,
        "index_seconds": round(index_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "minutes_per_epoch": round(training_seconds / 60.0 / max(1, epochs_completed), 4),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_json(summary_path, summary)

    print("\n=== result ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
