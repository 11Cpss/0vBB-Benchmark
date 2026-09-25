"""One cell of the MJD RoPE classification benchmark (signal vs. background).

Trains an ``MJDRopeTransformer(task="classification", position_encoding="rope")``
(one of the three ``mjd_transformer`` tokenizations) on a seeded, cached
50k/5k/50k subsample of the official MJD clean/non-clean split, via the
shared, unchanged ``mjdbench.training.train_model`` / ``evaluate_model``, and
writes ``run_config.json`` / ``run_summary.json`` into its own run directory
alongside the ``best.pt`` / ``history.json`` / ``metrics.json`` /
``predictions.npz`` the shared trainer already produces -- the same output
contract documented in ``mjd_detector/README.md``.

The first invocation for a given (n_train, n_val, n_test, seed) builds a
cache under ``MJD_ROPE_CACHE_DIR``; every later invocation (a different
tokenization, a rope_base sweep, a re-run) reuses it, so only one full pass
over the slow, unchunked HDF5 shards is ever paid.

Example (CPU smoke, tiny slice of real data)::

    MJD_ROPE_TOKENIZATION=raw_patches MJD_ROPE_N_TRAIN=200 MJD_ROPE_N_VAL=50 \\
    MJD_ROPE_N_TEST=200 MJD_ROPE_EPOCHS=1 MJD_ROPE_DEVICE=cpu \\
    python mjd_rope_classifier/scripts/run_mjd_rope_classification.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path


def _find_project_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (
            (candidate / "mjd_detector" / "mjdbench").is_dir()
            and (candidate / "mjd_rope_classifier" / "mjdrope_bench").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "could not locate the project root (needs mjd_detector/ and "
        "mjd_rope_classifier/)"
    )


PROJECT_ROOT = _find_project_root()
for _source_root in (
    PROJECT_ROOT / "mjd_detector",
    PROJECT_ROOT / "next_detector",
    PROJECT_ROOT / "mjd_rope_classifier",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

import torch  # noqa: E402

from mjdbench.config import DataConfig, TrainingConfig  # noqa: E402
from mjdbench.training import evaluate_model, set_seed, train_model  # noqa: E402

from mjd_rope_transformer import (  # noqa: E402
    DEFAULT_MJD_ROPE_BASE,
    MJDRopeTransformer,
)
from mjd_transformer.tokenization import TokenizationConfig  # noqa: E402
from mjdrope_bench import (  # noqa: E402
    SubsetSizes,
    build_classification_cache,
    load_cached_loaders,
)


TOKENIZATIONS = ("raw_patches", "segment_summary", "pulse_entities")

# Fixed architecture -- identical to the MJD official matrix.
D_MODEL = 64
N_HEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
NUM_FREQUENCIES = 6

# 3800-sample waveforms (not the 10,000 samples mjd_detector/README.md
# assumes) -> 190 tokens for all three tokenizations, keeping sequence
# length (and attention cost) comparable across cells.
WAVEFORM_LENGTH = 3800
PATCH_SIZE = 20
TOKEN_COUNT = WAVEFORM_LENGTH // PATCH_SIZE  # 190


def make_tokenization_config(name: str) -> TokenizationConfig:
    if name == "raw_patches":
        return TokenizationConfig(tokenization="raw_patches", patch_size=PATCH_SIZE)
    if name == "segment_summary":
        return TokenizationConfig(
            tokenization="segment_summary", token_count=TOKEN_COUNT
        )
    if name == "pulse_entities":
        return TokenizationConfig(
            tokenization="pulse_entities",
            token_count=TOKEN_COUNT,
            uniform_entity_fraction=0.5,
            entity_context_size=9,
        )
    raise ValueError(f"MJD_ROPE_TOKENIZATION must be one of {TOKENIZATIONS}, got {name!r}")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


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
    data_dir = _env_str("MJD_ROPE_DATA_DIR", "/vast/pvenkata/vis224/MJD")
    cache_dir = _env_str(
        "MJD_ROPE_CACHE_DIR", str(Path(data_dir).parent / "mjd_rope_cache")
    )
    output_root = (
        Path(
            _env_str(
                "MJD_ROPE_OUTPUT_ROOT",
                str(
                    PROJECT_ROOT
                    / "mjd_rope_classifier"
                    / "results"
                    / "mjd_rope_classification_v1"
                ),
            )
        )
        .expanduser()
        .resolve()
    )
    tokenization = _env_str("MJD_ROPE_TOKENIZATION", "raw_patches")
    rope_base = _env_float("MJD_ROPE_BASE", DEFAULT_MJD_ROPE_BASE)
    rope_time_axis_only = _env_flag("MJD_ROPE_TIME_AXIS_ONLY", False)
    run_suffix = _env_str("MJD_ROPE_RUN_SUFFIX", "")
    run_id = _env_str("MJD_ROPE_RUN_ID", f"classification__{tokenization}__rope{run_suffix}")

    sizes = SubsetSizes(
        n_train=_env_int("MJD_ROPE_N_TRAIN", 50_000),
        n_val=_env_int("MJD_ROPE_N_VAL", 5_000),
        n_test=_env_int("MJD_ROPE_N_TEST", 50_000),
        seed=_env_int("MJD_ROPE_SEED", 42),
    )
    epochs = _env_int("MJD_ROPE_EPOCHS", 50)
    batch_size = _env_int("MJD_ROPE_BATCH_SIZE", 64)
    device = _env_str("MJD_ROPE_DEVICE", "auto")
    num_workers = _env_int("MJD_ROPE_NUM_WORKERS", 0)
    overwrite = _env_flag("MJD_ROPE_OVERWRITE", False)

    if tokenization not in TOKENIZATIONS:
        raise ValueError(f"MJD_ROPE_TOKENIZATION must be one of {TOKENIZATIONS}, got {tokenization!r}")

    run_dir = output_root / run_id
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "run_summary.json"

    print(f"Project root: {PROJECT_ROOT}")
    print(f"MJD data:     {data_dir}")
    print(f"Cache dir:    {cache_dir}")
    print(f"Run id:       {run_id}")
    print(f"Run dir:      {run_dir}")
    print(f"Tokenization: {tokenization} ({TOKEN_COUNT} tokens)")
    print(f"rope_base:    {rope_base} (time_axis_only={rope_time_axis_only})")
    print(f"Sizes:        train={sizes.n_train} val={sizes.n_val} test={sizes.n_test} seed={sizes.seed}")

    if config_path.is_file() and summary_path.is_file() and not overwrite:
        print(f"Run already complete: {summary_path}. Set MJD_ROPE_OVERWRITE=1 to redo.")
        return 0

    training_config = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        device=device,
        seed=sizes.seed,
    )
    tokenization_config = make_tokenization_config(tokenization)
    data_config = DataConfig(data_root=data_dir, seed=sizes.seed)

    set_seed(training_config.seed, training_config.deterministic)

    cache_start = time.perf_counter()
    cache_path = build_classification_cache(
        data_root=data_dir,
        cache_dir=cache_dir,
        sizes=sizes,
        data_config=data_config,
    )
    cache_seconds = time.perf_counter() - cache_start
    print(f"Cache:        {cache_path} ({cache_seconds:.1f}s)")

    loaders = load_cached_loaders(
        cache_path, batch_size=batch_size, num_workers=num_workers
    )
    print("Counts:", loaders["counts"])

    # Seed again immediately before construction so initial weights are
    # reproducible independently of how much RNG the cache build consumed.
    set_seed(training_config.seed, training_config.deterministic)
    model = MJDRopeTransformer(
        task="classification",
        position_encoding="rope",
        tokenization_config=tokenization_config,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_frequencies=NUM_FREQUENCIES,
        rope_base=rope_base,
        rope_time_axis_only=rope_time_axis_only,
    )
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_config = model.config_dict()
    print(f"Tokens: {TOKEN_COUNT} x feature_dim {model.feature_dim}; parameters: {parameter_count:,}")

    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"partial output exists for {run_id}: {run_dir}. Archive it or set MJD_ROPE_OVERWRITE=1."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        config_path,
        {
            "run_id": run_id,
            "task": "classification",
            "classification_target": "clean_vs_non_clean (all four PSD cuts: low_avse, high_avse, dcr, lq)",
            "tokenization": tokenization,
            "token_count": TOKEN_COUNT,
            "position_encoding": "rope",
            "rope_base": rope_base,
            "rope_time_axis_only": rope_time_axis_only,
            "model": model_config,
            "parameter_count": int(parameter_count),
            "training_config": training_config.to_dict(),
            "data_config": data_config.to_dict(),
            "sizes": {"n_train": sizes.n_train, "n_val": sizes.n_val, "n_test": sizes.n_test, "seed": sizes.seed},
            "counts": loaders["counts"],
            "cache_path": str(cache_path),
        },
    )

    training_start = time.perf_counter()
    history = train_model(
        model,
        loaders["train_loader"],
        loaders["validation_loader"],
        task="classification",
        config=training_config,
        output_dir=run_dir,
    )
    training_seconds = time.perf_counter() - training_start
    epochs_completed = len(history)

    def _checkpoint_score(row: dict) -> float:
        # Mirrors mjdbench.training.train_model's own checkpoint rule: rank
        # by validation macro AUC, falling back to -validation_loss when AUC
        # is undefined (e.g. a tiny smoke split with only one class).
        auc = row["validation_metrics"]["macro_auc"]
        return auc if math.isfinite(auc) else -row["validation_loss"]

    best_row = max(history, key=_checkpoint_score)

    evaluation_start = time.perf_counter()
    metrics = evaluate_model(
        model,
        loaders["test_loader"],
        task="classification",
        device=training_config.device,
        output_dir=run_dir,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    summary = {
        "run_id": run_id,
        "task": "classification",
        "tokenization": tokenization,
        "position_encoding": "rope",
        "rope_base": rope_base,
        "rope_time_axis_only": rope_time_axis_only,
        "seed": sizes.seed,
        "num_tokens": TOKEN_COUNT,
        "feature_dim": int(model.feature_dim),
        "d_model": D_MODEL,
        "nhead": N_HEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "parameter_count": int(parameter_count),
        "n_train": loaders["counts"]["train"],
        "n_val": loaders["counts"]["validation"],
        "n_test": loaders["counts"]["test"],
        "epochs_completed": epochs_completed,
        "best_epoch": best_row["epoch"],
        "best_val_macro_auc": best_row["validation_metrics"]["macro_auc"],
        "cache_seconds": round(cache_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "minutes_per_epoch": round(training_seconds / 60.0 / max(1, epochs_completed), 4),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "test_auc": metrics.get("auc"),
        "test_accuracy": metrics.get("accuracy"),
        "test_clean_fraction": metrics.get("clean_fraction"),
        "test_events": metrics.get("events"),
        "test_loss": metrics.get("loss"),
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
