"""One cell of the CUORE Δt regression benchmark matrix.

Trains a ``CuoreWaveformTransformer`` (one mjd_detector tokenization x one of
three positional encodings) on the CUORE first-two-pulse Δt regression task via
the shared EnergyBench ``train_model`` / ``evaluate_regression``, and writes a
flat ``run_summary.json`` into its own run directory.

It deliberately does NOT write a shared results CSV: the 9 array tasks would
race on a read-modify-rewrite and silently drop rows. Use
``collate_results.py`` after the array drains.

Example (CPU smoke)::

    CUORE_MAX_EVENTS=128 CUORE_EPOCHS=1 CUORE_DEVICE=cpu CUORE_BATCH_SIZE=8 \\
    CUORE_TOKENIZATION=segment_summary CUORE_POSITION_ENCODING=rope \\
    CUORE_OUTPUT_ROOT=$PWD/cuore_detector/results/smoke \\
    python cuore_detector/scripts/run_cuore_regression.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path


def _find_project_root() -> Path:
    for candidate in (
        Path(__file__).resolve(),
        *Path(__file__).resolve().parents,
    ):
        if (
            (candidate / "evalutaions_workflow" / "simple_energybench").is_dir()
            and (candidate / "mjd_detector" / "mjd_transformer").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "could not locate the project root (needs evalutaions_workflow/ and "
        "mjd_detector/)"
    )


PROJECT_ROOT = _find_project_root()
for _source_root in (
    PROJECT_ROOT / "evalutaions_workflow",
    PROJECT_ROOT / "mjd_detector",
    PROJECT_ROOT / "next_detector",
    PROJECT_ROOT / "cuore_detector",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

import torch  # noqa: E402

from mjd_transformer.tokenization import TokenizationConfig  # noqa: E402
from simple_energybench import (  # noqa: E402
    EvaluationConfig,
    TrainingConfig,
    set_seed,
    train_model,
)

from cuore_transformer import (  # noqa: E402
    DEFAULT_ROPE_BASE,
    POSITION_ENCODINGS,
    CuoreWaveformTransformer,
)
from cuorebench import (  # noqa: E402
    CuoreDataConfig,
    evaluate_cuore_regression,
    prepare_cuore_regression_data,
)


# ---------------------------------------------------------------------------
# environment parsing
# ---------------------------------------------------------------------------
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


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


# Fixed architecture -- identical to the MJD official matrix.
D_MODEL = 64
N_HEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
NUM_FREQUENCIES = 6

TOKENIZATIONS = ("raw_patches", "segment_summary", "pulse_entities")


def make_tokenization_config(name: str) -> TokenizationConfig:
    """Official MJD per-tokenization settings (500 tokens for all three)."""

    if name == "raw_patches":
        return TokenizationConfig(tokenization="raw_patches", patch_size=20)
    if name == "segment_summary":
        return TokenizationConfig(tokenization="segment_summary", token_count=500)
    if name == "pulse_entities":
        return TokenizationConfig(
            tokenization="pulse_entities",
            token_count=500,
            uniform_entity_fraction=0.5,
            entity_context_size=9,
        )
    raise ValueError(
        f"CUORE_TOKENIZATION must be one of {TOKENIZATIONS}, got {name!r}"
    )


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    data_dir = _env_str("CUORE_DATA_DIR", "/vast/pvenkata/vis224/CUORE")
    output_root = (
        Path(
            _env_str(
                "CUORE_OUTPUT_ROOT",
                str(PROJECT_ROOT / "cuore_detector" / "results" / "cuore_matrix_v1"),
            )
        )
        .expanduser()
        .resolve()
    )
    tokenization = _env_str("CUORE_TOKENIZATION", "segment_summary")
    position_encoding = _env_str("CUORE_POSITION_ENCODING", "coordinate_mlp")
    run_suffix = _env_str("CUORE_RUN_SUFFIX", "")
    run_id = _env_str(
        "CUORE_RUN_ID",
        f"regression__{tokenization}__{position_encoding}{run_suffix}",
    )
    rope_base = _env_float("CUORE_ROPE_BASE", DEFAULT_ROPE_BASE)
    rope_time_axis_only = _env_flag("CUORE_ROPE_TIME_AXIS_ONLY", False)
    dt_scale = _env_float("CUORE_DT_SCALE", 3000.0)
    tailelevated_weight = _env_float("CUORE_TAILELEVATED_WEIGHT", 1.0)
    zero_dt_weight = _env_float("CUORE_ZERO_DT_WEIGHT", 1.0)
    weight_normalization = _env_str("CUORE_WEIGHT_NORMALIZATION", "none")
    val_fraction = _env_float("CUORE_VAL_FRACTION", 0.15)
    seed = _env_int("CUORE_SEED", 42)
    epochs = _env_int("CUORE_EPOCHS", 50)
    batch_size = _env_int("CUORE_BATCH_SIZE", 64)
    max_events = _env_optional_int("CUORE_MAX_EVENTS")
    device = _env_str("CUORE_DEVICE", "auto")
    num_workers = _env_int("CUORE_NUM_WORKERS", 0)
    overwrite = _env_flag("CUORE_OVERWRITE", False)

    if position_encoding not in POSITION_ENCODINGS:
        raise ValueError(
            f"CUORE_POSITION_ENCODING must be one of {POSITION_ENCODINGS}, "
            f"got {position_encoding!r}"
        )

    run_dir = output_root / run_id
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "run_summary.json"

    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUORE data:   {data_dir}")
    print(f"Run id:       {run_id}")
    print(f"Run dir:      {run_dir}")
    print(f"Tokenization: {tokenization}")
    print(
        f"Encoding:     {position_encoding}"
        + (
            f" (rope_base={rope_base}, time_axis_only={rope_time_axis_only})"
            if position_encoding == "rope"
            else ""
        )
    )

    if config_path.is_file() and summary_path.is_file() and not overwrite:
        print(f"Run already complete: {summary_path}. Set CUORE_OVERWRITE=1 to redo.")
        return 0

    training_config = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )
    evaluation_config = EvaluationConfig()
    tokenization_config = make_tokenization_config(tokenization)

    set_seed(training_config.seed, training_config.deterministic)

    data_config = CuoreDataConfig(
        data_dir=data_dir,
        val_fraction=val_fraction,
        split_seed=seed,
        dt_scale=dt_scale,
        max_events=max_events,
        tailelevated_weight=tailelevated_weight,
        zero_dt_weight=zero_dt_weight,
        weight_normalization=weight_normalization,
    )

    train_loader, val_loader, test_loader, meta = prepare_cuore_regression_data(
        data_config, training_config
    )
    print("Scaled target range:", meta["scaled_target_range"])
    print("Counts: train={n_train} val={n_val} test={n_test}".format(**meta))
    print("Δt(ms) train stats:", meta["train_dt_ms_stats"])
    print("has_second_pulse fraction:", meta["has_second_pulse_fraction"])
    print("train sample_weight:", meta["splits"]["train"]["sample_weight_stats"])

    scaled_lo, scaled_hi = meta["scaled_target_range"]
    if not (0.0 <= scaled_lo and scaled_hi <= 3.0):
        raise ValueError(
            f"scaled Δt target outside [0, 3]: [{scaled_lo}, {scaled_hi}]"
        )

    # Seed again immediately before construction so initial weights are
    # reproducible independently of how much RNG the loader consumed.
    set_seed(training_config.seed, training_config.deterministic)
    model = CuoreWaveformTransformer(
        position_encoding=position_encoding,
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
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    model_config = model.config_dict()
    num_tokens = (
        meta["waveform_length"] // tokenization_config.patch_size
        if tokenization == "raw_patches"
        else tokenization_config.token_count
    )
    print(
        f"Tokens: {num_tokens} x feature_dim {model.feature_dim}; "
        f"parameters: {parameter_count:,}"
    )

    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"partial output exists for {run_id}: {run_dir}. "
            "Archive it or set CUORE_OVERWRITE=1."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        config_path,
        {
            "run_id": run_id,
            "task": "regression",
            "target": "pulseFinder/dtBetweenPeaks[:, 0] (first-two-pulse Δt, ms)",
            "note": (
                "EnergyBench artifacts in evaluation/ are named for the energy "
                "task they were written for: the target rides under the batch "
                "key 'energy', plots are labelled 'Energy [MeV]', "
                "predictions.npz stores energy_true/energy_pred, and "
                "metrics.json RMSE/MAE/bias are in SCALED units (Δt/dt_scale). "
                "Multiply by dt_scale for ms, or just read "
                "evaluation/cuore_regression_ms.json, which is labelled "
                "honestly."
            ),
            "model": _json_safe(model_config),
            "parameter_count": int(parameter_count),
            "num_tokens": int(num_tokens),
            "training_config": training_config.to_dict(),
            "evaluation_config": evaluation_config.to_dict(),
            "data_meta": _json_safe(
                {key: meta[key] for key in meta if key != "splits"}
            ),
            "split_meta": _json_safe(meta["splits"]),
        },
    )

    training_start = time.perf_counter()
    history = train_model(
        model,
        train_loader,
        val_loader,
        config=training_config,
        task="regression",
        output_dir=run_dir / "training",
        overwrite=overwrite,
    )
    training_seconds = time.perf_counter() - training_start

    evaluation_start = time.perf_counter()
    evaluation = evaluate_cuore_regression(
        model,
        test_loader,
        output_dir=run_dir / "evaluation",
        data_config=data_config,
        evaluation_config=evaluation_config,
        device=training_config.device,
        overwrite=overwrite,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    scaled_metrics = evaluation["scaled"]
    ms = evaluation["ms"]
    epochs_completed = int(history["epochs_completed"])

    summary = {
        "run_id": run_id,
        "task": "regression",
        "tokenization": tokenization,
        "position_encoding": position_encoding,
        # Recorded explicitly: CUORE_SEED drives both weight init and the
        # train/val partition, so multi-seed studies must be able to group on
        # it. The test split is unaffected (all of cuoreTest.h5 is always used).
        "seed": seed,
        "split_seed": seed,
        "rope_base": rope_base if position_encoding == "rope" else None,
        "rope_time_axis_only": (
            rope_time_axis_only if position_encoding == "rope" else None
        ),
        "num_tokens": int(num_tokens),
        "feature_dim": int(model.feature_dim),
        "patch_size": int(tokenization_config.patch_size),
        "token_count": int(tokenization_config.token_count),
        "d_model": D_MODEL,
        "nhead": N_HEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "num_frequencies": NUM_FREQUENCIES,
        "parameter_count": int(parameter_count),
        "dt_scale": dt_scale,
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_test": meta["n_test"],
        "train_weight_sum": meta["train_weight_sum"],
        "epochs_completed": epochs_completed,
        "best_epoch": history["best_epoch"],
        "best_val_rmse_scaled": history["best_metric"],
        "stopped_early": bool(history["stopped_early"]),
        "training_seconds": round(training_seconds, 3),
        "minutes_per_epoch": round(
            training_seconds / 60.0 / max(1, epochs_completed), 4
        ),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "test_rmse_scaled": scaled_metrics.get("rmse"),
        "test_rmse_ms": ms["overall_ms"]["rmse_ms"],
        "test_mae_ms": ms["overall_ms"]["mae_ms"],
        "test_bias_ms": ms["overall_ms"]["bias_ms"],
        "test_r2": ms["overall_ms"]["r2"],
        "test_median_abs_error_ms": ms["overall_ms"]["median_abs_error_ms"],
        "test_frac_within_100ms": ms["overall_ms"]["frac_within_100ms"],
        "test_frac_within_250ms": ms["overall_ms"]["frac_within_250ms"],
        "test_rmse_ms_pileup_only": ms["pileup_ms"]["rmse_ms"],
        "test_mae_ms_pileup_only": ms["pileup_ms"]["mae_ms"],
        "test_bias_ms_pileup_only": ms["pileup_ms"]["bias_ms"],
        "test_r2_pileup": ms["pileup_ms"]["r2"],
        "test_rmse_ms_clean": ms["clean_ms"]["rmse_ms"],
        "test_mae_ms_clean": ms["clean_ms"]["mae_ms"],
        "test_bias_ms_clean": ms["clean_ms"]["bias_ms"],
        # Suffixed so nobody accidentally ranks on them -- both are meaningless
        # for a target with a ~51% point mass at zero.
        "test_ers_DEGENERATE": scaled_metrics.get("ers"),
        "test_frac_res68_DEGENERATE": scaled_metrics.get(
            "fractional_resolution_68"
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        ),
    }
    _write_json(summary_path, summary)

    print("\n=== result ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
