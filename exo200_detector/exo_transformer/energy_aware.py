"""Frozen EnergyBench evaluation adapter for EXO-200 Transformer outputs.

This module never runs model inference.  It aligns saved test logits with the
official EXOBench test split, attaches ``Rotated_energy`` as evaluation-only
metadata, and delegates all scientific metrics to Wing's frozen EnergyBench.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exobench.config import DataConfig
from exobench.data import EXOWaveformDataset, _split_plan, discover_files


ENERGY_FIELD = "Rotated_energy"
EXPECTED_TEST_EVENTS = 140_383
EXPECTED_TEST_RUNS = (8967, 8970, 8971, 9010)
TOKENIZATIONS = ("raw_patches", "segment_summary", "pulse_entities")
POSITION_ENCODINGS = ("coordinate_mlp", "fourier_coordinates")
DEFAULT_FROZEN_ENERGYBENCH_ROOT = (
    Path(__file__).resolve().parents[1] / "frozen_energybench"
)
RUN_IDS = tuple(
    f"classification__{tokenization}__{position_encoding}"
    for tokenization in TOKENIZATIONS
    for position_encoding in POSITION_ENCODINGS
)


@dataclass(frozen=True)
class EnergyBenchRuntime:
    """Imported interfaces and provenance for the frozen EnergyBench copy."""

    prediction_bundle: type
    run_evaluation: Callable[..., dict[str, Any]]
    load_manifest: Callable[[], dict[str, Any]]
    source_root: Path
    source_sha256: str
    source_files_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Hash Python sources with their relative paths in stable order."""

    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no Python source files found under {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_energybench_runtime(
    source_root: str | Path | None = None,
) -> EnergyBenchRuntime:
    """Import the vendored EnergyBench and capture a source-tree fingerprint."""

    root = Path(
        DEFAULT_FROZEN_ENERGYBENCH_ROOT if source_root is None else source_root
    ).expanduser().resolve()
    package_root = root / "energybench"
    if not package_root.is_dir():
        raise FileNotFoundError(
            f"frozen EnergyBench package not found at {package_root}"
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    config_module = importlib.import_module("energybench.config")
    data_module = importlib.import_module("energybench.data")
    evaluation_module = importlib.import_module("energybench.evaluation")
    imported_package = Path(
        importlib.import_module("energybench").__file__
    ).resolve().parent
    if imported_package != package_root:
        raise ImportError(
            "a different energybench package was already imported: "
            f"expected {package_root}, found {imported_package}"
        )
    return EnergyBenchRuntime(
        prediction_bundle=data_module.PredictionBundle,
        run_evaluation=evaluation_module.run_evaluation,
        load_manifest=config_module.load_manifest,
        source_root=root,
        source_sha256=_tree_sha256(package_root),
        source_files_sha256={
            path.relative_to(package_root).as_posix(): _sha256(path)
            for path in sorted(package_root.rglob("*.py"))
            if path.is_file()
        },
    )


def build_test_metadata(
    config: DataConfig,
    *,
    expected_runs: Iterable[int] = EXPECTED_TEST_RUNS,
    expected_events: int | None = EXPECTED_TEST_EVENTS,
) -> dict[str, np.ndarray]:
    """Reconstruct canonical EXOBench test identities and evaluation energy."""

    paths = discover_files(config.data_root)
    source = EXOWaveformDataset(paths, config=config)
    try:
        plan = _split_plan(source, config)
        required_runs = tuple(sorted(int(value) for value in expected_runs))
        actual_runs = tuple(sorted(int(value) for value in plan.runs["test"]))
        if actual_runs != required_runs:
            raise ValueError(
                f"EXO test runs differ: expected {required_runs}, found {actual_runs}"
            )
        indices = plan.indices["test"]
        if expected_events is not None and indices.size != int(expected_events):
            raise ValueError(
                f"EXO test count differs: expected {expected_events}, "
                f"found {indices.size}"
            )
        labels = source.labels_for_indices(indices).astype(np.int64, copy=False)
        event_id = np.empty(indices.size, dtype="U96")
        category = np.where(labels == 0, "signal", "background")
        energy = np.empty(indices.size, dtype=np.float64)
        group_id = np.empty(indices.size, dtype="U32")
        run_number = np.empty(indices.size, dtype=np.int64)
        event_number = np.empty(indices.size, dtype=np.int64)

        file_indices = np.searchsorted(source._offsets, indices, side="right") - 1
        for file_index in np.unique(file_indices):
            positions = np.flatnonzero(file_indices == file_index)
            info = source._files[int(file_index)]
            rows = indices[positions] - info.offset
            with h5py.File(info.path, "r") as handle:
                if ENERGY_FIELD not in handle:
                    raise KeyError(f"{info.path}: missing {ENERGY_FIELD!r}")
                values = np.asarray(handle[ENERGY_FIELD][rows], dtype=np.float64)
            energy[positions] = values
            group_id[positions] = str(info.run_number)
            run_number[positions] = int(info.run_number)
            selected_events = info.event_numbers[rows].astype(np.int64, copy=False)
            event_number[positions] = selected_events
            for position, selected_event in zip(
                positions.tolist(), selected_events.tolist()
            ):
                event_id[position] = (
                    f"EXO200::{info.path.name}::{int(selected_event)}"
                )
    finally:
        source.close()

    if not np.isin(labels, (0, 1)).all():
        raise ValueError("canonical EXO test labels must be binary")
    if not np.isfinite(energy).all() or np.any(energy <= 0.0):
        raise ValueError(f"{ENERGY_FIELD} must be finite and positive")
    if np.unique(event_id).size != event_id.size:
        raise ValueError("canonical EXO test event IDs are not unique")
    if tuple(sorted(np.unique(run_number).tolist())) != required_runs:
        raise ValueError("canonical metadata contains unexpected test runs")

    return {
        "event_id": event_id,
        "label": labels,
        "category": category,
        "energy_condition": energy,
        "sample_weight": np.ones(indices.size, dtype=np.float64),
        "group_id": group_id,
        "split": np.full(indices.size, "test", dtype="U4"),
        "run_number": run_number,
        "event_number": event_number,
    }


def build_evaluation_config(
    model_id: str,
    load_manifest: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Return Wing's frozen EXO-200 classification configuration."""

    config = load_manifest()
    config.update(
        {
            "task_id": "exo200-signal-vs-background",
            "model_id": model_id,
            "dataset": {
                "experiment": "EXO-200",
                "dataset_id": "zeronu-benchmark-exo200",
                "dataset_version": "local-hdf5-inventory-2026-08-21",
                "split": "test",
                "selection_split": "validation",
                "energy_condition_kind": "rotated_reconstructed_energy",
                "energy_target_kind": "not_applicable",
                "energy_unit": "keV",
            },
            "columns": {
                "event_id": "event_id",
                "label": "label",
                "score": "score",
                "energy_condition": "energy_condition",
                "energy_true": None,
                "energy_pred": None,
                "category": "category",
                "sample_weight": "sample_weight",
                "group_id": "group_id",
                "split": "split",
            },
            "classification": {
                "enabled": True,
                "positive_label": "0",
                "signal_categories": ["signal"],
                "background_categories": ["background"],
                "pair_mode": "pooled",
                "score_direction": "lower",
                "score_space": "logit",
                "energy_bins": 6,
                "matching_target": "overlap",
                "min_per_class": 20,
                "min_valid_bins": 2,
                "min_coverage": 0.5,
                "support_trim_quantile": 0.005,
                "energy_roi": None,
                "target_tpr": 0.90,
                "bootstrap": 0,
                "confidence": 0.95,
            },
            "regression": {
                "enabled": False,
                "histogram_bins": 50,
                "performance_bins": 10,
                "energy_floor": None,
                "histogram_edges": None,
                "bootstrap": 0,
                "confidence": 0.95,
            },
            "dependence": {
                "enabled": True,
                "energy_bins": 8,
                "score_bins": 20,
                "min_per_bin": 20,
                "distance_correlation_max_samples": 1200,
            },
            "runtime": {"seed": 42, "make_plots": True},
        }
    )
    return config


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_summary(report: dict[str, Any]) -> dict[str, Any]:
    classification = report["classification"]
    aggregates = classification["aggregates"]
    pairs = classification.get("pairs", [])
    if len(pairs) != 1:
        raise ValueError(f"expected one pooled classification pair, found {len(pairs)}")
    pair = pairs[0]
    dependence = report["energy_dependence"]
    coverage = pair.get("coverage")
    if coverage is None and isinstance(pair.get("matching"), dict):
        coverage = pair["matching"].get("coverage")
    if isinstance(coverage, dict):
        scalar_coverage = next(
            (
                coverage[name]
                for name in ("overall", "minimum", "matched_fraction", "fraction")
                if name in coverage and np.isscalar(coverage[name])
            ),
            None,
        )
        if scalar_coverage is None:
            matched_weight_fractions = [
                float(value)
                for name, value in coverage.items()
                if name.endswith("matched_weight_fraction")
                and value is not None
                and np.isscalar(value)
            ]
            matched_count_fractions = [
                float(value)
                for name, value in coverage.items()
                if name.endswith("matched_count_fraction")
                and value is not None
                and np.isscalar(value)
            ]
            candidates = matched_weight_fractions or matched_count_fractions
            scalar_coverage = min(candidates) if candidates else None
        coverage = scalar_coverage
    matched_status = pair.get("matched_auc_status", pair.get("status"))
    if matched_status is None:
        # Mirror frozen_energybench.energybench.reporting._matched_auc_status
        # without importing a private reporting helper.
        if aggregates.get("matched_auc_macro") is not None:
            matched_status = "ok"
        elif report.get("classification", {}).get("status") != "ok":
            matched_status = report.get("classification", {}).get(
                "status", "not_applicable"
            )
        else:
            matched_status = "not_evaluable_incomplete_pair_set"
    return {
        "inclusive_auc": aggregates.get(
            "inclusive_auc_macro", pair.get("inclusive_auc")
        ),
        "common_support_auc": pair.get(
            "common_support_auc", pair.get("inclusive_common_support_auc")
        ),
        "energy_matched_auc": aggregates.get(
            "matched_auc_macro", pair.get("matched_auc")
        ),
        "shortcut_gap": pair.get("shortcut_gap"),
        "matched_auc_status": matched_status,
        "matched_coverage": coverage,
        "energy_independence_score": dependence.get(
            "overall_energy_independence_score"
        ),
        "worst_energy_independence_score": dependence.get(
            "worst_group_energy_independence_score"
        ),
    }


def _validate_prediction_arrays(
    prediction_path: Path,
    canonical_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(prediction_path) as loaded:
        if not {"target", "prediction"}.issubset(loaded.files):
            raise KeyError(f"{prediction_path} lacks target or prediction")
        target = np.asarray(loaded["target"]).reshape(-1).astype(np.int64)
        score = np.asarray(loaded["prediction"]).reshape(-1).astype(np.float64)
    if target.shape != canonical_labels.shape or score.shape != target.shape:
        raise ValueError(
            f"{prediction_path}: expected {canonical_labels.size} aligned rows; "
            f"found target={target.shape}, prediction={score.shape}"
        )
    if not np.isin(target, (0, 1)).all() or not np.isfinite(score).all():
        raise ValueError(f"{prediction_path}: targets must be binary and scores finite")
    if not np.array_equal(target, canonical_labels):
        raise ValueError(
            f"prediction/test metadata alignment failed for {prediction_path.parent.name}"
        )
    return target, score


def _guard_existing(
    destination: Path,
    provenance: dict[str, Any],
    provenance_path: Path,
) -> None:
    # Version 1 briefly wrote this adapter-owned file inside EnergyBench's
    # protected output directory. Validate and remove only that known file so
    # EnergyBench can enforce its own artifact allowlist.
    legacy_path = destination / "input_provenance.json"
    if legacy_path.is_file():
        legacy = _read_json(legacy_path)
        if legacy != provenance:
            raise RuntimeError(
                f"stale EnergyBench provenance in {legacy_path}; archive or "
                "remove the evaluation directory before continuing"
            )
        legacy_path.unlink()
    if not destination.exists():
        return
    existing_files = [path for path in destination.iterdir() if path.is_file()]
    if not provenance_path.is_file():
        if existing_files:
            raise RuntimeError(
                f"refusing unverified existing evaluation artifacts in {destination}"
            )
        return
    existing = _read_json(provenance_path)
    if existing != provenance:
        raise RuntimeError(
            f"stale EnergyBench artifacts in {destination}; archive or remove that "
            "directory before evaluating changed inputs"
        )


def evaluate_transformer_runs(
    *,
    data_config: DataConfig,
    output_root: str | Path,
    energybench_source: str | Path | None = None,
    expected_runs: Iterable[int] = EXPECTED_TEST_RUNS,
    expected_events: int | None = EXPECTED_TEST_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate every completed Transformer run and write combined artifacts."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = load_energybench_runtime(energybench_source)
    metadata = build_test_metadata(
        data_config,
        expected_runs=expected_runs,
        expected_events=expected_events,
    )
    exobench_data_path = Path(sys.modules[_split_plan.__module__].__file__).resolve()
    exobench_data_sha256 = _sha256(exobench_data_path)

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, str]] = []
    reports: dict[str, Any] = {}
    reference_evaluation_fingerprint: str | None = None
    reference_protocol_fingerprint: str | None = None

    for run_id in RUN_IDS:
        run_dir = root / run_id
        required = {
            "predictions": run_dir / "predictions.npz",
            "checkpoint": run_dir / "best.pt",
            "run_config": run_dir / "run_config.json",
            "run_summary": run_dir / "run_summary.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            status_rows.append(
                {"run_id": run_id, "status": "incomplete", "detail": ", ".join(missing)}
            )
            continue

        target, score = _validate_prediction_arrays(
            required["predictions"], metadata["label"]
        )
        columns = {
            key: value
            for key, value in metadata.items()
            if key not in {"run_number", "event_number"}
        }
        columns["score"] = score
        provenance = {
            "adapter": "EXO-200 Transformer predictions + Rotated_energy",
            "run_id": run_id,
            "checkpoint": str(required["checkpoint"]),
            "checkpoint_sha256": _sha256(required["checkpoint"]),
            "prediction_source": str(required["predictions"]),
            "prediction_source_sha256": _sha256(required["predictions"]),
            "run_config_sha256": _sha256(required["run_config"]),
            "energybench_source": str(runtime.source_root / "energybench"),
            "energybench_source_sha256": runtime.source_sha256,
            "energybench_source_files_sha256": runtime.source_files_sha256,
            "exobench_split_source": str(exobench_data_path),
            "exobench_split_source_sha256": exobench_data_sha256,
            "energy_field": ENERGY_FIELD,
            "energy_unit": "keV",
            "split": "test",
            "test_events": int(target.size),
        }
        destination = run_dir / "energybench_classification"
        provenance_path = run_dir / "energybench_input_provenance.json"
        _guard_existing(destination, provenance, provenance_path)
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )

        canonical_path = run_dir / "energybench_predictions.npz"
        np.savez_compressed(
            canonical_path,
            **columns,
            __metadata__=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
        bundle = runtime.prediction_bundle(
            columns=columns, metadata=provenance, source=canonical_path
        )
        report = runtime.run_evaluation(
            bundle,
            build_evaluation_config(run_id, runtime.load_manifest),
            destination,
            strict=True,
            allow_existing=True,
        )
        evaluation_fingerprint = report["evaluation_fingerprint"]
        protocol_fingerprint = report["protocol_fingerprint"]
        if reference_evaluation_fingerprint is None:
            reference_evaluation_fingerprint = evaluation_fingerprint
            reference_protocol_fingerprint = protocol_fingerprint
        elif evaluation_fingerprint != reference_evaluation_fingerprint:
            raise RuntimeError("EnergyBench evaluation fingerprints differ across models")
        elif protocol_fingerprint != reference_protocol_fingerprint:
            raise RuntimeError("EnergyBench protocol fingerprints differ across models")

        run_summary = _read_json(required["run_summary"])
        representation = _read_json(required["run_config"]).get("representation", {})
        metrics = _pair_summary(report)
        tokenization_value = representation.get(
            "tokenization", run_summary.get("tokenization")
        )
        if isinstance(tokenization_value, dict):
            tokenization_value = tokenization_value.get("tokenization")
        row = {
            "run_id": run_id,
            "tokenization": tokenization_value,
            "position_encoding": representation.get(
                "position_encoding", run_summary.get("position_encoding")
            ),
            "best_validation_auc": run_summary.get("best_validation_auc"),
            **metrics,
            "epochs_completed": run_summary.get("epochs_completed"),
            "minutes_per_epoch": run_summary.get("minutes_per_epoch"),
            "test_events": int(target.size),
            "evaluation_fingerprint": evaluation_fingerprint,
            "protocol_fingerprint": protocol_fingerprint,
            "prediction_sha256": provenance["prediction_source_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
        }
        rows.append(row)
        status_rows.append({"run_id": run_id, "status": "complete", "detail": ""})
        reports[run_id] = report

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            "energy_matched_auc", ascending=False, na_position="last"
        ).reset_index(drop=True)
    statuses = pd.DataFrame(status_rows)
    results.to_csv(root / "energybench_transformer_results.csv", index=False)
    combined = {
        "energy_condition": ENERGY_FIELD,
        "energy_unit": "keV",
        "expected_run_ids": list(RUN_IDS),
        "completed_run_ids": results.get("run_id", pd.Series(dtype=str)).tolist(),
        "evaluation_fingerprint": reference_evaluation_fingerprint,
        "protocol_fingerprint": reference_protocol_fingerprint,
        "energybench_source": str(runtime.source_root / "energybench"),
        "energybench_source_sha256": runtime.source_sha256,
        "energybench_source_files_sha256": runtime.source_files_sha256,
        "exobench_split_source": str(exobench_data_path),
        "exobench_split_source_sha256": exobench_data_sha256,
        "test_runs": sorted(np.unique(metadata["run_number"]).astype(int).tolist()),
        "test_events": int(metadata["label"].size),
        "status": statuses.to_dict(orient="records"),
        "results": results.to_dict(orient="records"),
        "reports": reports,
    }
    (root / "energybench_transformer_summary.json").write_text(
        json.dumps(_json_safe(combined), indent=2, allow_nan=False), encoding="utf-8"
    )
    return results, statuses, reports


def save_comparison_plots(
    results: pd.DataFrame,
    reports: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Write compact six-model comparison plots from completed reports."""

    if results.empty:
        return []
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    labels = [
        f"{row.tokenization}\n{row.position_encoding}"
        for row in results.itertuples()
    ]
    positions = np.arange(len(results))

    inclusive = pd.to_numeric(results["inclusive_auc"], errors="coerce")
    matched = pd.to_numeric(results["energy_matched_auc"], errors="coerce")
    if bool((inclusive.notna() & matched.notna()).any()):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(
            positions - 0.18,
            inclusive,
            width=0.36,
            label="Inclusive AUC",
            color="#B9CBE0",
        )
        ax.bar(
            positions + 0.18,
            matched,
            width=0.36,
            label="Energy-matched AUC",
            color="#2563A6",
        )
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("EXO-200 Transformer classification")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = destination / "transformer_auc_comparison.png"
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)
        written.append(path)

    independence = pd.to_numeric(
        results["energy_independence_score"], errors="coerce"
    )
    if independence.notna().any():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(positions, independence, color="#6B7D2A")
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_ylabel("Energy-independence score")
        ax.set_title("EXO-200 class-conditional score/energy independence")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = destination / "transformer_energy_independence.png"
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    curves = 0
    for run_id in results["run_id"]:
        pair = reports[run_id]["classification"]["pairs"][0]
        curve = pair.get("matching", {}).get("matched")
        if not curve:
            continue
        label = run_id.removeprefix("classification__").replace("__", " + ")
        ax.plot(curve["fpr"], curve["tpr"], linewidth=1.8, label=label)
        curves += 1
    if curves:
        ax.plot([0, 1], [0, 1], "--", color="#6B7280", linewidth=1)
        ax.set_xlabel("False-positive rate (background acceptance)")
        ax.set_ylabel("True-positive rate (signal efficiency)")
        ax.set_title("EXO-200 energy-matched ROC")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = destination / "transformer_energy_matched_roc.png"
        fig.savefig(path, dpi=220, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


__all__ = [
    "ENERGY_FIELD",
    "EXPECTED_TEST_EVENTS",
    "EXPECTED_TEST_RUNS",
    "POSITION_ENCODINGS",
    "RUN_IDS",
    "TOKENIZATIONS",
    "EnergyBenchRuntime",
    "build_evaluation_config",
    "build_test_metadata",
    "evaluate_transformer_runs",
    "load_energybench_runtime",
    "save_comparison_plots",
]
