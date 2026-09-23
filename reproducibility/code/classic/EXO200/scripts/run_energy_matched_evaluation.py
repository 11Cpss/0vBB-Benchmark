#!/usr/bin/env python3
"""Run the frozen EnergyBench classification protocol on four EXO-200 models."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMER_ROOT = Path("/home/wenyu/summer")
for path in (PROJECT_ROOT, SUMMER_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from energybench.config import load_manifest
from energybench.data import PredictionBundle
from energybench.evaluation import run_evaluation
from exobench.config import DataConfig
from exobench.data import EXOWaveformDataset, _split_plan


MODEL_NAMES = (
    "cnn_004_multiview_late_fusion",
    "gnn_001_static_gine",
    "seq_001_bigru",
    "ssm_001_pointmamba",
)
ENERGY_FIELD = "Rotated_energy"
DISPLAY_NAMES = {
    "cnn_004_multiview_late_fusion": "CNN-004",
    "gnn_001_static_gine": "Static GINE",
    "seq_001_bigru": "BiGRU",
    "ssm_001_pointmamba": "PointMamba",
}


def _save_combined_plots(reports: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#2563A6", "#D97706", "#6B7D2A", "#B5476B"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )

    ordered = sorted(
        MODEL_NAMES,
        key=lambda name: reports[name]["classification"]["aggregates"]["matched_auc_macro"],
    )
    inclusive = [
        reports[name]["classification"]["aggregates"]["inclusive_auc_macro"]
        for name in ordered
    ]
    matched = [
        reports[name]["classification"]["aggregates"]["matched_auc_macro"]
        for name in ordered
    ]
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.12, top=0.82)
    ax.barh(y - 0.17, inclusive, height=0.30, color="#B9CBE0", edgecolor="#385C7A", label="Inclusive")
    ax.barh(y + 0.17, matched, height=0.30, color="#2563A6", edgecolor="#163A5F", label="Energy matched")
    ax.set_yticks(y, [DISPLAY_NAMES[name] for name in ordered])
    ax.set_xlim(0.82, 0.97)
    ax.set_xlabel("ROC-AUC")
    fig.suptitle("EXO-200 classification: inclusive vs energy-matched AUC", x=0.22, y=0.96, ha="left", weight="bold", fontsize=16)
    fig.text(0.22, 0.895, "Matched on Rotated_energy · held-out test · 140,383 events", color="#4B5563")
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for row, (raw, adjusted) in enumerate(zip(inclusive, matched)):
        ax.text(raw + 0.0012, row - 0.17, f"{raw:.4f}", va="center", fontsize=9)
        ax.text(adjusted + 0.0012, row + 0.17, f"{adjusted:.4f}", va="center", fontsize=9, weight="bold")
    fig.savefig(output_dir / "four_model_auc_comparison.png", dpi=220, facecolor="white")
    fig.savefig(output_dir / "four_model_auc_comparison.svg", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.11, top=0.83)
    for color, name in zip(palette, MODEL_NAMES):
        pair = reports[name]["classification"]["pairs"][0]
        matched_curve = pair["matching"]["matched"]
        ax.plot(
            matched_curve["fpr"],
            matched_curve["tpr"],
            color=color,
            linewidth=2.1,
            label=f"{DISPLAY_NAMES[name]} · {pair['matched_auc']:.4f}",
        )
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", linewidth=1.1, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate (background acceptance)")
    ax.set_ylabel("True-positive rate (signal efficiency)")
    fig.suptitle("EXO-200 four-model energy-matched ROC", x=0.13, y=0.96, ha="left", weight="bold", fontsize=16)
    fig.text(0.13, 0.90, "Solid curves use overlap-target weights in 6 Rotated_energy bins", color="#4B5563")
    ax.grid(color="#D1D5DB", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(output_dir / "four_model_energy_matched_roc.png", dpi=220, facecolor="white")
    fig.savefig(output_dir / "four_model_energy_matched_roc.svg", facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_metadata(config: DataConfig) -> dict[str, np.ndarray]:
    paths = sorted(config.data_root.glob("*.h5"))
    source = EXOWaveformDataset(paths, config=config)
    plan = _split_plan(source, config)
    indices = plan.indices["test"]
    labels = source.labels_for_indices(indices)

    event_id = np.empty(indices.size, dtype="U96")
    category = np.where(labels == 0, "signal", "background")
    energy = np.empty(indices.size, dtype=np.float64)
    group_id = np.empty(indices.size, dtype="U32")
    file_indices = np.searchsorted(source._offsets, indices, side="right") - 1
    for file_index in np.unique(file_indices):
        positions = np.flatnonzero(file_indices == file_index)
        info = source._files[int(file_index)]
        rows = indices[positions] - info.offset
        with h5py.File(info.path, "r") as handle:
            values = np.asarray(handle[ENERGY_FIELD][rows], dtype=np.float64)
        energy[positions] = values
        group_id[positions] = str(info.run_number)
        for position, row in zip(positions.tolist(), rows.tolist()):
            event_id[position] = (
                f"EXO200::{info.path.name}::{int(info.event_numbers[int(row)])}"
            )
    source.close()
    if not np.isfinite(energy).all() or np.any(energy <= 0.0):
        raise ValueError(f"{ENERGY_FIELD} must be finite and positive")
    return {
        "event_id": event_id,
        "label": labels.astype(np.int64),
        "category": category,
        "energy_condition": energy,
        "sample_weight": np.ones(indices.size, dtype=np.float64),
        "group_id": group_id,
        "split": np.full(indices.size, "test", dtype="U4"),
    }


def _config(model_name: str) -> dict:
    config = load_manifest()
    config.update(
        {
            "task_id": "exo200-signal-vs-background",
            "model_id": model_name,
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
                # Stored logits increase toward the EXO background class. EnergyBench
                # is standardized to signal-positive ROC, so direction is reversed.
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
                # Events share run-level correlations; retain the MJD/NEXT choice
                # to disable event bootstrap until group bootstrap is supported.
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


def main() -> int:
    data_config = DataConfig()
    metadata = _test_metadata(data_config)
    output_root = PROJECT_ROOT / "outputs" / "classification"
    reports = {}
    reference_fingerprint = None
    for model_name in MODEL_NAMES:
        model_dir = output_root / model_name
        prediction_path = model_dir / "predictions.npz"
        checkpoint_path = model_dir / "best.pt"
        with np.load(prediction_path) as predictions:
            target = np.asarray(predictions["target"], dtype=np.int64)
            score = np.asarray(predictions["prediction"], dtype=np.float64)
        if not np.array_equal(target, metadata["label"]):
            raise ValueError(f"prediction/test metadata alignment failed for {model_name}")
        columns = {**metadata, "score": score}
        provenance = {
            "adapter": "EXO200 final predictions + Rotated_energy",
            "model_id": model_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "prediction_source": str(prediction_path),
            "prediction_source_sha256": _sha256(prediction_path),
            "energy_field": ENERGY_FIELD,
            "energy_unit": "keV",
            "split": "test",
        }
        canonical_path = model_dir / "energybench_predictions.npz"
        np.savez_compressed(
            canonical_path,
            **columns,
            __metadata__=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
        bundle = PredictionBundle(columns=columns, metadata=provenance, source=canonical_path)
        destination = model_dir / "energybench_classification"
        report = run_evaluation(
            bundle,
            _config(model_name),
            destination,
            strict=True,
            allow_existing=True,
        )
        fingerprint = report["evaluation_fingerprint"]
        if reference_fingerprint is None:
            reference_fingerprint = fingerprint
        elif fingerprint != reference_fingerprint:
            raise RuntimeError("EnergyBench evaluation fingerprints differ across models")
        reports[model_name] = {
            "evaluation_fingerprint": fingerprint,
            "protocol_fingerprint": report["protocol_fingerprint"],
            "classification": report["classification"],
            "energy_dependence": report["energy_dependence"],
            "artifacts": report["artifacts"],
        }
        print(f"completed {model_name}: {destination}", flush=True)

    combined = output_root / "energy_matched_evaluation"
    combined.mkdir(parents=True, exist_ok=True)
    (combined / "summary.json").write_text(
        json.dumps(
            {
                "energy_condition": ENERGY_FIELD,
                "energy_unit": "keV",
                "evaluation_fingerprint": reference_fingerprint,
                "models": reports,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _save_combined_plots(reports, combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
