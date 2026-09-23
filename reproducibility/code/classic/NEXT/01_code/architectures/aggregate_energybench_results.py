#!/usr/bin/env python3
"""Aggregate EnergyBench model/task results and build comparison figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ARCHITECTURES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ARCHITECTURES_ROOT.parents[1]
DEFAULT_CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "03_training_runs"
    / "energybench_campaigns"
    / "20260808_energybench_rewrite_v2"
)

FAMILY = {
    "cnn": "CNN",
    "point": "Point Cloud",
    "gnn": "Graph Neural Network",
    "hybrid": "Hybrid CNN-GNN",
    "classic": "Classical ML",
    "seq": "Sequence Model",
    "mixer": "MLP Mixer",
    "topo": "Topological Model",
    "ssm": "State Space Model",
    "sparse": "Sparse 3D CNN",
}

ARCHITECTURE = {
    "cnn_001_two_conv_baseline": "Two-convolution three-view projection CNN",
    "cnn_002_global_energy_skip": "Projection CNN with global-energy skip connection",
    "cnn_003_residual_spatial": "Residual spatial three-view projection CNN",
    "cnn_004_multiview_late_fusion": "Multi-view 2D CNN with late fusion",
    "cnn_005_multiscale_projection": "Multi-scale projection CNN",
    "cnn_006_dense_3d_resnet": "Dense 3D residual CNN",
    "point_001_deepsets": "DeepSets point-cloud encoder",
    "point_002_pointnetpp": "PointNet++ hierarchical point-cloud network",
    "gnn_001_static_gine": "Static-radius graph with GINE message passing",
    "gnn_002_particlenet_edgeconv": "ParticleNet-style dynamic EdgeConv graph network",
    "gnn_003_egnn": "E(n)-equivariant graph neural network",
    "gnn_004_gravnet": "GravNet learned-neighbour graph network",
    "hybrid_001_cnn_gnn": "Late-fusion projection CNN and graph network",
    "classic_001_topology_xgboost": "Engineered topology features with XGBoost",
    "point_003_pointmlp": "PointMLP hierarchical point-cloud network",
    "seq_001_bigru": "Hilbert-ordered bidirectional GRU",
    "seq_002_dilated_tcn": "Hilbert-ordered dilated temporal CNN",
    "mixer_001_projection_mlp_mixer": "Projection-token MLP-Mixer",
    "gnn_005_dimenet_lite": "DimeNet-lite directional graph network",
    "point_004_rigid_kpconv": "Rigid kernel-point convolution network",
    "topo_001_persistence_perslay": "Persistent-homology features with PersLay",
    "ssm_001_pointmamba": "Hilbert-ordered point state-space model",
    "sparse_001_submanifold_resnet": "Submanifold sparse 3D residual CNN",
}

MISSING_RUN_METADATA = {
    "ssm_001_pointmamba": {
        "model_name": "PointMambaLiteClassifier",
        "input_kind": "sequence",
        "parameter_count": 316993,
    }
}

RESULT_COLUMNS = [
    "task",
    "n_events",
    "auc",
    "matched_auc",
    "matched_auc_status",
    "common_support_auc",
    "shortcut_gap",
    "energy_independence_score",
    "worst_energy_independence_score",
    "ers",
    "event_score",
    "histogram_similarity",
    "histogram_overlap",
    "jsd_bits",
    "wasserstein_1",
    "mae",
    "rmse",
    "bias",
    "r2",
    "mae_skill",
    "fractional_bias",
    "fractional_resolution_68",
    "balanced_fractional_mae",
    "finite_fraction",
]

IDENTITY_COLUMNS = [
    "evaluation_status",
    "job_id",
    "model_category",
    "architecture_id",
    "model_name",
    "architecture",
    "input_representation",
    "backend",
    "source_module",
    "trainable_parameters",
    "parameter_count_type",
    "tree_count",
    "tree_node_count",
    "best_epoch",
    "epochs_completed",
]

PATH_COLUMNS = [
    "result_csv",
    "metrics_json",
    "predictions_npz",
    "plot_1",
    "plot_2",
]


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one result row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in columns})


def classic_structure(checkpoint: Path) -> tuple[int | None, int | None]:
    if not checkpoint.is_file():
        return None, None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = payload.get("model") or {}
    return model.get("tree_count"), model.get("tree_node_count")


def spec_metadata(architecture_id: str) -> dict[str, Any]:
    prefix = architecture_id.split("_", 1)[0]
    config_path = ARCHITECTURES_ROOT / architecture_id / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    return {
        "model_category": FAMILY.get(prefix, prefix),
        "architecture": ARCHITECTURE.get(architecture_id, architecture_id),
        "source_module": (
            "next_cnn"
            if architecture_id in {
                "cnn_001_two_conv_baseline",
                "cnn_002_global_energy_skip",
                "cnn_003_residual_spatial",
            }
            else "next_alt"
        ),
    }


def collect_rows(campaign_root: Path, reevaluation_root: Path) -> list[dict[str, Any]]:
    source_manifest = json.loads((campaign_root / "manifest.json").read_text(encoding="utf-8"))
    reevaluation_manifest = json.loads((reevaluation_root / "manifest.json").read_text(encoding="utf-8"))
    status_by_id = {job["job_id"]: job for job in reevaluation_manifest["jobs"]}
    rows: list[dict[str, Any]] = []
    for source_job in source_manifest["jobs"]:
        job_id = str(source_job["job_id"])
        architecture_id = str(source_job["architecture_id"])
        task = str(source_job["task"])
        run_root = campaign_root / "runs" / architecture_id / task
        summary_path = run_root / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        fallback = MISSING_RUN_METADATA.get(architecture_id, {})
        evaluation_job = status_by_id[job_id]
        evaluation_dir = Path(evaluation_job["output_dir"])
        result_path = evaluation_dir / "results.csv"
        result = read_csv_row(result_path) if result_path.is_file() else {}
        metadata = spec_metadata(architecture_id)
        backend = summary.get("backend")
        if backend is None:
            config_path = ARCHITECTURES_ROOT / architecture_id / "config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            backend = "xgboost" if architecture_id.startswith("classic_") else "torch"
        tree_count = tree_nodes = None
        if backend == "xgboost":
            tree_count, tree_nodes = classic_structure(run_root / "training" / "best_model.pt")
        plots = (
            ("energy_matched_roc.png", "score_energy_dependence.png")
            if task == "classification"
            else ("energy_regression.png", "energy_histograms.png")
        )
        row: dict[str, Any] = {
            "evaluation_status": evaluation_job["status"],
            "job_id": job_id,
            "architecture_id": architecture_id,
            "model_name": summary.get("model_name") or fallback.get("model_name") or ARCHITECTURE.get(architecture_id, architecture_id),
            "input_representation": summary.get("input_kind") or fallback.get("input_kind", ""),
            "backend": backend,
            "trainable_parameters": summary.get("parameter_count", fallback.get("parameter_count")),
            "parameter_count_type": "trainable neural-network parameters" if backend == "torch" else "not applicable (tree ensemble)",
            "tree_count": tree_count,
            "tree_node_count": tree_nodes,
            "best_epoch": summary.get("best_epoch"),
            "epochs_completed": summary.get("epochs_completed"),
            "result_csv": str(result_path) if result_path.is_file() else "",
            "metrics_json": str(evaluation_dir / "metrics.json") if (evaluation_dir / "metrics.json").is_file() else "",
            "predictions_npz": str(evaluation_dir / "predictions.npz") if (evaluation_dir / "predictions.npz").is_file() else "",
            "plot_1": str(evaluation_dir / plots[0]) if (evaluation_dir / plots[0]).is_file() else "",
            "plot_2": str(evaluation_dir / plots[1]) if (evaluation_dir / plots[1]).is_file() else "",
            **metadata,
            **{key: result.get(key, "") for key in RESULT_COLUMNS},
        }
        row["task"] = task
        rows.append(row)
    return rows


def copy_plots(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for index, key in enumerate(("plot_1", "plot_2"), start=1):
            source_value = row.get(key)
            if not source_value:
                continue
            source = Path(str(source_value))
            output = destination / f"{row['architecture_id']}__{row['task']}__{source.name}"
            shutil.copy2(source, output)
            row[key] = str(output)


def numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else math.nan


def label(architecture_id: str) -> str:
    return architecture_id.replace("_", " ").replace("classification", "cls")


def style_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#D9DEE7", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def classification_ranking(rows: list[dict[str, Any]], path: Path) -> None:
    values = [r for r in rows if r["task"] == "classification" and r["evaluation_status"] == "DONE"]
    values.sort(key=lambda r: numeric(r, "matched_auc"))
    y = np.arange(len(values))
    matched = [numeric(r, "matched_auc") for r in values]
    inclusive = [numeric(r, "auc") for r in values]
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.barh(y, matched, color="#3B6FB6", edgecolor="#244A7C", linewidth=0.6, label="Energy-matched AUC")
    ax.scatter(inclusive, y, color="#D69E2E", edgecolor="#6D4C12", s=28, zorder=3, label="Inclusive AUC")
    ax.set_yticks(y, [label(r["architecture_id"]) for r in values], fontsize=8)
    ax.set_xlim(0.86, 1.0)
    ax.set_xlabel("AUC")
    fig.suptitle("Classification performance by architecture", x=0.125, y=0.985, ha="left", weight="bold")
    fig.text(0.125, 0.958, "Held-out test set, n=116,549 per model; canonical 5 keV EnergyBench protocol", fontsize=9, color="#556070")
    ax.legend(loc="lower right", frameon=False)
    style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parameter_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    values = [r for r in rows if r["task"] == "classification" and r["evaluation_status"] == "DONE" and r.get("trainable_parameters") not in (None, "")]
    x = np.asarray([numeric(r, "trainable_parameters") for r in values])
    y = np.asarray([numeric(r, "matched_auc") for r in values])
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(x, y, s=125, color="#3B6FB6", edgecolor="#244A7C", alpha=0.92)
    for index, (xv, yv) in enumerate(zip(x, y), start=1):
        ax.text(xv, yv, str(index), ha="center", va="center", fontsize=6.5, color="white", weight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Energy-matched AUC")
    ax.set_ylim(0.86, 1.0)
    fig.suptitle("Classification performance and model size", x=0.08, y=0.985, ha="left", weight="bold")
    fig.text(0.08, 0.95, "Torch models only; XGBoost excluded because tree size is not a trainable-parameter count", fontsize=9, color="#556070")
    midpoint = math.ceil(len(values) / 2)
    for index, row in enumerate(values, start=1):
        column = 0 if index <= midpoint else 1
        position = index - 1 if column == 0 else index - midpoint - 1
        fig.text(
            0.74 + column * 0.13,
            0.86 - position * 0.064,
            f"{index:>2}  {row['architecture_id']}",
            fontsize=7.2,
            color="#303846",
        )
    style_axis(ax)
    fig.tight_layout(rect=(0.06, 0.04, 0.72, 0.92))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def independence_ranking(rows: list[dict[str, Any]], path: Path) -> None:
    values = [r for r in rows if r["task"] == "classification" and r["evaluation_status"] == "DONE"]
    values.sort(key=lambda r: numeric(r, "energy_independence_score"))
    y = np.arange(len(values))
    scores = [numeric(r, "energy_independence_score") for r in values]
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.barh(y, scores, color="#6B8E5A", edgecolor="#3E5833", linewidth=0.6)
    ax.set_yticks(y, [label(r["architecture_id"]) for r in values], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Energy independence score (higher is better)")
    fig.suptitle("Classification score energy independence", x=0.125, y=0.985, ha="left", weight="bold")
    fig.text(0.125, 0.958, "Held-out test set; canonical 5 keV EnergyBench dependence analysis", fontsize=9, color="#556070")
    style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def regression_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    values = [r for r in rows if r["task"] == "regression" and r["evaluation_status"] == "DONE"]
    names = [label(r["architecture_id"]) for r in values]
    y = np.arange(len(values))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(y, [numeric(r, "ers") for r in values], color="#3B6FB6", edgecolor="#244A7C")
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].set_xlim(0, 1.0)
    axes[0].set_xlabel("ERS-v1 (higher is better)")
    axes[0].set_title("Energy regression score", loc="left", weight="bold")
    axes[1].barh(y, [numeric(r, "rmse") for r in values], color="#D69E2E", edgecolor="#6D4C12")
    axes[1].set_yticks(y, names, fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("RMSE in MeV (log scale; lower is better)")
    axes[1].set_title("Energy regression error", loc="left", weight="bold")
    for axis in axes:
        style_axis(axis)
    fig.suptitle("Regression performance by architecture", x=0.02, ha="left", weight="bold")
    fig.text(0.02, 0.92, "Held-out test set, n=116,549 per model; canonical EnergyBench protocol", fontsize=9, color="#556070")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def field_dictionary(path: Path) -> None:
    descriptions = {
        "evaluation_status": "DONE or UNAVAILABLE in the 2026-08-18 canonical reevaluation",
        "job_id": "Unique architecture:task key",
        "model_category": "Normalized model family",
        "architecture_id": "Repository architecture identifier",
        "model_name": "Implemented model class/name",
        "architecture": "Human-readable architecture description",
        "input_representation": "Input representation consumed by the model",
        "backend": "torch or xgboost",
        "source_module": "Source implementation family from config/run metadata",
        "trainable_parameters": "Number of trainable neural-network parameters; blank for XGBoost",
        "parameter_count_type": "Definition used for model-size fields",
        "tree_count": "XGBoost tree count; blank for neural networks",
        "tree_node_count": "Total XGBoost tree nodes; blank for neural networks",
        "best_epoch": "Training epoch/boosting round selected on validation data",
        "epochs_completed": "Total training epochs/rounds completed",
        "task": "classification or regression",
        "n_events": "Held-out test events evaluated",
        "auc": "Inclusive classification ROC AUC",
        "matched_auc": "Formal energy-matched classification AUC",
        "matched_auc_status": "Validity/status of matched AUC",
        "common_support_auc": "AUC within common energy support",
        "shortcut_gap": "Inclusive minus energy-matched performance diagnostic",
        "energy_independence_score": "Overall score-energy independence; higher is better",
        "worst_energy_independence_score": "Worst class/group independence score",
        "ers": "Energy Regression Score v1; higher is better",
        "event_score": "Event-level component of ERS-v1",
        "histogram_similarity": "Truth/prediction spectrum similarity",
        "histogram_overlap": "Truth/prediction spectrum overlap",
        "jsd_bits": "Jensen-Shannon divergence in bits",
        "wasserstein_1": "First Wasserstein distance in MeV",
        "mae": "Mean absolute error in MeV",
        "rmse": "Root mean square error in MeV",
        "bias": "Mean prediction bias in MeV",
        "r2": "Coefficient of determination",
        "mae_skill": "MAE skill relative to workflow baseline",
        "fractional_bias": "Weighted fractional prediction bias",
        "fractional_resolution_68": "Central 68% fractional resolution",
        "balanced_fractional_mae": "Truth-balanced fractional MAE",
        "finite_fraction": "Fraction of finite regression predictions",
        "result_csv": "Source reevaluation results.csv",
        "metrics_json": "Full nested reevaluation metrics",
        "predictions_npz": "Held-out predictions used in reevaluation",
        "plot_1": "First corresponding workflow plot copied into the collection",
        "plot_2": "Second corresponding workflow plot copied into the collection",
    }
    write_csv(path, [{"column": key, "description": descriptions.get(key, "")} for key in [*IDENTITY_COLUMNS, *RESULT_COLUMNS, *PATH_COLUMNS]], ["column", "description"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--reevaluation-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    reevaluation_root = (args.reevaluation_root or campaign_root / "reevaluation_20260818").expanduser().resolve()
    output_dir = (args.output_dir or reevaluation_root / "all_model_results").expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = collect_rows(campaign_root, reevaluation_root)
    copy_plots(rows, output_dir / "individual_plots")
    columns = [*IDENTITY_COLUMNS, *RESULT_COLUMNS, *PATH_COLUMNS]
    write_csv(output_dir / "all_model_results.csv", rows, columns)
    field_dictionary(output_dir / "field_dictionary.csv")
    figures = output_dir / "comparison_figures"
    figures.mkdir()
    classification_ranking(rows, figures / "classification_auc_ranking.png")
    parameter_scatter(rows, figures / "classification_parameter_efficiency.png")
    independence_ranking(rows, figures / "classification_energy_independence.png")
    regression_comparison(rows, figures / "regression_performance.png")
    keys = [row["job_id"] for row in rows]
    done = [row for row in rows if row["evaluation_status"] == "DONE"]
    qa = {
        "row_count": len(rows),
        "unique_job_ids": len(set(keys)),
        "duplicate_job_ids": sorted({key for key in keys if keys.count(key) > 1}),
        "done_rows": len(done),
        "unavailable_rows": len(rows) - len(done),
        "classification_rows": sum(row["task"] == "classification" for row in rows),
        "regression_rows": sum(row["task"] == "regression" for row in rows),
        "done_rows_with_116549_events": sum(
            row.get("n_events") == "116549" for row in done
        ),
        "individual_plot_count": len(list((output_dir / "individual_plots").glob("*.png"))),
        "comparison_figure_count": len(list(figures.glob("*.png"))),
        "missing_result_jobs": [row["job_id"] for row in rows if row["evaluation_status"] != "DONE"],
        "notes": [
            "Classification and regression metrics occupy different columns; non-applicable cells are blank.",
            "XGBoost tree counts are reported separately and are not treated as neural-network parameters.",
            "PointMamba is retained as UNAVAILABLE because the workflow campaign produced no held-out predictions.",
        ],
    }
    (output_dir / "qa_summary.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
