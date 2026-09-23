#!/usr/bin/env python3
"""Evaluate final MJD predictions, skipping already completed evaluations."""

from __future__ import annotations

import csv
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ENERGYBENCH_ROOT = Path("/home/wenyu/summer/evalutaions_workflow")
ARCHITECTURES = (
    "cnn_004_multiview_late_fusion",
    "gnn_001_static_gine",
    "seq_001_bigru",
    "ssm_001_pointmamba",
)
REGRESSION_COLUMNS = (
    "task", "n_events", "ers", "event_score", "histogram_similarity",
    "histogram_overlap", "jsd_bits", "wasserstein_1", "mae", "rmse",
    "bias", "r2", "mae_skill", "fractional_bias",
    "fractional_resolution_68", "balanced_fractional_mae", "finite_fraction",
)


def _load_energybench() -> dict[str, Any]:
    sys.path.insert(0, str(ENERGYBENCH_ROOT))
    from simple_energybench import EvaluationConfig, evaluate_classification
    from simple_energybench.metrics import evaluate_regression_metrics
    from simple_energybench.plotting import plot_energy_histograms, plot_energy_regression

    return {
        "EvaluationConfig": EvaluationConfig,
        "evaluate_classification": evaluate_classification,
        "evaluate_regression_metrics": evaluate_regression_metrics,
        "plot_energy_histograms": plot_energy_histograms,
        "plot_energy_regression": plot_energy_regression,
    }


def _predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"target", "prediction"}:
            raise ValueError(f"unexpected prediction fields in {path}: {archive.files}")
        target = np.asarray(archive["target"], dtype=np.float64).reshape(-1)
        prediction = np.asarray(archive["prediction"], dtype=np.float64).reshape(-1)
    if target.size == 0 or target.shape != prediction.shape:
        raise ValueError(f"invalid prediction arrays in {path}")
    return target, prediction


def _write_row(path: Path, row: dict[str, Any], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in columns})


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


class _Replay(torch.nn.Identity):
    """Give stored predictions a meaningful model name in EnergyBench plots."""


def _classification_batches(
    labels: np.ndarray, scores: np.ndarray, energies_mev: np.ndarray, batch_size: int = 4096
) -> Iterator[dict[str, Any]]:
    for start in range(0, labels.size, batch_size):
        stop = min(start + batch_size, labels.size)
        yield {
            "inputs": torch.from_numpy(scores[start:stop].astype(np.float32)),
            "label": torch.from_numpy(labels[start:stop].astype(np.int64)),
            "energy": energies_mev[start:stop],
            "event_id": np.asarray([f"mjd-test-{i}" for i in range(start, stop)]),
            "split": np.full(stop - start, "test"),
        }


def _classification_truth(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from mjdbench.data import MJDWaveformDataset, discover_files

    source = MJDWaveformDataset(discover_files(data_root, "test"), task="classification")
    try:
        labels = np.concatenate([item.clean for item in source._files])  # noqa: SLF001
        energies = np.concatenate([item.energies for item in source._files])  # noqa: SLF001
    finally:
        source.close()
    return labels.astype(np.int64), energies.astype(np.float64) / 1000.0


def _evaluate_classifications(energybench: dict[str, Any], force: bool = False) -> tuple[int, int]:
    missing: list[tuple[str, Path, np.ndarray, np.ndarray]] = []
    for architecture in ARCHITECTURES:
        model_dir = OUTPUT_ROOT / "classification" / architecture
        destination = model_dir / "energybench_classification" / "clean"
        if (destination / "results.csv").is_file() and not force:
            print(f"SKIP completed classification: {architecture}")
            continue
        target, prediction = _predictions(model_dir / "predictions.npz")
        missing.append((architecture, destination, target, prediction))
    if not missing:
        return 0, len(ARCHITECTURES)

    config = json.loads((OUTPUT_ROOT / "classification" / missing[0][0] / "run_config.json").read_text())
    labels, energies_mev = _classification_truth(Path(config["data"]["data_root"]))
    # The user-selected overflow policy assigns every event above 3000 keV
    # to the final canonical bin instead of dropping it.
    energies_mev = np.clip(energies_mev, 0.0, 3.0)
    for architecture, destination, target, prediction in missing:
        if target.size != labels.size or not np.array_equal(target.astype(np.int64), labels):
            raise ValueError(f"stored classification targets do not match official test data: {architecture}")
        in_protocol = np.isfinite(energies_mev)
        replay_type = type(architecture, (_Replay,), {})
        result = energybench["evaluate_classification"](
            replay_type(),
            _classification_batches(
                labels[in_protocol], prediction[in_protocol], energies_mev[in_protocol]
            ),
            device="cpu",
            output_dir=destination,
            config=energybench["EvaluationConfig"](),
            overwrite=force,
        )
        print(f"DONE classification {architecture}: AUC={result['auc']:.8f}")
    return len(missing), len(ARCHITECTURES) - len(missing)


def _evaluate_regressions(energybench: dict[str, Any], force: bool = False) -> tuple[int, int]:
    exported = skipped = 0
    protocol = energybench["EvaluationConfig"]()
    for architecture in ARCHITECTURES:
        model_dir = OUTPUT_ROOT / "regression" / architecture
        metrics_path = model_dir / "energybench_metrics.json"
        result_path = model_dir / "results.csv"
        plots_exist = all((model_dir / name).is_file() for name in (
            "energy_regression.png", "energy_histograms.png"
        ))
        if metrics_path.is_file() and plots_exist and not force:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not result_path.is_file():
                row = {"task": "regression", "n_events": metrics.get("n_total")}
                row.update({key: metrics.get(key, "") for key in REGRESSION_COLUMNS[2:]})
                row["ers"] = metrics.get("ers", metrics.get("energy_regression_score", ""))
                row["histogram_similarity"] = metrics.get(
                    "histogram_similarity", metrics.get("hist_similarity", "")
                )
                _write_row(result_path, row, REGRESSION_COLUMNS)
                print(f"STANDARDIZED existing regression CSV: {architecture}")
            else:
                print(f"SKIP completed regression: {architecture}")
            skipped += 1
            continue

        target_kev, prediction_kev = _predictions(model_dir / "predictions.npz")
        target_mev, prediction_mev = target_kev / 1000.0, prediction_kev / 1000.0
        in_protocol = np.isfinite(target_mev)
        target_mev = target_mev[in_protocol]
        prediction_mev = prediction_mev[in_protocol]
        target_mev = np.clip(target_mev, 0.0, 3.0)
        metrics = energybench["evaluate_regression_metrics"](
            target_mev, prediction_mev, None, protocol
        )
        metrics["ers"] = metrics.get("ers", metrics.get("energy_regression_score"))
        metrics["histogram_similarity"] = metrics.get(
            "histogram_similarity", metrics.get("hist_similarity")
        )
        energybench["plot_energy_regression"](
            target_mev, prediction_mev, metrics,
            model_dir / "energy_regression.png", "MeV", protocol.seed,
        )
        energybench["plot_energy_histograms"](
            metrics, model_dir / "energy_histograms.png", "MeV"
        )
        metrics_path.write_text(
            json.dumps(_json_ready(metrics), indent=2) + "\n", encoding="utf-8"
        )
        row = {"task": "regression", "n_events": int(target_mev.size)}
        row.update({key: metrics.get(key, "") for key in REGRESSION_COLUMNS[2:]})
        _write_row(result_path, row, REGRESSION_COLUMNS)
        print(f"DONE regression {architecture}: ERS={metrics['ers']:.8f}")
        exported += 1
    return exported, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="reevaluate completed outputs")
    args = parser.parse_args()
    energybench = _load_energybench()
    class_done, class_skipped = _evaluate_classifications(energybench, args.force)
    reg_done, reg_skipped = _evaluate_regressions(energybench, args.force)
    print(f"Evaluation complete: ran={class_done + reg_done}, skipped={class_skipped + reg_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
