from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

from exobench.config import DataConfig
from exobench.data import EXOWaveformDataset, _split_plan, discover_files
from exo_transformer import energy_aware
from exo_transformer.protocol import load_split_manifest, validate_split_manifest


def _write_exo_file(root: Path, run: int, events: int = 4) -> None:
    path = root / f"{run}-000.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["n_events"] = events
        handle.attrs["run_number"] = run
        waveforms = handle.create_group("Waveforms")
        positions = handle.create_group("Charge_Clusters_Pos")
        nccl = np.asarray([1, 2, 1, 2], dtype=np.int32)
        for index in range(events):
            waveforms.create_dataset(
                str(index), data=np.zeros((226, 300), dtype=np.int16)
            )
            positions.create_dataset(
                str(index), data=np.zeros((3, int(nccl[index])), dtype=np.float32)
            )
        handle.create_dataset("Charge_cluster_number", data=nccl)
        handle.create_dataset(
            "event_number", data=np.arange(events, dtype=np.int32) + run * 100
        )
        handle.create_dataset(
            "Rotated_energy",
            data=np.linspace(500.0, 2500.0, events, dtype=np.float32),
        )


def _fake_metadata() -> dict[str, np.ndarray]:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return {
        "event_id": np.asarray([f"event-{i}" for i in range(4)]),
        "label": labels,
        "category": np.where(labels == 0, "signal", "background"),
        "energy_condition": np.asarray([500.0, 600.0, 700.0, 800.0]),
        "sample_weight": np.ones(4, dtype=np.float64),
        "group_id": np.asarray(["8967"] * 4),
        "split": np.asarray(["test"] * 4),
        "run_number": np.asarray([8967] * 4, dtype=np.int64),
        "event_number": np.arange(4, dtype=np.int64),
    }


def _fake_report() -> dict:
    return {
        "evaluation_fingerprint": "evaluation-fingerprint",
        "protocol_fingerprint": "protocol-fingerprint",
        "classification": {
            "aggregates": {
                "inclusive_auc_macro": 0.91,
                "matched_auc_macro": 0.89,
            },
            "pairs": [
                {
                    "inclusive_auc": 0.91,
                    "common_support_auc": 0.90,
                    "matched_auc": 0.89,
                    "shortcut_gap": 0.01,
                    "matched_auc_status": "ok",
                    "coverage": 0.95,
                    "matching": {
                        "matched": {
                            "fpr": [0.0, 1.0],
                            "tpr": [0.0, 1.0],
                        }
                    },
                }
            ],
        },
        "energy_dependence": {
            "overall_energy_independence_score": 0.87,
            "worst_group_energy_independence_score": 0.84,
        },
    }


class EnergyAwareTests(unittest.TestCase):
    def test_vendored_energybench_matches_recorded_fingerprint(self) -> None:
        source = energy_aware.DEFAULT_FROZEN_ENERGYBENCH_ROOT
        provenance = json.loads(
            (source / "SOURCE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            energy_aware._tree_sha256(source / "energybench"),
            provenance["tree_sha256"],
        )

    def test_official_split_manifest_contract(self) -> None:
        manifest = load_split_manifest()
        prepared = SimpleNamespace(
            counts=manifest["counts"],
            class_counts=manifest["class_counts"],
            runs=manifest["runs"],
            overlap_counts=manifest["overlap_counts"],
        )
        self.assertEqual(validate_split_manifest(prepared), manifest)
        prepared.counts = {**manifest["counts"], "test": 1}
        with self.assertRaisesRegex(RuntimeError, "counts differ"):
            validate_split_manifest(prepared)

    def test_frozen_configuration(self) -> None:
        config = energy_aware.build_evaluation_config("model", lambda: {})
        self.assertEqual(config["classification"]["positive_label"], "0")
        self.assertEqual(config["classification"]["score_direction"], "lower")
        self.assertEqual(config["classification"]["energy_bins"], 6)
        self.assertEqual(config["classification"]["matching_target"], "overlap")
        self.assertEqual(config["dependence"]["energy_bins"], 8)
        self.assertEqual(config["dependence"]["score_bins"], 20)

    def test_metadata_follows_exobench_test_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for run in (100, 101, 102, 103):
                _write_exo_file(root, run)
            config = DataConfig(
                data_root=root,
                validation_fraction=0.25,
                test_fraction=0.25,
                seed=42,
            )
            source = EXOWaveformDataset(discover_files(root), config=config)
            try:
                plan = _split_plan(source, config)
                expected_labels = source.labels_for_indices(plan.indices["test"])
                expected_runs = plan.runs["test"]
                expected_events = plan.counts["test"]
            finally:
                source.close()
            metadata = energy_aware.build_test_metadata(
                config,
                expected_runs=expected_runs,
                expected_events=expected_events,
            )
            np.testing.assert_array_equal(metadata["label"], expected_labels)
            self.assertEqual(len(np.unique(metadata["event_id"])), expected_events)
            self.assertTrue(np.isfinite(metadata["energy_condition"]).all())
            self.assertEqual(
                sorted(np.unique(metadata["run_number"]).tolist()), expected_runs
            )

    def test_prediction_alignment_rejects_changed_order(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npz"
            np.savez(path, target=labels[::-1], prediction=np.arange(4.0))
            with self.assertRaisesRegex(ValueError, "alignment failed"):
                energy_aware._validate_prediction_arrays(path, labels)

    def test_stale_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "energybench_classification"
            destination.mkdir()
            provenance_path = root / "energybench_input_provenance.json"
            provenance_path.write_text(
                json.dumps({"prediction_sha256": "old"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                energy_aware._guard_existing(
                    destination,
                    {"prediction_sha256": "new"},
                    provenance_path,
                )

    def test_matching_legacy_provenance_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "energybench_classification"
            destination.mkdir()
            provenance = {"prediction_sha256": "same"}
            legacy_path = destination / "input_provenance.json"
            legacy_path.write_text(json.dumps(provenance), encoding="utf-8")
            energy_aware._guard_existing(
                destination,
                provenance,
                root / "energybench_input_provenance.json",
            )
            self.assertFalse(legacy_path.exists())

    def test_completed_run_is_evaluated_without_loading_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = energy_aware.RUN_IDS[0]
            run_dir = root / run_id
            run_dir.mkdir(parents=True)
            labels = _fake_metadata()["label"]
            np.savez(
                run_dir / "predictions.npz",
                target=labels,
                prediction=np.asarray([1.0, -1.0, 0.5, -0.5]),
            )
            (run_dir / "best.pt").write_bytes(b"checkpoint-is-only-hashed")
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "representation": {
                            "tokenization": {"tokenization": "raw_patches"},
                            "position_encoding": "coordinate_mlp",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "tokenization": "raw_patches",
                        "position_encoding": "coordinate_mlp",
                        "best_validation_auc": 0.88,
                        "epochs_completed": 4,
                        "minutes_per_epoch": 1.5,
                    }
                ),
                encoding="utf-8",
            )

            class Bundle:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            runtime = energy_aware.EnergyBenchRuntime(
                prediction_bundle=Bundle,
                run_evaluation=lambda *args, **kwargs: _fake_report(),
                load_manifest=lambda: {},
                source_root=root,
                source_sha256="energybench-hash",
                source_files_sha256={"metrics.py": "metrics-hash"},
            )
            with mock.patch.object(
                energy_aware, "load_energybench_runtime", return_value=runtime
            ), mock.patch.object(
                energy_aware, "build_test_metadata", return_value=_fake_metadata()
            ):
                results, statuses, reports = energy_aware.evaluate_transformer_runs(
                    data_config=DataConfig(data_root=root),
                    output_root=root,
                    energybench_source=root,
                    expected_events=4,
                    expected_runs=(8967,),
                )

            self.assertEqual(results.shape[0], 1)
            self.assertEqual(results.iloc[0]["energy_matched_auc"], 0.89)
            self.assertEqual(results.iloc[0]["energy_independence_score"], 0.87)
            self.assertEqual(statuses["status"].value_counts()["incomplete"], 5)
            self.assertIn(run_id, reports)
            self.assertTrue((run_dir / "energybench_predictions.npz").is_file())
            self.assertTrue((root / "energybench_transformer_results.csv").is_file())
            self.assertTrue((root / "energybench_transformer_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
