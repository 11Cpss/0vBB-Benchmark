from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from next_cnn import data as data_module
from next_cnn.data import (
    EventRecord,
    NextIterableDataset,
    ProjectionConfig,
    SourceFile,
)


def _load_training_module():
    path = (
        PROJECT_ROOT
        / "01_code"
        / "architectures"
        / "cnn_003_residual_spatial"
        / "train_energy_regression.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cnn003_energy_training_for_tests", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


training = _load_training_module()


def _source(name: str, label: int) -> SourceFile:
    category = "0nubb" if label == 1 else "Bi214"
    return SourceFile(
        path=Path("/unused") / name,
        relative_path=name,
        group_id=name,
        label=label,
        category=category,
        split="train",
    )


def _event(source: SourceFile, index: int, last: bool) -> EventRecord:
    return EventRecord(
        event_id="%s::%d" % (source.relative_path, index),
        source_event_id=index,
        group_id=source.group_id,
        label=source.label,
        category=source.category,
        split=source.split,
        coordinates=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        energies=np.asarray([1.0], dtype=np.float32),
        energy_sum=1.0,
        is_last_in_file=last,
    )


class DatasetShuffleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.files = [
            _source("signal_a.h5", 1),
            _source("signal_b.h5", 1),
            _source("background_a.h5", 0),
            _source("background_b.h5", 0),
        ]
        counts = {
            "signal_a.h5": 5,
            "signal_b.h5": 5,
            "background_a.h5": 15,
            "background_b.h5": 15,
        }
        self.events = {
            source.relative_path: [
                _event(source, index, index == counts[source.relative_path] - 1)
                for index in range(counts[source.relative_path])
            ]
            for source in self.files
        }

    def _iter_file_events(self, source: SourceFile):
        yield from self.events[source.relative_path]

    def _ids(self, dataset: NextIterableDataset):
        with mock.patch.object(
            data_module, "iter_file_events", self._iter_file_events
        ):
            return [row["event_id"] for row in dataset]

    def test_buffered_shuffle_is_complete_reproducible_and_epoch_specific(self):
        first = NextIterableDataset(
            self.files,
            projection=ProjectionConfig(
                normalize_energy=False,
                input_scale=1.0,
                representation="binary_occupancy",
            ),
            shuffle_files=True,
            balance_classes=False,
            seed=17,
            event_shuffle_buffer_size=128,
        )
        first.set_epoch(3)
        ids_first = self._ids(first)

        repeated = NextIterableDataset(
            self.files,
            projection=first.projection,
            shuffle_files=True,
            balance_classes=False,
            seed=17,
            event_shuffle_buffer_size=128,
        )
        repeated.set_epoch(3)
        ids_repeated = self._ids(repeated)
        repeated.set_epoch(4)
        ids_next_epoch = self._ids(repeated)

        expected = {
            event.event_id
            for events in self.events.values()
            for event in events
        }
        self.assertEqual(ids_first, ids_repeated)
        self.assertNotEqual(ids_first, ids_next_epoch)
        self.assertEqual(set(ids_first), expected)
        self.assertEqual(len(ids_first), len(expected))
        tail_labels = [
            1 if event_id.startswith("signal") else 0
            for event_id in ids_first[-20:]
        ]
        self.assertIn(1, tail_labels)
        self.assertIn(0, tail_labels)

    def test_balanced_mode_keeps_pairwise_alternation_and_shorter_class_size(self):
        dataset = NextIterableDataset(
            self.files,
            projection=ProjectionConfig(
                normalize_energy=False,
                input_scale=1.0,
                representation="binary_occupancy",
            ),
            shuffle_files=True,
            balance_classes=True,
            seed=11,
            event_shuffle_buffer_size=128,
        )
        with mock.patch.object(
            data_module, "iter_file_events", self._iter_file_events
        ):
            labels = [int(row["label"]) for row in dataset]
        self.assertEqual(len(labels), 20)
        self.assertEqual(labels, [1, 0] * 10)

    def test_worker_file_slices_have_no_overlap_or_loss(self):
        dataset = NextIterableDataset(
            self.files,
            projection=ProjectionConfig(
                normalize_energy=False,
                input_scale=1.0,
                representation="binary_occupancy",
            ),
            shuffle_files=True,
            balance_classes=False,
            seed=23,
            event_shuffle_buffer_size=0,
        )
        worker_ids = []
        with mock.patch.object(
            data_module, "iter_file_events", self._iter_file_events
        ):
            for worker_id in (0, 1):
                worker = SimpleNamespace(id=worker_id, num_workers=2)
                with mock.patch.object(
                    data_module._torch.utils.data,
                    "get_worker_info",
                    return_value=worker,
                ):
                    worker_ids.append(
                        {row["event_id"] for row in dataset}
                    )
        expected = {
            event.event_id
            for events in self.events.values()
            for event in events
        }
        self.assertFalse(worker_ids[0].intersection(worker_ids[1]))
        self.assertEqual(worker_ids[0].union(worker_ids[1]), expected)

    def test_invalid_shuffle_buffer_is_rejected(self):
        with self.assertRaises(ValueError):
            NextIterableDataset(self.files, event_shuffle_buffer_size=-1)
        with self.assertRaises(ValueError):
            NextIterableDataset(self.files, event_shuffle_buffer_size=1.5)


class TrainingContractTests(unittest.TestCase):
    def test_compact_model_zero_initialization_predicts_standardized_zero(self):
        model = training.EnergyRegressor(
            base_channels=4,
            pooled_size=1,
            head_features=32,
        )
        training.initialize_regression_output(model)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 45861)
        with torch.inference_mode():
            output = model(torch.rand(2, 3, 128, 128))
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_selection_uses_rmse_then_mae(self):
        incumbent = {"energy_rmse_mev": 0.01, "energy_mae_mev": 0.008}
        lower_rmse = {"energy_rmse_mev": 0.009, "energy_mae_mev": 0.02}
        tied_lower_mae = {
            "energy_rmse_mev": 0.01,
            "energy_mae_mev": 0.007,
        }
        tied_higher_mae = {
            "energy_rmse_mev": 0.01,
            "energy_mae_mev": 0.009,
        }
        self.assertTrue(training.validation_improved(lower_rmse, incumbent))
        self.assertTrue(training.validation_improved(tied_lower_mae, incumbent))
        self.assertFalse(
            training.validation_improved(tied_higher_mae, incumbent)
        )

    def test_early_stopping_uses_strict_min_delta(self):
        self.assertTrue(training.early_stopping_improved(0.009, 0.01, 1e-6))
        self.assertFalse(
            training.early_stopping_improved(0.0099995, 0.01, 1e-6)
        )

    def test_metrics_report_prediction_spread_and_correlation(self):
        target = np.asarray([1.0, 2.0, 3.0])
        prediction = np.asarray([1.5, 2.5, 3.5])
        metrics = training.complete_regression_metrics(
            target,
            prediction,
            energy_std=1.0,
            objective_name="mse",
            smooth_l1_beta=1.0,
        )
        self.assertAlmostEqual(metrics["loss"], 0.25)
        self.assertAlmostEqual(metrics["smooth_l1_loss"], 0.125)
        self.assertAlmostEqual(metrics["energy_pearson_r"], 1.0)
        self.assertGreater(metrics["energy_prediction_std_mev"], 0.0)

    def test_legacy_v2_checkpoint_defaults_to_supported_contract(self):
        checkpoint = {
            "format_version": 2,
            "task": "energy_regression",
            "model_name": "ResidualSpatialNextCNN",
            "model_config": {
                "base_channels": 16,
                "pooled_size": 4,
                "head_features": 256,
            },
            "energy_target_config": {
                "normalizer": {
                    "mean": 2.45,
                    "std": 0.01,
                    "fit_split": "train",
                    "transform": "standardize",
                }
            },
            "projection_config": {
                "grid_size": 128,
                "bin_size": 30.0,
                "origin": [-1920.0, -1920.0, -120.0],
                "normalize_energy": False,
                "input_scale": 1.0,
                "representation": "binary_occupancy",
            },
        }
        self.assertIs(
            training.validate_checkpoint(checkpoint, Path("legacy.pt")),
            checkpoint,
        )
        self.assertEqual(training.normalize_regression_loss("mse"), "mse")
        self.assertEqual(
            training.normalize_regression_loss("smooth_l1"), "smooth_l1"
        )

    def test_checkpoint_records_new_objective_selection_and_shuffle_contract(self):
        model = training.EnergyRegressor(
            base_channels=4,
            pooled_size=1,
            head_features=32,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        projection = ProjectionConfig(
            normalize_energy=False,
            input_scale=1.0,
            representation="binary_occupancy",
        )
        baselines = {
            "constant_train_mean": {"validation": {}},
            "geometry_linear": {"validation": {}},
        }
        payload = training.checkpoint_payload(
            model,
            optimizer,
            0,
            [],
            projection,
            {"inventory": {}, "training_groups": [], "validation_groups": []},
            "_test",
            1,
            1,
            2,
            2,
            False,
            "disabled",
            "mse",
            1.0,
            512,
            12,
            1e-6,
            {
                "patience_counter": 0,
                "reference_rmse_mev": 0.01,
                "stopped_early": False,
            },
            baselines,
            2.45,
            0.01,
            20,
        )
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(payload["objective"]["name"], "mse")
        self.assertEqual(
            payload["selection"]["metric"],
            "validation_energy_rmse_mev",
        )
        self.assertEqual(
            payload["training_config"]["data"][
                "event_shuffle_buffer_size"
            ],
            512,
        )
        self.assertEqual(payload["early_stopping"]["patience"], 12)

    def test_acceptance_requires_every_baseline_and_bias_check(self):
        baselines = {
            "constant_train_mean": {
                "validation": {
                    "energy_rmse_mev": 0.011,
                    "energy_mae_mev": 0.009,
                }
            },
            "geometry_linear": {
                "validation": {"energy_rmse_mev": 0.0105}
            },
        }
        passing = {
            "energy_rmse_mev": 0.010,
            "energy_mae_mev": 0.008,
            "energy_bias_mev": 0.0002,
            "energy_r2": 0.03,
        }
        failing = dict(passing, energy_rmse_mev=0.0107)
        self.assertTrue(training.validation_acceptance(passing, baselines)["passed"])
        self.assertFalse(
            training.validation_acceptance(failing, baselines)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
