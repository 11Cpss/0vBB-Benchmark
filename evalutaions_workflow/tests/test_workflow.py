"""Focused synthetic tests for the standalone workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch

from simple_energybench import (
    EvaluationConfig,
    ProjectionConfig,
    SimpleClassifier,
    SimpleRegressor,
    TrainingConfig,
    evaluate_classification,
    evaluate_regression,
    prepare_dataset,
    set_seed,
    train_model,
)
from simple_energybench.metrics import (
    evaluate_regression_metrics,
    make_fixed_energy_bins,
    weighted_roc_curve,
)
from simple_energybench.plotting import plot_score_energy_dependence


def _write_hdf5(path: Path, event_counts: int, label: int, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_per_event = 3
    dtype = np.dtype(
        [
            ("index", "<i8"),
            ("values_block_0", "<i8", (1,)),
            ("values_block_1", "<f4", (4,)),
            ("values_block_2", "S6", (1,)),
        ]
    )
    rows = np.zeros(event_counts * rows_per_event, dtype=dtype)
    cursor = 0
    raw_label = b"Signal" if label == 1 else b"Bkg"
    for event_index in range(event_counts):
        event_id = offset + event_index * 2
        energy = np.float32(0.066 + 0.0008 * event_index + 0.0002 * label)
        for hit_index in range(rows_per_event):
            rows[cursor]["index"] = cursor
            rows[cursor]["values_block_0"][0] = event_id
            rows[cursor]["values_block_1"] = (
                5.0 + hit_index,
                5.0 + event_index % 3,
                5.0 + label,
                energy,
            )
            rows[cursor]["values_block_2"][0] = raw_label
            cursor += 1
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("MC/hits/table", data=rows)
        dataset.attrs["values_block_0_kind"] = np.bytes_("(lp0\nVevent_id\np1\na.")
        dataset.attrs["values_block_1_kind"] = np.bytes_(
            "(lp0\nVx\np1\naVy\np2\naVz\np3\naVenergy\np4\na."
        )
        dataset.attrs["values_block_2_kind"] = np.bytes_("(lp0\nVlabel\np1\na.")


def _make_dataset(root: Path) -> None:
    for index, count in enumerate((3, 7, 11)):
        _write_hdf5(
            root / "0nubb_part_1" / f"signal_{index}.h5",
            count,
            1,
            10_000 + index * 1_000,
        )
    for index, count in enumerate((5, 9, 13)):
        _write_hdf5(
            root / "Bi_part_1" / f"background_{index}.h5",
            count,
            0,
            20_000 + index * 1_000,
        )


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        _make_dataset(self.data_root)
        self.projection = ProjectionConfig(
            grid_size=8,
            bin_size=10.0,
            origin=(0.0, 0.0, 0.0),
            input_scale=10.0,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_standard_config_and_seed(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.epochs, 50)
        self.assertAlmostEqual(config.learning_rate, 0.0005)
        set_seed(7)
        first = (np.random.random(), torch.rand(1).item())
        set_seed(7)
        second = (np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_fixed_energy_bins_and_tied_auc(self) -> None:
        edges = make_fixed_energy_bins(
            np.asarray([2.431, 2.436, 2.441]), width_kev=5.0, energy_unit="MeV"
        )
        np.testing.assert_allclose(
            edges,
            np.asarray([2.430, 2.435, 2.440, 2.445]),
            atol=1e-12,
            rtol=0,
        )
        self.assertTrue(np.allclose(np.diff(edges), 0.005, atol=1e-12, rtol=0))
        self.assertGreaterEqual(edges[-1], 2.441)
        constant = make_fixed_energy_bins(np.asarray([2.45, 2.45]))
        self.assertEqual(len(constant), 2)
        self.assertAlmostEqual(constant[0], 2.45)
        self.assertAlmostEqual(constant[1] - constant[0], 0.005)
        roc = weighted_roc_curve(
            np.asarray([0, 1, 0, 1]),
            np.asarray([0.1, 0.9, 0.5, 0.5]),
            np.ones(4),
            target_tpr=0.9,
        )
        self.assertAlmostEqual(roc["auc"], 0.875)

    def test_canonical_energy_grid_boundaries_and_validation(self) -> None:
        full = make_fixed_energy_bins(np.asarray([0.0, 3.0]))
        self.assertEqual(len(full) - 1, 600)
        self.assertEqual(full[0], 0.0)
        self.assertEqual(full[-1], 3.0)
        self.assertTrue(
            np.allclose(np.diff(full), 0.005, atol=1e-12, rtol=0)
        )

        # Internal edges belong to the bin beginning at that edge, exactly as
        # they do on the unsliced global grid.
        exact_internal = make_fixed_energy_bins(np.asarray([2.435, 2.440]))
        np.testing.assert_allclose(
            exact_internal,
            np.asarray([2.435, 2.440, 2.445]),
            atol=1e-12,
            rtol=0,
        )
        final_endpoint = make_fixed_energy_bins(np.asarray([3.0, 3.0]))
        np.testing.assert_allclose(
            final_endpoint,
            np.asarray([2.995, 3.0]),
            atol=1e-12,
            rtol=0,
        )

        with self.assertRaisesRegex(ValueError, r"\[0, 3000\] keV"):
            make_fixed_energy_bins(np.asarray([-1.0e-9, 1.0]))
        with self.assertRaisesRegex(ValueError, r"\[0, 3000\] keV"):
            make_fixed_energy_bins(np.asarray([1.0, 3.0 + 1.0e-12]))
        with self.assertRaisesRegex(ValueError, "fixed at 5 keV"):
            make_fixed_energy_bins(np.asarray([1.0, 2.0]), width_kev=10.0)
        with self.assertRaisesRegex(ValueError, "fixed at 5 keV"):
            EvaluationConfig(energy_bin_width_kev=10.0)
        with self.assertRaisesRegex(ValueError, "fixed at 600"):
            EvaluationConfig(energy_grid_bin_count=601)
        config_payload = EvaluationConfig().to_dict()
        self.assertEqual(config_payload["energy_grid_min_kev"], 0.0)
        self.assertEqual(config_payload["energy_grid_max_kev"], 3000.0)
        self.assertEqual(config_payload["energy_grid_bin_count"], 600)

    def _prepared(self, mode: str = "classification"):
        return prepare_dataset(
            self.data_root,
            batch_size=4,
            projection=self.projection,
            mode=mode,
            seed=42,
            num_workers=0,
            manifest_path=self.root / "split.json",
            verbose=False,
        )

    def test_event_split_is_complete_reproducible_and_slice_based(self) -> None:
        prepared = self._prepared()
        counts = prepared.counts
        self.assertEqual(counts["total"], 48)
        self.assertEqual(
            counts["train"] + counts["validation"] + counts["test"], 48
        )
        manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
        allocated = sum(
            int(item["event_stop"]) - int(item["event_start"])
            for rows in manifest["splits"].values()
            for item in rows
        )
        self.assertEqual(allocated, 48)
        ranges_by_path: dict[str, list[str]] = {}
        for split_name, rows in manifest["splits"].items():
            for row in rows:
                ranges_by_path.setdefault(row["relative_path"], []).append(split_name)
        self.assertTrue(any(len(names) > 1 for names in ranges_by_path.values()))
        prepared_again = self._prepared()
        self.assertEqual(prepared.counts, prepared_again.counts)
        self.assertEqual(
            manifest,
            json.loads(prepared_again.manifest_path.read_text(encoding="utf-8")),
        )
        batch = next(iter(prepared.train_loader))
        self.assertEqual(tuple(batch["inputs"].shape[1:]), (3, 8, 8))
        self.assertEqual(batch["label"].ndim, 1)
        self.assertEqual(batch["energy"].ndim, 1)

        event_ids: list[str] = []
        for loader in (
            prepared.train_loader,
            prepared.validation_loader,
            prepared.test_loader,
        ):
            for split_batch in loader:
                event_ids.extend(split_batch["event_id"])
        self.assertEqual(len(event_ids), 48)
        self.assertEqual(len(set(event_ids)), 48)

        old_digest = manifest["inventory"]["digest"]
        _write_hdf5(
            self.data_root / "0nubb_part_1" / "new_signal.h5",
            2,
            1,
            99_000,
        )
        refreshed = self._prepared()
        refreshed_manifest = json.loads(
            refreshed.manifest_path.read_text(encoding="utf-8")
        )
        self.assertNotEqual(old_digest, refreshed_manifest["inventory"]["digest"])
        self.assertEqual(refreshed.counts["total"], 50)

    def test_evaluation_validation_and_sparse_plot(self) -> None:
        invalid_batch = {
            "inputs": torch.zeros(4, 1),
            "label": torch.tensor([0.0, 0.2, 0.8, 1.0]),
            "energy": torch.linspace(2.4, 2.5, 4),
            "event_id": [f"custom-{index}" for index in range(4)],
        }
        invalid_model = torch.nn.Identity()
        self.assertTrue(invalid_model.training)

        with self.assertRaises(TypeError):
            evaluate_classification(
                torch.nn.Identity(),
                [invalid_batch],
                "cpu",
                self.root / "invalid_config",
                {"energy_bin_wdth_kev": 5.0},
            )
        with self.assertRaisesRegex(ValueError, "numeric 0/1"):
            evaluate_classification(
                invalid_model,
                [invalid_batch],
                "cpu",
                self.root / "invalid_labels",
            )
        self.assertTrue(invalid_model.training)

        placeholder = self.root / "sparse_dependence.png"
        returned = plot_score_energy_dependence({"groups": {}}, placeholder)
        self.assertEqual(returned, placeholder)
        self.assertTrue(placeholder.is_file())

        truth = np.asarray([2.440, 2.445, 2.450, 2.455, 2.460, 2.465])
        prediction = truth + np.asarray([0.001, -0.002, 0.0, 0.003, -0.001, 0.002])
        config = EvaluationConfig(min_per_bin=1, performance_bins=3)
        metrics = evaluate_regression_metrics(truth, prediction, None, config)
        # ERS keeps equal-population truth quantiles rather than replacing its
        # event term with the canonical fixed-width diagnostic bins.
        self.assertFalse(
            np.allclose(
                np.diff(metrics["performance_bin_edges"]),
                0.005,
                atol=1e-12,
                rtol=0,
            )
        )
        self.assertEqual(len(metrics["performance_bin_edges"]), 4)
        protocol = metrics["protocol"]
        self.assertEqual(protocol["energy_grid_min_kev"], 0.0)
        self.assertEqual(protocol["energy_grid_max_kev"], 3000.0)
        self.assertEqual(protocol["energy_grid_bin_count"], 600)
        self.assertEqual(
            protocol["selected_energy_bin_count"],
            len(metrics["histogram_edges"]) - 1,
        )
        expected_ers = metrics["finite_fraction"] * np.sqrt(
            metrics["event_score"] * metrics["histogram_similarity"]
        )
        self.assertAlmostEqual(metrics["ers"], expected_ers)
        missing = evaluate_regression_metrics(
            truth, np.full_like(truth, np.nan), None, config
        )
        self.assertEqual(missing["status"], "no_finite_predictions")
        self.assertEqual(missing["ers"], 0.0)
        with self.assertRaisesRegex(ValueError, r"\[0, 3000\] keV"):
            evaluate_regression_metrics(
                np.asarray([2.5, 3.001]), np.asarray([2.5, 3.0]), None, config
            )

    def test_training_and_evaluation_artifacts(self) -> None:
        data = self._prepared()
        training_config = replace(
            TrainingConfig(),
            batch_size=4,
            epochs=1,
            early_stopping_patience=1,
            use_amp=False,
            device="cpu",
        )
        classifier = SimpleClassifier(base_channels=2)
        classification_history = train_model(
            classifier,
            data.train_loader,
            data.validation_loader,
            training_config,
            "classification",
            self.root / "classification_training",
        )
        self.assertEqual(classification_history["best_epoch"], 1)
        evaluation_config = EvaluationConfig(
            min_per_class=1,
            min_valid_bins=1,
            support_trim_quantile=0.0,
            min_coverage=0.0,
            score_bins=5,
            min_per_bin=1,
            performance_bins=3,
        )
        classification = evaluate_classification(
            classifier,
            data.test_loader,
            "cpu",
            self.root / "classification_evaluation",
            evaluation_config,
        )
        self.assertIn("auc", classification)
        classification_protocol = classification["classification"]["protocol"]
        dependence_protocol = classification["energy_dependence"]["protocol"]
        for protocol in (classification_protocol, dependence_protocol):
            self.assertEqual(protocol["energy_grid_min_kev"], 0.0)
            self.assertEqual(protocol["energy_grid_max_kev"], 3000.0)
            self.assertEqual(protocol["energy_grid_bin_count"], 600)
        self.assertEqual(
            classification_protocol["selected_energy_bin_count"],
            classification["classification"]["n_bins_actual"],
        )
        self.assertEqual(
            dependence_protocol["selected_energy_bin_count"],
            len(classification["energy_dependence"]["energy_bin_edges"]) - 1,
        )
        for name in (
            "metrics.json",
            "results.csv",
            "predictions.npz",
            "energy_matched_roc.png",
            "score_energy_dependence.png",
        ):
            self.assertTrue((self.root / "classification_evaluation" / name).is_file())

        regression_data = self._prepared("regression_energy")
        regressor = SimpleRegressor(base_channels=2)
        regression_history = train_model(
            regressor,
            regression_data.train_loader,
            regression_data.validation_loader,
            training_config,
            "regression",
            self.root / "regression_training",
        )
        self.assertEqual(regression_history["best_epoch"], 1)
        regression = evaluate_regression(
            regressor,
            regression_data.test_loader,
            "cpu",
            self.root / "regression_evaluation",
            evaluation_config,
        )
        self.assertIn("ers", regression)
        regression_protocol = regression["energy_regression"]["protocol"]
        self.assertEqual(regression_protocol["energy_grid_min_kev"], 0.0)
        self.assertEqual(regression_protocol["energy_grid_max_kev"], 3000.0)
        self.assertEqual(regression_protocol["energy_grid_bin_count"], 600)
        self.assertEqual(
            regression_protocol["selected_energy_bin_count"],
            len(regression["energy_regression"]["histogram_edges"]) - 1,
        )
        for name in (
            "metrics.json",
            "results.csv",
            "predictions.npz",
            "energy_regression.png",
            "energy_histograms.png",
        ):
            self.assertTrue((self.root / "regression_evaluation" / name).is_file())

    def test_training_accepts_nested_mapping_inputs(self) -> None:
        class NestedInputClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output = torch.nn.Linear(3, 1)
                self.saw_expected_structure = False

            def forward(self, inputs):
                if not (
                    isinstance(inputs, dict)
                    and isinstance(inputs["features"], tuple)
                    and isinstance(inputs["features"][1], list)
                    and isinstance(inputs["context"], dict)
                ):
                    raise AssertionError("nested input containers were not preserved")
                tensors = (
                    inputs["features"][0],
                    inputs["features"][1][0],
                    inputs["context"]["offset"],
                )
                if any(tensor.device != self.output.weight.device for tensor in tensors):
                    raise AssertionError("nested tensor was not moved to the model device")
                self.saw_expected_structure = True
                return self.output(torch.cat(tensors, dim=1)).squeeze(1)

        batch = {
            "inputs": {
                "features": (
                    torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
                    [torch.tensor([[1.0], [0.0], [1.0], [0.0]])],
                ),
                "context": {
                    "offset": torch.tensor([[0.2], [0.3], [0.4], [0.5]])
                },
            },
            "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "sample_weight": torch.ones(4),
        }
        config = replace(
            TrainingConfig(),
            batch_size=4,
            epochs=1,
            early_stopping_patience=1,
            use_amp=False,
            device="cpu",
        )
        model = NestedInputClassifier()
        history = train_model(
            model,
            [batch],
            [batch],
            config,
            "classification",
            self.root / "nested_input_training",
        )

        self.assertTrue(model.saw_expected_structure)
        self.assertEqual(history["history"][0]["train_events"], 4)
        self.assertEqual(history["history"][0]["val_events"], 4)


if __name__ == "__main__":
    unittest.main()
