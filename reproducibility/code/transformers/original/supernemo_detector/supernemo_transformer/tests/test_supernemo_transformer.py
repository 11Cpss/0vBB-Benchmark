from __future__ import annotations

import unittest

import numpy as np
import torch

from supernemo_transformer import (
    CoordinateMLPEncoding,
    FourierXYZEncoding,
    SuperNEMOTokenizationConfig,
    SuperNEMOTransformerClassifier,
    tokenize_tracker_event,
)


def _event(number_of_hits: int = 24) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(number_of_hits, dtype=np.float32)
    xyz = np.column_stack(
        (
            46.0 * index,
            22.0 * (index % 5),
            30.0 * (index % 3),
        )
    ).astype(np.float32)
    radius = (index % 24).astype(np.float32)
    return xyz, radius


class SuperNEMOTokenizerTests(unittest.TestCase):
    def test_entity_shape_features_and_missing_radius(self) -> None:
        xyz, radius = _event(24)
        radius[[2, 9]] = np.nan
        config = SuperNEMOTokenizationConfig(tokenization="entity")
        tokens, coverage = tokenize_tracker_event(
            xyz, radius, event_key="event-12", config=config
        )
        self.assertEqual(tokens["coords"].shape, (24, 3))
        self.assertEqual(tokens["features"].shape, (24, 4))
        self.assertEqual(coverage, 1.0)
        self.assertTrue(np.isfinite(tokens["features"]).all())
        self.assertEqual(int(np.sum(tokens["features"][:, 3] == 0.0)), 2)
        np.testing.assert_allclose(tokens["features"][:, 0], 1.0 / 24.0)
        np.testing.assert_allclose(tokens["features"][:, 1], np.log1p(24.0))
        np.testing.assert_allclose(tokens["coords"].mean(axis=0), 0.0, atol=1.0e-6)

    def test_entity_is_permutation_invariant_and_truncation_is_deterministic(self) -> None:
        xyz, radius = _event(20)
        config = SuperNEMOTokenizationConfig(tokenization="entity", max_tokens=7)
        first, first_coverage = tokenize_tracker_event(
            xyz, radius, event_key="event-7", config=config
        )
        permutation = np.random.default_rng(9).permutation(len(xyz))
        second, second_coverage = tokenize_tracker_event(
            xyz[permutation],
            radius[permutation],
            event_key="event-7",
            config=config,
        )
        for name in ("coords", "features"):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(first_coverage, 7 / 20)
        self.assertEqual(first_coverage, second_coverage)

    def test_44mm_patch_matches_entities_but_88mm_aggregates(self) -> None:
        xyz = np.asarray(
            [[0.0, 0.0, 0.0], [46.0, 0.0, 0.0], [92.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        radius = np.asarray([4.0, 8.0, 12.0], dtype=np.float32)
        patch_44, _ = tokenize_tracker_event(
            xyz,
            radius,
            event_key="grid",
            config=SuperNEMOTokenizationConfig(
                tokenization="patch", patch_size_mm=44.0
            ),
        )
        patch_88, coverage = tokenize_tracker_event(
            xyz,
            radius,
            event_key="grid",
            config=SuperNEMOTokenizationConfig(tokenization="patch"),
        )
        self.assertEqual(len(patch_44["coords"]), 3)
        self.assertEqual(len(patch_88["coords"]), 2)
        self.assertEqual(patch_88["features"].shape, (2, 8))
        self.assertEqual(coverage, 1.0)
        self.assertAlmostEqual(float(patch_88["features"][:, 0].sum()), 1.0)

    def test_patch_truncation_prefers_occupancy_and_records_coverage(self) -> None:
        xyz = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [200.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        radius = np.ones(4, dtype=np.float32)
        tokens, coverage = tokenize_tracker_event(
            xyz,
            radius,
            event_key="patch-truncation",
            config=SuperNEMOTokenizationConfig(tokenization="patch", max_tokens=2),
        )
        self.assertEqual(len(tokens["coords"]), 2)
        self.assertEqual(coverage, 0.75)

    def test_summary_has_sixteen_balanced_tokens_and_full_coverage(self) -> None:
        xyz, radius = _event(24)
        config = SuperNEMOTokenizationConfig(tokenization="summary")
        first, coverage = tokenize_tracker_event(
            xyz, radius, event_key="summary", config=config
        )
        permutation = np.random.default_rng(5).permutation(len(xyz))
        second, _ = tokenize_tracker_event(
            xyz[permutation],
            radius[permutation],
            event_key="summary",
            config=config,
        )
        self.assertEqual(first["coords"].shape, (16, 3))
        self.assertEqual(first["features"].shape, (16, 8))
        self.assertEqual(coverage, 1.0)
        np.testing.assert_allclose(first["features"][:, 0].sum(), 1.0)
        group_sizes = np.rint(first["features"][:, 0] * len(xyz)).astype(int)
        self.assertEqual(set(group_sizes.tolist()), {1, 2})
        self.assertEqual(int(group_sizes.sum()), len(xyz))
        np.testing.assert_array_equal(first["coords"], second["coords"])
        np.testing.assert_array_equal(first["features"], second["features"])

    def test_invalid_events_are_rejected(self) -> None:
        config = SuperNEMOTokenizationConfig(tokenization="entity")
        with self.assertRaises(ValueError):
            tokenize_tracker_event(
                np.empty((0, 3)), np.empty((0,)), event_key="empty", config=config
            )
        with self.assertRaises(ValueError):
            tokenize_tracker_event(
                np.zeros((2, 3)),
                np.asarray([1.0, -1.0]),
                event_key="negative",
                config=config,
            )


class SuperNEMOModelTests(unittest.TestCase):
    def test_position_encoders(self) -> None:
        coordinates = torch.randn(2, 17, 3)
        for encoder in (
            CoordinateMLPEncoding(d_model=8),
            FourierXYZEncoding(d_model=8, num_frequencies=2),
        ):
            output = encoder(coordinates)
            self.assertEqual(tuple(output.shape), (2, 17, 8))
            self.assertTrue(bool(torch.isfinite(output).all().item()))
            output.sum().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in encoder.parameters()))

    def test_position_encoders_reject_invalid_coordinates(self) -> None:
        encoder = FourierXYZEncoding(d_model=8, num_frequencies=2)
        with self.assertRaises(ValueError):
            encoder(torch.zeros(2, 3))
        with self.assertRaises(TypeError):
            encoder(torch.zeros(1, 2, 3, dtype=torch.int64))
        invalid = torch.zeros(1, 2, 3)
        invalid[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            encoder(invalid)

    def test_all_six_model_combinations(self) -> None:
        for feature_dim in (4, 8):
            inputs = {
                "coords": torch.randn(2, 19, 3),
                "features": torch.randn(2, 19, feature_dim),
                "mask": torch.ones(2, 19, dtype=torch.bool),
            }
            inputs["mask"][1, 15:] = False
            for encoding in ("coordinate_mlp", "fourier_xyz"):
                model = SuperNEMOTransformerClassifier(
                    position_encoding=encoding,
                    feature_dim=feature_dim,
                    d_model=8,
                    nhead=2,
                    num_layers=1,
                    dim_feedforward=16,
                    dropout=0.0,
                    num_frequencies=2,
                )
                output = model(inputs)
                self.assertEqual(tuple(output.shape), (2,))
                self.assertTrue(bool(torch.isfinite(output).all().item()))
                output.sum().backward()

    def test_model_rejects_empty_event_mask(self) -> None:
        model = SuperNEMOTransformerClassifier(
            position_encoding="coordinate_mlp", feature_dim=4
        )
        with self.assertRaises(ValueError):
            model(
                {
                    "coords": torch.zeros(1, 2, 3),
                    "features": torch.zeros(1, 2, 4),
                    "mask": torch.zeros(1, 2, dtype=torch.bool),
                }
            )


if __name__ == "__main__":
    unittest.main()
