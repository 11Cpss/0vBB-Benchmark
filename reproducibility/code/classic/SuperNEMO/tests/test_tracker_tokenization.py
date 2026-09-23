"""Regression tests for the three SuperNEMO tracker tokenizations."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from supernemobench.config import (
    ARCHITECTURES,
    TRANSFORMER_ARCHITECTURE_IDS,
    DataConfig,
)
from supernemobench.data import SuperNEMOEventDataset, collate_events
from supernemobench.models import build_model
from supernemobench.tokenization import (
    SuperNEMOTrackerTokenizationConfig,
    tokenize_tracker_event,
)


TOKENIZER_SCHEMAS = (
    ("sampled_hits", 128, 4),
    ("voxel", 128, 4),
    ("summary_features", 16, 6),
)


@pytest.mark.parametrize(
    ("tokenization", "default_max_tokens", "feature_dim"),
    TOKENIZER_SCHEMAS,
)
def test_tokenizer_defaults_shapes_and_finite_contiguous_outputs(
    tokenization: str,
    default_max_tokens: int,
    feature_dim: int,
) -> None:
    config = SuperNEMOTrackerTokenizationConfig(tokenization=tokenization)
    assert config.max_tokens == default_max_tokens
    assert config.feature_dim == feature_dim
    assert config.sampling_key == "local_event_number"

    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [44.0, 0.0, 10.0],
            [88.0, 44.0, 20.0],
            [132.0, 44.0, 30.0],
        ],
        dtype=np.float32,
    )
    tracker_radius = np.asarray([12.0, np.nan, 24.0, 6.0], dtype=np.float32)
    tokens, coverage = tokenize_tracker_event(
        coordinates,
        tracker_radius,
        sampling_key="schema-fixture",
        config=config,
    )

    assert set(tokens) == {"coords", "features"}
    assert tokens["coords"].shape[1:] == (3,)
    assert tokens["features"].shape == (len(tokens["coords"]), feature_dim)
    assert tokens["coords"].dtype == np.dtype(np.float32)
    assert tokens["features"].dtype == np.dtype(np.float32)
    assert tokens["coords"].flags.c_contiguous
    assert tokens["features"].flags.c_contiguous
    assert np.isfinite(tokens["coords"]).all()
    assert np.isfinite(tokens["features"]).all()
    assert 0.0 < coverage <= 1.0


def test_sampled_hits_is_event_deterministic_and_preserves_radius_mask() -> None:
    number_of_hits = 140
    index = np.arange(number_of_hits, dtype=np.float32)
    coordinates = np.column_stack((3.0 * index, index % 11.0, index % 7.0))
    tracker_radius = (index % 24.0).astype(np.float32)
    tracker_radius[::2] = np.nan
    config = SuperNEMOTrackerTokenizationConfig(tokenization="sampled_hits")

    first, first_coverage = tokenize_tracker_event(
        coordinates,
        tracker_radius,
        sampling_key="event-a",
        config=config,
    )
    repeated, repeated_coverage = tokenize_tracker_event(
        coordinates,
        tracker_radius,
        sampling_key="event-a",
        config=config,
    )
    permutation = np.random.default_rng(20260829).permutation(number_of_hits)
    permuted, permuted_coverage = tokenize_tracker_event(
        coordinates[permutation],
        tracker_radius[permutation],
        sampling_key="event-a",
        config=config,
    )
    other_event, _ = tokenize_tracker_event(
        coordinates,
        tracker_radius,
        sampling_key="event-b",
        config=config,
    )

    np.testing.assert_array_equal(first["coords"], repeated["coords"])
    np.testing.assert_array_equal(first["features"], repeated["features"])
    np.testing.assert_array_equal(first["coords"], permuted["coords"])
    np.testing.assert_array_equal(first["features"], permuted["features"])
    assert first_coverage == repeated_coverage == pytest.approx(128.0 / 140.0)
    assert permuted_coverage == first_coverage
    assert not np.array_equal(first["coords"], other_event["coords"])
    np.testing.assert_allclose(first["features"][:, 0], 1.0 / number_of_hits)
    np.testing.assert_allclose(first["features"][:, 1], np.log1p(number_of_hits))
    assert set(np.unique(first["features"][:, 3])).issubset({0.0, 1.0})
    assert np.any(first["features"][:, 3] == 0.0)
    assert np.all(first["features"][first["features"][:, 3] == 0.0, 2] == 0.0)

    small_coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [20.0, 40.0, 60.0]],
        dtype=np.float32,
    )
    small_radius = np.asarray([12.0, np.nan, 24.0], dtype=np.float32)
    small, coverage = tokenize_tracker_event(
        small_coordinates,
        small_radius,
        sampling_key="untruncated",
        config=config,
    )
    expected_coords = (
        small_coordinates - small_coordinates.mean(axis=0, keepdims=True)
    ) / 1_000.0
    expected_features = np.column_stack(
        (
            np.full(3, 1.0 / 3.0),
            np.full(3, np.log1p(3)),
            np.asarray([0.5, 0.0, 1.0]),
            np.asarray([1.0, 0.0, 1.0]),
        )
    ).astype(np.float32)
    np.testing.assert_allclose(small["coords"], expected_coords, atol=1.0e-7)
    np.testing.assert_allclose(small["features"], expected_features, atol=1.0e-7)
    assert coverage == 1.0


def test_dataset_sampling_key_is_independent_of_class_source(tmp_path) -> None:
    tokenization = SuperNEMOTrackerTokenizationConfig(tokenization="sampled_hits")
    dataset = SuperNEMOEventDataset(
        data_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest={
            "splits": {
                "train": [
                    {
                        "source_key": "2nubb",
                        "split": "train",
                        "event_start": 0,
                        "event_stop": 1,
                    }
                ]
            }
        },
        task="classification",
        split="train",
        input_kind="sequence",
        config=DataConfig(
            data_root=tmp_path,
            manifest_path=tmp_path / "manifest.json",
        ),
        tokenization_config=tokenization,
        shuffle_slices=False,
    )
    index = np.arange(140, dtype=np.float32)
    coordinates = np.column_stack((13.0 * index, index % 17.0, index % 9.0))
    tracker_radius = (index % 24.0).astype(np.float32)
    tracker_radius[::5] = np.nan

    signal = dataset._sample(
        "2nubb", 37, coordinates, tracker_radius, 1_500.0
    )
    background = dataset._sample(
        "Bi214", 37, coordinates, tracker_radius, 1_500.0
    )

    np.testing.assert_array_equal(
        signal["inputs"]["coords"], background["inputs"]["coords"]
    )
    np.testing.assert_array_equal(
        signal["inputs"]["features"], background["inputs"]["features"]
    )
    assert signal["token_coverage"] == background["token_coverage"]
    assert signal["event_id"] != background["event_id"]
    assert signal["category"] == "2nu"
    assert background["category"] == "Bi214"
    assert signal["target"] == 1.0
    assert background["target"] == 0.0


def test_voxel_uses_hit_centroids_and_valid_radius_aggregation() -> None:
    coordinates = np.asarray(
        [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [70.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    tracker_radius = np.asarray([12.0, np.nan, 24.0], dtype=np.float32)
    config = SuperNEMOTrackerTokenizationConfig(
        tokenization="voxel",
        center_coordinates=False,
        voxel_size_mm=60.0,
    )
    tokens, coverage = tokenize_tracker_event(
        coordinates,
        tracker_radius,
        sampling_key="voxel-centroid",
        config=config,
    )

    expected_coords = np.asarray([[3.0, 4.0, 5.0], [70.0, 0.0, 0.0]]) / 1_000.0
    expected_features = np.asarray(
        [
            [2.0 / 3.0, np.log1p(2), 0.5, 0.5],
            [1.0 / 3.0, np.log1p(1), 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tokens["coords"], expected_coords, atol=1.0e-7)
    np.testing.assert_allclose(tokens["features"], expected_features, atol=1.0e-7)
    assert coverage == 1.0

    truncated_config = SuperNEMOTrackerTokenizationConfig(
        tokenization="voxel",
        max_tokens=1,
        center_coordinates=False,
        voxel_size_mm=60.0,
    )
    truncated, truncated_coverage = tokenize_tracker_event(
        coordinates,
        tracker_radius,
        sampling_key="voxel-truncation",
        config=truncated_config,
    )
    np.testing.assert_allclose(truncated["coords"], expected_coords[:1], atol=1.0e-7)
    np.testing.assert_allclose(
        truncated["features"], expected_features[:1], atol=1.0e-7
    )
    assert truncated_coverage == pytest.approx(2.0 / 3.0)


def test_summary_features_use_morton_groups_and_retain_all_hits() -> None:
    number_of_hits = 33
    x = np.arange(number_of_hits, dtype=np.float32) * 10.0
    coordinates = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    tracker_radius = (1.0 + x % 23.0).astype(np.float32)
    config = SuperNEMOTrackerTokenizationConfig(tokenization="summary_features")

    tokens, coverage = tokenize_tracker_event(
        coordinates,
        tracker_radius,
        sampling_key="summary-default",
        config=config,
    )
    assert tokens["coords"].shape == (16, 3)
    assert tokens["features"].shape == (16, 6)
    represented_counts = tokens["features"][:, 0] * number_of_hits
    np.testing.assert_allclose(represented_counts, np.round(represented_counts))
    assert int(represented_counts.max() - represented_counts.min()) <= 1
    assert tokens["features"][:, 0].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        tokens["features"][:, 1], np.log1p(represented_counts), atol=1.0e-6
    )
    np.testing.assert_allclose(tokens["features"][:, 2], 1.0 / number_of_hits)
    assert np.all(tokens["features"][:, 3] >= 0.0)
    assert coverage == 1.0

    permutation = np.random.default_rng(20260829).permutation(number_of_hits)
    permuted, permuted_coverage = tokenize_tracker_event(
        coordinates[permutation],
        tracker_radius[permutation],
        sampling_key="summary-permuted",
        config=config,
    )
    np.testing.assert_allclose(permuted["coords"], tokens["coords"], atol=1.0e-7)
    np.testing.assert_allclose(permuted["features"], tokens["features"], atol=1.0e-7)
    assert permuted_coverage == coverage

    manual_config = SuperNEMOTrackerTokenizationConfig(
        tokenization="summary_features",
        max_tokens=2,
        center_coordinates=False,
    )
    manual_coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [100.0, 0.0, 0.0], [110.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    manual_radius = np.asarray([12.0, np.nan, 24.0, 12.0], dtype=np.float32)
    manual, manual_coverage = tokenize_tracker_event(
        manual_coordinates,
        manual_radius,
        sampling_key="summary-manual",
        config=manual_config,
    )
    np.testing.assert_allclose(
        manual["coords"],
        np.asarray([[5.0, 0.0, 0.0], [105.0, 0.0, 0.0]]) / 1_000.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(manual["features"][:, 0], [0.5, 0.5])
    np.testing.assert_allclose(manual["features"][:, 1], np.log1p([2.0, 2.0]))
    np.testing.assert_allclose(manual["features"][:, 2], [0.25, 0.25])
    np.testing.assert_allclose(manual["features"][:, 3], [0.005, 0.005])
    np.testing.assert_allclose(manual["features"][:, 4], [0.5, 0.75])
    np.testing.assert_allclose(manual["features"][:, 5], [0.5, 1.0])
    assert manual_coverage == 1.0


@pytest.mark.parametrize(
    ("tracker_radius", "error"),
    (
        (np.asarray([1.0, np.inf], dtype=np.float32), "not infinity"),
        (np.asarray([1.0, -0.1], dtype=np.float32), "non-negative"),
    ),
)
def test_tracker_radius_rejects_invalid_finite_semantics(
    tracker_radius: np.ndarray,
    error: str,
) -> None:
    coordinates = np.zeros((2, 3), dtype=np.float32)
    config = SuperNEMOTrackerTokenizationConfig(tokenization="sampled_hits")
    with pytest.raises(ValueError, match=error):
        tokenize_tracker_event(
            coordinates,
            tracker_radius,
            sampling_key="invalid-radius",
            config=config,
        )


@pytest.mark.parametrize("architecture_id", TRANSFORMER_ARCHITECTURE_IDS)
def test_tokenized_classification_batch_forward_and_backward(
    architecture_id: str,
) -> None:
    architecture = ARCHITECTURES[architecture_id]
    assert architecture.tokenization is not None
    samples = []
    token_counts = []
    for event_index, (number_of_hits, target, category) in enumerate(
        ((19, 0.0, "Bi214"), (25, 1.0, "2nu"))
    ):
        index = np.arange(number_of_hits, dtype=np.float32)
        coordinates = np.column_stack(
            (
                44.0 * (index % 9.0),
                44.0 * ((3.0 * index) % 11.0),
                17.0 * index,
            )
        ).astype(np.float32)
        tracker_radius = (2.0 + index % 20.0).astype(np.float32)
        tracker_radius[event_index :: 7] = np.nan
        event_id = f"SuperNEMO::{category}::{event_index}"
        inputs, coverage = tokenize_tracker_event(
            coordinates,
            tracker_radius,
            sampling_key=f"SuperNEMO::ev_no::{event_index}",
            config=architecture.tokenization,
        )
        token_counts.append(len(inputs["coords"]))
        samples.append(
            {
                "inputs": inputs,
                "target": np.float32(target),
                "energy": np.float64(1_000.0 + 100.0 * event_index),
                "event_id": event_id,
                "group_id": event_id,
                "category": category,
                "split": "train",
                "token_coverage": np.float32(coverage),
            }
        )

    batch = collate_events(samples)
    feature_dim = int(architecture.model["feature_dim"])
    assert batch["inputs"]["coords"].shape == (2, max(token_counts), 3)
    assert batch["inputs"]["features"].shape == (
        2,
        max(token_counts),
        feature_dim,
    )
    assert batch["inputs"]["mask"].dtype == torch.bool
    assert batch["inputs"]["mask"].sum(dim=1).tolist() == token_counts
    assert batch["token_coverage"].shape == (2,)
    assert torch.all((batch["token_coverage"] > 0.0) & (batch["token_coverage"] <= 1.0))
    assert batch["category"] == ["Bi214", "2nu"]
    torch.testing.assert_close(batch["target"], torch.tensor([0.0, 1.0]))

    model = build_model(architecture_id, "classification")
    model.train()
    logits = model(batch["inputs"])
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    loss = F.binary_cross_entropy_with_logits(logits, batch["target"])
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
