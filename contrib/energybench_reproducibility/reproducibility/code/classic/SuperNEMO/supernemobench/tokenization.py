"""Topology-only tracker-hit tokenization for SuperNEMO Transformers.

The raw SuperNEMO tracker table provides one XYZ position and one tracker
radius (``tR``) per hit.  Calorimeter energies ``E1`` and ``E2`` are event
constants, not per-hit deposits, so they deliberately do not enter these
representations.

This module returns variable-length, unpadded token arrays.  Batch padding and
mask construction remain the responsibility of the shared data collator.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


TokenizationName = Literal[
    "sampled_hits",
    "voxel",
    "summary_features",
]


@dataclass(frozen=True)
class SuperNEMOTrackerTokenizationConfig:
    """Configuration for one SuperNEMO tracker-token representation."""

    tokenization: TokenizationName
    max_tokens: int | None = None
    coordinate_scale_mm: float = 1000.0
    center_coordinates: bool = True
    voxel_size_mm: float = 60.0
    summary_morton_bits: int = 10
    tracker_radius_scale_mm: float = 24.0
    seed: int = 42
    sampling_key: Literal["local_event_number"] = "local_event_number"

    def __post_init__(self) -> None:
        if self.tokenization not in {
            "sampled_hits",
            "voxel",
            "summary_features",
        }:
            raise ValueError(
                "tokenization must be 'sampled_hits', 'voxel', "
                "or 'summary_features'"
            )

        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = 16 if self.tokenization == "summary_features" else 128
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        object.__setattr__(self, "max_tokens", max_tokens)

        if not isinstance(self.center_coordinates, bool):
            raise TypeError("center_coordinates must be Boolean")

        for name in (
            "coordinate_scale_mm",
            "voxel_size_mm",
            "tracker_radius_scale_mm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        if (
            isinstance(self.summary_morton_bits, bool)
            or not isinstance(self.summary_morton_bits, int)
            or not 1 <= self.summary_morton_bits <= 21
        ):
            raise ValueError("summary_morton_bits must be an integer in [1, 21]")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.sampling_key != "local_event_number":
            raise ValueError("sampling_key must be 'local_event_number'")

    @property
    def feature_dim(self) -> int:
        """Return the content-feature dimension for the selected tokenizer."""

        return 6 if self.tokenization == "summary_features" else 4

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation record."""

        return asdict(self)


def _stable_event_seed(sampling_key: str, base_seed: int) -> int:
    text = f"{base_seed}::{sampling_key}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little")


def _validate_event(
    coordinates: np.ndarray,
    tracker_radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.iscomplexobj(coordinates):
        raise TypeError("coordinates must be real-valued")
    if np.iscomplexobj(tracker_radius):
        raise TypeError("tracker_radius must be real-valued")

    coords = np.asarray(coordinates, dtype=np.float32).copy()
    radius = np.asarray(tracker_radius, dtype=np.float32).copy()

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape [number_of_hits, 3]")
    if radius.ndim != 1 or len(radius) != len(coords):
        raise ValueError("tracker_radius must have shape [number_of_hits]")
    if len(coords) == 0:
        raise ValueError("an event must contain at least one tracker hit")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates must contain only finite values")
    if np.isinf(radius).any():
        raise ValueError("tracker_radius may contain NaN but not infinity")

    radius_valid = ~np.isnan(radius)
    if np.any(radius[radius_valid] < 0.0):
        raise ValueError("finite tracker_radius values must be non-negative")

    radius_filled = np.where(radius_valid, radius, np.float32(0.0)).astype(
        np.float32,
        copy=False,
    )
    return coords, radius_filled, radius_valid


def _canonical_hit_order(
    coordinates: np.ndarray,
    radius_filled: np.ndarray,
    radius_valid: np.ndarray,
) -> np.ndarray:
    """Return a label-neutral, permutation-stable order for tracker hits."""

    return np.lexsort(
        (
            radius_filled,
            radius_valid.astype(np.uint8),
            coordinates[:, 2],
            coordinates[:, 1],
            coordinates[:, 0],
        )
    )


def _morton_spatial_order(coordinates: np.ndarray, bits: int) -> np.ndarray:
    """Return a deterministic locality-preserving order for 3-D hits."""

    coordinate_minimum = coordinates.min(axis=0, keepdims=True)
    coordinate_extent = np.ptp(coordinates, axis=0, keepdims=True)
    spatial_scale = float(coordinate_extent.max())

    normalized = np.zeros_like(coordinates, dtype=np.float32)
    if spatial_scale > 0.0:
        normalized = (coordinates - coordinate_minimum) / np.float32(spatial_scale)

    maximum_integer = (1 << bits) - 1
    quantized = np.floor(
        np.clip(normalized, 0.0, 1.0) * maximum_integer
    ).astype(np.uint64)

    morton_codes = np.zeros(len(coordinates), dtype=np.uint64)
    for bit in range(bits):
        for dimension in range(3):
            source_bit = (
                quantized[:, dimension] >> np.uint64(bit)
            ) & np.uint64(1)
            destination_bit = np.uint64(3 * bit + dimension)
            morton_codes |= source_bit << destination_bit

    return np.argsort(morton_codes, kind="stable")


def _group_statistics(
    coordinates: np.ndarray,
    radius_normalized: np.ndarray,
    radius_valid: np.ndarray,
    group_ids: np.ndarray,
    number_of_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(group_ids, minlength=number_of_groups).astype(np.int64)
    if np.any(counts <= 0):
        raise RuntimeError("tokenization produced an empty tracker-hit group")

    coordinate_sums = np.stack(
        [
            np.bincount(
                group_ids,
                weights=coordinates[:, dimension],
                minlength=number_of_groups,
            )
            for dimension in range(3)
        ],
        axis=1,
    )
    centroids = coordinate_sums / counts[:, None]

    valid_counts = np.bincount(
        group_ids,
        weights=radius_valid.astype(np.float32),
        minlength=number_of_groups,
    )
    radius_sums = np.bincount(
        group_ids,
        weights=radius_normalized,
        minlength=number_of_groups,
    )
    mean_valid_radius = np.divide(
        radius_sums,
        valid_counts,
        out=np.zeros(number_of_groups, dtype=np.float64),
        where=valid_counts > 0.0,
    )
    valid_fraction = valid_counts / counts

    return centroids, counts, mean_valid_radius, valid_fraction


def _sampled_hit_tokens(
    coordinates: np.ndarray,
    radius_normalized: np.ndarray,
    radius_valid: np.ndarray,
    *,
    sampling_key: str,
    config: SuperNEMOTrackerTokenizationConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    number_of_hits = len(coordinates)
    number_to_keep = min(number_of_hits, int(config.max_tokens))

    if number_to_keep < number_of_hits:
        generator = np.random.default_rng(
            _stable_event_seed(sampling_key, config.seed)
        )
        selected = generator.choice(
            number_of_hits,
            size=number_to_keep,
            replace=False,
        )
        selected.sort()
    else:
        selected = np.arange(number_of_hits, dtype=np.int64)

    inverse_hit_count = np.float32(1.0 / number_of_hits)
    log_hit_count = np.float32(np.log1p(number_of_hits))
    features = np.column_stack(
        (
            np.full(number_to_keep, inverse_hit_count, dtype=np.float32),
            np.full(number_to_keep, log_hit_count, dtype=np.float32),
            radius_normalized[selected],
            radius_valid[selected].astype(np.float32),
        )
    ).astype(np.float32, copy=False)

    coverage = float(number_to_keep / number_of_hits)
    return coordinates[selected], features, coverage


def _voxel_tokens(
    coordinates: np.ndarray,
    radius_normalized: np.ndarray,
    radius_valid: np.ndarray,
    *,
    config: SuperNEMOTrackerTokenizationConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    voxel_indices = np.floor(coordinates / float(config.voxel_size_mm)).astype(
        np.int64
    )
    _, group_ids = np.unique(voxel_indices, axis=0, return_inverse=True)
    number_of_voxels = int(group_ids.max()) + 1

    centroids, counts, mean_radius, valid_fraction = _group_statistics(
        coordinates,
        radius_normalized,
        radius_valid,
        group_ids,
        number_of_voxels,
    )

    if number_of_voxels > int(config.max_tokens):
        selected = np.argsort(-counts, kind="stable")[: int(config.max_tokens)]
        centroids = centroids[selected]
        counts = counts[selected]
        mean_radius = mean_radius[selected]
        valid_fraction = valid_fraction[selected]

    spatial_order = np.lexsort(
        (centroids[:, 2], centroids[:, 1], centroids[:, 0])
    )
    centroids = centroids[spatial_order]
    counts = counts[spatial_order]
    mean_radius = mean_radius[spatial_order]
    valid_fraction = valid_fraction[spatial_order]

    number_of_hits = len(coordinates)
    features = np.column_stack(
        (
            counts / number_of_hits,
            np.log1p(counts),
            mean_radius,
            valid_fraction,
        )
    ).astype(np.float32, copy=False)
    coverage = float(counts.sum(dtype=np.int64) / number_of_hits)
    return centroids, features, coverage


def _summary_feature_tokens(
    coordinates: np.ndarray,
    radius_normalized: np.ndarray,
    radius_valid: np.ndarray,
    *,
    config: SuperNEMOTrackerTokenizationConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    number_of_hits = len(coordinates)
    number_of_tokens = min(number_of_hits, int(config.max_tokens))
    spatial_order = _morton_spatial_order(coordinates, config.summary_morton_bits)

    ordered_group_ids = (
        np.arange(number_of_hits, dtype=np.int64)
        * number_of_tokens
        // number_of_hits
    ).astype(np.int64)
    group_ids = np.empty(number_of_hits, dtype=np.int64)
    group_ids[spatial_order] = ordered_group_ids

    centroids, counts, mean_radius, valid_fraction = _group_statistics(
        coordinates,
        radius_normalized,
        radius_valid,
        group_ids,
        number_of_tokens,
    )
    offsets = coordinates - centroids[group_ids]
    squared_distances = np.sum(offsets * offsets, axis=1)
    mean_squared_distance = np.bincount(
        group_ids,
        weights=squared_distances,
        minlength=number_of_tokens,
    ) / counts
    spatial_rms = np.sqrt(np.maximum(mean_squared_distance, 0.0))

    inverse_hit_count = np.float32(1.0 / number_of_hits)
    features = np.column_stack(
        (
            counts / number_of_hits,
            np.log1p(counts),
            np.full(number_of_tokens, inverse_hit_count, dtype=np.float32),
            spatial_rms / float(config.coordinate_scale_mm),
            mean_radius,
            valid_fraction,
        )
    ).astype(np.float32, copy=False)
    return centroids, features, 1.0


def tokenize_tracker_event(
    coordinates: np.ndarray,
    tracker_radius: np.ndarray,
    *,
    sampling_key: str,
    config: SuperNEMOTrackerTokenizationConfig,
) -> tuple[dict[str, np.ndarray], float]:
    """Convert one complete tracker event into unpadded Transformer tokens.

    Returns ``({"coords": ..., "features": ...}, coverage)``. ``coverage``
    is the fraction of original tracker hits represented by the retained
    tokens; summary-feature tokens always retain every hit.
    """

    if not isinstance(config, SuperNEMOTrackerTokenizationConfig):
        raise TypeError("config must be a SuperNEMOTrackerTokenizationConfig")
    if not isinstance(sampling_key, str) or not sampling_key:
        raise ValueError("sampling_key must be a non-empty string")

    coords, radius_filled, radius_valid = _validate_event(
        coordinates,
        tracker_radius,
    )
    # Raw HDF5 row order is not a physical feature. Canonicalizing before both
    # centering and sampling makes every tokenizer invariant to a permutation
    # of otherwise identical tracker hits.
    hit_order = _canonical_hit_order(coords, radius_filled, radius_valid)
    coords = coords[hit_order]
    radius_filled = radius_filled[hit_order]
    radius_valid = radius_valid[hit_order]
    if config.center_coordinates:
        center = coords.mean(axis=0, keepdims=True, dtype=np.float64)
        coords = (coords.astype(np.float64) - center).astype(np.float32)

    radius_normalized = (
        radius_filled / np.float32(config.tracker_radius_scale_mm)
    ).astype(np.float32, copy=False)

    if config.tokenization == "sampled_hits":
        token_coords, features, coverage = _sampled_hit_tokens(
            coords,
            radius_normalized,
            radius_valid,
            sampling_key=sampling_key,
            config=config,
        )
    elif config.tokenization == "voxel":
        token_coords, features, coverage = _voxel_tokens(
            coords,
            radius_normalized,
            radius_valid,
            config=config,
        )
    elif config.tokenization == "summary_features":
        token_coords, features, coverage = _summary_feature_tokens(
            coords,
            radius_normalized,
            radius_valid,
            config=config,
        )
    else:  # pragma: no cover - guarded by the frozen config validation.
        raise RuntimeError(f"unsupported tokenization: {config.tokenization}")

    scaled_coords = (
        np.asarray(token_coords, dtype=np.float64)
        / float(config.coordinate_scale_mm)
    ).astype(np.float32)
    token_features = np.asarray(features, dtype=np.float32)

    if scaled_coords.ndim != 2 or scaled_coords.shape[1] != 3:
        raise RuntimeError("tokenizer produced invalid coordinate shape")
    if token_features.shape != (len(scaled_coords), config.feature_dim):
        raise RuntimeError("tokenizer produced invalid feature shape")
    if len(scaled_coords) == 0:
        raise RuntimeError("tokenizer produced an empty event")
    if not np.isfinite(scaled_coords).all() or not np.isfinite(token_features).all():
        raise RuntimeError("tokenizer produced non-finite output")
    if not math.isfinite(coverage) or not 0.0 < coverage <= 1.0:
        raise RuntimeError("tokenizer produced invalid representation coverage")

    return {
        "coords": np.ascontiguousarray(scaled_coords),
        "features": np.ascontiguousarray(token_features),
    }, float(coverage)


__all__ = [
    "SuperNEMOTrackerTokenizationConfig",
    "TokenizationName",
    "tokenize_tracker_event",
]
