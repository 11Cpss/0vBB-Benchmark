"""Deterministic topology-only tokenization for SuperNEMO tracker events."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


TokenizationName = Literal["entity", "patch", "summary"]


@dataclass(frozen=True)
class SuperNEMOTokenizationConfig:
    """Frozen settings for one SuperNEMO Transformer representation."""

    tokenization: TokenizationName
    max_tokens: int = 512
    summary_groups: int = 16
    patch_size_mm: float = 88.0
    patch_origin_mm: tuple[float, float, float] = (-2816.0, -2816.0, -2816.0)
    coordinate_scale_mm: float = 1000.0
    tracker_radius_scale_mm: float = 24.0
    morton_bits: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if self.tokenization not in {"entity", "patch", "summary"}:
            raise ValueError("tokenization must be 'entity', 'patch', or 'summary'")
        for name in ("max_tokens", "summary_groups", "morton_bits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.morton_bits > 21:
            raise ValueError("morton_bits must not exceed 21")
        if self.tokenization == "summary" and self.summary_groups > self.max_tokens:
            raise ValueError("summary_groups must not exceed max_tokens")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "patch_size_mm",
            "coordinate_scale_mm",
            "tracker_radius_scale_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if len(self.patch_origin_mm) != 3 or any(
            not math.isfinite(float(value)) for value in self.patch_origin_mm
        ):
            raise ValueError("patch_origin_mm must contain three finite values")

    @property
    def feature_dim(self) -> int:
        return 4 if self.tokenization == "entity" else 8

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["patch_origin_mm"] = list(self.patch_origin_mm)
        return payload


def _validate_event(
    coordinates: np.ndarray,
    tracker_radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.iscomplexobj(coordinates) or np.iscomplexobj(tracker_radius):
        raise TypeError("tracker inputs must be real-valued")
    xyz = np.asarray(coordinates, dtype=np.float32).copy()
    radius = np.asarray(tracker_radius, dtype=np.float32).copy()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape [number_of_hits, 3]")
    if radius.ndim != 1 or len(radius) != len(xyz):
        raise ValueError("tracker_radius must have shape [number_of_hits]")
    if len(xyz) == 0:
        raise ValueError("an event must contain at least one tracker hit")
    if not np.isfinite(xyz).all():
        raise ValueError("coordinates must contain only finite values")
    if np.isinf(radius).any():
        raise ValueError("tracker_radius may contain NaN but not infinity")
    valid = np.isfinite(radius)
    if np.any(radius[valid] < 0.0):
        raise ValueError("finite tracker_radius values must be non-negative")
    filled = np.where(valid, radius, np.float32(0.0)).astype(np.float32, copy=False)
    return xyz, filled, valid


def _canonical_order(
    xyz: np.ndarray,
    radius: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    return np.lexsort(
        (
            radius,
            valid.astype(np.uint8),
            xyz[:, 2],
            xyz[:, 1],
            xyz[:, 0],
        )
    )


def _event_seed(event_key: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}::{event_key}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _morton_order(xyz: np.ndarray, bits: int) -> np.ndarray:
    minimum = xyz.min(axis=0, keepdims=True)
    extent = np.ptp(xyz, axis=0, keepdims=True)
    scale = float(extent.max())
    normalized = np.zeros_like(xyz, dtype=np.float32)
    if scale > 0.0:
        normalized = (xyz - minimum) / np.float32(scale)
    maximum_integer = (1 << bits) - 1
    quantized = np.floor(np.clip(normalized, 0.0, 1.0) * maximum_integer).astype(
        np.uint64
    )
    codes = np.zeros(len(xyz), dtype=np.uint64)
    for bit in range(bits):
        for dimension in range(3):
            source = (quantized[:, dimension] >> np.uint64(bit)) & np.uint64(1)
            codes |= source << np.uint64(3 * bit + dimension)
    return np.argsort(codes, kind="stable")


def _group_tokens(
    xyz: np.ndarray,
    radius_normalized: np.ndarray,
    radius_valid: np.ndarray,
    group_ids: np.ndarray,
    number_of_groups: int,
    *,
    event_center: np.ndarray,
    config: SuperNEMOTokenizationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(group_ids, minlength=number_of_groups).astype(np.int64)
    if np.any(counts <= 0):
        raise RuntimeError("tokenization produced an empty group")
    centroids = np.stack(
        [
            np.bincount(group_ids, weights=xyz[:, axis], minlength=number_of_groups)
            for axis in range(3)
        ],
        axis=1,
    ) / counts[:, None]

    valid_counts = np.bincount(
        group_ids,
        weights=radius_valid.astype(np.float64),
        minlength=number_of_groups,
    )
    radius_sum = np.bincount(
        group_ids,
        weights=radius_normalized,
        minlength=number_of_groups,
    )
    radius_square_sum = np.bincount(
        group_ids,
        weights=radius_normalized * radius_normalized,
        minlength=number_of_groups,
    )
    mean_radius = np.divide(
        radius_sum,
        valid_counts,
        out=np.zeros(number_of_groups, dtype=np.float64),
        where=valid_counts > 0.0,
    )
    radius_variance = np.divide(
        radius_square_sum,
        valid_counts,
        out=np.zeros(number_of_groups, dtype=np.float64),
        where=valid_counts > 0.0,
    ) - mean_radius * mean_radius
    radius_std = np.sqrt(np.maximum(radius_variance, 0.0))
    maximum_radius = np.zeros(number_of_groups, dtype=np.float64)
    for group in range(number_of_groups):
        members = (group_ids == group) & radius_valid
        if np.any(members):
            maximum_radius[group] = float(np.max(radius_normalized[members]))

    offsets = xyz.astype(np.float64) - centroids[group_ids]
    squared_distance = np.sum(offsets * offsets, axis=1)
    spatial_rms = np.sqrt(
        np.bincount(
            group_ids,
            weights=squared_distance,
            minlength=number_of_groups,
        )
        / counts
    )
    number_of_hits = len(xyz)
    features = np.column_stack(
        (
            counts / number_of_hits,
            np.log1p(counts),
            np.full(number_of_groups, 1.0 / number_of_hits),
            mean_radius,
            radius_std,
            maximum_radius,
            valid_counts / counts,
            spatial_rms / float(config.coordinate_scale_mm),
        )
    ).astype(np.float32, copy=False)
    coordinates = (
        (centroids - event_center[None, :]) / float(config.coordinate_scale_mm)
    ).astype(np.float32)
    return coordinates, features, counts


def _entity_tokens(
    xyz: np.ndarray,
    radius_normalized: np.ndarray,
    radius_valid: np.ndarray,
    *,
    event_key: str,
    event_center: np.ndarray,
    config: SuperNEMOTokenizationConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    number_of_hits = len(xyz)
    keep = min(number_of_hits, config.max_tokens)
    if keep < number_of_hits:
        generator = np.random.default_rng(_event_seed(event_key, config.seed))
        selected = np.sort(generator.choice(number_of_hits, size=keep, replace=False))
    else:
        selected = np.arange(number_of_hits, dtype=np.int64)
    coordinates = (
        (xyz[selected].astype(np.float64) - event_center[None, :])
        / float(config.coordinate_scale_mm)
    ).astype(np.float32)
    features = np.column_stack(
        (
            np.full(keep, 1.0 / number_of_hits),
            np.full(keep, np.log1p(number_of_hits)),
            radius_normalized[selected],
            radius_valid[selected].astype(np.float32),
        )
    ).astype(np.float32, copy=False)
    return coordinates, features, float(keep / number_of_hits)


def _patch_tokens(
    xyz: np.ndarray,
    radius_normalized: np.ndarray,
    radius_valid: np.ndarray,
    *,
    event_center: np.ndarray,
    config: SuperNEMOTokenizationConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    origin = np.asarray(config.patch_origin_mm, dtype=np.float64)
    cells = np.floor(
        (xyz.astype(np.float64) - origin[None, :]) / float(config.patch_size_mm)
    ).astype(np.int64)
    _, group_ids = np.unique(cells, axis=0, return_inverse=True)
    groups = int(group_ids.max()) + 1
    coordinates, features, counts = _group_tokens(
        xyz,
        radius_normalized,
        radius_valid,
        group_ids,
        groups,
        event_center=event_center,
        config=config,
    )
    if groups > config.max_tokens:
        selected = np.argsort(-counts, kind="stable")[: config.max_tokens]
        coordinates = coordinates[selected]
        features = features[selected]
        retained = int(counts[selected].sum())
    else:
        retained = len(xyz)
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    return coordinates[order], features[order], float(retained / len(xyz))


def _summary_tokens(
    xyz: np.ndarray,
    radius_normalized: np.ndarray,
    radius_valid: np.ndarray,
    *,
    event_center: np.ndarray,
    config: SuperNEMOTokenizationConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    groups = min(len(xyz), config.summary_groups, config.max_tokens)
    order = _morton_order(xyz, config.morton_bits)
    ordered_groups = np.arange(len(xyz), dtype=np.int64) * groups // len(xyz)
    group_ids = np.empty(len(xyz), dtype=np.int64)
    group_ids[order] = ordered_groups
    coordinates, features, _ = _group_tokens(
        xyz,
        radius_normalized,
        radius_valid,
        group_ids,
        groups,
        event_center=event_center,
        config=config,
    )
    return coordinates, features, 1.0


def tokenize_tracker_event(
    coordinates: np.ndarray,
    tracker_radius: np.ndarray,
    *,
    event_key: str,
    config: SuperNEMOTokenizationConfig,
) -> tuple[dict[str, np.ndarray], float]:
    """Convert one complete event into unpadded topology tokens and coverage."""

    if not isinstance(config, SuperNEMOTokenizationConfig):
        raise TypeError("config must be a SuperNEMOTokenizationConfig")
    if not isinstance(event_key, str) or not event_key:
        raise ValueError("event_key must be a non-empty string")
    xyz, radius, valid = _validate_event(coordinates, tracker_radius)
    order = _canonical_order(xyz, radius, valid)
    xyz, radius, valid = xyz[order], radius[order], valid[order]
    event_center = xyz.mean(axis=0, dtype=np.float64)
    radius_normalized = radius / np.float32(config.tracker_radius_scale_mm)

    if config.tokenization == "entity":
        token_xyz, features, coverage = _entity_tokens(
            xyz,
            radius_normalized,
            valid,
            event_key=event_key,
            event_center=event_center,
            config=config,
        )
    elif config.tokenization == "patch":
        token_xyz, features, coverage = _patch_tokens(
            xyz,
            radius_normalized,
            valid,
            event_center=event_center,
            config=config,
        )
    else:
        token_xyz, features, coverage = _summary_tokens(
            xyz,
            radius_normalized,
            valid,
            event_center=event_center,
            config=config,
        )

    if token_xyz.shape != (len(token_xyz), 3):
        raise RuntimeError("tokenizer produced invalid coordinates")
    if features.shape != (len(token_xyz), config.feature_dim):
        raise RuntimeError("tokenizer produced invalid features")
    if not 0 < len(token_xyz) <= config.max_tokens:
        raise RuntimeError("tokenizer produced an invalid token count")
    if not np.isfinite(token_xyz).all() or not np.isfinite(features).all():
        raise RuntimeError("tokenizer produced non-finite values")
    if not math.isfinite(coverage) or not 0.0 < coverage <= 1.0:
        raise RuntimeError("tokenizer produced invalid coverage")
    return {
        "coords": np.ascontiguousarray(token_xyz, dtype=np.float32),
        "features": np.ascontiguousarray(features, dtype=np.float32),
    }, float(coverage)


__all__ = [
    "SuperNEMOTokenizationConfig",
    "TokenizationName",
    "tokenize_tracker_event",
]
