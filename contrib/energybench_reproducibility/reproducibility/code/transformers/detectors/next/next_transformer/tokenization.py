"""Tokenization strategies for the NEXT Transformer.

This module converts one raw detector event into a fixed-size collection
of Transformer tokens.

It does not handle:

- HDF5 loading
- Dataset splitting
- Labels
- Training
- Evaluation
- Energy-matched AUC

Those responsibilities belong to the shared EnergyBench workflow.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


TokenizationName = Literal[
    "voxel",
    "sampled_hits",
    "summary_features",
]

VoxelTruncation = Literal[
    "occupancy",
    "energy",
]


@dataclass(frozen=True)
class TokenizationConfig:
    """Configuration for one Transformer token representation."""

    tokenization: TokenizationName

    # Maximum sequence length seen by the Transformer.
    max_tokens: int = 512

    # Side length of each 3D voxel in detector coordinate units.
    # Only used for voxel tokenization.
    voxel_size: float = 15.0

    # Divide coordinates by this value after optional centering.
    coordinate_scale: float = 1000.0

    # Remove the event's absolute detector location.
    center_coordinates: bool = True

    # How to choose voxels if there are more than max_tokens.
    voxel_truncation: VoxelTruncation = "occupancy"

    # Number of bits per spatial dimension in the Morton code.
    # Only used for summary-feature tokenization.
    summary_morton_bits: int = 10

    # Combined with event_id for deterministic hit sampling.
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration immediately after construction."""

        if self.tokenization not in {
            "voxel",
            "sampled_hits",
            "summary_features",
        }:
            raise ValueError(
                "tokenization must be 'voxel', "
                "'sampled_hits', or 'summary_features'"
            )

        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError(
                "max_tokens must be a positive integer"
            )

        if (
            not math.isfinite(self.voxel_size)
            or self.voxel_size <= 0.0
        ):
            raise ValueError(
                "voxel_size must be finite and positive"
            )

        if (
            not math.isfinite(self.coordinate_scale)
            or self.coordinate_scale <= 0.0
        ):
            raise ValueError(
                "coordinate_scale must be finite and positive"
            )

        if self.voxel_truncation not in {
            "occupancy",
            "energy",
        }:
            raise ValueError(
                "voxel_truncation must be "
                "'occupancy' or 'energy'"
            )

        if (
            isinstance(self.summary_morton_bits, bool)
            or not isinstance(self.summary_morton_bits, int)
            or not 1 <= self.summary_morton_bits <= 21
        ):
            raise ValueError(
                "summary_morton_bits must be "
                "an integer in [1, 21]"
            )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer"
            )

    def to_dict(self) -> dict[str, Any]:
        """Create a JSON-serializable configuration dictionary."""

        return asdict(self)


def stable_event_seed(
    event_id: str,
    base_seed: int = 42,
) -> int:
    """Convert an event ID into a reproducible random seed."""

    text = f"{base_seed}::{event_id}"

    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
    )


def validate_event(
    coordinates: np.ndarray,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate and standardize one raw detector event."""

    coords = np.asarray(
        coordinates,
        dtype=np.float32,
    ).copy()

    hit_energy = np.asarray(
        energies,
        dtype=np.float32,
    ).copy()

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "coordinates must have shape "
            "[number_of_hits, 3]"
        )

    if (
        hit_energy.ndim != 1
        or len(hit_energy) != len(coords)
    ):
        raise ValueError(
            "energies must have shape "
            "[number_of_hits]"
        )

    if len(coords) == 0:
        raise ValueError(
            "an event must contain at least one hit"
        )

    if not np.isfinite(coords).all():
        raise ValueError(
            "coordinates must contain only finite values"
        )

    if not np.isfinite(hit_energy).all():
        raise ValueError(
            "energies must contain only finite values"
        )

    if np.any(hit_energy < 0.0):
        raise ValueError(
            "hit energies must be non-negative"
        )

    total_energy = float(
        hit_energy.sum(dtype=np.float64)
    )

    if (
        not math.isfinite(total_energy)
        or total_energy <= 0.0
    ):
        raise ValueError(
            "event total energy must be finite and positive"
        )

    return coords, hit_energy, total_energy


def pad_tokens(
    coordinates: np.ndarray,
    features: np.ndarray,
    max_tokens: int,
) -> dict[str, np.ndarray]:
    """Pad one variable-length event to a fixed sequence length.

    The returned mask uses:

        True  = real token
        False = padding
    """

    coords = np.asarray(
        coordinates,
        dtype=np.float32,
    )

    token_features = np.asarray(
        features,
        dtype=np.float32,
    )

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "token coordinates must have shape [N, 3]"
        )

    if token_features.ndim != 2:
        raise ValueError(
            "token features must have shape "
            "[N, feature_dimension]"
        )

    if len(coords) != len(token_features):
        raise ValueError(
            "coordinates and features must have "
            "the same number of tokens"
        )

    if len(coords) == 0:
        raise ValueError(
            "an event must produce at least one token"
        )

    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError(
            "max_tokens must be a positive integer"
        )

    if not np.isfinite(coords).all():
        raise ValueError(
            "token coordinates contain non-finite values"
        )

    if not np.isfinite(token_features).all():
        raise ValueError(
            "token features contain non-finite values"
        )

    number_of_tokens = min(
        len(coords),
        max_tokens,
    )

    feature_dimension = (
        token_features.shape[1]
    )

    padded_coords = np.zeros(
        (max_tokens, 3),
        dtype=np.float32,
    )

    padded_features = np.zeros(
        (max_tokens, feature_dimension),
        dtype=np.float32,
    )

    valid_mask = np.zeros(
        max_tokens,
        dtype=bool,
    )

    padded_coords[:number_of_tokens] = (
        coords[:number_of_tokens]
    )

    padded_features[:number_of_tokens] = (
        token_features[:number_of_tokens]
    )

    valid_mask[:number_of_tokens] = True

    return {
        "coords": padded_coords,
        "features": padded_features,
        "mask": valid_mask,
    }


def _morton_spatial_order(
    coordinates: np.ndarray,
    *,
    bits: int,
) -> np.ndarray:
    """Return a deterministic locality-preserving ordering of 3D hits.

    XYZ coordinates are quantized into a cube, and their binary bits are
    interleaved to produce Morton codes. Nearby points will generally be
    close in the resulting one-dimensional ordering.

    The Morton codes are used only to form spatial groups. They are not
    passed to the Transformer.
    """

    coords = np.asarray(
        coordinates,
        dtype=np.float32,
    )

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "coordinates must have shape "
            "[number_of_hits, 3]"
        )

    if len(coords) == 0:
        raise ValueError(
            "Morton ordering requires at least one coordinate"
        )

    coordinate_minimum = coords.min(
        axis=0,
        keepdims=True,
    )

    coordinate_extent = np.ptp(
        coords,
        axis=0,
        keepdims=True,
    )

    # Use one scale for all three axes so the event's physical
    # aspect ratio is not distorted.
    spatial_scale = float(
        coordinate_extent.max()
    )

    normalized = np.zeros_like(
        coords,
        dtype=np.float32,
    )

    if spatial_scale > 0.0:
        normalized = (
            coords - coordinate_minimum
        ) / np.float32(spatial_scale)

    maximum_integer = (1 << bits) - 1

    quantized = np.floor(
        np.clip(
            normalized,
            0.0,
            1.0,
        )
        * maximum_integer
    ).astype(np.uint64)

    morton_codes = np.zeros(
        len(coords),
        dtype=np.uint64,
    )

    for bit in range(bits):
        for dimension in range(3):
            source_bit = (
                quantized[:, dimension]
                >> np.uint64(bit)
            ) & np.uint64(1)

            destination_bit = np.uint64(
                3 * bit + dimension
            )

            morton_codes |= (
                source_bit
                << destination_bit
            )

    return np.argsort(
        morton_codes,
        kind="stable",
    )


class NEXTTokenBuilder:
    """Convert a complete NEXT event into Transformer inputs.

    EnergyBench calls this object once for each event:

        inputs, coverage = token_builder(
            coordinates,
            energies,
            event_id=event_id,
        )
    """

    def __init__(
        self,
        config: TokenizationConfig,
    ) -> None:
        if not isinstance(
            config,
            TokenizationConfig,
        ):
            raise TypeError(
                "config must be a TokenizationConfig"
            )

        self.config = config

    @property
    def feature_dim(self) -> int:
        """Return the number of content features per token."""

        if (
            self.config.tokenization
            == "summary_features"
        ):
            return 4

        return 2

    def __call__(
        self,
        coordinates: np.ndarray,
        energies: np.ndarray,
        *,
        event_id: str,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Tokenize one complete detector event."""

        (
            coords,
            hit_energy,
            total_energy,
        ) = validate_event(
            coordinates,
            energies,
        )

        # Remove absolute detector location while preserving
        # relative distances and event shape.
        if self.config.center_coordinates:
            event_center = coords.mean(
                axis=0,
                keepdims=True,
                dtype=np.float64,
            ).astype(np.float32)

            coords = (
                coords - event_center
            )

        if self.config.tokenization == "voxel":
            (
                token_coords,
                token_features,
                retained_energy,
            ) = self._voxel_tokenize(
                coords=coords,
                hit_energy=hit_energy,
                total_energy=total_energy,
            )

        elif (
            self.config.tokenization
            == "sampled_hits"
        ):
            (
                token_coords,
                token_features,
                retained_energy,
            ) = self._sampled_hit_tokenize(
                coords=coords,
                hit_energy=hit_energy,
                total_energy=total_energy,
                event_id=event_id,
            )

        elif (
            self.config.tokenization
            == "summary_features"
        ):
            (
                token_coords,
                token_features,
                retained_energy,
            ) = self._summary_feature_tokenize(
                coords=coords,
                hit_energy=hit_energy,
                total_energy=total_energy,
            )

        else:
            raise RuntimeError(
                "Unsupported tokenization: "
                f"{self.config.tokenization}"
            )

        # Place coordinates closer to order-one values.
        token_coords = (
            token_coords
            / np.float32(
                self.config.coordinate_scale
            )
        )

        inputs = pad_tokens(
            coordinates=token_coords,
            features=token_features,
            max_tokens=self.config.max_tokens,
        )

        coverage = (
            retained_energy
            / total_energy
        )

        coverage = float(
            np.clip(
                coverage,
                0.0,
                1.0,
            )
        )

        return inputs, coverage

    def _voxel_tokenize(
        self,
        *,
        coords: np.ndarray,
        hit_energy: np.ndarray,
        total_energy: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Combine hits occupying the same fixed 3D voxel."""

        voxel_indices = np.floor(
            coords / self.config.voxel_size
        ).astype(np.int32)

        (
            _,
            inverse,
        ) = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True,
        )

        number_of_voxels = (
            int(inverse.max()) + 1
        )

        hit_counts = np.bincount(
            inverse,
            minlength=number_of_voxels,
        ).astype(np.float32)

        coordinate_sums = np.stack(
            [
                np.bincount(
                    inverse,
                    weights=coords[:, dimension],
                    minlength=number_of_voxels,
                )
                for dimension in range(3)
            ],
            axis=1,
        ).astype(np.float32)

        voxel_centroids = (
            coordinate_sums
            / hit_counts[:, None]
        )

        voxel_energy = np.bincount(
            inverse,
            weights=hit_energy,
            minlength=number_of_voxels,
        ).astype(np.float32)

        if (
            number_of_voxels
            > self.config.max_tokens
        ):
            if (
                self.config.voxel_truncation
                == "occupancy"
            ):
                priority = hit_counts

            else:
                priority = voxel_energy

            selected = np.argsort(
                -priority,
                kind="stable",
            )[:self.config.max_tokens]

            voxel_centroids = (
                voxel_centroids[selected]
            )

            hit_counts = (
                hit_counts[selected]
            )

            voxel_energy = (
                voxel_energy[selected]
            )

        spatial_order = np.lexsort(
            (
                voxel_centroids[:, 2],
                voxel_centroids[:, 1],
                voxel_centroids[:, 0],
            )
        )

        voxel_centroids = (
            voxel_centroids[spatial_order]
        )

        hit_counts = (
            hit_counts[spatial_order]
        )

        voxel_energy = (
            voxel_energy[spatial_order]
        )

        energy_fraction = (
            voxel_energy
            / np.float32(total_energy)
        )

        log_occupancy = np.log1p(
            hit_counts
        )

        token_features = np.column_stack(
            [
                energy_fraction,
                log_occupancy,
            ]
        ).astype(np.float32)

        retained_energy = float(
            voxel_energy.sum(dtype=np.float64)
        )

        return (
            voxel_centroids.astype(np.float32),
            token_features,
            retained_energy,
        )

    def _sampled_hit_tokenize(
        self,
        *,
        coords: np.ndarray,
        hit_energy: np.ndarray,
        total_energy: float,
        event_id: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Represent an event using selected original detector hits."""

        number_of_hits = len(coords)

        if (
            number_of_hits
            > self.config.max_tokens
        ):
            random_generator = (
                np.random.default_rng(
                    stable_event_seed(
                        event_id,
                        self.config.seed,
                    )
                )
            )

            selected_indices = (
                random_generator.choice(
                    number_of_hits,
                    size=self.config.max_tokens,
                    replace=False,
                )
            )

            # Restore original hit ordering after sampling.
            selected_indices.sort()

            selected_coords = (
                coords[selected_indices]
            )

            selected_energy = (
                hit_energy[selected_indices]
            )

        else:
            selected_coords = coords
            selected_energy = hit_energy

        energy_fraction = (
            selected_energy
            / np.float32(total_energy)
        )

        log_total_hit_count = np.float32(
            np.log1p(number_of_hits)
        )

        token_features = np.column_stack(
            [
                energy_fraction,
                np.full(
                    len(selected_coords),
                    log_total_hit_count,
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        retained_energy = float(
            selected_energy.sum(dtype=np.float64)
        )

        return (
            selected_coords.astype(np.float32),
            token_features,
            retained_energy,
        )

    def _summary_feature_tokenize(
        self,
        *,
        coords: np.ndarray,
        hit_energy: np.ndarray,
        total_energy: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Create balanced spatial groups and summarize each group.

        All hits are sorted using a locality-preserving Morton order.
        Consecutive spatially ordered hits are assigned to balanced
        groups. Each group becomes one Transformer token.
        """

        number_of_hits = len(coords)

        number_of_tokens = min(
            number_of_hits,
            self.config.max_tokens,
        )

        spatial_order = (
            _morton_spatial_order(
                coords,
                bits=(
                    self.config
                    .summary_morton_bits
                ),
            )
        )

        # The integer formula distributes hits as evenly as possible
        # across number_of_tokens groups.
        ordered_group_ids = (
            np.arange(
                number_of_hits,
                dtype=np.int64,
            )
            * number_of_tokens
            // number_of_hits
        ).astype(np.int32)

        # Convert the ordered group IDs back into original-hit order.
        group_ids = np.empty(
            number_of_hits,
            dtype=np.int32,
        )

        group_ids[spatial_order] = (
            ordered_group_ids
        )

        hit_counts = np.bincount(
            group_ids,
            minlength=number_of_tokens,
        ).astype(np.float32)

        coordinate_sums = np.stack(
            [
                np.bincount(
                    group_ids,
                    weights=coords[:, dimension],
                    minlength=number_of_tokens,
                )
                for dimension in range(3)
            ],
            axis=1,
        ).astype(np.float32)

        group_centroids = (
            coordinate_sums
            / hit_counts[:, None]
        )

        group_energy = np.bincount(
            group_ids,
            weights=hit_energy,
            minlength=number_of_tokens,
        ).astype(np.float32)

        maximum_hit_energy = np.zeros(
            number_of_tokens,
            dtype=np.float32,
        )

        np.maximum.at(
            maximum_hit_energy,
            group_ids,
            hit_energy,
        )

        offsets = (
            coords
            - group_centroids[group_ids]
        )

        squared_distances = np.sum(
            offsets**2,
            axis=1,
        )

        mean_squared_distance = (
            np.bincount(
                group_ids,
                weights=squared_distances,
                minlength=number_of_tokens,
            )
            / hit_counts
        )

        rms_distance = np.sqrt(
            np.maximum(
                mean_squared_distance,
                0.0,
            )
        ).astype(np.float32)

        # Summary feature 1:
        # fraction of total event energy in the spatial group.
        energy_fraction = (
            group_energy
            / np.float32(total_energy)
        )

        # Summary feature 2:
        # compressed number of raw hits in the group.
        log_occupancy = np.log1p(
            hit_counts
        )

        # Summary feature 3:
        # largest individual hit energy relative to total energy.
        maximum_energy_fraction = (
            maximum_hit_energy
            / np.float32(total_energy)
        )

        # Summary feature 4:
        # spatial spread of the hits in the group.
        normalized_rms_distance = (
            rms_distance
            / np.float32(
                self.config.coordinate_scale
            )
        )

        token_features = np.column_stack(
            [
                energy_fraction,
                log_occupancy,
                maximum_energy_fraction,
                normalized_rms_distance,
            ]
        ).astype(np.float32)

        # Summary tokenization represents every original hit.
        retained_energy = float(
            group_energy.sum(dtype=np.float64)
        )

        return (
            group_centroids.astype(np.float32),
            token_features,
            retained_energy,
        )

    def config_dict(self) -> dict[str, Any]:
        """Return settings for experiment metadata."""

        return self.config.to_dict()


__all__ = [
    "NEXTTokenBuilder",
    "TokenizationConfig",
    "pad_tokens",
    "stable_event_seed",
]