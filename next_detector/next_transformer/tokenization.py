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

    # Combined with event_id for deterministic hit sampling.
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration immediately after construction."""

        if self.tokenization not in {
            "voxel",
            "sampled_hits",
        }:
            raise ValueError(
                "tokenization must be "
                "'voxel' or 'sampled_hits'"
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
    """Convert an event ID into a reproducible random seed.

    Python's normal ``hash()`` can change after restarting Python.
    A cryptographic digest gives us the same result every time.

    Therefore:

        same base seed + same event ID
        -> same sampled hits
    """

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
    """Validate and standardize one raw event."""

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

    Transformer batches require every event to have the same tensor
    dimensions. Real tokens are placed first, followed by zero padding.

    The mask distinguishes the real tokens from padding:

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

    number_of_tokens = min(
        len(coords),
        max_tokens,
    )

    feature_dimension = token_features.shape[1]

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

        # Event centering removes absolute detector position while
        # preserving relative distances and event shape.
        if self.config.center_coordinates:
            event_center = coords.mean(
                axis=0,
                keepdims=True,
                dtype=np.float64,
            ).astype(np.float32)

            coords = coords - event_center

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

        else:
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

        # Fixed physical scaling places coordinates closer to
        # order-one values without changing the event geometry.
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

        # Coverage measures how much of the original event energy
        # remains after sampling or truncation.
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
        """Combine hits occupying the same 3D voxel."""

        # Convert continuous coordinates into integer voxel IDs.
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

        # Number of raw hits assigned to each voxel.
        hit_counts = np.bincount(
            inverse,
            minlength=number_of_voxels,
        ).astype(np.float32)

        # Sum XYZ separately for every voxel.
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

        # Mean raw-hit coordinate inside each voxel.
        voxel_centroids = (
            coordinate_sums
            / hit_counts[:, None]
        )

        # Total deposited energy inside each voxel.
        voxel_energy = np.bincount(
            inverse,
            weights=hit_energy,
            minlength=number_of_voxels,
        ).astype(np.float32)

        # If an event produces too many voxels, retain only
        # max_tokens according to the selected rule.
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

        # Produce a deterministic spatial order.
        # np.lexsort uses its last key as the primary key.
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

        # Feature 1: fraction of total event energy deposited
        # inside this voxel.
        energy_fraction = (
            voxel_energy
            / np.float32(total_energy)
        )

        # Feature 2: compressed measure of local occupancy.
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

            # Restore the original hit order after sampling.
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

        # Feature 1: fraction of the complete event energy
        # deposited by this hit.
        energy_fraction = (
            selected_energy
            / np.float32(total_energy)
        )

        # Feature 2: global event-size information repeated
        # for every selected hit.
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

    def config_dict(self) -> dict[str, Any]:
        """Return settings for experiment metadata."""

        return self.config.to_dict()


__all__ = [
    "NEXTTokenBuilder",
    "TokenizationConfig",
    "pad_tokens",
    "stable_event_seed",
]