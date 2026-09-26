"""Explicit configuration for the SuperNEMO RoPE classification benchmark.

The task, event key, split rule, and energy definition follow the published
SuperNEMO Transformer benchmark so that RoPE results are directly comparable
with its coordinate-MLP and Fourier-XYZ models:

- signal ``2nubb`` (label 1) vs. background ``Bi214`` (label 0);
- one event = the contiguous rows sharing an ``ev_no`` in one source file;
- 80/10/10 train/validation/test split by seeded blocks of 4096 events;
- the model sees tracker-hit topology only; ``E1 + E2`` is used solely as the
  conditioning energy of the energy-matched evaluation.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(
    os.environ.get("SUPERNEMO_ROPE_DATA_DIR", "data/SuperNEMO")
).expanduser()

SPLIT_NAMES = ("train", "validation", "test")

EVENT_ID_FIELD = "ev_no"
COORDINATE_FIELDS = ("tX", "tY", "tZ")
RADIUS_FIELD = "tR"
ENERGY_FIELDS = ("E1", "E2")
KEV_PER_MEV = 1000.0


@dataclass(frozen=True)
class SourceSpec:
    """Identity and task eligibility of one raw HDF5 source file."""

    source_key: str
    file_name: str
    raw_label: str
    category: str
    classification_label: int | None


# The order is part of the split contract: a source's block permutation is
# seeded with its position in this tuple. All four sources are listed, even
# though only two are classified, so the positions match the published split.
SOURCE_SPECS = (
    SourceSpec("0nubb", "data_0nubb_merged.h5", "0nubb", "0nubb", None),
    SourceSpec("2nubb", "data_2nubb_merged.h5", "2nubb", "2nu", 1),
    SourceSpec("Bi214", "data_Bi214_merged.h5", "Bi214", "Bi214", 0),
    SourceSpec("Tl208", "data_Tl208_merged.h5", "Tl208", "Tl208", None),
)
SOURCE_BY_KEY = {spec.source_key: spec for spec in SOURCE_SPECS}
SOURCE_POSITION = {spec.source_key: index for index, spec in enumerate(SOURCE_SPECS)}
CLASSIFICATION_SOURCES = tuple(
    spec for spec in SOURCE_SPECS if spec.classification_label is not None
)

# Events per (split, category) in the published SuperNEMO split manifest,
# produced with the default DataConfig below. ``verify_published_counts``
# compares against this to prove the split was reproduced exactly.
PUBLISHED_CLASSIFICATION_COUNTS: dict[str, dict[str, int]] = {
    "train": {"2nu": 2_629_632, "Bi214": 2_105_618},
    "validation": {"2nu": 326_804, "Bi214": 266_240},
    "test": {"2nu": 327_680, "Bi214": 262_144},
}


@dataclass(frozen=True)
class DataConfig:
    """Raw-data location, split rule, and optional per-split event caps."""

    data_root: Path = DEFAULT_DATA_ROOT
    seed: int = 42
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_block_events: int = 4096
    # ``None`` keeps every event of the split. A cap keeps a seeded set of
    # whole blocks per source, in proportion to that source's share of the
    # split, so reads stay contiguous and the class ratio is preserved.
    max_train_events: int | None = None
    max_validation_events: int | None = None
    max_test_events: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        fractions = tuple(float(item) for item in self.split_fractions)
        if len(fractions) != 3 or any(item <= 0.0 for item in fractions):
            raise ValueError("split_fractions must contain three positive values")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("split_fractions must sum to one")
        object.__setattr__(self, "split_fractions", fractions)
        if int(self.split_block_events) <= 0:
            raise ValueError("split_block_events must be positive")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        for name in ("max_train_events", "max_validation_events", "max_test_events"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive or None")

    def event_cap(self, split: str) -> int | None:
        """Return the configured event cap for one split, if any."""

        return {
            "train": self.max_train_events,
            "validation": self.max_validation_events,
            "test": self.max_test_events,
        }[split]

    @property
    def is_published_split(self) -> bool:
        """True when this config reproduces the published split exactly."""

        return (
            int(self.seed) == 42
            and self.split_fractions == (0.8, 0.1, 0.1)
            and int(self.split_block_events) == 4096
            and all(self.event_cap(split) is None for split in SPLIT_NAMES)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["split_fractions"] = list(self.split_fractions)
        return payload


__all__ = [
    "CLASSIFICATION_SOURCES",
    "COORDINATE_FIELDS",
    "DEFAULT_DATA_ROOT",
    "ENERGY_FIELDS",
    "EVENT_ID_FIELD",
    "KEV_PER_MEV",
    "PUBLISHED_CLASSIFICATION_COUNTS",
    "RADIUS_FIELD",
    "SOURCE_BY_KEY",
    "SOURCE_POSITION",
    "SOURCE_SPECS",
    "SPLIT_NAMES",
    "DataConfig",
    "SourceSpec",
]
