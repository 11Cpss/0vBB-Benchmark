"""SuperNEMO event split, streaming loaders, and evaluation helpers."""

from .config import (
    DEFAULT_DATA_ROOT,
    PUBLISHED_CLASSIFICATION_COUNTS,
    DataConfig,
)
from .evaluation import EnergyWindowLoader
from .data import (
    PreparedData,
    SuperNEMOEventDataset,
    block_assignments,
    build_split,
    collate_events,
    index_sources,
    prepare_dataset,
    scan_event_offsets,
    split_counts,
    verify_published_counts,
)


__all__ = [
    "DEFAULT_DATA_ROOT",
    "PUBLISHED_CLASSIFICATION_COUNTS",
    "DataConfig",
    "EnergyWindowLoader",
    "PreparedData",
    "SuperNEMOEventDataset",
    "block_assignments",
    "build_split",
    "collate_events",
    "index_sources",
    "prepare_dataset",
    "scan_event_offsets",
    "split_counts",
    "verify_published_counts",
]
