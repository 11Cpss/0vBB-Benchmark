"""SuperNEMO tracker-hit tokenization and RoPE Transformer factory."""

from .model import (
    DEFAULT_SUPERNEMO_ROPE_BASE,
    POSITION_ENCODINGS,
    build_supernemo_transformer,
    rope_frequency_table,
    coarsest_unambiguous_extents,
)
from .tokenization import (
    SuperNEMOTrackerTokenizationConfig,
    tokenize_tracker_event,
)


__all__ = [
    "DEFAULT_SUPERNEMO_ROPE_BASE",
    "POSITION_ENCODINGS",
    "SuperNEMOTrackerTokenizationConfig",
    "build_supernemo_transformer",
    "rope_frequency_table",
    "coarsest_unambiguous_extents",
    "tokenize_tracker_event",
]
