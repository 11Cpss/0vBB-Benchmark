"""Transformer for the MJD RoPE classification benchmark.

Reuses ``mjd_transformer``'s three tokenizers and two additive coordinate
encoders unchanged and adds rotary attention (``rope``) as a third positional
encoding, built from ``next_transformer.rotary_attention``.
"""

from __future__ import annotations

from .model import (
    DEFAULT_MJD_ROPE_BASE,
    POSITION_ENCODINGS,
    MJDRopePositionEncodingName,
    MJDRopeTransformer,
    MJDRopeTransformerClassifier,
)

__all__ = [
    "DEFAULT_MJD_ROPE_BASE",
    "POSITION_ENCODINGS",
    "MJDRopePositionEncodingName",
    "MJDRopeTransformer",
    "MJDRopeTransformerClassifier",
]
