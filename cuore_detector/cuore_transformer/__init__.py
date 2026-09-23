"""Transformer for the CUORE first-two-pulse Δt regression benchmark.

Reuses ``mjd_transformer``'s three tokenizers and two coordinate encoders
unchanged and adds rotary attention (``rope``) as a third positional encoding,
built from ``next_transformer.rotary_attention``.
"""

from __future__ import annotations

from .model import (
    DEFAULT_ROPE_BASE,
    POSITION_ENCODINGS,
    CuorePositionEncodingName,
    CuoreWaveformTransformer,
)

__all__ = [
    "DEFAULT_ROPE_BASE",
    "POSITION_ENCODINGS",
    "CuorePositionEncodingName",
    "CuoreWaveformTransformer",
]
