"""Waveform Transformer for the CUORE first-two-pulse Δt regression benchmark.

This model is a deliberate superset of ``mjd_transformer.MJDTransformer``:

- the three MJD tokenizers are reused unchanged (``build_tokenizer``), so a
  CUORE cell and the corresponding MJD cell see the same token construction;
- ``coordinate_mlp`` and ``fourier_coordinates`` are reused unchanged
  (``build_position_encoder``), and those six cells are architecturally
  identical to ``MJDTransformer(task="regression", ...)``;
- a third positional encoding, ``rope``, is added by swapping the encoder for
  ``next_transformer.rotary_attention.RotaryTransformerEncoder`` and threading
  the token coordinates into every attention layer instead of adding a position
  embedding at the input.

Nothing in ``mjd_detector`` or ``next_detector`` is modified.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor, nn

from mjd_transformer.positional_encoding import build_position_encoder
from mjd_transformer.tokenization import TokenizationConfig, build_tokenizer
from next_transformer.rotary_attention import RotaryTransformerEncoder


CuorePositionEncodingName = Literal[
    "coordinate_mlp",
    "fourier_coordinates",
    "rope",
]

POSITION_ENCODINGS: tuple[str, ...] = (
    "coordinate_mlp",
    "fourier_coordinates",
    "rope",
)

# Rotary base for the MJD coordinate convention.
#
# MJD tokenizers emit coords[..., 0] = linspace(-1, 1, N) (span 2) plus two
# amplitude-derived channels of order 1. With d_model=64 and nhead=4 the
# head_dim is 16 -> 8 rotation pairs split [3, 3, 2] across the three axes, and
# pair j of an axis with `count` pairs rotates at
# theta_j = rope_base ** (1 - j / count).
#
# RoPE attention depends only on Δcoord through cos(theta * Δcoord), which is
# injective only on [0, pi]. The largest Δt actually present in the data is
# 6812.7 ms out of a 10 s record = 0.681 of the span = Δcoord 1.363, so the
# SLOWEST channel must satisfy
#
#     theta_slow * 1.363 <= pi   ->   base**(1/3) <= 2.305   ->   base <= 12.25
#
# Below that bound two different Δt values collapse to the same phase on every
# channel -- the aliasing failure documented in
# notes/papers/rope_relative_encoding_limitation.md.
#
# base = 8 sits inside the bound with ~13% headroom (slow channel 2.73 rad over
# the full observed range) and gives a clean octave ladder 8 / 4 / 2. The fast
# channel is then coarse (0.032 rad per raw_patches token); with only 3 pairs on
# the time axis no base can be both unambiguous and finely resolving -- that is
# a structural limit of head_dim=16, not a tuning failure.
#
# The value 128 used by the earlier CUORE patch tokenizer does NOT transfer: it
# was derived for coords in [0, 1) with 100 tokens. Under this convention its
# slow channel turns 6.87 rad > 2*pi. Sweep {4, 8, 16, 32} to measure how much
# fast-end resolution is worth.
DEFAULT_ROPE_BASE: float = 8.0


class CuoreWaveformTransformer(nn.Module):
    """Tokenize one CUORE waveform and regress a single scalar per event.

    ``forward`` takes the raw (already normalized) waveform, exactly like
    ``MJDTransformer``, so tokenization happens inside the model and every
    baseline receives identical inputs.
    """

    def __init__(
        self,
        *,
        position_encoding: CuorePositionEncodingName,
        tokenization_config: TokenizationConfig | None = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_frequencies: int = 6,
        rope_base: float = DEFAULT_ROPE_BASE,
        rope_time_axis_only: bool = False,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            position_encoding=position_encoding,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_frequencies=num_frequencies,
            rope_base=rope_base,
        )

        selected_tokens = tokenization_config or TokenizationConfig()
        self.task = "regression"
        self.position_encoding_name = position_encoding
        self.tokenization_config = selected_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = float(dropout)
        self.num_frequencies = num_frequencies
        self.rope_base = float(rope_base)
        self.rope_time_axis_only = bool(rope_time_axis_only)

        self.tokenizer = build_tokenizer(selected_tokens)
        self.feature_dim = int(self.tokenizer.feature_dim)
        self.coordinate_dim = int(self.tokenizer.coordinate_dim)

        self._uses_rotary_attention = position_encoding == "rope"
        if self._uses_rotary_attention:
            head_dim = d_model // nhead
            if head_dim % 2 != 0 or head_dim < 6:
                raise ValueError(
                    "position_encoding='rope' requires d_model // nhead to be "
                    "even and at least 6 so channels split across the three "
                    f"coordinate axes; got head_dim={head_dim}"
                )
            if self.coordinate_dim != 3:
                raise ValueError(
                    "position_encoding='rope' requires exactly 3 coordinate "
                    f"channels; tokenizer emits {self.coordinate_dim}"
                )

        # Module construction order is deliberately identical to
        # MJDTransformer.__init__ so that, for the two additive encodings, the
        # same seed produces bit-identical weights. tests/ asserts this.
        self.content_projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.position_encoder = (
            None
            if self._uses_rotary_attention
            else build_position_encoder(
                position_encoding,
                coordinate_dim=self.coordinate_dim,
                d_model=d_model,
                num_frequencies=num_frequencies,
            )
        )
        self.input_norm = nn.LayerNorm(d_model)
        if self._uses_rotary_attention:
            # Rotary attention has no additive position embedding: position
            # enters by rotating Q/K inside every self-attention layer.
            self.transformer = RotaryTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rope_base=rope_base,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )

        self.output_norm = nn.LayerNorm(d_model)
        hidden = max(1, d_model // 2)
        self.task_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Return one scalar regression prediction per event, shape ``[B]``."""

        token_batch = self.tokenizer(waveform)
        coordinates = token_batch["coords"]
        features = token_batch["features"]
        valid_mask = token_batch["mask"]
        self._validate_token_batch(coordinates, features, valid_mask)

        content = self.content_projection(features)
        padding_mask = ~valid_mask

        if self._uses_rotary_attention:
            token_embeddings = self.input_norm(content)
            rotation_coordinates = coordinates
            if self.rope_time_axis_only:
                # Diagnostic mode: rotate by normalized time only. The other two
                # axes rotate by angle 0, i.e. identity. Not part of the
                # official matrix -- it makes the RoPE cells see strictly less
                # than the additive encodings.
                zeros = torch.zeros_like(coordinates[..., 0])
                rotation_coordinates = torch.stack(
                    (coordinates[..., 0], zeros, zeros), dim=-1
                )
            transformed = self.transformer(
                token_embeddings,
                rotation_coordinates,
                key_padding_mask=padding_mask,
            )
        else:
            position = self.position_encoder(coordinates)
            token_embeddings = self.input_norm(content + position)
            transformed = self.transformer(
                token_embeddings,
                src_key_padding_mask=padding_mask,
            )

        numeric_mask = valid_mask.unsqueeze(-1).to(transformed.dtype)
        pooled = (transformed * numeric_mask).sum(dim=1)
        pooled = pooled / numeric_mask.sum(dim=1).clamp_min(1.0)
        pooled = self.output_norm(pooled)
        return self.task_head(pooled).squeeze(-1)

    def config_dict(self) -> dict[str, Any]:
        """Return the complete representation and architecture configuration."""

        return {
            "task": self.task,
            "tokenization": self.tokenization_config.to_dict(),
            "position_encoding": self.position_encoding_name,
            "coordinate_dim": self.coordinate_dim,
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "num_frequencies": self.num_frequencies,
            "pooling": "masked_mean",
            "attention": (
                "rotary" if self._uses_rotary_attention else "scaled_dot_product"
            ),
            "rope_base": self.rope_base if self._uses_rotary_attention else None,
            "rope_time_axis_only": (
                self.rope_time_axis_only if self._uses_rotary_attention else None
            ),
            "rope_thetas_per_axis": self._rope_thetas(),
        }

    def _rope_thetas(self) -> dict[str, list[float]] | None:
        """Resolved rotation frequencies per coordinate axis, for auditing."""

        if not self._uses_rotary_attention:
            return None
        from next_transformer.rotary_attention import _axis_pair_counts

        counts = _axis_pair_counts((self.d_model // self.nhead) // 2)
        axis_names = ("time", "amplitude", "first_difference")
        return {
            axis_names[axis]: [
                self.rope_base ** (1.0 - index / count)
                for index in range(count)
            ]
            for axis, count in enumerate(counts)
        }

    def _validate_token_batch(
        self,
        coordinates: Tensor,
        features: Tensor,
        valid_mask: Tensor,
    ) -> None:
        if coordinates.ndim != 3 or coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("token coordinates have an invalid shape")
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("token features have an invalid shape")
        if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
            raise ValueError("token mask must be Boolean with shape [B, N]")
        if coordinates.shape[:2] != features.shape[:2]:
            raise ValueError("coordinates and features must share [B, N]")
        if coordinates.shape[:2] != valid_mask.shape:
            raise ValueError("coordinates and mask must share [B, N]")
        if not bool(valid_mask.any(dim=1).all().item()):
            raise ValueError("every waveform must produce at least one valid token")

    @staticmethod
    def _validate_configuration(
        *,
        position_encoding: str,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_frequencies: int,
        rope_base: float,
    ) -> None:
        if position_encoding not in set(POSITION_ENCODINGS):
            raise ValueError(
                "position_encoding must be one of "
                + ", ".join(repr(name) for name in POSITION_ENCODINGS)
            )
        integers = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "num_frequencies": num_frequencies,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if (
            isinstance(rope_base, bool)
            or not isinstance(rope_base, (int, float))
            or not math.isfinite(float(rope_base))
            or float(rope_base) <= 1.0
        ):
            raise ValueError("rope_base must be finite and greater than 1.0")


__all__ = [
    "DEFAULT_ROPE_BASE",
    "POSITION_ENCODINGS",
    "CuorePositionEncodingName",
    "CuoreWaveformTransformer",
]
