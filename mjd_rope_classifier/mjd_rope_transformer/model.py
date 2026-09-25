"""Transformer for the MJD clean/non-clean (signal vs. background) classifier.

This model is a deliberate merge of two pieces that already exist elsewhere
in this repository:

- ``mjd_transformer.model.MJDTransformer``'s ``task`` parameter and shared
  training contract (one raw value per event; classification returns a
  logit consumed by ``mjdbench.training``'s unchanged ``train_model`` /
  ``evaluate_model``);
- ``cuore_transformer.model.CuoreWaveformTransformer``'s rotary-attention
  (``rope``) branch, itself built from
  ``next_transformer.rotary_attention.RotaryTransformerEncoder``.

The three MJD tokenizers (``build_tokenizer``) and the two additive
coordinate encoders (``build_position_encoder``) are reused unchanged, so the
six non-RoPE cells are architecturally identical to
``MJDTransformer(task="classification", ...)``. Nothing in ``mjd_detector``
or ``next_detector`` is modified.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor, nn

from mjd_transformer.positional_encoding import build_position_encoder
from mjd_transformer.tokenization import TokenizationConfig, build_tokenizer
from next_transformer.rotary_attention import RotaryTransformerEncoder


Task = Literal["classification", "regression"]

MJDRopePositionEncodingName = Literal[
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
# MJD tokenizers (mjd_transformer.tokenization) emit three coordinate
# channels, all in [-1, 1] (span 2): normalized time, normalized
# amplitude/region mean, and a local/first-difference term. With d_model=64
# and nhead=4, head_dim=16 -> 8 rotation pairs, split [3, 3, 2] across the
# three axes (next_transformer.rotary_attention._axis_pair_counts). Pair j of
# an axis with `count` pairs rotates at theta_j = rope_base ** (1 - j/count),
# so the slowest channel on that axis is theta_slow = rope_base ** (1/count).
#
# RoPE attention depends on Δcoord only through cos(theta * Δcoord), which is
# injective only while the argument stays within [0, pi]. The worst case
# pairwise gap on any axis spanning [-1, 1] is Δcoord = 2.0 (first token vs.
# last token), so every axis's slowest channel must satisfy
#
#     theta_slow * 2.0 <= pi   ->   rope_base ** (1/count) <= pi/2
#
#   time / amplitude axes (count=3): rope_base <= (pi/2)**3 ~= 3.876
#   first-difference axis (count=2, the binding constraint):
#                                    rope_base <= (pi/2)**2 ~= 2.467
#
# base=2.0 sits inside the binding bound with ~19% headroom, matching the
# margin cuore_transformer.model.DEFAULT_ROPE_BASE uses for its own (looser,
# task-specific) bound. This value does NOT transfer from CUORE's
# DEFAULT_ROPE_BASE=8.0: that bound was relaxed using the largest Δt actually
# present in CUORE's regression target, a task-specific quantity with no
# analog here, where every pairwise token interaction across the full event
# matters for classification.
DEFAULT_MJD_ROPE_BASE: float = 2.0


class MJDRopeTransformer(nn.Module):
    """Tokenize one dense MJD waveform and predict one event-level value.

    ``forward`` takes the raw (already normalized) waveform, exactly like
    ``MJDTransformer``, so tokenization happens inside the model and every
    baseline receives an identical input tensor.
    """

    def __init__(
        self,
        *,
        task: Task,
        position_encoding: MJDRopePositionEncodingName,
        tokenization_config: TokenizationConfig | None = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_frequencies: int = 6,
        rope_base: float = DEFAULT_MJD_ROPE_BASE,
        rope_time_axis_only: bool = False,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            task=task,
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
        self.task = task
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
        # MJDTransformer.__init__ so that, for the two additive encodings,
        # the same seed produces bit-identical weights.
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
        """Return one classification logit or regression prediction per event."""

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
                # Diagnostic mode: rotate by normalized time only. The other
                # two axes rotate by angle 0, i.e. identity. Not part of the
                # official matrix -- it makes the RoPE cells see strictly
                # less than the additive encodings.
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
        task: str,
        position_encoding: str,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_frequencies: int,
        rope_base: float,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
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


class MJDRopeTransformerClassifier(MJDRopeTransformer):
    """Convenience classifier with the shared clean/non-clean output contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(task="classification", **kwargs)


__all__ = [
    "DEFAULT_MJD_ROPE_BASE",
    "POSITION_ENCODINGS",
    "MJDRopePositionEncodingName",
    "MJDRopeTransformer",
    "MJDRopeTransformerClassifier",
    "Task",
]
