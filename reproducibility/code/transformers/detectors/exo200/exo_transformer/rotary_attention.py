"""Rotary self-attention for EXO-200 sensor-aware coordinates.

EXO tokens carry six continuous/indicator coordinates rather than a single
sequence index.  This module assigns each Query/Key channel pair to one of
those coordinate axes and rotates it inside every attention layer.  Values
are deliberately left unchanged.  Consequently, this is a pure relative
positional encoding rather than an additive coordinate embedding.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .positional_encoding import validate_coordinates


def rotate_half(value: Tensor) -> Tensor:
    """Return the quarter-turn companion used by rotary embeddings."""

    if value.shape[-1] % 2:
        raise ValueError("the rotary dimension must be even")
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def apply_rotary_embedding(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate ``value`` using broadcast-compatible cosine and sine tensors."""

    if value.shape[-1] != cosine.shape[-1] or value.shape[-1] != sine.shape[-1]:
        raise ValueError("value, cosine, and sine must share their last dimension")
    return value * cosine + rotate_half(value) * sine


def axis_pair_counts(pairs_total: int, coordinate_dim: int) -> tuple[int, ...]:
    """Distribute rotary channel pairs evenly across coordinate axes."""

    for value, name in (
        (pairs_total, "pairs_total"),
        (coordinate_dim, "coordinate_dim"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if pairs_total < coordinate_dim:
        raise ValueError(
            "each coordinate requires at least one rotary pair; "
            f"received {pairs_total} pairs for {coordinate_dim} coordinates"
        )

    base, remainder = divmod(pairs_total, coordinate_dim)
    return tuple(base + int(axis < remainder) for axis in range(coordinate_dim))


class RotaryPositionAngles(nn.Module):
    """Compute rotary angles from arbitrary continuous token coordinates."""

    def __init__(
        self,
        *,
        head_dim: int,
        coordinate_dim: int,
        rope_base: float = math.pi / 2.0,
    ) -> None:
        super().__init__()
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        if (
            isinstance(coordinate_dim, bool)
            or not isinstance(coordinate_dim, int)
            or coordinate_dim <= 0
        ):
            raise ValueError("coordinate_dim must be a positive integer")
        if (
            isinstance(rope_base, bool)
            or not isinstance(rope_base, (int, float))
            or not math.isfinite(float(rope_base))
            or float(rope_base) <= 1.0
        ):
            raise ValueError("rope_base must be a finite number greater than 1")

        self.head_dim = head_dim
        self.coordinate_dim = coordinate_dim
        self.rope_base = float(rope_base)
        counts = axis_pair_counts(head_dim // 2, coordinate_dim)
        self.pair_counts = counts

        axes: list[int] = []
        rates: list[float] = []
        for axis, count in enumerate(counts):
            for pair_index in range(count):
                axes.append(axis)
                rates.append(self.rope_base ** (1.0 - pair_index / count))

        self.register_buffer("axis_index", torch.tensor(axes, dtype=torch.long))
        self.register_buffer("theta", torch.tensor(rates, dtype=torch.float32))

    def forward(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        validate_coordinates(coordinates, self.coordinate_dim)
        axis_index = self.axis_index.to(device=coordinates.device)
        theta = self.theta.to(device=coordinates.device, dtype=coordinates.dtype)
        angles = coordinates[..., axis_index] * theta
        cosine_half = torch.cos(angles)
        sine_half = torch.sin(angles)
        return (
            torch.cat((cosine_half, cosine_half), dim=-1),
            torch.cat((sine_half, sine_half), dim=-1),
        )


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with coordinate-driven rotary Q/K."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        coordinate_dim: int,
        dropout: float = 0.0,
        rope_base: float = math.pi / 2.0,
    ) -> None:
        super().__init__()
        for value, name in ((d_model, "d_model"), (nhead, "nhead")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary_angles = RotaryPositionAngles(
            head_dim=self.head_dim,
            coordinate_dim=coordinate_dim,
            rope_base=rope_base,
        )

    def _split_heads(self, value: Tensor) -> Tensor:
        batch_size, tokens, _ = value.shape
        return value.view(batch_size, tokens, self.nhead, self.head_dim).transpose(1, 2)

    def forward(
        self,
        value: Tensor,
        coordinates: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if value.ndim != 3 or value.shape[-1] != self.d_model:
            raise ValueError("attention input must have shape [B, N, d_model]")
        if key_padding_mask is not None:
            if key_padding_mask.shape != value.shape[:2] or key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be Boolean with shape [B, N]")
            if bool(key_padding_mask.all(dim=1).any().item()):
                raise ValueError("an attention row cannot mask every token")

        query = self._split_heads(self.q_proj(value))
        key = self._split_heads(self.k_proj(value))
        values = self._split_heads(self.v_proj(value))
        cosine, sine = self.rotary_angles(coordinates)
        cosine = cosine.to(dtype=query.dtype).unsqueeze(1)
        sine = sine.to(dtype=query.dtype).unsqueeze(1)
        query = apply_rotary_embedding(query, cosine, sine)
        key = apply_rotary_embedding(key, cosine, sine)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, values)
        batch_size, _, tokens, _ = context.shape
        context = context.transpose(1, 2).contiguous().view(
            batch_size, tokens, self.d_model
        )
        return self.out_proj(context)


class RotaryTransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer layer whose self-attention uses RoPE."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        coordinate_dim: int,
        dim_feedforward: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RotarySelfAttention(
            d_model=d_model,
            nhead=nhead,
            coordinate_dim=coordinate_dim,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        value: Tensor,
        coordinates: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attention = self.self_attn(self.norm1(value), coordinates, key_padding_mask)
        value = value + self.dropout1(attention)
        feedforward = self.linear2(
            self.dropout_ffn(self.activation(self.linear1(self.norm2(value))))
        )
        return value + self.dropout2(feedforward)


class RotaryTransformerEncoder(nn.Module):
    """Stack rotary encoder layers while passing coordinates to each layer."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        coordinate_dim: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        self.layers = nn.ModuleList(
            RotaryTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                coordinate_dim=coordinate_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        value: Tensor,
        coordinates: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            value = layer(value, coordinates, key_padding_mask)
        return value


__all__ = [
    "RotaryPositionAngles",
    "RotarySelfAttention",
    "RotaryTransformerEncoder",
    "RotaryTransformerEncoderLayer",
    "apply_rotary_embedding",
    "axis_pair_counts",
    "rotate_half",
]
