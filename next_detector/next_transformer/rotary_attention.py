"""Rotary positional encoding (RoPE) for the NEXT Transformer.

Unlike the encoders in positional_encoding.py, RoPE is not an additive
embedding computed once at the input. It rotates each token's Query and
Key vectors, inside every self-attention layer, by an angle derived from
that token's position. Two tokens whose relative geometric relationship
is the same produce the same rotated dot product regardless of where in
space that relationship occurs.

Standard RoPE (RoFormer, LLaMA, ...) assigns rotation angles from an
integer sequence position. NEXT tokens instead carry continuous XYZ
coordinates with no meaningful order, so this module generalizes RoPE
to three continuous axes: each attention head's channels are split into
three groups (x, y, z), and each group is rotated using that axis's
coordinate value.

Shapes:
    coordinates: [batch, number_of_tokens, 3]
    token embeddings: [batch, number_of_tokens, d_model]
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .positional_encoding import validate_coordinates


def rotate_half(
    x: torch.Tensor,
) -> torch.Tensor:
    """Swap and negate the two halves of the last dimension.

    [a, b] -> [-b, a]

    This is the standard helper used to build a 2D rotation out of
    elementwise multiplies: for a channel pair (i, i + D/2), rotating
    by angle theta is:

        x1' = x1 * cos(theta) - x2 * sin(theta)
        x2' = x2 * cos(theta) + x1 * sin(theta)

    which is exactly ``x * cos + rotate_half(x) * sin`` once cos/sin
    are duplicated across both halves.
    """

    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat(
        [-x2, x1],
        dim=-1,
    )


def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate x by the angles encoded in cos/sin.

    x, cos, and sin must share their trailing dimension (head_dim).
    cos and sin broadcast over any leading dimensions of x.
    """

    return x * cos + rotate_half(x) * sin


def _axis_pair_counts(
    pairs_total: int,
) -> list[int]:
    """Split channel pairs as evenly as possible across x, y, z.

    Any remainder is assigned to the earlier axes first, so:
        8  -> [3, 3, 2]
        9  -> [3, 3, 3]
        10 -> [4, 3, 3]
    """

    base_count = pairs_total // 3
    remainder = pairs_total % 3

    counts = [base_count, base_count, base_count]

    for axis in range(remainder):
        counts[axis] += 1

    return counts


class RotaryPositionAngles(nn.Module):
    """Compute per-token rotation angles from continuous XYZ coordinates.

    head_dim channels are split into pairs; each pair is assigned to one
    of the x, y, or z axes and given a rotation frequency:

        theta[j] = rope_base ** (1 - j / pairs_for_this_axis)

    so theta ranges from rope_base (fastest, j=0) down to
    rope_base ** (1 / pairs_for_this_axis) (slowest). Unlike NLP RoPE
    (integer sequence positions, where a rate of 1 is already
    meaningful and base only sets how slowly the coarsest channels
    rotate), NEXT coordinates are continuous and ~O(0.1) in magnitude
    after the tokenizer centers and scales them, so rope_base must
    also set the *fastest* rotation rate rather than being pinned
    at 1.
    """

    def __init__(
        self,
        head_dim: int,
        rope_base: float = 10.0,
    ) -> None:
        super().__init__()

        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError(
                "head_dim must be a positive integer"
            )

        if head_dim % 2 != 0 or head_dim < 6:
            raise ValueError(
                "head_dim must be even and at least 6 so "
                "channels can be split into rotation pairs "
                "across x, y, and z"
            )

        if (
            isinstance(rope_base, bool)
            or not isinstance(rope_base, (int, float))
            or not math.isfinite(rope_base)
            or rope_base <= 1.0
        ):
            raise ValueError(
                "rope_base must be a finite number greater than 1.0"
            )

        self.head_dim = head_dim
        self.rope_base = float(rope_base)

        pairs_total = head_dim // 2
        counts = _axis_pair_counts(pairs_total)

        axis_index_values: list[int] = []
        theta_values: list[float] = []

        for axis, count in enumerate(counts):
            for pair_index in range(count):
                axis_index_values.append(axis)
                # rope_base is the fastest (pair_index=0) rotation
                # rate, in radians per unit of (already centered and
                # scaled) coordinate. Unlike NLP RoPE, where integer
                # sequence positions make a rate of 1 already
                # meaningful, our coordinates are ~O(0.1) in
                # magnitude, so the fastest channel needs its own
                # explicit scale rather than being pinned at 1.0.
                theta_values.append(
                    rope_base
                    ** (1.0 - pair_index / count)
                )

        axis_index = torch.tensor(
            axis_index_values,
            dtype=torch.long,
        )
        theta = torch.tensor(
            theta_values,
            dtype=torch.float32,
        )

        # Non-trainable: saved in state_dict, moves with .to(device).
        self.register_buffer("axis_index", axis_index)
        self.register_buffer("theta", theta)

    def forward(
        self,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) rotation tensors, each [B, N, head_dim]."""

        validate_coordinates(coordinates)

        axis_index = self.axis_index.to(
            device=coordinates.device,
        )
        theta = self.theta.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

        # [B, N, 3] -> [B, N, pairs_total]
        selected = coordinates[..., axis_index]
        angle = selected * theta

        cos_half = torch.cos(angle)
        sin_half = torch.sin(angle)

        cos_full = torch.cat(
            [cos_half, cos_half],
            dim=-1,
        )
        sin_full = torch.cat(
            [sin_half, sin_half],
            dim=-1,
        )

        return cos_full, sin_full


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with rotary Q/K.

    Q and K are rotated by RotaryPositionAngles before the attention
    dot product; V is left unrotated. Uses four separate d_model ->
    d_model projections so the parameter count exactly matches
    torch.nn.MultiheadAttention with the same d_model/nhead.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        rope_base: float = 10.0,
    ) -> None:
        super().__init__()

        for name, value in (
            ("d_model", d_model),
            ("nhead", nhead),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead"
            )

        if (
            not math.isfinite(dropout)
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError(
                "dropout must be finite and in [0, 1)"
            )

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
            rope_base=rope_base,
        )

    def _split_heads(
        self,
        projected: torch.Tensor,
        batch_size: int,
        num_tokens: int,
    ) -> torch.Tensor:
        return (
            projected
            .view(
                batch_size,
                num_tokens,
                self.nhead,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply rotary self-attention.

        x: [B, N, d_model]
        coords: [B, N, 3]
        key_padding_mask: [B, N], True = ignore this key position
        """

        batch_size, num_tokens, _ = x.shape

        query = self._split_heads(
            self.q_proj(x), batch_size, num_tokens
        )
        key = self._split_heads(
            self.k_proj(x), batch_size, num_tokens
        )
        value = self._split_heads(
            self.v_proj(x), batch_size, num_tokens
        )

        cos, sin = self.rotary_angles(coords)

        # Broadcast the same rotation across every head.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        query = apply_rotary_embedding(query, cos, sin)
        key = apply_rotary_embedding(key, cos, sin)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(
                mask,
                float("-inf"),
            )

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, value)
        context = (
            context
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_model)
        )

        return self.out_proj(context)


class RotaryTransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer using RotarySelfAttention.

    Mirrors torch.nn.TransformerEncoderLayer(norm_first=True,
    activation="gelu") but threads coords through to self-attention
    instead of relying on an additive position embedding.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RotarySelfAttention(
            d_model=d_model,
            nhead=nhead,
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
        x: torch.Tensor,
        coords: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.self_attn(
            self.norm1(x),
            coords,
            key_padding_mask,
        )
        x = x + self.dropout1(attention_output)

        feedforward_output = self.linear2(
            self.dropout_ffn(
                self.activation(
                    self.linear1(self.norm2(x))
                )
            )
        )
        x = x + self.dropout2(feedforward_output)

        return x


class RotaryTransformerEncoder(nn.Module):
    """Stack of RotaryTransformerEncoderLayer, coords passed to each."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()

        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers <= 0
        ):
            raise ValueError(
                "num_layers must be a positive integer"
            )

        self.layers = nn.ModuleList(
            [
                RotaryTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, coords, key_padding_mask)

        return x


__all__ = [
    "RotaryPositionAngles",
    "RotarySelfAttention",
    "RotaryTransformerEncoder",
    "RotaryTransformerEncoderLayer",
    "apply_rotary_embedding",
    "rotate_half",
]