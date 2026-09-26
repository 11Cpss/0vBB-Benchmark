"""Transformer over padded SuperNEMO tracker-hit tokens.

This is the shared architecture for the coordinate-MLP and Fourier-XYZ
experiments.  The two variants differ only in their position encoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Optional

import torch
from torch import Tensor, nn

# Source model SHA256:
# 7d7789a2f5a4eb8fb144648951be4fd2e3e214c4d9e1b99a3327dd8ab6147686
# Source position-encoding SHA256:
# 51d1c20a103edc8e1b3fa3d9f8619a6afafdea4d39fd1b2360320ce57f35e5fc

PositionEncoding = Literal["coordinate_mlp", "fourier_xyz"]
Pooling = Literal["masked_mean"]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dropout(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return probability


def _tensor_invariant(condition: Tensor, message: str) -> None:
    """Check data invariants without synchronizing every valid CUDA batch."""

    if condition.device.type == "cuda":
        assert_async = getattr(torch, "_assert_async", None)
        if assert_async is not None:
            assert_async(condition, message)
        elif not bool(condition):
            raise ValueError(message)
    elif not bool(condition):
        raise ValueError(message)


class _CoordinateMLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coords: Tensor) -> Tensor:
        return self.network(coords)


class _FourierXYZ(nn.Module):
    def __init__(self, d_model: int, num_frequencies: int) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_frequencies, dtype=torch.float32),
        )
        encoded_dim = 3 + 6 * num_frequencies
        self.network = nn.Sequential(
            nn.Linear(encoded_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coords: Tensor) -> Tensor:
        frequencies = self.frequencies.to(device=coords.device, dtype=coords.dtype)
        angles = math.pi * coords.unsqueeze(-1) * frequencies
        encoded = torch.cat(
            (
                coords,
                torch.sin(angles).flatten(start_dim=-2),
                torch.cos(angles).flatten(start_dim=-2),
            ),
            dim=-1,
        )
        return self.network(encoded)


class TrackerHitTransformer(nn.Module):
    """Map one padded 3-D tracker-hit sequence to one scalar per event."""

    def __init__(
        self,
        *,
        position_encoding: PositionEncoding,
        feature_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_frequencies: int = 6,
        pooling: Pooling = "masked_mean",
    ) -> None:
        super().__init__()
        feature_dim = _positive_int("feature_dim", feature_dim)
        d_model = _positive_int("d_model", d_model)
        nhead = _positive_int("nhead", nhead)
        num_layers = _positive_int("num_layers", num_layers)
        dim_feedforward = _positive_int("dim_feedforward", dim_feedforward)
        num_frequencies = _positive_int("num_frequencies", num_frequencies)
        dropout = _dropout(dropout)
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if position_encoding not in {"coordinate_mlp", "fourier_xyz"}:
            raise ValueError(
                "position_encoding must be 'coordinate_mlp' or 'fourier_xyz'"
            )
        if pooling != "masked_mean":
            raise ValueError("pooling must be 'masked_mean'")

        self.position_encoding = position_encoding
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_frequencies = num_frequencies
        self.pooling = pooling

        self.content_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if position_encoding == "coordinate_mlp":
            self.position_encoder: nn.Module = _CoordinateMLP(d_model)
        else:
            self.position_encoder = _FourierXYZ(d_model, num_frequencies)
        self.input_norm = nn.LayerNorm(d_model)

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
        self.scalar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _unpack_inputs(
        self,
        coords_or_batch: Tensor | Mapping[str, Any],
        features: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(coords_or_batch, Mapping):
            if features is not None or mask is not None:
                raise TypeError(
                    "features and mask must be omitted when the first argument is a batch"
                )
            missing = {"coords", "features", "mask"} - set(coords_or_batch)
            if missing:
                raise KeyError(
                    "missing Transformer input keys: " + ", ".join(sorted(missing))
                )
            coords = coords_or_batch["coords"]
            features = coords_or_batch["features"]
            mask = coords_or_batch["mask"]
        else:
            coords = coords_or_batch

        if not isinstance(coords, Tensor):
            raise TypeError("coords must be a torch.Tensor")
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if not isinstance(mask, Tensor):
            raise TypeError("mask must be a torch.Tensor")
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [batch, tokens, 3]")
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [batch, tokens, {self.feature_dim}]"
            )
        if mask.ndim != 2:
            raise ValueError("mask must have shape [batch, tokens]")
        if coords.shape[:2] != features.shape[:2] or coords.shape[:2] != mask.shape:
            raise ValueError(
                "coords, features, and mask must share batch and token dimensions"
            )
        if coords.shape[0] < 1 or coords.shape[1] < 1:
            raise ValueError("input must contain at least one event and one token slot")
        if not coords.is_floating_point():
            raise TypeError("coords must have a floating-point dtype")
        if not features.is_floating_point():
            raise TypeError("features must have a floating-point dtype")
        if coords.dtype != features.dtype:
            raise TypeError("coords and features must have the same dtype")
        if mask.dtype != torch.bool:
            raise TypeError("mask must have Boolean dtype")
        if coords.device != features.device or coords.device != mask.device:
            raise ValueError("coords, features, and mask must be on the same device")

        _tensor_invariant(torch.isfinite(coords).all(), "coords contain non-finite values")
        _tensor_invariant(
            torch.isfinite(features).all(), "features contain non-finite values"
        )
        _tensor_invariant(
            mask.any(dim=1).all(),
            "every event must contain at least one valid token",
        )
        return coords, features, mask

    def forward(
        self,
        coords: Tensor | Mapping[str, Any],
        features: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        coords, features, mask = self._unpack_inputs(coords, features, mask)
        tokens = self.input_norm(
            self.content_projection(features) + self.position_encoder(coords)
        )
        tokens = self.transformer(tokens, src_key_padding_mask=~mask)
        numeric_mask = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        pooled = (tokens * numeric_mask).sum(dim=1) / numeric_mask.sum(dim=1)
        return self.scalar_head(self.output_norm(pooled)).squeeze(-1)

    def config_dict(self) -> dict[str, Any]:
        """Return every constructor setting needed to reproduce the model."""

        return {
            "position_encoding": self.position_encoding,
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "num_frequencies": self.num_frequencies,
            "pooling": self.pooling,
        }


__all__ = ["TrackerHitTransformer"]
