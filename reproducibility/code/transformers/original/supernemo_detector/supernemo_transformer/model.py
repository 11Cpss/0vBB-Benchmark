"""SuperNEMO Transformer classifier compatible with SuperNEMOBench."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .positional_encoding import PositionEncodingName, build_position_encoder


class SuperNEMOTransformerClassifier(nn.Module):
    """Return one 2nu-directed raw logit for each padded tracker-token event."""

    def __init__(
        self,
        *,
        position_encoding: PositionEncodingName,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_frequencies: int = 6,
        pooling: str = "masked_mean",
    ) -> None:
        super().__init__()
        for name, value in {
            "feature_dim": feature_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "num_frequencies": num_frequencies,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if pooling != "masked_mean":
            raise ValueError("pooling must be 'masked_mean'")

        self.position_encoding_name = position_encoding
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = float(dropout)
        self.num_frequencies = num_frequencies
        self.pooling = pooling
        self.content_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.position_encoder = build_position_encoder(
            position_encoding,
            d_model=d_model,
            num_frequencies=num_frequencies,
        )
        self.input_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)
        hidden = max(1, d_model // 2)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        coordinates, features, valid = self._validate_inputs(inputs)
        embeddings = self.input_norm(
            self.content_projection(features) + self.position_encoder(coordinates)
        )
        transformed = self.transformer(
            embeddings,
            src_key_padding_mask=~valid,
        )
        numeric_mask = valid.unsqueeze(-1).to(transformed.dtype)
        pooled = (transformed * numeric_mask).sum(dim=1)
        pooled = pooled / numeric_mask.sum(dim=1).clamp_min(1.0)
        return self.classification_head(self.output_norm(pooled)).squeeze(-1)

    def _validate_inputs(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping")
        if set(inputs) != {"coords", "features", "mask"}:
            raise ValueError("inputs must contain exactly coords, features, and mask")
        coordinates, features, valid = (
            inputs["coords"],
            inputs["features"],
            inputs["mask"],
        )
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("coords must have shape [batch, tokens, 3]")
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [batch, tokens, {self.feature_dim}]"
            )
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("mask must be Boolean with shape [batch, tokens]")
        if coordinates.shape[:2] != features.shape[:2] or valid.shape != coordinates.shape[:2]:
            raise ValueError("coords, features, and mask must share batch/token dimensions")
        if not coordinates.is_floating_point() or not features.is_floating_point():
            raise TypeError("coords and features must be floating point")
        if not bool(torch.isfinite(coordinates).all().item()) or not bool(
            torch.isfinite(features).all().item()
        ):
            raise ValueError("model inputs must contain only finite values")
        if not bool(valid.any(dim=1).all().item()):
            raise ValueError("every event must contain at least one valid token")
        return coordinates, features, valid

    def config_dict(self) -> dict[str, Any]:
        return {
            "task": "classification",
            "position_encoding": self.position_encoding_name,
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "num_frequencies": self.num_frequencies,
            "pooling": self.pooling,
            "positive_class": "2nu",
        }


__all__ = ["SuperNEMOTransformerClassifier"]
