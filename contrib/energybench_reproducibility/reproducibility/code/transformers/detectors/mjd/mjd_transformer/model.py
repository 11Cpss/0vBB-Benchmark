"""Transformer models compatible with the shared ``mjdbench`` workflow."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .positional_encoding import PositionEncodingName, build_position_encoder
from .tokenization import TokenizationConfig, build_tokenizer


Task = Literal["classification", "regression"]


class MJDTransformer(nn.Module):
    """Tokenize one dense waveform and predict one event-level value.

    The shared MJD training loop supplies a waveform tensor and expects one
    raw value per event.  For classification that value is a logit; for
    regression it is the predicted energy in the target's units.
    """

    def __init__(
        self,
        *,
        task: Task,
        position_encoding: PositionEncodingName,
        tokenization_config: TokenizationConfig | None = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_frequencies: int = 6,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            task=task,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_frequencies=num_frequencies,
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

        self.tokenizer = build_tokenizer(selected_tokens)
        self.feature_dim = int(self.tokenizer.feature_dim)
        self.coordinate_dim = int(self.tokenizer.coordinate_dim)
        self.content_projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.position_encoder = build_position_encoder(
            position_encoding,
            coordinate_dim=self.coordinate_dim,
            d_model=d_model,
            num_frequencies=num_frequencies,
        )
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
        position = self.position_encoder(coordinates)
        token_embeddings = self.input_norm(content + position)
        transformed = self.transformer(
            token_embeddings,
            src_key_padding_mask=~valid_mask,
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
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_frequencies: int,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
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


class MJDTransformerClassifier(MJDTransformer):
    """Convenience classifier with the shared clean/non-clean output contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(task="classification", **kwargs)


class MJDTransformerRegressor(MJDTransformer):
    """Convenience regressor with the shared energy-output contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(task="regression", **kwargs)


__all__ = [
    "MJDTransformer",
    "MJDTransformerClassifier",
    "MJDTransformerRegressor",
    "Task",
]
