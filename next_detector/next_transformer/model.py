"""Transformer classifier for NEXT detector events.

The model receives a dictionary produced by NEXTTokenBuilder:

    coords:   [batch, tokens, 3]
    features: [batch, tokens, feature_dim]
    mask:     [batch, tokens]

It returns one raw classification logit per event:

    logits: [batch]
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from .positional_encoding import (
    PositionEncodingName,
    build_position_encoder,
)
from .rotary_attention import RotaryTransformerEncoder


class NEXTTransformerClassifier(nn.Module):
    """Classify a complete NEXT event using token self-attention."""

    def __init__(
        self,
        *,
        position_encoding: PositionEncodingName,
        feature_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_frequencies: int = 6,
        rope_base: float = 10.0,
    ) -> None:
        super().__init__()

        self._validate_configuration(
            position_encoding=position_encoding,
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_frequencies=num_frequencies,
            rope_base=rope_base,
        )

        # Save constructor settings for checkpoints and reproducibility.
        self.position_encoding_name = (
            position_encoding
        )
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = (
            dim_feedforward
        )
        self.dropout = dropout
        self.num_frequencies = (
            num_frequencies
        )
        self.rope_base = rope_base

        # Convert each token's content features from feature_dim
        # values into the Transformer's d_model dimensions.
        self.content_projection = nn.Sequential(
            nn.Linear(
                feature_dim,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        # Normalize after combining content and position.
        self.input_norm = nn.LayerNorm(
            d_model
        )

        self._uses_rotary_attention = (
            position_encoding == "rope"
        )

        if self._uses_rotary_attention:
            # Rotary attention has no additive position embedding:
            # position enters by rotating Q/K inside every
            # self-attention layer instead.
            self.position_encoder = None

            self.transformer = RotaryTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rope_base=rope_base,
            )
        else:
            # Convert each token's XYZ coordinates into a d_model
            # position embedding.
            self.position_encoder = (
                build_position_encoder(
                    position_encoding,
                    d_model=d_model,
                    num_frequencies=num_frequencies,
                )
            )

            encoder_layer = (
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )

            self.transformer = (
                nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers,

                    # Avoid nested-tensor warnings with norm_first=True.
                    enable_nested_tensor=False,
                )
            )

        # Normalize the pooled event representation.
        self.output_norm = nn.LayerNorm(
            d_model
        )

        # Convert one event representation into one raw logit.
        self.classification_head = nn.Sequential(
            nn.Linear(
                d_model,
                d_model // 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                d_model // 2,
                1,
            ),
        )

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Convert padded event tokens into one raw logit per event."""

        (
            coords,
            features,
            valid_mask,
        ) = self._validate_inputs(
            inputs
        )

        # Content tells the model what is present at each token.
        content_embedding = (
            self.content_projection(
                features
            )
        )

        # Our mask:
        #     True  = real token
        #     False = padding
        #
        # PyTorch's padding mask:
        #     True  = ignore this token
        #     False = use this token
        #
        # Therefore, it must be inverted.
        padding_mask = ~valid_mask

        if self._uses_rotary_attention:
            # Position is not added here. It enters by rotating Q/K
            # inside every self-attention layer instead.
            token_embeddings = self.input_norm(
                content_embedding
            )

            transformed_tokens = (
                self.transformer(
                    token_embeddings,
                    coords,
                    key_padding_mask=padding_mask,
                )
            )
        else:
            # Position tells the model where each token is located.
            position_embedding = (
                self.position_encoder(
                    coords
                )
            )

            # Both tensors have shape:
            #
            #     [batch, tokens, d_model]
            #
            # Adding them creates one complete representation
            # per token.
            token_embeddings = self.input_norm(
                content_embedding
                + position_embedding
            )

            transformed_tokens = (
                self.transformer(
                    token_embeddings,
                    src_key_padding_mask=padding_mask,
                )
            )

        # Convert the Boolean mask into numeric zeros and ones.
        numeric_mask = (
            valid_mask
            .unsqueeze(-1)
            .to(transformed_tokens.dtype)
        )

        # Zero out padding embeddings before pooling.
        event_representation = (
            transformed_tokens
            * numeric_mask
        ).sum(dim=1)

        # Count the number of real tokens per event.
        number_of_valid_tokens = (
            numeric_mask
            .sum(dim=1)
            .clamp_min(1.0)
        )

        # Masked mean pooling:
        #
        # sum of real token embeddings
        # --------------------------------
        # number of real tokens
        event_representation = (
            event_representation
            / number_of_valid_tokens
        )

        event_representation = (
            self.output_norm(
                event_representation
            )
        )

        logits = (
            self.classification_head(
                event_representation
            )
            .squeeze(-1)
        )

        return logits

    def _validate_inputs(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Validate the nested EnergyBench model inputs."""

        if not isinstance(inputs, Mapping):
            raise TypeError(
                "inputs must be a mapping containing "
                "coords, features, and mask"
            )

        required_keys = {
            "coords",
            "features",
            "mask",
        }

        missing_keys = (
            required_keys
            - set(inputs)
        )

        if missing_keys:
            raise KeyError(
                "missing Transformer input keys: "
                + ", ".join(
                    sorted(missing_keys)
                )
            )

        coords = inputs["coords"]
        features = inputs["features"]
        valid_mask = inputs["mask"]

        if (
            coords.ndim != 3
            or coords.shape[-1] != 3
        ):
            raise ValueError(
                "coords must have shape "
                "[batch, tokens, 3]"
            )

        if (
            features.ndim != 3
            or features.shape[-1]
            != self.feature_dim
        ):
            raise ValueError(
                "features must have shape "
                f"[batch, tokens, {self.feature_dim}]"
            )

        if valid_mask.ndim != 2:
            raise ValueError(
                "mask must have shape "
                "[batch, tokens]"
            )

        if (
            coords.shape[:2]
            != features.shape[:2]
            or coords.shape[:2]
            != valid_mask.shape
        ):
            raise ValueError(
                "coords, features, and mask must "
                "share batch and token dimensions"
            )

        if not coords.is_floating_point():
            raise TypeError(
                "coords must be floating point"
            )

        if not features.is_floating_point():
            raise TypeError(
                "features must be floating point"
            )

        if valid_mask.dtype != torch.bool:
            raise TypeError(
                "mask must have Boolean dtype"
            )

        if not torch.isfinite(coords).all():
            raise ValueError(
                "coords contain non-finite values"
            )

        if not torch.isfinite(features).all():
            raise ValueError(
                "features contain non-finite values"
            )

        # Every event must have at least one real token.
        if (
            ~valid_mask.any(dim=1)
        ).any():
            raise ValueError(
                "every event must contain "
                "at least one valid token"
            )

        return (
            coords,
            features,
            valid_mask,
        )

    def config_dict(self) -> dict[str, Any]:
        """Return model settings for the EnergyBench checkpoint."""

        return {
            "position_encoding": (
                self.position_encoding_name
            ),
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": (
                self.dim_feedforward
            ),
            "dropout": self.dropout,
            "num_frequencies": (
                self.num_frequencies
            ),
            "rope_base": self.rope_base,
        }

    @staticmethod
    def _validate_configuration(
        *,
        position_encoding: str,
        feature_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_frequencies: int,
        rope_base: float,
    ) -> None:
        """Validate model hyperparameters before creating layers."""

        integer_settings = {
            "feature_dim": feature_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": (
                dim_feedforward
            ),
            "num_frequencies": (
                num_frequencies
            ),
        }

        for name, value in (
            integer_settings.items()
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be "
                    "a positive integer"
                )

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible "
                "by nhead"
            )

        if (
            not math.isfinite(dropout)
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError(
                "dropout must be finite "
                "and in [0, 1)"
            )

        if (
            isinstance(rope_base, bool)
            or not isinstance(rope_base, (int, float))
            or not math.isfinite(rope_base)
            or rope_base <= 1.0
        ):
            raise ValueError(
                "rope_base must be finite "
                "and greater than 1.0"
            )

        if position_encoding == "rope":
            head_dim = d_model // nhead

            if head_dim % 2 != 0 or head_dim < 6:
                raise ValueError(
                    "position_encoding='rope' requires "
                    "d_model // nhead (head_dim) to be even "
                    "and at least 6, so channels can be split "
                    "across x, y, z rotation groups; got "
                    f"head_dim={head_dim} for d_model={d_model}, "
                    f"nhead={nhead}"
                )


__all__ = [
    "NEXTTransformerClassifier",
]