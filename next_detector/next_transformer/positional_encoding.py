"""Positional encodings for the NEXT Transformer.

A Transformer does not automatically understand that each token has a
physical 3D location. These modules convert XYZ coordinates into vectors
with the same dimension as the token content embeddings.

Input shape:
    [batch, number_of_tokens, 3]

Output shape:
    [batch, number_of_tokens, d_model]
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


PositionEncodingName = Literal[
    "coordinate_mlp",
    "fourier_xyz",
    "rope",
]


def validate_coordinates(
    coordinates: torch.Tensor,
) -> None:
    """Validate a batch of XYZ token coordinates."""

    if not isinstance(
        coordinates,
        torch.Tensor,
    ):
        raise TypeError(
            "coordinates must be a PyTorch tensor"
        )

    if (
        coordinates.ndim != 3
        or coordinates.shape[-1] != 3
    ):
        raise ValueError(
            "coordinates must have shape "
            "[batch, number_of_tokens, 3]"
        )

    if not coordinates.is_floating_point():
        raise TypeError(
            "coordinates must be floating point"
        )

    if not torch.isfinite(
        coordinates
    ).all():
        raise ValueError(
            "coordinates must contain only finite values"
        )


class CoordinateMLPEncoding(nn.Module):
    """Learn a position representation directly from XYZ coordinates.

    Each coordinate triplet:

        [x, y, z]

    is passed through a small neural network:

        3
        -> d_model
        -> GELU
        -> d_model
    """

    def __init__(
        self,
        d_model: int,
    ) -> None:
        super().__init__()

        if (
            isinstance(d_model, bool)
            or not isinstance(d_model, int)
            or d_model <= 0
        ):
            raise ValueError(
                "d_model must be a positive integer"
            )

        self.d_model = d_model

        self.network = nn.Sequential(
            nn.Linear(
                in_features=3,
                out_features=d_model,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=d_model,
                out_features=d_model,
            ),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Convert XYZ coordinates into learned position embeddings."""

        validate_coordinates(
            coordinates
        )

        return self.network(
            coordinates
        )


class FourierXYZEncoding(nn.Module):
    """Represent XYZ using several sine and cosine frequencies.

    Fourier features allow the model to describe spatial variation at
    multiple physical scales.

    Low-frequency waves capture broad spatial structure.
    High-frequency waves capture finer spatial differences.
    """

    def __init__(
        self,
        d_model: int,
        num_frequencies: int = 6,
    ) -> None:
        super().__init__()

        if (
            isinstance(d_model, bool)
            or not isinstance(d_model, int)
            or d_model <= 0
        ):
            raise ValueError(
                "d_model must be a positive integer"
            )

        if (
            isinstance(num_frequencies, bool)
            or not isinstance(
                num_frequencies,
                int,
            )
            or num_frequencies <= 0
        ):
            raise ValueError(
                "num_frequencies must be "
                "a positive integer"
            )

        self.d_model = d_model
        self.num_frequencies = (
            num_frequencies
        )

        frequencies = (
            2.0
            ** torch.arange(
                num_frequencies,
                dtype=torch.float32,
            )
        )

        # A registered buffer:
        # - is saved in the model state
        # - moves to the GPU with the model
        # - is not a trainable parameter
        self.register_buffer(
            "frequencies",
            frequencies,
        )

        # Original XYZ contributes 3 values.
        #
        # For each frequency, each of X, Y and Z contributes:
        #     one sine value
        #     one cosine value
        #
        # Therefore:
        #     3 + (3 * frequencies * 2)
        encoded_dimension = (
            3
            + 6 * num_frequencies
        )

        self.network = nn.Sequential(
            nn.Linear(
                in_features=encoded_dimension,
                out_features=d_model,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=d_model,
                out_features=d_model,
            ),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Convert XYZ into multiscale Fourier position embeddings."""

        validate_coordinates(
            coordinates
        )

        # Match the coordinate tensor's device and numerical dtype.
        frequencies = self.frequencies.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

        # Before:
        #     coordinates [B, N, 3]
        #
        # After unsqueeze:
        #     [B, N, 3, 1]
        #
        # After multiplying by frequencies:
        #     [B, N, 3, F]
        angles = (
            math.pi
            * coordinates.unsqueeze(-1)
            * frequencies
        )

        sine_features = torch.sin(
            angles
        ).flatten(
            start_dim=-2
        )

        cosine_features = torch.cos(
            angles
        ).flatten(
            start_dim=-2
        )

        # Combine:
        # - original XYZ
        # - sine features
        # - cosine features
        encoded_coordinates = torch.cat(
            [
                coordinates,
                sine_features,
                cosine_features,
            ],
            dim=-1,
        )

        return self.network(
            encoded_coordinates
        )


def build_position_encoder(
    name: PositionEncodingName,
    *,
    d_model: int,
    num_frequencies: int = 6,
) -> nn.Module:
    """Construct the requested positional encoder.

    This factory keeps model.py simple. It converts a configuration name
    into the corresponding PyTorch module.
    """

    if name == "coordinate_mlp":
        return CoordinateMLPEncoding(
            d_model=d_model,
        )

    if name == "fourier_xyz":
        return FourierXYZEncoding(
            d_model=d_model,
            num_frequencies=num_frequencies,
        )

    if name == "rope":
        raise ValueError(
            "build_position_encoder does not support 'rope'; "
            "rotary attention rotates Q/K inside every "
            "self-attention layer rather than producing an "
            "additive position embedding, so it is constructed "
            "directly inside NEXTTransformerClassifier"
        )

    raise ValueError(
        "position encoding must be "
        "'coordinate_mlp', 'fourier_xyz', or 'rope'"
    )


__all__ = [
    "CoordinateMLPEncoding",
    "FourierXYZEncoding",
    "PositionEncodingName",
    "build_position_encoder",
]