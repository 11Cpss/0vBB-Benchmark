"""Coordinate encodings for EXO-200 sensor-aware tokens."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


PositionEncodingName = Literal["coordinate_mlp", "fourier_coordinates"]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_coordinates(coordinates: Tensor, coordinate_dim: int) -> None:
    if not isinstance(coordinates, Tensor):
        raise TypeError("coordinates must be a PyTorch tensor")
    if coordinates.ndim != 3 or coordinates.shape[-1] != coordinate_dim:
        raise ValueError(
            f"coordinates must have shape [batch, tokens, {coordinate_dim}]"
        )
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must be floating point")
    if not bool(torch.isfinite(coordinates).all().item()):
        raise ValueError("coordinates must contain only finite values")


class CoordinateMLPEncoding(nn.Module):
    """Learn a nonlinear embedding of physical token coordinates."""

    def __init__(self, coordinate_dim: int, d_model: int) -> None:
        super().__init__()
        self.coordinate_dim = _positive_integer(coordinate_dim, "coordinate_dim")
        self.d_model = _positive_integer(d_model, "d_model")
        self.network = nn.Sequential(
            nn.Linear(self.coordinate_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        validate_coordinates(coordinates, self.coordinate_dim)
        return self.network(coordinates)


class FourierCoordinateEncoding(nn.Module):
    """Embed continuous and indicator coordinates with Fourier features."""

    def __init__(
        self,
        coordinate_dim: int,
        d_model: int,
        num_frequencies: int = 6,
    ) -> None:
        super().__init__()
        self.coordinate_dim = _positive_integer(coordinate_dim, "coordinate_dim")
        self.d_model = _positive_integer(d_model, "d_model")
        self.num_frequencies = _positive_integer(num_frequencies, "num_frequencies")
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32),
        )
        encoded_dim = self.coordinate_dim * (1 + 2 * self.num_frequencies)
        self.network = nn.Sequential(
            nn.Linear(encoded_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        validate_coordinates(coordinates, self.coordinate_dim)
        frequencies = self.frequencies.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        angles = math.pi * coordinates.unsqueeze(-1) * frequencies
        encoded = torch.cat(
            (
                coordinates,
                torch.sin(angles).flatten(start_dim=-2),
                torch.cos(angles).flatten(start_dim=-2),
            ),
            dim=-1,
        )
        return self.network(encoded)


def build_position_encoder(
    name: PositionEncodingName,
    *,
    coordinate_dim: int,
    d_model: int,
    num_frequencies: int = 6,
) -> nn.Module:
    if name == "coordinate_mlp":
        return CoordinateMLPEncoding(coordinate_dim, d_model)
    if name == "fourier_coordinates":
        return FourierCoordinateEncoding(
            coordinate_dim,
            d_model,
            num_frequencies,
        )
    raise ValueError(
        "position encoding must be 'coordinate_mlp' or 'fourier_coordinates'"
    )


__all__ = [
    "CoordinateMLPEncoding",
    "FourierCoordinateEncoding",
    "PositionEncodingName",
    "build_position_encoder",
    "validate_coordinates",
]
