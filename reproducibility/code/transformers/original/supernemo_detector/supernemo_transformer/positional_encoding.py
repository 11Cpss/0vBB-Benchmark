"""Physical XYZ encodings for SuperNEMO tracker tokens."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


PositionEncodingName = Literal["coordinate_mlp", "fourier_xyz"]


def _validate_coordinates(coordinates: torch.Tensor) -> None:
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("coordinates must be a PyTorch tensor")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [batch, tokens, 3]")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must be floating point")
    if not bool(torch.isfinite(coordinates).all().item()):
        raise ValueError("coordinates must contain only finite values")


class CoordinateMLPEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        self.network = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        _validate_coordinates(coordinates)
        return self.network(coordinates)


class FourierXYZEncoding(nn.Module):
    def __init__(self, d_model: int, num_frequencies: int = 6) -> None:
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if (
            isinstance(num_frequencies, bool)
            or not isinstance(num_frequencies, int)
            or num_frequencies <= 0
        ):
            raise ValueError("num_frequencies must be a positive integer")
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        )
        encoded_dim = 3 + 6 * num_frequencies
        self.network = nn.Sequential(
            nn.Linear(encoded_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        _validate_coordinates(coordinates)
        frequencies = self.frequencies.to(
            device=coordinates.device, dtype=coordinates.dtype
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
    d_model: int,
    num_frequencies: int = 6,
) -> nn.Module:
    if name == "coordinate_mlp":
        return CoordinateMLPEncoding(d_model)
    if name == "fourier_xyz":
        return FourierXYZEncoding(d_model, num_frequencies)
    raise ValueError("position encoding must be 'coordinate_mlp' or 'fourier_xyz'")


__all__ = [
    "CoordinateMLPEncoding",
    "FourierXYZEncoding",
    "PositionEncodingName",
    "build_position_encoder",
]
