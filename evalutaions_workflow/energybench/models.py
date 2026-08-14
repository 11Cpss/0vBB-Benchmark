"""Small reference CNNs for the standard three-view NEXT input.

These models are examples, not framework requirements.  Any ``nn.Module``
that maps ``[batch, 3, height, width]`` inputs to one scalar per event can be
passed directly to the training and evaluation functions.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _feature_extractor(base_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.AdaptiveAvgPool2d(output_size=1),
    )


class SimpleCNNClassifier(nn.Module):
    """Return one raw positive-class logit per projected event."""

    def __init__(self, base_channels: int = 8) -> None:
        super().__init__()
        self.base_channels = _validate_base_channels(base_channels)
        # Keep these public names compatible with the useful legacy baseline.
        self.features = _feature_extractor(self.base_channels)
        self.classifier = nn.Linear(self.base_channels * 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map ``[B, 3, H, W]`` images to logits with shape ``[B]``."""

        _validate_inputs(inputs)
        features = self.features(inputs).flatten(start_dim=1)
        return self.classifier(features).squeeze(1)

    def config_dict(self) -> dict[str, Any]:
        """Return constructor parameters suitable for a checkpoint."""

        return {"base_channels": self.base_channels}


class SimpleCNNRegressor(nn.Module):
    """Return one predicted physical energy in MeV per projected event."""

    def __init__(self, base_channels: int = 8) -> None:
        super().__init__()
        self.base_channels = _validate_base_channels(base_channels)
        self.features = _feature_extractor(self.base_channels)
        self.regressor = nn.Linear(self.base_channels * 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map ``[B, 3, H, W]`` images to MeV predictions with shape ``[B]``."""

        _validate_inputs(inputs)
        features = self.features(inputs).flatten(start_dim=1)
        return self.regressor(features).squeeze(1)

    def config_dict(self) -> dict[str, Any]:
        """Return constructor parameters suitable for a checkpoint."""

        return {"base_channels": self.base_channels}


# Descriptive legacy names remain useful when loading the simple two-conv
# baseline checkpoint, but they introduce no separate implementation path.
SimpleNextCNN = SimpleCNNClassifier
SimpleNextEnergyRegressor = SimpleCNNRegressor
SimpleClassifier = SimpleCNNClassifier
SimpleRegressor = SimpleCNNRegressor


def _validate_base_channels(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("base_channels must be a positive integer")
    return int(value)


def _validate_inputs(inputs: torch.Tensor) -> None:
    if inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError("inputs must have shape [batch, 3, height, width]")
    if not inputs.is_floating_point():
        raise TypeError("inputs must be a floating-point tensor")


__all__ = [
    "SimpleCNNClassifier",
    "SimpleCNNRegressor",
    "SimpleClassifier",
    "SimpleNextCNN",
    "SimpleNextEnergyRegressor",
    "SimpleRegressor",
]
