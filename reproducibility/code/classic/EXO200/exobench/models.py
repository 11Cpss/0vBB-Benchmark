"""MJD CNN-001 adapted only for EXO-200's 226 waveform channels."""

from __future__ import annotations

import torch
from torch import nn

from .config import INPUT_SHAPE, NUM_CLASSES, ModelConfig


_MODEL_DEFAULTS = ModelConfig()


class WaveformCNNBackbone(nn.Module):
    """The MJD CNN-001 feature extractor with configurable input channels."""

    def __init__(
        self,
        input_channels: int = INPUT_SHAPE[0],
        base_channels: int = _MODEL_DEFAULTS.base_channels,
    ) -> None:
        super().__init__()
        if int(input_channels) <= 0:
            raise ValueError("input_channels must be positive")
        if int(base_channels) <= 0:
            raise ValueError("base_channels must be positive")
        incoming = int(input_channels)
        channels = int(base_channels)
        self.input_channels = incoming
        self.base_channels = channels
        self.network = nn.Sequential(
            nn.Conv1d(incoming, channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels * 2, channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.output_features = channels * 4

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2 and self.input_channels == 1:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != self.input_channels:
            raise ValueError(
                f"waveform must have shape [B, {self.input_channels}, L]"
            )
        if not waveform.is_floating_point():
            raise TypeError("waveform must be floating point")
        return self.network(waveform)


class WaveformCNNClassifier(nn.Module):
    """Return one raw logit whose positive class is background (label 1)."""

    def __init__(
        self,
        input_channels: int = INPUT_SHAPE[0],
        base_channels: int = _MODEL_DEFAULTS.base_channels,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if int(num_classes) != NUM_CLASSES:
            raise ValueError("EXO-200 classification requires num_classes=2")
        self.backbone = WaveformCNNBackbone(input_channels, base_channels)
        self.head = nn.Linear(self.backbone.output_features, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(waveform)).squeeze(1)


__all__ = ["WaveformCNNBackbone", "WaveformCNNClassifier"]
