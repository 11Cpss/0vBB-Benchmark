"""Paired 1D CNN models with one shared waveform architecture."""

from __future__ import annotations

import torch
from torch import nn


class WaveformCNNBackbone(nn.Module):
    """Feature extractor used unchanged by both MJD tasks."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if int(base_channels) <= 0:
            raise ValueError("base_channels must be positive")
        channels = int(base_channels)
        self.base_channels = channels
        self.network = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, stride=2, padding=4),
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
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [B, L] or [B, 1, L]")
        if not waveform.is_floating_point():
            raise TypeError("waveform must be floating point")
        return self.network(waveform)


class WaveformCNNClassifier(nn.Module):
    """Predict one clean-versus-non-clean logit from a waveform."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.backbone = WaveformCNNBackbone(base_channels)
        self.head = nn.Linear(self.backbone.output_features, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(waveform)).squeeze(1)


class WaveformCNNRegressor(nn.Module):
    """Predict physical energy in keV from one clean-event waveform."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.backbone = WaveformCNNBackbone(base_channels)
        self.head = nn.Linear(self.backbone.output_features, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(waveform)).squeeze(1)


__all__ = [
    "WaveformCNNBackbone",
    "WaveformCNNClassifier",
    "WaveformCNNRegressor",
]
