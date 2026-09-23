"""The paired CNN-001 MJD models share this exact backbone definition."""

from __future__ import annotations

from torch import nn

from mjdbench.models import WaveformCNNClassifier, WaveformCNNRegressor


def build_model(task: str, base_channels: int = 16) -> nn.Module:
    if task == "classification":
        return WaveformCNNClassifier(base_channels=base_channels)
    if task == "regression":
        return WaveformCNNRegressor(base_channels=base_channels)
    raise ValueError("task must be 'classification' or 'regression'")


__all__ = ["build_model"]
