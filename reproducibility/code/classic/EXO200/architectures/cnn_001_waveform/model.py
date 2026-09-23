"""The EXO-200 classifier uses the MJD CNN-001 definition."""

from __future__ import annotations

from torch import nn

from exobench.config import INPUT_SHAPE, NUM_CLASSES, ModelConfig
from exobench.models import WaveformCNNClassifier


_MODEL_DEFAULTS = ModelConfig()


def build_model(
    *,
    input_channels: int = INPUT_SHAPE[0],
    base_channels: int = _MODEL_DEFAULTS.base_channels,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    return WaveformCNNClassifier(
        input_channels=input_channels,
        base_channels=base_channels,
        num_classes=num_classes,
    )


__all__ = ["build_model"]
