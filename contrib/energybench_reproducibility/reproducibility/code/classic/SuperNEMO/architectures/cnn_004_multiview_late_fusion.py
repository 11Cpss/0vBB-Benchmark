"""Local, source-faithful copy of Cnn004."""

# Model origin: ../NEXT/src/next_alt/models/cnn.py
# Source SHA256: 5fd7286ad4b0643e306217316588d8b3e64b9a0ece4552d7ea2191033fd8c436

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - reported by the training CLI
    raise RuntimeError(
        "Alternative NEXT models require PyTorch; activate the GPU environment"
    ) from exc


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _stage_blocks(value: Sequence[int]) -> tuple[int, int, int, int]:
    blocks = tuple(_positive_int("stage_blocks entry", item) for item in value)
    if len(blocks) != 4:
        raise ValueError("stage_blocks must contain exactly four entries")
    return blocks  # type: ignore[return-value]


def _dropout(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return probability


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _batch_tensor(
    batch: Mapping[str, Any],
    key: str,
    dimensions: int,
    channels: int,
) -> "torch.Tensor":
    if not isinstance(batch, Mapping):
        raise TypeError("forward expects a batch mapping")
    if key not in batch:
        raise KeyError(f"batch is missing required tensor {key!r}")
    tensor = batch[key]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"batch[{key!r}] must be a torch.Tensor")
    if tensor.ndim != dimensions or tensor.shape[1] != channels:
        shape = "B,C," + ",".join("spatial" for _ in range(dimensions - 2))
        raise ValueError(
            f"batch[{key!r}] must have shape ({shape}) with C={channels}; "
            f"received {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"batch[{key!r}] must have a floating-point dtype")
    return tensor


class _ResidualBlock2D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(output_channels), output_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        residual = self.skip(inputs)
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class _Shared2DEncoder(nn.Module):
    """Compact four-stage residual encoder shared across views or scales."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        stage_blocks: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        stages = []
        incoming = base_channels
        for stage_index, block_count in enumerate(stage_blocks):
            outgoing = base_channels * (2**stage_index)
            blocks = [
                _ResidualBlock2D(
                    incoming,
                    outgoing,
                    stride=1 if stage_index == 0 else 2,
                )
            ]
            blocks.extend(
                _ResidualBlock2D(outgoing, outgoing)
                for _ in range(block_count - 1)
            )
            stages.append(nn.Sequential(*blocks))
            incoming = outgoing
        self.stages = nn.Sequential(*stages)
        self.output_features = incoming

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        features = self.stages(self.stem(inputs))
        return features.mean(dim=(-2, -1))


class MultiViewLateFusionCNN(nn.Module):
    """Encode XY/XZ/YZ independently with shared weights, then fuse late.

    ``batch['projections']`` has shape ``(B, 3, 128, 128)``.  Treating the three
    projections as a view axis prevents early convolutions from confusing the
    non-aligned detector planes with ordinary RGB channels.  Learned view
    identities preserve which plane is which, while a shared attention scorer
    controls each view's contribution before ordered concatenation.
    """

    INPUT_VIEWS = 3

    def __init__(
        self,
        base_channels: int = 16,
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
        fusion_features: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_channels = _positive_int("base_channels", base_channels)
        self.stage_blocks = _stage_blocks(stage_blocks)
        self.fusion_features = _positive_int("fusion_features", fusion_features)
        self.dropout = _dropout(dropout)

        self.encoder = _Shared2DEncoder(
            input_channels=1,
            base_channels=self.base_channels,
            stage_blocks=self.stage_blocks,
        )
        encoded = self.encoder.output_features
        attention_features = max(encoded // 2, 1)
        self.view_identity = nn.Parameter(torch.empty(1, self.INPUT_VIEWS, encoded))
        nn.init.normal_(self.view_identity, mean=0.0, std=0.02)
        self.view_attention = nn.Sequential(
            nn.Linear(encoded, attention_features),
            nn.SiLU(inplace=True),
            nn.Linear(attention_features, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.INPUT_VIEWS * encoded, self.fusion_features),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_features, 1),
        )

    def forward(self, batch: Mapping[str, Any]) -> "torch.Tensor":
        images = _batch_tensor(batch, "projections", dimensions=4, channels=3)
        batch_size, views, height, width = images.shape
        view_images = images.reshape(batch_size * views, 1, height, width)
        view_features = self.encoder(view_images).reshape(batch_size, views, -1)
        identified = view_features + self.view_identity
        attention = torch.softmax(self.view_attention(identified), dim=1)
        fused = (identified * attention).flatten(start_dim=1)
        return self.classifier(fused).squeeze(1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "base_channels": self.base_channels,
            "stage_blocks": list(self.stage_blocks),
            "fusion_features": self.fusion_features,
            "dropout": self.dropout,
        }


__all__ = ["MultiViewLateFusionCNN"]

