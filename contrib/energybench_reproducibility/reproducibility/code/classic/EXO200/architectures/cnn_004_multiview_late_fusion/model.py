"""MJD CNN-004 adapted only at the EXO-200 waveform representation boundary."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from exobench.config import INPUT_FIELD, INPUT_SHAPE
from exobench.waveform_points import serialize_exo_waveform


MODEL_CONFIG = {
    "base_channels": 16,
    "stage_blocks": [2, 2, 2, 2],
    "fusion_features": 256,
    "dropout": 0.1,
}
REPRESENTATION_CONFIG = {
    "source": INPUT_FIELD,
    "adapter": "channel_major_flatten",
    "input_shape": list(INPUT_SHAPE),
    "serialized_shape": [1, math.prod(INPUT_SHAPE)],
    "views": "multiresolution_stft",
    "window": "hann",
    "n_fft": [64, 128, 256],
    "hop_length": [16, 32, 64],
    "magnitude_transform": "log1p",
    "image_size": 128,
}
TRAINING_DEFAULTS = {
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 12,
    "early_stopping_min_delta": 0.0,
    "seed": 42,
    "deterministic": False,
    "use_amp": True,
    "amp_precision": "auto",
    "num_workers": 0,
}
ARCHITECTURE_CONFIG = {
    "model": MODEL_CONFIG,
    "representation": REPRESENTATION_CONFIG,
    "training": TRAINING_DEFAULTS,
}


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class _Shared2DEncoder(nn.Module):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(inputs))
        return features.mean(dim=(-2, -1))


class MultiViewLateFusionBackbone(nn.Module):
    """Encode three STFT views independently with shared MJD CNN-004 weights."""

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
        for n_fft in REPRESENTATION_CONFIG["n_fft"]:
            self.register_buffer(
                f"hann_window_{n_fft}",
                torch.hann_window(n_fft),
                persistent=False,
            )

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
        self.fusion = nn.Sequential(
            nn.Linear(self.INPUT_VIEWS * encoded, self.fusion_features),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout),
        )
        self.output_features = self.fusion_features

    def _stft_views(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = serialize_exo_waveform(waveform)
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError("serialized waveform must have shape [B, C*T]")
        if not waveform.is_floating_point():
            raise TypeError("waveform must be floating point")

        views = []
        image_size = int(REPRESENTATION_CONFIG["image_size"])
        for n_fft, hop_length in zip(
            REPRESENTATION_CONFIG["n_fft"],
            REPRESENTATION_CONFIG["hop_length"],
        ):
            window = getattr(self, f"hann_window_{n_fft}")
            spectrum = torch.stft(
                waveform.float(),
                n_fft=n_fft,
                hop_length=hop_length,
                window=window.float(),
                return_complex=True,
            )
            magnitude = torch.log1p(spectrum.abs()).unsqueeze(1)
            views.append(
                F.interpolate(
                    magnitude,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(views, dim=1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        images = self._stft_views(waveform)
        batch_size, views, height, width = images.shape
        view_images = images.reshape(batch_size * views, 1, height, width)
        view_features = self.encoder(view_images).reshape(batch_size, views, -1)
        identified = view_features + self.view_identity
        attention = torch.softmax(self.view_attention(identified), dim=1)
        return self.fusion((identified * attention).flatten(start_dim=1))


class MultiViewLateFusionClassifier(nn.Module):
    """Return one background-positive EXO-200 classification logit per event."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = MultiViewLateFusionBackbone(**MODEL_CONFIG)
        self.head = nn.Linear(self.backbone.output_features, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(waveform)).squeeze(1)


def build_model() -> nn.Module:
    return MultiViewLateFusionClassifier()


__all__ = [
    "ARCHITECTURE_CONFIG",
    "MODEL_CONFIG",
    "MultiViewLateFusionBackbone",
    "MultiViewLateFusionClassifier",
    "REPRESENTATION_CONFIG",
    "TRAINING_DEFAULTS",
    "build_model",
]
