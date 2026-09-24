"""Sensor-aware tokenization for EXO-200 v1 waveforms.

The shared :mod:`exobench` loader supplies baseline-subtracted, event-normalized
waveforms with shape ``[batch, 226, 300]``.  Tokenization stays inside the
Transformer so every architecture receives the same shared preprocessing and
run-level split.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TokenizationName = Literal[
    "raw_patches",
    "pulse_entities",
    "segment_summary",
]

INPUT_CHANNELS = 226
INPUT_SAMPLES = 300

# name, inclusive channel start, exclusive channel stop, detector side,
# sensor-family index (U, V, APD).
SENSOR_BLOCKS = (
    ("positive_u", 0, 38, 1.0, 0),
    ("positive_v", 38, 76, 1.0, 1),
    ("negative_u", 76, 114, -1.0, 0),
    ("negative_v", 114, 152, -1.0, 1),
    ("positive_apd", 152, 189, 1.0, 2),
    ("negative_apd", 189, 226, -1.0, 2),
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class TokenizationConfig:
    """Configuration for one EXO-200 sensor-aware representation."""

    tokenization: TokenizationName = "segment_summary"
    channel_regions: int = 7
    time_regions: int = 12
    uniform_entity_fraction: float = 0.5
    entity_context_size: int = 9

    def __post_init__(self) -> None:
        if self.tokenization not in {
            "raw_patches",
            "pulse_entities",
            "segment_summary",
        }:
            raise ValueError(
                "tokenization must be 'raw_patches', 'pulse_entities', "
                "or 'segment_summary'"
            )
        _positive_integer(self.channel_regions, "channel_regions")
        _positive_integer(self.time_regions, "time_regions")
        if self.channel_regions > min(
            stop - start for _, start, stop, _, _ in SENSOR_BLOCKS
        ):
            raise ValueError("channel_regions cannot exceed the smallest sensor block")
        if INPUT_SAMPLES % self.time_regions != 0:
            raise ValueError(
                "time_regions must divide the 300 EXO time samples exactly"
            )
        if not isinstance(self.uniform_entity_fraction, (int, float)) or isinstance(
            self.uniform_entity_fraction, bool
        ):
            raise ValueError("uniform_entity_fraction must be a real number")
        if not 0.0 < float(self.uniform_entity_fraction) < 1.0:
            raise ValueError(
                "uniform_entity_fraction must be strictly between 0 and 1"
            )
        uniform_per_block = round(self.tokens_per_block * self.uniform_entity_fraction)
        if uniform_per_block % self.channel_regions != 0:
            raise ValueError(
                "the number of uniform entities per sensor block must be divisible "
                "by channel_regions"
            )
        _positive_integer(self.entity_context_size, "entity_context_size")
        if self.entity_context_size % 2 == 0:
            raise ValueError("entity_context_size must be odd")

    @property
    def tokens_per_block(self) -> int:
        return self.channel_regions * self.time_regions

    @property
    def token_count(self) -> int:
        return len(SENSOR_BLOCKS) * self.tokens_per_block

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tokens_per_block"] = self.tokens_per_block
        payload["token_count"] = self.token_count
        return payload


def _validate_waveform(waveform: Tensor) -> Tensor:
    if not isinstance(waveform, Tensor):
        raise TypeError("waveform must be a PyTorch tensor")
    if waveform.ndim != 3 or tuple(waveform.shape[1:]) != (
        INPUT_CHANNELS,
        INPUT_SAMPLES,
    ):
        raise ValueError("waveform must have shape [B, 226, 300]")
    if waveform.shape[0] < 1:
        raise ValueError("waveform batch must be non-empty")
    if not waveform.is_floating_point():
        raise TypeError("waveform must be floating point")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError("waveform must contain only finite values")
    return waveform


def _region_bounds(length: int, regions: int) -> tuple[tuple[int, int], ...]:
    """Split ``length`` entries into balanced, non-empty contiguous regions."""

    base, remainder = divmod(length, regions)
    sizes = [base + (index < remainder) for index in range(regions)]
    bounds: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        stop = start + int(size)
        bounds.append((start, stop))
        start = stop
    return tuple(bounds)


def _static_region_coordinates(
    *,
    channel_regions: int,
    time_regions: int,
    device: torch.device | None = None,
) -> Tensor:
    rows: list[list[float]] = []
    for _, start, stop, side, family in SENSOR_BLOCKS:
        block_channels = stop - start
        for local_start, local_stop in _region_bounds(
            block_channels,
            channel_regions,
        ):
            local_center = 0.5 * (local_start + local_stop - 1)
            normalized_channel = (
                2.0 * local_center / max(block_channels - 1, 1) - 1.0
            )
            family_one_hot = [0.0, 0.0, 0.0]
            family_one_hot[family] = 1.0
            for time_index in range(time_regions):
                normalized_time = 2.0 * (time_index + 0.5) / time_regions - 1.0
                rows.append(
                    [normalized_time, normalized_channel, side, *family_one_hot]
                )
    return torch.tensor(rows, dtype=torch.float32, device=device)


class _RegionTokenizer(nn.Module):
    coordinate_dim = 6

    def __init__(self, *, channel_regions: int, time_regions: int) -> None:
        super().__init__()
        _positive_integer(channel_regions, "channel_regions")
        _positive_integer(time_regions, "time_regions")
        if channel_regions > min(
            stop - start for _, start, stop, _, _ in SENSOR_BLOCKS
        ):
            raise ValueError("channel_regions cannot exceed the smallest sensor block")
        if INPUT_SAMPLES % time_regions != 0:
            raise ValueError(
                "time_regions must divide the 300 EXO time samples exactly"
            )
        self.channel_regions = channel_regions
        self.time_regions = time_regions
        self.time_width = INPUT_SAMPLES // time_regions
        self.token_count = len(SENSOR_BLOCKS) * channel_regions * time_regions
        self.maximum_channels_per_region = max(
            math.ceil((stop - start) / channel_regions)
            for _, start, stop, _, _ in SENSOR_BLOCKS
        )
        self.register_buffer(
            "coordinate_template",
            _static_region_coordinates(
                channel_regions=channel_regions,
                time_regions=time_regions,
            ),
            persistent=False,
        )

    def _coordinates_and_mask(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        coordinates = self.coordinate_template.to(
            device=waveform.device,
            dtype=waveform.dtype,
        ).unsqueeze(0).expand(waveform.shape[0], -1, -1)
        mask = torch.ones(
            waveform.shape[0],
            self.token_count,
            dtype=torch.bool,
            device=waveform.device,
        )
        return coordinates, mask


class RawSensorPatchTokenizer(_RegionTokenizer):
    """Preserve raw samples inside sensor-aware channel/time regions."""

    def __init__(self, channel_regions: int = 7, time_regions: int = 12) -> None:
        super().__init__(
            channel_regions=channel_regions,
            time_regions=time_regions,
        )
        self.feature_dim = self.maximum_channels_per_region * self.time_width

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        waveform = _validate_waveform(waveform)
        region_tokens: list[Tensor] = []
        for _, block_start, block_stop, _, _ in SENSOR_BLOCKS:
            block = waveform[:, block_start:block_stop, :]
            for region_start, region_stop in _region_bounds(
                block.shape[1], self.channel_regions
            ):
                region = block[:, region_start:region_stop, :]
                missing_channels = self.maximum_channels_per_region - region.shape[1]
                if missing_channels:
                    region = F.pad(region, (0, 0, 0, missing_channels))
                tokens = (
                    region.reshape(
                        waveform.shape[0],
                        self.maximum_channels_per_region,
                        self.time_regions,
                        self.time_width,
                    )
                    .permute(0, 2, 1, 3)
                    .reshape(waveform.shape[0], self.time_regions, self.feature_dim)
                )
                region_tokens.append(tokens)
        features = torch.cat(region_tokens, dim=1)
        coordinates, mask = self._coordinates_and_mask(waveform)
        return {"coords": coordinates, "features": features, "mask": mask}


class SensorRegionSummaryTokenizer(_RegionTokenizer):
    """Summarize the same sensor regions without including padded values."""

    feature_dim = 4

    def __init__(self, channel_regions: int = 7, time_regions: int = 12) -> None:
        super().__init__(
            channel_regions=channel_regions,
            time_regions=time_regions,
        )

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        waveform = _validate_waveform(waveform)
        region_tokens: list[Tensor] = []
        for _, block_start, block_stop, _, _ in SENSOR_BLOCKS:
            block = waveform[:, block_start:block_stop, :]
            for region_start, region_stop in _region_bounds(
                block.shape[1], self.channel_regions
            ):
                region = block[:, region_start:region_stop, :].reshape(
                    waveform.shape[0],
                    region_stop - region_start,
                    self.time_regions,
                    self.time_width,
                )
                values = region.permute(0, 2, 1, 3).flatten(start_dim=2)
                mean = values.mean(dim=-1)
                rms = values.square().mean(dim=-1).clamp_min(0.0).sqrt()
                maximum = values.abs().amax(dim=-1)
                midpoint = self.time_width // 2
                first_half = region[..., :midpoint].mean(dim=(1, 3))
                second_half = region[..., midpoint:].mean(dim=(1, 3))
                temporal_change = second_half - first_half
                region_tokens.append(
                    torch.stack((mean, rms, maximum, temporal_change), dim=-1)
                )
        features = torch.cat(region_tokens, dim=1)
        coordinates, mask = self._coordinates_and_mask(waveform)
        return {"coords": coordinates, "features": features, "mask": mask}


class PulseEntityTokenizer(nn.Module):
    """Select uniform and high-importance channel/time points per sensor block."""

    coordinate_dim = 6
    feature_dim = 2

    def __init__(
        self,
        *,
        channel_regions: int = 7,
        time_regions: int = 12,
        uniform_fraction: float = 0.5,
        context_size: int = 9,
    ) -> None:
        super().__init__()
        _positive_integer(channel_regions, "channel_regions")
        _positive_integer(time_regions, "time_regions")
        if channel_regions > min(
            stop - start for _, start, stop, _, _ in SENSOR_BLOCKS
        ):
            raise ValueError("channel_regions cannot exceed the smallest sensor block")
        if INPUT_SAMPLES % time_regions != 0:
            raise ValueError("time_regions must divide the 300 EXO time samples exactly")
        if not isinstance(uniform_fraction, (int, float)) or isinstance(
            uniform_fraction, bool
        ):
            raise ValueError("uniform_fraction must be a real number")
        if not 0.0 < float(uniform_fraction) < 1.0:
            raise ValueError("uniform_fraction must be strictly between 0 and 1")
        _positive_integer(context_size, "context_size")
        if context_size % 2 == 0:
            raise ValueError("context_size must be odd")
        self.channel_regions = channel_regions
        self.time_regions = time_regions
        self.tokens_per_block = channel_regions * time_regions
        self.token_count = len(SENSOR_BLOCKS) * self.tokens_per_block
        self.uniform_count = round(self.tokens_per_block * uniform_fraction)
        self.important_count = self.tokens_per_block - self.uniform_count
        if self.uniform_count % channel_regions != 0:
            raise ValueError(
                "uniform entities per block must be divisible by channel_regions"
            )
        self.uniform_time_points = self.uniform_count // channel_regions
        self.context_size = context_size

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        waveform = _validate_waveform(waveform)
        feature_blocks: list[Tensor] = []
        coordinate_blocks: list[Tensor] = []

        for _, block_start, block_stop, side, family in SENSOR_BLOCKS:
            block = waveform[:, block_start:block_stop, :]
            batch_size, channels, samples = block.shape
            temporal_change = torch.diff(
                block,
                dim=-1,
                prepend=block[..., :1],
            )
            local_rms = (
                F.avg_pool1d(
                    block.reshape(batch_size * channels, 1, samples).square(),
                    kernel_size=self.context_size,
                    stride=1,
                    padding=self.context_size // 2,
                )
                .clamp_min(0.0)
                .sqrt()
                .reshape(batch_size, channels, samples)
            )
            importance = (block.abs() + temporal_change.abs()).flatten(start_dim=1)

            channel_anchors = torch.tensor(
                [
                    (start + stop - 1) // 2
                    for start, stop in _region_bounds(channels, self.channel_regions)
                ],
                device=waveform.device,
                dtype=torch.long,
            )
            time_anchors = torch.linspace(
                0,
                samples - 1,
                self.uniform_time_points,
                device=waveform.device,
                dtype=waveform.dtype,
            ).round().to(torch.long)
            uniform_indices = (
                channel_anchors[:, None] * samples + time_anchors[None, :]
            ).flatten()

            ranking_scores = importance.clone()
            ranking_scores[:, uniform_indices] = -torch.inf
            important_indices = torch.argsort(
                ranking_scores,
                dim=1,
                descending=True,
                stable=True,
            )[:, : self.important_count]
            selected_indices = torch.cat(
                (uniform_indices.expand(batch_size, -1), important_indices),
                dim=1,
            ).sort(dim=1).values

            flat_signal = block.flatten(start_dim=1)
            flat_rms = local_rms.flatten(start_dim=1)
            selected_signal = flat_signal.gather(1, selected_indices)
            selected_rms = flat_rms.gather(1, selected_indices)
            features = torch.stack((selected_signal, selected_rms), dim=-1)

            selected_channels = torch.div(
                selected_indices,
                samples,
                rounding_mode="floor",
            )
            selected_times = selected_indices.remainder(samples)
            normalized_time = 2.0 * selected_times.to(waveform.dtype) / (samples - 1) - 1.0
            normalized_channel = (
                2.0 * selected_channels.to(waveform.dtype) / max(channels - 1, 1) - 1.0
            )
            side_coordinate = torch.full_like(normalized_time, side)
            family_coordinates = torch.zeros(
                batch_size,
                self.tokens_per_block,
                3,
                dtype=waveform.dtype,
                device=waveform.device,
            )
            family_coordinates[..., family] = 1.0
            coordinates = torch.cat(
                (
                    normalized_time.unsqueeze(-1),
                    normalized_channel.unsqueeze(-1),
                    side_coordinate.unsqueeze(-1),
                    family_coordinates,
                ),
                dim=-1,
            )
            feature_blocks.append(features)
            coordinate_blocks.append(coordinates)

        all_features = torch.cat(feature_blocks, dim=1)
        all_coordinates = torch.cat(coordinate_blocks, dim=1)
        mask = torch.ones(
            waveform.shape[0],
            self.token_count,
            dtype=torch.bool,
            device=waveform.device,
        )
        return {
            "coords": all_coordinates,
            "features": all_features,
            "mask": mask,
        }


def build_tokenizer(config: TokenizationConfig) -> nn.Module:
    if not isinstance(config, TokenizationConfig):
        raise TypeError("config must be a TokenizationConfig")
    common = {
        "channel_regions": config.channel_regions,
        "time_regions": config.time_regions,
    }
    if config.tokenization == "raw_patches":
        return RawSensorPatchTokenizer(**common)
    if config.tokenization == "segment_summary":
        return SensorRegionSummaryTokenizer(**common)
    if config.tokenization == "pulse_entities":
        return PulseEntityTokenizer(
            **common,
            uniform_fraction=config.uniform_entity_fraction,
            context_size=config.entity_context_size,
        )
    raise RuntimeError(f"unsupported tokenization: {config.tokenization}")


__all__ = [
    "PulseEntityTokenizer",
    "RawSensorPatchTokenizer",
    "SENSOR_BLOCKS",
    "SensorRegionSummaryTokenizer",
    "TokenizationConfig",
    "TokenizationName",
    "build_tokenizer",
]
