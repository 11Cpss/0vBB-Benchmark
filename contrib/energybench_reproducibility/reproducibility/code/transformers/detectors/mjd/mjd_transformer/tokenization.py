"""Batched waveform tokenization for the MJD Transformer.

The shared ``mjdbench.data`` loader returns dense waveforms with shape
``[batch, 1, samples]``. Tokenization stays inside the model so CNN and
Transformer baselines receive the exact same preprocessed waveform and split.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class TokenizationConfig:
    """Configuration for one MJD waveform representation."""

    tokenization: TokenizationName = "segment_summary"
    token_count: int = 500
    patch_size: int = 20
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
        _validate_positive_integer(self.token_count, "token_count")
        _validate_positive_integer(self.patch_size, "patch_size")
        if not isinstance(self.uniform_entity_fraction, (int, float)) or isinstance(
            self.uniform_entity_fraction, bool
        ):
            raise ValueError("uniform_entity_fraction must be a real number")
        if not 0.0 < float(self.uniform_entity_fraction) < 1.0:
            raise ValueError("uniform_entity_fraction must be strictly between 0 and 1")
        _validate_positive_integer(self.entity_context_size, "entity_context_size")
        if self.entity_context_size % 2 == 0:
            raise ValueError("entity_context_size must be odd")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_waveform(waveform: Tensor) -> Tensor:
    """Return a waveform with canonical shape ``[B, 1, L]``."""

    if not isinstance(waveform, Tensor):
        raise TypeError("waveform must be a PyTorch tensor")
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("waveform must have shape [B, L] or [B, 1, L]")
    if waveform.shape[0] < 1 or waveform.shape[-1] < 1:
        raise ValueError("waveform batch and sample dimensions must be non-empty")
    if not waveform.is_floating_point():
        raise TypeError("waveform must be floating point")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError("waveform must contain only finite values")
    return waveform


class SegmentSummaryTokenizer(nn.Module):
    """Summarize fixed-count temporal regions using mean and RMS amplitude."""

    coordinate_dim = 3
    feature_dim = 2

    def __init__(self, token_count: int = 500) -> None:
        super().__init__()
        _validate_positive_integer(token_count, "token_count")
        self.token_count = token_count

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        waveform = _validate_waveform(waveform)

        segment_mean = F.adaptive_avg_pool1d(
            waveform, self.token_count
        ).squeeze(1)
        segment_rms = (
            F.adaptive_avg_pool1d(waveform.square(), self.token_count)
            .clamp_min(0.0)
            .sqrt()
            .squeeze(1)
        )

        amplitude = segment_mean.abs().amax(dim=1, keepdim=True)
        amplitude = amplitude.clamp_min(torch.finfo(segment_mean.dtype).eps)
        normalized_mean = segment_mean / amplitude
        first_difference = torch.diff(
            normalized_mean,
            dim=1,
            prepend=normalized_mean[:, :1],
        )
        normalized_time = torch.linspace(
            -1.0,
            1.0,
            self.token_count,
            device=waveform.device,
            dtype=waveform.dtype,
        ).expand(waveform.shape[0], -1)

        coordinates = torch.stack(
            (normalized_time, normalized_mean, first_difference), dim=-1
        )
        features = torch.stack((segment_mean, segment_rms), dim=-1)
        mask = torch.ones(
            waveform.shape[0],
            self.token_count,
            dtype=torch.bool,
            device=waveform.device,
        )
        return {
            "coords": coordinates,
            "features": features,
            "mask": mask,
        }


class RawPatchTokenizer(nn.Module):
    """Represent contiguous waveform regions using their raw samples."""

    coordinate_dim = 3

    def __init__(self, patch_size: int = 20) -> None:
        super().__init__()

        # Validate that patch_size is a positive integer.
        if (
            isinstance(patch_size, bool)
            or not isinstance(patch_size, int)
            or patch_size <= 0
        ):
            raise ValueError("patch_size must be a positive integer")

        self.patch_size = patch_size

        # Each token contains patch_size raw amplitudes.
        self.feature_dim = patch_size

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        waveform = _validate_waveform(waveform)

        batch_size = waveform.shape[0]
        waveform_length = waveform.shape[-1]

        # Ensure waveform_length is divisible by patch_size.
        if waveform_length % self.patch_size != 0:
            raise ValueError("waveform_length must be divisible by patch_size")

        number_of_tokens = waveform_length // self.patch_size

        # Remove the one-channel dimension: [B, 1, L] -> [B, L].
        waveform_1d = waveform.squeeze(1)

        # Reshape [B, L] into [B, number_of_tokens, patch_size].
        patches = waveform_1d.reshape(
            batch_size,
            number_of_tokens,
            self.patch_size,
        )

        features = patches
        patch_mean = patches.mean(dim=-1)

        abs_patch_mean = patch_mean.abs()
        amplitude = abs_patch_mean.amax(dim=-1, keepdim=True)
        # Prevent division by zero.
        epsilon = torch.finfo(patch_mean.dtype).eps
        amplitude = amplitude.clamp_min(epsilon)
        normalized_mean = patch_mean / amplitude

        first_difference = torch.diff(
            normalized_mean,
            dim=-1,
            prepend=normalized_mean[:, :1],
        )

        normalized_time = torch.linspace(
            -1.0,
            1.0,
            number_of_tokens,
            device=waveform.device,
            dtype=waveform.dtype,
        ).expand(batch_size, -1)

        coordinate_tokens = torch.stack(
            (normalized_time, normalized_mean, first_difference),
            dim=-1,
        )
        mask = torch.ones(
            batch_size,
            number_of_tokens,
            dtype=torch.bool,
            device=waveform.device,
        )
        return {
            "mask": mask,
            "features": features,
            "coords": coordinate_tokens,
        }


class PulseEntityTokenizer(nn.Module):
    """Select deterministic waveform points as pulse/time entity tokens.

    A fixed fraction of tokens provides uniform time coverage. The remaining
    tokens are selected by absolute normalized amplitude plus absolute local
    change. Selection uses waveform inputs only; labels and energy targets are
    never consulted. Selected entities are restored to chronological order.
    """

    coordinate_dim = 3
    feature_dim = 2

    def __init__(
        self,
        token_count: int = 500,
        uniform_fraction: float = 0.5,
        context_size: int = 9,
    ) -> None:
        super().__init__()
        _validate_positive_integer(token_count, "token_count")
        if token_count < 2:
            raise ValueError("pulse entity token_count must be at least 2")
        if not isinstance(uniform_fraction, (int, float)) or isinstance(
            uniform_fraction, bool
        ):
            raise ValueError("uniform_fraction must be a real number")
        if not 0.0 < float(uniform_fraction) < 1.0:
            raise ValueError("uniform_fraction must be strictly between 0 and 1")
        _validate_positive_integer(context_size, "context_size")
        if context_size % 2 == 0:
            raise ValueError("context_size must be odd")

        self.token_count = token_count
        self.uniform_fraction = float(uniform_fraction)
        self.context_size = context_size

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        waveform = _validate_waveform(waveform)
        batch_size = waveform.shape[0]
        waveform_length = waveform.shape[-1]
        if self.token_count > waveform_length:
            raise ValueError("token_count cannot exceed waveform_length")

        signal = waveform.squeeze(1)
        scale = signal.abs().amax(dim=1, keepdim=True)
        scale = scale.clamp_min(torch.finfo(signal.dtype).eps)
        normalized_signal = signal / scale
        local_change = torch.diff(
            normalized_signal,
            dim=1,
            prepend=normalized_signal[:, :1],
        )
        local_rms = (
            F.avg_pool1d(
                waveform.square(),
                kernel_size=self.context_size,
                stride=1,
                padding=self.context_size // 2,
            )
            .clamp_min(0.0)
            .sqrt()
            .squeeze(1)
        )

        importance = normalized_signal.abs() + local_change.abs()
        uniform_count = round(self.token_count * self.uniform_fraction)
        uniform_count = min(max(uniform_count, 1), self.token_count - 1)
        important_count = self.token_count - uniform_count

        uniform_indices = torch.linspace(
            0,
            waveform_length - 1,
            uniform_count,
            device=waveform.device,
            dtype=waveform.dtype,
        ).round().to(torch.long)
        ranking_scores = importance.clone()
        ranking_scores[:, uniform_indices] = -torch.inf
        ranked_indices = torch.argsort(
            ranking_scores,
            dim=1,
            descending=True,
            stable=True,
        )
        important_indices = ranked_indices[:, :important_count]
        selected_indices = torch.cat(
            (
                uniform_indices.expand(batch_size, -1),
                important_indices,
            ),
            dim=1,
        ).sort(dim=1).values

        selected_signal = signal.gather(1, selected_indices)
        selected_rms = local_rms.gather(1, selected_indices)
        selected_normalized = normalized_signal.gather(1, selected_indices)
        selected_change = local_change.gather(1, selected_indices)
        full_time = torch.linspace(
            -1.0,
            1.0,
            waveform_length,
            device=waveform.device,
            dtype=waveform.dtype,
        ).expand(batch_size, -1)
        selected_time = full_time.gather(1, selected_indices)

        features = torch.stack((selected_signal, selected_rms), dim=-1)
        coordinates = torch.stack(
            (selected_time, selected_normalized, selected_change),
            dim=-1,
        )
        mask = torch.ones(
            batch_size,
            self.token_count,
            dtype=torch.bool,
            device=waveform.device,
        )
        return {
            "coords": coordinates,
            "features": features,
            "mask": mask,
        }


def build_tokenizer(config: TokenizationConfig) -> nn.Module:
    """Construct the tokenizer described by ``config``."""

    if not isinstance(config, TokenizationConfig):
        raise TypeError("config must be a TokenizationConfig")
    if config.tokenization == "raw_patches":
        return RawPatchTokenizer(patch_size=config.patch_size)
    if config.tokenization == "pulse_entities":
        return PulseEntityTokenizer(
            token_count=config.token_count,
            uniform_fraction=config.uniform_entity_fraction,
            context_size=config.entity_context_size,
        )
    if config.tokenization == "segment_summary":
        return SegmentSummaryTokenizer(token_count=config.token_count)
    raise RuntimeError(f"unsupported tokenization: {config.tokenization}")


__all__ = [
    "PulseEntityTokenizer",
    "RawPatchTokenizer",
    "SegmentSummaryTokenizer",
    "TokenizationConfig",
    "TokenizationName",
    "build_tokenizer",
]
