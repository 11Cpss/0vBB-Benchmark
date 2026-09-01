"""Shared waveform serialization and point-token helpers for EXO-200 models."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import INPUT_SHAPE


def serialize_exo_waveform(waveform: Tensor) -> Tensor:
    """Serialize ``[B, 226, 300]`` waveforms without dropping information.

    EXO channels remain in their stored order, and each channel's time samples
    remain contiguous.  The result is the single-waveform interface used by the
    corresponding MJD architectures: ``[B, 1, 226 * 300]``.
    """

    expected = tuple(int(value) for value in INPUT_SHAPE)
    if waveform.ndim != 3 or tuple(waveform.shape[1:]) != expected:
        raise ValueError(f"waveform must have shape [B, {expected[0]}, {expected[1]}]")
    if not waveform.is_floating_point():
        raise TypeError("waveform must be floating point")
    return waveform.reshape(waveform.shape[0], 1, expected[0] * expected[1])


class WaveformPointTokenizer(nn.Module):
    """Convert one serialized waveform into fixed-size point tokens."""

    def __init__(self, point_count: int = 512) -> None:
        super().__init__()
        if isinstance(point_count, bool) or int(point_count) != point_count:
            raise ValueError("point_count must be a positive integer")
        self.point_count = int(point_count)
        if self.point_count <= 0:
            raise ValueError("point_count must be a positive integer")

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [B, 1, L]")
        if waveform.shape[-1] < 1:
            raise ValueError("waveform length must be positive")
        if not waveform.is_floating_point():
            raise TypeError("waveform must be floating point")

        segment_mean = F.adaptive_avg_pool1d(waveform, self.point_count).squeeze(1)
        segment_rms = F.adaptive_avg_pool1d(
            waveform.square(), self.point_count
        ).clamp_min(0.0).sqrt().squeeze(1)
        amplitude = segment_mean.abs().amax(dim=1, keepdim=True)
        amplitude = amplitude.clamp_min(torch.finfo(segment_mean.dtype).eps)
        normalized_mean = segment_mean / amplitude
        first_difference = torch.diff(
            normalized_mean,
            dim=1,
            prepend=normalized_mean[:, :1],
        )
        time = torch.linspace(
            -1.0,
            1.0,
            self.point_count,
            device=waveform.device,
            dtype=waveform.dtype,
        ).expand(waveform.shape[0], -1)
        coords = torch.stack((time, normalized_mean, first_difference), dim=-1)
        features = torch.stack((segment_mean, segment_rms), dim=-1)
        mask = torch.ones(
            waveform.shape[0],
            self.point_count,
            dtype=torch.bool,
            device=waveform.device,
        )
        return coords, features, mask


def gather_points(values: Tensor, indices: Tensor) -> Tensor:
    """Gather batched point values with batched point indices."""

    batch_shape = (values.shape[0],) + (1,) * (indices.ndim - 1)
    batch = torch.arange(values.shape[0], device=values.device).view(batch_shape)
    return values[batch, indices]


def masked_pool(values: Tensor, mask: Tensor) -> Tensor:
    """Concatenate the valid-token mean and maximum."""

    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1).to(values.dtype)
    mean = (values * expanded.to(values.dtype)).sum(dim=1) / count
    floor = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~expanded, floor).amax(dim=1)
    maximum = torch.where(
        mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
    )
    return torch.cat((mean, maximum), dim=-1)


def _quantized_coordinates(coords: Tensor, mask: Tensor, bits: int) -> Tensor:
    levels = (1 << int(bits)) - 1
    detached = coords.detach().float()
    positive_inf = torch.full_like(detached, float("inf"))
    negative_inf = torch.full_like(detached, -float("inf"))
    lower = torch.where(mask.unsqueeze(-1), detached, positive_inf).amin(
        dim=1, keepdim=True
    )
    upper = torch.where(mask.unsqueeze(-1), detached, negative_inf).amax(
        dim=1, keepdim=True
    )
    span = (upper - lower).clamp_min(torch.finfo(detached.dtype).eps)
    normalized = ((detached - lower) / span).clamp(0.0, 1.0)
    quantized = torch.floor(normalized * float(levels) + 0.5).to(torch.int64)
    return torch.where(mask.unsqueeze(-1), quantized, torch.zeros_like(quantized))


def _hilbert_codes(
    coords: Tensor,
    mask: Tensor,
    bits: int,
    *,
    transposed: bool,
) -> Tensor:
    axes = _quantized_coordinates(coords, mask, bits)
    if transposed:
        axes = axes[..., (1, 0, 2)]
    axes = axes.clone()
    top = 1 << (int(bits) - 1)

    q = top
    while q > 1:
        p = q - 1
        for axis in range(3):
            old_first = axes[..., 0].clone()
            old_current = axes[..., axis].clone()
            selected = (old_current & q) != 0
            exchange = (old_first ^ old_current) & p
            first = torch.where(selected, old_first ^ p, old_first ^ exchange)
            current = torch.where(selected, old_current, old_current ^ exchange)
            axes[..., 0] = first
            if axis != 0:
                axes[..., axis] = current
        q >>= 1

    axes[..., 1] ^= axes[..., 0]
    axes[..., 2] ^= axes[..., 1]
    correction = torch.zeros_like(axes[..., 0])
    q = top
    while q > 1:
        correction ^= torch.where(
            (axes[..., 2] & q) != 0,
            torch.full_like(correction, q - 1),
            torch.zeros_like(correction),
        )
        q >>= 1
    axes ^= correction.unsqueeze(-1)

    code = torch.zeros_like(axes[..., 0])
    for bit in range(int(bits) - 1, -1, -1):
        for axis in range(3):
            code = (code << 1) | ((axes[..., axis] >> bit) & 1)
    return code.masked_fill(~mask, torch.iinfo(torch.int64).max)


def dual_hilbert_sequences(
    coords: Tensor,
    features: Tensor,
    mask: Tensor,
    bits: int = 10,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Serialize points with MJD's Hilbert and trans-Hilbert orders."""

    if int(bits) != bits or not 1 <= int(bits) <= 20:
        raise ValueError("bits must be an integer between 1 and 20")
    point_input = torch.cat((coords, features), dim=-1)
    with torch.no_grad():
        hilbert_order = torch.argsort(
            _hilbert_codes(coords, mask, int(bits), transposed=False),
            dim=1,
            stable=True,
        )
        trans_order = torch.argsort(
            _hilbert_codes(coords, mask, int(bits), transposed=True),
            dim=1,
            stable=True,
        )
    hilbert = gather_points(point_input, hilbert_order)
    trans_hilbert = gather_points(point_input, trans_order)
    hilbert_mask = gather_points(mask.unsqueeze(-1), hilbert_order).squeeze(-1)
    trans_mask = gather_points(mask.unsqueeze(-1), trans_order).squeeze(-1)
    return hilbert, trans_hilbert, hilbert_mask, trans_mask


__all__ = [
    "WaveformPointTokenizer",
    "dual_hilbert_sequences",
    "gather_points",
    "masked_pool",
    "serialize_exo_waveform",
]
