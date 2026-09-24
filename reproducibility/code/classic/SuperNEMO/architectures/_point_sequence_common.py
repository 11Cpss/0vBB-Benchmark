"""Shared, source-faithful point-sequence helpers for SEQ001 and SSM001."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Tuple

import torch
from torch import Tensor

# Model origin: ../NEXT/src/next_alt/models/point_sequence.py
# Source SHA256: 9d6e2b46ff93a99836836d2c9ff92e71354cc17cb6ae7e8569b23676da9b14db

def _positive_int(name: str, value: int, *, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return converted


def _dropout(value: float) -> float:
    converted = float(value)
    if not 0.0 <= converted < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return converted


def _unpack_points(
    coords_or_batch: Tensor | Mapping[str, Tensor],
    features: Optional[Tensor],
    mask: Optional[Tensor],
    feature_dim: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(coords_or_batch, Mapping):
        batch = coords_or_batch
        if "coords" in batch:
            coords = batch["coords"]
        elif "coordinates" in batch:
            coords = batch["coordinates"]
        elif "points" in batch:
            coords = batch["points"]
        else:
            raise KeyError("point batch requires 'coords', 'coordinates' or 'points'")
        features = batch.get("features")
        mask = batch.get("mask")
    else:
        coords = coords_or_batch

    if not isinstance(coords, Tensor) or not isinstance(features, Tensor):
        raise TypeError("coords and features must be torch tensors")
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (batch, nodes, 3)")
    if features.ndim != 3 or features.shape[:2] != coords.shape[:2]:
        raise ValueError("features must have shape (batch, nodes, feature_dim)")
    if features.shape[-1] != feature_dim:
        raise ValueError(
            f"expected {feature_dim} node features, got {features.shape[-1]}"
        )
    if coords.shape[1] < 1:
        raise ValueError("point batches must contain at least one padded node slot")
    if mask is None:
        mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
    if not isinstance(mask, Tensor) or mask.shape != coords.shape[:2]:
        raise ValueError("mask must have shape (batch, nodes)")
    if coords.device != features.device or coords.device != mask.device:
        raise ValueError("coords, features and mask must be on the same device")
    mask = mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every point-cloud event must contain at least one valid node")
    return coords, features, mask


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    """Gather ``(B,N,C)`` values with ``(B,Q)`` or ``(B,Q,K)`` indices."""

    batch_shape = (values.shape[0],) + (1,) * (indices.ndim - 1)
    batch = torch.arange(values.shape[0], device=values.device).view(batch_shape)
    return values[batch, indices]


def _masked_pool(values: Tensor, mask: Tensor) -> Tensor:
    """Concatenate the valid-token mean and max."""

    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1).to(values.dtype)
    mean = (values * expanded.to(values.dtype)).sum(dim=1) / count
    floor = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~expanded, floor).amax(dim=1)
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)
def _quantized_coordinates(coords: Tensor, mask: Tensor, bits: int) -> Tensor:
    """Map each event's valid bounding box to an integer Hilbert cube."""

    levels = (1 << _positive_int("hilbert_bits", bits, maximum=20)) - 1
    detached = coords.detach().float()
    positive_inf = torch.full_like(detached, float("inf"))
    negative_inf = torch.full_like(detached, -float("inf"))
    lower = torch.where(mask.unsqueeze(-1), detached, positive_inf).amin(dim=1, keepdim=True)
    upper = torch.where(mask.unsqueeze(-1), detached, negative_inf).amax(dim=1, keepdim=True)
    span = (upper - lower).clamp_min(torch.finfo(detached.dtype).eps)
    normalized = ((detached - lower) / span).clamp(0.0, 1.0)
    quantized = torch.floor(normalized * float(levels) + 0.5).to(torch.int64)
    return torch.where(mask.unsqueeze(-1), quantized, torch.zeros_like(quantized))


def _hilbert_codes(coords: Tensor, mask: Tensor, bits: int, *, transposed: bool) -> Tensor:
    """Return 3-D Hilbert integer codes using Skilling's transpose algorithm.

    ``transposed=True`` swaps the x/y axes before encoding.  This is the exact
    Trans-Hilbert convention used by this project and is intentionally stated in
    every model card because the PointMamba paper does not prescribe one unique
    coordinate-axis implementation for the transposed curve.
    """

    axes = _quantized_coordinates(coords, mask, bits)
    if transposed:
        axes = axes[..., (1, 0, 2)]
    axes = axes.clone()
    top = 1 << (int(bits) - 1)

    # Inverse undo followed by Gray encoding, vectorized over batch and nodes.
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


def _hilbert_orders(coords: Tensor, mask: Tensor, bits: int) -> Tuple[Tensor, Tensor]:
    """Return deterministic Hilbert and Trans-Hilbert permutations."""

    with torch.no_grad():
        standard = torch.argsort(
            _hilbert_codes(coords, mask, bits, transposed=False),
            dim=1,
            stable=True,
        )
        transposed = torch.argsort(
            _hilbert_codes(coords, mask, bits, transposed=True),
            dim=1,
            stable=True,
        )
    return standard, transposed


def _dual_sequences(
    coords: Tensor,
    features: Tensor,
    mask: Tensor,
    bits: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    point_input = torch.cat((coords, features), dim=-1)
    hilbert_order, trans_order = _hilbert_orders(coords, mask, bits)
    hilbert = _gather(point_input, hilbert_order)
    trans_hilbert = _gather(point_input, trans_order)
    hilbert_mask = _gather(mask.unsqueeze(-1), hilbert_order).squeeze(-1)
    trans_mask = _gather(mask.unsqueeze(-1), trans_order).squeeze(-1)
    return hilbert, trans_hilbert, hilbert_mask, trans_mask

