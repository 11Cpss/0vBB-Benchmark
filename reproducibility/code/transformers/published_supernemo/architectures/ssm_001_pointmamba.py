"""Local, source-faithful copy of SSM001."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._point_sequence_common import (
    _dropout,
    _dual_sequences,
    _gather,
    _masked_pool,
    _positive_int,
    _unpack_points,
)

# Source: /home/wenyu/summer/src/next_alt/models/point_sequence.py
# Source SHA256: 9d6e2b46ff93a99836836d2c9ff92e71354cc17cb6ae7e8569b23676da9b14db

def _chunked_selective_scan(
    drive: Tensor,
    delta: Tensor,
    input_b: Tensor,
    output_c: Tensor,
    negative_a: Tensor,
    skip: Tensor,
    mask: Tensor,
    chunk_size: int,
) -> Tensor:
    """Diagonal selective recurrence using vectorized closed-form chunks.

    For each channel/state pair this computes
    ``h_t = exp(delta_t A) h_(t-1) + delta_t B_t x_t`` and
    ``y_t = C_t h_t + D x_t``.  Chunk boundaries carry the exact final state;
    the bounded cumulative log-product guards the closed-form prefix evaluation.
    All recurrence arithmetic is float32 even under AMP.
    """

    original_dtype = drive.dtype
    drive32 = drive.float()
    delta32 = delta.float()
    input_b32 = input_b.float()
    output_c32 = output_c.float()
    negative_a32 = negative_a.float()
    skip32 = skip.float()
    valid = mask.bool()
    batch_size, sequence_length, inner_dim = drive32.shape
    state_dim = negative_a32.shape[1]
    state = torch.zeros(
        (batch_size, inner_dim, state_dim),
        device=drive.device,
        dtype=torch.float32,
    )
    outputs = []
    for start in range(0, sequence_length, chunk_size):
        stop = min(start + chunk_size, sequence_length)
        chunk_mask = valid[:, start:stop]
        chunk_drive = drive32[:, start:stop]
        chunk_delta = delta32[:, start:stop]
        transition = torch.exp(
            chunk_delta.unsqueeze(-1) * negative_a32.unsqueeze(0).unsqueeze(0)
        )
        transition = torch.where(
            chunk_mask.unsqueeze(-1).unsqueeze(-1),
            transition,
            torch.ones_like(transition),
        )
        update = (
            chunk_delta.unsqueeze(-1)
            * input_b32[:, start:stop].unsqueeze(2)
            * chunk_drive.unsqueeze(-1)
        )
        update = update * chunk_mask.unsqueeze(-1).unsqueeze(-1).to(update.dtype)

        log_prefix = torch.cumsum(torch.log(transition.clamp_min(1.0e-12)), dim=1)
        log_prefix = log_prefix.clamp(min=-60.0, max=0.0)
        prefix = torch.exp(log_prefix)
        states = prefix * (
            state.unsqueeze(1)
            + torch.cumsum(update * torch.exp(-log_prefix), dim=1)
        )
        state = states[:, -1]
        chunk_output = (
            states * output_c32[:, start:stop].unsqueeze(2)
        ).sum(dim=-1) + skip32.view(1, 1, -1) * chunk_drive
        chunk_output = chunk_output * chunk_mask.unsqueeze(-1).to(chunk_output.dtype)
        outputs.append(chunk_output)
    return torch.cat(outputs, dim=1).to(original_dtype)


class _SelectiveSSMBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        state_dim: int,
        dt_rank: int,
        conv_kernel: int,
        scan_chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.scan_chunk_size = scan_chunk_size
        self.norm = nn.LayerNorm(model_dim)
        self.in_projection = nn.Linear(model_dim, 2 * inner_dim)
        self.depthwise_convolution = nn.Conv1d(
            inner_dim,
            inner_dim,
            conv_kernel,
            groups=inner_dim,
        )
        self.conv_padding = conv_kernel - 1
        self.parameter_projection = nn.Linear(
            inner_dim,
            dt_rank + 2 * state_dim,
            bias=False,
        )
        self.delta_projection = nn.Linear(dt_rank, inner_dim)
        initial_a = torch.linspace(0.1, 1.0, state_dim).log()
        self.a_log = nn.Parameter(initial_a.repeat(inner_dim, 1))
        self.skip = nn.Parameter(torch.ones(inner_dim))
        self.out_projection = nn.Linear(inner_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.delta_projection.bias, -4.0)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        residual = values
        projected = self.in_projection(self.norm(values))
        drive, gate = projected.chunk(2, dim=-1)
        drive = drive * mask.unsqueeze(-1).to(drive.dtype)
        drive = self.depthwise_convolution(
            F.pad(drive.transpose(1, 2), (self.conv_padding, 0))
        ).transpose(1, 2)
        drive = F.silu(drive) * mask.unsqueeze(-1).to(drive.dtype)
        parameters = self.parameter_projection(drive)
        raw_delta, input_b, output_c = torch.split(
            parameters,
            (parameters.shape[-1] - 2 * self.state_dim, self.state_dim, self.state_dim),
            dim=-1,
        )
        delta = F.softplus(self.delta_projection(raw_delta))
        negative_a = -torch.exp(self.a_log)
        scanned = _chunked_selective_scan(
            drive,
            delta,
            input_b,
            output_c,
            negative_a,
            self.skip,
            mask,
            self.scan_chunk_size,
        )
        update = self.out_projection(scanned * F.silu(gate))
        output = residual + self.dropout(update)
        return output * mask.unsqueeze(-1).to(output.dtype)


class PointMambaLiteClassifier(nn.Module):
    """PointMamba-inspired classifier with a pure-PyTorch selective scan."""

    def __init__(
        self,
        feature_dim: int = 2,
        model_dim: int = 128,
        inner_dim: int = 192,
        state_dim: int = 16,
        dt_rank: int = 16,
        num_layers: int = 3,
        conv_kernel: int = 4,
        hilbert_bits: int = 10,
        scan_chunk_size: int = 32,
        classifier_dim: int = 160,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.model_dim = _positive_int("model_dim", model_dim)
        self.inner_dim = _positive_int("inner_dim", inner_dim)
        self.state_dim = _positive_int("state_dim", state_dim)
        self.dt_rank = _positive_int("dt_rank", dt_rank)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.conv_kernel = _positive_int("conv_kernel", conv_kernel)
        self.hilbert_bits = _positive_int("hilbert_bits", hilbert_bits, maximum=20)
        self.scan_chunk_size = _positive_int("scan_chunk_size", scan_chunk_size)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.input_encoder = nn.Sequential(
            nn.Linear(3 + self.feature_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.SiLU(),
        )
        self.order_scale = nn.Parameter(torch.ones(2, self.model_dim))
        self.order_shift = nn.Parameter(torch.zeros(2, self.model_dim))
        self.blocks = nn.ModuleList(
            _SelectiveSSMBlock(
                self.model_dim,
                self.inner_dim,
                self.state_dim,
                self.dt_rank,
                self.conv_kernel,
                self.scan_chunk_size,
                self.dropout,
            )
            for _ in range(self.num_layers)
        )
        self.final_norm = nn.LayerNorm(self.model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.model_dim, self.classifier_dim),
            nn.LayerNorm(self.classifier_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_dim, 1),
        )

    def _compact_dual_order(
        self,
        hilbert: Tensor,
        trans_hilbert: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, padded_points, _ = hilbert.shape
        lengths = mask.sum(dim=1)
        positions = torch.arange(2 * padded_points, device=mask.device)[None]
        hilbert_slot = positions < lengths[:, None]
        trans_slot = (positions >= lengths[:, None]) & (positions < 2 * lengths[:, None])
        source_position = torch.where(
            hilbert_slot,
            positions,
            positions - lengths[:, None],
        ).clamp(min=0, max=padded_points - 1)
        standard_values = _gather(hilbert, source_position)
        trans_values = _gather(trans_hilbert, source_position)
        values = torch.where(
            hilbert_slot.unsqueeze(-1),
            standard_values,
            torch.where(trans_slot.unsqueeze(-1), trans_values, torch.zeros_like(trans_values)),
        )
        values = torch.where(
            hilbert_slot.unsqueeze(-1),
            values * self.order_scale[0] + self.order_shift[0],
            torch.where(
                trans_slot.unsqueeze(-1),
                values * self.order_scale[1] + self.order_shift[1],
                torch.zeros_like(values),
            ),
        )
        return values, hilbert_slot | trans_slot

    def forward(
        self,
        coords: Tensor | Mapping[str, Tensor],
        features: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        coords, features, mask = _unpack_points(coords, features, mask, self.feature_dim)
        hilbert, trans_hilbert, hilbert_mask, trans_mask = _dual_sequences(
            coords, features, mask, self.hilbert_bits
        )
        hilbert = self.input_encoder(hilbert)
        trans_hilbert = self.input_encoder(trans_hilbert)
        # Both masks contain the same valid count; using their conjunction makes
        # that invariant explicit before compacting the two orders per event.
        common_mask = hilbert_mask & trans_mask
        values, sequence_mask = self._compact_dual_order(
            hilbert, trans_hilbert, common_mask
        )
        for block in self.blocks:
            values = block(values, sequence_mask)
        values = self.final_norm(values) * sequence_mask.unsqueeze(-1).to(values.dtype)
        return self.classifier(_masked_pool(values, sequence_mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "model_dim": self.model_dim,
            "inner_dim": self.inner_dim,
            "state_dim": self.state_dim,
            "dt_rank": self.dt_rank,
            "num_layers": self.num_layers,
            "conv_kernel": self.conv_kernel,
            "hilbert_bits": self.hilbert_bits,
            "scan_chunk_size": self.scan_chunk_size,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


__all__ = ["PointMambaLiteClassifier"]

