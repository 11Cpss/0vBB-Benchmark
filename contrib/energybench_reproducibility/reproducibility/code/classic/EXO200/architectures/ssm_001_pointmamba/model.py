"""MJD PointMamba-lite adapted for EXO-200 binary classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from exobench.config import INPUT_FIELD, INPUT_SHAPE
from exobench.waveform_points import (
    WaveformPointTokenizer,
    dual_hilbert_sequences,
    gather_points,
    masked_pool,
    serialize_exo_waveform,
)


MODEL_CONFIG = {
    "feature_dim": 2,
    "model_dim": 128,
    "inner_dim": 192,
    "state_dim": 16,
    "dt_rank": 16,
    "num_layers": 3,
    "conv_kernel": 4,
    "hilbert_bits": 10,
    "scan_chunk_size": 32,
    "classifier_dim": 160,
    "dropout": 0.10,
}

REPRESENTATION_CONFIG = {
    "source": INPUT_FIELD,
    "adapter": "channel_major_flatten",
    "input_shape": list(INPUT_SHAPE),
    "serialized_shape": [1, INPUT_SHAPE[0] * INPUT_SHAPE[1]],
    "tokenizer": "adaptive_mean_rms",
    "point_count": 512,
}

TRAINING_DEFAULTS = {
    "batch_size": 4,
    "epochs": 50,
    "learning_rate": 3.0e-4,
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
            (
                parameters.shape[-1] - 2 * self.state_dim,
                self.state_dim,
                self.state_dim,
            ),
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


class PointMambaBackbone(nn.Module):
    """MJD selective SSM over two Hilbert orders of EXO waveform tokens."""

    def __init__(
        self,
        *,
        point_count: int,
        feature_dim: int,
        model_dim: int,
        inner_dim: int,
        state_dim: int,
        dt_rank: int,
        num_layers: int,
        conv_kernel: int,
        hilbert_bits: int,
        scan_chunk_size: int,
        classifier_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = WaveformPointTokenizer(point_count)
        self.hilbert_bits = int(hilbert_bits)
        self.input_encoder = nn.Sequential(
            nn.Linear(3 + feature_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
        )
        self.order_scale = nn.Parameter(torch.ones(2, model_dim))
        self.order_shift = nn.Parameter(torch.zeros(2, model_dim))
        self.blocks = nn.ModuleList(
            _SelectiveSSMBlock(
                model_dim,
                inner_dim,
                state_dim,
                dt_rank,
                conv_kernel,
                scan_chunk_size,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.event_encoder = nn.Sequential(
            nn.Linear(2 * model_dim, classifier_dim),
            nn.LayerNorm(classifier_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output_features = classifier_dim

    def _compact_dual_order(
        self,
        hilbert: Tensor,
        trans_hilbert: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _, padded_points, _ = hilbert.shape
        lengths = mask.sum(dim=1)
        positions = torch.arange(2 * padded_points, device=mask.device)[None]
        hilbert_slot = positions < lengths[:, None]
        trans_slot = (positions >= lengths[:, None]) & (
            positions < 2 * lengths[:, None]
        )
        source_position = torch.where(
            hilbert_slot,
            positions,
            positions - lengths[:, None],
        ).clamp(min=0, max=padded_points - 1)
        standard_values = gather_points(hilbert, source_position)
        trans_values = gather_points(trans_hilbert, source_position)
        values = torch.where(
            hilbert_slot.unsqueeze(-1),
            standard_values,
            torch.where(
                trans_slot.unsqueeze(-1),
                trans_values,
                torch.zeros_like(trans_values),
            ),
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

    def forward(self, waveform: Tensor) -> Tensor:
        serialized = serialize_exo_waveform(waveform)
        coords, features, mask = self.tokenizer(serialized)
        hilbert, trans_hilbert, hilbert_mask, trans_mask = dual_hilbert_sequences(
            coords,
            features,
            mask,
            self.hilbert_bits,
        )
        hilbert = self.input_encoder(hilbert)
        trans_hilbert = self.input_encoder(trans_hilbert)
        common_mask = hilbert_mask & trans_mask
        values, sequence_mask = self._compact_dual_order(
            hilbert,
            trans_hilbert,
            common_mask,
        )
        for block in self.blocks:
            values = block(values, sequence_mask)
        values = self.final_norm(values) * sequence_mask.unsqueeze(-1).to(values.dtype)
        return self.event_encoder(masked_pool(values, sequence_mask))


class PointMambaClassifier(nn.Module):
    """Return one raw logit whose positive class is background (label 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = PointMambaBackbone(
            point_count=REPRESENTATION_CONFIG["point_count"],
            **MODEL_CONFIG,
        )
        self.head = nn.Linear(self.backbone.output_features, 1)

    def forward(self, waveform: Tensor) -> Tensor:
        output = self.head(self.backbone(waveform))
        return output.squeeze(1)


def build_model() -> nn.Module:
    return PointMambaClassifier()


__all__ = [
    "ARCHITECTURE_CONFIG",
    "MODEL_CONFIG",
    "REPRESENTATION_CONFIG",
    "TRAINING_DEFAULTS",
    "PointMambaBackbone",
    "PointMambaClassifier",
    "build_model",
]
