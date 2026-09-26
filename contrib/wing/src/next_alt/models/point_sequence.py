"""Pure-PyTorch point-MLP and serialized point-sequence classifiers.

The models in this module share the existing NEXT point-batch contract:
``coords`` has shape ``(B, N, 3)``, ``features`` has shape ``(B, N, F)``,
and ``mask`` has shape ``(B, N)``.  Sequence models derive both Hilbert and
Trans-Hilbert orders from those tensors inside the model, so every serialized
backbone sees exactly the same deterministic ordering implementation.

No implementation in this file uses attention or a compiled point/SSM kernel.
In particular, :class:`PointMambaLiteClassifier` contains a numerically guarded,
chunked, pure-PyTorch diagonal selective scan.  It is a portable fallback, not
a reproduction of the hardware-aware Mamba CUDA kernel.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


def _knn_indices(coords: Tensor, mask: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Build one fixed Euclidean kNN graph, excluding each node itself."""

    k_eff = min(_positive_int("k", k), coords.shape[1])
    with torch.no_grad():
        distances = torch.cdist(coords.detach().float(), coords.detach().float())
        distances.masked_fill_(~mask[:, None, :], float("inf"))
        distances.masked_fill_(~mask[:, :, None], float("inf"))
        diagonal = torch.eye(coords.shape[1], dtype=torch.bool, device=coords.device)[None]
        distances.masked_fill_(diagonal, float("inf"))
        selected, indices = distances.topk(k_eff, dim=-1, largest=False)
        neighbour_mask = torch.isfinite(selected)
    return indices, neighbour_mask


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


class _ResidualPointMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        expanded_dim = hidden_dim * expansion
        self.local = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        nodes: Tensor,
        coords: Tensor,
        mask: Tensor,
        neighbours: Tensor,
        neighbour_mask: Tensor,
    ) -> Tensor:
        neighbour_nodes = _gather(nodes, neighbours)
        neighbour_coords = _gather(coords, neighbours)
        relative = neighbour_coords - coords.unsqueeze(2)
        distance = relative.square().sum(dim=-1, keepdim=True).sqrt()
        messages = self.local(
            torch.cat((neighbour_nodes - nodes.unsqueeze(2), relative, distance), dim=-1)
        )
        floor = torch.finfo(messages.dtype).min
        aggregate = messages.masked_fill(
            ~neighbour_mask.unsqueeze(-1), floor
        ).amax(dim=2)
        aggregate = torch.where(
            neighbour_mask.any(dim=2, keepdim=True),
            aggregate,
            torch.zeros_like(aggregate),
        )
        nodes = self.local_norm(nodes + aggregate)
        nodes = nodes + self.ffn(self.ffn_norm(nodes))
        return nodes * mask.unsqueeze(-1).to(nodes.dtype)


class PointMLPClassifier(nn.Module):
    """Residual point-MLP classifier on one fixed Euclidean kNN graph."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        expansion: int = 2,
        k: int = 16,
        classifier_dim: int = 160,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.num_blocks = _positive_int("num_blocks", num_blocks)
        self.expansion = _positive_int("expansion", expansion)
        self.k = _positive_int("k", k)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.input_encoder = nn.Sequential(
            nn.Linear(3 + self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _ResidualPointMLPBlock(self.hidden_dim, self.expansion, self.dropout)
            for _ in range(self.num_blocks)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.classifier_dim),
            nn.LayerNorm(self.classifier_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_dim, 1),
        )

    def forward(
        self,
        coords: Tensor | Mapping[str, Tensor],
        features: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        coords, features, mask = _unpack_points(coords, features, mask, self.feature_dim)
        neighbours, neighbour_mask = _knn_indices(coords, mask, self.k)
        nodes = self.input_encoder(torch.cat((coords, features), dim=-1))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        for block in self.blocks:
            nodes = block(nodes, coords, mask, neighbours, neighbour_mask)
        return self.classifier(_masked_pool(nodes, mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "expansion": self.expansion,
            "k": self.k,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


class HilbertBiGRUClassifier(nn.Module):
    """Shared bidirectional GRU over Hilbert and Trans-Hilbert point orders."""

    def __init__(
        self,
        feature_dim: int = 2,
        embedding_dim: int = 96,
        hidden_dim: int = 128,
        num_layers: int = 2,
        hilbert_bits: int = 10,
        classifier_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.embedding_dim = _positive_int("embedding_dim", embedding_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.hilbert_bits = _positive_int("hilbert_bits", hilbert_bits, maximum=20)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.input_encoder = nn.Sequential(
            nn.Linear(3 + self.feature_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * self.hidden_dim, self.classifier_dim),
            nn.LayerNorm(self.classifier_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_dim, 1),
        )

    def _encode_order(self, sequence: Tensor, mask: Tensor) -> Tensor:
        encoded = self.input_encoder(sequence)
        lengths = mask.sum(dim=1).to(device="cpu", dtype=torch.int64)
        packed = pack_padded_sequence(
            encoded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        return _masked_pool(output, mask)

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
        event = torch.cat(
            (
                self._encode_order(hilbert, hilbert_mask),
                self._encode_order(trans_hilbert, trans_mask),
            ),
            dim=-1,
        )
        return self.classifier(event).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "hilbert_bits": self.hilbert_bits,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


class _CausalTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            dilation=dilation,
        )
        self.convolution2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            dilation=dilation,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _causal_convolution(self, values: Tensor, convolution: nn.Conv1d) -> Tensor:
        transposed = values.transpose(1, 2)
        padded = F.pad(transposed, (self.left_padding, 0))
        return convolution(padded).transpose(1, 2)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        update = self._causal_convolution(values, self.convolution1)
        update = self.dropout(F.gelu(self.norm1(update)))
        update = self._causal_convolution(update, self.convolution2)
        update = self.dropout(F.gelu(self.norm2(update)))
        return (values + update) * mask.unsqueeze(-1).to(values.dtype)


class HilbertTCNClassifier(nn.Module):
    """Dilated causal TCN shared by Hilbert and Trans-Hilbert sequences."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 128,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dilation_base: int = 2,
        hilbert_bits: int = 10,
        classifier_dim: int = 192,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.num_blocks = _positive_int("num_blocks", num_blocks)
        self.kernel_size = _positive_int("kernel_size", kernel_size)
        self.dilation_base = _positive_int("dilation_base", dilation_base)
        self.hilbert_bits = _positive_int("hilbert_bits", hilbert_bits, maximum=20)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.input_encoder = nn.Sequential(
            nn.Linear(3 + self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _CausalTCNBlock(
                self.hidden_dim,
                self.kernel_size,
                self.dilation_base**index,
                self.dropout,
            )
            for index in range(self.num_blocks)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.classifier_dim),
            nn.LayerNorm(self.classifier_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_dim, 1),
        )

    def _encode_order(self, sequence: Tensor, mask: Tensor) -> Tensor:
        values = self.input_encoder(sequence) * mask.unsqueeze(-1).to(sequence.dtype)
        for block in self.blocks:
            values = block(values, mask)
        return _masked_pool(values, mask)

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
        event = torch.cat(
            (
                self._encode_order(hilbert, hilbert_mask),
                self._encode_order(trans_hilbert, trans_mask),
            ),
            dim=-1,
        )
        return self.classifier(event).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "kernel_size": self.kernel_size,
            "dilation_base": self.dilation_base,
            "hilbert_bits": self.hilbert_bits,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
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


__all__ = [
    "HilbertBiGRUClassifier",
    "HilbertTCNClassifier",
    "PointMLPClassifier",
    "PointMambaLiteClassifier",
]
