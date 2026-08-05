"""Pure-PyTorch projection-mixer, KPConv-style, and sparse-voxel models.

The implementations in this module follow the format-v3 NEXT classifier
contract: ``forward`` accepts one batch mapping and returns one unnormalised
classification logit per event.  The KPConv and submanifold operators are
portable fallbacks written only with PyTorch tensor operations.  They are not
reproductions of the official high-performance compiled CUDA kernels.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _dropout(value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return converted


def _positive_int_sequence(name: str, values: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of positive integers")
    converted = tuple(_positive_int(f"{name} entry", value) for value in values)
    if not converted:
        raise ValueError(f"{name} must not be empty")
    return converted


def _masked_pool(values: Tensor, mask: Tensor) -> Tensor:
    """Concatenate masked mean and max; map an empty event to zeros."""

    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1).to(values.dtype)
    mean = (values * expanded.to(values.dtype)).sum(dim=1) / count
    floor = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~expanded, floor).amax(dim=1)
    maximum = torch.where(
        mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
    )
    return torch.cat((mean, maximum), dim=-1)


class _MixerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class _MixerBlock(nn.Module):
    def __init__(
        self,
        token_count: int,
        embedding_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.token_mlp = _MixerMLP(token_count, token_mlp_dim, dropout)
        self.channel_norm = nn.LayerNorm(embedding_dim)
        self.channel_mlp = _MixerMLP(
            embedding_dim, channel_mlp_dim, dropout
        )

    def forward(self, tokens: Tensor) -> Tensor:
        mixed_tokens = self.token_norm(tokens).transpose(1, 2)
        tokens = tokens + self.token_mlp(mixed_tokens).transpose(1, 2)
        return tokens + self.channel_mlp(self.channel_norm(tokens))


class ProjectionMLPMixerClassifier(nn.Module):
    """Patch MLP-Mixer classifier for the three fixed NEXT projections."""

    def __init__(
        self,
        input_channels: int = 3,
        grid_size: int = 128,
        patch_size: int = 16,
        embedding_dim: int = 128,
        depth: int = 6,
        token_mlp_dim: int = 32,
        channel_mlp_dim: int = 256,
        classifier_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_channels = _positive_int("input_channels", input_channels)
        self.grid_size = _positive_int("grid_size", grid_size)
        self.patch_size = _positive_int("patch_size", patch_size)
        if self.grid_size % self.patch_size != 0:
            raise ValueError("grid_size must be divisible by patch_size")
        self.embedding_dim = _positive_int("embedding_dim", embedding_dim)
        self.depth = _positive_int("depth", depth)
        self.token_mlp_dim = _positive_int("token_mlp_dim", token_mlp_dim)
        self.channel_mlp_dim = _positive_int(
            "channel_mlp_dim", channel_mlp_dim
        )
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        patches_per_axis = self.grid_size // self.patch_size
        self.token_count = patches_per_axis * patches_per_axis

        self.patch_embedding = nn.Conv2d(
            self.input_channels,
            self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.blocks = nn.ModuleList(
            _MixerBlock(
                token_count=self.token_count,
                embedding_dim=self.embedding_dim,
                token_mlp_dim=self.token_mlp_dim,
                channel_mlp_dim=self.channel_mlp_dim,
                dropout=self.dropout,
            )
            for _ in range(self.depth)
        )
        self.final_norm = nn.LayerNorm(self.embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.classifier_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_dim, 1),
        )

    def forward(self, batch: Mapping[str, Tensor] | Tensor) -> Tensor:
        projections = batch.get("projections") if isinstance(batch, Mapping) else batch
        if not isinstance(projections, Tensor):
            raise TypeError("projections must be a torch tensor")
        expected = (self.input_channels, self.grid_size, self.grid_size)
        if projections.ndim != 4 or tuple(projections.shape[1:]) != expected:
            raise ValueError(
                "projections must have shape (batch, %d, %d, %d)"
                % expected
            )
        tokens = self.patch_embedding(projections).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        event_embedding = self.final_norm(tokens).mean(dim=1)
        return self.classifier(event_embedding).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "grid_size": self.grid_size,
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim,
            "depth": self.depth,
            "token_mlp_dim": self.token_mlp_dim,
            "channel_mlp_dim": self.channel_mlp_dim,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


def _unpack_points(
    coords_or_batch: Tensor | Mapping[str, Tensor],
    features: Optional[Tensor],
    mask: Optional[Tensor],
    feature_dim: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(coords_or_batch, Mapping):
        coords = coords_or_batch.get("coords")
        features = coords_or_batch.get("features")
        mask = coords_or_batch.get("mask")
    else:
        coords = coords_or_batch
    if not isinstance(coords, Tensor) or not isinstance(features, Tensor):
        raise TypeError("coords and features must be torch tensors")
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (batch, points, 3)")
    if features.ndim != 3 or features.shape[:2] != coords.shape[:2]:
        raise ValueError("features must have shape (batch, points, feature_dim)")
    if features.shape[-1] != feature_dim:
        raise ValueError(f"expected {feature_dim} features per point")
    if coords.shape[1] < 1:
        raise ValueError("point batches need at least one padded point slot")
    if mask is None:
        mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
    if not isinstance(mask, Tensor) or mask.shape != coords.shape[:2]:
        raise ValueError("mask must have shape (batch, points)")
    if coords.device != features.device or coords.device != mask.device:
        raise ValueError("coords, features and mask must be on the same device")
    return coords, features, mask.bool()


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    batch_index = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch_index, indices]


def _fixed_kernel_points(count: int, radius: float) -> Tensor:
    """Place one kernel at zero and the rest on a deterministic sphere."""

    points = torch.zeros((count, 3), dtype=torch.float32)
    if count == 1:
        return points
    sphere_count = count - 1
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(sphere_count):
        z_value = 1.0 - 2.0 * (index + 0.5) / sphere_count
        xy_radius = math.sqrt(max(0.0, 1.0 - z_value * z_value))
        angle = index * golden_angle
        points[index + 1] = torch.tensor(
            [xy_radius * math.cos(angle), xy_radius * math.sin(angle), z_value]
        )
    points[1:] *= 0.65 * radius
    return points


def _knn_indices(
    coords: Tensor,
    mask: Tensor,
    k: int,
    radius: float,
) -> Tuple[Tensor, Tensor]:
    k_effective = min(k, coords.shape[1])
    with torch.no_grad():
        distances = torch.cdist(coords.detach().float(), coords.detach().float())
        distances.masked_fill_(~mask[:, None, :], float("inf"))
        distances.masked_fill_(~mask[:, :, None], float("inf"))
        selected_distances, indices = distances.topk(
            k_effective, dim=-1, largest=False
        )
        neighbour_mask = torch.isfinite(selected_distances) & (
            selected_distances <= radius
        )
    return indices, neighbour_mask


class _RigidKernelPointConvolution(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neighbour_count: int,
        kernel_point_count: int,
        radius: float,
        sigma: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neighbour_count = neighbour_count
        self.radius = radius
        self.sigma = sigma
        self.register_buffer(
            "kernel_points", _fixed_kernel_points(kernel_point_count, radius)
        )
        self.weight = nn.Parameter(
            torch.empty(kernel_point_count, input_dim, output_dim)
        )
        self.bias = nn.Parameter(torch.empty(output_dim))
        bound = 1.0 / math.sqrt(kernel_point_count * input_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, coords: Tensor, features: Tensor, mask: Tensor) -> Tensor:
        indices, neighbour_mask = _knn_indices(
            coords, mask, self.neighbour_count, self.radius
        )
        neighbour_coords = _gather(coords, indices)
        neighbour_features = _gather(features, indices)
        relative = neighbour_coords - coords.unsqueeze(2)
        kernel_points = self.kernel_points.to(
            device=coords.device, dtype=coords.dtype
        )
        kernel_distance = torch.linalg.vector_norm(
            relative.unsqueeze(3) - kernel_points.view(1, 1, 1, -1, 3),
            dim=-1,
        )
        influence = (1.0 - kernel_distance / self.sigma).clamp_min(0.0)
        influence = influence * neighbour_mask.unsqueeze(-1).to(influence.dtype)
        influence = influence / influence.sum(dim=2, keepdim=True).clamp_min(
            torch.finfo(influence.dtype).eps
        )
        kernel_features = torch.einsum(
            "bnqk,bnqc->bnkc", influence, neighbour_features
        )
        output = torch.einsum("bnkc,kco->bno", kernel_features, self.weight)
        output = output + self.bias
        return output * mask.unsqueeze(-1).to(output.dtype)


class _KPResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neighbour_count: int,
        kernel_point_count: int,
        radius: float,
        sigma: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.convolution = _RigidKernelPointConvolution(
            input_dim,
            output_dim,
            neighbour_count,
            kernel_point_count,
            radius,
            sigma,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim)
        )

    def forward(self, coords: Tensor, features: Tensor, mask: Tensor) -> Tensor:
        update = self.dropout(self.activation(self.norm(self.convolution(
            coords, features, mask
        ))))
        output = self.activation(update + self.shortcut(features))
        return output * mask.unsqueeze(-1).to(output.dtype)


class RigidKPConvClassifier(nn.Module):
    """Rigid KPConv-style point classifier with a portable PyTorch backend."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 96, 128),
        neighbour_count: int = 24,
        kernel_point_count: int = 15,
        base_radius: float = 0.12,
        base_sigma: float = 0.06,
        radius_multiplier: float = 1.5,
        classifier_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dims = _positive_int_sequence("hidden_dims", hidden_dims)
        self.neighbour_count = _positive_int(
            "neighbour_count", neighbour_count
        )
        self.kernel_point_count = _positive_int(
            "kernel_point_count", kernel_point_count
        )
        self.base_radius = _positive_float("base_radius", base_radius)
        self.base_sigma = _positive_float("base_sigma", base_sigma)
        self.radius_multiplier = _positive_float(
            "radius_multiplier", radius_multiplier
        )
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)

        self.input_encoder = nn.Sequential(
            nn.Linear(3 + self.feature_dim, self.hidden_dims[0]),
            nn.LayerNorm(self.hidden_dims[0]),
            nn.SiLU(),
        )
        blocks = []
        input_dim = self.hidden_dims[0]
        for stage, output_dim in enumerate(self.hidden_dims):
            scale = self.radius_multiplier ** stage
            blocks.append(
                _KPResidualBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    neighbour_count=self.neighbour_count,
                    kernel_point_count=self.kernel_point_count,
                    radius=self.base_radius * scale,
                    sigma=self.base_sigma * scale,
                    dropout=self.dropout,
                )
            )
            input_dim = output_dim
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dims[-1], self.classifier_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_dim, 1),
        )

    def forward(
        self,
        coords: Tensor | Mapping[str, Tensor],
        features: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        coords, features, mask = _unpack_points(
            coords, features, mask, self.feature_dim
        )
        nodes = self.input_encoder(torch.cat((coords, features), dim=-1))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        for block in self.blocks:
            nodes = block(coords, nodes, mask)
        return self.classifier(_masked_pool(nodes, mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "neighbour_count": self.neighbour_count,
            "kernel_point_count": self.kernel_point_count,
            "base_radius": self.base_radius,
            "base_sigma": self.base_sigma,
            "radius_multiplier": self.radius_multiplier,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


_SUBMANIFOLD_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))


def _unpack_sparse(
    batch: Mapping[str, Tensor], feature_dim: int
) -> Tuple[Tensor, Tensor, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("forward expects a sparse-voxel batch mapping")
    coords = batch.get("voxel_coords")
    features = batch.get("voxel_features")
    mask = batch.get("voxel_mask")
    if not isinstance(coords, Tensor) or not isinstance(features, Tensor):
        raise TypeError("voxel_coords and voxel_features must be torch tensors")
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("voxel_coords must have shape (batch, voxels, 3)")
    if coords.dtype not in (torch.int32, torch.int64):
        raise TypeError("voxel_coords must have int32 or int64 dtype")
    if features.ndim != 3 or features.shape[:2] != coords.shape[:2]:
        raise ValueError(
            "voxel_features must have shape (batch, voxels, feature_dim)"
        )
    if features.shape[-1] != feature_dim:
        raise ValueError(f"expected {feature_dim} features per voxel")
    if coords.shape[1] < 1:
        raise ValueError("sparse batches need at least one padded voxel slot")
    if not isinstance(mask, Tensor) or mask.shape != coords.shape[:2]:
        raise ValueError("voxel_mask must have shape (batch, voxels)")
    if coords.device != features.device or coords.device != mask.device:
        raise ValueError(
            "voxel_coords, voxel_features and voxel_mask must share a device"
        )
    return coords.to(torch.int64), features, mask.bool()


class _SubmanifoldConvolution(nn.Module):
    """27-neighbour active-site convolution using sorted integer hashes."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer(
            "offsets", torch.tensor(_SUBMANIFOLD_OFFSETS, dtype=torch.int64)
        )
        self.weight = nn.Parameter(
            torch.empty(len(_SUBMANIFOLD_OFFSETS), input_dim, output_dim)
        )
        self.bias = nn.Parameter(torch.empty(output_dim))
        bound = 1.0 / math.sqrt(len(_SUBMANIFOLD_OFFSETS) * input_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _hash(coords: Tensor, minimum: Tensor, spans: Tensor) -> Tensor:
        shifted = coords - minimum
        return (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2] + shifted[:, 2]

    def forward(self, coords: Tensor, features: Tensor, mask: Tensor) -> Tensor:
        batch_outputs = []
        padded_count = coords.shape[1]
        for batch_index in range(coords.shape[0]):
            active_index = torch.nonzero(
                mask[batch_index], as_tuple=False
            ).squeeze(-1)
            if active_index.numel() == 0:
                batch_outputs.append(
                    features.new_zeros((padded_count, self.output_dim))
                )
                continue
            active_coords = coords[batch_index].index_select(0, active_index)
            active_features = features[batch_index].index_select(0, active_index)
            minimum = active_coords.amin(dim=0)
            maximum = active_coords.amax(dim=0)
            spans = maximum - minimum + 1
            keys = self._hash(active_coords, minimum, spans)
            sorted_keys, order = keys.sort()
            sorted_features = active_features.index_select(0, order)
            neighbour_coords = (
                active_coords[:, None, :] + self.offsets[None, :, :]
            )
            inside = (
                (neighbour_coords >= minimum.view(1, 1, 3))
                & (neighbour_coords <= maximum.view(1, 1, 3))
            ).all(dim=-1)
            flat_neighbour_coords = neighbour_coords.reshape(-1, 3)
            flat_neighbour_keys = self._hash(
                flat_neighbour_coords, minimum, spans
            )
            flat_locations = torch.searchsorted(
                sorted_keys, flat_neighbour_keys
            )
            safe_locations = flat_locations.clamp_max(
                sorted_keys.numel() - 1
            )
            present = inside.reshape(-1) & (
                flat_locations < sorted_keys.numel()
            ) & (
                sorted_keys.index_select(0, safe_locations)
                == flat_neighbour_keys
            )
            neighbour_features = sorted_features.index_select(
                0, safe_locations
            ).reshape(
                active_index.numel(), len(_SUBMANIFOLD_OFFSETS), self.input_dim
            )
            neighbour_features = neighbour_features * present.reshape(
                active_index.numel(), len(_SUBMANIFOLD_OFFSETS), 1
            ).to(neighbour_features.dtype)
            active_output = torch.einsum(
                "voc,ocd->vd", neighbour_features, self.weight
            ) + self.bias
            padded_output = features.new_zeros((padded_count, self.output_dim))
            padded_output = padded_output.index_copy(
                0, active_index, active_output
            )
            batch_outputs.append(padded_output)
        return torch.stack(batch_outputs, dim=0)


class _SubmanifoldResidualBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.convolution1 = _SubmanifoldConvolution(input_dim, output_dim)
        self.norm1 = nn.LayerNorm(output_dim)
        self.convolution2 = _SubmanifoldConvolution(output_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.shortcut = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim)
        )

    def forward(self, coords: Tensor, features: Tensor, mask: Tensor) -> Tensor:
        update = self.activation(self.norm1(self.convolution1(
            coords, features, mask
        )))
        update = self.dropout(update)
        update = self.norm2(self.convolution2(coords, update, mask))
        output = self.activation(update + self.shortcut(features))
        return output * mask.unsqueeze(-1).to(output.dtype)


class SubmanifoldSparseResNetClassifier(nn.Module):
    """Sparse 3-D residual classifier with a pure-PyTorch SubMConv fallback."""

    def __init__(
        self,
        feature_dim: int = 2,
        stage_channels: Sequence[int] = (24, 40, 64),
        stage_blocks: Sequence[int] = (1, 1, 1),
        classifier_dim: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.stage_channels = _positive_int_sequence(
            "stage_channels", stage_channels
        )
        self.stage_blocks = _positive_int_sequence("stage_blocks", stage_blocks)
        if len(self.stage_channels) != len(self.stage_blocks):
            raise ValueError("stage_channels and stage_blocks must have equal length")
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)

        self.stem = _SubmanifoldConvolution(
            self.feature_dim, self.stage_channels[0]
        )
        self.stem_norm = nn.LayerNorm(self.stage_channels[0])
        self.activation = nn.SiLU()
        blocks = []
        input_dim = self.stage_channels[0]
        for output_dim, block_count in zip(
            self.stage_channels, self.stage_blocks
        ):
            for _ in range(block_count):
                blocks.append(
                    _SubmanifoldResidualBlock(
                        input_dim, output_dim, self.dropout
                    )
                )
                input_dim = output_dim
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.stage_channels[-1], self.classifier_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_dim, 1),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        coords, features, mask = _unpack_sparse(batch, self.feature_dim)
        nodes = self.activation(self.stem_norm(self.stem(coords, features, mask)))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        for block in self.blocks:
            nodes = block(coords, nodes, mask)
        return self.classifier(_masked_pool(nodes, mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "stage_channels": list(self.stage_channels),
            "stage_blocks": list(self.stage_blocks),
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


__all__ = [
    "ProjectionMLPMixerClassifier",
    "RigidKPConvClassifier",
    "SubmanifoldSparseResNetClassifier",
]
