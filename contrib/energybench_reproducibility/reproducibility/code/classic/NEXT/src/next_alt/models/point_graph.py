"""Pure-PyTorch point-cloud and graph classifiers for NEXT events.

All point models accept either ``model(coords, features, mask)`` or a mapping
containing ``coords`` (or ``points``), ``features`` and ``mask``.  The hybrid
model additionally accepts an ``image`` tensor.  No model depends on PyG/DGL;
small padded batches are handled with masked dense k-nearest-neighbour graphs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dropout(value: float) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return value


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.SiLU(),
    )


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
    return coords, features, mask.bool()


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    """Gather ``(B,N,C)`` values with ``(B,Q,K)`` node indices."""

    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch, indices]


def _knn(
    query: Tensor,
    support: Tensor,
    query_mask: Tensor,
    support_mask: Tensor,
    k: int,
    *,
    exclude_self: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return neighbour indices, finite-neighbour mask and squared distances."""

    k_eff = min(_positive_int("k", k), support.shape[1])
    # Neighbour selection is discrete.  Building it without autograd also avoids
    # retaining the O(B*N^2) distance matrix for backward.
    with torch.no_grad():
        distances = torch.cdist(query.detach().float(), support.detach().float())
        distances.masked_fill_(~support_mask[:, None, :], float("inf"))
        distances.masked_fill_(~query_mask[:, :, None], float("inf"))
        if exclude_self:
            if query.shape[1] != support.shape[1]:
                raise ValueError("exclude_self requires matching query/support sizes")
            diagonal = torch.eye(
                query.shape[1], dtype=torch.bool, device=query.device
            )[None]
            distances.masked_fill_(diagonal, float("inf"))
        selected_distance, indices = distances.topk(k_eff, dim=-1, largest=False)
        neighbour_mask = torch.isfinite(selected_distance)
    # Recompute selected distances from the live tensors.  kNN indices are
    # discrete, but GravNet's distance weights must still train its learned
    # coordinate projection.
    selected_support = _gather(support, indices)
    squared_distance = (selected_support - query.unsqueeze(2)).square().sum(dim=-1)
    return indices, neighbour_mask, squared_distance


def _masked_pool(values: Tensor, mask: Tensor) -> Tensor:
    """Concatenate masked mean and max; empty events map to all zeros."""

    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1)
    mean = (values * expanded.to(values.dtype)).sum(dim=1) / count.to(values.dtype)
    floor = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~expanded, floor).amax(dim=1)
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


def _masked_neighbour_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


def _masked_neighbour_max(values: Tensor, mask: Tensor) -> Tensor:
    floor = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~mask.unsqueeze(-1), floor).amax(dim=2)
    return torch.where(mask.any(dim=2, keepdim=True), maximum, torch.zeros_like(maximum))


class DeepSetsClassifier(nn.Module):
    """Permutation-invariant point classifier with mean/max event pooling."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 128,
        embedding_dim: int = 192,
        classifier_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.embedding_dim = _positive_int("embedding_dim", embedding_dim)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.node_encoder = nn.Sequential(
            nn.Linear(3 + self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.classifier_dim),
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
        coords, features, mask = _unpack_points(coords, features, mask, self.feature_dim)
        nodes = self.node_encoder(torch.cat((coords, features), dim=-1))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        return self.classifier(_masked_pool(nodes, mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


def _farthest_point_sample(coords: Tensor, mask: Tensor, samples: int) -> Tuple[Tensor, Tensor]:
    """Deterministic batched farthest-point sampling with padded outputs."""

    batch_size, node_count, _ = coords.shape
    sample_count = min(_positive_int("samples", samples), node_count)
    valid_count = mask.sum(dim=1)
    denom = valid_count.clamp_min(1).to(coords.dtype).view(batch_size, 1, 1)
    centroid = (coords * mask.unsqueeze(-1).to(coords.dtype)).sum(dim=1, keepdim=True) / denom
    radial = (coords - centroid).square().sum(dim=-1).masked_fill(~mask, -1.0)
    farthest = radial.argmax(dim=1)
    minimum_distance = torch.full(
        (batch_size, node_count), float("inf"), device=coords.device, dtype=coords.dtype
    )
    available = mask.clone()
    indices = torch.zeros(batch_size, sample_count, dtype=torch.long, device=coords.device)
    sampled_mask = torch.zeros(batch_size, sample_count, dtype=torch.bool, device=coords.device)
    batch_index = torch.arange(batch_size, device=coords.device)

    with torch.no_grad():
        for step in range(sample_count):
            active = valid_count > step
            indices[:, step] = farthest
            sampled_mask[:, step] = active
            chosen = coords[batch_index, farthest]
            distance = (coords - chosen[:, None]).square().sum(dim=-1)
            minimum_distance = torch.minimum(minimum_distance, distance)
            available[batch_index, farthest] &= ~active
            scores = minimum_distance.masked_fill(~available, -1.0)
            farthest = scores.argmax(dim=1)
    return indices, sampled_mask


class _SetAbstraction(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, k: int, dropout: float) -> None:
        super().__init__()
        self.k = k
        self.local_mlp = _mlp(input_dim + 4, output_dim, output_dim, dropout)

    def forward(
        self,
        support_coords: Tensor,
        support_features: Tensor,
        support_mask: Tensor,
        query_coords: Tensor,
        query_mask: Tensor,
    ) -> Tensor:
        indices, neighbour_mask, _ = _knn(
            query_coords, support_coords, query_mask, support_mask, self.k
        )
        neighbour_coords = _gather(support_coords, indices)
        neighbour_features = _gather(support_features, indices)
        relative = neighbour_coords - query_coords.unsqueeze(2)
        distance = relative.square().sum(dim=-1, keepdim=True).sqrt()
        messages = self.local_mlp(
            torch.cat((neighbour_features, relative, distance), dim=-1)
        )
        output = _masked_neighbour_max(messages, neighbour_mask)
        return output * query_mask.unsqueeze(-1).to(output.dtype)


class PointNetPPClassifier(nn.Module):
    """Lightweight PointNet++-style hierarchy using FPS and local kNN pooling."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 96,
        stage1_dim: int = 128,
        stage2_dim: int = 192,
        stage1_points: int = 64,
        stage2_points: int = 16,
        k: int = 16,
        classifier_dim: int = 160,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.stage1_dim = _positive_int("stage1_dim", stage1_dim)
        self.stage2_dim = _positive_int("stage2_dim", stage2_dim)
        self.stage1_points = _positive_int("stage1_points", stage1_points)
        self.stage2_points = _positive_int("stage2_points", stage2_points)
        self.k = _positive_int("k", k)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.input_encoder = _mlp(3 + self.feature_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.stage1 = _SetAbstraction(self.hidden_dim, self.stage1_dim, self.k, self.dropout)
        self.stage2 = _SetAbstraction(self.stage1_dim, self.stage2_dim, self.k, self.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.stage2_dim, self.classifier_dim),
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
        coords, features, mask = _unpack_points(coords, features, mask, self.feature_dim)
        encoded = self.input_encoder(torch.cat((coords, features), dim=-1))
        index1, mask1 = _farthest_point_sample(coords, mask, self.stage1_points)
        coords1 = _gather(coords, index1.unsqueeze(-1)).squeeze(2)
        features1 = self.stage1(coords, encoded, mask, coords1, mask1)
        index2, mask2 = _farthest_point_sample(coords1, mask1, self.stage2_points)
        coords2 = _gather(coords1, index2.unsqueeze(-1)).squeeze(2)
        features2 = self.stage2(coords1, features1, mask1, coords2, mask2)
        return self.classifier(_masked_pool(features2, mask2)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "stage1_dim": self.stage1_dim,
            "stage2_dim": self.stage2_dim,
            "stage1_points": self.stage1_points,
            "stage2_points": self.stage2_points,
            "k": self.k,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


class _GINELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, train_eps: bool) -> None:
        super().__init__()
        self.edge_encoder = _mlp(4, hidden_dim, hidden_dim, dropout)
        self.message_mlp = _mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.update_mlp = _mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        epsilon = torch.zeros(())
        if train_eps:
            self.epsilon = nn.Parameter(epsilon)
        else:
            self.register_buffer("epsilon", epsilon)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: Tensor,
        coords: Tensor,
        mask: Tensor,
        indices: Tensor,
        neighbour_mask: Tensor,
    ) -> Tensor:
        neighbours = _gather(nodes, indices)
        neighbour_coords = _gather(coords, indices)
        relative = neighbour_coords - coords.unsqueeze(2)
        distance = relative.square().sum(dim=-1, keepdim=True).sqrt()
        edge = self.edge_encoder(torch.cat((relative, distance), dim=-1))
        messages = self.message_mlp(neighbours + edge)
        messages = messages * neighbour_mask.unsqueeze(-1).to(messages.dtype)
        aggregate = messages.sum(dim=2)
        update = self.update_mlp((1.0 + self.epsilon) * nodes + aggregate)
        output = self.norm(nodes + self.dropout(update))
        return output * mask.unsqueeze(-1).to(output.dtype)


class StaticGINEClassifier(nn.Module):
    """Residual GINE network on one static geometric kNN graph."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 5,
        k: int = 12,
        classifier_dim: int = 160,
        dropout: float = 0.10,
        train_eps: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.k = _positive_int("k", k)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.train_eps = bool(train_eps)
        self.node_encoder = _mlp(3 + self.feature_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.layers = nn.ModuleList(
            _GINELayer(self.hidden_dim, self.dropout, self.train_eps)
            for _ in range(self.num_layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.classifier_dim),
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
        coords, features, mask = _unpack_points(coords, features, mask, self.feature_dim)
        nodes = self.node_encoder(torch.cat((coords, features), dim=-1))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        indices, neighbour_mask, _ = _knn(
            coords, coords, mask, mask, self.k, exclude_self=True
        )
        for layer in self.layers:
            nodes = layer(nodes, coords, mask, indices, neighbour_mask)
        return self.classifier(_masked_pool(nodes, mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "k": self.k,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
            "train_eps": self.train_eps,
        }


class _EdgeConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_mlp = _mlp(2 * input_dim + 4, output_dim, output_dim, dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        nodes: Tensor,
        coords: Tensor,
        mask: Tensor,
        indices: Tensor,
        neighbour_mask: Tensor,
    ) -> Tensor:
        neighbours = _gather(nodes, indices)
        relative_nodes = neighbours - nodes.unsqueeze(2)
        neighbour_coords = _gather(coords, indices)
        relative_coords = neighbour_coords - coords.unsqueeze(2)
        distance = relative_coords.square().sum(dim=-1, keepdim=True).sqrt()
        edge = torch.cat((nodes.unsqueeze(2).expand_as(neighbours), relative_nodes, relative_coords, distance), dim=-1)
        aggregate = _masked_neighbour_max(self.edge_mlp(edge), neighbour_mask)
        output = self.norm(self.skip(nodes) + self.dropout(aggregate))
        return output * mask.unsqueeze(-1).to(output.dtype)


class ParticleNetLiteClassifier(nn.Module):
    """ParticleNet-like dynamic EdgeConv network for voxel point clouds."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 96, 128),
        k: int = 16,
        classifier_dim: int = 192,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dims = tuple(_positive_int("hidden_dims item", value) for value in hidden_dims)
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        self.k = _positive_int("k", k)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.node_encoder = _mlp(3 + self.feature_dim, self.hidden_dims[0], self.hidden_dims[0], self.dropout)
        dimensions = (self.hidden_dims[0], *self.hidden_dims)
        self.layers = nn.ModuleList(
            _EdgeConvBlock(input_dim, output_dim, self.dropout)
            for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
        )
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
        coords, features, mask = _unpack_points(coords, features, mask, self.feature_dim)
        nodes = self.node_encoder(torch.cat((coords, features), dim=-1))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        for layer_index, layer in enumerate(self.layers):
            search = coords if layer_index == 0 else nodes / math.sqrt(nodes.shape[-1])
            indices, neighbour_mask, _ = _knn(
                search, search, mask, mask, self.k, exclude_self=True
            )
            nodes = layer(nodes, coords, mask, indices, neighbour_mask)
        return self.classifier(_masked_pool(nodes, mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "k": self.k,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


class _EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, message_dim: int, dropout: float, coord_scale: float) -> None:
        super().__init__()
        self.edge_mlp = _mlp(2 * hidden_dim + 1, message_dim, message_dim, dropout)
        self.feature_mlp = _mlp(hidden_dim + message_dim, hidden_dim, hidden_dim, dropout)
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(message_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1), nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.coord_scale = coord_scale

    def forward(
        self,
        nodes: Tensor,
        coords: Tensor,
        mask: Tensor,
        indices: Tensor,
        neighbour_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        neighbours = _gather(nodes, indices)
        neighbour_coords = _gather(coords, indices)
        relative = coords.unsqueeze(2) - neighbour_coords
        squared_distance = relative.square().sum(dim=-1, keepdim=True)
        message = self.edge_mlp(
            torch.cat((nodes.unsqueeze(2).expand_as(neighbours), neighbours, squared_distance), dim=-1)
        )
        edge_weight = neighbour_mask.unsqueeze(-1).to(message.dtype)
        message = message * edge_weight
        aggregate = message.sum(dim=2) / edge_weight.sum(dim=2).clamp_min(1.0)
        feature_update = self.feature_mlp(torch.cat((nodes, aggregate), dim=-1))
        nodes = self.norm(nodes + self.dropout(feature_update))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)

        # Add epsilon before sqrt: clamping after sqrt still leaves an infinite
        # derivative at zero for masked/self edges (0 * inf becomes NaN).
        direction = relative / (squared_distance + 1.0e-8).sqrt()
        coordinate_weight = self.coordinate_mlp(message) * edge_weight
        coordinate_update = (direction * coordinate_weight).sum(dim=2)
        coordinate_update = coordinate_update / edge_weight.sum(dim=2).clamp_min(1.0)
        coords = coords + self.coord_scale * coordinate_update
        coords = coords * mask.unsqueeze(-1).to(coords.dtype)
        return nodes, coords


class EGNNClassifier(nn.Module):
    """Distance-message EGNN with equivariant coordinate updates."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 128,
        message_dim: int = 128,
        num_layers: int = 5,
        k: int = 16,
        classifier_dim: int = 160,
        coord_scale: float = 0.10,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.message_dim = _positive_int("message_dim", message_dim)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.k = _positive_int("k", k)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.coord_scale = float(coord_scale)
        if not math.isfinite(self.coord_scale) or self.coord_scale < 0.0:
            raise ValueError("coord_scale must be finite and non-negative")
        self.dropout = _dropout(dropout)
        # Coordinates never enter scalar node features directly, preserving E(3)
        # invariance of the final event logit.
        self.node_encoder = _mlp(self.feature_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.layers = nn.ModuleList(
            _EGNNLayer(self.hidden_dim, self.message_dim, self.dropout, self.coord_scale)
            for _ in range(self.num_layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 2, self.classifier_dim),
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
        coords, features, mask = _unpack_points(coords, features, mask, self.feature_dim)
        nodes = self.node_encoder(features)
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        evolving_coords = coords * mask.unsqueeze(-1).to(coords.dtype)
        for layer in self.layers:
            indices, neighbour_mask, _ = _knn(
                evolving_coords, evolving_coords, mask, mask, self.k, exclude_self=True
            )
            nodes, evolving_coords = layer(
                nodes, evolving_coords, mask, indices, neighbour_mask
            )
        event_weight = mask.unsqueeze(-1).to(evolving_coords.dtype)
        event_center = (evolving_coords * event_weight).sum(dim=1, keepdim=True)
        event_center = event_center / event_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        radius_squared = (evolving_coords - event_center).square().sum(dim=-1, keepdim=True)
        geometry = _masked_pool(radius_squared, mask)
        event_features = torch.cat((_masked_pool(nodes, mask), geometry), dim=-1)
        return self.classifier(event_features).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "message_dim": self.message_dim,
            "num_layers": self.num_layers,
            "k": self.k,
            "classifier_dim": self.classifier_dim,
            "coord_scale": self.coord_scale,
            "dropout": self.dropout,
        }


class _GravNetBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        space_dim: int,
        propagate_dim: int,
        k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.k = k
        self.space_projection = nn.Linear(hidden_dim, space_dim)
        self.feature_projection = nn.Linear(hidden_dim, propagate_dim)
        self.update = _mlp(hidden_dim + 2 * propagate_dim, hidden_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, mask: Tensor) -> Tensor:
        learned_coords = self.space_projection(nodes)
        propagated = self.feature_projection(nodes)
        indices, neighbour_mask, squared_distance = _knn(
            learned_coords, learned_coords, mask, mask, self.k, exclude_self=True
        )
        neighbours = _gather(propagated, indices)
        weights = torch.exp(-squared_distance.to(neighbours.dtype)).unsqueeze(-1)
        weights = weights * neighbour_mask.unsqueeze(-1).to(neighbours.dtype)
        weighted_mean = (neighbours * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0e-8)
        maximum = _masked_neighbour_max(neighbours, neighbour_mask)
        update = self.update(torch.cat((nodes, weighted_mean, maximum), dim=-1))
        output = self.norm(nodes + self.dropout(update))
        return output * mask.unsqueeze(-1).to(output.dtype)


class GravNetClassifier(nn.Module):
    """GravNet-style graph network with learned-space dynamic neighbours."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        space_dim: int = 4,
        propagate_dim: int = 64,
        k: int = 16,
        classifier_dim: int = 160,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.space_dim = _positive_int("space_dim", space_dim)
        self.propagate_dim = _positive_int("propagate_dim", propagate_dim)
        self.k = _positive_int("k", k)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.node_encoder = _mlp(3 + self.feature_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.layers = nn.ModuleList(
            _GravNetBlock(
                self.hidden_dim, self.space_dim, self.propagate_dim, self.k, self.dropout
            )
            for _ in range(self.num_layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.classifier_dim),
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
        coords, features, mask = _unpack_points(coords, features, mask, self.feature_dim)
        nodes = self.node_encoder(torch.cat((coords, features), dim=-1))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        for layer in self.layers:
            nodes = layer(nodes, mask)
        return self.classifier(_masked_pool(nodes, mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "space_dim": self.space_dim,
            "propagate_dim": self.propagate_dim,
            "k": self.k,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _SharedViewEncoder(nn.Module):
    def __init__(self, base_channels: int, embedding_dim: int) -> None:
        super().__init__()
        channels = (base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels)
        blocks = []
        input_channels = 1
        for output_channels in channels:
            blocks.extend(
                (
                    nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.SiLU(),
                )
            )
            input_channels = output_channels
        self.backbone = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.projection = nn.Linear(channels[-1], embedding_dim)

    def forward(self, image: Tensor) -> Tensor:
        return self.projection(self.backbone(image).flatten(1))


class CNNGNNHybridClassifier(nn.Module):
    """Fuse a shared three-view 2D CNN with a dynamic EdgeConv graph."""

    def __init__(
        self,
        feature_dim: int = 2,
        image_base_channels: int = 16,
        image_embedding_dim: int = 128,
        graph_hidden_dim: int = 96,
        graph_embedding_dim: int = 192,
        num_graph_layers: int = 3,
        k: int = 16,
        classifier_dim: int = 192,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.image_base_channels = _positive_int("image_base_channels", image_base_channels)
        self.image_embedding_dim = _positive_int("image_embedding_dim", image_embedding_dim)
        self.graph_hidden_dim = _positive_int("graph_hidden_dim", graph_hidden_dim)
        self.graph_embedding_dim = _positive_int("graph_embedding_dim", graph_embedding_dim)
        self.num_graph_layers = _positive_int("num_graph_layers", num_graph_layers)
        self.k = _positive_int("k", k)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)

        self.view_encoder = _SharedViewEncoder(self.image_base_channels, self.image_embedding_dim)
        self.view_identity = nn.Parameter(torch.empty(1, 3, self.image_embedding_dim))
        nn.init.normal_(self.view_identity, mean=0.0, std=0.02)
        self.view_fusion = nn.Sequential(
            nn.Linear(2 * self.image_embedding_dim, self.image_embedding_dim), nn.SiLU()
        )
        self.node_encoder = _mlp(
            3 + self.feature_dim, self.graph_hidden_dim, self.graph_hidden_dim, self.dropout
        )
        self.graph_layers = nn.ModuleList(
            _EdgeConvBlock(self.graph_hidden_dim, self.graph_hidden_dim, self.dropout)
            for _ in range(self.num_graph_layers)
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(2 * self.graph_hidden_dim, self.graph_embedding_dim), nn.SiLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.image_embedding_dim + self.graph_embedding_dim, self.classifier_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_dim, 1),
        )

    def forward(
        self,
        image: Tensor | Mapping[str, Tensor],
        coords: Optional[Tensor] = None,
        features: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if isinstance(image, Mapping):
            batch = image
            if "image" in batch:
                image_tensor = batch["image"]
            elif "projections" in batch:
                image_tensor = batch["projections"]
            else:
                raise KeyError("hybrid batch requires 'image' or 'projections'")
            coords_or_batch: Tensor | Mapping[str, Tensor] = batch
        else:
            image_tensor = image
            if coords is None:
                raise TypeError("hybrid forward requires coords, features and mask")
            coords_or_batch = coords
        coords, features, mask = _unpack_points(
            coords_or_batch, features, mask, self.feature_dim
        )
        if not isinstance(image_tensor, Tensor):
            raise TypeError("image must be a torch tensor")
        if image_tensor.ndim == 5 and image_tensor.shape[2] == 1:
            image_tensor = image_tensor.squeeze(2)
        if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
            raise ValueError("image must have shape (batch, 3, height, width)")
        if image_tensor.shape[0] != coords.shape[0]:
            raise ValueError("image and point batch sizes must match")

        batch_size, view_count, height, width = image_tensor.shape
        views = image_tensor.reshape(batch_size * view_count, 1, height, width)
        view_embeddings = self.view_encoder(views).reshape(
            batch_size, view_count, self.image_embedding_dim
        )
        view_embeddings = view_embeddings + self.view_identity
        image_embedding = self.view_fusion(
            torch.cat((view_embeddings.mean(dim=1), view_embeddings.amax(dim=1)), dim=-1)
        )

        nodes = self.node_encoder(torch.cat((coords, features), dim=-1))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        for layer_index, layer in enumerate(self.graph_layers):
            search = coords if layer_index == 0 else nodes / math.sqrt(nodes.shape[-1])
            indices, neighbour_mask, _ = _knn(
                search, search, mask, mask, self.k, exclude_self=True
            )
            nodes = layer(nodes, coords, mask, indices, neighbour_mask)
        graph_embedding = self.graph_projection(_masked_pool(nodes, mask))
        return self.classifier(torch.cat((image_embedding, graph_embedding), dim=-1)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "image_base_channels": self.image_base_channels,
            "image_embedding_dim": self.image_embedding_dim,
            "graph_hidden_dim": self.graph_hidden_dim,
            "graph_embedding_dim": self.graph_embedding_dim,
            "num_graph_layers": self.num_graph_layers,
            "k": self.k,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


__all__ = [
    "DeepSetsClassifier",
    "PointNetPPClassifier",
    "StaticGINEClassifier",
    "ParticleNetLiteClassifier",
    "EGNNClassifier",
    "GravNetClassifier",
    "CNNGNNHybridClassifier",
]
