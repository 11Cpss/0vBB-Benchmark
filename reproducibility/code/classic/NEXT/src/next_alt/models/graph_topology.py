"""Directional graph and persistent-homology classifiers for NEXT events.

The implementations in this module deliberately depend only on PyTorch.  They
consume the same padded, centered point representation as the existing graph
models: ``coords`` has shape ``(B, N, 3)``, ``features`` has shape
``(B, N, feature_dim)``, and ``mask`` has shape ``(B, N)``.

``DimeNetLiteClassifier`` is a compact directional-message-passing model.  It
uses radial Bessel-style features and cosine angular features over geometric
triplets, but it is not a reproduction of the complete DimeNet architecture.

``PersistencePersLayClassifier`` computes the exact zero-dimensional
Vietoris--Rips persistence diagram of a capped point set (equivalently, the
minimum-spanning-tree edge lengths) and applies an unnormalised learned
PersLay-style weighting.  It does not claim to compute H1 or higher homology.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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
        raise ValueError("point batches require at least one padded node slot")
    if mask is None:
        mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
    if not isinstance(mask, Tensor) or mask.shape != coords.shape[:2]:
        raise ValueError("mask must have shape (batch, nodes)")
    if coords.device != features.device or coords.device != mask.device:
        raise ValueError("coords, features and mask must share one device")
    mask = mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every event must contain at least one valid point")
    return coords, features, mask


def _batched_index(values: Tensor, indices: Tensor) -> Tensor:
    """Index the node dimension of ``values`` with any batched index shape."""

    if values.ndim != 3 or indices.ndim < 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("invalid batched node gather shapes")
    batch_shape = (values.shape[0],) + (1,) * (indices.ndim - 1)
    batch = torch.arange(values.shape[0], device=values.device).reshape(batch_shape)
    return values[batch, indices]


def _knn(
    coords: Tensor,
    mask: Tensor,
    k: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build a directed masked kNN graph and return live edge distances."""

    k_eff = min(_positive_int("k", k), coords.shape[1])
    with torch.no_grad():
        distances = torch.cdist(coords.detach().float(), coords.detach().float())
        distances.masked_fill_(~mask[:, None, :], float("inf"))
        distances.masked_fill_(~mask[:, :, None], float("inf"))
        diagonal = torch.eye(coords.shape[1], dtype=torch.bool, device=coords.device)
        distances.masked_fill_(diagonal[None], float("inf"))
        selected_distance, indices = distances.topk(k_eff, dim=-1, largest=False)
        edge_mask = torch.isfinite(selected_distance)
    neighbours = _batched_index(coords, indices)
    live_distance = (neighbours - coords.unsqueeze(2)).square().sum(dim=-1).sqrt()
    return indices, edge_mask, live_distance


def _masked_pool(values: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1).to(values.dtype)
    mean = (values * expanded.to(values.dtype)).sum(dim=1) / count
    floor = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~expanded, floor).amax(dim=1)
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


def _radial_basis(distance: Tensor, cutoff: float, radial_dim: int) -> Tensor:
    """Smoothly cut off sine/Bessel-style radial features."""

    scaled = distance / float(cutoff)
    inside = scaled < 1.0
    clipped = scaled.clamp(min=0.0, max=1.0)
    frequencies = torch.arange(
        1, radial_dim + 1, dtype=distance.dtype, device=distance.device
    )
    phase = math.pi * clipped.unsqueeze(-1) * frequencies
    denominator = clipped.unsqueeze(-1)
    limit = math.pi * frequencies
    bessel = torch.where(
        denominator > 1.0e-4,
        torch.sin(phase) / denominator.clamp_min(1.0e-4),
        limit,
    )
    envelope = 0.5 * (torch.cos(math.pi * clipped) + 1.0)
    return bessel * envelope.unsqueeze(-1) * inside.unsqueeze(-1).to(distance.dtype)


def _cosine_angular_basis(cosine: Tensor, angular_dim: int) -> Tensor:
    """Return ``cos(m*theta)`` without materialising ``theta``."""

    value = cosine.clamp(-1.0, 1.0)
    terms = [torch.ones_like(value)]
    if angular_dim > 1:
        terms.append(value)
    for _ in range(2, angular_dim):
        terms.append(2.0 * value * terms[-1] - terms[-2])
    return torch.stack(terms, dim=-1)


class _DirectionalInteraction(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        interaction_dim: int,
        radial_dim: int,
        angular_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.edge_down = nn.Sequential(nn.Linear(hidden_dim, interaction_dim), nn.SiLU())
        self.radial_in = nn.Linear(radial_dim, interaction_dim, bias=False)
        self.radial_out = nn.Linear(radial_dim, interaction_dim, bias=False)
        self.angular = nn.Linear(angular_dim, interaction_dim, bias=False)
        self.edge_up = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        edge: Tensor,
        radial_ij: Tensor,
        radial_jk: Tensor,
        angular_jik: Tensor,
        triplet_mask: Tensor,
        neighbour_indices: Tensor,
    ) -> Tensor:
        batch = torch.arange(edge.shape[0], device=edge.device)[:, None, None]
        edge_kj = edge[batch, neighbour_indices]

        triplet = self.edge_down(edge_kj)
        triplet = triplet * self.radial_out(radial_jk)
        triplet = triplet * self.angular(angular_jik)
        triplet = triplet * triplet_mask.unsqueeze(-1).to(triplet.dtype)
        count = triplet_mask.sum(dim=3, keepdim=True).clamp_min(1).to(triplet.dtype)
        aggregate = triplet.sum(dim=3) / count.sqrt()
        aggregate = aggregate * self.radial_in(radial_ij)
        update = self.edge_up(aggregate)
        return self.norm(edge + self.dropout(update))


class DimeNetLiteClassifier(nn.Module):
    """DimeNet-inspired directional message passing on a static kNN graph."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 96,
        interaction_dim: int = 48,
        num_blocks: int = 3,
        k: int = 8,
        radial_dim: int = 6,
        angular_dim: int = 4,
        cutoff: float = 1.0,
        classifier_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.interaction_dim = _positive_int("interaction_dim", interaction_dim)
        self.num_blocks = _positive_int("num_blocks", num_blocks)
        self.k = _positive_int("k", k)
        self.radial_dim = _positive_int("radial_dim", radial_dim)
        self.angular_dim = _positive_int("angular_dim", angular_dim)
        self.cutoff = _positive_float("cutoff", cutoff)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)

        self.node_encoder = nn.Sequential(
            nn.Linear(3 + self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + self.radial_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.interactions = nn.ModuleList(
            _DirectionalInteraction(
                self.hidden_dim,
                self.interaction_dim,
                self.radial_dim,
                self.angular_dim,
                self.dropout,
            )
            for _ in range(self.num_blocks)
        )
        self.edge_to_node = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(self.num_blocks)
        )
        self.node_norms = nn.ModuleList(
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_blocks)
        )
        self.node_dropout = nn.Dropout(self.dropout)
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

        indices, edge_mask, distance_ij = _knn(coords, mask, self.k)
        neighbour_nodes = _batched_index(nodes, indices)
        center_nodes = nodes.unsqueeze(2).expand_as(neighbour_nodes)
        radial_ij = _radial_basis(distance_ij, self.cutoff, self.radial_dim)
        radial_ij = radial_ij * edge_mask.unsqueeze(-1).to(radial_ij.dtype)
        edge = self.edge_encoder(torch.cat((center_nodes, neighbour_nodes, radial_ij), dim=-1))
        edge = edge * edge_mask.unsqueeze(-1).to(edge.dtype)

        batch = torch.arange(coords.shape[0], device=coords.device)[:, None, None]
        neighbour_of_neighbour = indices[batch, indices]
        neighbour_of_neighbour_mask = edge_mask[batch, indices]
        radial_jk = radial_ij[batch, indices]

        coords_j = _batched_index(coords, indices)
        coords_k = _batched_index(coords, neighbour_of_neighbour)
        vector_ji = coords.unsqueeze(2) - coords_j
        vector_jk = coords_k - coords_j.unsqueeze(3)
        numerator = (vector_ji.unsqueeze(3) * vector_jk).sum(dim=-1)
        denominator = (
            vector_ji.square().sum(dim=-1).sqrt().unsqueeze(3)
            * vector_jk.square().sum(dim=-1).sqrt()
        ).clamp_min(1.0e-8)
        angular = _cosine_angular_basis(numerator / denominator, self.angular_dim)

        center_index = torch.arange(coords.shape[1], device=coords.device)[None, :, None, None]
        triplet_mask = (
            edge_mask.unsqueeze(3)
            & neighbour_of_neighbour_mask
            & (neighbour_of_neighbour != center_index)
        )
        angular = angular * triplet_mask.unsqueeze(-1).to(angular.dtype)
        radial_jk = radial_jk * neighbour_of_neighbour_mask.unsqueeze(-1).to(radial_jk.dtype)

        for interaction, edge_to_node, node_norm in zip(
            self.interactions, self.edge_to_node, self.node_norms
        ):
            edge = interaction(
                edge,
                radial_ij,
                radial_jk,
                angular,
                triplet_mask,
                indices,
            )
            edge = edge * edge_mask.unsqueeze(-1).to(edge.dtype)
            messages = edge_to_node(edge) * edge_mask.unsqueeze(-1).to(edge.dtype)
            degree = edge_mask.sum(dim=2, keepdim=True).clamp_min(1).to(edge.dtype)
            aggregate = messages.sum(dim=2) / degree.sqrt()
            nodes = node_norm(nodes + self.node_dropout(aggregate))
            nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)

        return self.classifier(_masked_pool(nodes, mask)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "interaction_dim": self.interaction_dim,
            "num_blocks": self.num_blocks,
            "k": self.k,
            "radial_dim": self.radial_dim,
            "angular_dim": self.angular_dim,
            "cutoff": self.cutoff,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


def _top_energy_points(
    coords: Tensor,
    features: Tensor,
    mask: Tensor,
    maximum: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    count = min(_positive_int("max_topology_points", maximum), coords.shape[1])
    score = features[..., 0].masked_fill(~mask, torch.finfo(features.dtype).min)
    indices = score.topk(count, dim=1, largest=True, sorted=True).indices
    return (
        _batched_index(coords, indices),
        _batched_index(features, indices),
        _batched_index(mask.unsqueeze(-1), indices).squeeze(-1).bool(),
    )


def _h0_persistence_diagram(
    coords: Tensor,
    features: Tensor,
    mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return exact finite H0 VR pairs and endpoint marks via batched Prim."""

    batch_size, point_count, _ = coords.shape
    device = coords.device
    batch = torch.arange(batch_size, device=device)
    distance = torch.cdist(coords.float(), coords.float())
    distance.masked_fill_(~mask[:, None, :], float("inf"))
    distance.masked_fill_(~mask[:, :, None], float("inf"))
    diagonal = torch.eye(point_count, dtype=torch.bool, device=device)
    distance.masked_fill_(diagonal[None], float("inf"))

    valid_count = mask.sum(dim=1)
    start = mask.to(torch.int64).argmax(dim=1)
    visited = torch.zeros_like(mask)
    visited[batch, start] = True
    best_distance = distance[batch, start]
    best_parent = start[:, None].expand(-1, point_count).clone()

    diagram = torch.zeros(
        (batch_size, max(point_count - 1, 1), 6),
        dtype=torch.float32,
        device=device,
    )
    diagram_mask = torch.zeros(
        (batch_size, max(point_count - 1, 1)), dtype=torch.bool, device=device
    )
    for step in range(1, point_count):
        candidates = best_distance.masked_fill(visited | ~mask, float("inf"))
        child = candidates.argmin(dim=1)
        active = valid_count > step
        parent = best_parent[batch, child]
        death = candidates[batch, child]
        finite = active & torch.isfinite(death)

        child_feature = features[batch, child]
        parent_feature = features[batch, parent]
        energy_sum = child_feature[:, 0] + parent_feature[:, 0]
        energy_difference = (child_feature[:, 0] - parent_feature[:, 0]).abs()
        log_count_mean = 0.5 * (child_feature[:, 1] + parent_feature[:, 1])
        row = torch.stack(
            (
                torch.zeros_like(death),
                torch.where(finite, death, torch.zeros_like(death)),
                torch.where(finite, death, torch.zeros_like(death)),
                energy_sum,
                energy_difference,
                log_count_mean,
            ),
            dim=-1,
        )
        diagram[:, step - 1] = torch.where(finite[:, None], row, torch.zeros_like(row))
        diagram_mask[:, step - 1] = finite

        visited[batch, child] |= active
        proposal = distance[batch, child]
        improve = active[:, None] & ~visited & (proposal < best_distance)
        best_distance = torch.where(improve, proposal, best_distance)
        best_parent = torch.where(improve, child[:, None], best_parent)
    return diagram, diagram_mask


class PersistencePersLayClassifier(nn.Module):
    """Exact H0 persistence followed by learned, non-attentive PersLay pooling."""

    def __init__(
        self,
        feature_dim: int = 2,
        max_topology_points: int = 96,
        diagram_hidden_dim: int = 96,
        embedding_dim: int = 128,
        classifier_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.max_topology_points = _positive_int(
            "max_topology_points", max_topology_points
        )
        self.diagram_hidden_dim = _positive_int(
            "diagram_hidden_dim", diagram_hidden_dim
        )
        self.embedding_dim = _positive_int("embedding_dim", embedding_dim)
        self.classifier_dim = _positive_int("classifier_dim", classifier_dim)
        self.dropout = _dropout(dropout)
        self.diagram_encoder = nn.Sequential(
            nn.Linear(6, self.diagram_hidden_dim),
            nn.LayerNorm(self.diagram_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.diagram_hidden_dim, self.embedding_dim),
            nn.SiLU(),
        )
        weight_hidden = max(8, self.diagram_hidden_dim // 4)
        self.weight_network = nn.Sequential(
            nn.Linear(1, weight_hidden),
            nn.SiLU(),
            nn.Linear(weight_hidden, 1),
            nn.Softplus(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * self.embedding_dim + 4, self.classifier_dim),
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
        selected_coords, selected_features, selected_mask = _top_energy_points(
            coords, features, mask, self.max_topology_points
        )
        # Persistence pairing is discrete.  The detector representation is an
        # input rather than a learnable coordinate transform, so retaining its
        # graph is unnecessary and would make the O(M^2) matrix expensive.
        with torch.no_grad():
            diagram, diagram_mask = _h0_persistence_diagram(
                selected_coords.float(), selected_features.float(), selected_mask
            )

        encoded = self.diagram_encoder(diagram.to(dtype=coords.dtype))
        weights = self.weight_network(diagram[..., 2:3].to(dtype=coords.dtype))
        valid = diagram_mask.unsqueeze(-1)
        weighted = encoded * weights * valid.to(encoded.dtype)
        count = valid.sum(dim=1).clamp_min(1).to(encoded.dtype)
        weighted_sum = weighted.sum(dim=1) / count.sqrt()
        mean = (encoded * valid.to(encoded.dtype)).sum(dim=1) / count
        floor = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~valid, floor).amax(dim=1)
        maximum = torch.where(
            diagram_mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
        )

        persistence = diagram[..., 2].to(encoded.dtype)
        raw_count = diagram_mask.sum(dim=1).to(encoded.dtype)
        safe_count = raw_count.clamp_min(1.0)
        total_persistence = (persistence * diagram_mask.to(encoded.dtype)).sum(dim=1)
        mean_persistence = total_persistence / safe_count
        max_persistence = persistence.masked_fill(~diagram_mask, 0.0).amax(dim=1)
        statistics = torch.stack(
            (
                raw_count / float(self.max_topology_points),
                total_persistence / safe_count.sqrt(),
                mean_persistence,
                max_persistence,
            ),
            dim=-1,
        )
        pooled = torch.cat((weighted_sum, mean, maximum, statistics), dim=-1)
        return self.classifier(pooled).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "max_topology_points": self.max_topology_points,
            "diagram_hidden_dim": self.diagram_hidden_dim,
            "embedding_dim": self.embedding_dim,
            "classifier_dim": self.classifier_dim,
            "dropout": self.dropout,
        }


__all__ = ["DimeNetLiteClassifier", "PersistencePersLayClassifier"]
