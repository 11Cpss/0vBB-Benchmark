"""Local, source-faithful copy of Gnn001."""

# Source: /home/wenyu/summer/src/next_alt/models/point_graph.py
# Source SHA256: ade0c49038cc1170948085f23828f49b344875e8c25b20c77fce511ac8a2ae54

from __future__ import annotations

from collections.abc import Mapping
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


__all__ = ["StaticGINEClassifier"]

