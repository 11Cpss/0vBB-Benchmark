"""Static geometric GINE classifier adapted to EXO-200 waveforms."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from exobench.config import INPUT_FIELD, INPUT_SHAPE
from exobench.waveform_points import (
    WaveformPointTokenizer,
    gather_points,
    masked_pool,
    serialize_exo_waveform,
)


MODEL_CONFIG = {
    "feature_dim": 2,
    "hidden_dim": 128,
    "num_layers": 5,
    "k": 12,
    "classifier_dim": 160,
    "dropout": 0.10,
    "train_eps": True,
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
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 5.0e-4,
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


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.SiLU(),
    )


def _knn(coords: Tensor, mask: Tensor, k: int) -> tuple[Tensor, Tensor]:
    k_eff = min(k, coords.shape[1])
    with torch.no_grad():
        distances = torch.cdist(coords.detach().float(), coords.detach().float())
        distances.masked_fill_(~mask[:, None, :], float("inf"))
        distances.masked_fill_(~mask[:, :, None], float("inf"))
        diagonal = torch.eye(
            coords.shape[1], dtype=torch.bool, device=coords.device
        )[None]
        distances.masked_fill_(diagonal, float("inf"))
        selected, indices = distances.topk(k_eff, dim=-1, largest=False)
    return indices, torch.isfinite(selected)


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
        neighbours = gather_points(nodes, indices)
        neighbour_coords = gather_points(coords, indices)
        relative = neighbour_coords - coords.unsqueeze(2)
        distance = relative.square().sum(dim=-1, keepdim=True).sqrt()
        edge = self.edge_encoder(torch.cat((relative, distance), dim=-1))
        messages = self.message_mlp(neighbours + edge)
        messages = messages * neighbour_mask.unsqueeze(-1).to(messages.dtype)
        aggregate = messages.sum(dim=2)
        update = self.update_mlp((1.0 + self.epsilon) * nodes + aggregate)
        output = self.norm(nodes + self.dropout(update))
        return output * mask.unsqueeze(-1).to(output.dtype)


class StaticGINEBackbone(nn.Module):
    """MJD StaticGINE feature extractor over serialized EXO waveforms."""

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 5,
        k: int = 12,
        dropout: float = 0.10,
        train_eps: bool = True,
        point_count: int = 512,
    ) -> None:
        super().__init__()
        self.tokenizer = WaveformPointTokenizer(point_count=point_count)
        self.node_encoder = _mlp(
            3 + feature_dim,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        self.layers = nn.ModuleList(
            _GINELayer(hidden_dim, dropout, train_eps)
            for _ in range(num_layers)
        )
        self.k = k
        self.output_features = 2 * hidden_dim

    def forward(self, waveform: Tensor) -> Tensor:
        serialized = serialize_exo_waveform(waveform)
        coords, features, mask = self.tokenizer(serialized)
        nodes = self.node_encoder(torch.cat((coords, features), dim=-1))
        nodes = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        indices, neighbour_mask = _knn(coords, mask, self.k)
        for layer in self.layers:
            nodes = layer(nodes, coords, mask, indices, neighbour_mask)
        return masked_pool(nodes, mask)


class StaticGINEClassifier(nn.Module):
    """Return one raw background logit for EXO binary classification."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = StaticGINEBackbone(
            feature_dim=MODEL_CONFIG["feature_dim"],
            hidden_dim=MODEL_CONFIG["hidden_dim"],
            num_layers=MODEL_CONFIG["num_layers"],
            k=MODEL_CONFIG["k"],
            dropout=MODEL_CONFIG["dropout"],
            train_eps=MODEL_CONFIG["train_eps"],
            point_count=REPRESENTATION_CONFIG["point_count"],
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_features, MODEL_CONFIG["classifier_dim"]),
            nn.SiLU(),
            nn.Dropout(MODEL_CONFIG["dropout"]),
            nn.Linear(MODEL_CONFIG["classifier_dim"], 1),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        return self.head(self.backbone(waveform)).squeeze(1)


def build_model() -> nn.Module:
    return StaticGINEClassifier()


__all__ = [
    "ARCHITECTURE_CONFIG",
    "MODEL_CONFIG",
    "REPRESENTATION_CONFIG",
    "StaticGINEBackbone",
    "StaticGINEClassifier",
    "TRAINING_DEFAULTS",
    "build_model",
]
