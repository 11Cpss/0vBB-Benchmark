"""Paired MJD models using the frozen NEXT Hilbert BiGRU architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mjdbench.waveform_points import (
    WaveformPointTokenizer,
    dual_hilbert_sequences,
    masked_pool,
)


MODEL_CONFIG = {
    "feature_dim": 2,
    "embedding_dim": 96,
    "hidden_dim": 128,
    "num_layers": 2,
    "hilbert_bits": 10,
    "classifier_dim": 256,
    "dropout": 0.10,
}

REPRESENTATION_CONFIG = {
    "source": "waveform",
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


class HilbertBiGRUBackbone(nn.Module):
    """Shared BiGRU encoder over Hilbert and Trans-Hilbert waveform points."""

    def __init__(
        self,
        *,
        point_count: int,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        hilbert_bits: int,
        classifier_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = WaveformPointTokenizer(point_count)
        self.hilbert_bits = int(hilbert_bits)
        self.input_encoder = nn.Sequential(
            nn.Linear(3 + feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(8 * hidden_dim, classifier_dim),
            nn.LayerNorm(classifier_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output_features = classifier_dim

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
        return masked_pool(output, mask)

    def forward(self, waveform: Tensor) -> Tensor:
        coords, features, mask = self.tokenizer(waveform)
        hilbert, trans_hilbert, hilbert_mask, trans_mask = dual_hilbert_sequences(
            coords,
            features,
            mask,
            self.hilbert_bits,
        )
        event = torch.cat(
            (
                self._encode_order(hilbert, hilbert_mask),
                self._encode_order(trans_hilbert, trans_mask),
            ),
            dim=-1,
        )
        return self.event_encoder(event)


class HilbertBiGRU(nn.Module):
    """Task head on the shared Hilbert BiGRU backbone."""

    def __init__(self, task: str) -> None:
        super().__init__()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.backbone = HilbertBiGRUBackbone(
            point_count=REPRESENTATION_CONFIG["point_count"],
            **MODEL_CONFIG,
        )
        self.head = nn.Linear(self.backbone.output_features, 1)

    def forward(self, waveform: Tensor) -> Tensor:
        output = self.head(self.backbone(waveform))
        return output.squeeze(1)


def build_model(task: str) -> nn.Module:
    return HilbertBiGRU(task)


__all__ = [
    "ARCHITECTURE_CONFIG",
    "MODEL_CONFIG",
    "REPRESENTATION_CONFIG",
    "TRAINING_DEFAULTS",
    "build_model",
]
