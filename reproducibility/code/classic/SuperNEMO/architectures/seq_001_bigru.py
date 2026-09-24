"""Local, source-faithful copy of SEQ001."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ._point_sequence_common import (
    _dropout,
    _dual_sequences,
    _masked_pool,
    _positive_int,
    _unpack_points,
)

# Model origin: ../NEXT/src/next_alt/models/point_sequence.py
# Source SHA256: 9d6e2b46ff93a99836836d2c9ff92e71354cc17cb6ae7e8569b23676da9b14db

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


__all__ = ["HilbertBiGRUClassifier"]

