"""Held-out background-quantile score decorrelation.

For a calibration background sample, this module estimates

    u = F_background(score | energy-bin)

so a fixed threshold on ``u`` has approximately energy-flat background
acceptance.  The calibration sample must be independent of the final test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .utils import json_ready, weighted_quantile


@dataclass
class BackgroundQuantileDecorrelator:
    energy_edges: np.ndarray
    sorted_scores: List[np.ndarray]
    cumulative_weights: List[np.ndarray]
    score_direction: str = "higher"
    method: str = "background_conditional_ecdf_v1"

    @classmethod
    def fit(
        cls,
        energy: Any,
        score: Any,
        is_background: Any,
        sample_weight: Optional[Any] = None,
        n_energy_bins: int = 12,
        min_per_bin: int = 30,
        score_direction: str = "higher",
    ) -> "BackgroundQuantileDecorrelator":
        energy = np.asarray(energy, dtype=float)
        score = np.asarray(score, dtype=float)
        background = np.asarray(is_background, dtype=bool)
        if not (energy.ndim == score.ndim == background.ndim == 1):
            raise ValueError("energy, score, and is_background must be 1D")
        if not (len(energy) == len(score) == len(background)):
            raise ValueError("energy, score, and is_background lengths differ")
        if sample_weight is None:
            weight = np.ones(len(energy), dtype=float)
        else:
            weight = np.asarray(sample_weight, dtype=float)
            if weight.ndim != 1 or len(weight) != len(energy):
                raise ValueError("sample_weight must be 1D and event-aligned")
        if score_direction not in {"higher", "lower"}:
            raise ValueError("score_direction must be 'higher' or 'lower'")
        if n_energy_bins < 1:
            raise ValueError("n_energy_bins must be positive")
        if min_per_bin < 2:
            raise ValueError("min_per_bin must be at least 2")

        mask = (
            background
            & np.isfinite(energy)
            & np.isfinite(score)
            & np.isfinite(weight)
            & (weight > 0)
        )
        if int(np.sum(mask)) < min_per_bin:
            raise ValueError(
                "only %d finite calibration-background events; need at least %d"
                % (int(np.sum(mask)), min_per_bin)
            )
        e, s, w = energy[mask], score[mask], weight[mask]
        if score_direction == "lower":
            s = -s

        maximum_bins = max(1, min(int(n_energy_bins), len(e) // min_per_bin))
        quantiles = np.linspace(0.0, 1.0, maximum_bins + 1)
        edges = np.unique(weighted_quantile(e, quantiles, w))
        if len(edges) < 2:
            span = max(abs(float(e[0])) * 1e-9, 1e-12)
            edges = np.asarray([float(e[0]) - span, float(e[0]) + span])
        sorted_scores = []
        cumulative_weights = []
        for index in range(len(edges) - 1):
            if index == len(edges) - 2:
                in_bin = (e >= edges[index]) & (e <= edges[index + 1])
            else:
                in_bin = (e >= edges[index]) & (e < edges[index + 1])
            if int(np.sum(in_bin)) < 2:
                raise ValueError(
                    "energy bin %d has fewer than 2 calibration backgrounds; "
                    "use fewer bins" % index
                )
            sb, wb = s[in_bin], w[in_bin]
            order = np.argsort(sb, kind="mergesort")
            sb, wb = sb[order], wb[order]
            cumulative = np.r_[0.0, np.cumsum(wb)]
            cumulative /= cumulative[-1]
            sorted_scores.append(sb)
            cumulative_weights.append(cumulative)

        return cls(
            energy_edges=np.asarray(edges, dtype=float),
            sorted_scores=sorted_scores,
            cumulative_weights=cumulative_weights,
            score_direction=score_direction,
        )

    def transform(self, energy: Any, score: Any) -> np.ndarray:
        energy = np.asarray(energy, dtype=float)
        score = np.asarray(score, dtype=float)
        if energy.ndim != 1 or score.ndim != 1 or len(energy) != len(score):
            raise ValueError("energy and score must be aligned 1D arrays")
        internal_score = score if self.score_direction == "higher" else -score
        output = np.full(len(score), np.nan, dtype=float)
        finite = np.isfinite(energy) & np.isfinite(internal_score)
        bin_index = np.searchsorted(
            self.energy_edges[1:-1], energy[finite], side="right"
        )
        finite_indices = np.flatnonzero(finite)
        for index in range(len(self.sorted_scores)):
            selected = bin_index == index
            if not np.any(selected):
                continue
            destination = finite_indices[selected]
            values = internal_score[destination]
            reference = self.sorted_scores[index]
            cdf = self.cumulative_weights[index]
            left = np.searchsorted(reference, values, side="left")
            right = np.searchsorted(reference, values, side="right")
            # Mid-distribution transform handles score ties without assigning
            # the whole atom to one edge of the ECDF.
            output[destination] = 0.5 * (cdf[left] + cdf[right])
        return output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "score_direction": self.score_direction,
            "energy_edges": json_ready(self.energy_edges),
            "sorted_scores": json_ready(self.sorted_scores),
            "cumulative_weights": json_ready(self.cumulative_weights),
            "n_energy_bins": len(self.sorted_scores),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BackgroundQuantileDecorrelator":
        method = payload.get("method", "background_conditional_ecdf_v1")
        if method != "background_conditional_ecdf_v1":
            raise ValueError("unsupported decorrelator method %r" % method)
        return cls(
            energy_edges=np.asarray(payload["energy_edges"], dtype=float),
            sorted_scores=[
                np.asarray(values, dtype=float) for values in payload["sorted_scores"]
            ],
            cumulative_weights=[
                np.asarray(values, dtype=float)
                for values in payload["cumulative_weights"]
            ],
            score_direction=str(payload.get("score_direction", "higher")),
            method=method,
        )
