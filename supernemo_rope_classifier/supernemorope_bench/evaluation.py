"""Evaluation helpers for the shared EnergyBench classification protocol.

EnergyBench evaluates on a fixed physical-energy grid (0-3 MeV in 5 keV bins) and
raises if any evaluated event lies outside it. SuperNEMO ``Bi214`` events reach
3.18 MeV (0.013% of them exceed 3 MeV; ``2nubb`` tops out at 2.83 MeV), so the
test loader is wrapped to evaluate only events inside the grid and to report
how many it set aside. Training and validation are unaffected: they never bin
by energy.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader


def _select_events(value: Any, keep: torch.Tensor) -> Any:
    """Restrict every per-event field of a collated batch to the kept events."""

    count = int(keep.shape[0])
    if isinstance(value, torch.Tensor):
        return value[keep] if value.ndim >= 1 and value.shape[0] == count else value
    if isinstance(value, Mapping):
        return {key: _select_events(item, keep) for key, item in value.items()}
    if isinstance(value, list) and len(value) == count:
        return [item for item, flag in zip(value, keep.tolist()) if flag]
    return value


class EnergyWindowLoader:
    """Iterate a loader, yielding only events with ``energy <= max_energy_mev``.

    ``dropped`` maps category -> number of events set aside during the most
    recent full pass.
    """

    def __init__(self, loader: DataLoader, *, max_energy_mev: float) -> None:
        if not max_energy_mev > 0.0:
            raise ValueError("max_energy_mev must be positive")
        self.loader = loader
        self.max_energy_mev = float(max_energy_mev)
        self.dropped: dict[str, int] = {}

    @property
    def dataset(self) -> Any:
        return self.loader.dataset

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.dropped = {}
        for batch in self.loader:
            keep = batch["energy"] <= self.max_energy_mev
            if bool(keep.all()):
                yield batch
                continue
            for category, flag in zip(batch["category"], keep.tolist()):
                if not flag:
                    self.dropped[category] = self.dropped.get(category, 0) + 1
            if bool(keep.any()):
                yield _select_events(batch, keep)


__all__ = ["EnergyWindowLoader"]
