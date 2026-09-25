"""Seeded, cached MJD classification subsamples.

The official ``MJD_Train_*``/``MJD_Test_*`` HDF5 shards are unchunked, so
random-row access is ~40x slower than a contiguous read (measured: ~770
events/s random vs. ~30,500 events/s contiguous). Training directly against
random ``DataLoader`` batches would be I/O-bound and would re-pay that cost
every epoch.

This module draws a fixed-size, seeded random subsample once, applies
``mjdbench``'s baseline-subtraction and classification amplitude
normalization exactly as-is (via ``MJDWaveformDataset.__getitems__`` --
reused, not reimplemented), and caches the result to disk as float32
tensors. All three tokenizations read the *same* cache: tokenization happens
inside the model (``mjd_rope_transformer.model``), so a single materialized
waveform tensor serves every cell, matching the repo's stated
CNN/Transformer-parity design.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mjdbench.config import DataConfig
from mjdbench.data import MJDWaveformDataset, discover_files

# Reused, not reimplemented: identical seeded permutation to the one
# mjdbench.data.prepare_dataset uses for the official 90/10 train/validation
# boundary. Importing the private helper keeps our subsample strictly inside
# that boundary (no validation event can leak into training) instead of
# re-deriving a parallel splitting rule.
from mjdbench.data import _subset_indices


@dataclass(frozen=True)
class SubsetSizes:
    """How many events to draw from each official partition."""

    n_train: int = 50_000
    n_val: int = 5_000
    n_test: int = 50_000
    seed: int = 42


def _cache_key(data_root: Path, sizes: SubsetSizes, data_config: DataConfig) -> str:
    payload: dict[str, Any] = {
        "data_root": str(data_root),
        "n_train": sizes.n_train,
        "n_val": sizes.n_val,
        "n_test": sizes.n_test,
        "seed": sizes.seed,
        "baseline_samples": data_config.baseline_samples,
        "classification_amplitude_normalization": (
            data_config.classification_amplitude_normalization
        ),
        "split_seed": data_config.seed,
        "validation_fraction": data_config.validation_fraction,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"mjd_classification_{sizes.n_train}_{sizes.n_val}_{sizes.n_test}_seed{sizes.seed}_{digest}.pt"


def _materialize(
    source: MJDWaveformDataset,
    indices: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Load one subsample through the unchanged mjdbench preprocessing path."""

    waveforms: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    for start in range(0, len(indices), chunk_size):
        chunk = [int(index) for index in indices[start : start + chunk_size]]
        items = source.__getitems__(chunk)
        waveforms.append(torch.stack([item["inputs"].squeeze(0) for item in items]))
        labels.append(
            torch.stack([item["clean"] for item in items]).to(torch.float32)
        )
        ids.append(torch.stack([item["id"] for item in items]))
    return {
        "waveform": torch.cat(waveforms).to(torch.float32),
        "clean": torch.cat(labels),
        "id": torch.cat(ids),
    }


def build_classification_cache(
    *,
    data_root: str | Path,
    cache_dir: str | Path,
    sizes: SubsetSizes = SubsetSizes(),
    data_config: DataConfig | None = None,
) -> Path:
    """Build (or reuse) a cached seeded MJD classification subsample.

    Returns the path to a ``torch.save``d dict with keys ``train``,
    ``validation``, ``test`` (each ``{"waveform", "clean", "id"}``) and
    ``meta``.
    """

    root = Path(data_root).expanduser().resolve()
    config = data_config or DataConfig(data_root=root)
    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _cache_key(root, sizes, config)
    if cache_path.is_file():
        return cache_path

    train_paths = discover_files(root, "train")
    test_paths = discover_files(root, "test")
    if not train_paths:
        raise FileNotFoundError(f"no MJD_Train_*.hdf5 files found in {root}")
    if not test_paths:
        raise FileNotFoundError(f"no MJD_Test_*.hdf5 files found in {root}")

    options = dict(
        task="classification",
        baseline_samples=config.baseline_samples,
        classification_amplitude_normalization=(
            config.classification_amplitude_normalization
        ),
    )
    train_source = MJDWaveformDataset(train_paths, **options)
    test_source = MJDWaveformDataset(test_paths, **options)
    try:
        # Official 90/10 seed=42 train/validation boundary (mjdbench.data).
        # Both arrays are already uniformly random permutations of their
        # pool, so a prefix of each is itself an unbiased subsample.
        official_train_indices, official_val_indices = _subset_indices(
            len(train_source), config.validation_fraction, config.seed
        )
        if sizes.n_train > len(official_train_indices):
            raise ValueError(
                f"n_train={sizes.n_train} exceeds the official training pool "
                f"({len(official_train_indices)})"
            )
        if sizes.n_val > len(official_val_indices):
            raise ValueError(
                f"n_val={sizes.n_val} exceeds the official validation pool "
                f"({len(official_val_indices)})"
            )
        train_indices = official_train_indices[: sizes.n_train]
        val_indices = official_val_indices[: sizes.n_val]

        # The official test pool has no train/val boundary to respect: draw
        # an explicit seeded random sample rather than a contiguous prefix
        # (a prefix would effectively only cover the first shard).
        if sizes.n_test > len(test_source):
            raise ValueError(
                f"n_test={sizes.n_test} exceeds the official test pool "
                f"({len(test_source)})"
            )
        test_rng = np.random.default_rng(sizes.seed)
        test_indices = np.sort(
            test_rng.choice(len(test_source), size=sizes.n_test, replace=False)
        )

        cache = {
            "train": _materialize(train_source, train_indices),
            "validation": _materialize(train_source, val_indices),
            "test": _materialize(test_source, test_indices),
            "meta": {
                "data_root": str(root),
                "n_train": sizes.n_train,
                "n_val": sizes.n_val,
                "n_test": sizes.n_test,
                "seed": sizes.seed,
                "split_seed": config.seed,
                "baseline_samples": config.baseline_samples,
                "classification_amplitude_normalization": (
                    config.classification_amplitude_normalization
                ),
            },
        }
    finally:
        train_source.close()
        test_source.close()

    # Per-process suffix so two concurrent builders (e.g. array tasks that
    # started before the cache existed) never write the same temp file; the
    # rename onto the final path is atomic, so last writer wins harmlessly.
    tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
    torch.save(cache, tmp_path)
    tmp_path.replace(cache_path)
    return cache_path


class _CachedClassificationDataset(Dataset):
    """In-memory dataset over one materialized split."""

    def __init__(self, split: dict[str, torch.Tensor]) -> None:
        self.waveform = split["waveform"]
        self.clean = split["clean"]

    def __len__(self) -> int:
        return int(self.waveform.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.waveform[index].unsqueeze(0),
            "clean": self.clean[index],
        }


def load_cached_loaders(
    cache_path: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Load a cache built by ``build_classification_cache`` into DataLoaders.

    Returns a dict with ``train_loader``, ``validation_loader``,
    ``test_loader`` (each yielding the ``{"inputs", "clean"}`` batches
    ``mjdbench.training`` already expects), ``counts``, and ``meta``.
    """

    cache = torch.load(Path(cache_path).expanduser(), weights_only=False)
    pin_memory = torch.cuda.is_available()
    loaders: dict[str, DataLoader] = {}
    counts: dict[str, int] = {}
    for split_name, shuffle in (
        ("train", True),
        ("validation", False),
        ("test", False),
    ):
        dataset = _CachedClassificationDataset(cache[split_name])
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        counts[split_name] = len(dataset)
    return {
        "train_loader": loaders["train"],
        "validation_loader": loaders["validation"],
        "test_loader": loaders["test"],
        "counts": counts,
        "meta": cache["meta"],
    }


__all__ = [
    "SubsetSizes",
    "build_classification_cache",
    "load_cached_loaders",
]
