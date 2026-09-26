"""Event indexing, the published block split, and streaming SuperNEMO loaders.

Raw rows are tracker hits; one event is the run of consecutive rows sharing an
``ev_no`` in one source file. The release stores every column contiguously and
uncompressed, so a block of consecutive events is a handful of slice reads.

Batches follow the shared EnergyBench contract (``inputs``, ``label``,
``energy`` in MeV, ``event_id``, ``category``, ``group_id``, ``split``,
``projection_coverage``), so the unchanged ``simple_energybench.train_model``
and ``evaluate_classification`` consume them directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from supernemo_rope_transformer.tokenization import (
    SuperNEMOTrackerTokenizationConfig,
    tokenize_tracker_event,
)

from .config import (
    CLASSIFICATION_SOURCES,
    COORDINATE_FIELDS,
    ENERGY_FIELDS,
    EVENT_ID_FIELD,
    KEV_PER_MEV,
    PUBLISHED_CLASSIFICATION_COUNTS,
    RADIUS_FIELD,
    SOURCE_BY_KEY,
    SOURCE_POSITION,
    SPLIT_NAMES,
    DataConfig,
)


@dataclass(frozen=True)
class EventSlice:
    """A contiguous half-open range of event ordinals within one source."""

    source_key: str
    split: str
    event_start: int
    event_stop: int

    def __post_init__(self) -> None:
        if self.source_key not in SOURCE_BY_KEY:
            raise ValueError(f"unknown source key: {self.source_key!r}")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {self.split!r}")
        if self.event_start < 0 or self.event_stop <= self.event_start:
            raise ValueError("event slice must be a non-empty half-open range")

    @property
    def event_count(self) -> int:
        return self.event_stop - self.event_start


def scan_event_offsets(path: Path) -> np.ndarray:
    """Return int64 row offsets ``[event_count + 1]`` for one source file.

    The release numbers events ``0..N-1`` with no gaps or repeats. Anything
    else would make an event ordinal ambiguous, so it is rejected rather than
    silently re-numbered.
    """

    with h5py.File(path, "r") as handle:
        event_numbers = np.asarray(handle[EVENT_ID_FIELD][:], dtype=np.int64)
    if event_numbers.size == 0:
        raise ValueError(f"{path}: file contains no rows")
    starts = np.concatenate(
        ([0], np.flatnonzero(event_numbers[1:] != event_numbers[:-1]) + 1)
    )
    if not np.array_equal(event_numbers[starts], np.arange(starts.size)):
        raise ValueError(
            f"{path}: {EVENT_ID_FIELD} must run 0..N-1 in contiguous row blocks"
        )
    return np.concatenate((starts, [event_numbers.size])).astype(np.int64)


def _check_source_label(path: Path, raw_label: str) -> None:
    """Confirm the file holds the process its name claims (cheap: two reads)."""

    with h5py.File(path, "r") as handle:
        labels = handle["label"]
        found = {labels[0], labels[len(labels) - 1]}
    decoded = {item.decode() if isinstance(item, bytes) else str(item) for item in found}
    if decoded != {raw_label}:
        raise ValueError(f"{path}: expected label {raw_label!r}, found {sorted(decoded)}")


def block_assignments(
    source_key: str,
    event_count: int,
    config: DataConfig,
) -> dict[str, list[EventSlice]]:
    """Allocate locality-preserving event blocks to splits, reproducibly.

    Events are cut into consecutive blocks of ``split_block_events``; a
    permutation seeded by ``(seed, source position)`` deals whole blocks to
    train/validation/test, and adjacent same-split blocks are merged. Whole
    blocks keep neighbouring (correlated) events in the same split.
    """

    block_size = int(config.split_block_events)
    blocks = [
        (start, min(start + block_size, event_count))
        for start in range(0, event_count, block_size)
    ]
    generator = np.random.default_rng(
        np.random.SeedSequence([int(config.seed), SOURCE_POSITION[source_key]])
    )
    permutation = generator.permutation(len(blocks))
    desired = np.asarray(config.split_fractions, dtype=np.float64) * len(blocks)
    base = np.floor(desired).astype(np.int64)
    remainder = len(blocks) - int(base.sum())
    if remainder:
        order = np.argsort(-(desired - base), kind="stable")
        base[order[:remainder]] += 1
    assigned = np.empty(len(blocks), dtype=np.int8)
    cursor = 0
    for split_index, count in enumerate(base.tolist()):
        assigned[permutation[cursor : cursor + count]] = split_index
        cursor += count

    result: dict[str, list[EventSlice]] = {name: [] for name in SPLIT_NAMES}
    current: int | None = None
    run_start = run_stop = 0
    for block_index, (start, stop) in enumerate(blocks):
        split_index = int(assigned[block_index])
        if current is None:
            current, run_start, run_stop = split_index, start, stop
        elif split_index == current and start == run_stop:
            run_stop = stop
        else:
            result[SPLIT_NAMES[current]].append(
                EventSlice(source_key, SPLIT_NAMES[current], run_start, run_stop)
            )
            current, run_start, run_stop = split_index, start, stop
    if current is not None:
        result[SPLIT_NAMES[current]].append(
            EventSlice(source_key, SPLIT_NAMES[current], run_start, run_stop)
        )
    return result


def _split_into_blocks(
    slices: Sequence[EventSlice],
    block_events: int,
) -> list[EventSlice]:
    """Cut slices into pieces of at most ``block_events`` events."""

    pieces: list[EventSlice] = []
    for item in slices:
        for start in range(item.event_start, item.event_stop, block_events):
            pieces.append(
                EventSlice(
                    item.source_key,
                    item.split,
                    start,
                    min(start + block_events, item.event_stop),
                )
            )
    return pieces


def _apportion(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Split ``total`` across keys in proportion to ``weights`` (largest remainder)."""

    weight_sum = sum(weights.values())
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    shares = {key: int(math.floor(value)) for key, value in exact.items()}
    leftover = total - sum(shares.values())
    for key in sorted(exact, key=lambda k: (-(exact[k] - shares[k]), k))[:leftover]:
        shares[key] += 1
    return shares


def _cap_split(
    per_source: Mapping[str, list[EventSlice]],
    cap: int,
    *,
    split: str,
    config: DataConfig,
) -> dict[str, list[EventSlice]]:
    """Keep about ``cap`` events of a split as seeded whole blocks per source."""

    sizes = {key: sum(item.event_count for item in items) for key, items in per_source.items()}
    total = sum(sizes.values())
    if cap >= total:
        return {key: list(items) for key, items in per_source.items()}
    quotas = _apportion(cap, sizes)
    split_index = SPLIT_NAMES.index(split)
    capped: dict[str, list[EventSlice]] = {}
    for key, items in per_source.items():
        blocks = _split_into_blocks(items, int(config.split_block_events))
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [int(config.seed), SOURCE_POSITION[key], split_index, 1]
            )
        )
        chosen: list[EventSlice] = []
        remaining = quotas[key]
        for block_index in generator.permutation(len(blocks)):
            if remaining <= 0:
                break
            block = blocks[int(block_index)]
            take = min(block.event_count, remaining)
            chosen.append(
                EventSlice(key, split, block.event_start, block.event_start + take)
            )
            remaining -= take
        capped[key] = sorted(chosen, key=lambda item: item.event_start)
    return capped


def build_split(
    event_counts: Mapping[str, int],
    config: DataConfig,
) -> dict[str, list[EventSlice]]:
    """Return the classification slices of every split for the given sources."""

    per_split: dict[str, dict[str, list[EventSlice]]] = {name: {} for name in SPLIT_NAMES}
    for spec in CLASSIFICATION_SOURCES:
        allocation = block_assignments(
            spec.source_key, int(event_counts[spec.source_key]), config
        )
        for split in SPLIT_NAMES:
            per_split[split][spec.source_key] = allocation[split]
    result: dict[str, list[EventSlice]] = {}
    for split in SPLIT_NAMES:
        sources = per_split[split]
        cap = config.event_cap(split)
        if cap is not None:
            sources = _cap_split(sources, int(cap), split=split, config=config)
        result[split] = [item for key in sources for item in sources[key]]
    return result


def split_counts(
    split_slices: Mapping[str, Sequence[EventSlice]],
) -> dict[str, dict[str, int]]:
    """Events per split and category, keyed like ``PUBLISHED_CLASSIFICATION_COUNTS``."""

    counts: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        by_category: dict[str, int] = {}
        for item in split_slices[split]:
            category = SOURCE_BY_KEY[item.source_key].category
            by_category[category] = by_category.get(category, 0) + item.event_count
        counts[split] = dict(sorted(by_category.items()))
    return counts


def verify_published_counts(counts: Mapping[str, Mapping[str, int]]) -> None:
    """Raise unless the counts equal the published SuperNEMO split exactly."""

    expected = {
        split: dict(sorted(by_category.items()))
        for split, by_category in PUBLISHED_CLASSIFICATION_COUNTS.items()
    }
    observed = {split: dict(sorted(by.items())) for split, by in counts.items()}
    if observed != expected:
        raise ValueError(
            "split does not reproduce the published SuperNEMO split; "
            f"expected {expected}, got {observed}"
        )


def _proportional_interleave(
    streams: Mapping[str, Iterator[dict[str, Any]]],
    assigned: Mapping[str, int],
    order: Sequence[str],
) -> Iterator[dict[str, Any]]:
    """Merge per-source streams so every prefix keeps the natural class ratio.

    Without this a block-shuffled epoch would emit one class for thousands of
    consecutive events and every mini-batch would be single-class.
    """

    emitted = {key: 0 for key in order}
    remaining = sum(assigned.values())
    while remaining:
        active = [key for key in order if emitted[key] < assigned[key]]
        chosen = active[0]
        for candidate in active[1:]:
            if emitted[candidate] * assigned[chosen] < emitted[chosen] * assigned[candidate]:
                chosen = candidate
        try:
            yield next(streams[chosen])
        except StopIteration as error:
            raise RuntimeError(
                f"{chosen}: event stream ended before its assigned count"
            ) from error
        emitted[chosen] += 1
        remaining -= 1


class SuperNEMOEventDataset(IterableDataset):
    """Stream tokenized events for one split from the raw HDF5 sources.

    ``shuffle=True`` (training) deals shuffled blocks per source and merges the
    sources proportionally; otherwise slices are visited in order. With
    ``num_workers > 0`` each worker takes a disjoint stride of blocks/slices,
    so every event is emitted exactly once per epoch.
    """

    def __init__(
        self,
        *,
        data_root: Path,
        slices: Sequence[EventSlice],
        offsets: Mapping[str, np.ndarray],
        tokenization_config: SuperNEMOTrackerTokenizationConfig,
        split: str,
        shuffle: bool,
        seed: int,
        block_events: int,
    ) -> None:
        super().__init__()
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {split!r}")
        if not isinstance(tokenization_config, SuperNEMOTrackerTokenizationConfig):
            raise TypeError(
                "tokenization_config must be a SuperNEMOTrackerTokenizationConfig"
            )
        if not slices:
            raise ValueError(f"no events in split {split!r}")
        self.data_root = Path(data_root)
        self.slices = list(slices)
        self.offsets = dict(offsets)
        self.tokenization_config = tokenization_config
        self.split = split
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.block_events = int(block_events)
        self.epoch = 0
        self._length = sum(item.event_count for item in self.slices)

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _make_sample(
        self,
        source_key: str,
        event_number: int,
        coordinates: np.ndarray,
        tracker_radius: np.ndarray,
        energy_kev: float,
    ) -> dict[str, Any]:
        spec = SOURCE_BY_KEY[source_key]
        event_id = f"SuperNEMO::{source_key}::{event_number}"
        # The source is one-to-one with the class, so it must not influence the
        # deterministic hit sampling; the local ordinal alone is the key.
        inputs, coverage = tokenize_tracker_event(
            coordinates,
            tracker_radius,
            sampling_key=f"SuperNEMO::ev_no::{event_number}",
            config=self.tokenization_config,
        )
        return {
            "inputs": inputs,
            "label": np.float32(spec.classification_label),
            "energy": np.float64(energy_kev / KEV_PER_MEV),
            "event_id": event_id,
            "group_id": event_id,
            "category": spec.category,
            "split": self.split,
            "projection_coverage": np.float32(coverage),
        }

    def _iter_slice(
        self,
        item: EventSlice,
        handle: h5py.File,
    ) -> Iterator[dict[str, Any]]:
        offsets = self.offsets[item.source_key]
        row_start = int(offsets[item.event_start])
        row_stop = int(offsets[item.event_stop])
        relative = offsets[item.event_start : item.event_stop + 1] - row_start
        counts = np.diff(relative)
        fields = (*COORDINATE_FIELDS, *ENERGY_FIELDS, RADIUS_FIELD)
        arrays = {
            field: np.asarray(handle[field][row_start:row_stop], dtype=np.float32)
            for field in fields
        }

        for field in COORDINATE_FIELDS:
            if not np.isfinite(arrays[field]).all():
                raise ValueError(f"{item.source_key}:{field} has a non-finite value")
        for field in ENERGY_FIELDS:
            values = arrays[field]
            if not np.isfinite(values).all() or (values <= 0.0).any():
                raise ValueError(
                    f"{item.source_key}:{field} must be finite and positive"
                )
            # E1/E2 are event constants repeated on every hit row.
            if not np.array_equal(values, np.repeat(values[relative[:-1]], counts)):
                raise ValueError(
                    f"{item.source_key}:{field} varies within an event "
                    f"in events {item.event_start}:{item.event_stop}"
                )
        radius = arrays[RADIUS_FIELD]
        if np.isinf(radius).any() or (np.isfinite(radius) & (radius < 0.0)).any():
            raise ValueError(f"{item.source_key}:{RADIUS_FIELD} has an invalid value")

        coordinates = np.column_stack([arrays[field] for field in COORDINATE_FIELDS])
        first_energy = arrays[ENERGY_FIELDS[0]][relative[:-1]].astype(np.float64)
        second_energy = arrays[ENERGY_FIELDS[1]][relative[:-1]].astype(np.float64)
        for local, event_number in enumerate(range(item.event_start, item.event_stop)):
            lo, hi = int(relative[local]), int(relative[local + 1])
            yield self._make_sample(
                item.source_key,
                event_number,
                coordinates[lo:hi],
                radius[lo:hi],
                float(first_energy[local] + second_energy[local]),
            )

    def _open(self, handles: dict[str, h5py.File], source_key: str) -> h5py.File:
        if source_key not in handles:
            handles[source_key] = h5py.File(
                self.data_root / SOURCE_BY_KEY[source_key].file_name, "r"
            )
        return handles[source_key]

    def _iter_training(
        self,
        worker: Any,
        handles: dict[str, h5py.File],
    ) -> Iterator[dict[str, Any]]:
        by_source: dict[str, list[EventSlice]] = {}
        for piece in _split_into_blocks(self.slices, self.block_events):
            by_source.setdefault(piece.source_key, []).append(piece)
        order = [key for key in SOURCE_POSITION if key in by_source]
        for key in order:
            generator = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, SOURCE_POSITION[key], 0])
            )
            generator.shuffle(by_source[key])
            if worker is not None:
                by_source[key] = by_source[key][worker.id :: worker.num_workers]
        # Only the tie-break between equally due sources varies by epoch.
        np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 1])).shuffle(
            order
        )
        order = [key for key in order if by_source[key]]
        assigned = {
            key: sum(piece.event_count for piece in by_source[key]) for key in order
        }
        # The handle is chosen from ``piece.source_key`` rather than the loop
        # variable ``key``: a generator expression evaluates ``key`` lazily, so
        # it would see only the last source and read every slice from the wrong
        # file.
        streams = {
            key: (
                sample
                for piece in by_source[key]
                for sample in self._iter_slice(
                    piece, self._open(handles, piece.source_key)
                )
            )
            for key in order
        }
        yield from _proportional_interleave(streams, assigned, order)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        handles: dict[str, h5py.File] = {}
        try:
            if self.shuffle:
                yield from self._iter_training(worker, handles)
                return
            slices = self.slices
            if worker is not None:
                slices = slices[worker.id :: worker.num_workers]
            for item in slices:
                yield from self._iter_slice(item, self._open(handles, item.source_key))
        finally:
            for handle in handles.values():
                handle.close()


def collate_events(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad tokens to the longest event in the batch; keep metadata outside inputs."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    counts = [len(item["inputs"]["coords"]) for item in samples]
    longest = max(counts)
    coord_dim = samples[0]["inputs"]["coords"].shape[1]
    feature_dim = samples[0]["inputs"]["features"].shape[1]
    coords = torch.zeros((len(samples), longest, coord_dim), dtype=torch.float32)
    features = torch.zeros((len(samples), longest, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(samples), longest), dtype=torch.bool)
    for index, (item, count) in enumerate(zip(samples, counts)):
        coords[index, :count] = torch.from_numpy(np.asarray(item["inputs"]["coords"]))
        features[index, :count] = torch.from_numpy(np.asarray(item["inputs"]["features"]))
        mask[index, :count] = True
    return {
        "inputs": {"coords": coords, "features": features, "mask": mask},
        "label": torch.as_tensor([item["label"] for item in samples], dtype=torch.float32),
        "energy": torch.as_tensor([item["energy"] for item in samples], dtype=torch.float64),
        "projection_coverage": torch.as_tensor(
            [item["projection_coverage"] for item in samples], dtype=torch.float32
        ),
        "event_id": [str(item["event_id"]) for item in samples],
        "group_id": [str(item["group_id"]) for item in samples],
        "category": [str(item["category"]) for item in samples],
        "split": [str(item["split"]) for item in samples],
    }


@dataclass(frozen=True)
class PreparedData:
    """Loaders and split bookkeeping for one tokenization."""

    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    counts: dict[str, dict[str, int]]
    slices: dict[str, list[EventSlice]]


def index_sources(data_config: DataConfig) -> dict[str, np.ndarray]:
    """Validate and index each classification source file; return row offsets."""

    offsets: dict[str, np.ndarray] = {}
    for spec in CLASSIFICATION_SOURCES:
        path = data_config.data_root / spec.file_name
        if not path.is_file():
            raise FileNotFoundError(
                f"missing SuperNEMO source {path}; set SUPERNEMO_ROPE_DATA_DIR "
                "to the directory holding the data_*_merged.h5 files"
            )
        _check_source_label(path, spec.raw_label)
        offsets[spec.source_key] = scan_event_offsets(path)
    return offsets


def prepare_dataset(
    *,
    data_config: DataConfig,
    tokenization_config: SuperNEMOTrackerTokenizationConfig,
    batch_size: int,
    num_workers: int,
    verify_published: bool | None = None,
) -> PreparedData:
    """Index the sources, build the split, and return the three loaders.

    ``verify_published`` defaults to on whenever ``data_config`` claims to be the
    published split (default seed/fractions/block size, no caps), so a data
    directory that does not match the published files fails loudly.
    """

    if int(batch_size) <= 0 or int(num_workers) < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    offsets = index_sources(data_config)
    event_counts = {key: len(value) - 1 for key, value in offsets.items()}
    split_slices = build_split(event_counts, data_config)
    counts = split_counts(split_slices)
    if verify_published is None:
        verify_published = data_config.is_published_split
    if verify_published:
        verify_published_counts(counts)

    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "shuffle": False,
        "collate_fn": collate_events,
    }
    loaders = {
        split: DataLoader(
            SuperNEMOEventDataset(
                data_root=data_config.data_root,
                slices=split_slices[split],
                offsets=offsets,
                tokenization_config=tokenization_config,
                split=split,
                shuffle=split == "train",
                seed=int(data_config.seed),
                block_events=int(data_config.split_block_events),
            ),
            **loader_options,
        )
        for split in SPLIT_NAMES
    }
    return PreparedData(
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        test_loader=loaders["test"],
        counts=counts,
        slices=split_slices,
    )


__all__ = [
    "EventSlice",
    "PreparedData",
    "SuperNEMOEventDataset",
    "block_assignments",
    "build_split",
    "collate_events",
    "index_sources",
    "prepare_dataset",
    "scan_event_offsets",
    "split_counts",
    "verify_published_counts",
]
