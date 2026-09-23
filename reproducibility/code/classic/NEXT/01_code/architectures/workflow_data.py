"""Bridge the standalone event-count split to ``next_alt`` representations.

The standalone workflow owns dataset discovery and split assignment.  Its
``FileSlice`` objects describe half-open *event ordinal* ranges, while the
alternative architectures need the original three-dimensional hit records.
This module joins those two contracts without attempting to reconstruct 3-D
events from the standalone workflow's 2-D projections.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from next_alt.config import INPUT_KINDS
from next_alt.data import (
    RepresentationConfig,
    padded_event_collate,
    represent_event,
)
from next_cnn.data import EventRecord, SourceFile, iter_file_events


SPLIT_NAMES = ("train", "validation", "test")

# These are deliberately the only tensors exposed below ``batch["inputs"]``.
# In particular, padding bookkeeping such as ``num_points`` is not a model
# input in the next_alt forward contract.
_MODEL_INPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "projection2d": ("projections",),
    "multiscale2d": ("projections", "fine_projections"),
    "dense3d": ("volume",),
    "points": ("coords", "features", "mask"),
    "graph": ("coords", "features", "mask"),
    "hybrid": ("projections", "coords", "features", "mask"),
    "sequence": ("coords", "features", "mask"),
    "topology": ("coords", "features", "mask"),
    "sparse3d": ("voxel_coords", "voxel_features", "voxel_mask"),
}


@dataclass(frozen=True)
class _SliceRecord:
    """Pickle-friendly structural copy of a standalone ``FileSlice``."""

    relative_path: str
    category: str
    label: int
    split: str
    event_start: int
    event_stop: int

    @property
    def event_count(self) -> int:
        return self.event_stop - self.event_start


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if converted != value or converted < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return converted


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"file slice is missing {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise ValueError(f"file slice is missing {name!r}")
    return getattr(value, name)


def _coerce_slice(value: Any, expected_split: str) -> _SliceRecord:
    relative_path = str(_field(value, "relative_path"))
    category = str(_field(value, "category"))
    split = str(_field(value, "split"))
    label = _integer(_field(value, "label"), "file slice label", minimum=0)
    event_start = _integer(
        _field(value, "event_start"), "file slice event_start", minimum=0
    )
    event_stop = _integer(
        _field(value, "event_stop"), "file slice event_stop", minimum=1
    )
    if not relative_path:
        raise ValueError("file slice relative_path must not be empty")
    if not category:
        raise ValueError("file slice category must not be empty")
    if split != expected_split:
        raise ValueError(
            f"{expected_split!r} loader contains a slice assigned to {split!r}"
        )
    if label not in (0, 1):
        raise ValueError("NEXT labels must be binary (0 or 1)")
    if event_stop <= event_start:
        raise ValueError("file slice event_stop must be greater than event_start")
    return _SliceRecord(
        relative_path=relative_path,
        category=category,
        label=label,
        split=split,
        event_start=event_start,
        event_stop=event_stop,
    )


def _manifest_slices(prepared_data: Any) -> Mapping[str, Sequence[Any]]:
    manifest_path = getattr(prepared_data, "manifest_path", None)
    if manifest_path is None:
        raise TypeError(
            "prepared_data loaders do not expose file_slices and no "
            "manifest_path is available"
        )
    path = Path(manifest_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits = payload["splits"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read split slices from manifest: {path}") from exc
    if not isinstance(splits, Mapping):
        raise ValueError("split manifest 'splits' must be a mapping")
    return splits


def _prepared_slices(prepared_data: Any) -> dict[str, list[_SliceRecord]]:
    """Read the exact ``FileSlice`` lists created by ``prepare_dataset``."""

    raw_by_split: dict[str, Sequence[Any]] = {}
    direct = getattr(prepared_data, "file_slices", None)
    if direct is None:
        direct = getattr(prepared_data, "split_slices", None)
    if isinstance(direct, Mapping):
        for split in SPLIT_NAMES:
            if split in direct:
                raw_by_split[split] = direct[split]

    loader_names = {
        "train": "train_loader",
        "validation": "validation_loader",
        "test": "test_loader",
    }
    for split, loader_name in loader_names.items():
        if split in raw_by_split:
            continue
        loader = getattr(prepared_data, loader_name, None)
        dataset = getattr(loader, "dataset", None)
        file_slices = getattr(dataset, "file_slices", None)
        if file_slices is not None:
            raw_by_split[split] = file_slices

    if len(raw_by_split) != len(SPLIT_NAMES):
        manifest_slices = _manifest_slices(prepared_data)
        for split in SPLIT_NAMES:
            if split not in raw_by_split:
                try:
                    raw_by_split[split] = manifest_slices[split]
                except KeyError as exc:
                    raise ValueError(
                        f"split manifest is missing {split!r} slices"
                    ) from exc

    result = {
        split: [_coerce_slice(item, split) for item in raw_by_split[split]]
        for split in SPLIT_NAMES
    }
    for split, slices in result.items():
        if not slices:
            raise ValueError(f"prepared_data {split!r} split contains no events")

    # A malformed/custom PreparedData must not silently duplicate an event.
    ranges: dict[str, list[tuple[int, int, str, int, str]]] = {}
    for split, slices in result.items():
        for item in slices:
            ranges.setdefault(item.relative_path, []).append(
                (
                    item.event_start,
                    item.event_stop,
                    split,
                    item.label,
                    item.category,
                )
            )
    for relative_path, file_ranges in ranges.items():
        ordered = sorted(file_ranges)
        previous_stop = -1
        previous_label: int | None = None
        previous_category: str | None = None
        for start, stop, _, label, category in ordered:
            if start < previous_stop:
                raise ValueError(
                    f"overlapping event slices would duplicate events in "
                    f"{relative_path!r}"
                )
            if previous_label is not None and (
                label != previous_label or category != previous_category
            ):
                raise ValueError(
                    f"inconsistent class metadata for {relative_path!r}"
                )
            previous_stop = stop
            previous_label = label
            previous_category = category

    counts = getattr(prepared_data, "counts", None)
    if isinstance(counts, Mapping):
        for split, slices in result.items():
            if split not in counts:
                continue
            expected = _integer(
                counts[split], f"prepared_data.counts[{split!r}]", minimum=0
            )
            actual = sum(item.event_count for item in slices)
            if actual != expected:
                raise ValueError(
                    f"{split} FileSlice count is {actual}, expected {expected}"
                )
    return result


def _prepared_seed(prepared_data: Any) -> int:
    loader = getattr(prepared_data, "train_loader", None)
    dataset = getattr(loader, "dataset", None)
    raw_seed = getattr(dataset, "seed", getattr(prepared_data, "seed", 42))
    return _integer(raw_seed, "prepared-data seed", minimum=0)


def _buffer_shuffle(
    stream: Iterator[EventRecord],
    buffer_size: int,
    generator: np.random.Generator,
) -> Iterator[EventRecord]:
    buffer: list[EventRecord] = []
    for event in stream:
        if len(buffer) < buffer_size:
            buffer.append(event)
            continue
        selected = int(generator.integers(0, len(buffer)))
        yield buffer[selected]
        buffer[selected] = event
    generator.shuffle(buffer)
    yield from buffer


class ArchitectureEventDataset(IterableDataset):
    """Stream exact event-count slices as one ``next_alt`` input kind."""

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        file_slices: Sequence[_SliceRecord],
        *,
        input_kind: str,
        representation: Mapping[str, Any] | RepresentationConfig | None,
        seed: int,
        shuffle_slices: bool,
        shuffle_buffer_size: int,
    ) -> None:
        super().__init__()
        root = Path(data_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"NEXT data directory does not exist: {root}")
        if input_kind not in INPUT_KINDS:
            raise ValueError(f"unsupported input_kind: {input_kind!r}")
        if not file_slices:
            raise ValueError("ArchitectureEventDataset needs at least one file slice")

        self.data_root = root
        self.file_slices = list(file_slices)
        self.input_kind = str(input_kind)
        self.representation = RepresentationConfig.from_mapping(representation)
        self.seed = _integer(seed, "seed", minimum=0)
        self.shuffle_slices = bool(shuffle_slices)
        self.shuffle_buffer_size = _integer(
            shuffle_buffer_size, "shuffle_buffer_size", minimum=0
        )
        self.epoch = 0
        self._length = sum(item.event_count for item in self.file_slices)

        # Resolve every target before worker processes start.  Besides clearer
        # errors, the containment check prevents a malformed relative path from
        # escaping the selected dataset root.
        for item in self.file_slices:
            source_path = (self.data_root / item.relative_path).resolve()
            try:
                source_path.relative_to(self.data_root)
            except ValueError as exc:
                raise ValueError(
                    f"file slice escapes data_root: {item.relative_path!r}"
                ) from exc
            if not source_path.is_file():
                raise FileNotFoundError(f"manifest source file is missing: {source_path}")

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible slice and buffer order for one epoch."""

        self.epoch = _integer(epoch, "epoch", minimum=0)

    def _slice_events(self, item: _SliceRecord) -> Iterator[EventRecord]:
        source = SourceFile(
            path=(self.data_root / item.relative_path).resolve(),
            relative_path=item.relative_path,
            group_id=item.relative_path,
            label=item.label,
            category=item.category,
            split=item.split,
        )
        produced = 0
        for ordinal, event in enumerate(iter_file_events(source)):
            if ordinal < item.event_start:
                continue
            if ordinal >= item.event_stop:
                break
            produced += 1
            yield event
        if produced != item.event_count:
            raise ValueError(
                f"slice {item.relative_path!r} [{item.event_start}, "
                f"{item.event_stop}) requested {item.event_count} events but "
                f"the source produced {produced}"
            )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        slices = list(self.file_slices)
        if self.shuffle_slices:
            slice_generator = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch])
            )
            slice_generator.shuffle(slices)

        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        if worker is not None:
            # Shard complete, non-overlapping FileSlices.  Each event is owned
            # by exactly one worker, including slices of a boundary file.
            slices = slices[worker.id :: worker.num_workers]

        events: Iterator[EventRecord] = chain.from_iterable(
            self._slice_events(item) for item in slices
        )
        if self.shuffle_slices and self.shuffle_buffer_size > 0:
            events = _buffer_shuffle(
                events,
                self.shuffle_buffer_size,
                np.random.default_rng(
                    np.random.SeedSequence([self.seed, self.epoch, worker_id])
                ),
            )
        for event in events:
            yield represent_event(event, self.input_kind, self.representation)


def architecture_event_collate(
    samples: Sequence[Mapping[str, Any]], *, input_kind: str
) -> dict[str, Any]:
    """Collate one alternative representation into the EnergyBench contract."""

    if input_kind not in _MODEL_INPUT_FIELDS:
        raise ValueError(f"unsupported input_kind: {input_kind!r}")
    collated = padded_event_collate(samples)
    inputs: dict[str, torch.Tensor] = {}
    for field in _MODEL_INPUT_FIELDS[input_kind]:
        value = collated.get(field)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"collated {input_kind!r} input {field!r} must be a tensor"
            )
        inputs[field] = value

    coverage = collated.get(
        "projection_coverage", collated.get("representation_coverage")
    )
    if not isinstance(coverage, torch.Tensor):
        raise TypeError("collated representation must provide tensor coverage")
    label = collated["label"].to(dtype=torch.float32)
    return {
        "inputs": inputs,
        "label": label,
        "energy": collated["energy_target"].to(dtype=torch.float64),
        "event_id": collated["event_id"],
        "category": collated["category"],
        "group_id": collated["group_id"],
        "split": collated["split"],
        "sample_weight": torch.ones_like(label, dtype=torch.float32),
        "projection_coverage": coverage.to(dtype=torch.float32),
    }


def build_architecture_loaders(
    prepared_data: Any,
    data_root: str | os.PathLike[str],
    input_kind: str,
    representation: Mapping[str, Any] | RepresentationConfig | None,
    batch_size: int,
    num_workers: int,
    shuffle_buffer_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build alternative-model loaders for the standalone workflow's splits.

    ``prepared_data`` must be the object returned by the standalone
    :func:`prepare_dataset` (or an object exposing the same loader/slice
    attributes).  The returned tuple is ``(train, validation, test)``.
    """

    kind = str(input_kind)
    if kind not in INPUT_KINDS or kind not in _MODEL_INPUT_FIELDS:
        raise ValueError(f"unsupported input_kind: {kind!r}")
    selected_batch_size = _integer(batch_size, "batch_size", minimum=1)
    selected_num_workers = _integer(num_workers, "num_workers", minimum=0)
    selected_buffer_size = _integer(
        shuffle_buffer_size, "shuffle_buffer_size", minimum=0
    )
    slices = _prepared_slices(prepared_data)
    seed = _prepared_seed(prepared_data)
    selected_representation = RepresentationConfig.from_mapping(representation)

    datasets = {
        split: ArchitectureEventDataset(
            data_root,
            slices[split],
            input_kind=kind,
            representation=selected_representation,
            seed=seed,
            shuffle_slices=split == "train",
            shuffle_buffer_size=(
                selected_buffer_size if split == "train" else 0
            ),
        )
        for split in SPLIT_NAMES
    }

    # functools.partial would also be pickle-safe, but this small callable
    # keeps DataLoader reprs readable and carries no dataset state.
    collate = _ArchitectureCollate(kind)
    options = {
        "batch_size": selected_batch_size,
        "num_workers": selected_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "collate_fn": collate,
    }
    loaders = {
        split: DataLoader(datasets[split], **options) for split in SPLIT_NAMES
    }
    return (
        loaders["train"],
        loaders["validation"],
        loaders["test"],
    )


@dataclass(frozen=True)
class _ArchitectureCollate:
    input_kind: str

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return architecture_event_collate(samples, input_kind=self.input_kind)


__all__ = [
    "ArchitectureEventDataset",
    "architecture_event_collate",
    "build_architecture_loaders",
]
