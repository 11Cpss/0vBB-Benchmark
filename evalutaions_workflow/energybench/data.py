"""Event-count splitting and lazy NEXT HDF5 loading.

The split manifest stores only relative HDF5 paths, event counts, and
half-open event-ordinal ranges.  Hit data remain in the source files and are
read lazily when a :class:`~torch.utils.data.DataLoader` is iterated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import ProjectionConfig


DEFAULT_DATA_ROOT = Path(
    os.environ.get("SIMPLE_ENERGYBENCH_DATA", "data/NEXT")
).expanduser()
DEFAULT_CLASS_MAP: dict[str, int] = {"0nubb": 1, "Bi214": 0}
SPLIT_NAMES = ("train", "validation", "test")
HDF5_TABLE = "MC/hits/table"
MANIFEST_VERSION = 1
InputMode = Literal[
    "classification", "regression_energy", "regression_topology"
]

EventInputBuilder = Callable[..., tuple[Any, float]]


@dataclass(frozen=True)
class FileSlice:
    """A half-open range of event ordinals from one HDF5 file."""

    relative_path: str
    category: str
    label: int
    split: str
    event_start: int
    event_stop: int

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {self.split!r}")
        if self.label not in (0, 1):
            raise ValueError("NEXT labels must be binary (0 or 1)")
        if self.event_start < 0 or self.event_stop <= self.event_start:
            raise ValueError("a file slice must contain a non-empty event range")

    @property
    def event_count(self) -> int:
        """Number of events represented by this slice."""

        return self.event_stop - self.event_start

    def to_dict(self) -> dict[str, Any]:
        """Return the compact form stored in the JSON manifest."""

        return {
            "relative_path": self.relative_path,
            "category": self.category,
            "label": self.label,
            "split": self.split,
            "event_start": self.event_start,
            "event_stop": self.event_stop,
            "event_count": self.event_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FileSlice":
        """Construct a validated slice from a manifest entry."""

        return cls(
            relative_path=str(payload["relative_path"]),
            category=str(payload["category"]),
            label=int(payload["label"]),
            split=str(payload["split"]),
            event_start=int(payload["event_start"]),
            event_stop=int(payload["event_stop"]),
        )


@dataclass(frozen=True)
class PreparedData:
    """DataLoaders and bookkeeping returned by :func:`prepare_dataset`."""

    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    counts: dict[str, Any]
    manifest_path: Path
    mode: InputMode
    projection: ProjectionConfig

    @property
    def val_loader(self) -> DataLoader:
        """Short alias for ``validation_loader``."""

        return self.validation_loader


@dataclass(frozen=True)
class _InventoryFile:
    path: Path
    relative_path: str
    category: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _CountedFile:
    relative_path: str
    category: str
    label: int
    event_count: int
    size: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "category": self.category,
            "label": self.label,
            "event_count": self.event_count,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


def prepare_dataset(
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
    *,
    batch_size: int = 64,
    projection: ProjectionConfig | None = None,
    mode: InputMode = "classification",
    split_fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    num_workers: int = 0,
    manifest_path: str | os.PathLike[str] | None = None,
    max_files_per_class: int | None = None,
    class_map: Mapping[str, int] | None = None,
    shuffle_buffer_size: int = 0,
    pin_memory: bool | None = None,
    chunk_rows: int = 262_144,
    verbose: bool = True,
    input_builder: EventInputBuilder | None = None,
) -> PreparedData:
    """Prepare reproducible event-level train/validation/test DataLoaders.

    Files are shuffled separately per class, then partition boundaries are
    placed at the nearest cumulative event-count targets.  A boundary may cut
    through a file; its two event-ordinal ranges then belong to different
    partitions.  No event or hit data are copied into the manifest.

    Args:
        data_root: Directory containing ``0nubb_part_*`` and ``Bi_part_*``.
        batch_size: DataLoader batch size.
        projection: Detector projection geometry.  The energy representation
            is selected automatically from ``mode``.
        mode: ``classification`` (energy-normalized), ``regression_energy``
            (energy-preserving), or ``regression_topology`` (binary occupancy).
        split_fractions: Train, validation, and test fractions.
        seed: File-shuffle and training-stream seed.
        num_workers: DataLoader worker processes.
        manifest_path: JSON cache path.  By default it is stored alongside the
            standalone package, not in the possibly read-only dataset.
        max_files_per_class: Optional deterministic file subset for controlled studies.
        class_map: Binary labels for the ``0nubb`` and ``Bi214`` categories.
        shuffle_buffer_size: Optional bounded event shuffle for training.
        pin_memory: Defaults to whether CUDA is available.
        chunk_rows: Number of HDF5 hit rows read at a time.
        verbose: Print event-count and boundary-file summaries.
        input_builder: Optional callable that converts one complete raw event
            into model-specific inputs. It receives ``coordinates``,
            ``energies``, and the stable ``event_id`` and returns
            ``(inputs, coverage)``. When omitted, the existing three-view CNN
            projection is used. This option does not alter the split manifest,
            labels, event energies, or evaluation metadata.
    """

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NEXT data directory does not exist: {root}")
    _positive_integer(batch_size, "batch_size")
    _nonnegative_integer(num_workers, "num_workers")
    _nonnegative_integer(shuffle_buffer_size, "shuffle_buffer_size")
    _positive_integer(chunk_rows, "chunk_rows")
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if max_files_per_class is not None:
        _positive_integer(max_files_per_class, "max_files_per_class")

    fractions = _validate_fractions(split_fractions)
    labels = _validate_class_map(class_map)
    selected_projection, selected_mode = _projection_for_mode(projection, mode)
    inventory, fingerprint = _build_inventory(root)
    cache_path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else Path(__file__).resolve().parents[1] / "split_manifest.json"
    )
    settings = {
        "split_fractions": list(fractions),
        "seed": int(seed),
        "class_map": dict(sorted(labels.items())),
        "max_files_per_class": max_files_per_class,
    }

    manifest = _load_compatible_manifest(cache_path, fingerprint, settings)
    if manifest is None:
        counted = _count_selected_files(
            inventory,
            labels,
            seed=int(seed),
            max_files_per_class=max_files_per_class,
            chunk_rows=int(chunk_rows),
            verbose=verbose,
        )
        slices, counts, boundary_files = _allocate_slices(counted, fractions)
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "inventory": fingerprint,
            "settings": settings,
            "files": [item.to_dict() for item in counted],
            "splits": {
                name: [item.to_dict() for item in slices[name]]
                for name in SPLIT_NAMES
            },
            "counts": counts,
            "boundary_files": boundary_files,
        }
        _write_manifest(cache_path, manifest)
        if verbose:
            print(f"Wrote event split manifest: {cache_path}")
    elif verbose:
        print(f"Using cached event split manifest: {cache_path}")

    split_slices = {
        name: [FileSlice.from_dict(item) for item in manifest["splits"][name]]
        for name in SPLIT_NAMES
    }
    _validate_manifest_accounting(split_slices, manifest["counts"])
    counts = dict(manifest["counts"])
    counts["boundary_files"] = list(manifest.get("boundary_files", []))

    datasets = {
        name: NextEventDataset(
            root,
            split_slices[name],
            projection=selected_projection,
            mode=selected_mode,
            seed=int(seed),
            shuffle_slices=name == "train",
            shuffle_buffer_size=(
                int(shuffle_buffer_size) if name == "train" else 0
            ),
            chunk_rows=int(chunk_rows),
            input_builder=input_builder,
        )
        for name in SPLIT_NAMES
    }
    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "drop_last": False,
    }
    loaders = {
        name: DataLoader(datasets[name], **loader_options) for name in SPLIT_NAMES
    }
    if verbose:
        _report_counts(counts)
    return PreparedData(
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        test_loader=loaders["test"],
        counts=counts,
        manifest_path=cache_path,
        mode=selected_mode,
        projection=selected_projection,
    )


class NextEventDataset(IterableDataset):
    """Lazy event stream over manifest file slices."""

    def __init__(
        self,
        data_root: Path,
        file_slices: Sequence[FileSlice],
        *,
        projection: ProjectionConfig,
        mode: InputMode,
        seed: int,
        shuffle_slices: bool,
        shuffle_buffer_size: int,
        chunk_rows: int,
        input_builder: EventInputBuilder | None = None,
    ) -> None:
        super().__init__()
        if not file_slices:
            raise ValueError("each split must contain at least one event")
        self.data_root = Path(data_root)
        self.file_slices = list(file_slices)
        self.projection = projection
        self.mode = mode
        self.seed = int(seed)
        self.shuffle_slices = bool(shuffle_slices)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.chunk_rows = int(chunk_rows)
        self.input_builder = input_builder
        self.epoch = 0
        self._length = sum(item.event_count for item in self.file_slices)

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic training order for one epoch."""

        if isinstance(epoch, bool) or int(epoch) != epoch or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        slices = list(self.file_slices)
        if self.shuffle_slices:
            generator = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch])
            )
            generator.shuffle(slices)
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        if worker is not None:
            slices = slices[worker.id :: worker.num_workers]
        stream = chain.from_iterable(
            _iter_file_slice(
                self.data_root,
                file_slice,
                projection=self.projection,
                chunk_rows=self.chunk_rows,
                input_builder=self.input_builder,
            )
            for file_slice in slices
        )
        if self.shuffle_slices and self.shuffle_buffer_size > 0:
            yield from _buffer_shuffle(
                stream,
                self.shuffle_buffer_size,
                np.random.default_rng(
                    np.random.SeedSequence([self.seed, self.epoch, worker_id])
                ),
            )
        else:
            yield from stream


def project_event(
    coordinates: np.ndarray,
    energies: np.ndarray,
    config: ProjectionConfig,
) -> tuple[np.ndarray, float]:
    """Create the three ``XY/XZ/YZ`` projections for one event."""

    xyz = np.asarray(coordinates, dtype=np.float32)
    weights = np.asarray(energies, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_hits, 3)")
    if weights.ndim != 1 or len(weights) != len(xyz) or len(weights) == 0:
        raise ValueError("energies must have shape (n_hits,) for a non-empty event")
    if np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(weights)):
        raise ValueError("coordinates and energies must be finite")
    if np.any(weights < 0.0):
        raise ValueError("hit energies must be non-negative")
    total_energy = float(np.sum(weights, dtype=np.float64))
    if not math.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("event total energy must be finite and positive")

    origin = np.asarray(config.origin, dtype=np.float32)
    indices = np.floor(
        (xyz - origin) / np.float32(config.bin_size)
    ).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < config.grid_size), axis=1)
    image = np.zeros(
        (3, config.grid_size, config.grid_size), dtype=np.float32
    )
    if np.any(valid):
        x_index, y_index, z_index = indices[valid].T
        deposited = (
            weights[valid]
            if config.representation == "energy"
            else np.ones(np.count_nonzero(valid), dtype=np.float32)
        )
        np.add.at(image[0], (y_index, x_index), deposited)
        np.add.at(image[1], (z_index, x_index), deposited)
        np.add.at(image[2], (z_index, y_index), deposited)

    if config.representation == "binary_occupancy":
        np.minimum(image, np.float32(1.0), out=image)
        coverage = float(np.count_nonzero(valid)) / float(len(valid))
    else:
        kept_energy = float(np.sum(weights[valid], dtype=np.float64))
        coverage = kept_energy / total_energy
        if config.normalize_energy:
            image /= np.float32(total_energy)
    image *= np.float32(config.input_scale)
    return image, coverage


def count_file_events(
    path: str | os.PathLike[str], *, chunk_rows: int = 262_144
) -> int:
    """Count contiguous event-ID runs without loading a whole HDF5 table."""

    source = Path(path)
    _positive_integer(chunk_rows, "chunk_rows")
    with h5py.File(source, "r") as handle:
        if HDF5_TABLE not in handle:
            raise ValueError(f"missing /{HDF5_TABLE} in {source}")
        dataset = handle[HDF5_TABLE]
        _validate_hdf_schema(dataset, source)
        if len(dataset) == 0:
            raise ValueError(f"NEXT HDF5 table is empty: {source}")
        previous_id: int | None = None
        seen: set[int] = set()
        event_count = 0
        for row_start in range(0, len(dataset), int(chunk_rows)):
            event_ids = _read_event_ids(
                dataset, row_start, min(row_start + int(chunk_rows), len(dataset))
            )
            internal = np.flatnonzero(event_ids[1:] != event_ids[:-1]) + 1
            if previous_id is None or int(event_ids[0]) != previous_id:
                starts = np.r_[0, internal]
            else:
                starts = internal
            for event_id in event_ids[starts]:
                numeric_id = int(event_id)
                if numeric_id in seen:
                    raise ValueError(f"event IDs are not contiguous in {source}")
                seen.add(numeric_id)
            event_count += int(len(starts))
            previous_id = int(event_ids[-1])
    return event_count


def _build_inventory(
    root: Path,
) -> tuple[list[_InventoryFile], dict[str, Any]]:
    paths = list(root.glob("0nubb_part_*/*.h5"))
    paths.extend(root.glob("Bi_part_*/*.h5"))
    if not paths:
        raise FileNotFoundError(f"no NEXT HDF5 files found below {root}")
    inventory: list[_InventoryFile] = []
    digest = hashlib.sha256()
    category_counts = {category: 0 for category in DEFAULT_CLASS_MAP}
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix()
        category = _category_from_path(path)
        stat = path.stat()
        item = _InventoryFile(
            path=path,
            relative_path=relative_path,
            category=category,
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )
        inventory.append(item)
        category_counts[category] += 1
        for value in (relative_path, str(item.size), str(item.mtime_ns)):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return inventory, {
        "algorithm": "sha256-relative-path-size-mtime-ns-v1",
        "digest": digest.hexdigest(),
        "file_count": len(inventory),
        "category_counts": category_counts,
    }


def _count_selected_files(
    inventory: Sequence[_InventoryFile],
    class_map: Mapping[str, int],
    *,
    seed: int,
    max_files_per_class: int | None,
    chunk_rows: int,
    verbose: bool,
) -> list[_CountedFile]:
    selected: list[_InventoryFile] = []
    for category in DEFAULT_CLASS_MAP:
        category_files = sorted(
            (item for item in inventory if item.category == category),
            key=lambda item: item.relative_path,
        )
        category_seed = int.from_bytes(
            hashlib.sha256(category.encode("utf-8")).digest()[:4], "little"
        )
        generator = np.random.default_rng(
            np.random.SeedSequence([seed, category_seed])
        )
        generator.shuffle(category_files)
        if max_files_per_class is not None:
            category_files = category_files[: int(max_files_per_class)]
        if not category_files:
            raise ValueError(f"no selected files for category {category!r}")
        selected.extend(category_files)

    if verbose:
        print(f"Counting events in {len(selected):,} HDF5 files ...")
    counted: list[_CountedFile] = []
    progress_step = max(1, min(250, len(selected) // 10 or 1))
    for index, item in enumerate(selected, start=1):
        counted.append(
            _CountedFile(
                relative_path=item.relative_path,
                category=item.category,
                label=int(class_map[item.category]),
                event_count=count_file_events(item.path, chunk_rows=chunk_rows),
                size=item.size,
                mtime_ns=item.mtime_ns,
            )
        )
        if verbose and (index % progress_step == 0 or index == len(selected)):
            print(f"  counted {index:,}/{len(selected):,} files")
    return counted


def _allocate_slices(
    counted: Sequence[_CountedFile],
    fractions: tuple[float, float, float],
) -> tuple[dict[str, list[FileSlice]], dict[str, Any], list[str]]:
    slices: dict[str, list[FileSlice]] = {name: [] for name in SPLIT_NAMES}
    class_counts: dict[str, dict[str, int]] = {}
    for category in DEFAULT_CLASS_MAP:
        files = [item for item in counted if item.category == category]
        total = sum(item.event_count for item in files)
        targets = _nearest_partition_counts(total, fractions)
        boundaries = np.cumsum([0, *targets], dtype=np.int64)
        cursor = 0
        for source in files:
            file_start = cursor
            file_stop = cursor + source.event_count
            for split_index, split in enumerate(SPLIT_NAMES):
                overlap_start = max(file_start, int(boundaries[split_index]))
                overlap_stop = min(file_stop, int(boundaries[split_index + 1]))
                if overlap_stop > overlap_start:
                    slices[split].append(
                        FileSlice(
                            relative_path=source.relative_path,
                            category=source.category,
                            label=source.label,
                            split=split,
                            event_start=overlap_start - file_start,
                            event_stop=overlap_stop - file_start,
                        )
                    )
            cursor = file_stop
        class_counts[category] = {
            "total": total,
            **{name: targets[index] for index, name in enumerate(SPLIT_NAMES)},
        }

    split_totals = {
        name: sum(item.event_count for item in slices[name]) for name in SPLIT_NAMES
    }
    total_events = sum(split_totals.values())
    if total_events <= 0:
        raise ValueError("the selected dataset contains no events")
    for name in SPLIT_NAMES:
        if split_totals[name] == 0:
            raise ValueError(
                f"the {name} split is empty; select more files/events or change fractions"
            )
    counts: dict[str, Any] = {
        "total": total_events,
        **split_totals,
        "fractions": {
            name: split_totals[name] / total_events for name in SPLIT_NAMES
        },
        "by_class": class_counts,
    }
    membership: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        for item in slices[split]:
            membership.setdefault(item.relative_path, set()).add(split)
    boundary_files = sorted(
        relative for relative, split_set in membership.items() if len(split_set) > 1
    )
    return slices, counts, boundary_files


def _nearest_partition_counts(
    total: int, fractions: tuple[float, float, float]
) -> tuple[int, int, int]:
    train_end = int(math.floor(total * fractions[0] + 0.5))
    validation_end = int(
        math.floor(total * (fractions[0] + fractions[1]) + 0.5)
    )
    return train_end, validation_end - train_end, total - validation_end


def _iter_file_slice(
    data_root: Path,
    file_slice: FileSlice,
    *,
    projection: ProjectionConfig,
    chunk_rows: int,
    input_builder: EventInputBuilder | None = None,
) -> Iterator[dict[str, Any]]:
    path = data_root / file_slice.relative_path
    if not path.is_file():
        raise FileNotFoundError(f"manifest source file is missing: {path}")
    expected_label = b"Signal" if file_slice.category == "0nubb" else b"Bkg"
    with h5py.File(path, "r") as handle:
        if HDF5_TABLE not in handle:
            raise ValueError(f"missing /{HDF5_TABLE} in {path}")
        dataset = handle[HDF5_TABLE]
        _validate_hdf_schema(dataset, path)
        current_id: int | None = None
        current_ordinal = -1
        coordinate_parts: list[np.ndarray] = []
        energy_parts: list[np.ndarray] = []
        seen: set[int] = set()

        def finish_current() -> dict[str, Any] | None:
            if current_id is None or not (
                file_slice.event_start <= current_ordinal < file_slice.event_stop
            ):
                return None
            coordinates = (
                coordinate_parts[0]
                if len(coordinate_parts) == 1
                else np.concatenate(coordinate_parts, axis=0)
            )
            energies = (
                energy_parts[0]
                if len(energy_parts) == 1
                else np.concatenate(energy_parts, axis=0)
            )
            return _event_sample(
                coordinates,
                energies,
                source_event_id=current_id,
                file_slice=file_slice,
                projection=projection,
                input_builder=input_builder,
            )

        for row_start in range(0, len(dataset), chunk_rows):
            rows = dataset[row_start : min(row_start + chunk_rows, len(dataset))]
            event_ids = np.asarray(rows["values_block_0"][:, 0], dtype=np.int64)
            values = np.asarray(rows["values_block_1"], dtype=np.float32)
            labels = np.asarray(rows["values_block_2"][:, 0]).astype("S")
            if values.ndim != 2 or values.shape[1] != 4:
                raise ValueError(f"invalid x/y/z/energy block in {path}")
            if np.any(~np.isfinite(values)) or np.any(values[:, 3] < 0.0):
                raise ValueError(f"invalid coordinates or energies in {path}")
            if np.any(labels != expected_label):
                found = sorted(
                    value.decode("utf-8", errors="replace")
                    for value in np.unique(labels)
                )
                raise ValueError(
                    f"label mismatch in {path}: expected "
                    f"{expected_label.decode()!r}, found {found!r}"
                )
            starts = np.r_[0, np.flatnonzero(event_ids[1:] != event_ids[:-1]) + 1]
            stops = np.r_[starts[1:], len(event_ids)]
            for start, stop in zip(starts, stops):
                source_event_id = int(event_ids[start])
                if current_id != source_event_id:
                    sample = finish_current()
                    if sample is not None:
                        yield sample
                    if current_ordinal + 1 >= file_slice.event_stop:
                        return
                    if source_event_id in seen:
                        raise ValueError(f"event IDs are not contiguous in {path}")
                    seen.add(source_event_id)
                    current_id = source_event_id
                    current_ordinal += 1
                    coordinate_parts = []
                    energy_parts = []
                if file_slice.event_start <= current_ordinal < file_slice.event_stop:
                    coordinate_parts.append(values[start:stop, :3])
                    energy_parts.append(values[start:stop, 3])
        sample = finish_current()
        if sample is not None:
            yield sample


def _event_sample(
    coordinates: np.ndarray,
    energies: np.ndarray,
    *,
    source_event_id: int,
    file_slice: FileSlice,
    projection: ProjectionConfig,
    input_builder: EventInputBuilder | None = None,
) -> dict[str, Any]:
    total_energy = float(np.sum(energies, dtype=np.float64))
    if not math.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError(
            f"event {source_event_id} in {file_slice.relative_path} "
            "has non-positive energy"
        )
    event_id = f"NEXT::{file_slice.relative_path}::{source_event_id}"

    if input_builder is None:
        # Default representation used by the CNN baselines.
        inputs, coverage = project_event(coordinates, energies, projection)
    else:
        # Architecture-specific representation, such as padded point tokens.
        inputs, coverage = input_builder(
            coordinates,
            energies,
            event_id=event_id,
        )

    coverage = float(coverage)
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError(
            "input_builder coverage must be finite and between 0 and 1"
        )

    return {
        "inputs": inputs,
        "label": np.float32(file_slice.label),
        "energy": np.float64(total_energy),
        "event_id": event_id,
        "category": file_slice.category,
        "group_id": file_slice.relative_path,
        "split": file_slice.split,
        "sample_weight": np.float32(1.0),
        "projection_coverage": np.float32(coverage),
    }


def _buffer_shuffle(
    stream: Iterator[dict[str, Any]],
    buffer_size: int,
    generator: np.random.Generator,
) -> Iterator[dict[str, Any]]:
    buffer: list[dict[str, Any]] = []
    for item in stream:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = int(generator.integers(0, len(buffer)))
        yield buffer[index]
        buffer[index] = item
    generator.shuffle(buffer)
    yield from buffer


def _projection_for_mode(
    projection: ProjectionConfig | None, mode: str
) -> tuple[ProjectionConfig, InputMode]:
    base = ProjectionConfig() if projection is None else projection
    if not isinstance(base, ProjectionConfig):
        raise TypeError("projection must be a ProjectionConfig")
    aliases = {
        "classification": "classification",
        "regression": "regression_energy",
        "regression_energy": "regression_energy",
        "regression_topology": "regression_topology",
    }
    try:
        selected: InputMode = aliases[str(mode).strip().lower()]  # type: ignore[assignment]
    except KeyError as exc:
        raise ValueError(
            "mode must be classification, regression_energy, or regression_topology"
        ) from exc
    if selected == "classification":
        return replace(base, representation="energy", normalize_energy=True), selected
    if selected == "regression_energy":
        return replace(base, representation="energy", normalize_energy=False), selected
    return replace(
        base,
        representation="binary_occupancy",
        normalize_energy=False,
        input_scale=1.0,
    ), selected


def _category_from_path(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent.startswith("0nubb_part_"):
        return "0nubb"
    if parent.startswith("bi_part_"):
        return "Bi214"
    raise ValueError(f"cannot infer NEXT category from {path.parent}")


def _validate_hdf_schema(dataset: h5py.Dataset, path: Path) -> None:
    expected = ("index", "values_block_0", "values_block_1", "values_block_2")
    if dataset.ndim != 1 or dataset.dtype.names != expected:
        raise ValueError(f"unexpected NEXT table fields in {path}: {dataset.dtype}")
    event_shape = dataset.dtype.fields["values_block_0"][0].shape
    value_shape = dataset.dtype.fields["values_block_1"][0].shape
    label_shape = dataset.dtype.fields["values_block_2"][0].shape
    if event_shape != (1,) or value_shape != (4,) or label_shape != (1,):
        raise ValueError(f"unexpected NEXT table block shapes in {path}")
    block_0 = _attribute_text(dataset.attrs.get("values_block_0_kind", ""))
    block_1 = _attribute_text(dataset.attrs.get("values_block_1_kind", ""))
    block_2 = _attribute_text(dataset.attrs.get("values_block_2_kind", ""))
    if block_0 and "Vevent_id" not in block_0:
        raise ValueError(f"NEXT event_id metadata is invalid in {path}")
    if block_2 and "Vlabel" not in block_2:
        raise ValueError(f"NEXT label metadata is invalid in {path}")
    if block_1:
        positions = [block_1.find("V" + name) for name in ("x", "y", "z", "energy")]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            raise ValueError(f"NEXT x/y/z/energy metadata is invalid in {path}")


def _read_event_ids(
    dataset: h5py.Dataset, start: int, stop: int
) -> np.ndarray:
    rows = dataset.fields("values_block_0")[start:stop]
    event_ids = np.asarray(rows[:, 0], dtype=np.int64)
    if len(event_ids) == 0:
        raise ValueError("internal error: attempted to count an empty HDF5 chunk")
    return event_ids


def _attribute_text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _load_compatible_manifest(
    path: Path,
    inventory: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != MANIFEST_VERSION:
        return None
    if payload.get("inventory") != dict(inventory):
        return None
    if payload.get("settings") != dict(settings):
        return None
    try:
        for name in SPLIT_NAMES:
            if not payload["splits"][name]:
                return None
        if not payload["files"] or not payload["counts"]:
            return None
    except (KeyError, TypeError):
        return None
    return payload


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _validate_manifest_accounting(
    slices: Mapping[str, Sequence[FileSlice]], counts: Mapping[str, Any]
) -> None:
    actual = {
        name: sum(item.event_count for item in slices[name]) for name in SPLIT_NAMES
    }
    if any(actual[name] != int(counts[name]) for name in SPLIT_NAMES):
        raise ValueError("cached split manifest has inconsistent event counts")
    if sum(actual.values()) != int(counts["total"]):
        raise ValueError("cached split manifest has an inconsistent total")
    seen_ranges: dict[str, list[tuple[int, int]]] = {}
    for item in chain.from_iterable(slices[name] for name in SPLIT_NAMES):
        seen_ranges.setdefault(item.relative_path, []).append(
            (item.event_start, item.event_stop)
        )
    for relative_path, ranges in seen_ranges.items():
        ordered = sorted(ranges)
        for (_, previous_stop), (current_start, _) in zip(ordered, ordered[1:]):
            if current_start < previous_stop:
                raise ValueError(
                    f"cached split manifest overlaps events in {relative_path}"
                )


def _report_counts(counts: Mapping[str, Any]) -> None:
    total = int(counts["total"])
    print(f"Total events: {total:,}")
    labels = {"train": "Train", "validation": "Validation", "test": "Test"}
    for name in SPLIT_NAMES:
        value = int(counts[name])
        print(f"{labels[name]}: {value:,} events ({100.0 * value / total:.2f}%)")
    print("Per class:")
    for category, values in counts["by_class"].items():
        print(
            f"  {category}: {int(values['total']):,} total "
            f"(train {int(values['train']):,}, "
            f"validation {int(values['validation']):,}, "
            f"test {int(values['test']):,})"
        )
    boundaries = counts.get("boundary_files", [])
    if boundaries:
        print(f"Boundary files shared by adjacent splits: {len(boundaries)}")
        for relative_path in boundaries:
            print(f"  {relative_path}")


def _validate_fractions(values: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError("split_fractions must contain train/validation/test values")
    fractions = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in fractions):
        raise ValueError("split fractions must be finite and positive")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("split fractions must sum to 1")
    return fractions  # type: ignore[return-value]


def _validate_class_map(value: Mapping[str, int] | None) -> dict[str, int]:
    class_map = dict(DEFAULT_CLASS_MAP if value is None else value)
    if set(class_map) != set(DEFAULT_CLASS_MAP):
        raise ValueError("class_map must define exactly '0nubb' and 'Bi214'")
    if any(
        isinstance(label, bool) or int(label) != label for label in class_map.values()
    ) or set(int(label) for label in class_map.values()) != {0, 1}:
        raise ValueError("class_map must assign the binary labels 0 and 1")
    return {category: int(class_map[category]) for category in DEFAULT_CLASS_MAP}


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


__all__ = [
    "DEFAULT_CLASS_MAP",
    "DEFAULT_DATA_ROOT",
    "EventInputBuilder",
    "FileSlice",
    "InputMode",
    "NextEventDataset",
    "PreparedData",
    "SPLIT_NAMES",
    "count_file_events",
    "prepare_dataset",
    "project_event",
]
