"""NEXT HDF5 reading, leakage-safe splits, and orthographic projections."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch as _torch

    _IterableDatasetBase = _torch.utils.data.IterableDataset
except ImportError:  # The pure NumPy/HDF helpers remain independently usable.
    _torch = None
    _IterableDatasetBase = object


DEFAULT_DATA_ROOT = Path("/home/klz/Data/zeronu_benchmark/NEXT")
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ProjectionConfig:
    """Fixed detector-coordinate projection used as the CNN input."""

    grid_size: int = 128
    bin_size: float = 30.0
    origin: Tuple[float, float, float] = (-1920.0, -1920.0, -120.0)
    normalize_energy: bool = True
    input_scale: float = 100.0
    representation: str = "energy"

    def __post_init__(self) -> None:
        if int(self.grid_size) != self.grid_size or self.grid_size < 8:
            raise ValueError("grid_size must be an integer >= 8")
        if not np.isfinite(self.bin_size) or self.bin_size <= 0:
            raise ValueError("bin_size must be finite and positive")
        if len(self.origin) != 3 or not np.all(np.isfinite(self.origin)):
            raise ValueError("origin must contain three finite coordinates")
        if not np.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError("input_scale must be finite and positive")
        if self.representation not in {"energy", "binary_occupancy"}:
            raise ValueError(
                "representation must be energy or binary_occupancy"
            )
        if self.representation != "energy" and self.normalize_energy:
            raise ValueError(
                "normalize_energy is only valid for energy-weighted projections"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_size": int(self.grid_size),
            "bin_size": float(self.bin_size),
            "origin": [float(value) for value in self.origin],
            "normalize_energy": bool(self.normalize_energy),
            "input_scale": float(self.input_scale),
            "representation": str(self.representation),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProjectionConfig":
        return cls(
            grid_size=int(payload.get("grid_size", 128)),
            bin_size=float(payload.get("bin_size", 30.0)),
            origin=tuple(payload.get("origin", (-1920.0, -1920.0, -120.0))),
            normalize_energy=bool(payload.get("normalize_energy", True)),
            input_scale=float(payload.get("input_scale", 100.0)),
            representation=str(payload.get("representation", "energy")),
        )


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    group_id: str
    label: int
    category: str
    split: str


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    source_event_id: int
    group_id: str
    label: int
    category: str
    split: str
    coordinates: np.ndarray
    energies: np.ndarray
    energy_sum: float
    is_last_in_file: bool = False


def _hash_fraction(text: str, seed: int) -> float:
    digest = hashlib.blake2b(
        (str(seed) + ":" + text).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="big") / float(2**64)


def assign_split(
    group_id: str,
    seed: int = 42,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
) -> str:
    """Assign an entire source HDF5 file to one deterministic split."""

    values = np.asarray(fractions, dtype=float)
    if values.shape != (3,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("split fractions must contain three positive values")
    if not np.isclose(float(np.sum(values)), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("split fractions must sum to 1")
    draw = _hash_fraction(group_id, seed)
    if draw < values[0]:
        return "train"
    if draw < values[0] + values[1]:
        return "validation"
    return "test"


def _class_from_path(path: Path) -> Tuple[int, str]:
    parent = path.parent.name.lower()
    if parent.startswith("0nubb_part_"):
        return 1, "0nubb"
    if parent.startswith("bi_part_"):
        return 0, "Bi214"
    raise ValueError("cannot infer NEXT class from directory: %s" % path.parent)


def discover_source_files(
    root: Any = DEFAULT_DATA_ROOT,
    split: Optional[str] = None,
    split_seed: int = 42,
    split_fractions: Sequence[float] = (0.8, 0.1, 0.1),
    max_files_per_class: Optional[int] = None,
) -> List[SourceFile]:
    """Find extracted NEXT files and apply a stable file-level split."""

    data_root = Path(root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError("NEXT data directory does not exist: %s" % data_root)
    if split is not None and split not in SPLIT_NAMES:
        raise ValueError("split must be one of: %s" % ", ".join(SPLIT_NAMES))
    if max_files_per_class is not None and max_files_per_class <= 0:
        max_files_per_class = None

    paths = list(data_root.glob("0nubb_part_*/*.h5"))
    paths.extend(data_root.glob("Bi_part_*/*.h5"))
    if not paths:
        raise FileNotFoundError(
            "no extracted NEXT HDF5 files found below %s" % data_root
        )

    by_category: Dict[str, List[SourceFile]] = {"0nubb": [], "Bi214": []}
    for path in paths:
        relative = path.relative_to(data_root).as_posix()
        label, category = _class_from_path(path)
        assigned = assign_split(relative, split_seed, split_fractions)
        if split is not None and assigned != split:
            continue
        by_category[category].append(
            SourceFile(
                path=path,
                relative_path=relative,
                group_id=relative,
                label=label,
                category=category,
                split=assigned,
            )
        )

    selected: List[SourceFile] = []
    for category in ("0nubb", "Bi214"):
        ranked = sorted(
            by_category[category],
            key=lambda item: (
                _hash_fraction("selection:" + item.relative_path, split_seed),
                item.relative_path,
            ),
        )
        if max_files_per_class is not None:
            ranked = ranked[: int(max_files_per_class)]
        if not ranked:
            raise ValueError("split %r contains no %s files" % (split, category))
        selected.extend(ranked)
    return sorted(selected, key=lambda item: item.relative_path)


def dataset_inventory(root: Any = DEFAULT_DATA_ROOT) -> Dict[str, Any]:
    """Fingerprint the extracted file inventory without reading 15 GB of data."""

    data_root = Path(root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError("NEXT data directory does not exist: %s" % data_root)
    paths = list(data_root.glob("0nubb_part_*/*.h5"))
    paths.extend(data_root.glob("Bi_part_*/*.h5"))
    if not paths:
        raise FileNotFoundError(
            "no extracted NEXT HDF5 files found below %s" % data_root
        )
    digest = hashlib.sha256()
    category_counts = {"0nubb": 0, "Bi214": 0}
    for path in sorted(paths, key=lambda item: item.relative_to(data_root).as_posix()):
        relative = path.relative_to(data_root).as_posix()
        _, category = _class_from_path(path)
        category_counts[category] += 1
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256-relative-path-size-v1",
        "digest": digest.hexdigest(),
        "file_count": len(paths),
        "category_counts": category_counts,
    }


def _attribute_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _validate_hdf_schema(dataset: Any, path: Path) -> None:
    names = dataset.dtype.names
    expected = (
        "index",
        "values_block_0",
        "values_block_1",
        "values_block_2",
    )
    if names != expected:
        raise ValueError(
            "unexpected NEXT table fields in %s: %r" % (path, names)
        )
    block_0 = _attribute_text(dataset.attrs.get("values_block_0_kind", ""))
    block_1 = _attribute_text(dataset.attrs.get("values_block_1_kind", ""))
    block_2 = _attribute_text(dataset.attrs.get("values_block_2_kind", ""))
    if "Vevent_id" not in block_0 or "Vlabel" not in block_2:
        raise ValueError("NEXT event_id/label metadata is missing in %s" % path)
    positions = [block_1.find("V" + name) for name in ("x", "y", "z", "energy")]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValueError(
            "NEXT coordinate columns are not ordered x,y,z,energy in %s" % path
        )


def iter_file_events(source: SourceFile) -> Iterator[EventRecord]:
    """Read one HDF5 table and yield one record per contiguous event ID."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "NEXT CNN data loading requires h5py; "
            "install the CUDA PyTorch requirement first, then "
            "activate the project environment and run "
            "`python -m pip install -e .`"
        ) from exc

    with h5py.File(str(source.path), "r") as handle:
        if "MC/hits/table" not in handle:
            raise ValueError("missing /MC/hits/table in %s" % source.path)
        dataset = handle["MC/hits/table"]
        _validate_hdf_schema(dataset, source.path)
        rows = dataset[:]

    if rows.size == 0:
        raise ValueError("NEXT HDF5 table is empty: %s" % source.path)
    event_ids = np.asarray(rows["values_block_0"][:, 0], dtype=np.int64)
    values = np.asarray(rows["values_block_1"], dtype=np.float32)
    raw_labels = np.asarray(rows["values_block_2"][:, 0]).astype("S")
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("NEXT x,y,z,energy block has invalid shape in %s" % source.path)
    if np.any(~np.isfinite(values)):
        raise ValueError("NEXT table contains non-finite coordinates/energy: %s" % source.path)
    if np.any(values[:, 3] < 0):
        raise ValueError("NEXT table contains negative voxel energy: %s" % source.path)

    expected_raw_label = b"Signal" if source.label == 1 else b"Bkg"
    if np.any(raw_labels != expected_raw_label):
        found = sorted(value.decode("utf-8", errors="replace") for value in np.unique(raw_labels))
        raise ValueError(
            "label mismatch in %s: expected %r, found %r"
            % (source.path, expected_raw_label.decode(), found)
        )

    starts = np.r_[0, np.flatnonzero(event_ids[1:] != event_ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(event_ids)]
    boundary_ids = event_ids[starts]
    if len(np.unique(boundary_ids)) != len(boundary_ids):
        raise ValueError("event IDs are not contiguous in %s" % source.path)

    for event_index, (start, stop, source_event_id) in enumerate(
        zip(starts, stops, boundary_ids)
    ):
        coordinates = values[start:stop, :3]
        energies = values[start:stop, 3]
        energy_sum = float(np.sum(energies, dtype=np.float64))
        if not np.isfinite(energy_sum) or energy_sum <= 0:
            raise ValueError(
                "event %s has non-positive total energy in %s"
                % (source_event_id, source.path)
            )
        yield EventRecord(
            event_id="NEXT::%s::%d" % (source.relative_path, source_event_id),
            source_event_id=int(source_event_id),
            group_id=source.group_id,
            label=source.label,
            category=source.category,
            split=source.split,
            coordinates=coordinates,
            energies=energies,
            energy_sum=energy_sum,
            is_last_in_file=event_index == len(starts) - 1,
        )


def project_event(
    coordinates: np.ndarray,
    energies: np.ndarray,
    config: ProjectionConfig = ProjectionConfig(),
) -> Tuple[np.ndarray, float]:
    """Accumulate XY, XZ, and YZ energy maps into a ``(3, H, W)`` tensor."""

    xyz = np.asarray(coordinates, dtype=np.float32)
    weight = np.asarray(energies, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_voxels, 3)")
    if weight.ndim != 1 or len(weight) != len(xyz):
        raise ValueError("energies must have shape (n_voxels,)")
    if len(weight) == 0:
        raise ValueError("cannot project an empty event")
    if np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(weight)):
        raise ValueError("coordinates and energies must be finite")
    if np.any(weight < 0):
        raise ValueError("voxel energies must be non-negative")
    total_energy = float(np.sum(weight, dtype=np.float64))
    if total_energy <= 0:
        raise ValueError("event total energy must be positive")

    origin = np.asarray(config.origin, dtype=np.float32)
    indices = np.floor((xyz - origin) / np.float32(config.bin_size)).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < config.grid_size), axis=1)
    kept_energy = float(np.sum(weight[valid], dtype=np.float64))
    image = np.zeros(
        (3, config.grid_size, config.grid_size), dtype=np.float32
    )
    if np.any(valid):
        x_index, y_index, z_index = indices[valid].T
        if config.representation == "energy":
            kept_weight = weight[valid]
        else:
            # Topology-only input: every in-range voxel contributes one,
            # independently of its deposited-energy amplitude.
            kept_weight = np.ones(np.count_nonzero(valid), dtype=np.float32)
        np.add.at(image[0], (y_index, x_index), kept_weight)
        np.add.at(image[1], (z_index, x_index), kept_weight)
        np.add.at(image[2], (z_index, y_index), kept_weight)
    if config.representation == "binary_occupancy":
        np.minimum(image, np.float32(1.0), out=image)
    elif config.normalize_energy:
        image /= np.float32(total_energy)
    image *= np.float32(config.input_scale)
    if config.representation == "energy":
        coverage = kept_energy / total_energy
    else:
        coverage = float(np.count_nonzero(valid)) / float(len(valid))
    return image, coverage


class NextIterableDataset(_IterableDatasetBase):
    """PyTorch-compatible iterable dataset with lazy optional imports."""

    def __init__(
        self,
        files: Iterable[SourceFile],
        projection: ProjectionConfig = ProjectionConfig(),
        shuffle_files: bool = False,
        balance_classes: bool = False,
        seed: int = 42,
        event_shuffle_buffer_size: int = 0,
        include_classification_metadata: bool = True,
    ) -> None:
        self.files = list(files)
        if not self.files:
            raise ValueError("NextIterableDataset needs at least one source file")
        if (
            int(event_shuffle_buffer_size) != event_shuffle_buffer_size
            or event_shuffle_buffer_size < 0
        ):
            raise ValueError(
                "event_shuffle_buffer_size must be a non-negative integer"
            )
        self.projection = projection
        self.shuffle_files = bool(shuffle_files)
        self.balance_classes = bool(balance_classes)
        self.seed = int(seed)
        self.event_shuffle_buffer_size = int(event_shuffle_buffer_size)
        self.include_classification_metadata = bool(
            include_classification_metadata
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if _torch is None:
            raise RuntimeError(
                "NEXT CNN training requires PyTorch; "
                "install `requirements/next-cnn-cu128.txt` in the GPU environment"
            )

        files = list(self.files)
        if self.shuffle_files:
            generator = np.random.default_rng(self.seed + self.epoch)
            generator.shuffle(files)
        worker = _torch.utils.data.get_worker_info()

        def events(sources: Sequence[SourceFile]) -> Iterator[EventRecord]:
            for source in sources:
                yield from iter_file_events(source)

        def shuffled_events(
            stream: Iterator[EventRecord],
            worker_id: int,
        ) -> Iterator[EventRecord]:
            buffer_size = self.event_shuffle_buffer_size
            if buffer_size == 0:
                yield from stream
                return
            generator = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, self.epoch, int(worker_id)]
                )
            )
            buffer: List[EventRecord] = []
            for event in stream:
                if len(buffer) < buffer_size:
                    buffer.append(event)
                    continue
                index = int(generator.integers(0, len(buffer)))
                selected = buffer[index]
                buffer[index] = event
                yield selected
            generator.shuffle(buffer)
            yield from buffer

        def projected(event: EventRecord) -> Dict[str, Any]:
            image, coverage = project_event(
                event.coordinates, event.energies, self.projection
            )
            row = {
                "image": image,
                "event_id": event.event_id,
                "energy_condition": np.float64(event.energy_sum),
                "energy_target": np.float64(event.energy_sum),
                "projection_coverage": np.float32(coverage),
                "source_file_complete": np.bool_(event.is_last_in_file),
                "split": event.split,
                "group_id": event.group_id,
            }
            if self.include_classification_metadata:
                row["label"] = np.float32(event.label)
                row["category"] = event.category
            return row

        # Regression keeps every selected event.  Shuffle the complete file
        # stream before applying a bounded event buffer so a larger class can
        # no longer form a deterministic single-class tail at every epoch.
        # The default (non-shuffled) inference order remains unchanged.
        if not self.balance_classes and self.shuffle_files:
            worker_id = 0 if worker is None else int(worker.id)
            if worker is not None:
                files = files[worker.id :: worker.num_workers]
            for event in shuffled_events(events(files), worker_id):
                yield projected(event)
            return

        by_label = {
            label: [source for source in files if source.label == label]
            for label in (1, 0)
        }
        if worker is not None:
            by_label = {
                label: sources[worker.id :: worker.num_workers]
                for label, sources in by_label.items()
            }

        # Alternate classes at event level.  This gives ordinary DataLoader
        # batches both labels without loading the full dataset or a large
        # shuffle buffer into memory.
        active = [iter(events(by_label[label])) for label in (1, 0) if by_label[label]]
        if self.balance_classes and len(active) == 2:
            signal_events, background_events = active
            while True:
                try:
                    pair = (next(signal_events), next(background_events))
                except StopIteration:
                    return
                for event in pair:
                    yield projected(event)
        while active:
            remaining = []
            for iterator in active:
                try:
                    event = next(iterator)
                except StopIteration:
                    continue
                remaining.append(iterator)
                yield projected(event)
            active = remaining


__all__ = [
    "DEFAULT_DATA_ROOT",
    "EventRecord",
    "NextIterableDataset",
    "ProjectionConfig",
    "SPLIT_NAMES",
    "SourceFile",
    "assign_split",
    "dataset_inventory",
    "discover_source_files",
    "iter_file_events",
    "project_event",
]
