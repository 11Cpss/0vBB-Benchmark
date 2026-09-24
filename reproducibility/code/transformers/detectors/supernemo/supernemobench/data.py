"""Strict event indexing, shared splits, and topology-only input adapters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import (
    CATEGORY_FIELD,
    CLASSIFICATION_LABELS,
    COORDINATE_FIELDS,
    ENERGY_FIELDS,
    ENERGY_TARGET,
    ENERGY_UNIT,
    EVENT_ID_FIELD,
    EVENT_CONSTANT_FIELDS,
    POINT_FEATURES,
    PROJECTION_PLANES,
    REQUIRED_FIELDS,
    RADIUS_FIELD,
    SOURCE_BY_KEY,
    SOURCE_SPECS,
    SPLIT_NAMES,
    DataConfig,
)
from .tokenization import (
    SuperNEMOTrackerTokenizationConfig,
    tokenize_tracker_event,
)


Task = Literal["classification", "energy"]
InputKind = Literal["projection2d", "graph", "sequence"]
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class EventSlice:
    """One contiguous range of complete event ordinals from a raw source."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "split": self.split,
            "event_start": self.event_start,
            "event_stop": self.event_stop,
            "event_count": self.event_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EventSlice":
        return cls(
            source_key=str(payload["source_key"]),
            split=str(payload["split"]),
            event_start=int(payload["event_start"]),
            event_stop=int(payload["event_stop"]),
        )


class _PrefetchLoader:
    """Prepare at most two batches ahead without changing their order."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    @property
    def dataset(self) -> Any:
        """Expose the underlying dataset so epoch updates reach it directly."""

        return self.loader.dataset

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        batches: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=2)
        stopped = threading.Event()
        sentinel = object()

        def send(is_error: bool, payload: Any) -> bool:
            while not stopped.is_set():
                try:
                    batches.put((is_error, payload), timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                for batch in self.loader:
                    if not send(False, batch):
                        return
            except BaseException as error:
                if not send(True, error):
                    return
            send(False, sentinel)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            while True:
                is_error, payload = batches.get()
                if payload is sentinel:
                    break
                if is_error:
                    raise payload
                yield payload
        finally:
            stopped.set()
            producer.join(timeout=1.0)


@dataclass(frozen=True)
class PreparedData:
    train_loader: _PrefetchLoader
    validation_loader: _PrefetchLoader
    test_loader: _PrefetchLoader
    counts: dict[str, Any]
    manifest_path: Path
    task: Task
    input_kind: InputKind
    tokenization_config: SuperNEMOTrackerTokenizationConfig | None

    @property
    def val_loader(self) -> _PrefetchLoader:
        return self.validation_loader


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_inventory(config: DataConfig) -> list[dict[str, Any]]:
    root = config.data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SuperNEMO data directory does not exist: {root}")
    expected = {spec.file_name for spec in SOURCE_SPECS}
    actual = {path.name for path in root.glob("*.h5") if path.is_file()}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise FileNotFoundError("missing canonical SuperNEMO files: " + ", ".join(missing))
    if unexpected:
        raise ValueError(
            "unrecognized HDF5 sources require an explicit category mapping: "
            + ", ".join(unexpected)
        )
    inventory = []
    for spec in SOURCE_SPECS:
        path = (root / spec.file_name).resolve()
        stat = path.stat()
        inventory.append(
            {
                "source_key": spec.source_key,
                "file_name": spec.file_name,
                "raw_label": spec.raw_label,
                "category": spec.category,
                "classification_label": spec.classification_label,
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return inventory


def _settings(config: DataConfig) -> dict[str, Any]:
    return {
        "split_fractions": list(config.split_fractions),
        "seed": int(config.seed),
        "split_block_events": int(config.split_block_events),
        "classification_labels": dict(CLASSIFICATION_LABELS),
        "energy_fields": list(ENERGY_FIELDS),
        "energy_target": ENERGY_TARGET,
        "energy_unit": ENERGY_UNIT,
        "required_fields": list(REQUIRED_FIELDS),
    }


def _index_name(source_key: str) -> str:
    return f"{source_key}_event_offsets.npy"


def _first_bad(mask: np.ndarray, row_start: int) -> int:
    return row_start + int(np.flatnonzero(mask)[0])


def _numeric_profile() -> dict[str, Any]:
    return {
        "min": math.inf,
        "max": -math.inf,
        "nan": 0,
        "positive_infinity": 0,
        "negative_infinity": 0,
    }


def _update_numeric_profile(profile: dict[str, Any], values: np.ndarray) -> None:
    profile["nan"] += int(np.count_nonzero(np.isnan(values)))
    profile["positive_infinity"] += int(np.count_nonzero(np.isposinf(values)))
    profile["negative_infinity"] += int(np.count_nonzero(np.isneginf(values)))
    finite = values[np.isfinite(values)]
    if finite.size:
        profile["min"] = min(float(profile["min"]), float(np.min(finite)))
        profile["max"] = max(float(profile["max"]), float(np.max(finite)))


def _finish_numeric_profile(profile: dict[str, Any]) -> dict[str, Any]:
    result = dict(profile)
    if math.isinf(float(result["min"])) or math.isinf(float(result["max"])):
        result["min"] = None
        result["max"] = None
    return result


def _scan_source(
    path: Path,
    source_key: str,
    config: DataConfig,
    index_path: Path,
) -> dict[str, Any]:
    """Perform the one full validation pass and write event row offsets."""

    spec = SOURCE_BY_KEY[source_key]
    with h5py.File(path, "r") as handle:
        available = set(handle.keys())
        missing = sorted(set(REQUIRED_FIELDS) - available)
        if missing:
            raise KeyError(f"{path}: missing fields {missing}; available={sorted(available)}")
        lengths: dict[str, int] = {}
        dtypes: dict[str, str] = {}
        for field in REQUIRED_FIELDS:
            dataset = handle[field]
            if dataset.ndim != 1:
                raise ValueError(f"{path}:{field} must be one-dimensional, got {dataset.shape}")
            lengths[field] = int(dataset.shape[0])
            dtypes[field] = str(dataset.dtype)
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path}: inconsistent field lengths: {lengths}")
        row_count = next(iter(lengths.values()))
        if row_count == 0:
            raise ValueError(f"{path}: empty HDF5 source")
        if not np.issubdtype(handle[EVENT_ID_FIELD].dtype, np.integer):
            raise TypeError(f"{path}: {EVENT_ID_FIELD} must have integer dtype")
        for field in REQUIRED_FIELDS:
            if field in {EVENT_ID_FIELD, CATEGORY_FIELD}:
                continue
            if not np.issubdtype(handle[field].dtype, np.floating):
                raise TypeError(f"{path}:{field} must have floating dtype")

        first_event = int(handle[EVENT_ID_FIELD][0])
        last_event = int(handle[EVENT_ID_FIELD][-1])
        if first_event != 0 or last_event < first_event:
            raise ValueError(
                f"{path}: expected contiguous local event IDs starting at zero; "
                f"found endpoints {first_event}..{last_event}"
            )
        event_count = last_event + 1
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp.npy")
        offsets = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.int64,
            shape=(event_count + 1,),
        )
        offsets[:] = -1
        offsets[0] = 0

        numeric_profiles = {
            field: _numeric_profile()
            for field in REQUIRED_FIELDS
            if field not in {EVENT_ID_FIELD, CATEGORY_FIELD}
        }
        energy_profile = _numeric_profile()
        previous_event: int | None = None
        previous_values: dict[str, float] = {}
        t_r_affected_events = 0
        last_bad_t_r_event: int | None = None
        event_constant_mismatches = {field: 0 for field in EVENT_CONSTANT_FIELDS}

        try:
            for row_start in range(0, row_count, config.scan_chunk_rows):
                row_stop = min(row_start + config.scan_chunk_rows, row_count)
                event_ids = np.asarray(
                    handle[EVENT_ID_FIELD][row_start:row_stop], dtype=np.int64
                )
                if event_ids.size == 0:
                    continue
                differences = np.diff(event_ids)
                invalid_step = (differences < 0) | (differences > 1)
                if np.any(invalid_step):
                    bad_row = row_start + int(np.flatnonzero(invalid_step)[0]) + 1
                    raise ValueError(
                        f"{path}: ev_no gap/decrease at row {bad_row}: "
                        f"{int(handle[EVENT_ID_FIELD][bad_row - 1])}->"
                        f"{int(handle[EVENT_ID_FIELD][bad_row])}"
                    )
                if previous_event is not None:
                    boundary_step = int(event_ids[0]) - previous_event
                    if boundary_step not in (0, 1):
                        raise ValueError(
                            f"{path}: ev_no gap/decrease across chunk boundary at row {row_start}"
                        )
                    if boundary_step == 1:
                        offsets[int(event_ids[0])] = row_start
                change_positions = np.flatnonzero(differences == 1) + 1
                if change_positions.size:
                    new_ids = event_ids[change_positions]
                    offsets[new_ids] = row_start + change_positions

                arrays = {
                    field: np.asarray(handle[field][row_start:row_stop], dtype=np.float32)
                    for field in numeric_profiles
                }
                for field, values in arrays.items():
                    _update_numeric_profile(numeric_profiles[field], values)
                    finite = np.isfinite(values)
                    if field != RADIUS_FIELD and not np.all(finite):
                        bad_row = _first_bad(~finite, row_start)
                        raise ValueError(f"{path}:{field} has a non-finite value at row {bad_row}")
                    if field in EVENT_CONSTANT_FIELDS:
                        same_event = event_ids[1:] == event_ids[:-1]
                        mismatched = same_event & (values[1:] != values[:-1])
                        mismatch_count = int(np.count_nonzero(mismatched))
                        event_constant_mismatches[field] += mismatch_count
                        if mismatch_count:
                            bad_row = row_start + int(np.flatnonzero(mismatched)[0]) + 1
                            raise ValueError(
                                f"{path}:{field} changes within event {int(event_ids[bad_row-row_start])} "
                                f"at row {bad_row}"
                            )
                        if (
                            previous_event is not None
                            and int(event_ids[0]) == previous_event
                            and float(values[0]) != previous_values[field]
                        ):
                            raise ValueError(
                                f"{path}:{field} changes within event {previous_event} "
                                f"across row {row_start}"
                            )
                        previous_values[field] = float(values[-1])

                e1 = arrays[ENERGY_FIELDS[0]]
                e2 = arrays[ENERGY_FIELDS[1]]
                if np.any(e1 <= 0.0) or np.any(e2 <= 0.0):
                    invalid = (e1 <= 0.0) | (e2 <= 0.0)
                    raise ValueError(
                        f"{path}: non-positive electron energy at row {_first_bad(invalid, row_start)}"
                    )
                energy = np.asarray(e1 + e2, dtype=np.float32)
                if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
                    invalid = (~np.isfinite(energy)) | (energy <= 0.0)
                    raise ValueError(
                        f"{path}: invalid E1+E2 at row {_first_bad(invalid, row_start)}"
                    )
                _update_numeric_profile(energy_profile, energy)

                bad_t_r = ~np.isfinite(arrays[RADIUS_FIELD])
                if np.any(bad_t_r):
                    bad_ids = np.unique(event_ids[bad_t_r])
                    t_r_affected_events += int(bad_ids.size)
                    if last_bad_t_r_event is not None and int(bad_ids[0]) == last_bad_t_r_event:
                        t_r_affected_events -= 1
                    last_bad_t_r_event = int(bad_ids[-1])

                raw_labels = np.asarray(handle[CATEGORY_FIELD][row_start:row_stop])
                expected = spec.raw_label.encode("utf-8")
                valid_labels = raw_labels == expected
                if not np.all(valid_labels):
                    bad_row = _first_bad(~valid_labels, row_start)
                    found = handle[CATEGORY_FIELD][bad_row]
                    raise ValueError(
                        f"{path}: label mismatch at row {bad_row}; "
                        f"expected {spec.raw_label!r}, found {found!r}"
                    )
                previous_event = int(event_ids[-1])

            offsets[event_count] = row_count
            if np.any(offsets < 0):
                missing_event = int(np.flatnonzero(offsets < 0)[0])
                raise ValueError(f"{path}: no row boundary found for event ordinal {missing_event}")
            hit_counts = np.diff(offsets)
            if np.any(hit_counts <= 0):
                event = int(np.flatnonzero(hit_counts <= 0)[0])
                raise ValueError(f"{path}: event {event} contains no hit rows")
            offsets.flush()
            del offsets
            offsets = None
            os.replace(temporary, index_path)
        except BaseException:
            if offsets is not None:
                del offsets
            if temporary.exists():
                temporary.unlink()
            raise

    index_values = np.load(index_path, mmap_mode="r")
    hit_counts = np.diff(index_values)
    profile = {
        "row_count": row_count,
        "event_count": event_count,
        "event_id_min": first_event,
        "event_id_max": last_event,
        "hits_per_event": {
            "min": int(np.min(hit_counts)),
            "max": int(np.max(hit_counts)),
        },
        "field_lengths": lengths,
        "field_dtypes": dtypes,
        "numeric": {
            field: _finish_numeric_profile(values)
            for field, values in numeric_profiles.items()
        },
        "energy": {
            **_finish_numeric_profile(energy_profile),
            "definition": ENERGY_TARGET,
            "unit": ENERGY_UNIT,
        },
        "event_constant_mismatches": event_constant_mismatches,
        "label_values": [spec.raw_label],
        "empty_labels": 0,
        "tR_affected_events": t_r_affected_events,
        "duplicate_event_keys": 0,
        "event_id_gaps": 0,
    }
    return profile


def _block_assignments(
    source_key: str,
    event_count: int,
    config: DataConfig,
) -> dict[str, list[EventSlice]]:
    """Allocate locality-preserving event blocks reproducibly by category."""

    block_size = int(config.split_block_events)
    blocks = [
        (start, min(start + block_size, event_count))
        for start in range(0, event_count, block_size)
    ]
    source_position = [spec.source_key for spec in SOURCE_SPECS].index(source_key)
    generator = np.random.default_rng(
        np.random.SeedSequence([int(config.seed), int(source_position)])
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
        selected = permutation[cursor : cursor + count]
        assigned[selected] = split_index
        cursor += count
    if cursor != len(blocks):
        raise RuntimeError("internal split block allocation error")

    result = {name: [] for name in SPLIT_NAMES}
    current_split: int | None = None
    current_start = 0
    current_stop = 0
    for block_index, (start, stop) in enumerate(blocks):
        split_index = int(assigned[block_index])
        if current_split is None:
            current_split, current_start, current_stop = split_index, start, stop
        elif split_index == current_split and start == current_stop:
            current_stop = stop
        else:
            split = SPLIT_NAMES[current_split]
            result[split].append(EventSlice(source_key, split, current_start, current_stop))
            current_split, current_start, current_stop = split_index, start, stop
    if current_split is not None:
        split = SPLIT_NAMES[current_split]
        result[split].append(EventSlice(source_key, split, current_start, current_stop))
    return result


def _task_counts(
    split_slices: Mapping[str, Sequence[EventSlice]],
    *,
    task: Task,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        by_category: dict[str, int] = {}
        for item in split_slices[split]:
            spec = SOURCE_BY_KEY[item.source_key]
            if task == "classification" and spec.classification_label is None:
                continue
            by_category[spec.category] = by_category.get(spec.category, 0) + item.event_count
        result[split] = {
            "total": int(sum(by_category.values())),
            "by_category": dict(sorted(by_category.items())),
        }
    return result


def _validate_split_slices(
    split_slices: Mapping[str, Sequence[EventSlice]],
    files: Sequence[Mapping[str, Any]],
) -> None:
    event_counts = {str(item["source_key"]): int(item["profile"]["event_count"]) for item in files}
    for source_key, event_count in event_counts.items():
        ranges = sorted(
            (
                item.event_start,
                item.event_stop,
                split,
            )
            for split in SPLIT_NAMES
            for item in split_slices[split]
            if item.source_key == source_key
        )
        cursor = 0
        for start, stop, _ in ranges:
            if start != cursor:
                relation = "overlap" if start < cursor else "gap"
                raise ValueError(f"split manifest has an event {relation} for {source_key} at {cursor}")
            cursor = stop
        if cursor != event_count:
            raise ValueError(
                f"split manifest covers {cursor} of {event_count} events for {source_key}"
            )


def build_manifest(config: DataConfig | None = None) -> dict[str, Any]:
    """Build the sole shared event index and 80/10/10 split manifest."""

    selected = config or DataConfig()
    inventory = _source_inventory(selected)
    manifest_dir = selected.manifest_path.resolve().parent
    files: list[dict[str, Any]] = []
    split_slices: dict[str, list[EventSlice]] = {name: [] for name in SPLIT_NAMES}
    for record in inventory:
        source_key = str(record["source_key"])
        source_path = selected.data_root.resolve() / str(record["file_name"])
        index_path = manifest_dir / _index_name(source_key)
        profile = _scan_source(source_path, source_key, selected, index_path)
        allocation = _block_assignments(
            source_key,
            int(profile["event_count"]),
            selected,
        )
        for split in SPLIT_NAMES:
            split_slices[split].extend(allocation[split])
        files.append(
            {
                **record,
                "index_file": index_path.name,
                "profile": profile,
            }
        )
    _validate_split_slices(split_slices, files)
    excluded = {
        SOURCE_BY_KEY[item["source_key"]].category: int(item["profile"]["event_count"])
        for item in files
        if SOURCE_BY_KEY[item["source_key"]].classification_label is None
    }
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "data_root": str(selected.data_root.resolve()),
        "inventory": inventory,
        "settings": _settings(selected),
        "files": files,
        "splits": {
            split: [item.to_dict() for item in split_slices[split]]
            for split in SPLIT_NAMES
        },
        "counts": {
            "energy": _task_counts(split_slices, task="energy"),
            "classification": _task_counts(split_slices, task="classification"),
            "classification_excluded": dict(sorted(excluded.items())),
        },
        "event_key": ["source_key", EVENT_ID_FIELD],
        "grouping": {
            "unit": "complete_event",
            "higher_level_fields_available": [],
            "note": "No run or simulation-batch field exists in the release.",
        },
        "input_policy": {
            "fields": list(COORDINATE_FIELDS),
            "representation": "topology_only",
            "excluded_from_model": [
                *[field for field in REQUIRED_FIELDS if field not in COORDINATE_FIELDS],
                "category",
                "source_key",
                "file_name",
            ],
        },
    }
    digest_payload = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    payload["content_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    _json_dump(selected.manifest_path.resolve(), payload)
    return payload


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read split manifest {path}: {error}") from error
    if int(payload.get("schema_version", -1)) != MANIFEST_VERSION:
        raise ValueError(f"unsupported split manifest schema in {path}")
    recorded_digest = payload.pop("content_sha256", None)
    calculated_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    if recorded_digest != calculated_digest:
        raise ValueError(f"split manifest integrity check failed: {path}")
    payload["content_sha256"] = recorded_digest
    return payload


def load_manifest(config: DataConfig | None = None) -> dict[str, Any]:
    """Load a compatible manifest, building it only when it does not exist."""

    selected = config or DataConfig()
    path = selected.manifest_path.resolve()
    if not path.exists():
        return build_manifest(selected)
    payload = _read_manifest(path)
    inventory = _source_inventory(selected)
    if payload.get("inventory") != inventory or payload.get("settings") != _settings(selected):
        raise ValueError(
            f"stale split manifest {path}; source files or split settings changed. "
            "Rebuild it explicitly before training."
        )
    if payload.get("data_root") != str(selected.data_root.resolve()):
        raise ValueError(f"manifest data_root does not match {selected.data_root.resolve()}")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != len(SOURCE_SPECS):
        raise ValueError("split manifest has an invalid file inventory")
    split_slices = {
        split: [EventSlice.from_dict(item) for item in payload["splits"][split]]
        for split in SPLIT_NAMES
    }
    _validate_split_slices(split_slices, files)
    for record in files:
        index_path = path.parent / str(record["index_file"])
        if not index_path.is_file():
            raise FileNotFoundError(f"manifest event index is missing: {index_path}")
        offsets = np.load(index_path, mmap_mode="r")
        expected = int(record["profile"]["event_count"]) + 1
        if offsets.dtype != np.int64 or offsets.ndim != 1 or len(offsets) != expected:
            raise ValueError(f"invalid event-offset index: {index_path}")
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(record["profile"]["row_count"]):
            raise ValueError(f"event-offset endpoints do not match manifest: {index_path}")
    return payload


def _projection_input(coordinates: np.ndarray, config: DataConfig) -> np.ndarray:
    xyz = np.asarray(coordinates, dtype=np.float32)
    origin = np.asarray(config.projection_origin_mm, dtype=np.float32)
    indices = np.floor((xyz - origin) / np.float32(config.projection_bin_size_mm)).astype(
        np.int64
    )
    valid = np.all((indices >= 0) & (indices < config.projection_grid_size), axis=1)
    if not np.all(valid):
        point = xyz[int(np.flatnonzero(~valid)[0])].tolist()
        raise ValueError(f"tracker coordinate falls outside the configured CNN grid: {point}")
    image = np.zeros(
        (len(PROJECTION_PLANES), config.projection_grid_size, config.projection_grid_size),
        dtype=np.float32,
    )
    coordinate_index = {field: index for index, field in enumerate(COORDINATE_FIELDS)}
    for channel, (vertical, horizontal) in enumerate(PROJECTION_PLANES):
        image[
            channel,
            indices[:, coordinate_index[vertical]],
            indices[:, coordinate_index[horizontal]],
        ] = np.float32(config.projection_input_scale)
    return image


def _point_input(
    coordinates: np.ndarray,
    config: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(coordinates, dtype=np.float32)
    center = np.mean(xyz, axis=0, dtype=np.float64)
    centered = xyz.astype(np.float64) - center[None, :]
    cells = np.floor(centered / float(config.point_bin_size_mm)).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_cells)).astype(np.int64)
    if len(unique_cells) > config.max_points:
        raise ValueError(
            f"event needs {len(unique_cells)} topology points, exceeding the fixed "
            f"model limit {config.max_points}; no truncation was performed"
        )
    centers = (unique_cells.astype(np.float64) + 0.5) * float(config.point_bin_size_mm)
    quantized_center = np.average(centers, axis=0, weights=counts.astype(np.float64))
    centers -= quantized_center[None, :]
    occupancy_fraction = counts.astype(np.float64) / float(np.sum(counts))
    features = np.column_stack((occupancy_fraction, np.log1p(counts))).astype(
        np.float32,
        copy=False,
    )
    coordinates_scaled = (centers / float(config.coordinate_scale_mm)).astype(np.float32)
    return coordinates_scaled, features


class SuperNEMOEventDataset(IterableDataset):
    """Stream exact manifest event slices and adapt only the model representation."""

    def __init__(
        self,
        *,
        data_root: Path,
        manifest_path: Path,
        manifest: Mapping[str, Any],
        task: Task,
        split: str,
        input_kind: InputKind,
        config: DataConfig,
        tokenization_config: SuperNEMOTrackerTokenizationConfig | None,
        shuffle_slices: bool,
    ) -> None:
        super().__init__()
        if task not in {"classification", "energy"}:
            raise ValueError(f"unknown task: {task!r}")
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {split!r}")
        if input_kind not in {"projection2d", "graph", "sequence"}:
            raise ValueError(f"unknown input kind: {input_kind!r}")
        if tokenization_config is not None:
            if not isinstance(
                tokenization_config, SuperNEMOTrackerTokenizationConfig
            ):
                raise TypeError(
                    "tokenization_config must be a SuperNEMOTrackerTokenizationConfig"
                )
            if input_kind != "sequence" or task != "classification":
                raise ValueError(
                    "tracker tokenization is supported only for classification sequences"
                )
        all_slices = [EventSlice.from_dict(item) for item in manifest["splits"][split]]
        self.slices = [
            item
            for item in all_slices
            if task == "energy" or SOURCE_BY_KEY[item.source_key].classification_label is not None
        ]
        if not self.slices:
            raise ValueError(f"no eligible {task} events in {split}")
        self.data_root = data_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.task = task
        self.split = split
        self.input_kind = input_kind
        self.config = config
        self.tokenization_config = tokenization_config
        self.shuffle_slices = bool(shuffle_slices)
        self.epoch = 0
        self._length = sum(item.event_count for item in self.slices)

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _sample(
        self,
        source_key: str,
        event_number: int,
        coordinates: np.ndarray,
        tracker_radius: np.ndarray | None,
        energy_kev: float,
    ) -> dict[str, Any]:
        spec = SOURCE_BY_KEY[source_key]
        event_id = f"SuperNEMO::{source_key}::{event_number}"
        token_coverage: float | None = None
        if self.input_kind == "projection2d":
            inputs = {"projections": _projection_input(coordinates, self.config)}
        elif self.tokenization_config is not None:
            if tracker_radius is None:
                raise RuntimeError("tracker tokenization requires the tR field")
            # source_key maps one-to-one to the class, so it must not influence
            # deterministic hit selection. Local event ordinals can safely repeat
            # across sources and give equal-length events the same RNG choices.
            sampling_key = f"SuperNEMO::ev_no::{event_number}"
            inputs, token_coverage = tokenize_tracker_event(
                coordinates,
                tracker_radius,
                sampling_key=sampling_key,
                config=self.tokenization_config,
            )
        else:
            coords, features = _point_input(coordinates, self.config)
            inputs = {"coords": coords, "features": features}
        target = (
            float(spec.classification_label)
            if self.task == "classification" and spec.classification_label is not None
            else float(energy_kev)
        )
        sample = {
            "inputs": inputs,
            "target": np.float32(target),
            # Preserve the float64 sum of the two source float32 calorimeter
            # fields for audit/evaluation. The model target remains float32.
            "energy": np.float64(energy_kev),
            "event_id": event_id,
            "group_id": event_id,
            "category": spec.category,
            "split": self.split,
        }
        if token_coverage is not None:
            sample["token_coverage"] = np.float32(token_coverage)
        return sample

    def _iter_slice(
        self,
        item: EventSlice,
        handle: h5py.File,
        offsets: np.ndarray,
    ) -> Iterator[dict[str, Any]]:
        row_start = int(offsets[item.event_start])
        row_stop = int(offsets[item.event_stop])
        row_count = row_stop - row_start
        relative_offsets = np.asarray(
            offsets[item.event_start : item.event_stop + 1], dtype=np.int64
        ) - np.int64(row_start)
        if (
            relative_offsets.shape != (item.event_count + 1,)
            or int(relative_offsets[0]) != 0
            or int(relative_offsets[-1]) != row_count
            or np.any(np.diff(relative_offsets) <= 0)
        ):
            raise ValueError(
                f"{item.source_key}: manifest event boundaries are not aligned "
                f"for events {item.event_start}:{item.event_stop}"
            )
        runtime_fields = (*COORDINATE_FIELDS, *ENERGY_FIELDS)
        if self.tokenization_config is not None:
            runtime_fields = (*runtime_fields, RADIUS_FIELD)
        arrays = {
            field: np.asarray(handle[field][row_start:row_stop], dtype=np.float32)
            for field in runtime_fields
        }
        if any(len(values) != row_count for values in arrays.values()):
            raise ValueError(f"{item.source_key}: runtime field length mismatch")
        for field in COORDINATE_FIELDS:
            if np.any(~np.isfinite(arrays[field])):
                raise ValueError(f"{item.source_key}:{field} contains a non-finite input value")
        for field in ENERGY_FIELDS:
            values = arrays[field]
            invalid = ~np.isfinite(values)
            if np.any(invalid):
                bad_row = row_start + int(np.flatnonzero(invalid)[0])
                raise ValueError(
                    f"{item.source_key}:{field} contains a non-finite energy "
                    f"value at row {bad_row}"
                )
            non_positive = values <= 0.0
            if np.any(non_positive):
                bad_row = row_start + int(np.flatnonzero(non_positive)[0])
                raise ValueError(
                    f"{item.source_key}:{field} contains a non-positive energy "
                    f"value at row {bad_row}"
                )
        if self.tokenization_config is not None:
            radius = arrays[RADIUS_FIELD]
            invalid = np.isinf(radius) | (np.isfinite(radius) & (radius < 0.0))
            if np.any(invalid):
                bad_row = row_start + int(np.flatnonzero(invalid)[0])
                raise ValueError(
                    f"{item.source_key}:{RADIUS_FIELD} contains an invalid tracker "
                    f"radius at row {bad_row}"
                )

        for event_number in range(item.event_start, item.event_stop):
            local_index = event_number - item.event_start
            local_start = int(relative_offsets[local_index])
            local_stop = int(relative_offsets[local_index + 1])
            e1 = arrays[ENERGY_FIELDS[0]][local_start:local_stop]
            e2 = arrays[ENERGY_FIELDS[1]][local_start:local_stop]
            if not np.all(e1 == e1[0]) or not np.all(e2 == e2[0]):
                raise ValueError(
                    f"{item.source_key}: E1/E2 are not event-aligned for event {event_number}"
                )
            energy = float(e1[0]) + float(e2[0])
            if not math.isfinite(energy) or energy <= 0.0:
                raise ValueError(f"{item.source_key}: invalid energy for event {event_number}")
            coordinates = np.column_stack(
                [arrays[field][local_start:local_stop] for field in COORDINATE_FIELDS]
            ).astype(np.float32, copy=False)
            tracker_radius = (
                arrays[RADIUS_FIELD][local_start:local_stop]
                if self.tokenization_config is not None
                else None
            )
            yield self._sample(
                item.source_key,
                event_number,
                coordinates,
                tracker_radius,
                energy,
            )

    def _training_slices_by_source(
        self,
        worker: Any,
    ) -> tuple[list[str], dict[str, list[EventSlice]]]:
        """Return deterministic, disjoint locality chunks for proportional merging."""

        by_source: dict[str, list[EventSlice]] = {}
        maximum = int(self.config.split_block_events)
        for item in self.slices:
            chunks = by_source.setdefault(item.source_key, [])
            for start in range(item.event_start, item.event_stop, maximum):
                chunks.append(
                    EventSlice(
                        item.source_key,
                        item.split,
                        start,
                        min(start + maximum, item.event_stop),
                    )
                )

        canonical_sources = [
            spec.source_key for spec in SOURCE_SPECS if spec.source_key in by_source
        ]
        source_positions = {
            spec.source_key: index for index, spec in enumerate(SOURCE_SPECS)
        }
        for source_key in canonical_sources:
            generator = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(self.config.seed),
                        int(self.epoch),
                        int(source_positions[source_key]),
                        0,
                    ]
                )
            )
            generator.shuffle(by_source[source_key])
            if worker is not None:
                by_source[source_key] = by_source[source_key][
                    worker.id :: worker.num_workers
                ]

        # Vary only tie-breaking between epochs. The proportional scheduler below
        # still emits every assigned event exactly once at its natural frequency.
        scheduler = np.random.default_rng(
            np.random.SeedSequence(
                [int(self.config.seed), int(self.epoch), 1]
            )
        )
        scheduler.shuffle(canonical_sources)
        return (
            [source for source in canonical_sources if by_source[source]],
            by_source,
        )

    def _iter_training_events(
        self,
        worker: Any,
        handles: dict[str, h5py.File],
        indexes: dict[str, np.ndarray],
    ) -> Iterator[dict[str, Any]]:
        """Interleave sources in natural proportions without sampling events."""

        source_order, by_source = self._training_slices_by_source(worker)
        streams: dict[str, Iterator[dict[str, Any]]] = {}
        assigned: dict[str, int] = {}
        emitted = {source: 0 for source in source_order}
        for source in source_order:
            spec = SOURCE_BY_KEY[source]
            handles[source] = h5py.File(self.data_root / spec.file_name, "r")
            indexes[source] = np.load(
                self.manifest_path.parent / _index_name(source),
                mmap_mode="r",
            )
            assigned[source] = sum(item.event_count for item in by_source[source])
            streams[source] = (
                sample
                for item in by_source[source]
                for sample in self._iter_slice(
                    item,
                    handles[item.source_key],
                    indexes[item.source_key],
                )
            )

        remaining = sum(assigned.values())
        while remaining:
            active = [
                source for source in source_order if emitted[source] < assigned[source]
            ]
            if not active:
                raise RuntimeError("training source scheduler exhausted before its event count")
            chosen = active[0]
            for candidate in active[1:]:
                if (
                    emitted[candidate] * assigned[chosen]
                    < emitted[chosen] * assigned[candidate]
                ):
                    chosen = candidate
            try:
                yield next(streams[chosen])
            except StopIteration as error:
                raise RuntimeError(
                    f"{chosen}: event stream ended before its assigned count"
                ) from error
            emitted[chosen] += 1
            remaining -= 1

        for source, stream in streams.items():
            try:
                next(stream)
            except StopIteration:
                continue
            raise RuntimeError(f"{source}: event stream exceeded its assigned count")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        handles: dict[str, h5py.File] = {}
        indexes: dict[str, np.ndarray] = {}
        try:
            if self.shuffle_slices:
                yield from self._iter_training_events(worker, handles, indexes)
                return

            slices = list(self.slices)
            if worker is not None:
                slices = slices[worker.id :: worker.num_workers]
            for item in slices:
                if item.source_key not in handles:
                    spec = SOURCE_BY_KEY[item.source_key]
                    handles[item.source_key] = h5py.File(
                        self.data_root / spec.file_name,
                        "r",
                    )
                    indexes[item.source_key] = np.load(
                        self.manifest_path.parent / _index_name(item.source_key),
                        mmap_mode="r",
                    )
                yield from self._iter_slice(
                    item,
                    handles[item.source_key],
                    indexes[item.source_key],
                )
        finally:
            for handle in handles.values():
                handle.close()


def collate_events(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad topology points while keeping all audit metadata outside inputs."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch: dict[str, Any] = {
        "target": torch.as_tensor([item["target"] for item in samples], dtype=torch.float32),
        "energy": torch.as_tensor([item["energy"] for item in samples], dtype=torch.float64),
        "event_id": [str(item["event_id"]) for item in samples],
        "group_id": [str(item["group_id"]) for item in samples],
        "category": [str(item["category"]) for item in samples],
        "split": [str(item["split"]) for item in samples],
    }
    coverage_presence = ["token_coverage" in item for item in samples]
    if any(coverage_presence) and not all(coverage_presence):
        raise ValueError("token_coverage must be present for every event or none")
    if all(coverage_presence):
        coverage = np.asarray(
            [item["token_coverage"] for item in samples], dtype=np.float32
        )
        if coverage.shape != (len(samples),) or np.any(~np.isfinite(coverage)):
            raise ValueError("token_coverage must contain one finite value per event")
        if np.any((coverage <= 0.0) | (coverage > 1.0)):
            raise ValueError("token_coverage values must be in (0, 1]")
        batch["token_coverage"] = torch.from_numpy(coverage)
    first_inputs = samples[0]["inputs"]
    if "projections" in first_inputs:
        if all(coverage_presence):
            raise ValueError("projection inputs cannot carry tracker token coverage")
        if not all(set(item["inputs"]) == {"projections"} for item in samples):
            raise ValueError("inconsistent CNN input fields")
        batch["inputs"] = {
            "projections": torch.from_numpy(
                np.stack([np.asarray(item["inputs"]["projections"]) for item in samples])
            ).to(dtype=torch.float32)
        }
        return batch
    if not all(set(item["inputs"]) == {"coords", "features"} for item in samples):
        raise ValueError("inconsistent point input fields")
    counts = [int(len(item["inputs"]["coords"])) for item in samples]
    if any(count <= 0 for count in counts):
        raise ValueError("every event must contain at least one topology point")
    first_coords = np.asarray(first_inputs["coords"])
    first_features = np.asarray(first_inputs["features"])
    if first_coords.ndim != 2 or first_coords.shape[1] != len(COORDINATE_FIELDS):
        raise ValueError("point coordinates must have shape [number_of_points, 3]")
    if first_features.ndim != 2 or first_features.shape[1] <= 0:
        raise ValueError("point features must have a positive feature dimension")
    maximum = max(counts)
    coordinate_dim = len(COORDINATE_FIELDS)
    feature_dim = int(first_features.shape[1])
    coords = torch.zeros((len(samples), maximum, coordinate_dim), dtype=torch.float32)
    features = torch.zeros((len(samples), maximum, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
    for index, (item, count) in enumerate(zip(samples, counts)):
        item_coords = torch.as_tensor(item["inputs"]["coords"], dtype=torch.float32)
        item_features = torch.as_tensor(item["inputs"]["features"], dtype=torch.float32)
        if item_coords.shape != (count, coordinate_dim) or item_features.shape != (
            count,
            feature_dim,
        ):
            raise ValueError(
                "invalid point input shape or inconsistent batch feature dimension"
            )
        if not bool(torch.isfinite(item_coords).all()) or not bool(
            torch.isfinite(item_features).all()
        ):
            raise ValueError("point inputs must contain only finite values")
        coords[index, :count] = item_coords
        features[index, :count] = item_features
        mask[index, :count] = True
    batch["inputs"] = {"coords": coords, "features": features, "mask": mask}
    return batch


def prepare_dataset(
    *,
    task: Task,
    input_kind: InputKind,
    data_config: DataConfig | None = None,
    tokenization_config: SuperNEMOTrackerTokenizationConfig | None = None,
    batch_size: int,
    num_workers: int,
) -> PreparedData:
    """Create all loaders from one immutable event split manifest."""

    config = data_config or DataConfig()
    if task not in {"classification", "energy"}:
        raise ValueError("task must be 'classification' or 'energy'")
    if input_kind not in {"projection2d", "graph", "sequence"}:
        raise ValueError("unsupported model input kind")
    if tokenization_config is not None:
        if not isinstance(tokenization_config, SuperNEMOTrackerTokenizationConfig):
            raise TypeError(
                "tokenization_config must be a SuperNEMOTrackerTokenizationConfig"
            )
        if input_kind != "sequence" or task != "classification":
            raise ValueError(
                "tracker tokenization is supported only for classification sequences"
            )
    if int(batch_size) <= 0 or int(num_workers) < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    manifest = load_manifest(config)
    datasets = {
        split: SuperNEMOEventDataset(
            data_root=config.data_root,
            manifest_path=config.manifest_path,
            manifest=manifest,
            task=task,
            split=split,
            input_kind=input_kind,
            config=config,
            tokenization_config=tokenization_config,
            shuffle_slices=split == "train",
        )
        for split in SPLIT_NAMES
    }
    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "shuffle": False,
        "collate_fn": collate_events,
        "persistent_workers": False,
    }
    loaders = {
        split: _PrefetchLoader(DataLoader(datasets[split], **loader_options))
        for split in SPLIT_NAMES
    }
    counts = manifest["counts"][task]
    return PreparedData(
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        test_loader=loaders["test"],
        counts=counts,
        manifest_path=config.manifest_path.resolve(),
        task=task,
        input_kind=input_kind,
        tokenization_config=tokenization_config,
    )


def manifest_summary(config: DataConfig | None = None) -> dict[str, Any]:
    """Return the validated counts and field profiles without loading events."""

    manifest = load_manifest(config)
    return {
        "files": manifest["files"],
        "counts": manifest["counts"],
        "settings": manifest["settings"],
        "event_key": manifest["event_key"],
        "grouping": manifest["grouping"],
        "input_policy": manifest["input_policy"],
    }


__all__ = [
    "EventSlice",
    "PreparedData",
    "SuperNEMOEventDataset",
    "build_manifest",
    "collate_events",
    "load_manifest",
    "manifest_summary",
    "prepare_dataset",
]
