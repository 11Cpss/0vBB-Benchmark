"""Lazy EXO-200 HDF5 loading and leakage-safe grouped splits."""

from __future__ import annotations

import queue
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    BACKGROUND_LABEL,
    CLASS_NAMES,
    SIGNAL_LABEL,
    SIGNAL_NCCL,
    TASK,
    DataConfig,
)


@dataclass(frozen=True)
class _FileInfo:
    path: Path
    offset: int
    length: int
    run_number: int
    event_numbers: np.ndarray
    labels: np.ndarray
    energies: np.ndarray
    nccl_counts: dict[int, int]
    label_diagnostics: dict[str, int]
    waveform_dtype: str
    waveform_compression: str | None
    waveform_compression_opts: Any
    waveform_chunks: tuple[int, ...] | None


@dataclass(frozen=True)
class _SplitPlan:
    indices: dict[str, np.ndarray]
    counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    runs: dict[str, list[int]]
    overlap_counts: dict[str, int]


@dataclass(frozen=True)
class EXODataLoaders:
    train_loader: Any
    validation_loader: Any
    test_loader: Any
    counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    runs: dict[str, list[int]]
    overlap_counts: dict[str, int]
    task: str = TASK

    @property
    def val_loader(self) -> Any:
        return self.validation_loader

    def require_two_classes(self) -> None:
        missing = {
            split: [name for name in CLASS_NAMES if counts.get(name, 0) == 0]
            for split, counts in self.class_counts.items()
        }
        missing = {split: names for split, names in missing.items() if names}
        if missing:
            raise ValueError(
                "cannot start EXO-200 binary classification: each train, validation, "
                "and test split must contain signal (label 0, nccl == 1) and background "
                f"(label 1, nccl > 1); missing classes: {missing}; observed counts: "
                f"{self.class_counts}"
            )


def discover_files(data_root: str | Path) -> list[Path]:
    """Return sorted EXO-200 HDF5 files without modifying the data directory."""

    root = Path(data_root).expanduser().resolve()
    return sorted(root.glob("*.h5"))


def _validate_index_group(
    path: Path,
    field: str,
    group: h5py.Group,
    expected_length: int,
) -> None:
    invalid: list[str] = []
    for key in group.keys():
        if (
            not key.isdecimal()
            or key != str(int(key))
            or not 0 <= int(key) < expected_length
        ):
            if len(invalid) < 5:
                invalid.append(key)
    if len(group) != expected_length or invalid:
        raise ValueError(
            f"{path}: group {field!r} must contain exactly string keys "
            f"0..{expected_length - 1}; found {len(group)} entries and invalid keys "
            f"{invalid}"
        )


def _label_diagnostics(values: np.ndarray) -> dict[str, int]:
    diagnostics = {
        "negative": 0,
        "less_than_one": 0,
        "nan": 0,
        "positive_infinity": 0,
        "negative_infinity": 0,
        "non_integer": 0,
        "uninterpretable": 0,
    }
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        diagnostics["uninterpretable"] = int(values.size)
        return diagnostics
    numeric = values.astype(np.float64, copy=False)
    diagnostics["nan"] = int(np.isnan(numeric).sum())
    diagnostics["positive_infinity"] = int(np.isposinf(numeric).sum())
    diagnostics["negative_infinity"] = int(np.isneginf(numeric).sum())
    finite = np.isfinite(numeric)
    diagnostics["negative"] = int(np.sum(finite & (numeric < 0.0)))
    diagnostics["less_than_one"] = int(
        np.sum(
            finite
            & (numeric >= 0.0)
            & (numeric < 1.0)
            & (numeric == np.floor(numeric))
        )
    )
    diagnostics["non_integer"] = int(
        np.sum(finite & (numeric != np.floor(numeric)))
    )
    return diagnostics


def _labels_from_nccl(
    path: Path,
    field: str,
    values: np.ndarray,
    expected_length: int,
) -> tuple[np.ndarray, dict[str, int], dict[int, int]]:
    if values.shape != (expected_length,):
        raise ValueError(
            f"{path}: field {field!r} must have shape ({expected_length},), "
            f"found {values.shape}"
        )
    diagnostics = _label_diagnostics(values)
    if any(diagnostics.values()):
        raise ValueError(
            f"{path}: field {field!r} contains invalid nccl values: {diagnostics}"
        )
    numeric = values.astype(np.float64, copy=False)
    outside_int64 = np.isfinite(numeric) & (numeric >= float(2**63))
    if bool(outside_int64.any()):
        diagnostics["uninterpretable"] += int(outside_int64.sum())
        raise ValueError(
            f"{path}: field {field!r} contains invalid nccl values: {diagnostics}"
        )
    integer_values = values.astype(np.int64, copy=False)
    unique, counts = np.unique(integer_values, return_counts=True)
    nccl_counts = {
        int(value): int(count) for value, count in zip(unique.tolist(), counts.tolist())
    }
    labels = (integer_values > SIGNAL_NCCL).astype(np.int64, copy=False)
    return labels, diagnostics, nccl_counts


class EXOWaveformDataset(Dataset):
    """Map-style lazy reader over EXO waveform groups.

    ``Charge_cluster_number`` is converted to labels during construction and is
    never returned as an input. ``Charge_Clusters_Pos`` is schema-checked but
    excluded because its column count directly reveals nccl.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        config: DataConfig | None = None,
    ) -> None:
        if not paths:
            raise FileNotFoundError("no EXO-200 .h5 files were supplied")
        self.config = config or DataConfig()
        self._files: list[_FileInfo] = []
        self._handles: dict[Path, h5py.File] = {}
        offset = 0

        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            try:
                with h5py.File(path, "r") as handle:
                    for attribute in ("n_events", self.config.group_attribute):
                        if attribute not in handle.attrs:
                            raise KeyError(f"missing root attribute {attribute!r}")
                    length = int(handle.attrs["n_events"])
                    run_number = int(handle.attrs[self.config.group_attribute])
                    if length <= 0:
                        raise ValueError("root attribute 'n_events' must be positive")
                    file_run = path.stem.split("-", maxsplit=1)[0]
                    if file_run.isdecimal() and int(file_run) != run_number:
                        raise ValueError(
                            f"filename run {file_run} disagrees with root run_number "
                            f"{run_number}"
                        )

                    required = (
                        self.config.input_field,
                        self.config.label_field,
                        self.config.event_field,
                        self.config.position_field,
                        "Rotated_energy",
                    )
                    missing = [name for name in required if name not in handle]
                    if missing:
                        raise KeyError(f"missing required fields: {missing}")
                    waveforms = handle[self.config.input_field]
                    positions = handle[self.config.position_field]
                    if not isinstance(waveforms, h5py.Group):
                        raise TypeError(
                            f"field {self.config.input_field!r} must be an HDF5 group"
                        )
                    if not isinstance(positions, h5py.Group):
                        raise TypeError(
                            f"field {self.config.position_field!r} must be an HDF5 group"
                        )
                    _validate_index_group(
                        path, self.config.input_field, waveforms, length
                    )
                    _validate_index_group(
                        path, self.config.position_field, positions, length
                    )

                    for field, value in handle.items():
                        if isinstance(value, h5py.Dataset):
                            if value.ndim < 1 or int(value.shape[0]) != length:
                                raise ValueError(
                                    f"{path}: event field {field!r} has shape "
                                    f"{value.shape}, inconsistent with n_events={length}"
                                )

                    event_dataset = handle[self.config.event_field]
                    event_numbers = np.asarray(event_dataset[:])
                    if event_numbers.shape != (length,) or not np.issubdtype(
                        event_numbers.dtype, np.integer
                    ):
                        raise ValueError(
                            f"{path}: field {self.config.event_field!r} must be an "
                            f"integer array with shape ({length},)"
                        )
                    event_numbers = event_numbers.astype(np.int64, copy=False)

                    nccl = np.asarray(handle[self.config.label_field][:])
                    labels, diagnostics, nccl_counts = _labels_from_nccl(
                        path,
                        self.config.label_field,
                        nccl,
                        length,
                    )

                    energies = np.asarray(handle["Rotated_energy"][:], dtype=np.float64)
                    if energies.shape != (length,):
                        raise ValueError("Rotated_energy must have one scalar keV value per row")

                    example = waveforms["0"]
                    if tuple(example.shape) != self.config.input_shape:
                        raise ValueError(
                            f"{path}: {self.config.input_field!r}/0 has shape "
                            f"{example.shape}, expected {self.config.input_shape}"
                        )
                    if np.dtype(example.dtype) != np.dtype(self.config.input_dtype):
                        raise TypeError(
                            f"{path}: {self.config.input_field!r}/0 has dtype "
                            f"{example.dtype}, expected {self.config.input_dtype}"
                        )
                    info = _FileInfo(
                        path=path,
                        offset=offset,
                        length=length,
                        run_number=run_number,
                        event_numbers=event_numbers,
                        labels=labels,
                        energies=energies,
                        nccl_counts=nccl_counts,
                        label_diagnostics=diagnostics,
                        waveform_dtype=str(example.dtype),
                        waveform_compression=example.compression,
                        waveform_compression_opts=example.compression_opts,
                        waveform_chunks=example.chunks,
                    )
            except OSError as error:
                raise OSError(f"cannot open EXO-200 file {path}: {error}") from error
            except (KeyError, TypeError, ValueError) as error:
                raise type(error)(f"{path}: {error}") from error
            self._files.append(info)
            offset += length

        self._offsets = np.asarray(
            [*(info.offset for info in self._files), offset], dtype=np.int64
        )
        self._validate_event_identities()

    def _validate_event_identities(self) -> None:
        by_run: dict[int, list[_FileInfo]] = defaultdict(list)
        for info in self._files:
            by_run[info.run_number].append(info)
        for run_number, files in by_run.items():
            events = np.concatenate([info.event_numbers for info in files])
            unique, counts = np.unique(events, return_counts=True)
            duplicate_events = unique[counts > 1]
            if duplicate_events.size:
                raise ValueError(
                    f"files {[str(info.path) for info in files]}: field "
                    f"{self.config.event_field!r} contains "
                    f"{int(np.sum(counts[counts > 1] - 1))} duplicate "
                    f"(run_number, event_number) identities for run {run_number}; "
                    f"example event numbers: {duplicate_events[:5].tolist()}"
                )

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def close(self) -> None:
        handles = getattr(self, "_handles", {})
        for handle in handles.values():
            handle.close()
        handles.clear()

    def __del__(self) -> None:
        self.close()

    def _handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return int(index)

    def _preprocess(self, path: Path, row: int, waveform: np.ndarray) -> np.ndarray:
        if tuple(waveform.shape) != self.config.input_shape:
            raise ValueError(
                f"{path}: {self.config.input_field}/{row} has shape {waveform.shape}, "
                f"expected {self.config.input_shape}"
            )
        if np.dtype(waveform.dtype) != np.dtype(self.config.input_dtype):
            raise TypeError(
                f"{path}: {self.config.input_field}/{row} has dtype {waveform.dtype}, "
                f"expected {self.config.input_dtype}"
            )
        values = np.asarray(waveform, dtype=np.float32)
        baseline_count = min(self.config.baseline_samples, values.shape[-1])
        baseline = np.mean(
            values[:, :baseline_count], axis=-1, dtype=np.float64
        ).astype(np.float32, copy=False)
        values = np.asarray(values - baseline[:, np.newaxis], dtype=np.float32)
        if self.config.classification_amplitude_normalization:
            scale = float(np.max(np.abs(values)))
            if scale > 0.0:
                values /= scale
        if not np.isfinite(values).all():
            raise FloatingPointError(
                f"{path}: preprocessing produced non-finite values for row {row}"
            )
        return values

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        normalized = np.asarray(
            [self._normalize_index(int(index)) for index in indices], dtype=np.int64
        )
        if normalized.size == 0:
            return []
        file_indices = np.searchsorted(
            self._offsets, normalized, side="right"
        ) - 1
        outputs: list[dict[str, Any] | None] = [None] * normalized.size

        for file_index in np.unique(file_indices):
            positions = np.flatnonzero(file_indices == file_index)
            info = self._files[int(file_index)]
            rows = normalized[positions] - info.offset
            handle = self._handle(info.path)
            waveform_group = handle[self.config.input_field]
            for position, row_value in zip(positions.tolist(), rows.tolist()):
                row = int(row_value)
                raw_waveform = np.asarray(waveform_group[str(row)][:])
                waveform = self._preprocess(info.path, row, raw_waveform)
                outputs[position] = {
                    "inputs": torch.from_numpy(waveform.copy()),
                    "energy_keV": torch.tensor(float(info.energies[row]), dtype=torch.float64),
                    "event_id": f"EXO200::{info.path.name}::{int(info.event_numbers[row])}",
                    "source_file": info.path.name,
                    "label": torch.tensor(int(info.labels[row]), dtype=torch.int64),
                    "row": torch.tensor(row, dtype=torch.int64),
                    "run_number": torch.tensor(info.run_number, dtype=torch.int64),
                    "event_number": torch.tensor(
                        int(info.event_numbers[row]), dtype=torch.int64
                    ),
                    "file_index": torch.tensor(int(file_index), dtype=torch.int64),
                }

        if any(output is None for output in outputs):
            raise RuntimeError("failed to load an EXO-200 batch")
        return [output for output in outputs if output is not None]

    def labels_for_indices(self, indices: np.ndarray) -> np.ndarray:
        labels = np.empty(indices.size, dtype=np.int64)
        file_indices = np.searchsorted(self._offsets, indices, side="right") - 1
        for file_index in np.unique(file_indices):
            positions = np.flatnonzero(file_indices == file_index)
            info = self._files[int(file_index)]
            rows = indices[positions] - info.offset
            labels[positions] = info.labels[rows]
        return labels


class _IndexedDataset(Dataset):
    def __init__(self, source: EXOWaveformDataset, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.source[int(self.indices[index])]

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        mapped = self.indices[np.asarray(indices, dtype=np.int64)]
        return self.source.__getitems__(mapped.tolist())


class _PrefetchLoader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
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


def _group_statistics(source: EXOWaveformDataset) -> dict[int, np.ndarray]:
    statistics: dict[int, np.ndarray] = {}
    for info in source._files:
        values = statistics.setdefault(info.run_number, np.zeros(3, dtype=np.int64))
        values[0] += info.length
        values[1] += int(np.sum(info.labels == SIGNAL_LABEL))
        values[2] += int(np.sum(info.labels == BACKGROUND_LABEL))
    return statistics


def _choose_group_subset(
    statistics: dict[int, np.ndarray],
    fraction: float,
    seed: int,
) -> set[int]:
    if len(statistics) < 2:
        raise ValueError("a grouped split requires at least two run groups")
    group_ids = np.asarray(sorted(statistics), dtype=np.int64)
    np.random.default_rng(seed).shuffle(group_ids)
    vectors = [statistics[int(group)] for group in group_ids]
    total = np.sum(vectors, axis=0)
    target = total.astype(np.float64) * float(fraction)
    active_classes = np.flatnonzero(total[1:] > 0) + 1
    best_mask = 0
    best_key: tuple[float, float] | None = None

    for mask in range(1, (1 << len(group_ids)) - 1):
        selected = np.zeros(3, dtype=np.int64)
        for index, vector in enumerate(vectors):
            if mask & (1 << index):
                selected += vector
        total_error = abs(float(selected[0]) - target[0]) / max(target[0], 1.0)
        class_errors = [
            abs(float(selected[index]) - target[index]) / max(target[index], 1.0)
            for index in active_classes.tolist()
        ]
        score = total_error + (
            float(np.mean(class_errors)) if class_errors else 0.0
        )
        for index in active_classes.tolist():
            if selected[index] == 0 or selected[index] == total[index]:
                score += 1000.0
        key = (score, abs(float(selected[0]) - target[0]))
        if best_key is None or key < best_key:
            best_key = key
            best_mask = mask

    return {
        int(group_ids[index])
        for index in range(len(group_ids))
        if best_mask & (1 << index)
    }


def _indices_for_runs(source: EXOWaveformDataset, runs: set[int]) -> np.ndarray:
    pieces = [
        np.arange(info.offset, info.offset + info.length, dtype=np.int64)
        for info in source._files
        if info.run_number in runs
    ]
    if not pieces:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(pieces)


def _split_plan(source: EXOWaveformDataset, config: DataConfig) -> _SplitPlan:
    statistics = _group_statistics(source)
    if len(statistics) < 3:
        raise ValueError("train, validation, and test require at least three run groups")
    test_runs = _choose_group_subset(statistics, config.test_fraction, config.seed)
    remaining_statistics = {
        run: values for run, values in statistics.items() if run not in test_runs
    }
    validation_runs = _choose_group_subset(
        remaining_statistics,
        config.validation_fraction,
        config.seed,
    )
    train_runs = set(statistics) - test_runs - validation_runs
    run_sets = {
        "train": train_runs,
        "validation": validation_runs,
        "test": test_runs,
    }
    if any(not values for values in run_sets.values()):
        raise ValueError(f"group split produced an empty split: {run_sets}")

    indices = {
        split: _indices_for_runs(source, values)
        for split, values in run_sets.items()
    }
    all_indices = np.concatenate(list(indices.values()))
    if all_indices.size != len(source) or np.unique(all_indices).size != len(source):
        raise RuntimeError("group split does not assign every event exactly once")

    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlap_counts = {
        # Global offsets identify source events one-to-one because construction
        # rejects duplicate (run_number, event_number) identities above.
        f"{left}_{right}": int(
            np.intersect1d(indices[left], indices[right], assume_unique=True).size
        )
        for left, right in pairs
    }
    if any(overlap_counts.values()):
        raise RuntimeError(f"event overlap detected across splits: {overlap_counts}")

    class_counts: dict[str, dict[str, int]] = {}
    for split, split_indices in indices.items():
        labels = source.labels_for_indices(split_indices)
        class_counts[split] = {
            CLASS_NAMES[SIGNAL_LABEL]: int(np.sum(labels == SIGNAL_LABEL)),
            CLASS_NAMES[BACKGROUND_LABEL]: int(np.sum(labels == BACKGROUND_LABEL)),
        }
    return _SplitPlan(
        indices=indices,
        counts={split: int(values.size) for split, values in indices.items()},
        class_counts=class_counts,
        runs={split: sorted(values) for split, values in run_sets.items()},
        overlap_counts=overlap_counts,
    )


def _full_event_scan(
    source: EXOWaveformDataset,
    *,
    sample_events_per_file: int,
) -> dict[str, Any]:
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    compression_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    position_mismatches = 0
    position_dtype_mismatches = 0
    position_nonfinite = 0
    position_mismatch_examples: list[dict[str, Any]] = []
    position_nonfinite_examples: list[dict[str, Any]] = []
    sampled_events = 0
    sampled_values = 0
    sample_min: int | None = None
    sample_max: int | None = None
    sample_zero = 0
    sample_negative = 0

    for info in source._files:
        with h5py.File(info.path, "r") as handle:
            waveforms = handle[source.config.input_field]
            positions = handle[source.config.position_field]
            nccl = np.asarray(handle[source.config.label_field][:])
            sample_rows = set(
                np.linspace(
                    0,
                    info.length - 1,
                    min(max(int(sample_events_per_file), 0), info.length),
                    dtype=np.int64,
                ).tolist()
            )
            for row in range(info.length):
                waveform = waveforms[str(row)]
                shape_counts[str(tuple(waveform.shape))] += 1
                dtype_counts[str(waveform.dtype)] += 1
                compression_counts[
                    f"{waveform.compression}:{waveform.compression_opts}"
                ] += 1
                chunk_counts[str(waveform.chunks)] += 1
                if tuple(waveform.shape) != source.config.input_shape:
                    raise ValueError(
                        f"{info.path}: {source.config.input_field}/{row} has shape "
                        f"{waveform.shape}, expected {source.config.input_shape}"
                    )
                if np.dtype(waveform.dtype) != np.dtype(source.config.input_dtype):
                    raise TypeError(
                        f"{info.path}: {source.config.input_field}/{row} has dtype "
                        f"{waveform.dtype}, expected {source.config.input_dtype}"
                    )
                position = np.asarray(positions[str(row)][:])
                if position.ndim != 2 or position.shape != (3, int(nccl[row])):
                    position_mismatches += 1
                    if len(position_mismatch_examples) < 10:
                        position_mismatch_examples.append(
                            {
                                "path": str(info.path),
                                "row": row,
                                "field": source.config.position_field,
                                "shape": list(position.shape),
                                "expected_shape": [3, int(nccl[row])],
                            }
                        )
                if position.dtype != np.dtype("float32"):
                    position_dtype_mismatches += 1
                    if len(position_mismatch_examples) < 10:
                        position_mismatch_examples.append(
                            {
                                "path": str(info.path),
                                "row": row,
                                "field": source.config.position_field,
                                "dtype": str(position.dtype),
                                "expected_dtype": "float32",
                            }
                        )
                if not np.isfinite(position).all():
                    invalid_count = int(
                        np.size(position) - np.isfinite(position).sum()
                    )
                    position_nonfinite += invalid_count
                    if len(position_nonfinite_examples) < 10:
                        position_nonfinite_examples.append(
                            {
                                "path": str(info.path),
                                "row": row,
                                "field": source.config.position_field,
                                "nonfinite_values": invalid_count,
                            }
                        )
                if row in sample_rows:
                    values = np.asarray(waveform[:])
                    sampled_events += 1
                    sampled_values += int(values.size)
                    value_min = int(values.min())
                    value_max = int(values.max())
                    sample_min = value_min if sample_min is None else min(sample_min, value_min)
                    sample_max = value_max if sample_max is None else max(sample_max, value_max)
                    sample_zero += int(np.sum(values == 0))
                    sample_negative += int(np.sum(values < 0))

    return {
        "waveform_shape_counts": dict(shape_counts),
        "waveform_dtype_counts": dict(dtype_counts),
        "waveform_compression_counts": dict(compression_counts),
        "waveform_chunk_counts": dict(chunk_counts),
        "position_shape_mismatches": position_mismatches,
        "position_dtype_mismatches": position_dtype_mismatches,
        "position_mismatch_examples": position_mismatch_examples,
        "position_nonfinite_values": position_nonfinite,
        "position_nonfinite_examples": position_nonfinite_examples,
        "sampled_waveform_events": sampled_events,
        "sampled_waveform_values": sampled_values,
        "sampled_waveform_min": sample_min,
        "sampled_waveform_max": sample_max,
        "sampled_waveform_zero_values": sample_zero,
        "sampled_waveform_negative_values": sample_negative,
    }


def inspect_data_root(
    data_config: DataConfig | None = None,
    *,
    full_event_scan: bool = False,
    sample_events_per_file: int = 0,
) -> dict[str, Any]:
    """Inspect schema, labels, identities, and the deterministic grouped split."""

    config = data_config or DataConfig()
    paths = discover_files(config.data_root)
    source = EXOWaveformDataset(paths, config=config)
    try:
        plan = _split_plan(source, config)
        nccl_counts: Counter[int] = Counter()
        diagnostics: Counter[str] = Counter()
        files = []
        for info in source._files:
            nccl_counts.update(info.nccl_counts)
            diagnostics.update(info.label_diagnostics)
            files.append(
                {
                    "path": str(info.path),
                    "run_number": info.run_number,
                    "events": info.length,
                    "event_number_min": int(info.event_numbers.min()),
                    "event_number_max": int(info.event_numbers.max()),
                    "nccl_counts": info.nccl_counts,
                    "signal": int(np.sum(info.labels == SIGNAL_LABEL)),
                    "background": int(np.sum(info.labels == BACKGROUND_LABEL)),
                }
            )
        report: dict[str, Any] = {
            "data_root": str(config.data_root.resolve()),
            "files": files,
            "file_count": len(source._files),
            "run_count": len({info.run_number for info in source._files}),
            "events": len(source),
            "input_field": config.input_field,
            "input_shape": list(config.input_shape),
            "input_dtype": config.input_dtype,
            "label_field": config.label_field,
            "class_names": list(config.class_names),
            "label_rule": {
                f"nccl == {SIGNAL_NCCL}": "signal (label 0)",
                f"nccl > {SIGNAL_NCCL}": "background (label 1)",
            },
            "nccl_counts": dict(sorted(nccl_counts.items())),
            "nccl_min": min(nccl_counts),
            "nccl_max": max(nccl_counts),
            "label_anomalies": dict(diagnostics),
            "class_counts": {
                CLASS_NAMES[SIGNAL_LABEL]: int(
                    sum(np.sum(info.labels == SIGNAL_LABEL) for info in source._files)
                ),
                CLASS_NAMES[BACKGROUND_LABEL]: int(
                    sum(np.sum(info.labels == BACKGROUND_LABEL) for info in source._files)
                ),
            },
            "event_identity": "(run_number, event_number)",
            "duplicate_event_identities": 0,
            "excluded_leakage_fields": [
                config.label_field,
                "nccl",
                "label",
                config.position_field,
            ],
            "split": {
                "counts": plan.counts,
                "class_counts": plan.class_counts,
                "runs": plan.runs,
                "overlap_counts": plan.overlap_counts,
            },
        }
        if full_event_scan:
            report["full_event_scan"] = _full_event_scan(
                source,
                sample_events_per_file=sample_events_per_file,
            )
        return report
    finally:
        source.close()


def prepare_dataset(
    *,
    data_config: DataConfig | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> EXODataLoaders:
    """Prepare run-grouped train, validation, and test loaders."""

    config = data_config or DataConfig()
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    paths = discover_files(config.data_root)
    if not paths:
        raise FileNotFoundError(f"no .h5 files found in {config.data_root}")
    source = EXOWaveformDataset(paths, config=config)
    plan = _split_plan(source, config)
    datasets = {
        split: _IndexedDataset(source, indices)
        for split, indices in plan.indices.items()
    }
    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return EXODataLoaders(
        train_loader=_PrefetchLoader(
            DataLoader(
                datasets["train"],
                shuffle=True,
                generator=generator,
                **loader_options,
            )
        ),
        validation_loader=_PrefetchLoader(
            DataLoader(datasets["validation"], shuffle=False, **loader_options)
        ),
        test_loader=_PrefetchLoader(
            DataLoader(datasets["test"], shuffle=False, **loader_options)
        ),
        counts=plan.counts,
        class_counts=plan.class_counts,
        runs=plan.runs,
        overlap_counts=plan.overlap_counts,
    )


__all__ = [
    "EXODataLoaders",
    "EXOWaveformDataset",
    "discover_files",
    "inspect_data_root",
    "prepare_dataset",
]
