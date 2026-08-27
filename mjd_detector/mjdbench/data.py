"""Lazy HDF5 loading and reproducible splits for MJD waveforms."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig


Task = Literal["classification", "regression", "inference"]
PSD_NAMES = ("low_avse", "high_avse", "dcr", "lq")

# The aliases make the reader tolerant of the human-readable and underscore
# spellings used in MJD documentation and derivative copies.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "waveform": ("raw_waveform", "raw waveform"),
    "energy": ("energy_label", "energy label", "energy"),
    "low_avse": ("psd_label_low_avse", "psd label low avse"),
    "high_avse": ("psd_label_high_avse", "psd label high avse"),
    "dcr": ("psd_label_dcr", "psd label dcr"),
    "lq": ("psd_label_lq", "psd label lq"),
    "tp0": ("tp0",),
    "detector": ("detector",),
    "run_number": ("run_number", "run number"),
    "id": ("id",),
}


@dataclass(frozen=True)
class _FileInfo:
    path: Path
    length: int
    fields: dict[str, str]
    selected_rows: np.ndarray | None
    labels: np.ndarray | None
    energies: np.ndarray | None
    clean: np.ndarray | None
    metadata: dict[str, np.ndarray]

    @property
    def selected_length(self) -> int:
        return self.length if self.selected_rows is None else int(self.selected_rows.size)


@dataclass(frozen=True)
class MJDDataLoaders:
    train_loader: Any
    validation_loader: Any
    test_loader: Any
    counts: dict[str, int]
    task: str

    @property
    def val_loader(self) -> Any:
        return self.validation_loader


def discover_files(data_root: str | Path, split: str) -> list[Path]:
    """Return sorted official MJD shard paths for one split."""

    normalized = str(split).strip().lower()
    names = {"train": "Train", "test": "Test", "npml": "NPML"}
    if normalized not in names:
        raise ValueError("split must be 'train', 'test', or 'npml'")
    root = Path(data_root).expanduser().resolve()
    return sorted(root.glob(f"MJD_{names[normalized]}_*.hdf5"))


def _all_dataset_paths(handle: h5py.File) -> set[str]:
    paths: set[str] = set()

    def visitor(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            paths.add(name)

    handle.visititems(visitor)
    return paths


def _resolve_fields(handle: h5py.File, *, labeled: bool) -> dict[str, str]:
    paths = _all_dataset_paths(handle)
    basenames = {Path(path).name: path for path in paths}
    required = ["waveform"]
    if labeled:
        required.extend(("energy", *PSD_NAMES))
    resolved: dict[str, str] = {}
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in paths:
                resolved[field] = alias
                break
            if alias in basenames:
                resolved[field] = basenames[alias]
                break
        if field in required and field not in resolved:
            raise KeyError(
                f"missing MJD field {field!r}; available datasets: {sorted(paths)}"
            )
    return resolved


def inspect_data_root(data_root: str | Path) -> list[dict[str, Any]]:
    """Inspect every local MJD shard without reading full waveform arrays."""

    root = Path(data_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("MJD_*.hdf5")):
        split = "npml" if "_NPML_" in path.name else (
            "train" if "_Train_" in path.name else "test"
        )
        record: dict[str, Any] = {
            "path": str(path),
            "split": split,
            "bytes": path.stat().st_size,
            "status": "ok",
        }
        try:
            with h5py.File(path, "r") as handle:
                fields = _resolve_fields(handle, labeled=split != "npml")
                waveform = handle[fields["waveform"]]
                record["events"] = int(waveform.shape[0])
                record["waveform_shape"] = list(waveform.shape[1:])
                record["waveform_dtype"] = str(waveform.dtype)
                record["fields"] = sorted(fields)
        except (OSError, KeyError, ValueError) as error:
            record["status"] = "incomplete" if isinstance(error, OSError) else "invalid"
            record["error"] = str(error)
        rows.append(record)
    return rows


class MJDWaveformDataset(Dataset):
    """Map-style lazy reader over one or more complete MJD shards."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        task: Task,
        baseline_samples: int = 200,
        classification_amplitude_normalization: bool = True,
        regression_waveform_scale: float = 1.0,
    ) -> None:
        if task not in {"classification", "regression", "inference"}:
            raise ValueError("unknown MJD task")
        if not paths:
            raise FileNotFoundError(f"no complete MJD files supplied for {task}")
        self.task = task
        self.baseline_samples = int(baseline_samples)
        self.classification_amplitude_normalization = bool(
            classification_amplitude_normalization
        )
        self.regression_waveform_scale = float(regression_waveform_scale)
        self._files: list[_FileInfo] = []
        self._handles: dict[Path, h5py.File] = {}

        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            try:
                with h5py.File(path, "r") as handle:
                    fields = _resolve_fields(handle, labeled=task != "inference")
                    waveform = handle[fields["waveform"]]
                    if waveform.ndim != 2:
                        raise ValueError(
                            f"{path.name}: raw_waveform must have shape [N, L]"
                        )
                    length = int(waveform.shape[0])
                    for field, dataset_path in fields.items():
                        if int(handle[dataset_path].shape[0]) != length:
                            raise ValueError(
                                f"{path.name}: field {field!r} has inconsistent length"
                            )
                    labels: np.ndarray | None = None
                    energies: np.ndarray | None = None
                    clean: np.ndarray | None = None
                    selected: np.ndarray | None = None
                    if task != "inference":
                        labels = np.column_stack(
                            [
                                np.asarray(handle[fields[name]][:]).reshape(-1)
                                for name in PSD_NAMES
                            ]
                        ).astype(np.float32, copy=False)
                        energies = np.asarray(
                            handle[fields["energy"]][:], dtype=np.float32
                        ).reshape(-1)
                        clean = np.all(labels == 1, axis=1)
                        if task == "regression":
                            selected = np.flatnonzero(clean).astype(
                                np.int64, copy=False
                            )
                    metadata = {
                        name: np.asarray(handle[fields[name]][:])
                        .reshape(-1)
                        .astype(np.int64, copy=False)
                        for name in ("id", "run_number", "detector", "tp0")
                        if name in fields
                    }
            except OSError as error:
                raise OSError(
                    f"cannot open {path}; the download may be incomplete: {error}"
                ) from error
            self._files.append(
                _FileInfo(
                    path,
                    length,
                    fields,
                    selected,
                    labels,
                    energies,
                    clean,
                    metadata,
                )
            )

        self._offsets = np.cumsum(
            [0, *(item.selected_length for item in self._files)], dtype=np.int64
        )
        if int(self._offsets[-1]) == 0:
            raise ValueError(f"no eligible events found for task {task!r}")

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

    def _output(
        self,
        info: _FileInfo,
        row: int,
        waveform: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {
            "inputs": torch.from_numpy(waveform.copy()).unsqueeze(0),
            "row": torch.tensor(row, dtype=torch.int64),
        }
        for name, values in info.metadata.items():
            output[name] = torch.tensor(int(values[row]), dtype=torch.int64)

        if self.task != "inference":
            if info.labels is None or info.energies is None or info.clean is None:
                raise RuntimeError("labeled MJD cache is unavailable")
            output["labels"] = torch.from_numpy(info.labels[row].copy())
            output["clean"] = torch.tensor(bool(info.clean[row]))
            output["energy"] = torch.tensor(
                float(info.energies[row]), dtype=torch.float32
            )
        return output

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, torch.Tensor]]:
        normalized = np.asarray(
            [self._normalize_index(int(index)) for index in indices], dtype=np.int64
        )
        if normalized.size == 0:
            return []
        file_indices = np.searchsorted(
            self._offsets, normalized, side="right"
        ) - 1
        outputs: list[dict[str, torch.Tensor] | None] = [None] * normalized.size

        for file_index in np.unique(file_indices):
            positions = np.flatnonzero(file_indices == file_index)
            info = self._files[int(file_index)]
            local = normalized[positions] - self._offsets[int(file_index)]
            rows = (
                local
                if info.selected_rows is None
                else info.selected_rows[local]
            ).astype(np.int64, copy=False)
            unique_rows, inverse = np.unique(rows, return_inverse=True)
            handle = self._handle(info.path)
            waveforms = np.asarray(
                handle[info.fields["waveform"]][unique_rows], dtype=np.float32
            ).reshape(unique_rows.size, -1)[inverse]

            baseline_count = min(self.baseline_samples, waveforms.shape[1])
            baseline = np.mean(
                waveforms[:, :baseline_count], axis=1, dtype=np.float64
            ).astype(np.float32, copy=False)
            waveforms = np.asarray(
                waveforms - baseline[:, np.newaxis], dtype=np.float32
            )
            if (
                self.task == "classification"
                and self.classification_amplitude_normalization
            ):
                scale = np.max(np.abs(waveforms), axis=1, keepdims=True)
                np.divide(waveforms, scale, out=waveforms, where=scale > 0.0)
            elif self.task == "regression":
                waveforms /= self.regression_waveform_scale

            for position, row, waveform in zip(
                positions.tolist(), rows.tolist(), waveforms
            ):
                outputs[position] = self._output(info, int(row), waveform)

        if any(output is None for output in outputs):
            raise RuntimeError("failed to load an MJD batch")
        return [output for output in outputs if output is not None]


class _IndexedDataset(Dataset):
    def __init__(self, source: MJDWaveformDataset, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.source[int(self.indices[index])]

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, torch.Tensor]]:
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


def _subset_indices(
    length: int, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    order = generator.permutation(length)
    validation_size = max(1, int(round(length * validation_fraction)))
    if validation_size >= length:
        raise ValueError("training data must contain at least two eligible events")
    return order[validation_size:], order[:validation_size]


def prepare_dataset(
    *,
    task: Literal["classification", "regression"],
    data_config: DataConfig | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    limit_per_split: int | None = None,
) -> MJDDataLoaders:
    """Prepare train/validation loaders and the untouched official test loader."""

    config = data_config or DataConfig()
    train_paths = discover_files(config.data_root, "train")
    test_paths = discover_files(config.data_root, "test")
    if not train_paths:
        raise FileNotFoundError(
            f"no MJD_Train_*.hdf5 files found in {config.data_root}"
        )
    if not test_paths:
        raise FileNotFoundError(
            f"no MJD_Test_*.hdf5 files found in {config.data_root}"
        )
    options = {
        "task": task,
        "baseline_samples": config.baseline_samples,
        "classification_amplitude_normalization": (
            config.classification_amplitude_normalization
        ),
        "regression_waveform_scale": config.regression_waveform_scale,
    }
    train_source = MJDWaveformDataset(train_paths, **options)
    test_source = MJDWaveformDataset(test_paths, **options)
    train_indices, validation_indices = _subset_indices(
        len(train_source), config.validation_fraction, config.seed
    )
    if limit_per_split is not None:
        limit = int(limit_per_split)
        train_indices = train_indices[:limit]
        validation_indices = validation_indices[:limit]
        test_indices = np.arange(min(limit, len(test_source)), dtype=np.int64)
    else:
        test_indices = np.arange(len(test_source), dtype=np.int64)

    train_data = _IndexedDataset(train_source, train_indices)
    validation_data = _IndexedDataset(train_source, validation_indices)
    test_data = _IndexedDataset(test_source, test_indices)
    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    return MJDDataLoaders(
        train_loader=_PrefetchLoader(
            DataLoader(train_data, shuffle=True, **loader_options)
        ),
        validation_loader=_PrefetchLoader(
            DataLoader(validation_data, shuffle=False, **loader_options)
        ),
        test_loader=_PrefetchLoader(
            DataLoader(test_data, shuffle=False, **loader_options)
        ),
        counts={
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        task=task,
    )


__all__ = [
    "FIELD_ALIASES",
    "MJDDataLoaders",
    "MJDWaveformDataset",
    "PSD_NAMES",
    "discover_files",
    "inspect_data_root",
    "prepare_dataset",
]
