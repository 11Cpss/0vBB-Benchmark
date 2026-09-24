"""Disk-backed token caches for NEXT Transformer experiments.

The cache is a performance layer around the shared Simple EnergyBench data
contract.  Raw HDF5 parsing and :class:`NEXTTokenBuilder` remain the source of
truth; this module stores their deterministic outputs in uncompressed NumPy
memory maps and reconstructs the same event dictionaries during training.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, default_collate

from next_training import ProjectionConfig
from next_training.data import FileSlice, NextEventDataset, SPLIT_NAMES

from .tokenization import NEXTTokenBuilder, TokenizationConfig


CACHE_SCHEMA_VERSION = 1
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "coords": np.dtype(np.float32),
    "features": np.dtype(np.float32),
    "mask": np.dtype(np.bool_),
    "energy": np.dtype(np.float64),
    "coverage": np.dtype(np.float32),
    "source_event_id": np.dtype(np.int64),
    "label": np.dtype(np.int8),
}


@dataclass(frozen=True)
class CachedPreparedData:
    """Cached DataLoaders and their shared split/cache provenance."""

    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    counts: dict[str, Any]
    manifest_path: Path
    cache_manifest_path: Path

    @property
    def val_loader(self) -> DataLoader:
        """Short alias matching EnergyBench's ``PreparedData`` object."""

        return self.validation_loader


@dataclass(frozen=True)
class _CachedBatchCollator:
    """Collate cached samples with optional execution-only optimizations.

    ``trim_padding`` removes only token columns that are padding for every
    event in the batch. ``compact_metadata`` omits fields that the shared
    classification training loop never reads. Both operations happen inside
    DataLoader workers before the batch is transferred to the main process.
    """

    trim_padding: bool = False
    compact_metadata: bool = False

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty cached batch")

        selected_samples: Sequence[Mapping[str, Any]] = samples
        if self.compact_metadata:
            selected_samples = [
                {
                    "inputs": sample["inputs"],
                    "label": sample["label"],
                    "sample_weight": sample["sample_weight"],
                }
                for sample in samples
            ]

        batch = default_collate(selected_samples)
        if not self.trim_padding:
            return batch

        inputs = batch.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TypeError("cached batch inputs must be a mapping")
        mask = inputs.get("mask")
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
            raise ValueError("cached batch mask must have shape [B, N]")
        if mask.dtype != torch.bool:
            raise TypeError("cached batch mask must have Boolean dtype")

        # Find the final column used by any event. This remains correct even
        # if a malformed mask contains a gap, so no valid token can be cut.
        used_columns = mask.any(dim=0)
        used_indices = torch.nonzero(used_columns, as_tuple=False)
        if used_indices.numel() == 0:
            raise ValueError("every cached batch must contain a valid token")
        token_stop = int(used_indices[-1, 0].item()) + 1

        mutable_inputs = dict(inputs)
        for name in ("coords", "features", "mask"):
            value = mutable_inputs.get(name)
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                raise ValueError(
                    f"cached batch input {name!r} must include batch and token axes"
                )
            if value.shape[1] != mask.shape[1]:
                raise ValueError("cached batch inputs must share the token axis")
            # Materialize compact storage before multiprocessing transfer.
            mutable_inputs[name] = value[:, :token_stop].contiguous()
        batch["inputs"] = mutable_inputs
        return batch


@dataclass(frozen=True)
class _CachedSlice:
    """One EnergyBench FileSlice and its rows in the cache arrays."""

    relative_path: str
    category: str
    label: int
    split: str
    event_start: int
    event_stop: int
    cache_start: int
    cache_stop: int

    @property
    def event_count(self) -> int:
        return self.cache_stop - self.cache_start

    def file_slice(self) -> FileSlice:
        return FileSlice(
            relative_path=self.relative_path,
            category=self.category,
            label=self.label,
            split=self.split,
            event_start=self.event_start,
            event_stop=self.event_stop,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "category": self.category,
            "label": self.label,
            "split": self.split,
            "event_start": self.event_start,
            "event_stop": self.event_stop,
            "event_count": self.event_count,
            "cache_start": self.cache_start,
            "cache_stop": self.cache_stop,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "_CachedSlice":
        result = cls(
            relative_path=str(value["relative_path"]),
            category=str(value["category"]),
            label=int(value["label"]),
            split=str(value["split"]),
            event_start=int(value["event_start"]),
            event_stop=int(value["event_stop"]),
            cache_start=int(value["cache_start"]),
            cache_stop=int(value["cache_stop"]),
        )
        if result.split not in SPLIT_NAMES:
            raise ValueError(f"unknown cached split: {result.split!r}")
        if result.label not in (0, 1):
            raise ValueError("cached labels must be binary")
        if result.event_stop - result.event_start != result.event_count:
            raise ValueError("source and cache slice lengths differ")
        if result.event_count <= 0:
            raise ValueError("cached slices must be non-empty")
        return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tokenization_source_path() -> Path:
    return Path(__file__).resolve().with_name("tokenization.py")


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")
    return value


def _load_source_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"split manifest does not exist: {path}")
    manifest = _read_json(path)
    if "splits" not in manifest or "counts" not in manifest:
        raise ValueError("split manifest must contain 'splits' and 'counts'")
    for split in SPLIT_NAMES:
        rows = manifest["splits"].get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"split manifest has no {split!r} slices")
        allocated = sum(FileSlice.from_dict(row).event_count for row in rows)
        expected = int(manifest["counts"].get(split, -1))
        if allocated != expected:
            raise ValueError(
                f"manifest {split} count is {expected:,}, but slices allocate "
                f"{allocated:,} events"
            )
    expected_total = int(manifest["counts"].get("total", -1))
    actual_total = sum(int(manifest["counts"][name]) for name in SPLIT_NAMES)
    if actual_total != expected_total:
        raise ValueError("split manifest total does not equal its split totals")
    return manifest


def _validate_source_files(
    data_root: Path, source_manifest: Mapping[str, Any]
) -> None:
    """Ensure selected source files still match the split inventory."""

    files = source_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("split manifest contains no selected source files")
    for value in files:
        if not isinstance(value, Mapping):
            raise ValueError("split manifest file records must be mappings")
        path = data_root / str(value["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifest source file is missing: {path}")
        stat = path.stat()
        expected_size = int(value["size"])
        expected_mtime = int(value["mtime_ns"])
        if stat.st_size != expected_size or stat.st_mtime_ns != expected_mtime:
            raise ValueError(
                "manifest source file changed after split creation: "
                f"{path}"
            )


def _build_layout(
    source_manifest: Mapping[str, Any],
) -> dict[str, list[_CachedSlice]]:
    layout: dict[str, list[_CachedSlice]] = {}
    for split in SPLIT_NAMES:
        cursor = 0
        slices: list[_CachedSlice] = []
        for value in source_manifest["splits"][split]:
            source = FileSlice.from_dict(value)
            cached = _CachedSlice(
                relative_path=source.relative_path,
                category=source.category,
                label=source.label,
                split=source.split,
                event_start=source.event_start,
                event_stop=source.event_stop,
                cache_start=cursor,
                cache_stop=cursor + source.event_count,
            )
            slices.append(cached)
            cursor = cached.cache_stop
        expected = int(source_manifest["counts"][split])
        if cursor != expected:
            raise ValueError(
                f"cache layout allocated {cursor:,} {split} events; "
                f"expected {expected:,}"
            )
        layout[split] = slices
    return layout


def _array_shapes(event_count: int, max_tokens: int) -> dict[str, tuple[int, ...]]:
    return {
        "coords": (event_count, max_tokens, 3),
        "features": (event_count, max_tokens, 2),
        "mask": (event_count, max_tokens),
        "energy": (event_count,),
        "coverage": (event_count,),
        "source_event_id": (event_count,),
        "label": (event_count,),
    }


def _array_specs(
    counts: Mapping[str, Any], max_tokens: int
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLIT_NAMES:
        shapes = _array_shapes(int(counts[split]), max_tokens)
        result[split] = {
            name: {
                "file": f"{split}/{name}.npy",
                "shape": list(shapes[name]),
                "dtype": ARRAY_DTYPES[name].name,
            }
            for name in ARRAY_DTYPES
        }
    return result


def _allocate_arrays(
    root: Path, specs: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> None:
    for split in SPLIT_NAMES:
        (root / split).mkdir(parents=True, exist_ok=True)
        for specification in specs[split].values():
            path = root / str(specification["file"])
            array = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.dtype(str(specification["dtype"])),
                shape=tuple(int(value) for value in specification["shape"]),
            )
            array.flush()
            del array


def _validate_array_files(
    root: Path, specs: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> None:
    for split in SPLIT_NAMES:
        for name, specification in specs[split].items():
            path = root / str(specification["file"])
            if not path.is_file():
                raise FileNotFoundError(f"cache array is missing: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = tuple(int(value) for value in specification["shape"])
            expected_dtype = np.dtype(str(specification["dtype"]))
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"cache array {name!r} has shape {array.shape}; "
                    f"expected {expected_shape}"
                )
            if array.dtype != expected_dtype:
                raise ValueError(
                    f"cache array {name!r} has dtype {array.dtype}; "
                    f"expected {expected_dtype}"
                )
            del array


def _cache_identity(
    split_manifest_path: Path, tokenization_config: TokenizationConfig
) -> dict[str, Any]:
    config = tokenization_config.to_dict()
    return {
        "source_manifest_sha256": _sha256_file(split_manifest_path),
        "tokenization_config": config,
        "tokenization_config_sha256": _sha256_json(config),
        "tokenization_source_sha256": _sha256_file(_tokenization_source_path()),
    }


def _cache_name(identity: Mapping[str, Any]) -> str:
    tokenization = str(identity["tokenization_config"]["tokenization"])
    split_hash = str(identity["source_manifest_sha256"])[:12]
    config_hash = str(identity["tokenization_config_sha256"])[:12]
    return f"{tokenization}_{split_hash}_{config_hash}"


def _expected_build_spec(
    data_root: Path,
    split_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    tokenization_config: TokenizationConfig,
    identity: Mapping[str, Any],
    layout: Mapping[str, Sequence[_CachedSlice]],
) -> dict[str, Any]:
    counts = dict(source_manifest["counts"])
    counts["boundary_files"] = list(source_manifest.get("boundary_files", []))
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "status": "building",
        "data_root": str(data_root),
        "source_manifest_path": str(split_manifest_path),
        **dict(identity),
        "counts": counts,
        "arrays": _array_specs(counts, tokenization_config.max_tokens),
        "slices": {
            split: [item.to_dict() for item in layout[split]]
            for split in SPLIT_NAMES
        },
    }


def _spec_signature(spec: Mapping[str, Any]) -> str:
    relevant = {
        key: spec[key]
        for key in (
            "cache_schema_version",
            "data_root",
            "source_manifest_path",
            "source_manifest_sha256",
            "tokenization_config",
            "tokenization_config_sha256",
            "tokenization_source_sha256",
            "counts",
            "arrays",
            "slices",
        )
    }
    return _sha256_json(relevant)


_WORKER_DATA_ROOT: Path | None = None
_WORKER_CONFIG: TokenizationConfig | None = None
_WORKER_ARRAYS: dict[str, dict[str, np.ndarray]] = {}


def _worker_initialize(
    data_root: str,
    config_payload: Mapping[str, Any],
    building_root: str,
    array_specs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    global _WORKER_DATA_ROOT, _WORKER_CONFIG, _WORKER_ARRAYS
    _WORKER_DATA_ROOT = Path(data_root)
    _WORKER_CONFIG = TokenizationConfig(**dict(config_payload))
    root = Path(building_root)
    _WORKER_ARRAYS = {
        split: {
            name: np.load(
                root / str(specification["file"]),
                mmap_mode="r+",
                allow_pickle=False,
            )
            for name, specification in array_specs[split].items()
        }
        for split in SPLIT_NAMES
    }


def _validate_worker_sample(
    sample: Mapping[str, Any], cached_slice: _CachedSlice, max_tokens: int
) -> tuple[Mapping[str, np.ndarray], int]:
    inputs = sample.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("token builder must produce a mapping of input arrays")
    coords = np.asarray(inputs.get("coords"))
    features = np.asarray(inputs.get("features"))
    mask = np.asarray(inputs.get("mask"))
    if coords.shape != (max_tokens, 3) or coords.dtype != np.float32:
        raise ValueError("token coordinates have the wrong shape or dtype")
    if features.shape != (max_tokens, 2) or features.dtype != np.float32:
        raise ValueError("token features have the wrong shape or dtype")
    if mask.shape != (max_tokens,) or mask.dtype != np.bool_:
        raise ValueError("token mask has the wrong shape or dtype")
    if not np.isfinite(coords).all() or not np.isfinite(features).all():
        raise ValueError("token inputs contain non-finite values")
    if not bool(mask.any()):
        raise ValueError("every cached event must contain a real token")
    label = int(sample["label"])
    if label != cached_slice.label:
        raise ValueError("sample label disagrees with its manifest slice")
    energy = float(sample["energy"])
    coverage = float(sample["projection_coverage"])
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("sample energy must be finite and positive")
    if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("sample coverage must be finite and between zero and one")
    expected_prefix = f"NEXT::{cached_slice.relative_path}::"
    event_id = str(sample["event_id"])
    if not event_id.startswith(expected_prefix):
        raise ValueError("sample event ID does not match its manifest source")
    try:
        source_event_id = int(event_id.rsplit("::", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"cannot parse source event ID from {event_id!r}") from error
    return {"coords": coords, "features": features, "mask": mask}, source_event_id


def _worker_build_slice(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_DATA_ROOT is None or _WORKER_CONFIG is None:
        raise RuntimeError("cache worker was not initialized")
    cached_slice = _CachedSlice.from_dict(task["slice"])
    builder = NEXTTokenBuilder(_WORKER_CONFIG)
    dataset = NextEventDataset(
        _WORKER_DATA_ROOT,
        [cached_slice.file_slice()],
        projection=ProjectionConfig(),
        mode="classification",
        seed=_WORKER_CONFIG.seed,
        shuffle_slices=False,
        shuffle_buffer_size=0,
        chunk_rows=262_144,
        input_builder=builder,
    )
    arrays = _WORKER_ARRAYS[cached_slice.split]
    cursor = cached_slice.cache_start
    label_counts = {"0": 0, "1": 0}
    for sample in dataset:
        if cursor >= cached_slice.cache_stop:
            raise ValueError("source slice yielded more events than expected")
        inputs, source_event_id = _validate_worker_sample(
            sample, cached_slice, _WORKER_CONFIG.max_tokens
        )
        arrays["coords"][cursor] = inputs["coords"]
        arrays["features"][cursor] = inputs["features"]
        arrays["mask"][cursor] = inputs["mask"]
        arrays["energy"][cursor] = np.float64(sample["energy"])
        arrays["coverage"][cursor] = np.float32(sample["projection_coverage"])
        arrays["source_event_id"][cursor] = np.int64(source_event_id)
        arrays["label"][cursor] = np.int8(cached_slice.label)
        label_counts[str(cached_slice.label)] += 1
        cursor += 1
    if cursor != cached_slice.cache_stop:
        raise ValueError(
            f"source slice yielded {cursor - cached_slice.cache_start:,} events; "
            f"expected {cached_slice.event_count:,}"
        )
    return {
        "key": str(task["key"]),
        "events": cached_slice.event_count,
        "labels": label_counts,
    }


def _run_build_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    tokenization_config: TokenizationConfig,
    building_root: Path,
    array_specs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    num_workers: int,
) -> Iterator[dict[str, Any]]:
    initializer_arguments = (
        str(data_root),
        tokenization_config.to_dict(),
        str(building_root),
        array_specs,
    )
    if num_workers == 0:
        _worker_initialize(*initializer_arguments)
        for task in tasks:
            yield _worker_build_slice(task)
        return
    context = mp.get_context("spawn")
    with context.Pool(
        processes=num_workers,
        initializer=_worker_initialize,
        initargs=initializer_arguments,
    ) as pool:
        yield from pool.imap_unordered(_worker_build_slice, tasks, chunksize=1)


def _flush_arrays(
    root: Path, specs: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> None:
    for split in SPLIT_NAMES:
        for specification in specs[split].values():
            array = np.load(
                root / str(specification["file"]),
                mmap_mode="r+",
                allow_pickle=False,
            )
            array.flush()
            del array


def build_token_cache(
    data_root: str | os.PathLike[str],
    split_manifest_path: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    tokenization_config: TokenizationConfig,
    *,
    num_workers: int = 8,
    resume: bool = True,
    overwrite: bool = False,
) -> Path:
    """Build or resume one deterministic token cache.

    Worker processes write disjoint memmap ranges, so no lock or centralized
    writer is required.  A cache becomes readable only after every manifest
    slice completes and the ``_SUCCESS`` marker is published.
    """

    if not isinstance(tokenization_config, TokenizationConfig):
        raise TypeError("tokenization_config must be a TokenizationConfig")
    if isinstance(num_workers, bool) or int(num_workers) != num_workers:
        raise ValueError("num_workers must be a non-negative integer")
    num_workers = int(num_workers)
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")

    source_root = Path(data_root).expanduser().resolve()
    source_manifest_path = Path(split_manifest_path).expanduser().resolve()
    output_root = Path(cache_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"NEXT data root does not exist: {source_root}")
    source_manifest = _load_source_manifest(source_manifest_path)
    _validate_source_files(source_root, source_manifest)
    layout = _build_layout(source_manifest)
    identity = _cache_identity(source_manifest_path, tokenization_config)
    cache_name = _cache_name(identity)
    final_path = output_root / cache_name
    building_path = output_root / f"{cache_name}.building"
    output_root.mkdir(parents=True, exist_ok=True)

    if final_path.exists():
        if overwrite:
            shutil.rmtree(final_path)
        else:
            validate_token_cache(
                final_path,
                source_root,
                source_manifest_path,
                tokenization_config,
            )
            return final_path
    if building_path.exists() and overwrite:
        shutil.rmtree(building_path)
    if building_path.exists() and not resume:
        raise FileExistsError(
            f"incomplete cache exists at {building_path}; enable resume or overwrite"
        )

    expected_spec = _expected_build_spec(
        source_root,
        source_manifest_path,
        source_manifest,
        tokenization_config,
        identity,
        layout,
    )
    signature = _spec_signature(expected_spec)
    build_spec_path = building_path / "build_spec.json"
    state_path = building_path / "build_state.json"

    if not building_path.exists():
        building_path.mkdir(parents=True)
        _allocate_arrays(building_path, expected_spec["arrays"])
        _atomic_write_json(build_spec_path, expected_spec)
        state = {
            "spec_signature": signature,
            "completed": [],
            "completed_events": 0,
        }
        _atomic_write_json(state_path, state)
    else:
        if not build_spec_path.is_file() or not state_path.is_file():
            raise ValueError("incomplete cache is missing its build metadata")
        existing_spec = _read_json(build_spec_path)
        if _spec_signature(existing_spec) != signature:
            raise ValueError(
                "incomplete cache does not match the requested split or tokenizer"
            )
        _validate_array_files(building_path, expected_spec["arrays"])
        state = _read_json(state_path)
        if state.get("spec_signature") != signature:
            raise ValueError("cache build state has the wrong specification hash")

    completed = {str(value) for value in state.get("completed", [])}
    slice_by_key = {
        f"{split}:{index}": cached_slice
        for split in SPLIT_NAMES
        for index, cached_slice in enumerate(layout[split])
    }
    unknown_keys = completed - set(slice_by_key)
    if unknown_keys:
        raise ValueError(
            "cache build state contains unknown slice keys: "
            + ", ".join(sorted(unknown_keys)[:5])
        )
    recomputed_events = sum(
        slice_by_key[key].event_count for key in completed
    )
    if int(state.get("completed_events", -1)) != recomputed_events:
        raise ValueError("cache build state's completed-event count is inconsistent")
    tasks: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        for index, cached_slice in enumerate(layout[split]):
            key = f"{split}:{index}"
            if key not in completed:
                tasks.append({"key": key, "slice": cached_slice.to_dict()})
    total_slices = sum(len(layout[split]) for split in SPLIT_NAMES)
    completed_events = recomputed_events
    print(
        f"Building {tokenization_config.tokenization!r} cache: "
        f"{len(tasks):,} pending / {total_slices:,} total slices"
    )
    results_since_save = 0
    for result in _run_build_tasks(
        tasks,
        data_root=source_root,
        tokenization_config=tokenization_config,
        building_root=building_path,
        array_specs=expected_spec["arrays"],
        num_workers=num_workers,
    ):
        completed.add(str(result["key"]))
        completed_events += int(result["events"])
        results_since_save += 1
        if results_since_save >= 100 or len(completed) == total_slices:
            state = {
                "spec_signature": signature,
                "completed": sorted(completed),
                "completed_events": completed_events,
            }
            _atomic_write_json(state_path, state)
            results_since_save = 0
            print(
                f"  cached {len(completed):,}/{total_slices:,} slices "
                f"({completed_events:,} events)"
            )

    if len(completed) != total_slices:
        raise RuntimeError("cache build ended before every slice completed")
    expected_events = int(source_manifest["counts"]["total"])
    if completed_events != expected_events:
        raise RuntimeError(
            f"cache build recorded {completed_events:,} events; "
            f"expected {expected_events:,}"
        )
    _flush_arrays(building_path, expected_spec["arrays"])
    _validate_array_files(building_path, expected_spec["arrays"])

    completed_manifest = {
        **expected_spec,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_slices": total_slices,
    }
    _atomic_write_json(building_path / "cache_manifest.json", completed_manifest)
    (building_path / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    building_path.replace(final_path)
    validate_token_cache(
        final_path,
        source_root,
        source_manifest_path,
        tokenization_config,
    )
    print(f"Completed cache: {final_path}")
    return final_path


def _load_complete_manifest(cache_dir: Path) -> dict[str, Any]:
    if cache_dir.name.endswith(".building"):
        raise ValueError("incomplete .building directories cannot be loaded")
    if not (cache_dir / "_SUCCESS").is_file():
        raise ValueError(f"cache is missing its _SUCCESS marker: {cache_dir}")
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if int(manifest.get("cache_schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported cache schema version")
    if manifest.get("status") != "complete":
        raise ValueError("cache manifest is not marked complete")
    if manifest.get("tokenization_source_sha256") != _sha256_file(
        _tokenization_source_path()
    ):
        raise ValueError("tokenization.py changed after this cache was built")
    config = manifest.get("tokenization_config")
    if not isinstance(config, dict) or manifest.get(
        "tokenization_config_sha256"
    ) != _sha256_json(config):
        raise ValueError("cache tokenization configuration hash is invalid")
    source_manifest = Path(str(manifest.get("source_manifest_path", "")))
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"cache source split manifest is unavailable: {source_manifest}"
        )
    if manifest.get("source_manifest_sha256") != _sha256_file(source_manifest):
        raise ValueError("source split manifest changed after this cache was built")
    _validate_array_files(cache_dir, manifest["arrays"])
    return manifest


def validate_token_cache(
    cache_dir: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
    split_manifest_path: str | os.PathLike[str],
    tokenization_config: TokenizationConfig,
) -> dict[str, Any]:
    """Validate cache provenance, structure, and requested representation."""

    root = Path(cache_dir).expanduser().resolve()
    source_root = Path(data_root).expanduser().resolve()
    source_manifest = Path(split_manifest_path).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"NEXT data root does not exist: {source_root}")
    manifest = _load_complete_manifest(root)
    identity = _cache_identity(source_manifest, tokenization_config)
    if Path(str(manifest["data_root"])).resolve() != source_root:
        raise ValueError("cache was built from a different NEXT data root")
    if Path(str(manifest["source_manifest_path"])).resolve() != source_manifest:
        raise ValueError("cache was built from a different split manifest")
    for field in (
        "source_manifest_sha256",
        "tokenization_config_sha256",
        "tokenization_source_sha256",
    ):
        if manifest.get(field) != identity[field]:
            raise ValueError(f"cache provenance mismatch: {field}")
    config = TokenizationConfig(**dict(manifest["tokenization_config"]))
    if config != tokenization_config:
        raise ValueError("cache tokenization settings do not match the request")
    layout = {
        split: [_CachedSlice.from_dict(value) for value in manifest["slices"][split]]
        for split in SPLIT_NAMES
    }
    for split in SPLIT_NAMES:
        expected_cursor = 0
        for item in layout[split]:
            if item.cache_start != expected_cursor:
                raise ValueError(f"{split} cache ranges are not contiguous")
            expected_cursor = item.cache_stop
        if expected_cursor != int(manifest["counts"][split]):
            raise ValueError(f"{split} cache ranges do not cover every event")
    return {
        "cache_dir": str(root),
        "cache_manifest_path": str(root / "cache_manifest.json"),
        "tokenization": config.tokenization,
        "counts": dict(manifest["counts"]),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "tokenization_config_sha256": manifest["tokenization_config_sha256"],
        "tokenization_source_sha256": manifest["tokenization_source_sha256"],
    }


def find_token_cache(
    cache_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
    split_manifest_path: str | os.PathLike[str],
    tokenization_config: TokenizationConfig,
) -> Path:
    """Find the one complete cache matching a split and tokenizer config."""

    root = Path(cache_root).expanduser().resolve()
    source_manifest = Path(split_manifest_path).expanduser().resolve()
    identity = _cache_identity(source_manifest, tokenization_config)
    expected = root / _cache_name(identity)
    validate_token_cache(
        expected,
        data_root,
        source_manifest,
        tokenization_config,
    )
    return expected


class CachedNEXTDataset(IterableDataset):
    """Reproduce EnergyBench FileSlice iteration from cached token arrays."""

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        slices: Sequence[_CachedSlice],
        *,
        seed: int,
        shuffle_slices: bool,
        array_specs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        super().__init__()
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {split!r}")
        if not slices:
            raise ValueError("cached datasets require at least one slice")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.slices = list(slices)
        self.seed = int(seed)
        self.shuffle_slices = bool(shuffle_slices)
        self.array_specs = {name: dict(value) for name, value in array_specs.items()}
        self.epoch = 0
        self._length = sum(item.event_count for item in self.slices)

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or int(epoch) != epoch or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = int(epoch)

    def _open_arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.load(
                self.cache_dir / str(specification["file"]),
                mmap_mode="c",
                allow_pickle=False,
            )
            for name, specification in self.array_specs.items()
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        slices = list(self.slices)
        if self.shuffle_slices:
            generator = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch])
            )
            generator.shuffle(slices)
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            slices = slices[worker.id :: worker.num_workers]
        arrays = self._open_arrays()
        try:
            for item in slices:
                for row in range(item.cache_start, item.cache_stop):
                    label = int(arrays["label"][row])
                    if label != item.label:
                        raise ValueError("cached label disagrees with slice metadata")
                    source_event_id = int(arrays["source_event_id"][row])
                    yield {
                        "inputs": {
                            "coords": np.asarray(arrays["coords"][row]),
                            "features": np.asarray(arrays["features"][row]),
                            "mask": np.asarray(arrays["mask"][row]),
                        },
                        "label": np.float32(label),
                        "energy": np.float64(arrays["energy"][row]),
                        "event_id": (
                            f"NEXT::{item.relative_path}::{source_event_id}"
                        ),
                        "category": item.category,
                        "group_id": item.relative_path,
                        "split": item.split,
                        "sample_weight": np.float32(1.0),
                        "projection_coverage": np.float32(
                            arrays["coverage"][row]
                        ),
                    }
        finally:
            arrays.clear()


def prepare_cached_dataset(
    cache_dir: str | os.PathLike[str],
    *,
    batch_size: int = 64,
    num_workers: int = 8,
    seed: int = 42,
    pin_memory: bool = True,
    trim_padding: bool = False,
    compact_training_batches: bool = False,
) -> CachedPreparedData:
    """Create cached train/validation/test DataLoaders for EnergyBench.

    ``trim_padding`` removes token columns that are padding for every event in
    a batch without changing the cache or its maximum token ceiling.
    ``compact_training_batches`` omits evaluation-only metadata from train and
    validation batches; the test loader always retains complete metadata.
    """

    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or int(num_workers) != num_workers or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(trim_padding, bool):
        raise TypeError("trim_padding must be Boolean")
    if not isinstance(compact_training_batches, bool):
        raise TypeError("compact_training_batches must be Boolean")
    root = Path(cache_dir).expanduser().resolve()
    manifest = _load_complete_manifest(root)
    source_seed = int(manifest.get("source_settings", {}).get("seed", seed))
    source_manifest = _read_json(Path(str(manifest["source_manifest_path"])))
    source_seed = int(source_manifest.get("settings", {}).get("seed", source_seed))
    if int(seed) != source_seed:
        raise ValueError(
            f"cached loader seed {seed} differs from split seed {source_seed}"
        )
    datasets: dict[str, CachedNEXTDataset] = {}
    for split in SPLIT_NAMES:
        slices = [_CachedSlice.from_dict(value) for value in manifest["slices"][split]]
        datasets[split] = CachedNEXTDataset(
            root,
            split,
            slices,
            seed=int(seed),
            shuffle_slices=split == "train",
            array_specs=manifest["arrays"][split],
        )
    shared_loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": False,
        "drop_last": False,
    }
    loaders: dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:
        compact_metadata = compact_training_batches and split != "test"
        collate_fn = None
        if trim_padding or compact_metadata:
            collate_fn = _CachedBatchCollator(
                trim_padding=trim_padding,
                compact_metadata=compact_metadata,
            )
        loaders[split] = DataLoader(
            datasets[split],
            collate_fn=collate_fn,
            **shared_loader_options,
        )
    return CachedPreparedData(
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        test_loader=loaders["test"],
        counts=dict(manifest["counts"]),
        manifest_path=Path(str(manifest["source_manifest_path"])),
        cache_manifest_path=root / "cache_manifest.json",
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CachedNEXTDataset",
    "CachedPreparedData",
    "build_token_cache",
    "find_token_cache",
    "prepare_cached_dataset",
    "validate_token_cache",
]
