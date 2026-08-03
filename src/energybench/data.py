"""Canonical prediction-table IO and schema resolution.

EnergyBench deliberately consumes prediction records rather than raw detector
events.  This keeps the statistical evaluator independent of model framework
and detector representation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .utils import array_digest


CANONICAL_ALIASES = {
    "event_id": ("event_id", "eventId", "eventid", "ev_no", "id"),
    "label": ("label", "class_id", "y_true", "target", "is_signal"),
    "score": (
        "score",
        "score_positive",
        "y_score",
        "signal_score",
        "probability",
        "logit",
    ),
    "energy_condition": (
        "energy_condition",
        "match_energy",
        "nuisance_energy",
        "energy",
        "Evis",
    ),
    "energy_true": (
        "energy_true",
        "true_energy",
        "energy_target",
        "target_energy",
        "E_true",
        "energy_label",
        "energy",
    ),
    "energy_pred": (
        "energy_pred",
        "pred_energy",
        "predicted_energy",
        "E_pred",
        "energy_prediction",
    ),
    "category": ("category", "class_name", "sample", "process", "source_class"),
    "sample_weight": ("sample_weight", "event_weight", "weight"),
    "group_id": ("group_id", "run_id", "file_id"),
    "split": ("split", "data_split"),
    "experiment": ("experiment",),
    "dataset_id": ("dataset_id", "dataset"),
    "dataset_version": ("dataset_version",),
    "model_id": ("model_id", "model"),
    "task_id": ("task_id",),
}


@dataclass
class PredictionBundle:
    columns: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[Path] = None

    def __post_init__(self) -> None:
        normalized = {}
        lengths = {}
        for name, values in self.columns.items():
            array = np.asarray(values)
            if array.ndim == 0:
                raise ValueError("column %r is scalar; every column needs an event axis" % name)
            normalized[str(name)] = array
            lengths[str(name)] = int(array.shape[0])
        if lengths and len(set(lengths.values())) != 1:
            raise ValueError("columns have different event counts: %s" % lengths)
        self.columns = normalized
        if self.source is not None:
            self.source = Path(self.source)

    @property
    def n_events(self) -> int:
        if not self.columns:
            return 0
        return int(next(iter(self.columns.values())).shape[0])

    def require(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(
                "missing column %r; available columns: %s"
                % (name, ", ".join(sorted(self.columns)))
            )
        return self.columns[name]

    def subset(self, mask_or_indices: Any) -> "PredictionBundle":
        return PredictionBundle(
            {name: values[mask_or_indices] for name, values in self.columns.items()},
            metadata=dict(self.metadata),
            source=self.source,
        )


def _parse_npz_metadata(archive: Any) -> Dict[str, Any]:
    if "__metadata__" not in archive.files:
        return {}
    raw = archive["__metadata__"]
    if raw.ndim == 0:
        text = str(raw.item())
    elif raw.size == 1:
        text = str(raw.ravel()[0])
    else:
        raise ValueError("__metadata__ must be a scalar JSON string")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON in NPZ __metadata__: %s" % exc)
    if not isinstance(value, dict):
        raise ValueError("NPZ __metadata__ must decode to an object")
    return value


def _load_npz(path: Path) -> PredictionBundle:
    with np.load(str(path), allow_pickle=False) as archive:
        metadata = _parse_npz_metadata(archive)
        columns = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "__metadata__"
        }
    return PredictionBundle(columns, metadata=metadata, source=path)


def _infer_csv_column(values: Sequence[str]) -> np.ndarray:
    stripped = [value.strip() for value in values]
    nonempty = [value for value in stripped if value != ""]
    if not nonempty:
        return np.asarray(stripped, dtype=str)
    try:
        parsed_int = [int(value) if value != "" else 0 for value in stripped]
        if len(nonempty) == len(stripped):
            return np.asarray(parsed_int, dtype=np.int64)
    except ValueError:
        pass
    try:
        return np.asarray(
            [float(value) if value != "" else np.nan for value in stripped],
            dtype=float,
        )
    except ValueError:
        return np.asarray(stripped, dtype=str)


def _load_csv(path: Path) -> PredictionBundle:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header: %s" % path)
        raw = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                raw[name].append(row.get(name, ""))
    columns = {name: _infer_csv_column(values) for name, values in raw.items()}
    return PredictionBundle(columns, source=path)


def _load_hdf5(path: Path) -> PredictionBundle:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "HDF5 input requires h5py; activate the project environment, "
            "then run `python -m pip install -e .`"
        ) from exc
    columns = {}
    with h5py.File(str(path), "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
                columns[name] = np.asarray(obj)

        handle.visititems(visitor)
        metadata = {
            str(key): value.item() if isinstance(value, np.generic) else value
            for key, value in handle.attrs.items()
            if np.asarray(value).ndim == 0
        }
    if not columns:
        raise ValueError("no array datasets found in HDF5 file %s" % path)
    return PredictionBundle(columns, metadata=metadata, source=path)


def _load_parquet(path: Path) -> PredictionBundle:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Parquet input requires pandas and pyarrow; "
            "activate the project environment, then run "
            "`python -m pip install -e .`"
        ) from exc
    frame = pd.read_parquet(path)
    return PredictionBundle(
        {str(name): frame[name].to_numpy() for name in frame.columns},
        source=path,
    )


def load_bundle(path: Any) -> PredictionBundle:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("prediction file does not exist: %s" % source)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        return _load_npz(source)
    if suffix in {".csv", ".tsv"}:
        if suffix == ".tsv":
            raise ValueError("TSV is not yet supported; save it as CSV or NPZ")
        return _load_csv(source)
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return _load_hdf5(source)
    if suffix in {".parquet", ".pq"}:
        return _load_parquet(source)
    raise ValueError(
        "unsupported prediction format %r; use .npz, .csv, .h5/.hdf5, or .parquet"
        % suffix
    )


def save_bundle(bundle: PredictionBundle, path: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() != ".npz":
        raise ValueError("canonical bundle output must end in .npz")
    payload = dict(bundle.columns)
    payload["__metadata__"] = np.asarray(
        json.dumps(bundle.metadata, ensure_ascii=False, sort_keys=True)
    )
    np.savez_compressed(str(destination), **payload)
    return destination


def resolve_column(
    bundle: PredictionBundle,
    role: str,
    explicit: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    if explicit:
        if explicit not in bundle.columns:
            raise KeyError(
                "%s column %r not found; available: %s"
                % (role, explicit, ", ".join(sorted(bundle.columns)))
            )
        return explicit
    aliases = CANONICAL_ALIASES.get(role, (role,))
    found = [name for name in aliases if name in bundle.columns]
    if found:
        return found[0]
    if required:
        raise KeyError(
            "could not infer %s column; use --%s-column. Available: %s"
            % (role, role.replace("_", "-"), ", ".join(sorted(bundle.columns)))
        )
    return None


def resolve_schema(
    bundle: PredictionBundle, overrides: Optional[Mapping[str, Optional[str]]] = None
) -> Dict[str, Optional[str]]:
    overrides = dict(overrides or {})
    return {
        role: resolve_column(bundle, role, explicit=overrides.get(role), required=False)
        for role in CANONICAL_ALIASES
    }


def evaluation_fingerprint(
    columns: Mapping[str, Any], event_id_name: Optional[str] = None
) -> str:
    """Hash the evaluation sample while intentionally excluding predictions.

    ``columns`` should contain only identity/truth/weight arrays.  If event IDs
    are provided, records are sorted by their string representation so model
    outputs may arrive in a different row order and still compare correctly.
    """

    if not columns:
        return hashlib.sha256(b"empty").hexdigest()
    arrays = {str(name): np.asarray(value) for name, value in columns.items()}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("fingerprint columns must have equal length")
    order = None
    if event_id_name and event_id_name in arrays:
        order = np.argsort(arrays[event_id_name].astype(str), kind="mergesort")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = arrays[name] if order is None else arrays[name][order]
        array_digest(digest, name, values)
    return digest.hexdigest()


def duplicate_event_ids(values: Any) -> int:
    ids = np.asarray(values).astype(str)
    return int(len(ids) - len(np.unique(ids)))
