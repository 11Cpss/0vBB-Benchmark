"""Adapter loading for arbitrary model frameworks.

An adapter is ordinary trusted Python code with the signature::

    predict(model_path: str, data_path: str, **kwargs) -> Mapping[str, array]

It may also return an iterable of mappings (one per inference batch).  The
statistical evaluator never imports the model framework itself.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from .data import PredictionBundle


def parse_adapter_arguments(items: Sequence[str]) -> Dict[str, Any]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError("adapter argument must be KEY=VALUE, received %r" % item)
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("adapter argument key cannot be empty")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parsed[key] = value
    return parsed


def _module_from_path(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError("adapter file does not exist: %s" % path)
    module_name = "_energybench_adapter_%s" % abs(hash(str(path.resolve())))
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("could not load adapter module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adapter(specification: str) -> Callable[..., Any]:
    if ":" not in specification:
        raise ValueError(
            "adapter must be MODULE:FUNCTION or /path/to/file.py:FUNCTION"
        )
    module_part, function_name = specification.rsplit(":", 1)
    if module_part.endswith(".py") or "/" in module_part:
        module = _module_from_path(Path(module_part).expanduser().resolve())
    else:
        module = importlib.import_module(module_part)
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise AttributeError(
            "adapter %r has no callable %r" % (module_part, function_name)
        )
    return function


def _mapping_to_columns(batch: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    columns = {}
    for name, values in batch.items():
        if str(name).startswith("__"):
            continue
        array = np.asarray(values)
        if array.ndim == 0:
            raise ValueError("adapter column %r is scalar" % name)
        columns[str(name)] = array
    lengths = {name: int(values.shape[0]) for name, values in columns.items()}
    if lengths and len(set(lengths.values())) != 1:
        raise ValueError(
            "adapter batch columns have different event counts: %s" % lengths
        )
    return columns


def normalize_adapter_output(result: Any) -> PredictionBundle:
    if isinstance(result, PredictionBundle):
        return result
    if isinstance(result, Mapping):
        metadata = dict(result.get("__metadata__", {}))
        return PredictionBundle(_mapping_to_columns(result), metadata=metadata)

    try:
        batches = list(result)
    except TypeError as exc:
        raise TypeError(
            "adapter output must be a mapping, PredictionBundle, or iterable "
            "of mappings"
        ) from exc
    if not batches:
        raise ValueError("adapter returned no inference batches")
    if not all(isinstance(batch, Mapping) for batch in batches):
        raise TypeError("every adapter batch must be a mapping")

    first_columns = set(_mapping_to_columns(batches[0]))
    if not first_columns:
        raise ValueError("adapter batch has no columns")
    pieces = {name: [] for name in first_columns}
    metadata = {}
    for batch_index, batch in enumerate(batches):
        columns = _mapping_to_columns(batch)
        if set(columns) != first_columns:
            raise ValueError(
                "adapter batch %d columns differ: expected %s, received %s"
                % (
                    batch_index,
                    sorted(first_columns),
                    sorted(columns),
                )
            )
        for name, values in columns.items():
            pieces[name].append(values)
        if "__metadata__" in batch:
            metadata.update(dict(batch["__metadata__"]))
    return PredictionBundle(
        {name: np.concatenate(values, axis=0) for name, values in pieces.items()},
        metadata=metadata,
    )


def run_adapter(
    specification: str,
    model_path: str,
    data_path: str,
    arguments: Optional[Mapping[str, Any]] = None,
) -> PredictionBundle:
    function = load_adapter(specification)
    result = function(
        model_path=str(model_path),
        data_path=str(data_path),
        **dict(arguments or {}),
    )
    bundle = normalize_adapter_output(result)
    bundle.metadata.setdefault("adapter", specification)
    bundle.metadata.setdefault("model_path", str(model_path))
    bundle.metadata.setdefault("data_path", str(data_path))
    return bundle
