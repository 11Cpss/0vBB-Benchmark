"""EnergyBench adapter for the campaign's classical topology XGBoost model."""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import yaml

from next_cnn.data import dataset_inventory, discover_source_files

from .adapter import _Progress, _file_sha256, _split_contract
from .data import RepresentationConfig, build_inference_loader
from .models.classic_topology import (
    TOPOLOGY_FEATURE_NAMES,
    TopologyFeatureExtractor,
)


ARCHITECTURE_ID = "classic_001_topology_xgboost"


def _status(enabled: bool, message: str) -> None:
    if enabled:
        print("NEXT classical predict: %s" % message, file=sys.stderr, flush=True)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("could not read classical checkpoint %s: %s" % (path, exc))
    if not isinstance(checkpoint, Mapping):
        raise ValueError("classical checkpoint root must be a mapping")
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(
            "unsupported classical checkpoint architecture %r"
            % checkpoint.get("architecture_id")
        )
    if checkpoint.get("backend") != "xgboost":
        raise ValueError("classical checkpoint backend must be xgboost")
    if checkpoint.get("input_kind") != "topology":
        raise ValueError("classical checkpoint input_kind must be topology")
    if checkpoint.get("task") != "classification":
        raise ValueError("classical checkpoint task must be classification")
    if checkpoint.get("label_semantics") != {"0nubb": 1, "Bi214": 0}:
        raise ValueError("classical checkpoint label semantics are incompatible")
    return checkpoint


def _representation(checkpoint: Mapping[str, Any]) -> RepresentationConfig:
    training = checkpoint.get("training_config", {})
    if not isinstance(training, Mapping):
        raise ValueError("classical checkpoint training_config must be a mapping")
    config_path = training.get("config_path")
    if not config_path:
        raise ValueError("classical checkpoint does not record its config snapshot")
    source = Path(str(config_path)).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("classical config snapshot is missing: %s" % source)
    try:
        with source.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("could not read classical config snapshot %s: %s" % (source, exc))
    if not isinstance(config, Mapping):
        raise ValueError("classical config snapshot root must be a mapping")
    if config.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("classical config snapshot architecture does not match checkpoint")
    return RepresentationConfig.from_mapping(config.get("representation"))


def _selected_groups(
    selection: Mapping[str, Any], field: str, data_root: Path
) -> list[str]:
    records = selection.get(field)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("classical checkpoint %s must be a sequence" % field)
    groups: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or not record.get("group_id"):
            raise ValueError("classical checkpoint %s contains an invalid record" % field)
        group_id = str(record["group_id"])
        if not (data_root / group_id).is_file():
            raise FileNotFoundError(
                "checkpoint selection file is missing from the dataset: %s" % group_id
            )
        groups.append(group_id)
    if len(groups) != len(set(groups)):
        raise ValueError("classical checkpoint %s contains duplicate groups" % field)
    return groups


def predict(
    model_path: str,
    data_path: str,
    batch_size: int = 128,
    device: str = "cpu",
    num_workers: int = 0,
    split: str = "test",
    max_files_per_class: int = 0,
    include_energy_condition: bool = True,
    show_progress: bool = True,
    progress_interval_seconds: float = 10.0,
    split_seed: Optional[int] = None,
    split_fractions: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    """Yield canonical event predictions from one trusted JSON checkpoint."""

    if kwargs:
        raise TypeError(
            "unsupported NEXT classical adapter arguments: %s"
            % ", ".join(sorted(kwargs))
        )
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or int(num_workers) != num_workers or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if (
        isinstance(max_files_per_class, bool)
        or int(max_files_per_class) != max_files_per_class
        or max_files_per_class < 0
    ):
        raise ValueError("max_files_per_class must be a non-negative integer")
    if not isinstance(include_energy_condition, (bool, np.bool_)):
        raise ValueError("include_energy_condition must be true or false")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise ValueError("show_progress must be true or false")
    progress_interval_seconds = float(progress_interval_seconds)
    if not np.isfinite(progress_interval_seconds) or progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be finite and positive")

    checkpoint_path = Path(model_path).expanduser().resolve()
    data_root = Path(data_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("classical checkpoint does not exist: %s" % checkpoint_path)
    if not data_root.is_dir():
        raise FileNotFoundError("NEXT dataset directory does not exist: %s" % data_root)

    _status(bool(show_progress), "loading checkpoint %s" % checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    representation = _representation(checkpoint)
    checkpoint_seed, checkpoint_fractions = _split_contract(checkpoint)
    if split_seed is not None and int(split_seed) != checkpoint_seed:
        raise ValueError("split_seed differs from the checkpoint")
    if split_fractions is not None and not np.allclose(
        np.asarray(split_fractions, dtype=float),
        np.asarray(checkpoint_fractions, dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("split_fractions differ from the checkpoint")

    selection = checkpoint.get("data_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("classical checkpoint data_selection must be a mapping")
    training_groups = _selected_groups(selection, "training_files", data_root)
    validation_groups = _selected_groups(selection, "validation_files", data_root)
    if set(training_groups).intersection(validation_groups):
        raise ValueError("checkpoint training and validation selections overlap")

    actual_inventory = dataset_inventory(data_root)
    maximum = None if int(max_files_per_class) == 0 else int(max_files_per_class)
    files = discover_source_files(
        root=data_root,
        split=str(split),
        split_seed=checkpoint_seed,
        split_fractions=checkpoint_fractions,
        max_files_per_class=maximum,
    )
    forbidden = set(training_groups)
    if str(split) == "test":
        forbidden.update(validation_groups)
    overlap = sorted({source.group_id for source in files}.intersection(forbidden))
    if overlap:
        raise ValueError(
            "prediction split overlaps checkpoint selection data; examples: %s"
            % ", ".join(overlap[:5])
        )

    extractor_config = checkpoint.get("feature_extractor")
    if not isinstance(extractor_config, Mapping):
        raise ValueError("classical checkpoint feature_extractor must be a mapping")
    recorded_names = tuple(str(value) for value in extractor_config.get("feature_names", ()))
    if recorded_names != TOPOLOGY_FEATURE_NAMES:
        raise ValueError("classical checkpoint topology feature order is incompatible")
    extractor = TopologyFeatureExtractor(
        max_topology_points=int(extractor_config["max_topology_points"]),
        connectivity_radius=float(extractor_config["connectivity_radius"]),
        blob_radius=float(extractor_config["blob_radius"]),
    )

    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping) or model_state.get("backend") != "xgboost":
        raise ValueError("classical checkpoint model state is not XGBoost")
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("classical NEXT inference requires xgboost") from exc
    raw = base64.b64decode(str(model_state["booster_raw_base64"]), validate=True)
    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    tree_limit = int(model_state["prediction_tree_limit"])
    if tree_limit < 1:
        raise ValueError("prediction_tree_limit must be positive")

    loader = build_inference_loader(
        files=files,
        representation_config=representation,
        input_kind="topology",
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device_type="cpu",
    )
    metadata = {
        "framework": "xgboost",
        "model_name": str(checkpoint.get("model_name")),
        "architecture_id": ARCHITECTURE_ID,
        "input_kind": "topology",
        "checkpoint_format_version": int(checkpoint.get("format_version", 1)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dataset_root": str(data_root),
        "data_split": str(split),
        "split_seed": checkpoint_seed,
        "split_fractions": list(checkpoint_fractions),
        "dataset_inventory": actual_inventory,
        "source_file_count": len(files),
        "representation": representation.to_dict(),
        "topology_feature_names": list(TOPOLOGY_FEATURE_NAMES),
        "prediction_tree_limit": tree_limit,
        "score_space": "logit",
        "positive_class": "0nubb",
        "requested_device": str(device),
        "inference_device": "cpu",
        "energy_condition_derivation": (
            "float64 sum of MC/hits/table values_block_1 energy rows per event_id"
        ),
        "energy_unit": "MeV",
        "energy_unit_source": (
            "NEXT zeronu benchmark data paper arXiv:2605.07566; "
            "source HDF5 attributes do not declare a unit"
        ),
        "energy_unit_reference": "https://arxiv.org/abs/2605.07566",
        "representation_coverage_definition": (
            "fraction of complete event deposited energy retained by the model representation"
        ),
        "projection_coverage_compatibility_alias": True,
    }
    _status(
        bool(show_progress),
        "selected %d %s files; starting %s on CPU"
        % (len(files), split, ARCHITECTURE_ID),
    )

    def batches() -> Iterator[Dict[str, Any]]:
        produced = 0
        completed_files = 0
        success = False
        progress = _Progress(
            total_files=len(files),
            split=str(split),
            device="cpu",
            enabled=bool(show_progress),
            interval_seconds=progress_interval_seconds,
        )
        try:
            for batch in loader:
                matrix = extractor.extract_batch(
                    batch["coords"], batch["features"], batch["mask"]
                )
                dmatrix = xgb.DMatrix(
                    matrix, feature_names=list(TOPOLOGY_FEATURE_NAMES)
                )
                score = np.asarray(
                    booster.predict(
                        dmatrix,
                        output_margin=True,
                        iteration_range=(0, tree_limit),
                    ),
                    dtype=np.float32,
                )
                if score.ndim != 1 or np.any(~np.isfinite(score)):
                    raise ValueError("classical model logits must be finite and one-dimensional")
                size = len(batch["event_id"])
                if len(score) != size:
                    raise ValueError("classical logits are not aligned with input events")
                coverage = batch["representation_coverage"].numpy().astype(np.float32)
                columns: Dict[str, Any] = {
                    "event_id": np.asarray(batch["event_id"], dtype=str),
                    "label": batch["label"].numpy().astype(np.int8),
                    "category": np.asarray(batch["category"], dtype=str),
                    "score": score,
                    "sample_weight": np.ones(size, dtype=np.float32),
                    "split": np.asarray(batch["split"], dtype=str),
                    "group_id": np.asarray(batch["group_id"], dtype=str),
                    "representation_coverage": coverage,
                    "projection_coverage": coverage,
                    "__metadata__": metadata,
                }
                if bool(include_energy_condition):
                    columns["energy_condition"] = batch["energy_condition"].numpy().astype(np.float64)
                produced += size
                completed_in_batch = int(batch["source_file_complete"].sum().item())
                completed_files += completed_in_batch
                progress.update(completed_in_batch, produced)
                yield columns
            if produced == 0:
                raise RuntimeError("NEXT classical inference produced no events")
            if completed_files != len(files):
                raise RuntimeError(
                    "NEXT inference completed %d/%d selected source files"
                    % (completed_files, len(files))
                )
            success = True
        finally:
            progress.close(success)

    return batches()


__all__ = ["predict"]
