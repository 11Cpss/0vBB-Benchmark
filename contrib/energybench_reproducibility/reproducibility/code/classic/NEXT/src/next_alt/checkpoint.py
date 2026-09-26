"""Format-version-3 checkpoint contract for alternative NEXT models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from .config import INPUT_KINDS


FORMAT_VERSION = 3
TASK = "binary_classification"


def build_v3_checkpoint(
    *,
    architecture_id: str,
    model_name: str,
    input_kind: str,
    model_config: Mapping[str, Any],
    model: torch.nn.Module,
    representation_config: Mapping[str, Any],
    split_config: Mapping[str, Any],
    data_selection: Mapping[str, Any],
    training_config: Mapping[str, Any],
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    history: Sequence[Mapping[str, Any]] = (),
    validation_metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a checkpoint payload with all adapter and provenance fields."""

    if input_kind not in INPUT_KINDS:
        raise ValueError("unsupported input_kind: %s" % input_kind)
    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "task": TASK,
        "architecture_id": str(architecture_id),
        "model_name": str(model_name),
        "input_kind": str(input_kind),
        "model_config": dict(model_config),
        "state_dict": model.state_dict(),
        "representation_config": dict(representation_config),
        "split_config": dict(split_config),
        "data_selection": dict(data_selection),
        "training_config": dict(training_config),
        "epoch": int(epoch),
        "history": list(history),
    }
    if validation_metrics is not None:
        payload["validation_metrics"] = dict(validation_metrics)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return validate_v3_checkpoint(payload)


def validate_v3_checkpoint(
    payload: Any,
    *,
    architecture_id: Optional[str] = None,
    input_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate v3 identity, model reconstruction, and data provenance fields."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    required = {
        "format_version",
        "task",
        "architecture_id",
        "model_name",
        "input_kind",
        "model_config",
        "state_dict",
        "representation_config",
        "split_config",
        "data_selection",
        "training_config",
        "epoch",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("v3 checkpoint is missing: %s" % ", ".join(missing))
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(
            "unsupported checkpoint format_version %r (expected %d)"
            % (payload["format_version"], FORMAT_VERSION)
        )
    if payload["task"] != TASK:
        raise ValueError("checkpoint task must be %s" % TASK)
    checkpoint_architecture = str(payload["architecture_id"])
    checkpoint_input_kind = str(payload["input_kind"])
    if not checkpoint_architecture:
        raise ValueError("checkpoint architecture_id cannot be empty")
    if not str(payload["model_name"]):
        raise ValueError("checkpoint model_name cannot be empty")
    if checkpoint_input_kind not in INPUT_KINDS:
        raise ValueError("checkpoint contains unsupported input_kind")
    if architecture_id is not None and checkpoint_architecture != architecture_id:
        raise ValueError(
            "checkpoint architecture_id %r does not match %r"
            % (checkpoint_architecture, architecture_id)
        )
    if input_kind is not None and checkpoint_input_kind != input_kind:
        raise ValueError(
            "checkpoint input_kind %r does not match %r"
            % (checkpoint_input_kind, input_kind)
        )
    for field in (
        "model_config",
        "state_dict",
        "representation_config",
        "split_config",
        "data_selection",
        "training_config",
    ):
        if not isinstance(payload[field], Mapping):
            raise ValueError("checkpoint %s must be a mapping" % field)
    selection = payload["data_selection"]
    for field in ("inventory", "training_groups", "validation_groups"):
        if field not in selection:
            raise ValueError("checkpoint data_selection is missing %s" % field)
    if not isinstance(selection["inventory"], Mapping):
        raise ValueError("checkpoint inventory must be a mapping")
    if not isinstance(selection["training_groups"], Sequence) or isinstance(
        selection["training_groups"], (str, bytes)
    ):
        raise ValueError("checkpoint training_groups must be a sequence")
    if not isinstance(selection["validation_groups"], Sequence) or isinstance(
        selection["validation_groups"], (str, bytes)
    ):
        raise ValueError("checkpoint validation_groups must be a sequence")
    try:
        epoch = int(payload["epoch"])
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint epoch must be a non-negative integer") from exc
    if epoch != payload["epoch"] or epoch < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    return dict(payload)


def save_v3_checkpoint(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically save a v3 checkpoint.

    Existing files are rejected unless ``overwrite`` is explicitly true.
    The training runner uses overwrite only after it has claimed a fresh pair
    of best/last paths at run start.
    """

    validated = validate_v3_checkpoint(payload)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing checkpoint: %s" % destination
        )
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(validated, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_v3_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    architecture_id: Optional[str] = None,
    input_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and validate a trusted local v3 checkpoint for training/adapters."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError("checkpoint does not exist: %s" % source)
    try:
        loaded = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        loaded = torch.load(source, map_location=map_location)
    return validate_v3_checkpoint(
        loaded,
        architecture_id=architecture_id,
        input_kind=input_kind,
    )


__all__ = [
    "FORMAT_VERSION",
    "TASK",
    "build_v3_checkpoint",
    "load_v3_checkpoint",
    "save_v3_checkpoint",
    "validate_v3_checkpoint",
]
