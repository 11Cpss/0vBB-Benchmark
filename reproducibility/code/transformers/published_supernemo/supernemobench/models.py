"""Strict registry for the configured local model experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from architectures import get_model_class
from .config import ARCHITECTURES, TASKS, TRANSFORMER_ARCHITECTURE_IDS
from .tokenization import SuperNEMOTrackerTokenizationConfig


_EXPECTED_MODEL_COUNT = 10


@dataclass(frozen=True)
class ModelSpec:
    """Checkpoint-facing identity for one locally stored architecture."""

    architecture_id: str
    model_name: str
    input_kind: str
    model_class: type[Any]
    tokenization: SuperNEMOTrackerTokenizationConfig | None


def _configured_model_ids() -> tuple[str, ...]:
    if not isinstance(ARCHITECTURES, Mapping):
        raise RuntimeError("ARCHITECTURES must be a mapping")
    identifiers = tuple(ARCHITECTURES)
    if len(identifiers) != _EXPECTED_MODEL_COUNT:
        raise RuntimeError(
            "SuperNEMO must configure exactly ten model experiments; "
            f"found {len(identifiers)}"
        )
    for identifier in identifiers:
        if not isinstance(identifier, str) or not identifier:
            raise RuntimeError("every configured model ID must be a non-empty string")
        if identifier != identifier.strip() or identifier != identifier.lower():
            raise RuntimeError(
                f"configured model ID must be exact lowercase text: {identifier!r}"
            )
        configured = ARCHITECTURES[identifier]
        if configured.architecture_id != identifier:
            raise RuntimeError(
                "ARCHITECTURES key and architecture_id disagree for "
                f"{identifier!r}: {configured.architecture_id!r}"
            )
        if not isinstance(configured.model, Mapping):
            raise RuntimeError(f"model options for {identifier!r} must be a mapping")
    configured_transformers = tuple(
        identifier
        for identifier in identifiers
        if ARCHITECTURES[identifier].tokenization is not None
    )
    if configured_transformers != TRANSFORMER_ARCHITECTURE_IDS:
        raise RuntimeError("active Transformer registry is not the frozen six")
    return identifiers


def _require_identifier(identifier: str) -> str:
    identifiers = _configured_model_ids()
    if not isinstance(identifier, str):
        raise TypeError("model identifier must be an exact lowercase string")
    if identifier in ARCHITECTURES:
        return identifier

    stripped = identifier.strip()
    if stripped in ARCHITECTURES:
        raise KeyError(
            f"model ID {identifier!r} contains surrounding whitespace; "
            f"use {stripped!r}"
        )
    case_matches = [item for item in identifiers if item.casefold() == identifier.casefold()]
    if case_matches:
        raise KeyError(
            f"model IDs are case-sensitive; use exact ID {case_matches[0]!r}"
        )

    alias_matches = [
        item
        for item, configured in ARCHITECTURES.items()
        if identifier == configured.model_name
    ]
    if alias_matches:
        raise KeyError(
            f"model-name alias {identifier!r} is not accepted; "
            f"use exact architecture ID {alias_matches[0]!r}"
        )
    raise KeyError(
        f"unknown SuperNEMO model ID {identifier!r}; aliases and short names are "
        f"not accepted; expected one of: {', '.join(identifiers)}"
    )


def _require_task(task: str) -> str:
    if not isinstance(task, str):
        raise TypeError("task must be an exact string")
    if task in TASKS:
        return task
    stripped = task.strip()
    if stripped in TASKS:
        raise ValueError(
            f"task {task!r} contains surrounding whitespace; use {stripped!r}"
        )
    case_matches = [item for item in TASKS if item.casefold() == task.casefold()]
    if case_matches:
        raise ValueError(f"tasks are case-sensitive; use {case_matches[0]!r}")
    raise ValueError(f"unknown task {task!r}; expected one of: {', '.join(TASKS)}")


def _checked_local_spec(identifier: str) -> ModelSpec:
    configured = ARCHITECTURES[identifier]
    model_class = get_model_class(configured.model_name)
    if model_class.__name__ != configured.model_name:
        raise RuntimeError(
            f"local model class name changed for {identifier!r}: "
            f"{model_class.__name__!r}"
        )
    return ModelSpec(
        architecture_id=configured.architecture_id,
        model_name=configured.model_name,
        input_kind=configured.input_kind,
        model_class=model_class,
        tokenization=configured.tokenization,
    )


def registered_models() -> tuple[str, ...]:
    """Return the exact lowercase model IDs after local registry validation."""

    identifiers = _configured_model_ids()
    for identifier in identifiers:
        _checked_local_spec(identifier)
    return identifiers


def get_model_spec(identifier: str) -> ModelSpec:
    """Resolve one exact ID against the local model registry."""

    resolved = _require_identifier(identifier)
    return _checked_local_spec(resolved)


def build_model(identifier: str, task: str) -> Any:
    """Build one unchanged local model copy and attach only task metadata."""

    resolved = _require_identifier(identifier)
    resolved_task = _require_task(task)
    spec = _checked_local_spec(resolved)
    configured = ARCHITECTURES[resolved]
    if resolved_task not in configured.tasks:
        raise ValueError(
            f"{resolved!r} does not support task {resolved_task!r}; expected one of: "
            + ", ".join(configured.tasks)
        )
    model = spec.model_class(**dict(configured.model))
    model.architecture_id = configured.architecture_id
    model.model_name = configured.model_name
    model.input_kind = configured.input_kind
    model.task = resolved_task
    model.tokenization_config = configured.tokenization
    return model


__all__ = ["ModelSpec", "build_model", "get_model_spec", "registered_models"]
