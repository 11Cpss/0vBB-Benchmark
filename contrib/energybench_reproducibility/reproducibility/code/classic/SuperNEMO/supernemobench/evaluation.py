"""SuperNEMO classic prediction export and checkpoint/data identity checks."""
from __future__ import annotations
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import numpy as np
from .config import (CATEGORY_FIELD, CLASSIFICATION_LABELS, COORDINATE_FIELDS,
    ENERGY_TARGET, ENERGY_UNIT, EVENT_CONSTANT_FIELDS, PROJECT_ROOT, RADIUS_FIELD)
from .tokenization import SuperNEMOTrackerTokenizationConfig
EVALUATION_MANIFEST = PROJECT_ROOT / "evaluation" / "selection.json"
SELECTION_SPLIT = "validation"
EVALUATION_SPLIT = "test"


@dataclass
class PredictionBundle:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    def require(self, name: str) -> np.ndarray:
        return self.arrays[name]
    @property
    def n_events(self) -> int:
        return len(self.arrays["event_id"])

def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA256 digest for one regular file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def dataset_provenance(
    *,
    manifest_path: str | Path,
    data_root: str | Path,
    task: str,
    counts: Mapping[str, Any],
    data_config: Mapping[str, Any],
    input_kind: str,
    representation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the checkpoint-facing identity of the validated data release."""

    path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read validated split manifest {path}: {error}") from error
    content_digest = manifest.get("content_sha256")
    if not isinstance(content_digest, str) or len(content_digest) != 64:
        raise ValueError("split manifest has no valid content_sha256")
    expected_counts = manifest.get("counts", {}).get(task)
    if expected_counts != dict(counts):
        raise ValueError("prepared-data counts disagree with the split manifest")
    root = Path(data_root).expanduser().resolve()
    if manifest.get("data_root") != str(root):
        raise ValueError("prepared data root disagrees with the split manifest")
    if input_kind not in {"projection2d", "graph", "sequence"}:
        raise ValueError(f"invalid input kind for provenance: {input_kind!r}")
    adapter_path = Path(__file__).with_name("data.py").resolve()
    payload = {
        "schema_version": 1,
        "dataset": "SuperNEMO",
        "task": str(task),
        "data_root": str(root),
        "manifest_path": str(path),
        "manifest_file_sha256": file_sha256(path),
        "manifest_content_sha256": content_digest,
        "counts": dict(counts),
        "inventory": manifest.get("inventory"),
        "grouping": manifest.get("grouping"),
        "input_policy": manifest.get("input_policy"),
        "data_config": dict(data_config),
        "input_kind": input_kind,
        "data_adapter": {
            "path": str(adapter_path),
            "sha256": file_sha256(adapter_path),
        },
        "evaluation_protocol": {
            "manifest_path": str(EVALUATION_MANIFEST.resolve()),
            "manifest_sha256": file_sha256(EVALUATION_MANIFEST),
            "selection_metric": "auc",
            "selection_split": "validation",
        },
        "classification_label_mapping": dict(CLASSIFICATION_LABELS),
        "energy": {"definition": ENERGY_TARGET, "unit": ENERGY_UNIT},
    }
    if representation_config is None:
        return payload
    if task != "classification" or input_kind != "sequence":
        raise ValueError(
            "tracker-token provenance is supported only for classification sequences"
        )
    try:
        tokenization = SuperNEMOTrackerTokenizationConfig(
            **dict(representation_config)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid tracker-token representation config: {error}") from error
    tokenization_path = Path(__file__).with_name("tokenization.py").resolve()
    evaluation_adapter_path = Path(__file__).resolve()
    manifest_input_policy = manifest.get("input_policy")
    if not isinstance(manifest_input_policy, Mapping):
        raise ValueError("split manifest has no valid input_policy")
    excluded = [
        str(field)
        for field in manifest_input_policy.get("excluded_from_model", ())
        if str(field) != RADIUS_FIELD
    ]
    required_excluded = (*EVENT_CONSTANT_FIELDS, CATEGORY_FIELD)
    if any(field not in excluded for field in required_excluded):
        raise ValueError("split manifest does not exclude every event-level model field")
    payload.update(
        {
            "schema_version": 2,
            "input_policy": {
                "excluded_from_model": excluded,
                "fields": [*COORDINATE_FIELDS, RADIUS_FIELD],
                "representation": "tracker_tokens",
                "missing_value_handling": {
                    RADIUS_FIELD: {
                        "accepted_missing_value": "NaN",
                        "imputed_value_before_scaling": 0.0,
                        "validity_indicator_feature": True,
                    }
                },
            },
            "representation_config": tokenization.to_dict(),
            "tokenization_source": {
                "path": str(tokenization_path),
                "sha256": file_sha256(tokenization_path),
            },
            "evaluation_adapter": {
                "path": str(evaluation_adapter_path),
                "sha256": file_sha256(evaluation_adapter_path),
            },
        }
    )
    return payload

def assert_same_provenance(
    recorded: Any,
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject checkpoints or predictions from a different data manifest."""

    if not isinstance(recorded, Mapping):
        raise ValueError(f"{context} does not contain dataset provenance")
    if dict(recorded) != dict(expected):
        keys = sorted(set(recorded) | set(expected))
        changed = [key for key in keys if recorded.get(key) != expected.get(key)]
        raise ValueError(
            f"{context} dataset provenance does not match this run; changed: "
            + ", ".join(changed)
        )


def _string_column(name: str, values: Any, size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != size:
        raise ValueError(f"{name} must be an aligned one-dimensional column")
    result = array.astype(np.str_, copy=False)
    if np.any(np.char.strip(result) == ""):
        raise ValueError(f"{name} must not contain empty strings")
    return result

def classification_bundle(
    *,
    event_id: Any,
    label: Any,
    category: Any,
    score: Any,
    energy_condition: Any,
    group_id: Any,
    split: Any,
    metadata: Mapping[str, Any],
) -> Any:
    """Validate and construct the sole canonical 2nu-vs-Bi214 bundle."""

    labels = np.asarray(label)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("label must be a non-empty one-dimensional column")
    if not np.all(np.logical_or(labels == 0, labels == 1)):
        raise ValueError("classification labels must contain only 0 and 1")
    labels = labels.astype(np.int8, copy=False)
    size = int(labels.size)
    ids = _string_column("event_id", event_id, size)
    categories = _string_column("category", category, size)
    groups = _string_column("group_id", group_id, size)
    splits = _string_column("split", split, size)
    if np.unique(ids).size != size:
        raise ValueError("event_id must be unique in one evaluation bundle")
    expected_label = np.asarray(
        [CLASSIFICATION_LABELS.get(item, -1) for item in categories],
        dtype=np.int8,
    )
    if not np.array_equal(labels, expected_label):
        raise ValueError("category and label columns disagree with 2nu=1/Bi214=0")
    scores = np.asarray(score, dtype=np.float32)
    energies = np.asarray(energy_condition, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != size or not np.all(np.isfinite(scores)):
        raise ValueError("score must be an aligned finite one-dimensional column")
    if (
        energies.ndim != 1
        or len(energies) != size
        or not np.all(np.isfinite(energies))
        or np.any(energies <= 0.0)
    ):
        raise ValueError(
            "energy_condition must be aligned, finite, positive E1+E2 values"
        )
    return PredictionBundle(
        {
            "event_id": ids,
            "label": labels,
            "category": categories,
            "score": scores,
            "energy_condition": energies,
            "sample_weight": np.ones(size, dtype=np.float32),
            "group_id": groups,
            "split": splits,
        },
        metadata=dict(metadata),
    )

def _save_bundle_atomic(bundle: Any, path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prediction bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        np.savez_compressed(temporary, **bundle.arrays, __metadata__=np.asarray(json.dumps(bundle.metadata, sort_keys=True)))
        try:
            # A hard link is an atomic create-if-absent operation on the same
            # filesystem, so concurrent evaluators cannot overwrite each other.
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite prediction bundle: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return path

def export_classification_bundle(bundle: PredictionBundle, *, architecture_id: str,
                                 provenance: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Export aligned test predictions; compute final metrics with benchmark/."""
    if np.unique(bundle.require("split").astype(str)).tolist() != [EVALUATION_SPLIT]:
        raise ValueError("Test export requires split='test'.")
    assert_same_provenance(bundle.metadata.get("dataset_provenance"), provenance,
                           context="prediction bundle")
    destination = Path(output_dir).expanduser().resolve()
    path = _save_bundle_atomic(bundle, destination / "test_predictions.npz")
    return {"events": bundle.n_events, "prediction_file": str(path),
            "architecture_id": architecture_id,
            "native_metrics": dict(bundle.metadata.get("native_metrics", {}))}


def save_validation_bundle(
    bundle: Any,
    *,
    output_dir: str | Path,
) -> Path:
    """Save validation predictions canonically without running locked-test metrics."""

    values = np.unique(bundle.require("split").astype(str)).tolist()
    if values != [SELECTION_SPLIT]:
        raise ValueError(
            f"validation export requires split={SELECTION_SPLIT!r}; received {values}"
        )
    return _save_bundle_atomic(
        bundle,
        Path(output_dir).expanduser().resolve() / "validation_predictions.npz",
    )
