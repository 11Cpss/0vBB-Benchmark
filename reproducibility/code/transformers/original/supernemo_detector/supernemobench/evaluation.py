"""Canonical SuperNEMO classifier export and strict EnergyBench evaluation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CATEGORY_FIELD,
    CLASSIFICATION_LABELS,
    COORDINATE_FIELDS,
    ENERGY_TARGET,
    ENERGY_UNIT,
    EVENT_CONSTANT_FIELDS,
    PROJECT_ROOT,
    RADIUS_FIELD,
)
from supernemo_transformer import (
    SuperNEMOTokenizationConfig,
    tokenize_tracker_event,
)


EVALUATION_MANIFEST = Path(
    os.environ.get(
        "SUPERNEMO_EVALUATION_MANIFEST",
        "/home/wenyu/SuperNEMO/evaluation/supernemo_2nu_vs_bi214.json",
    )
).expanduser()
ENERGYBENCH_PACKAGE = Path(
    os.environ.get(
        "SUPERNEMO_ENERGYBENCH_PACKAGE",
        str(
            PROJECT_ROOT.parent
            / "exo200_detector"
            / "frozen_energybench"
            / "energybench"
        ),
    )
).expanduser()
SELECTION_SPLIT = "validation"
EVALUATION_SPLIT = "test"


def _energybench_has_frozen_api() -> bool:
    """Return whether the active module is the frozen collaboration evaluator."""

    try:
        config = importlib.import_module("energybench.config")
        roc = importlib.import_module("energybench.roc")
        data = importlib.import_module("energybench.data")
        evaluation = importlib.import_module("energybench.evaluation")
        utils = importlib.import_module("energybench.utils")
    except (ImportError, AttributeError):
        return False
    return all(
        (
            callable(getattr(config, "load_manifest", None)),
            callable(getattr(roc, "evaluate_energy_matched_roc", None)),
            getattr(data, "PredictionBundle", None) is not None,
            callable(getattr(evaluation, "run_evaluation", None)),
            callable(getattr(utils, "write_json", None)),
        )
    )


def _ensure_energybench_api() -> None:
    """Load the frozen evaluator explicitly when another namesake is installed."""

    if _energybench_has_frozen_api():
        return
    package_dir = ENERGYBENCH_PACKAGE.resolve()
    initializer = package_dir / "__init__.py"
    required = (
        initializer,
        package_dir / "config.py",
        package_dir / "roc.py",
        package_dir / "data.py",
        package_dir / "evaluation.py",
        package_dir / "utils.py",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "The frozen EnergyBench package is unavailable or has the wrong API at "
            f"{package_dir}; missing: {', '.join(missing)}. Set "
            "SUPERNEMO_ENERGYBENCH_PACKAGE to the local frozen energybench directory."
        )
    for module_name in tuple(sys.modules):
        if module_name == "energybench" or module_name.startswith("energybench."):
            del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        "energybench",
        initializer,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the frozen EnergyBench from {package_dir}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["energybench"] = package
    try:
        spec.loader.exec_module(package)
    except Exception:
        sys.modules.pop("energybench", None)
        raise
    if not _energybench_has_frozen_api():
        raise RuntimeError(
            f"EnergyBench at {package_dir} does not expose Wing's required API"
        )


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


def energybench_code_sha256() -> str:
    """Match EnergyBench's own package-level code fingerprint."""

    _ensure_energybench_api()
    try:
        import energybench
    except ImportError as error:
        raise RuntimeError(
            "the repository's frozen EnergyBench package is required"
        ) from error
    package_dir = Path(energybench.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
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
    prediction_order_sha256: dict[str, str] = {}
    source_by_key = {
        str(item["source_key"]): item for item in manifest.get("inventory", ())
    }
    for split_name in (SELECTION_SPLIT, EVALUATION_SPLIT):
        digest = hashlib.sha256()
        for event_slice in manifest.get("splits", {}).get(split_name, ()):
            source_key = str(event_slice["source_key"])
            source = source_by_key.get(source_key, {})
            if task == "classification" and source.get("classification_label") is None:
                continue
            for event_number in range(
                int(event_slice["event_start"]), int(event_slice["event_stop"])
            ):
                digest.update(
                    f"SuperNEMO::{source_key}::{event_number}\n".encode("utf-8")
                )
        prediction_order_sha256[split_name] = digest.hexdigest()

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
            "evaluator_code_sha256": energybench_code_sha256(),
        },
        "prediction_order_sha256": prediction_order_sha256,
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
        tokenization = SuperNEMOTokenizationConfig(
            **dict(representation_config)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid tracker-token representation config: {error}") from error
    tokenization_source = inspect.getsourcefile(tokenize_tracker_event)
    if tokenization_source is None:
        raise ValueError("cannot identify the tokenizer source file")
    tokenization_path = Path(tokenization_source).resolve()
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


def matched_validation_auc(
    label: np.ndarray,
    score: np.ndarray,
    energy_condition: np.ndarray,
    *,
    seed: int,
    expected_manifest_sha256: str | None = None,
    expected_evaluator_sha256: str | None = None,
) -> float:
    """Apply the frozen headline matching rule on validation predictions."""

    _ensure_energybench_api()
    try:
        from energybench.config import load_manifest
        from energybench.roc import evaluate_energy_matched_roc
    except ImportError as error:
        raise RuntimeError(
            "the repository's frozen EnergyBench package is required"
        ) from error
    if (
        expected_manifest_sha256 is not None
        and file_sha256(EVALUATION_MANIFEST) != expected_manifest_sha256
    ):
        raise ValueError("EnergyBench protocol manifest changed during this run")
    if (
        expected_evaluator_sha256 is not None
        and energybench_code_sha256() != expected_evaluator_sha256
    ):
        raise ValueError("EnergyBench evaluator code changed during this run")
    protocol = load_manifest(EVALUATION_MANIFEST)
    classification = protocol["classification"]
    if int(protocol["runtime"]["seed"]) != int(seed):
        raise ValueError("training seed and EnergyBench protocol seed disagree")
    target = (
        "legacy_uniform"
        if classification["matching_target"] == "uniform"
        else "overlap"
    )
    energy_roi = classification.get("energy_roi")
    result = evaluate_energy_matched_roc(
        np.asarray(label, dtype=np.int8),
        np.asarray(score, dtype=np.float64),
        np.asarray(energy_condition, dtype=np.float64),
        positive_label=int(classification["positive_label"]),
        n_bins=int(classification["energy_bins"]),
        min_per_class=int(classification["min_per_class"]),
        target=target,
        target_tpr=float(classification["target_tpr"]),
        n_bootstrap=0,
        random_state=int(seed),
        support_trim_quantile=float(classification["support_trim_quantile"]),
        energy_roi=(
            None
            if energy_roi is None
            else (float(energy_roi[0]), float(energy_roi[1]))
        ),
    )
    valid_bins = sum(1 for item in result.bins if item.valid)
    coverage = min(
        result.coverage.signal_matched_weight_fraction,
        result.coverage.background_matched_weight_fraction,
    )
    if (
        result.status != "ok"
        or result.matched_auc is None
        or valid_bins < int(classification["min_valid_bins"])
        or coverage < float(classification["min_coverage"])
    ):
        raise RuntimeError(
            "validation energy-matched AUC is not evaluable under the frozen "
            f"protocol: status={result.status!r}, reason={result.reason!r}, "
            f"valid_bins={valid_bins}, coverage={coverage:.6g}"
        )
    return float(result.matched_auc)


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

    _ensure_energybench_api()
    try:
        from energybench.data import PredictionBundle
    except ImportError as error:
        raise RuntimeError(
            "the repository's frozen EnergyBench package is required"
        ) from error
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
    unique_splits = np.unique(splits).tolist()
    if len(unique_splits) != 1:
        raise ValueError("one evaluation bundle must contain exactly one split")
    provenance = metadata.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("prediction metadata has no dataset provenance")
    expected_orders = provenance.get("prediction_order_sha256")
    if not isinstance(expected_orders, Mapping):
        raise ValueError("dataset provenance has no prediction-order fingerprints")
    expected_order = expected_orders.get(unique_splits[0])
    actual_digest = hashlib.sha256()
    for event_identifier in ids:
        actual_digest.update(f"{event_identifier}\n".encode("utf-8"))
    if actual_digest.hexdigest() != expected_order:
        raise ValueError(
            f"event_id ordering does not match the split manifest for {unique_splits[0]!r}"
        )
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
    _ensure_energybench_api()
    try:
        from energybench.data import save_bundle
    except ImportError as error:
        raise RuntimeError("EnergyBench is not importable") from error
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prediction bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        save_bundle(bundle, temporary)
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


def evaluate_classification_bundle(
    bundle: Any,
    *,
    architecture_id: str,
    provenance: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Save one test bundle and run the user's EnergyBench in strict mode."""

    _ensure_energybench_api()
    try:
        from energybench.config import load_manifest
        from energybench.data import load_bundle
        from energybench.evaluation import run_evaluation
        from energybench.utils import write_json
    except ImportError as error:
        raise RuntimeError(
            "the repository's frozen EnergyBench package is required"
        ) from error
    destination = Path(output_dir).expanduser().resolve()
    evaluation_path = destination / "test_evaluation"
    legacy_paths = (
        destination / "test_predictions.npz",
        destination / "test_metrics.json",
    )
    occupied = [
        path for path in (*legacy_paths, evaluation_path) if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite existing test artifacts: "
            + ", ".join(str(path) for path in occupied)
        )
    split_values = np.unique(bundle.require("split").astype(str)).tolist()
    if split_values != [EVALUATION_SPLIT]:
        raise ValueError(
            f"strict test evaluation requires split={EVALUATION_SPLIT!r}; "
            f"received {split_values}"
        )
    assert_same_provenance(
        bundle.metadata.get("dataset_provenance"),
        provenance,
        context="prediction bundle",
    )
    config = load_manifest(EVALUATION_MANIFEST)
    protocol = provenance.get("evaluation_protocol", {})
    if protocol.get("manifest_sha256") != file_sha256(EVALUATION_MANIFEST):
        raise ValueError("checkpoint and current EnergyBench protocol manifest disagree")
    if config["dataset"]["dataset_version"] != provenance["manifest_content_sha256"]:
        raise ValueError("EnergyBench manifest and split manifest versions disagree")
    checkpoint_sha = str(bundle.metadata.get("checkpoint", {}).get("sha256", ""))
    if len(checkpoint_sha) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha.lower()
    ):
        raise ValueError("prediction metadata has no valid checkpoint SHA256")
    config["model_id"] = f"{architecture_id}@{checkpoint_sha[:12]}"

    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".test-evaluation-", dir=destination))
    try:
        staged_prediction = _save_bundle_atomic(
            bundle, staging / "test_predictions.npz"
        )
        loaded = load_bundle(staged_prediction)
        report = run_evaluation(
            loaded,
            config,
            staging / "energybench",
            strict=True,
            allow_existing=False,
        )
        quality = report.get("quality", {})
        evaluator_fingerprint = report.get("evaluator", {}).get("code_fingerprint")
        if evaluator_fingerprint != protocol.get("evaluator_code_sha256"):
            raise RuntimeError("EnergyBench evaluator code differs from the checkpoint")
        if quality.get("errors") or quality.get("warnings"):
            raise RuntimeError(
                "strict EnergyBench quality checks were not clean: "
                f"warnings={quality.get('warnings')}, errors={quality.get('errors')}"
            )
        classification = report.get("classification", {})
        matched_auc = classification.get("aggregates", {}).get("matched_auc_macro")
        if classification.get("status") != "ok" or matched_auc is None:
            raise RuntimeError("EnergyBench did not produce the required matched AUC")
        final_prediction = evaluation_path / "test_predictions.npz"
        report["input"]["path"] = str(final_prediction)
        write_json(staging / "energybench" / ".energybench" / "metrics.json", report)
        native_metrics = bundle.metadata.get("native_metrics", {})
        if not isinstance(native_metrics, Mapping):
            raise TypeError("prediction metadata native_metrics must be a mapping")
        metrics_payload = {
            "loss": native_metrics.get("loss"),
            "auc": native_metrics.get("auc"),
            "accuracy": native_metrics.get("accuracy"),
            "energy_matched_auc": float(matched_auc),
            "events": int(bundle.n_events),
            "energybench_model_id": config["model_id"],
            "evaluation_fingerprint": report.get("evaluation_fingerprint"),
            "protocol_fingerprint": report.get("protocol_fingerprint"),
        }
        for coverage_metric in (
            "token_coverage_mean",
            "token_coverage_minimum",
            "token_truncated_events",
            "token_truncated_event_fraction",
        ):
            if coverage_metric in native_metrics:
                metrics_payload[coverage_metric] = native_metrics[coverage_metric]
        staged_metrics = staging / "test_metrics.json"
        staged_metrics.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            os.rename(staging, evaluation_path)
        except OSError as error:
            if evaluation_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite test evaluation: {evaluation_path}"
                ) from error
            raise
        return metrics_payload
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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


__all__ = [
    "EVALUATION_MANIFEST",
    "assert_same_provenance",
    "classification_bundle",
    "dataset_provenance",
    "energybench_code_sha256",
    "evaluate_classification_bundle",
    "file_sha256",
    "matched_validation_auc",
    "save_validation_bundle",
]
