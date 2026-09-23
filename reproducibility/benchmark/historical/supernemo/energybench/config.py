"""Evaluation manifest loading and stable protocol defaults."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


DEFAULT_CONFIG = {
    "schema_version": 1,
    "task_id": "energy-evaluation",
    "model_id": "unnamed-model",
    "dataset": {
        "experiment": "unspecified",
        "dataset_id": "unspecified",
        "dataset_version": "unspecified",
        "split": "test",
        "selection_split": None,
        "energy_condition_kind": "unspecified",
        "energy_target_kind": "unspecified",
        "energy_unit": "unspecified",
    },
    "columns": {
        "event_id": None,
        "label": None,
        "score": None,
        "energy_condition": None,
        "energy_true": None,
        "energy_pred": None,
        "category": None,
        "sample_weight": None,
        "group_id": None,
        "split": None,
    },
    "classification": {
        "enabled": "auto",
        "positive_label": "1",
        "signal_categories": [],
        "background_categories": [],
        "pair_mode": "both",
        "score_direction": "higher",
        "score_space": "unspecified",
        "energy_bins": 6,
        "matching_target": "overlap",
        "min_per_class": 20,
        "min_valid_bins": 2,
        "support_trim_quantile": 0.005,
        "energy_roi": None,
        "min_coverage": 0.5,
        "target_tpr": 0.90,
        "bootstrap": 200,
        "confidence": 0.95,
    },
    "regression": {
        "enabled": "auto",
        "histogram_bins": 50,
        "performance_bins": 10,
        "energy_floor": None,
        "histogram_edges": None,
        "bootstrap": 200,
        "confidence": 0.95,
    },
    "dependence": {
        "enabled": True,
        "energy_bins": 8,
        "score_bins": 20,
        "min_per_bin": 20,
        "distance_correlation_max_samples": 1200,
    },
    "runtime": {
        "seed": 42,
        "make_plots": True,
    },
}


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_manifest(path: Optional[Any] = None) -> Dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("manifest does not exist: %s" % source)
    suffix = source.suffix.lower()
    with source.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML manifest requires PyYAML; use JSON or install PyYAML"
                ) from exc
            payload = yaml.safe_load(handle)
        else:
            raise ValueError("manifest must end in .json, .yaml, or .yml")
    if not isinstance(payload, Mapping):
        raise ValueError("manifest root must be a mapping/object")
    config = deep_merge(DEFAULT_CONFIG, payload)
    config["_manifest_path"] = str(source)
    return config


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply dot-separated command-line overrides, ignoring ``None`` values."""

    result = copy.deepcopy(dict(config))
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        parts = dotted_key.split(".")
        target = result
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], MutableMapping):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    def positive_integer(value: Any, name: str, minimum: int = 1) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be an integer" % name)
        if number != value or number < minimum:
            raise ValueError(
                "%s must be an integer >= %d" % (name, minimum)
            )
        return number

    def probability(
        value: Any, name: str, include_endpoints: bool = False
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be numeric" % name)
        valid = (
            0.0 <= number <= 1.0
            if include_endpoints
            else 0.0 < number < 1.0
        )
        if not valid:
            interval = "[0, 1]" if include_endpoints else "(0, 1)"
            raise ValueError("%s must lie in %s" % (name, interval))
        return number

    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("only manifest schema_version 1 is supported")
    classification = config["classification"]
    if classification["enabled"] not in {True, False, "auto"}:
        raise ValueError("classification.enabled must be true, false, or auto")
    if classification["score_direction"] not in {"higher", "lower"}:
        raise ValueError("classification.score_direction must be higher or lower")
    if classification["matching_target"] not in {"overlap", "uniform"}:
        raise ValueError("classification.matching_target must be overlap or uniform")
    if classification["pair_mode"] not in {"pooled", "categories", "both"}:
        raise ValueError(
            "classification.pair_mode must be pooled, categories, or both"
        )
    trim = float(classification["support_trim_quantile"])
    if not 0.0 <= trim < 0.5:
        raise ValueError("support_trim_quantile must be in [0, 0.5)")
    positive_integer(
        classification["energy_bins"], "classification.energy_bins"
    )
    positive_integer(
        classification["min_per_class"],
        "classification.min_per_class",
    )
    positive_integer(
        classification["min_valid_bins"],
        "classification.min_valid_bins",
    )
    positive_integer(
        classification["bootstrap"],
        "classification.bootstrap",
        minimum=0,
    )
    probability(
        classification["min_coverage"],
        "classification.min_coverage",
        include_endpoints=True,
    )
    probability(
        classification["target_tpr"], "classification.target_tpr"
    )
    probability(
        classification["confidence"], "classification.confidence"
    )
    roi = classification.get("energy_roi")
    if roi is not None:
        if len(roi) != 2 or not float(roi[1]) > float(roi[0]):
            raise ValueError(
                "classification.energy_roi must be [LOW, HIGH] with HIGH > LOW"
            )

    regression = config["regression"]
    if regression["enabled"] not in {True, False, "auto"}:
        raise ValueError("regression.enabled must be true, false, or auto")
    positive_integer(
        regression["histogram_bins"], "regression.histogram_bins"
    )
    positive_integer(
        regression["performance_bins"], "regression.performance_bins"
    )
    positive_integer(
        regression["bootstrap"], "regression.bootstrap", minimum=0
    )
    probability(regression["confidence"], "regression.confidence")
    if regression.get("energy_floor") is not None and not (
        float(regression["energy_floor"]) > 0.0
    ):
        raise ValueError("regression.energy_floor must be > 0")

    dependence = config["dependence"]
    positive_integer(
        dependence["energy_bins"], "dependence.energy_bins", minimum=2
    )
    positive_integer(
        dependence["score_bins"], "dependence.score_bins", minimum=2
    )
    positive_integer(
        dependence["min_per_bin"], "dependence.min_per_bin", minimum=2
    )
    positive_integer(
        dependence["distance_correlation_max_samples"],
        "dependence.distance_correlation_max_samples",
        minimum=4,
    )
    positive_integer(config["runtime"]["seed"], "runtime.seed", minimum=0)
    if config["dataset"].get("split") == "train":
        raise ValueError(
            "refusing to benchmark on split=train; use a validation or locked test split"
        )
