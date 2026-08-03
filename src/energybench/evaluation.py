"""End-to-end, model-independent evaluation orchestration."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import validate_config
from .data import (
    PredictionBundle,
    duplicate_event_ids,
    evaluation_fingerprint,
    resolve_schema,
)
from .dependence import evaluate_dependence
from .plotting import (
    plot_energy_histograms,
    plot_energy_matched_rocs,
    plot_energy_regression,
    plot_score_energy_dependence,
)
from .regression import evaluate_energy_regression
from .reporting import write_evaluation_results_csv
from .roc import evaluate_energy_matched_roc, weighted_roc_curve
from .utils import (
    file_sha256,
    json_ready,
    prepare_output_directory,
    runtime_versions,
    validate_output_directory,
    write_json,
)


EVALUATION_OUTPUT_ARTIFACTS = (
    "results.csv",
    "energy_matched_roc.png",
    "energy_regression.png",
    "energy_histograms.png",
    "score_energy_dependence.png",
    ".energybench",
    # Legacy EnergyBench artifacts removed only under explicit --allow-existing.
    "report.md",
    "pair_metrics.csv",
    "energy_regression_bins.csv",
    "matching_diagnostics",
)


def _enabled(value: Any, available: bool) -> bool:
    if isinstance(value, str) and value.lower() == "auto":
        return bool(available)
    return bool(value)


def _coerce_positive_label(values: np.ndarray, requested: Any) -> Any:
    unique = np.unique(values)
    matches = [value for value in unique if str(value) == str(requested)]
    if len(matches) == 1:
        return matches[0]
    if requested in unique:
        return requested
    raise ValueError(
        "positive label %r not found; labels are %s"
        % (requested, [item.item() if isinstance(item, np.generic) else item for item in unique])
    )


def _weights(bundle: PredictionBundle, column: Optional[str]) -> np.ndarray:
    if column is None:
        return np.ones(bundle.n_events, dtype=float)
    values = np.asarray(bundle.require(column), dtype=float)
    if values.ndim != 1:
        raise ValueError("sample weight column must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("sample weights contain NaN/Inf")
    if np.any(values < 0):
        raise ValueError("sample weights must be non-negative")
    if not np.any(values > 0):
        raise ValueError("sample weights have zero total mass")
    return values


def _roles(
    bundle: PredictionBundle,
    schema: Mapping[str, Optional[str]],
    config: Mapping[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[str], List[str], List[str]]:
    classification = config["classification"]
    category = (
        None
        if schema.get("category") is None
        else np.asarray(bundle.require(schema["category"])).astype(str)
    )
    warnings = []
    configured_signal = [str(value) for value in classification["signal_categories"]]
    configured_background = [
        str(value) for value in classification["background_categories"]
    ]

    if category is not None and configured_signal:
        positive = np.isin(category, configured_signal)
        if configured_background:
            background = np.isin(category, configured_background)
        else:
            background = ~positive
        keep = positive | background
        excluded = int(np.sum(~keep))
        if excluded:
            warnings.append(
                "%d events have categories outside the manifest pair set and were excluded"
                % excluded
            )
        if schema.get("label") is not None:
            labels = np.asarray(bundle.require(schema["label"]))
            try:
                positive_label = _coerce_positive_label(
                    labels, classification["positive_label"]
                )
                label_positive = labels == positive_label
                disagreements = int(np.sum(keep & (label_positive != positive)))
                if disagreements:
                    warnings.append(
                        "%d events disagree between label and manifest category roles; "
                        "manifest category roles were authoritative" % disagreements
                    )
            except ValueError:
                warnings.append(
                    "positive_label was absent, but explicit signal_categories defined roles"
                )
    elif schema.get("label") is not None:
        labels = np.asarray(bundle.require(schema["label"]))
        positive_label = _coerce_positive_label(
            labels, classification["positive_label"]
        )
        positive = labels == positive_label
        background = ~positive
        keep = np.ones(bundle.n_events, dtype=bool)
    else:
        raise ValueError(
            "classification needs a label column or category plus signal_categories"
        )

    if category is None:
        signal_names = ["positive"]
        background_names = ["negative"]
    else:
        signal_names = (
            configured_signal
            if configured_signal
            else sorted(np.unique(category[positive]).tolist())
        )
        background_names = (
            configured_background
            if configured_background
            else sorted(np.unique(category[background]).tolist())
        )
    binary_label = positive.astype(int)
    return binary_label, category, keep, signal_names, background_names, warnings


def _pair_specs(
    binary_label: np.ndarray,
    category: Optional[np.ndarray],
    keep: np.ndarray,
    signal_names: Sequence[str],
    background_names: Sequence[str],
    pair_mode: str,
) -> List[Dict[str, Any]]:
    specs = []
    if category is not None and pair_mode in {"categories", "both"}:
        for signal in signal_names:
            for background in background_names:
                mask = keep & ((category == signal) | (category == background))
                specs.append(
                    {
                        "name": "%s vs %s" % (signal, background),
                        "signal": signal,
                        "background": background,
                        "kind": "category_pair",
                        "mask": mask,
                    }
                )
    add_pooled = pair_mode == "pooled" or (
        pair_mode == "both" and len(specs) != 1
    )
    if add_pooled or not specs:
        specs.append(
            {
                "name": "all signal vs all background",
                "signal": "all_signal",
                "background": "all_background",
                "kind": "pooled",
                "mask": keep.copy(),
            }
        )
    return specs


def _common_support_roc(
    label: np.ndarray,
    score: np.ndarray,
    energy: np.ndarray,
    weight: np.ndarray,
    support: Optional[Tuple[float, float]],
    target_tpr: float,
) -> Any:
    if support is None:
        return None
    mask = (
        np.isfinite(energy)
        & (energy >= support[0])
        & (energy <= support[1])
    )
    if not np.any(mask & (label == 1)) or not np.any(mask & (label == 0)):
        return None
    return weighted_roc_curve(
        label[mask],
        score[mask],
        weight[mask],
        positive_label=1,
        target_tpr=target_tpr,
    )


def _classification(
    bundle: PredictionBundle,
    schema: Mapping[str, Optional[str]],
    config: Mapping[str, Any],
    weight: np.ndarray,
    output_dir: Path,
) -> Tuple[Dict[str, Any], List[Mapping[str, Any]], List[str]]:
    classification = config["classification"]
    warnings = []
    score_column = schema.get("score")
    role_available = score_column is not None and (
        schema.get("label") is not None
        or (
            schema.get("category") is not None
            and bool(classification["signal_categories"])
        )
    )
    if not _enabled(classification["enabled"], role_available):
        reason = (
            "disabled_by_manifest"
            if role_available
            else "missing_score_or_class_role"
        )
        return {"status": "not_applicable", "reason": reason}, [], warnings
    if not role_available:
        return {
            "status": "not_applicable",
            "reason": "classification requires score and class labels/roles",
        }, [], warnings

    score = np.asarray(bundle.require(score_column), dtype=float)
    if score.ndim != 1:
        raise ValueError("classification score must be one-dimensional")
    if classification["score_direction"] == "lower":
        oriented_score = -score
    else:
        oriented_score = score.copy()

    binary_label, category, keep, signal_names, background_names, role_warnings = _roles(
        bundle, schema, config
    )
    warnings.extend(role_warnings)
    specs = _pair_specs(
        binary_label,
        category,
        keep,
        signal_names,
        background_names,
        classification["pair_mode"],
    )
    energy_column = schema.get("energy_condition")
    energy = (
        None
        if energy_column is None
        else np.asarray(bundle.require(energy_column), dtype=float)
    )
    target = (
        "legacy_uniform"
        if classification["matching_target"] == "uniform"
        else "overlap"
    )
    energy_roi_raw = classification.get("energy_roi")
    energy_roi = (
        None
        if energy_roi_raw is None
        else (float(energy_roi_raw[0]), float(energy_roi_raw[1]))
    )

    pair_reports = []
    runtime_records = []
    for pair_index, spec in enumerate(specs):
        selected = np.asarray(spec["mask"], dtype=bool)
        pair_label = binary_label[selected]
        pair_score = oriented_score[selected]
        pair_weight = weight[selected]
        n_signal = int(np.sum(pair_label == 1))
        n_background = int(np.sum(pair_label == 0))
        base = {
            "name": spec["name"],
            "signal": spec["signal"],
            "background": spec["background"],
            "kind": spec["kind"],
            "n_events": int(np.sum(selected)),
            "n_signal": n_signal,
            "n_background": n_background,
        }
        if n_signal == 0 or n_background == 0:
            pair_reports.append(
                dict(
                    base,
                    metric_status="not_evaluable_missing_class",
                    reason="pair has no signal or no background events",
                    matched_auc=None,
                    inclusive_auc=None,
                )
            )
            continue
        inclusive = weighted_roc_curve(
            pair_label,
            pair_score,
            pair_weight,
            positive_label=1,
            target_tpr=float(classification["target_tpr"]),
        )
        if energy is None:
            pair_reports.append(
                dict(
                    base,
                    metric_status="not_applicable_no_energy_condition",
                    reason="energy_condition column is absent",
                    inclusive_auc=inclusive.auc,
                    inclusive=inclusive.to_dict(),
                    matched_auc=None,
                )
            )
            continue

        pair_energy = energy[selected]
        result = evaluate_energy_matched_roc(
            pair_label,
            pair_score,
            pair_energy,
            pair_weight,
            positive_label=1,
            n_bins=int(classification["energy_bins"]),
            min_per_class=int(classification["min_per_class"]),
            target=target,
            target_tpr=float(classification["target_tpr"]),
            n_bootstrap=int(classification["bootstrap"]),
            confidence_level=float(classification["confidence"]),
            random_state=int(config["runtime"]["seed"]) + pair_index * 1009,
            support_trim_quantile=float(
                classification.get("support_trim_quantile", 0.0)
            ),
            energy_roi=energy_roi,
        )
        valid_bins = sum(1 for item in result.bins if item.valid)
        if result.status != "ok" or result.matched_auc is None:
            metric_status = "not_evaluable_%s" % result.status
            primary_auc = None
            reason = result.reason
        else:
            coverage = min(
                result.coverage.signal_matched_weight_fraction,
                result.coverage.background_matched_weight_fraction,
            )
            exact_single_energy = (
                result.common_support is not None
                and result.common_support[0] == result.common_support[1]
            )
            if coverage < float(classification["min_coverage"]):
                metric_status = "not_evaluable_low_coverage"
                primary_auc = None
                reason = "matched weight coverage %.3f is below %.3f" % (
                    coverage,
                    float(classification["min_coverage"]),
                )
            elif (
                valid_bins < int(classification["min_valid_bins"])
                and not exact_single_energy
            ):
                metric_status = "not_evaluable_too_few_valid_bins"
                primary_auc = None
                reason = "%d valid bins, need %d" % (
                    valid_bins,
                    int(classification["min_valid_bins"]),
                )
            else:
                metric_status = "ok"
                primary_auc = result.matched_auc
                reason = None
        common_roc = _common_support_roc(
            pair_label,
            pair_score,
            pair_energy,
            pair_weight,
            result.common_support,
            float(classification["target_tpr"]),
        )
        common_auc = None if common_roc is None else common_roc.auc
        shortcut_gap = (
            None
            if common_auc is None or result.matched_auc is None
            else float(common_auc - result.matched_auc)
        )
        report = dict(
            base,
            metric_status=metric_status,
            reason=reason,
            matched_auc=primary_auc,
            diagnostic_matched_auc=result.matched_auc,
            inclusive_auc=result.inclusive_auc,
            inclusive_common_support_auc=common_auc,
            shortcut_gap=shortcut_gap,
            valid_bins=valid_bins,
            matching=result.to_dict(),
        )
        pair_reports.append(report)
        runtime_records.append(
            {
                **base,
                "result": result,
                "label": pair_label,
                "score": pair_score,
                "energy": pair_energy,
                "weight": pair_weight,
            }
        )

    atomic = [item for item in pair_reports if item["kind"] == "category_pair"]
    aggregate_source = atomic if atomic else pair_reports
    matched_values = [
        item.get("matched_auc")
        for item in aggregate_source
        if item.get("matched_auc") is not None
    ]
    inclusive_values = [
        item.get("inclusive_auc")
        for item in aggregate_source
        if item.get("inclusive_auc") is not None
    ]
    complete = bool(aggregate_source) and len(matched_values) == len(aggregate_source)
    aggregates = {
        "pair_set": "category_pairs" if atomic else "pooled",
        "n_pairs_expected": len(aggregate_source),
        "n_pairs_evaluable": len(matched_values),
        "complete_pair_set": complete,
        "matched_auc_macro": (
            float(np.mean(matched_values)) if complete else None
        ),
        "matched_auc_macro_available": (
            float(np.mean(matched_values)) if matched_values else None
        ),
        "inclusive_auc_macro": (
            float(np.mean(inclusive_values))
            if len(inclusive_values) == len(aggregate_source) and inclusive_values
            else None
        ),
    }
    status = "ok" if pair_reports else "not_evaluable"
    return {
        "status": status,
        "score_column": score_column,
        "score_direction": classification["score_direction"],
        "score_space": classification["score_space"],
        "energy_condition_column": energy_column,
        "pairs": pair_reports,
        "aggregates": aggregates,
    }, runtime_records, warnings


def _regression(
    bundle: PredictionBundle,
    schema: Mapping[str, Optional[str]],
    config: Mapping[str, Any],
    weight: np.ndarray,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    regression = config["regression"]
    true_column = schema.get("energy_true")
    pred_column = schema.get("energy_pred")
    available = true_column is not None and pred_column is not None
    if not _enabled(regression["enabled"], available):
        return {
            "status": "not_applicable",
            "reason": "disabled_by_manifest" if available else "missing_energy_target_or_prediction",
        }, None
    if not available:
        return {
            "status": "not_applicable",
            "reason": "energy regression needs energy_target and energy_pred",
        }, None
    truth = np.asarray(bundle.require(true_column), dtype=float)
    prediction = np.asarray(bundle.require(pred_column), dtype=float)
    result = evaluate_energy_regression(
        truth,
        prediction,
        weight,
        n_histogram_bins=int(regression["histogram_bins"]),
        histogram_edges=regression.get("histogram_edges"),
        n_energy_bins=int(regression["performance_bins"]),
        fractional_energy_floor=regression.get("energy_floor"),
        n_bootstrap=int(regression["bootstrap"]),
        confidence=float(regression["confidence"]),
        random_state=int(config["runtime"]["seed"]),
    )
    payload = result.to_dict()
    payload["energy_target_column"] = true_column
    payload["energy_pred_column"] = pred_column
    return payload, result


def _quality(
    bundle: PredictionBundle,
    schema: Mapping[str, Optional[str]],
    config: Mapping[str, Any],
    strict: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    warnings = []
    errors = []
    duplicate_count = None
    if schema.get("event_id") is None:
        warnings.append(
            "event_id is missing; cross-model alignment and duplicate checks are unavailable"
        )
        if strict:
            errors.append("strict mode requires event_id")
    else:
        duplicate_count = duplicate_event_ids(bundle.require(schema["event_id"]))
        event_ids = np.asarray(bundle.require(schema["event_id"])).astype(str)
        empty_ids = int(np.sum(np.char.strip(event_ids) == ""))
        if empty_ids:
            warnings.append("%d event IDs are empty" % empty_ids)
            if strict:
                errors.append("strict mode forbids empty event IDs")
        if duplicate_count:
            warnings.append("%d duplicate event IDs" % duplicate_count)
            if strict:
                errors.append("strict mode forbids duplicate event IDs")

    split_values = None
    if schema.get("split") is None:
        warnings.append(
            "split column is missing; event-level evaluation split cannot be verified"
        )
        if strict:
            errors.append("strict mode requires a split column")
    else:
        split_values = sorted(
            np.unique(np.asarray(bundle.require(schema["split"])).astype(str)).tolist()
        )
        expected = str(config["dataset"]["split"])
        unexpected = [value for value in split_values if value != expected]
        if unexpected:
            warnings.append(
                "prediction split values %s do not match manifest split %r"
                % (split_values, expected)
            )
            if strict:
                errors.append("strict mode requires one matching split")
    selection_split = config["dataset"].get("selection_split")
    if not selection_split:
        warnings.append(
            "selection_split is missing; checkpoint-selection independence is not declared"
        )
        if strict:
            errors.append("strict mode requires selection_split")
    elif selection_split == config["dataset"].get("split"):
        warnings.append(
            "selection_split equals evaluation split; reported performance has checkpoint-selection bias"
        )
        if strict:
            errors.append("strict mode requires selection and evaluation splits to differ")
    if schema.get("score") is not None:
        score = np.asarray(bundle.require(schema["score"]), dtype=float)
        nonfinite_score = int(np.sum(~np.isfinite(score)))
        if nonfinite_score:
            warnings.append(
                "%d classifier scores are NaN/Inf and will not enter ROC" % nonfinite_score
            )
            if strict:
                errors.append("strict mode requires finite classifier scores")
        if config["classification"].get("score_space") == "probability":
            outside = int(np.sum(np.isfinite(score) & ((score < 0) | (score > 1))))
            if outside:
                warnings.append(
                    "%d scores are outside [0,1] despite score_space=probability"
                    % outside
                )
                if strict:
                    errors.append("strict probability scores must be in [0,1]")
    if schema.get("energy_condition") is not None:
        condition = np.asarray(
            bundle.require(schema["energy_condition"]), dtype=float
        )
        nonfinite_condition = int(np.sum(~np.isfinite(condition)))
        if nonfinite_condition:
            warnings.append(
                "%d energy_condition values are NaN/Inf; pair coverage will decrease"
                % nonfinite_condition
            )
    decorrelation = bundle.metadata.get("decorrelation")
    uses_decorrelated_score = (
        str(config["classification"].get("score_space", "")).lower()
        == "conditional_quantile"
        or (
            isinstance(decorrelation, Mapping)
            and schema.get("score")
            == decorrelation.get("output_score_column")
        )
    )
    if uses_decorrelated_score:
        verification_status = (
            None
            if not isinstance(decorrelation, Mapping)
            else decorrelation.get("verification_status")
        )
        if verification_status != "verified_disjoint":
            warnings.append(
                "decorrelated score lacks verified-disjoint calibration/test provenance"
            )
            if strict:
                errors.append(
                    "strict mode requires verified-disjoint decorrelation provenance"
                )
    if strict:
        explicit_columns = config.get("columns", {})
        required_explicit = ["event_id", "split"]
        score_available = schema.get("score") is not None and (
            schema.get("label") is not None
            or (
                schema.get("category") is not None
                and bool(config["classification"]["signal_categories"])
            )
        )
        classification_enabled = _enabled(
            config["classification"]["enabled"], score_available
        )
        regression_available = (
            schema.get("energy_true") is not None
            and schema.get("energy_pred") is not None
        )
        regression_enabled = _enabled(
            config["regression"]["enabled"], regression_available
        )
        if classification_enabled and score_available:
            required_explicit.append("score")
            if explicit_columns.get("category"):
                required_explicit.append("category")
            else:
                required_explicit.append("label")
            if schema.get("energy_condition") is not None:
                required_explicit.append("energy_condition")
        if regression_enabled and regression_available:
            required_explicit.extend(["energy_true", "energy_pred"])
        missing_explicit = sorted(
            {
                role
                for role in required_explicit
                if not explicit_columns.get(role)
            }
        )
        if missing_explicit:
            errors.append(
                "strict mode requires explicit column mappings: %s"
                % ", ".join(missing_explicit)
            )

        required_provenance = (
            "experiment",
            "dataset_id",
            "dataset_version",
            "energy_unit",
        )
        missing = [
            key
            for key in required_provenance
            if str(config["dataset"].get(key, "unspecified")) in {"", "unspecified", "None"}
        ]
        if missing:
            errors.append(
                "strict mode requires dataset provenance: %s" % ", ".join(missing)
            )
        semantic_missing = []
        if (
            classification_enabled
            and score_available
            and schema.get("energy_condition") is not None
            and str(
                config["dataset"].get(
                    "energy_condition_kind", "unspecified"
                )
            )
            in {"", "unspecified", "None"}
        ):
            semantic_missing.append("energy_condition_kind")
        if (
            regression_enabled
            and regression_available
            and str(
                config["dataset"].get("energy_target_kind", "unspecified")
            )
            in {"", "unspecified", "None"}
        ):
            semantic_missing.append("energy_target_kind")
        if (
            classification_enabled
            and score_available
            and str(
                config["classification"].get("score_space", "unspecified")
            )
            in {"", "unspecified", "None"}
        ):
            semantic_missing.append("classification.score_space")
        if semantic_missing:
            errors.append(
                "strict mode requires task semantics: %s"
                % ", ".join(semantic_missing)
            )
    return {
        "n_events": bundle.n_events,
        "duplicate_event_ids": duplicate_count,
        "split_values": split_values,
        "warnings": warnings,
        "errors": errors,
        "strict": bool(strict),
    }, errors


def _fingerprint(
    bundle: PredictionBundle, schema: Mapping[str, Optional[str]]
) -> str:
    truth_roles = (
        "event_id",
        "label",
        "category",
        "energy_condition",
        "energy_true",
        "sample_weight",
        "group_id",
        "split",
        "experiment",
        "dataset_id",
        "dataset_version",
        "task_id",
    )
    columns = {
        role: bundle.require(schema[role])
        for role in truth_roles
        if role in schema and schema.get(role) is not None
    }
    return evaluation_fingerprint(columns, event_id_name="event_id")


def _evaluator_code_fingerprint() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _protocol_fingerprint(config: Mapping[str, Any]) -> str:
    protocol = {
        "schema_version": config.get("schema_version"),
        "task_id": config.get("task_id"),
        "dataset": config.get("dataset"),
        "classification": config.get("classification"),
        "regression": config.get("regression"),
        "dependence": config.get("dependence"),
        "seed": config.get("runtime", {}).get("seed"),
    }
    encoded = json.dumps(
        json_ready(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_evaluation(
    bundle: PredictionBundle,
    config: Mapping[str, Any],
    output_dir: Any,
    strict: bool = False,
    allow_existing: bool = False,
) -> Dict[str, Any]:
    """Run all applicable tasks, write machine-readable artifacts and plots."""

    validate_config(config)
    destination = Path(output_dir).expanduser().resolve()
    validate_output_directory(
        destination, allow_existing, EVALUATION_OUTPUT_ARTIFACTS
    )
    schema = resolve_schema(bundle, config.get("columns"))
    quality, strict_errors = _quality(bundle, schema, config, strict)
    if strict_errors:
        raise ValueError("; ".join(strict_errors))
    weight = _weights(bundle, schema.get("sample_weight"))

    classification, pair_runtime, class_warnings = _classification(
        bundle, schema, config, weight, destination
    )
    quality["warnings"].extend(class_warnings)
    regression, regression_runtime = _regression(bundle, schema, config, weight)

    dependence = {
        "status": "not_applicable",
        "reason": "classification score and energy_condition are required",
    }
    threshold = None
    for record in pair_runtime:
        if record["kind"] == "pooled" and record["result"].matched is not None:
            point = record["result"].matched.operating_point
            if point is not None:
                threshold = point.threshold
                break
    if threshold is None:
        for record in pair_runtime:
            if record["result"].matched is not None:
                point = record["result"].matched.operating_point
                if point is not None:
                    threshold = point.threshold
                    break
    if (
        config["dependence"]["enabled"]
        and schema.get("score") is not None
        and schema.get("energy_condition") is not None
        and classification.get("status") == "ok"
    ):
        binary_label, category, keep, _, _, _ = _roles(bundle, schema, config)
        score = np.asarray(bundle.require(schema["score"]), dtype=float)
        if config["classification"]["score_direction"] == "lower":
            score = -score
        energy = np.asarray(bundle.require(schema["energy_condition"]), dtype=float)
        dependence = evaluate_dependence(
            score[keep],
            energy[keep],
            binary_label[keep],
            None if category is None else category[keep],
            weight[keep],
            n_energy_bins=int(config["dependence"]["energy_bins"]),
            n_score_bins=int(config["dependence"]["score_bins"]),
            threshold=threshold,
            min_per_bin=int(config["dependence"]["min_per_bin"]),
            distance_correlation_max_samples=int(
                config["dependence"]["distance_correlation_max_samples"]
            ),
            seed=int(config["runtime"]["seed"]),
        )
        dependence["score_space"] = config["classification"]["score_space"]
        dependence["score_column"] = schema["score"]
        dependence["threshold_source"] = (
            "matched_roc_target_tpr" if threshold is not None else None
        )

    source_info = {
        "path": None if bundle.source is None else str(bundle.source),
        "sha256": (
            None
            if bundle.source is None or not bundle.source.is_file()
            else file_sha256(bundle.source)
        ),
        "metadata": bundle.metadata,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "evaluator": {
            "name": "energybench",
            "version": "0.1.0",
            "code_fingerprint": _evaluator_code_fingerprint(),
            "runtime_versions": runtime_versions(),
        },
        "task_id": config["task_id"],
        "model_id": config["model_id"],
        "dataset": config["dataset"],
        "protocol_fingerprint": _protocol_fingerprint(config),
        "evaluation_fingerprint": _fingerprint(bundle, schema),
        "input": source_info,
        "resolved_columns": schema,
        "quality": quality,
        "classification": classification,
        "energy_regression": regression,
        "energy_dependence": dependence,
        "artifacts": {},
    }

    prepare_output_directory(
        destination, allow_existing, EVALUATION_OUTPUT_ARTIFACTS
    )
    artifacts = {"results_csv": "results.csv"}

    if config["runtime"]["make_plots"]:
        if pair_runtime:
            plot_energy_matched_rocs(
                pair_runtime,
                destination / "energy_matched_roc.png",
                str(config["model_id"]),
            )
            artifacts["energy_matched_roc"] = "energy_matched_roc.png"
        if dependence.get("status") == "ok":
            path = destination / "score_energy_dependence.png"
            plot_score_energy_dependence(
                dependence,
                path,
                str(config["dataset"]["energy_unit"]),
                str(config["classification"]["score_space"]),
            )
            if path.is_file():
                artifacts["score_energy_dependence"] = path.name
        if regression_runtime is not None:
            truth = bundle.require(schema["energy_true"])
            prediction = bundle.require(schema["energy_pred"])
            path = destination / "energy_regression.png"
            plot_energy_regression(
                truth,
                prediction,
                weight,
                regression_runtime,
                path,
                str(config["dataset"]["energy_unit"]),
                str(config["model_id"]),
            )
            if path.is_file():
                artifacts["energy_regression"] = path.name
            hist_path = destination / "energy_histograms.png"
            plot_energy_histograms(
                regression_runtime,
                hist_path,
                str(config["dataset"]["energy_unit"]),
                str(config["model_id"]),
            )
            if hist_path.is_file():
                artifacts["energy_histograms"] = hist_path.name
    report["artifacts"] = artifacts
    ready_report = json_ready(report)
    write_evaluation_results_csv(ready_report, destination / "results.csv")
    internal = destination / ".energybench"
    write_json(internal / "resolved_manifest.json", config)
    write_json(internal / "metrics.json", ready_report)
    return json_ready(report)
