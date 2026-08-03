"""CSV result tables and multi-model comparison tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .utils import (
    prepare_output_directory,
    read_json,
    validate_output_directory,
)


COMPARISON_OUTPUT_ARTIFACTS = (
    "leaderboard.csv",
    ".energybench",
    # Legacy comparison artifacts removed only under explicit --allow-existing.
    "leaderboard.md",
    "comparison.json",
)


EVALUATION_RESULT_FIELDS = [
    "results_schema_version",
    "model_id",
    "task_id",
    "experiment",
    "dataset_id",
    "dataset_version",
    "split",
    "classification_status",
    "classification_reason",
    "matched_auc_status",
    "matched_auc_reason",
    "matched_auc_macro",
    "matched_auc_macro_available",
    "inclusive_auc_macro",
    "matched_pairs_evaluable",
    "matched_pairs_expected",
    "complete_pair_set",
    "energy_regression_status",
    "energy_regression_reason",
    "energy_regression_score_name",
    "energy_regression_score",
    "energy_regression_ci_low",
    "energy_regression_ci_high",
    "energy_regression_ci_level",
    "energy_regression_bootstrap_requested",
    "energy_regression_bootstrap_successful",
    "event_score",
    "histogram_similarity",
    "histogram_overlap",
    "jsd_bits",
    "wasserstein_1",
    "mae",
    "rmse",
    "bias",
    "r2",
    "fractional_bias",
    "fractional_resolution_68",
    "balanced_fractional_mae",
    "finite_fraction",
    "energy_dependence_status",
    "energy_dependence_reason",
    "energy_independence_score_status",
    "energy_independence_score_reason",
    "dependence_groups_evaluable",
    "dependence_groups_expected",
    "complete_dependence_group_set",
    "energy_independence_score_mean",
    "energy_independence_score_worst",
    "score_column",
    "score_space",
    "score_direction",
    "energy_condition_kind",
    "energy_target_kind",
    "energy_unit",
    "n_events",
    "strict",
    "warning_count",
    "error_count",
    "evaluation_fingerprint",
    "protocol_fingerprint",
    "code_fingerprint",
    "input_sha256",
    "created_at_utc",
    "source_schema_version",
]


def _matched_auc_status(
    classification: Mapping[str, Any], aggregates: Mapping[str, Any]
) -> tuple:
    if aggregates.get("matched_auc_macro") is not None:
        return "ok", None
    classification_status = classification.get("status", "not_applicable")
    if classification_status != "ok":
        return classification_status, classification.get("reason")
    expected = int(aggregates.get("n_pairs_expected") or 0)
    evaluable = int(aggregates.get("n_pairs_evaluable") or 0)
    if expected == 0:
        return "not_evaluable_no_pairs", "no classification pairs were defined"
    pair_reasons = sorted(
        {
            str(pair.get("reason"))
            for pair in classification.get("pairs", [])
            if pair.get("reason")
        }
    )
    reason = "%d/%d fixed pairs are evaluable" % (evaluable, expected)
    if pair_reasons:
        reason += "; " + "; ".join(pair_reasons)
    return "not_evaluable_incomplete_pair_set", reason


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _dependence_score_status(
    dependence: Mapping[str, Any]
) -> tuple:
    groups = dependence.get("groups", {})
    expected = len(groups) if isinstance(groups, Mapping) else 0
    evaluable = (
        sum(
            1
            for values in groups.values()
            if isinstance(values, Mapping) and values.get("status") == "ok"
        )
        if isinstance(groups, Mapping)
        else 0
    )
    raw_status = dependence.get("status", "not_applicable")
    if raw_status != "ok":
        return raw_status, dependence.get("reason"), evaluable, expected, False
    if evaluable != expected:
        return (
            "not_evaluable_incomplete_group_set",
            "%d/%d class-conditional groups are evaluable" % (evaluable, expected),
            evaluable,
            expected,
            False,
        )
    return "ok", None, evaluable, expected, True


def _competition_ranks(
    rows: Sequence[Mapping[str, Any]],
    score_field: str,
    eligible,
) -> Dict[int, int]:
    """Return descending competition ranks; tied scores receive the same rank."""

    indices = [
        index
        for index, row in enumerate(rows)
        if eligible(row) and row.get(score_field) is not None
    ]
    indices.sort(key=lambda index: float(rows[index][score_field]), reverse=True)
    ranks = {}
    previous_value = None
    previous_rank = None
    for position, index in enumerate(indices, 1):
        value = float(rows[index][score_field])
        if previous_value is not None and value == previous_value:
            rank = previous_rank
        else:
            rank = position
        ranks[index] = rank
        previous_value = value
        previous_rank = rank
    return ranks


def write_evaluation_results_csv(
    report: Mapping[str, Any], path: Path
) -> None:
    """Write the single user-facing, one-row summary table for an evaluation."""

    classification = report.get("classification", {})
    aggregates = classification.get("aggregates", {})
    regression = report.get("energy_regression", {})
    dependence = report.get("energy_dependence", {})
    dataset = report.get("dataset", {})
    quality = report.get("quality", {})
    regression_ci = regression.get("bootstrap", {}).get(
        "energy_regression_score", {}
    )
    matched_status, matched_reason = _matched_auc_status(
        classification, aggregates
    )
    (
        dependence_score_status,
        dependence_score_reason,
        dependence_groups_evaluable,
        dependence_groups_expected,
        complete_dependence_group_set,
    ) = _dependence_score_status(dependence)
    row = {
        "results_schema_version": 1,
        "model_id": report.get("model_id"),
        "task_id": report.get("task_id"),
        "experiment": dataset.get("experiment"),
        "dataset_id": dataset.get("dataset_id"),
        "dataset_version": dataset.get("dataset_version"),
        "split": dataset.get("split"),
        "classification_status": classification.get("status"),
        "classification_reason": classification.get("reason"),
        "matched_auc_status": matched_status,
        "matched_auc_reason": matched_reason,
        "matched_auc_macro": aggregates.get("matched_auc_macro"),
        "matched_auc_macro_available": aggregates.get(
            "matched_auc_macro_available"
        ),
        "inclusive_auc_macro": aggregates.get("inclusive_auc_macro"),
        "matched_pairs_evaluable": aggregates.get("n_pairs_evaluable"),
        "matched_pairs_expected": aggregates.get("n_pairs_expected"),
        "complete_pair_set": aggregates.get("complete_pair_set"),
        "energy_regression_status": regression.get("status"),
        "energy_regression_reason": regression.get("reason"),
        "energy_regression_score_name": regression.get("score_name"),
        "energy_regression_score": regression.get("energy_regression_score"),
        "energy_regression_ci_low": regression_ci.get("lower"),
        "energy_regression_ci_high": regression_ci.get("upper"),
        "energy_regression_ci_level": regression_ci.get("confidence"),
        "energy_regression_bootstrap_requested": regression_ci.get(
            "n_requested"
        ),
        "energy_regression_bootstrap_successful": regression_ci.get(
            "n_successful"
        ),
        "event_score": regression.get("event_score"),
        "histogram_similarity": regression.get("histogram_similarity"),
        "histogram_overlap": regression.get("histogram_overlap"),
        "jsd_bits": regression.get("jsd_bits"),
        "wasserstein_1": regression.get("wasserstein_1"),
        "mae": regression.get("mae"),
        "rmse": regression.get("rmse"),
        "bias": regression.get("bias"),
        "r2": regression.get("r2"),
        "fractional_bias": regression.get("fractional_bias"),
        "fractional_resolution_68": regression.get(
            "fractional_resolution_68"
        ),
        "balanced_fractional_mae": regression.get(
            "balanced_fractional_mae"
        ),
        "finite_fraction": regression.get("finite_fraction"),
        "energy_dependence_status": dependence.get("status"),
        "energy_dependence_reason": dependence.get("reason"),
        "energy_independence_score_status": dependence_score_status,
        "energy_independence_score_reason": dependence_score_reason,
        "dependence_groups_evaluable": dependence_groups_evaluable,
        "dependence_groups_expected": dependence_groups_expected,
        "complete_dependence_group_set": complete_dependence_group_set,
        "energy_independence_score_mean": (
            dependence.get("overall_energy_independence_score")
            if dependence_score_status == "ok"
            else None
        ),
        "energy_independence_score_worst": (
            dependence.get("worst_group_energy_independence_score")
            if dependence_score_status == "ok"
            else None
        ),
        "score_column": classification.get("score_column"),
        "score_space": classification.get("score_space"),
        "score_direction": classification.get("score_direction"),
        "energy_condition_kind": dataset.get("energy_condition_kind"),
        "energy_target_kind": dataset.get("energy_target_kind"),
        "energy_unit": dataset.get("energy_unit"),
        "n_events": quality.get("n_events"),
        "strict": quality.get("strict"),
        "warning_count": len(quality.get("warnings", [])),
        "error_count": len(quality.get("errors", [])),
        "evaluation_fingerprint": report.get("evaluation_fingerprint"),
        "protocol_fingerprint": report.get("protocol_fingerprint"),
        "code_fingerprint": report.get("evaluator", {}).get(
            "code_fingerprint"
        ),
        "input_sha256": report.get("input", {}).get("sha256"),
        "created_at_utc": report.get("created_at_utc"),
        "source_schema_version": report.get("schema_version"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_RESULT_FIELDS)
        writer.writeheader()
        writer.writerow(
            {field: _csv_value(row.get(field)) for field in EVALUATION_RESULT_FIELDS}
        )


def compare_evaluations(
    metric_paths: Sequence[Path],
    output_dir: Path,
    allow_mixed_data: bool = False,
    allow_existing: bool = False,
) -> Dict[str, Any]:
    validate_output_directory(
        output_dir, allow_existing, COMPARISON_OUTPUT_ARTIFACTS
    )
    reports = [read_json(path) for path in metric_paths]
    missing_fingerprints = [
        index
        for index, report in enumerate(reports)
        if not report.get("evaluation_fingerprint")
        or not report.get("protocol_fingerprint")
        or not report.get("evaluator", {}).get("code_fingerprint")
    ]
    if missing_fingerprints and not allow_mixed_data:
        raise ValueError(
            "one or more evaluations lack evaluation/protocol/code fingerprints; "
            "refusing an unverifiable leaderboard. Pass --allow-mixed-data "
            "only for an explicitly non-comparative inventory."
        )
    non_strict = [
        index
        for index, report in enumerate(reports)
        if report.get("quality", {}).get("strict") is not True
        or bool(report.get("quality", {}).get("errors"))
    ]
    if non_strict and not allow_mixed_data:
        raise ValueError(
            "one or more evaluations were not produced by a successful strict "
            "run; refusing a formal leaderboard. Pass --allow-mixed-data only "
            "for an explicitly non-comparative inventory."
        )
    fingerprints = sorted(
        {str(report.get("evaluation_fingerprint")) for report in reports}
    )
    protocols = sorted(
        {str(report.get("protocol_fingerprint")) for report in reports}
    )
    code_fingerprints = sorted(
        {
            str(report.get("evaluator", {}).get("code_fingerprint"))
            for report in reports
        }
    )
    if (
        len(fingerprints) > 1
        or len(protocols) > 1
        or len(code_fingerprints) > 1
    ) and not allow_mixed_data:
        raise ValueError(
            "evaluations use different event/truth, protocol, or code fingerprints; "
            "refusing an invalid leaderboard. Pass --allow-mixed-data only for "
            "an explicitly non-comparative inventory."
        )
    model_ids = [str(report.get("model_id") or "").strip() for report in reports]
    if not allow_mixed_data:
        if any(not model_id for model_id in model_ids):
            raise ValueError("formal comparison requires a non-empty model_id")
        if len(set(model_ids)) != len(model_ids):
            raise ValueError(
                "formal comparison requires unique model_id values; "
                "duplicate model IDs were supplied"
            )
    rows = []
    for path, report in zip(metric_paths, reports):
        classification = report.get("classification", {})
        aggregates = classification.get("aggregates", {})
        regression = report.get("energy_regression", {})
        dependence = report.get("energy_dependence", {})
        dataset = report.get("dataset", {})
        quality = report.get("quality", {})
        matched_status, matched_reason = _matched_auc_status(
            classification, aggregates
        )
        (
            dependence_score_status,
            dependence_score_reason,
            dependence_groups_evaluable,
            dependence_groups_expected,
            complete_dependence_group_set,
        ) = _dependence_score_status(dependence)
        rows.append(
            {
                "model_id": report.get("model_id"),
                "task_id": report.get("task_id"),
                "experiment": dataset.get("experiment"),
                "dataset_id": dataset.get("dataset_id"),
                "dataset_version": dataset.get("dataset_version"),
                "split": dataset.get("split"),
                "energy_target_kind": dataset.get("energy_target_kind"),
                "energy_unit": dataset.get("energy_unit"),
                "evaluation_fingerprint": report.get("evaluation_fingerprint"),
                "protocol_fingerprint": report.get("protocol_fingerprint"),
                "code_fingerprint": report.get("evaluator", {}).get(
                    "code_fingerprint"
                ),
                "classification_status": classification.get("status"),
                "classification_reason": classification.get("reason"),
                "matched_auc_status": matched_status,
                "matched_auc_reason": matched_reason,
                "matched_auc_macro": aggregates.get("matched_auc_macro"),
                "matched_auc_macro_available": aggregates.get(
                    "matched_auc_macro_available"
                ),
                "matched_pairs_evaluable": aggregates.get("n_pairs_evaluable"),
                "matched_pairs_expected": aggregates.get("n_pairs_expected"),
                "complete_pair_set": aggregates.get("complete_pair_set"),
                "inclusive_auc_macro": aggregates.get("inclusive_auc_macro"),
                "energy_regression_status": regression.get("status"),
                "energy_regression_reason": regression.get("reason"),
                "energy_regression_score_name": regression.get("score_name"),
                "energy_regression_score": regression.get(
                    "energy_regression_score"
                ),
                "histogram_similarity": regression.get("histogram_similarity"),
                "mae": regression.get("mae"),
                "rmse": regression.get("rmse"),
                "energy_independence_score_mean": (
                    dependence.get("overall_energy_independence_score")
                    if dependence_score_status == "ok"
                    else None
                ),
                "energy_independence_score_worst": (
                    dependence.get("worst_group_energy_independence_score")
                    if dependence_score_status == "ok"
                    else None
                ),
                "energy_dependence_status": dependence.get("status"),
                "energy_dependence_reason": dependence.get("reason"),
                "energy_independence_score_status": dependence_score_status,
                "energy_independence_score_reason": dependence_score_reason,
                "dependence_groups_evaluable": dependence_groups_evaluable,
                "dependence_groups_expected": dependence_groups_expected,
                "complete_dependence_group_set": complete_dependence_group_set,
                "strict": quality.get("strict"),
                "warning_count": len(quality.get("warnings", [])),
                "error_count": len(quality.get("errors", [])),
                "source_metrics": str(path),
            }
        )
    comparison_mode = (
        "non_comparative_inventory" if allow_mixed_data else "formal"
    )
    if allow_mixed_data:
        classification_ranks = {}
        regression_ranks = {}
        ordered_indices = list(range(len(rows)))
    else:
        classification_ranks = _competition_ranks(
            rows,
            "matched_auc_macro",
            lambda row: row.get("matched_auc_status") == "ok",
        )
        regression_ranks = _competition_ranks(
            rows,
            "energy_regression_score",
            lambda row: row.get("energy_regression_status")
            in {"ok", "no_finite_predictions"},
        )
        if classification_ranks:
            ordered_indices = sorted(
                range(len(rows)),
                key=lambda index: (
                    classification_ranks.get(index) is None,
                    classification_ranks.get(index, float("inf")),
                    str(rows[index].get("model_id") or ""),
                ),
            )
        elif regression_ranks:
            ordered_indices = sorted(
                range(len(rows)),
                key=lambda index: (
                    regression_ranks.get(index) is None,
                    regression_ranks.get(index, float("inf")),
                    str(rows[index].get("model_id") or ""),
                ),
            )
        else:
            ordered_indices = sorted(
                range(len(rows)),
                key=lambda index: str(rows[index].get("model_id") or ""),
            )
    output_rows = []
    for index in ordered_indices:
        row = rows[index]
        output_rows.append(
            {
                "classification_rank": classification_ranks.get(index, ""),
                "energy_regression_rank": regression_ranks.get(index, ""),
                "comparison_mode": comparison_mode,
                "comparable": not allow_mixed_data,
                **row,
            }
        )
    prepare_output_directory(
        output_dir, allow_existing, COMPARISON_OUTPUT_ARTIFACTS
    )
    csv_path = output_dir / "leaderboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]) if output_rows else []
        )
        if output_rows:
            writer.writeheader()
            writer.writerows(output_rows)
    summary = {
        "comparison_mode": comparison_mode,
        "allow_mixed_data": bool(allow_mixed_data),
        "n_evaluations": len(rows),
        "fingerprints": fingerprints,
        "protocol_fingerprints": protocols,
        "code_fingerprints": code_fingerprints,
        "mixed_data_protocol_or_code": (
            len(fingerprints) > 1
            or len(protocols) > 1
            or len(code_fingerprints) > 1
        ),
        "mixed_data_or_protocol": (
            len(fingerprints) > 1 or len(protocols) > 1
        ),
        "missing_fingerprint_input_indices": missing_fingerprints,
        "non_strict_input_indices": non_strict,
        "rows": output_rows,
    }
    return summary
