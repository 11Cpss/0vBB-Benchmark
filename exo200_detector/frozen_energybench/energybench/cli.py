"""Command-line interface for EnergyBench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .adapters import parse_adapter_arguments, run_adapter
from .config import apply_overrides, load_manifest
from .data import (
    PredictionBundle,
    duplicate_event_ids,
    load_bundle,
    resolve_column,
    resolve_schema,
    save_bundle,
)
from .decorrelation import BackgroundQuantileDecorrelator
from .evaluation import run_evaluation
from .reporting import compare_evaluations
from .utils import file_sha256, json_ready, write_json


def _add_column_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--event-id-column")
    parser.add_argument("--label-column")
    parser.add_argument("--score-column")
    parser.add_argument("--energy-condition-column")
    parser.add_argument("--energy-target-column")
    parser.add_argument("--energy-pred-column")
    parser.add_argument("--category-column")
    parser.add_argument("--weight-column")
    parser.add_argument("--group-column")
    parser.add_argument("--split-column")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="energybench",
        description=(
            "Model-agnostic energy-matched ROC and energy-regression evaluation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick NEXT workflow (keeps all expert commands available):\n"
            "  energybench next\n"
            "  energybench next path/to/checkpoint.pt\n"
            "  energybench next path/to/checkpoint.pt --dry-run\n"
            "\n"
            "Use `energybench COMMAND --help` for advanced options."
        ),
    )
    parser.add_argument("--version", action="version", version="energybench 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    next_model = subparsers.add_parser(
        "next",
        help="one-command NEXT checkpoint prediction and evaluation",
        description=(
            "一条命令完成 NEXT checkpoint 的预测、输入检查和严格评测。"
            "不传 checkpoint 时会显示本项目中的可选模型。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    next_model.add_argument(
        "checkpoint",
        nargs="?",
        help="checkpoint .pt path; omit to choose from 02_models/checkpoints",
    )
    next_model.add_argument(
        "--data",
        help="NEXT dataset root (normally read from the checkpoint)",
    )
    next_model.add_argument(
        "--output-dir",
        help="run directory (default: a new non-conflicting directory)",
    )
    next_model.add_argument(
        "--model-id",
        help="result model ID (default: derived from checkpoint name)",
    )
    next_model.add_argument(
        "--manifest",
        help="task manifest (normally selected automatically for v1/v2)",
    )
    next_model.add_argument("--device", default="cuda:0", help="cpu, cuda, cuda:0, ...")
    next_model.add_argument("--batch-size", type=int, default=32)
    next_model.add_argument("--num-workers", type=int, default=0)
    next_model.add_argument("--split", default="test")
    next_model.add_argument(
        "--max-files-per-class",
        type=int,
        default=0,
        help="0 uses every selected file; a small value is useful for a smoke run",
    )
    next_model.add_argument("--no-plots", action="store_true")
    next_model.add_argument(
        "--dry-run",
        action="store_true",
        help="show the resolved plan without inference or filesystem writes",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a canonical prediction table"
    )
    evaluate.add_argument("input", help=".npz/.csv/.h5/.parquet prediction table")
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--manifest", help="JSON/YAML task manifest")
    evaluate.add_argument("--model-id")
    evaluate.add_argument("--task-id")
    evaluate.add_argument("--experiment")
    evaluate.add_argument("--dataset-id")
    evaluate.add_argument("--dataset-version")
    evaluate.add_argument("--split")
    evaluate.add_argument("--selection-split")
    evaluate.add_argument("--energy-unit")
    evaluate.add_argument("--energy-condition-kind")
    evaluate.add_argument("--energy-target-kind")
    evaluate.add_argument(
        "--tasks", choices=["auto", "classification", "regression", "both"]
    )
    _add_column_arguments(evaluate)
    evaluate.add_argument("--positive-label")
    evaluate.add_argument("--signal-category", action="append")
    evaluate.add_argument("--background-category", action="append")
    evaluate.add_argument(
        "--pair-mode", choices=["pooled", "categories", "both"]
    )
    evaluate.add_argument("--score-direction", choices=["higher", "lower"])
    evaluate.add_argument(
        "--score-space",
        help="probability, logit, rank, or another explicitly named space",
    )
    evaluate.add_argument("--energy-bins", type=int)
    evaluate.add_argument("--matching-target", choices=["overlap", "uniform"])
    evaluate.add_argument("--min-per-class", type=int)
    evaluate.add_argument("--min-valid-bins", type=int)
    evaluate.add_argument("--min-coverage", type=float)
    evaluate.add_argument("--support-trim-quantile", type=float)
    evaluate.add_argument("--energy-roi", nargs=2, type=float, metavar=("LOW", "HIGH"))
    evaluate.add_argument("--target-tpr", type=float)
    evaluate.add_argument("--bootstrap", type=int)
    evaluate.add_argument("--roc-bootstrap", type=int)
    evaluate.add_argument("--regression-bootstrap", type=int)
    evaluate.add_argument("--histogram-bins", type=int)
    evaluate.add_argument("--performance-bins", type=int)
    evaluate.add_argument("--energy-floor", type=float)
    evaluate.add_argument("--dependence-energy-bins", type=int)
    evaluate.add_argument("--dependence-score-bins", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--no-plots", action="store_true")
    evaluate.add_argument(
        "--strict",
        action="store_true",
        help=(
            "require explicit semantic column mappings, provenance, unique "
            "event IDs, and independent selection/evaluation splits"
        ),
    )
    evaluate.add_argument(
        "--allow-existing",
        action="store_true",
        help="allow writing known artifacts into a non-empty output directory",
    )

    inspect = subparsers.add_parser(
        "inspect", help="show columns, metadata, inferred roles, and basic quality"
    )
    inspect.add_argument("input")
    _add_column_arguments(inspect)

    predict = subparsers.add_parser(
        "predict", help="run trusted user adapter and export canonical NPZ"
    )
    predict.add_argument("--adapter", required=True, help="MODULE:FUNCTION")
    predict.add_argument("--model", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument(
        "--adapter-arg", action="append", default=[], metavar="KEY=VALUE"
    )

    decorrelate = subparsers.add_parser(
        "decorrelate",
        help="fit held-out background conditional ECDF and transform test scores",
    )
    decorrelate.add_argument("input", help="test prediction table")
    decorrelate.add_argument("--calibration", required=True)
    decorrelate.add_argument("--output", required=True)
    decorrelate.add_argument("--artifact")
    decorrelate.add_argument("--score-column")
    decorrelate.add_argument("--energy-column")
    decorrelate.add_argument("--label-column")
    decorrelate.add_argument("--weight-column")
    decorrelate.add_argument("--event-id-column")
    decorrelate.add_argument("--split-column")
    decorrelate.add_argument("--background-label", default="0")
    decorrelate.add_argument("--score-direction", choices=["higher", "lower"], default="higher")
    decorrelate.add_argument("--energy-bins", type=int, default=12)
    decorrelate.add_argument("--min-per-bin", type=int, default=30)
    decorrelate.add_argument(
        "--allow-overlap",
        action="store_true",
        help="dangerous: allow calibration and test to share event IDs",
    )

    compare = subparsers.add_parser(
        "compare",
        help="build one CSV leaderboard from evaluation directories or metrics JSON files",
    )
    compare.add_argument("metrics", nargs="+")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--allow-mixed-data", action="store_true")
    compare.add_argument(
        "--allow-existing",
        action="store_true",
        help="replace only known EnergyBench artifacts in the output directory",
    )
    return parser


def _column_overrides(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    return {
        "event_id": getattr(args, "event_id_column", None),
        "label": getattr(args, "label_column", None),
        "score": getattr(args, "score_column", None),
        "energy_condition": getattr(args, "energy_condition_column", None),
        "energy_true": getattr(args, "energy_target_column", None),
        "energy_pred": getattr(args, "energy_pred_column", None),
        "category": getattr(args, "category_column", None),
        "sample_weight": getattr(args, "weight_column", None),
        "group_id": getattr(args, "group_column", None),
        "split": getattr(args, "split_column", None),
    }


def _evaluation_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_manifest(args.manifest)
    roc_bootstrap = args.roc_bootstrap
    regression_bootstrap = args.regression_bootstrap
    if args.bootstrap is not None:
        if roc_bootstrap is None:
            roc_bootstrap = args.bootstrap
        if regression_bootstrap is None:
            regression_bootstrap = args.bootstrap
    overrides = {
        "model_id": args.model_id,
        "task_id": args.task_id,
        "dataset.experiment": args.experiment,
        "dataset.dataset_id": args.dataset_id,
        "dataset.dataset_version": args.dataset_version,
        "dataset.split": args.split,
        "dataset.selection_split": args.selection_split,
        "dataset.energy_unit": args.energy_unit,
        "dataset.energy_condition_kind": args.energy_condition_kind,
        "dataset.energy_target_kind": args.energy_target_kind,
        "columns.event_id": args.event_id_column,
        "columns.label": args.label_column,
        "columns.score": args.score_column,
        "columns.energy_condition": args.energy_condition_column,
        "columns.energy_true": args.energy_target_column,
        "columns.energy_pred": args.energy_pred_column,
        "columns.category": args.category_column,
        "columns.sample_weight": args.weight_column,
        "columns.group_id": args.group_column,
        "columns.split": args.split_column,
        "classification.positive_label": args.positive_label,
        "classification.signal_categories": args.signal_category,
        "classification.background_categories": args.background_category,
        "classification.pair_mode": args.pair_mode,
        "classification.score_direction": args.score_direction,
        "classification.score_space": args.score_space,
        "classification.energy_bins": args.energy_bins,
        "classification.matching_target": args.matching_target,
        "classification.min_per_class": args.min_per_class,
        "classification.min_valid_bins": args.min_valid_bins,
        "classification.min_coverage": args.min_coverage,
        "classification.support_trim_quantile": args.support_trim_quantile,
        "classification.energy_roi": args.energy_roi,
        "classification.target_tpr": args.target_tpr,
        "classification.bootstrap": roc_bootstrap,
        "regression.bootstrap": regression_bootstrap,
        "regression.histogram_bins": args.histogram_bins,
        "regression.performance_bins": args.performance_bins,
        "regression.energy_floor": args.energy_floor,
        "dependence.energy_bins": args.dependence_energy_bins,
        "dependence.score_bins": args.dependence_score_bins,
        "runtime.seed": args.seed,
        "runtime.make_plots": False if args.no_plots else None,
    }
    if args.tasks == "classification":
        overrides["classification.enabled"] = True
        overrides["regression.enabled"] = False
    elif args.tasks == "regression":
        overrides["classification.enabled"] = False
        overrides["regression.enabled"] = True
    elif args.tasks == "both":
        overrides["classification.enabled"] = True
        overrides["regression.enabled"] = True
    elif args.tasks == "auto":
        overrides["classification.enabled"] = "auto"
        overrides["regression.enabled"] = "auto"
    return apply_overrides(config, overrides)


def _next_checkpoint_candidates() -> List[Path]:
    from .next_workflow import find_project_root

    root = find_project_root()
    if root is None:
        return []
    directory = root / "02_models" / "checkpoints"
    if not directory.is_dir():
        return []
    preferred = sorted(directory.glob("*_best.pt"))
    return preferred or sorted(directory.glob("*.pt"))


def _choose_next_checkpoint(requested: Optional[str]) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    candidates = _next_checkpoint_candidates()
    if not candidates:
        raise FileNotFoundError(
            "no NEXT checkpoints were found; pass one explicitly: "
            "energybench next path/to/checkpoint.pt"
        )
    if len(candidates) == 1:
        print("自动选择 checkpoint: %s" % candidates[0])
        return candidates[0]
    if not sys.stdin.isatty():
        names = ", ".join(str(path) for path in candidates)
        raise ValueError(
            "multiple NEXT checkpoints are available; pass one explicitly. "
            "Candidates: %s" % names
        )

    print("\n请选择要评测的 NEXT 模型：")
    for index, path in enumerate(candidates, 1):
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        print("  [%d] %s  (%.1f MB)" % (index, path.name, size_mb))
    while True:
        answer = input("输入编号（q 取消）: ").strip()
        if answer.lower() in {"q", "quit", "exit"}:
            raise ValueError("selection cancelled")
        try:
            selected = int(answer)
        except ValueError:
            selected = 0
        if 1 <= selected <= len(candidates):
            return candidates[selected - 1]
        print("请输入 1 到 %d 之间的编号。" % len(candidates))


def _print_evaluation_summary(report: Dict[str, Any], output: Path) -> None:
    classification = report.get("classification", {})
    aggregates = classification.get("aggregates", {})
    matched = aggregates.get("matched_auc_macro")
    inclusive = aggregates.get("inclusive_auc_macro")
    regression = report.get("energy_regression", {})
    regression_score = regression.get("energy_regression_score")

    print("Evaluation complete: %s" % output)
    print(
        "  energy-matched AUC macro: %s"
        % ("NA" if matched is None else "%.6f" % matched)
    )
    if inclusive is not None:
        print("  inclusive AUC macro: %.6f" % inclusive)
    if matched is None and classification.get("reason"):
        print("    classification: %s" % classification["reason"])
    print(
        "  energy regression ERS-v1: %s"
        % ("NA" if regression_score is None else "%.6f" % regression_score)
    )
    if regression_score is None and regression.get("reason"):
        print("    regression: %s" % regression["reason"])

    dependence = report.get("energy_dependence", {})
    independence = dependence.get("overall_energy_independence_score")
    if independence is not None:
        print("  energy independence: %.6f" % independence)
    warnings = report.get("quality", {}).get("warnings", [])
    if warnings:
        print("  warnings: %d (see .energybench/metrics.json)" % len(warnings))
    print("  table: %s" % (output / "results.csv"))
    figures = sorted(output.glob("*.png"))
    print("  figures: %d PNG file(s)" % len(figures))


def _cmd_next(args: argparse.Namespace) -> int:
    from .next_workflow import run_next_evaluation

    checkpoint = _choose_next_checkpoint(args.checkpoint)
    result = run_next_evaluation(
        checkpoint=checkpoint,
        data=args.data,
        output_root=args.output_dir,
        model_id=args.model_id,
        manifest=args.manifest,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        max_files_per_class=args.max_files_per_class,
        no_plots=args.no_plots,
        dry_run=args.dry_run,
        print_fn=print,
    )
    if result.get("dry_run"):
        print("\nDry run complete; no predictions or results were written.")
        return 0
    _print_evaluation_summary(
        result["report"], Path(result["evaluation_dir"])
    )
    print("  predictions: %s" % result["predictions_path"])
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.input)
    config = _evaluation_config(args)
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path.cwd() / ("evaluation_%s" % str(config["model_id"]).replace("/", "_"))
    )
    report = run_evaluation(
        bundle,
        config,
        output,
        strict=args.strict,
        allow_existing=args.allow_existing,
    )
    _print_evaluation_summary(report, output)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.input)
    schema = resolve_schema(bundle, _column_overrides(args))
    payload = {
        "source": str(bundle.source),
        "n_events": bundle.n_events,
        "columns": {
            name: {"shape": list(values.shape), "dtype": str(values.dtype)}
            for name, values in bundle.columns.items()
        },
        "metadata": bundle.metadata,
        "inferred_roles": schema,
        "duplicate_event_ids": (
            None
            if schema.get("event_id") is None
            else duplicate_event_ids(bundle.require(schema["event_id"]))
        ),
    }
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    arguments = parse_adapter_arguments(args.adapter_arg)
    bundle = run_adapter(args.adapter, args.model, args.data, arguments)
    destination = save_bundle(bundle, args.output)
    print("Wrote %d predictions to %s" % (bundle.n_events, destination))
    return 0


def _value_matching(values: np.ndarray, requested: str) -> np.ndarray:
    return values.astype(str) == str(requested)


def _cmd_decorrelate(args: argparse.Namespace) -> int:
    test_path = Path(args.input).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    if test_path == calibration_path and not args.allow_overlap:
        raise ValueError("calibration and test paths are identical")
    test = load_bundle(test_path)
    calibration = load_bundle(calibration_path)
    cal_score_name = resolve_column(
        calibration, "score", args.score_column, required=True
    )
    cal_energy_name = resolve_column(
        calibration, "energy_condition", args.energy_column, required=True
    )
    cal_label_name = resolve_column(
        calibration, "label", args.label_column, required=True
    )
    cal_weight_name = resolve_column(
        calibration, "sample_weight", args.weight_column, required=False
    )
    test_score_name = resolve_column(test, "score", args.score_column, required=True)
    test_energy_name = resolve_column(
        test, "energy_condition", args.energy_column, required=True
    )

    cal_ids_name = resolve_column(
        calibration, "event_id", args.event_id_column, required=False
    )
    test_ids_name = resolve_column(test, "event_id", args.event_id_column, required=False)
    cal_split_name = resolve_column(
        calibration, "split", args.split_column, required=False
    )
    test_split_name = resolve_column(
        test, "split", args.split_column, required=False
    )
    calibration_splits = (
        []
        if cal_split_name is None
        else sorted(
            np.unique(
                np.asarray(calibration.require(cal_split_name)).astype(str)
            ).tolist()
        )
    )
    test_splits = (
        []
        if test_split_name is None
        else sorted(
            np.unique(np.asarray(test.require(test_split_name)).astype(str))
            .tolist()
        )
    )
    if not args.allow_overlap and (not cal_ids_name or not test_ids_name):
        raise ValueError(
            "calibration and test both need event IDs to prove disjointness; "
            "use --allow-overlap only for an explicitly unverified run"
        )
    if cal_ids_name and test_ids_name and not args.allow_overlap:
        if duplicate_event_ids(calibration.require(cal_ids_name)):
            raise ValueError("calibration contains duplicate event IDs")
        if duplicate_event_ids(test.require(test_ids_name)):
            raise ValueError("test contains duplicate event IDs")
        overlap = np.intersect1d(
            np.asarray(calibration.require(cal_ids_name)).astype(str),
            np.asarray(test.require(test_ids_name)).astype(str),
        )
        if len(overlap):
            raise ValueError(
                "calibration and test share %d event IDs; use a held-out split"
                % len(overlap)
            )
    if not args.allow_overlap and (
        cal_split_name is None or test_split_name is None
    ):
        raise ValueError(
            "calibration and test both need split columns to prove distinct "
            "roles; use --allow-overlap only for an explicitly unverified run"
        )
    if not args.allow_overlap:
        shared_splits = sorted(set(calibration_splits) & set(test_splits))
        if shared_splits:
            raise ValueError(
                "calibration and test share split label(s) %s; use distinct "
                "held-out split roles" % shared_splits
            )

    cal_labels = np.asarray(calibration.require(cal_label_name))
    is_background = _value_matching(cal_labels, args.background_label)
    if not np.any(is_background):
        raise ValueError(
            "background label %r not present in calibration labels" % args.background_label
        )
    cal_weight = (
        None
        if cal_weight_name is None
        else np.asarray(calibration.require(cal_weight_name), dtype=float)
    )
    transform = BackgroundQuantileDecorrelator.fit(
        calibration.require(cal_energy_name),
        calibration.require(cal_score_name),
        is_background,
        cal_weight,
        n_energy_bins=args.energy_bins,
        min_per_bin=args.min_per_bin,
        score_direction=args.score_direction,
    )
    transformed_score = transform.transform(
        test.require(test_energy_name), test.require(test_score_name)
    )
    columns = dict(test.columns)
    if "score_raw" not in columns:
        columns["score_raw"] = np.asarray(test.require(test_score_name))
    if "score_decorrelated" in columns:
        raise ValueError("test table already contains score_decorrelated")
    columns["score_decorrelated"] = transformed_score
    metadata = dict(test.metadata)
    metadata["decorrelation"] = {
        "method": transform.method,
        "verification_status": (
            "unverified_override"
            if args.allow_overlap
            else "verified_disjoint"
        ),
        "allow_overlap": bool(args.allow_overlap),
        "calibration_path": str(calibration_path),
        "calibration_sha256": file_sha256(calibration_path),
        "test_input_sha256": file_sha256(test_path),
        "calibration_splits": calibration_splits,
        "test_splits": test_splits,
        "background_label": args.background_label,
        "n_calibration_background": int(np.sum(is_background)),
        "n_test_events": test.n_events,
        "energy_bins": len(transform.sorted_scores),
        "score_direction": args.score_direction,
        "output_score_column": "score_decorrelated",
    }
    destination = save_bundle(
        PredictionBundle(columns, metadata=metadata), args.output
    )
    artifact = (
        Path(args.artifact).expanduser().resolve()
        if args.artifact
        else destination.with_suffix(".decorrelator.json")
    )
    artifact_payload = transform.to_dict()
    artifact_payload["provenance"] = {
        "verification_status": (
            "unverified_override"
            if args.allow_overlap
            else "verified_disjoint"
        ),
        "allow_overlap": bool(args.allow_overlap),
        "calibration_path": str(calibration_path),
        "calibration_sha256": file_sha256(calibration_path),
        "test_input_path": str(test_path),
        "test_input_sha256": file_sha256(test_path),
        "calibration_splits": calibration_splits,
        "test_splits": test_splits,
        "event_id_overlap_checked": bool(
            not args.allow_overlap and cal_ids_name and test_ids_name
        ),
        "split_disjointness_checked": bool(
            not args.allow_overlap and cal_split_name and test_split_name
        ),
        "background_label": args.background_label,
        "n_calibration_events": calibration.n_events,
        "n_calibration_background": int(np.sum(is_background)),
        "n_test_events": test.n_events,
    }
    write_json(artifact, artifact_payload)
    print("Wrote decorrelated predictions to %s" % destination)
    print("Wrote calibration artifact to %s" % artifact)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    paths = []
    for item in args.metrics:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            candidates = [
                path / ".energybench" / "metrics.json",
                path / "metrics.json",
            ]
            path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
        if not path.is_file():
            raise FileNotFoundError("metrics file does not exist: %s" % path)
        paths.append(path)
    output = Path(args.output_dir).expanduser().resolve()
    summary = compare_evaluations(
        paths,
        output,
        args.allow_mixed_data,
        allow_existing=args.allow_existing,
    )
    write_json(output / ".energybench" / "comparison.json", summary)
    print("Compared %d evaluations in %s" % (len(paths), output))
    print("  table: %s" % (output / "leaderboard.csv"))
    if args.allow_mixed_data:
        print("  warning: non-comparative inventory; rows were not ranked")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    commands = {
        "next": _cmd_next,
        "evaluate": _cmd_evaluate,
        "inspect": _cmd_inspect,
        "predict": _cmd_predict,
        "decorrelate": _cmd_decorrelate,
        "compare": _cmd_compare,
    }
    try:
        status = commands[args.command](args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        message = str(exc)
        if isinstance(exc, KeyError) and len(message) >= 2:
            message = message.strip("'")
        parts = [part.strip() for part in message.split(";") if part.strip()]
        print("energybench %s: error" % args.command, file=sys.stderr)
        for part in parts:
            print("  - %s" % part, file=sys.stderr)
        print(
            "Run `energybench %s --help` for command options." % args.command,
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
