"""Merge per-run ``run_summary.json`` files into one results CSV and a ranking.

The 9 array tasks each write only their own ``run_summary.json``; nothing writes
a shared file during the array, so there is no race. This script runs once
afterwards, is idempotent, and is safe to re-run while later runs are still
finishing (use ``--require-all`` to insist the matrix is complete).

Stdlib only -- no pandas, deliberately: this is a 9-row table and the installed
pandas major version has moved on from what other collators in this repo were
written against.

Usage::

    python cuore_detector/scripts/collate_results.py \\
        --output-root cuore_detector/results/cuore_matrix_v1 [--require-all]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

TOKENIZATIONS = ("raw_patches", "segment_summary", "pulse_entities")
ENCODINGS = ("coordinate_mlp", "fourier_coordinates", "rope")

EXPECTED_RUN_IDS = tuple(
    f"regression__{tokenization}__{encoding}"
    for tokenization in TOKENIZATIONS
    for encoding in ENCODINGS
)

# Column order for the CSV; any key not listed is appended alphabetically.
PREFERRED_COLUMNS = (
    "run_id",
    "tokenization",
    "position_encoding",
    "num_tokens",
    "feature_dim",
    "parameter_count",
    "rope_base",
    "rope_time_axis_only",
    "n_train",
    "n_val",
    "n_test",
    "epochs_completed",
    "best_epoch",
    "best_val_rmse_scaled",
    "test_rmse_ms",
    "test_mae_ms",
    "test_bias_ms",
    "test_r2",
    "test_median_abs_error_ms",
    "test_rmse_ms_pileup_only",
    "test_mae_ms_pileup_only",
    "test_bias_ms_pileup_only",
    "test_r2_pileup",
    "test_rmse_ms_clean",
    "test_mae_ms_clean",
    "test_bias_ms_clean",
    "test_frac_within_100ms",
    "test_frac_within_250ms",
    "training_seconds",
    "minutes_per_epoch",
    "evaluation_seconds",
)

PRIMARY_KEY = "test_rmse_ms_pileup_only"
SECONDARY_KEY = "test_r2"


def _load_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("*/run_summary.json")):
        run_dir = summary_path.parent
        report = run_dir / "evaluation" / "cuore_regression_ms.json"
        if not report.is_file():
            print(
                f"  ! {run_dir.name}: run_summary.json exists but "
                "evaluation/cuore_regression_ms.json is missing -- treating "
                "the run as truncated and skipping it"
            )
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def _order_columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for row in rows:
        seen.update(row)
    ordered = [name for name in PREFERRED_COLUMNS if name in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in columns})
        # NamedTemporaryFile defaults to 0600; this is a shared collaboration
        # repo, so widen to the usual 0644 before publishing the result.
        Path(handle.name).chmod(0o644)
        Path(handle.name).replace(path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _sort_key(row: dict[str, Any]) -> tuple[float, float]:
    primary = row.get(PRIMARY_KEY)
    secondary = row.get(SECONDARY_KEY)
    primary_value = (
        float(primary)
        if isinstance(primary, (int, float)) and math.isfinite(float(primary))
        else math.inf
    )
    secondary_value = (
        float(secondary)
        if isinstance(secondary, (int, float)) and math.isfinite(float(secondary))
        else -math.inf
    )
    # ascending pile-up RMSE, then descending overall R^2
    return (primary_value, -secondary_value)


def _format(value: Any, digits: int = 1) -> str:
    if isinstance(value, bool) or value is None or value == "":
        return "-"
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return "nan"
        return f"{number:.{digits}f}"
    return str(value)


def _variant(row: dict[str, Any]) -> str:
    """Label a run's variant, derived from the run_id suffix.

    The official 9 have no suffix; sweeps and diagnostics carry one (e.g.
    ``_base32``, ``_timeonly``). Ranking them together would be meaningless --
    every RoPE variant shares position_encoding == "rope".
    """

    run_id = str(row.get("run_id", ""))
    stem = f"regression__{row.get('tokenization')}__{row.get('position_encoding')}"
    suffix = run_id[len(stem) :] if run_id.startswith(stem) else ""
    if not suffix:
        return "official"
    return suffix.lstrip("_")


def _table(rows: list[dict[str, Any]], *, show_variant: bool) -> list[str]:
    header = "| # | tokenization | encoding |"
    divider = "|--:|---|---|"
    if show_variant:
        header += " variant |"
        divider += "---|"
    header += " pile-up RMSE [ms] | RMSE [ms] | MAE [ms] | bias [ms] | R² | params |"
    divider += "--:|--:|--:|--:|--:|--:|"

    lines = [header, divider]
    for position, row in enumerate(sorted(rows, key=_sort_key), start=1):
        cells = [
            str(position),
            str(row.get("tokenization", "?")),
            str(row.get("position_encoding", "?")),
        ]
        if show_variant:
            cells.append(_variant(row))
        cells.extend(
            [
                _format(row.get(PRIMARY_KEY)),
                _format(row.get("test_rmse_ms")),
                _format(row.get("test_mae_ms")),
                _format(row.get("test_bias_ms")),
                _format(row.get("test_r2"), 4),
                _format(row.get("parameter_count"), 0),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _ranking_markdown(rows: list[dict[str, Any]]) -> str:
    official = [row for row in rows if _variant(row) == "official"]
    sweeps = [row for row in rows if _variant(row) != "official"]

    lines = [
        "# CUORE Δt regression -- 3 x 3 matrix",
        "",
        f"Ranked by `{PRIMARY_KEY}` ascending, tie-broken by `{SECONDARY_KEY}`"
        " descending.",
        "",
        "`ers` and `fractional_resolution_68` are NOT used: both are"
        " fraction-based and degenerate here, because ~51% of the Δt targets"
        " are exactly 0. `clean_ms.r2` is NaN by construction (zero variance).",
        "",
        "## Official matrix",
        "",
    ]
    lines.extend(_table(official, show_variant=False))

    if sweeps:
        lines.extend(
            [
                "",
                "## Sweeps and diagnostics (NOT part of the official matrix)",
                "",
                "These vary one hyperparameter of the RoPE cells. They are"
                " listed separately because every one of them shares"
                " `position_encoding == \"rope\"`, so ranking them alongside the"
                " official cells would compare a tuned family against untuned"
                " single configurations.",
                "",
            ]
        )
        lines.extend(_table(sweeps, show_variant=True))

    lines.extend(
        [
            "",
            "Parameter counts differ across the tokenization axis"
            " (`raw_patches` has feature_dim 20 vs 2) and across the encoding"
            " axis (`rope` adds no positional parameters). Capacity is not"
            " matched -- do not read these cells as a controlled capacity"
            " comparison. All runs are single-seed: differences smaller than"
            " the within-family spread are not interpretable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cuore_detector/results/cuore_matrix_v1"),
        help="directory containing one subdirectory per run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: <output-root>/cuore_transformer_results.csv)",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero unless all 9 expected runs are present",
    )
    arguments = parser.parse_args()

    output_root = arguments.output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise SystemExit(f"output root does not exist: {output_root}")

    print(f"Scanning {output_root}")
    rows = _load_rows(output_root)
    if not rows:
        raise SystemExit(f"no complete runs found under {output_root}")

    found = {str(row.get("run_id")) for row in rows}
    missing = [name for name in EXPECTED_RUN_IDS if name not in found]
    extra = sorted(found - set(EXPECTED_RUN_IDS))
    print(f"Found {len(rows)} complete run(s); {len(missing)} of 9 missing.")
    for name in missing:
        print(f"  missing: {name}")
    for name in extra:
        print(f"  extra (sweep/diagnostic): {name}")

    columns = _order_columns(rows)
    csv_path = arguments.out or (output_root / "cuore_transformer_results.csv")
    _write_csv(csv_path, sorted(rows, key=_sort_key), columns)
    print(f"Wrote {csv_path}")

    ranking = _ranking_markdown(rows)
    ranking_path = output_root / "cuore_ranking.md"
    ranking_path.write_text(ranking, encoding="utf-8")
    print(f"Wrote {ranking_path}\n")
    print(ranking)

    if missing and arguments.require_all:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
