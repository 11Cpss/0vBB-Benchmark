"""Merge per-run ``run_summary.json`` files into one results CSV and a ranking.

The 3 array tasks each write only their own ``run_summary.json``; nothing
writes a shared file during the array, so there is no race. This script runs
once afterwards, is idempotent, and is safe to re-run while later runs are
still finishing (use ``--require-all`` to insist the matrix is complete).

Stdlib only -- no pandas, deliberately: this is a 3-row table.

Usage::

    python mjd_rope_classifier/scripts/collate_results.py \\
        --output-root mjd_rope_classifier/results/mjd_rope_classification_v1 [--require-all]
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

EXPECTED_RUN_IDS = tuple(
    f"classification__{tokenization}__rope" for tokenization in TOKENIZATIONS
)

# Column order for the CSV; any key not listed is appended alphabetically.
PREFERRED_COLUMNS = (
    "run_id",
    "tokenization",
    "position_encoding",
    "rope_base",
    "rope_time_axis_only",
    "num_tokens",
    "feature_dim",
    "parameter_count",
    "n_train",
    "n_val",
    "n_test",
    "epochs_completed",
    "best_epoch",
    "best_val_macro_auc",
    "test_auc",
    "test_accuracy",
    "energy_independence_score",
    "worst_energy_independence_score",
    "test_clean_fraction",
    "test_events",
    "test_loss",
    "training_seconds",
    "minutes_per_epoch",
    "evaluation_seconds",
)

PRIMARY_KEY = "test_auc"


def _load_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("*/run_summary.json")):
        run_dir = summary_path.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            print(
                f"  ! {run_dir.name}: run_summary.json exists but "
                "metrics.json is missing -- treating the run as truncated "
                "and skipping it"
            )
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        # Optional: energy_independence.py (scripts/energy_independence.py)
        # writes this post hoc from predictions.npz; fold it in when present
        # so it sits alongside test_auc in the same table, matching how the
        # NEXT results notebook reports AUC next to energy independence.
        dependence_path = run_dir / "energy_independence.json"
        if dependence_path.is_file():
            with dependence_path.open("r", encoding="utf-8") as handle:
                dependence = json.load(handle)
            row["energy_independence_score"] = dependence.get(
                "overall_energy_independence_score"
            )
            row["worst_energy_independence_score"] = dependence.get(
                "worst_group_energy_independence_score"
            )
        rows.append(row)
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


def _sort_key(row: dict[str, Any]) -> float:
    value = row.get(PRIMARY_KEY)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return -float(value)  # descending AUC
    return math.inf


def _format(value: Any, digits: int = 4) -> str:
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

    The official 3 have no suffix; sweeps and diagnostics (e.g. a rope_base
    sweep) carry one. Ranking them together would compare a tuned family
    against untuned single configurations.
    """

    run_id = str(row.get("run_id", ""))
    stem = f"classification__{row.get('tokenization')}__rope"
    suffix = run_id[len(stem) :] if run_id.startswith(stem) else ""
    if not suffix:
        return "official"
    return suffix.lstrip("_")


def _table(rows: list[dict[str, Any]], *, show_variant: bool) -> list[str]:
    header = "| # | tokenization |"
    divider = "|--:|---|"
    if show_variant:
        header += " variant |"
        divider += "---|"
    header += " test AUC | test accuracy | energy independence | worst-group | rope_base | params |"
    divider += "--:|--:|--:|--:|--:|--:|"

    lines = [header, divider]
    for position, row in enumerate(sorted(rows, key=_sort_key), start=1):
        cells = [str(position), str(row.get("tokenization", "?"))]
        if show_variant:
            cells.append(_variant(row))
        cells.extend(
            [
                _format(row.get("test_auc")),
                _format(row.get("test_accuracy")),
                _format(row.get("energy_independence_score")),
                _format(row.get("worst_energy_independence_score")),
                _format(row.get("rope_base"), 2),
                _format(row.get("parameter_count"), 0),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _ranking_markdown(rows: list[dict[str, Any]]) -> str:
    official = [row for row in rows if _variant(row) == "official"]
    sweeps = [row for row in rows if _variant(row) != "official"]

    lines = [
        "# MJD RoPE classification -- 3-tokenization matrix",
        "",
        f"Ranked by `{PRIMARY_KEY}` descending (signal = clean, all four PSD"
        " cuts). Chance level: AUC 0.5, accuracy 0.62 (majority class,"
        " non-clean).",
        "",
        "`energy independence` / `worst-group` are EnergyBench's"
        " class-conditional score/energy independence score (run"
        " post hoc by `scripts/energy_independence.py` from each run's"
        " `predictions.npz`; 1.0 = the classifier's output distribution is"
        " identical across energy within each class, 0.0 = maximal"
        " class-conditional energy dependence). Blank when that script"
        " hasn't been run yet.",
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
            ]
        )
        lines.extend(_table(sweeps, show_variant=True))

    lines.extend(
        [
            "",
            "Parameter counts differ across the tokenization axis"
            " (`raw_patches` has feature_dim = patch_size vs. 2 for the"
            " other two). All runs are single-seed: differences smaller"
            " than the within-family spread are not interpretable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("mjd_rope_classifier/results/mjd_rope_classification_v1"),
        help="directory containing one subdirectory per run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: <output-root>/mjd_rope_classification_results.csv)",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero unless all 3 expected runs are present",
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
    print(f"Found {len(rows)} complete run(s); {len(missing)} of 3 missing.")
    for name in missing:
        print(f"  missing: {name}")
    for name in extra:
        print(f"  extra (sweep/diagnostic): {name}")

    columns = _order_columns(rows)
    csv_path = arguments.out or (output_root / "mjd_rope_classification_results.csv")
    _write_csv(csv_path, sorted(rows, key=_sort_key), columns)
    print(f"Wrote {csv_path}")

    ranking = _ranking_markdown(rows)
    ranking_path = output_root / "mjd_rope_ranking.md"
    ranking_path.write_text(ranking, encoding="utf-8")
    print(f"Wrote {ranking_path}\n")
    print(ranking)

    if missing and arguments.require_all:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
