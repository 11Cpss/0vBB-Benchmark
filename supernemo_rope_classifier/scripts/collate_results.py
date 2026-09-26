"""Merge per-run ``run_summary.json`` files into one results CSV and a ranking.

Each array task writes only its own ``run_summary.json``; nothing writes a
shared file during the array, so there is no race. This script runs once
afterwards, is idempotent, and is safe to re-run while later runs are still
finishing (use ``--require-all`` to insist the official matrix is complete).

The official matrix is the three tokenizations under ``rope`` at the default
``rope_base``. Position-encoding controls and ``rope_base`` sweeps are listed
separately: ranking a tuned family against untuned single configurations
would not be a like-for-like comparison.

Stdlib only.

Usage::

    python supernemo_rope_classifier/scripts/collate_results.py \\
        --output-root supernemo_rope_classifier/results/supernemo_rope_classification_v1 \\
        [--require-all]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

TOKENIZATIONS = ("sampled_hits", "voxel", "summary_features")

EXPECTED_RUN_IDS = tuple(f"classification__{name}__rope" for name in TOKENIZATIONS)

# Column order for the CSV; any key not listed is appended alphabetically.
PREFERRED_COLUMNS = (
    "run_id",
    "tokenization",
    "position_encoding",
    "rope_base",
    "max_tokens",
    "feature_dim",
    "parameter_count",
    "n_train",
    "n_val",
    "n_test",
    "published_split",
    "epochs_completed",
    "best_epoch",
    "stopped_early",
    "best_val_auc",
    "test_matched_auc",
    "test_matched_auc_status",
    "test_inclusive_auc",
    "test_common_support_auc",
    "test_shortcut_gap",
    "energy_independence_score",
    "worst_energy_independence_score",
    "training_seconds",
    "minutes_per_epoch",
    "evaluation_seconds",
)

PRIMARY_KEY = "test_matched_auc"


def _load_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("*/run_summary.json")):
        run_dir = summary_path.parent
        if not (run_dir / "evaluation" / "metrics.json").is_file():
            print(
                f"  ! {run_dir.name}: run_summary.json exists but "
                "evaluation/metrics.json is missing -- treating the run as "
                "truncated and skipping it"
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


def _sort_key(row: dict[str, Any]) -> float:
    value = row.get(PRIMARY_KEY)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return -float(value)  # descending matched AUC
    return math.inf  # unevaluable runs last


def _format(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None or value == "":
        return "-"
    if isinstance(value, (int, float)):
        number = float(value)
        return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"
    return str(value)


def _variant(row: dict[str, Any]) -> str:
    """Label a run's variant from its run_id.

    ``classification__<tokenization>__<encoding><suffix>``: the official cell is
    ``rope`` with no suffix; everything else is a control or a sweep.
    """

    run_id = str(row.get("run_id", ""))
    stem = f"classification__{row.get('tokenization')}__"
    remainder = run_id[len(stem) :] if run_id.startswith(stem) else run_id
    if remainder == "rope":
        return "official"
    return remainder


def _table(rows: list[dict[str, Any]], *, show_variant: bool) -> list[str]:
    header = "| # | tokenization |"
    divider = "|--:|---|"
    if show_variant:
        header += " variant |"
        divider += "---|"
    header += (
        " matched AUC | inclusive AUC | energy independence | worst-group |"
        " rope_base | params | epochs |"
    )
    divider += "--:|--:|--:|--:|--:|--:|--:|"

    lines = [header, divider]
    for position, row in enumerate(sorted(rows, key=_sort_key), start=1):
        cells = [str(position), str(row.get("tokenization", "?"))]
        if show_variant:
            cells.append(_variant(row))
        cells.extend(
            [
                _format(row.get("test_matched_auc"), 6),
                _format(row.get("test_inclusive_auc"), 6),
                _format(row.get("energy_independence_score")),
                _format(row.get("worst_energy_independence_score")),
                _format(row.get("rope_base"), 2),
                _format(row.get("parameter_count"), 0),
                _format(row.get("epochs_completed"), 0),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _ranking_markdown(rows: list[dict[str, Any]]) -> str:
    official = [row for row in rows if _variant(row) == "official"]
    others = [row for row in rows if _variant(row) != "official"]
    capped = [row for row in rows if row.get("published_split") is False]

    lines = [
        "# SuperNEMO RoPE classification -- 3-tokenization matrix",
        "",
        f"Ranked by `{PRIMARY_KEY}` (EnergyBench energy-matched AUC on the held-out"
        " test split) descending. Signal `2nubb` vs. background `Bi214`; the model"
        " sees tracker-hit topology only. `inclusive AUC` is supporting context;"
        " `energy independence` / `worst-group` are EnergyBench's class-conditional"
        " score/energy independence scores (1.0 = the score distribution is"
        " identical across energy within each class).",
        "",
    ]
    if capped:
        names = ", ".join(sorted(str(row.get("run_id")) for row in capped))
        lines.extend(
            [
                "> **Reduced-scale runs present** (per-split event caps, not the full"
                f" published split): {names}. Do not compare them with full-split runs.",
                "",
            ]
        )
    lines.extend(["## Official matrix", ""])
    lines.extend(_table(official, show_variant=False))

    if others:
        lines.extend(["", "## Controls and sweeps (NOT part of the official matrix)", ""])
        lines.extend(_table(others, show_variant=True))

    lines.extend(
        [
            "",
            "Tokenizations differ in sequence length and feature dimension"
            " (`summary_features` uses 16 tokens x 6 features; the others up to 128"
            " tokens x 4 features). All runs are single-seed: differences smaller"
            " than the seed-to-seed spread are not interpretable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("supernemo_rope_classifier/results/supernemo_rope_classification_v1"),
        help="directory containing one subdirectory per run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: <output-root>/supernemo_rope_classification_results.csv)",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero unless all 3 official runs are present",
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
    print(f"Found {len(rows)} complete run(s); {len(missing)} of 3 official missing.")
    for name in missing:
        print(f"  missing: {name}")
    for name in extra:
        print(f"  extra (control/sweep): {name}")

    columns = _order_columns(rows)
    csv_path = arguments.out or (output_root / "supernemo_rope_classification_results.csv")
    _write_csv(csv_path, sorted(rows, key=_sort_key), columns)
    print(f"Wrote {csv_path}")

    ranking = _ranking_markdown(rows)
    ranking_path = output_root / "supernemo_rope_ranking.md"
    ranking_path.write_text(ranking, encoding="utf-8")
    print(f"Wrote {ranking_path}\n")
    print(ranking)

    if missing and arguments.require_all:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
