"""Aggregate the multi-seed CUORE Δt matrix into the final benchmark.

Groups every completed run by its configuration (tokenization x positional
encoding x RoPE variant), pools across seeds, and reports mean +/- standard
deviation. Critically, it also estimates the *seed noise floor* -- the pooled
within-configuration standard deviation -- and refuses to call any pairwise
difference meaningful unless it clears that floor.

Stdlib only. Usage::

    python cuore_detector/scripts/final_benchmark.py \\
        --output-root cuore_detector/results/cuore_matrix_v1 \\
        --out cuore_detector/results_summary/cuore_matrix_v1/FINAL_BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    ("test_rmse_ms_pileup_only", "pile-up RMSE [ms]", "min"),
    ("test_rmse_ms", "RMSE [ms]", "min"),
    ("test_mae_ms", "MAE [ms]", "min"),
    ("test_bias_ms", "bias [ms]", "abs"),
    ("test_r2", "R²", "max"),
)
PRIMARY = "test_rmse_ms_pileup_only"

_SEED_SUFFIX = re.compile(r"_seed\d+$")


def _config_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """(tokenization, encoding, variant) with any _seedN suffix stripped."""

    run_id = str(row.get("run_id", ""))
    tokenization = str(row.get("tokenization", "?"))
    encoding = str(row.get("position_encoding", "?"))
    stem = f"regression__{tokenization}__{encoding}"
    suffix = run_id[len(stem) :] if run_id.startswith(stem) else ""
    suffix = _SEED_SUFFIX.sub("", suffix).lstrip("_")
    if encoding == "rope":
        variant = suffix or "base8"
    else:
        variant = suffix or "-"
    return tokenization, encoding, variant


def _load(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*/run_summary.json")):
        if not (path.parent / "evaluation" / "cuore_regression_ms.json").is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row.setdefault("seed", 42)  # runs from before seed was recorded
        rows.append(row)
    return rows


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                out.append(number)
    return out


def _stat(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev, len(values)


def _fmt(mean: float, stdev: float, count: int, digits: int = 1) -> str:
    if count == 0 or not math.isfinite(mean):
        return "-"
    if count == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {stdev:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cuore_detector/results/cuore_matrix_v1"),
    )
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    output_root = arguments.output_root.expanduser().resolve()
    rows = _load(output_root)
    if not rows:
        raise SystemExit(f"no complete runs under {output_root}")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_config_key(row)].append(row)

    # ---- seed noise floor: pooled within-configuration stdev on the primary
    spreads: list[float] = []
    for members in groups.values():
        values = _finite([row.get(PRIMARY) for row in members])
        if len(values) > 1:
            spreads.append(statistics.stdev(values))
    noise = statistics.fmean(spreads) if spreads else float("nan")
    max_seeds = max(len(members) for members in groups.values())

    summary: list[dict[str, Any]] = []
    for key, members in groups.items():
        tokenization, encoding, variant = key
        entry: dict[str, Any] = {
            "tokenization": tokenization,
            "encoding": encoding,
            "variant": variant,
            "seeds": sorted({int(row.get("seed", 42)) for row in members}),
        }
        for name, _, _ in METRICS:
            values = _finite([row.get(name) for row in members])
            entry[name] = _stat(values)
        summary.append(entry)

    official = [
        item
        for item in summary
        if item["variant"] in ("-", "base8")
    ]
    others = [item for item in summary if item not in official]
    official.sort(key=lambda item: item[PRIMARY][0])
    others.sort(key=lambda item: item[PRIMARY][0])

    lines = [
        "# CUORE first-two-pulse Δt regression -- final benchmark",
        "",
        "Task: regress the time gap between the first two detected pulses in a"
        " CUORE waveform, from `pulseFinder/dtBetweenPeaks[:, 0]`.",
        "Test set: all 1,000 events of `cuoreTest.h5`, identical for every run.",
        "Single-pulse events are kept with Δt = 0 (~51% of events).",
        "",
        f"Aggregated over up to **{max_seeds} seeds** per configuration"
        " (seed drives both weight initialization and the train/val partition;"
        " the test set is fixed). Cells show mean ± sd.",
        "",
        f"**Seed noise floor: ±{noise:.1f} ms** on pile-up RMSE (pooled"
        " within-configuration standard deviation). Differences smaller than"
        " roughly twice this are not interpretable.",
        "",
        "`ers` and `fractional_resolution_68` are excluded throughout: both are"
        " fraction-based and degenerate against the ~51% point mass at Δt = 0.",
        "",
        "## Official matrix (3 tokenizations × 3 positional encodings)",
        "",
        "| # | tokenization | encoding | n | "
        + " | ".join(label for _, label, _ in METRICS)
        + " |",
        "|--:|---|---|--:|" + "--:|" * len(METRICS),
    ]
    for position, item in enumerate(official, start=1):
        cells = [
            str(position),
            item["tokenization"],
            item["encoding"],
            str(len(item["seeds"])),
        ]
        for name, _, _ in METRICS:
            mean, stdev, count = item[name]
            cells.append(_fmt(mean, stdev, count, 4 if name == "test_r2" else 1))
        lines.append("| " + " | ".join(cells) + " |")

    if others:
        lines += [
            "",
            "## RoPE hyperparameter variants (not part of the official matrix)",
            "",
            "| # | tokenization | variant | n | "
            + " | ".join(label for _, label, _ in METRICS)
            + " |",
            "|--:|---|---|--:|" + "--:|" * len(METRICS),
        ]
        for position, item in enumerate(others, start=1):
            cells = [
                str(position),
                item["tokenization"],
                item["variant"],
                str(len(item["seeds"])),
            ]
            for name, _, _ in METRICS:
                mean, stdev, count = item[name]
                cells.append(
                    _fmt(mean, stdev, count, 4 if name == "test_r2" else 1)
                )
            lines.append("| " + " | ".join(cells) + " |")

    # ---- marginal means
    lines += ["", "## Marginal means on pile-up RMSE [ms]", ""]
    for axis, label in (("tokenization", "Tokenization"), ("encoding", "Encoding")):
        buckets: dict[str, list[float]] = defaultdict(list)
        for item in official:
            mean, _, count = item[PRIMARY]
            if count:
                buckets[item[axis]].append(mean)
        lines.append(f"**{label}** (averaged over the other axis):")
        lines.append("")
        for name, values in sorted(buckets.items(), key=lambda kv: statistics.fmean(kv[1])):
            lines.append(f"- `{name}`: {statistics.fmean(values):.1f}")
        lines.append("")

    # ---- what clears the noise floor
    lines += ["## Differences that clear the noise floor", ""]
    if official and math.isfinite(noise):
        best = official[0]
        threshold = 2.0 * noise
        beaten = [
            item
            for item in official[1:]
            if item[PRIMARY][0] - best[PRIMARY][0] > threshold
        ]
        indistinguishable = [
            item
            for item in official[1:]
            if item[PRIMARY][0] - best[PRIMARY][0] <= threshold
        ]
        head = (
            f"{best['tokenization']} x {best['encoding']} "
            f"({best[PRIMARY][0]:.1f} ms)"
        )
        lines.append(f"Best configuration: **{head}**.")
        lines.append("")
        if indistinguishable:
            lines.append(
                f"Statistically indistinguishable from it (within 2 x"
                f" {noise:.1f} ms):"
            )
            lines.append("")
            for item in indistinguishable:
                lines.append(
                    f"- {item['tokenization']} x {item['encoding']}"
                    f" ({item[PRIMARY][0]:.1f} ms,"
                    f" +{item[PRIMARY][0] - best[PRIMARY][0]:.1f})"
                )
            lines.append("")
        if beaten:
            lines.append("Genuinely worse than the best configuration:")
            lines.append("")
            for item in beaten:
                lines.append(
                    f"- {item['tokenization']} x {item['encoding']}"
                    f" ({item[PRIMARY][0]:.1f} ms,"
                    f" +{item[PRIMARY][0] - best[PRIMARY][0]:.1f})"
                )
            lines.append("")

    lines += [
        "## Caveats",
        "",
        "- Parameter counts are not matched: `raw_patches` has feature_dim 20"
        " vs 2 for the other tokenizations, and `rope` adds no positional"
        " parameters while `coordinate_mlp` adds ~4.4k and"
        " `fourier_coordinates` ~6.7k.",
        "- The tokenizations differ in information content, not just layout:"
        " `raw_patches` preserves all 10,000 samples, `segment_summary`"
        " compresses to 500 mean/RMS pairs, `pulse_entities` keeps 500 of"
        " 10,000 raw samples.",
        "- Labels are heuristic `pulseFinder` output, not ground truth;"
        " `tailElevated` events carry lower-bound Δt.",
        "- Reported RMSE/MAE/bias are dominated by a heavy tail: median"
        " absolute error is several times smaller than the mean.",
        "",
    ]

    text = "\n".join(lines)
    destination = arguments.out or (output_root / "FINAL_BENCHMARK.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    destination.chmod(0o644)
    print(f"Wrote {destination}\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
