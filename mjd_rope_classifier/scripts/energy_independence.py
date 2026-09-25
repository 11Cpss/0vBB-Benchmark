"""Energy-independence score for completed MJD RoPE classification runs.

Computes EnergyBench's class-conditional score/energy independence score
(``evalutaions_workflow/energybench/metrics.py::evaluate_energy_dependence``)
directly from each run's already-saved ``predictions.npz`` -- no re-inference,
no re-training. The score answers: does the classifier's output distribution
stay the same across energy, *within* each class (clean / non-clean)? A score
of 1.0 means the score distribution in every energy bin matches the pooled
distribution for that class (fully energy-independent); 0.0 means maximal
class-conditional dependence on energy.

Why this needs its own script instead of reusing ``evaluate_classification``:
that NEXT-specific wrapper re-runs model inference over a dataloader that
must yield NEXT-only fields (``projection_coverage``, ``group_id``,
``split``, ...) and hard-locks ``EvaluationConfig.energy_unit`` to ``"MeV"``.
MJD's own ``predictions.npz`` (target, prediction) plus the event energies
recovered via ``id`` is exactly the ``labels, scores, energies`` triple
``evaluate_energy_dependence`` wants -- so this script builds that triple and
calls the metric directly, passing a plain dict config (not an
``EvaluationConfig`` instance) so ``energy_unit="keV"`` is accepted; MJD
energies are natively keV, and the EnergyBench canonical grid is itself
defined in keV, so nothing is being worked around, just addressed in MJD's
native unit instead of NEXT's.

Events outside the canonical [0, 3000] keV EnergyBench grid are dropped
(reported per run) before scoring -- that cap is the shared EnergyBench
protocol, not an MJD-specific choice, and it fully covers 76Ge's 2039 keV
Q_betabeta, so nothing in the physics region of interest is lost.

Usage::

    python mjd_rope_classifier/scripts/energy_independence.py \\
        --output-root mjd_rope_classifier/results/mjd_rope_classification_v1 \\
        --data-dir /vast/pvenkata/vis224/MJD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def _find_project_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (
            (candidate / "evalutaions_workflow" / "energybench").is_dir()
            and (candidate / "mjd_rope_classifier" / "mjdrope_bench").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "could not locate the project root (needs evalutaions_workflow/ and "
        "mjd_rope_classifier/)"
    )


PROJECT_ROOT = _find_project_root()
for _source_root in (
    PROJECT_ROOT / "evalutaions_workflow",
    PROJECT_ROOT / "mjd_detector",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from energybench.metrics import evaluate_energy_dependence  # noqa: E402
from mjdbench.data import discover_files  # noqa: E402

# The shared EnergyBench protocol's fixed grid (evalutaions_workflow/
# energybench/config.py); events outside it cannot be scored.
CANONICAL_ENERGY_MIN_KEV = 0.0
CANONICAL_ENERGY_MAX_KEV = 3000.0

# Reasonable, protocol-consistent defaults for a non-NEXT detector: a plain
# dict (not EvaluationConfig, which hard-locks energy_unit="MeV") so MJD's
# native keV energies pass straight through.
DEFAULT_CONFIG: dict[str, Any] = {
    "energy_bin_width_kev": 5.0,
    "energy_unit": "keV",
    "score_bins": 20,
    "min_per_bin": 20,
    "distance_correlation_max_samples": 1200,
    "seed": 42,
}


def _load_id_to_energy(data_dir: str | Path) -> dict[int, float]:
    """Map every official MJD_Test_*.hdf5 event id to its energy_label (keV)."""

    mapping: dict[int, float] = {}
    for path in discover_files(data_dir, "test"):
        with h5py.File(path, "r") as handle:
            ids = np.asarray(handle["id"][:], dtype=np.int64)
            energies = np.asarray(handle["energy_label"][:], dtype=np.float64)
        mapping.update(zip(ids.tolist(), energies.tolist()))
    return mapping


def _score_run(
    run_dir: Path,
    cache_test_ids: np.ndarray,
    id_to_energy: dict[int, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.npz"
    predictions = np.load(predictions_path)
    target = np.asarray(predictions["target"], dtype=np.float64)
    score = np.asarray(predictions["prediction"], dtype=np.float64)
    if target.shape[0] != cache_test_ids.shape[0]:
        raise ValueError(
            f"{run_dir.name}: predictions.npz has {target.shape[0]} events, "
            f"cache has {cache_test_ids.shape[0]} -- was this run evaluated "
            "against a different cache?"
        )
    energy = np.asarray(
        [id_to_energy[int(event_id)] for event_id in cache_test_ids],
        dtype=np.float64,
    )

    in_range = (
        np.isfinite(energy)
        & (energy >= CANONICAL_ENERGY_MIN_KEV)
        & (energy <= CANONICAL_ENERGY_MAX_KEV)
    )
    n_total = int(target.size)
    n_dropped = int(n_total - int(np.sum(in_range)))

    dependence = evaluate_energy_dependence(
        target[in_range],
        score[in_range],
        energy[in_range],
        None,  # weights: uniform
        None,  # categories: derive signal/background from the label
        None,  # threshold: not needed for the independence score itself
        config,
    )

    groups = {
        name: {
            "energy_independence_score": metrics.get("energy_independence_score"),
            "n_valid": metrics.get("n_valid"),
            "n_retained": metrics.get("n_retained"),
            "status": metrics.get("status"),
        }
        for name, metrics in dependence.get("groups", {}).items()
    }

    return {
        "run_id": run_dir.name,
        "n_test_events": n_total,
        "n_dropped_out_of_canonical_range": n_dropped,
        "dropped_fraction": n_dropped / n_total if n_total else None,
        "status": dependence.get("status"),
        "definition": dependence.get("definition"),
        "overall_energy_independence_score": dependence.get(
            "overall_energy_independence_score"
        ),
        "worst_group_energy_independence_score": dependence.get(
            "worst_group_energy_independence_score"
        ),
        "groups": groups,
        "config": config,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "mjd_rope_classifier"
        / "results"
        / "mjd_rope_classification_v1",
        help="directory containing one subdirectory per completed run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="/vast/pvenkata/vis224/MJD",
        help="directory with the official MJD_Test_*.hdf5 shards",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help=(
            "the mjdrope_bench cache .pt file the runs were evaluated "
            "against (default: auto-detect the single cache under "
            "MJD_ROPE_CACHE_DIR / <data-dir>/../mjd_rope_cache)"
        ),
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise SystemExit(f"output root does not exist: {output_root}")

    cache_path = args.cache_path
    if cache_path is None:
        cache_dir = Path(args.data_dir).expanduser().resolve().parent / "mjd_rope_cache"
        candidates = sorted(cache_dir.glob("mjd_classification_*.pt"))
        if not candidates:
            raise SystemExit(f"no mjdrope_bench cache found under {cache_dir}")
        if len(candidates) > 1:
            raise SystemExit(
                f"multiple caches under {cache_dir}; pass --cache-path explicitly: "
                + ", ".join(str(path) for path in candidates)
            )
        cache_path = candidates[0]
    print(f"Cache:   {cache_path}")
    cache = torch.load(cache_path, weights_only=False)
    cache_test_ids = cache["test"]["id"].numpy()

    print(f"Data:    {args.data_dir}")
    id_to_energy = _load_id_to_energy(args.data_dir)
    print(f"Loaded energy_label for {len(id_to_energy):,} official test events")

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(output_root.glob("classification__*__rope")):
        if not (run_dir / "predictions.npz").is_file():
            print(f"  ! {run_dir.name}: no predictions.npz, skipping")
            continue
        result = _score_run(run_dir, cache_test_ids, id_to_energy, DEFAULT_CONFIG)
        rows.append(result)
        out_path = run_dir / "energy_independence.json"
        out_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"  {run_dir.name}: overall={result['overall_energy_independence_score']:.4f} "
            f"worst_group={result['worst_group_energy_independence_score']:.4f} "
            f"(dropped {result['n_dropped_out_of_canonical_range']}/"
            f"{result['n_test_events']} events > {CANONICAL_ENERGY_MAX_KEV:g} keV)"
        )
        print(f"    -> {out_path}")

    if not rows:
        raise SystemExit(f"no completed runs with predictions.npz found under {output_root}")

    print("\n=== summary (ranked by overall energy independence, descending) ===")
    for row in sorted(
        rows, key=lambda r: -(r["overall_energy_independence_score"] or -1.0)
    ):
        print(
            f"  {row['run_id']:<40s} overall={row['overall_energy_independence_score']:.4f}"
            f"  worst_group={row['worst_group_energy_independence_score']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
