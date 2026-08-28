"""Merge three isolated per-model RoPE training outputs into one tree.

Each of the three parallel jobs launched via run_rope_training.sbatch
writes into its own NEXT_OUTPUT_ROOT (results/job_007, job_008, job_009)
to avoid racing on a shared transformer_results.csv. This script merges
those three isolated results/final/ directories into a single combined
tree at results/combined/, in the shape next_energybench_results.ipynb
expects from one NEXT_RESULTS_ROOT.

Usage:
    python collate_results.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("/pscratch/sd/v/vsharma2/0vbb_benchmark/results")

JOB_OUTPUT_ROOTS = [
    RESULTS_ROOT / "job_007",
    RESULTS_ROOT / "job_008",
    RESULTS_ROOT / "job_009",
]

COMBINED_ROOT = RESULTS_ROOT / "combined"
COMBINED_FINAL = COMBINED_ROOT / "final"
COMBINED_SUMMARY_PATH = COMBINED_FINAL / "transformer_results.csv"


def main() -> None:
    COMBINED_FINAL.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    if COMBINED_SUMMARY_PATH.is_file():
        frames.append(pd.read_csv(COMBINED_SUMMARY_PATH))

    for job_root in JOB_OUTPUT_ROOTS:
        job_final = job_root / "final"
        job_summary_path = job_final / "transformer_results.csv"

        if not job_summary_path.is_file():
            raise FileNotFoundError(
                f"No completed summary at {job_summary_path}. "
                "Has this job's training+evaluation finished?"
            )

        frames.append(pd.read_csv(job_summary_path))

        for model_dir in job_final.iterdir():
            if not model_dir.is_dir():
                continue

            destination = COMBINED_FINAL / model_dir.name
            if destination.exists():
                print(
                    f"Skipping move, already present: {destination}",
                    file=sys.stderr,
                )
                continue

            shutil.move(str(model_dir), str(destination))

    combined = pd.concat(frames, ignore_index=True)

    duplicates = combined["model_id"][combined["model_id"].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate model_id rows after merge: {duplicates}")

    combined = combined.sort_values("model_id").reset_index(drop=True)

    temporary_path = COMBINED_SUMMARY_PATH.with_suffix(".csv.tmp")
    combined.to_csv(temporary_path, index=False)
    temporary_path.replace(COMBINED_SUMMARY_PATH)

    print(f"Combined {len(combined)} model rows into {COMBINED_SUMMARY_PATH}")
    for model_id in combined["model_id"]:
        print(" -", model_id)


if __name__ == "__main__":
    main()
