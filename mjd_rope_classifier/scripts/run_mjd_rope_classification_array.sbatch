#!/bin/bash
# MJD RoPE classification -- the 3-tokenization matrix as one SLURM array job
# on the Pitt CRC `gpu` cluster (h2p).
#
#   SLURM_ARRAY_TASK_ID = tokenization_index
#
#   idx  tokenization
#   ---  ----------------
#    0   raw_patches
#    1   segment_summary
#    2   pulse_entities
#
# All three tasks share position_encoding=rope and read/build the same
# cached 50k/5k/50k subsample (mjdrope_bench.data), so submit task 0 first
# (or with %1) once to build the cache, then the remaining tasks reuse it
# instead of racing to build it in parallel.
#
# Each task writes into its own <OUTPUT_ROOT>/<run_id>/ and only its own
# run_summary.json -- no shared file is written, so the tasks cannot race.
# Merge afterwards with collate_results.py.
#
# The log directory must exist BEFORE submitting (SLURM opens --output before
# the job body runs):
#   mkdir -p mjd_rope_classifier/scripts/logs
#
# Usage (from the repo root):
#   sbatch mjd_rope_classifier/scripts/run_mjd_rope_classification_array.sbatch [output_root]
#
# Build the cache once, then run the other two tokenizations:
#   sbatch --array=0 mjd_rope_classifier/scripts/run_mjd_rope_classification_array.sbatch
#   sbatch --array=1-2 mjd_rope_classifier/scripts/run_mjd_rope_classification_array.sbatch
#
# Re-running is safe: a run whose run_config.json and run_summary.json both
# exist is skipped unless MJD_ROPE_OVERWRITE=1.
#
# rope_base sweep (no code change):
#   sbatch --export=ALL,MJD_ROPE_BASE=1.5,MJD_ROPE_RUN_SUFFIX=_base1.5 \
#     mjd_rope_classifier/scripts/run_mjd_rope_classification_array.sbatch

#SBATCH --job-name=mjd_rope_matrix
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=mjd_rope_classifier/scripts/logs/%x-%A_%a.out

set -euo pipefail

TOKENIZATIONS=(raw_patches segment_summary pulse_entities)

IDX="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"
if (( IDX < 0 || IDX > 2 )); then
    echo "SLURM_ARRAY_TASK_ID must be in 0..2, got ${IDX}" >&2
    exit 2
fi

export MJD_ROPE_TOKENIZATION="${TOKENIZATIONS[$IDX]}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export MJD_ROPE_DATA_DIR="${MJD_ROPE_DATA_DIR:-/vast/pvenkata/vis224/MJD}"
export MJD_ROPE_OUTPUT_ROOT="${1:-${MJD_ROPE_OUTPUT_ROOT:-$PWD/mjd_rope_classifier/results/mjd_rope_classification_v1}}"
export MJD_ROPE_DEVICE="${MJD_ROPE_DEVICE:-auto}"
export MJD_ROPE_NUM_WORKERS="${MJD_ROPE_NUM_WORKERS:-0}"
export MJD_ROPE_EPOCHS="${MJD_ROPE_EPOCHS:-50}"
export MJD_ROPE_BATCH_SIZE="${MJD_ROPE_BATCH_SIZE:-64}"

module purge
module load anaconda3/2023.09-0-python_3.11.5
source activate /ix/pvenkata/vis224/envs/pyT

echo "host:  $(hostname)"
echo "job:   ${SLURM_ARRAY_JOB_ID:-?}_${IDX} on ${SLURM_JOB_PARTITION:-?}"
echo "cell:  ${MJD_ROPE_TOKENIZATION} x rope"
echo "out:   ${MJD_ROPE_OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

srun python mjd_rope_classifier/scripts/run_mjd_rope_classification.py
