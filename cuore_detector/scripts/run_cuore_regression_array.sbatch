#!/bin/bash
# CUORE first-two-pulse Δt regression -- the full 3 x 3 matrix as one SLURM
# array job on the Pitt CRC `gpu` cluster (h2p).
#
#   SLURM_ARRAY_TASK_ID = 3 * tokenization_index + encoding_index
#
#   idx  tokenization      position_encoding
#   ---  ----------------  -------------------
#    0   raw_patches       coordinate_mlp
#    1   raw_patches       fourier_coordinates
#    2   raw_patches       rope
#    3   segment_summary   coordinate_mlp
#    4   segment_summary   fourier_coordinates
#    5   segment_summary   rope
#    6   pulse_entities    coordinate_mlp
#    7   pulse_entities    fourier_coordinates
#    8   pulse_entities    rope
#
# Each task writes into its own <OUTPUT_ROOT>/<run_id>/ and only its own
# run_summary.json -- no shared file is written, so the tasks cannot race.
# Merge afterwards with collate_results.py.
#
# The log directory must exist BEFORE submitting (SLURM opens --output before
# the job body runs):
#   mkdir -p cuore_detector/scripts/logs
#
# Usage (from the repo root):
#   sbatch cuore_detector/scripts/run_cuore_regression_array.sbatch [output_root]
#
# Throttle to 3 concurrent tasks if the partition is contended:
#   sbatch --array=0-8%3 cuore_detector/scripts/run_cuore_regression_array.sbatch
#
# Re-running is safe: a run whose run_config.json and run_summary.json both
# exist is skipped unless CUORE_OVERWRITE=1.
#
# rope_base sweep (no code change):
#   sbatch --array=2,5,8 --export=ALL,CUORE_ROPE_BASE=32,CUORE_RUN_SUFFIX=_base32 \
#     cuore_detector/scripts/run_cuore_regression_array.sbatch

#SBATCH --job-name=cuore_matrix
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=cuore_detector/scripts/logs/%x-%A_%a.out

set -euo pipefail

TOKENIZATIONS=(raw_patches segment_summary pulse_entities)
ENCODINGS=(coordinate_mlp fourier_coordinates rope)

IDX="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"
if (( IDX < 0 || IDX > 8 )); then
    echo "SLURM_ARRAY_TASK_ID must be in 0..8, got ${IDX}" >&2
    exit 2
fi

export CUORE_TOKENIZATION="${TOKENIZATIONS[$(( IDX / 3 ))]}"
export CUORE_POSITION_ENCODING="${ENCODINGS[$(( IDX % 3 ))]}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export CUORE_DATA_DIR="${CUORE_DATA_DIR:-/vast/pvenkata/vis224/CUORE}"
export CUORE_OUTPUT_ROOT="${1:-${CUORE_OUTPUT_ROOT:-$PWD/cuore_detector/results/cuore_matrix_v1}}"
export CUORE_DEVICE="${CUORE_DEVICE:-auto}"
export CUORE_NUM_WORKERS="${CUORE_NUM_WORKERS:-0}"
export CUORE_EPOCHS="${CUORE_EPOCHS:-50}"
export CUORE_BATCH_SIZE="${CUORE_BATCH_SIZE:-64}"

module purge
module load anaconda3/2023.09-0-python_3.11.5
source activate /ix/pvenkata/vis224/envs/pyT

echo "host:  $(hostname)"
echo "job:   ${SLURM_ARRAY_JOB_ID:-?}_${IDX} on ${SLURM_JOB_PARTITION:-?}"
echo "cell:  ${CUORE_TOKENIZATION} x ${CUORE_POSITION_ENCODING}"
echo "out:   ${CUORE_OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

srun python cuore_detector/scripts/run_cuore_regression.py
