#!/bin/bash
# SuperNEMO RoPE classification -- the 3-tokenization matrix as one SLURM array
# job on the Pitt CRC `gpu` cluster (h2p).
#
#   SLURM_ARRAY_TASK_ID = tokenization_index
#
#   idx  tokenization
#   ---  ----------------
#    0   sampled_hits
#    1   voxel
#    2   summary_features
#
# All three tasks use position_encoding=rope, the published SuperNEMO split
# (2nubb vs Bi214, seed 42, 80/10/10 by 4096-event blocks), and stream events
# straight from the raw HDF5 files -- there is no cache to build, so the tasks
# do not depend on one another.
#
# Each task writes into its own <OUTPUT_ROOT>/<run_id>/ and only its own
# run_summary.json -- no shared file is written, so the tasks cannot race.
# Merge afterwards with collate_results.py.
#
# The log directory must exist BEFORE submitting (SLURM opens --output before
# the job body runs; the repo ships it with a .gitkeep):
#   mkdir -p supernemo_rope_classifier/scripts/logs
#
# Usage (from the repo root):
#   sbatch supernemo_rope_classifier/scripts/run_supernemo_rope_classification_array.sbatch [output_root]
#
# WALLTIME IS AN ESTIMATE, NOT A MEASUREMENT. The published split has 4.74M
# training events (74k steps/epoch at batch 64) and up to 50 epochs. The data
# pipeline alone sustains 5-19k events/s per worker (measured on CPU), so the
# GPU step rate should set the pace; time one capped epoch first and size
# --time from it:
#   sbatch --array=1 --time=00:30:00 \
#     --export=ALL,SUPERNEMO_ROPE_EPOCHS=1,SUPERNEMO_ROPE_MAX_TRAIN_EVENTS=200000,SUPERNEMO_ROPE_MAX_VALIDATION_EVENTS=20000,SUPERNEMO_ROPE_MAX_TEST_EVENTS=20000,SUPERNEMO_ROPE_RUN_SUFFIX=_probe \
#     supernemo_rope_classifier/scripts/run_supernemo_rope_classification_array.sbatch
#
# Re-running is safe: a run whose run_config.json and run_summary.json both
# exist is skipped unless SUPERNEMO_ROPE_OVERWRITE=1.
#
# rope_base sweep (no code change):
#   sbatch --export=ALL,SUPERNEMO_ROPE_BASE=16,SUPERNEMO_ROPE_RUN_SUFFIX=_base16 \
#     supernemo_rope_classifier/scripts/run_supernemo_rope_classification_array.sbatch
#
# Like-for-like position-encoding controls through the same code path:
#   sbatch --export=ALL,SUPERNEMO_ROPE_POSITION_ENCODING=coordinate_mlp \
#     supernemo_rope_classifier/scripts/run_supernemo_rope_classification_array.sbatch

#SBATCH --job-name=supernemo_rope_matrix
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=supernemo_rope_classifier/scripts/logs/%x-%A_%a.out

set -euo pipefail

TOKENIZATIONS=(sampled_hits voxel summary_features)

IDX="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"
if (( IDX < 0 || IDX > 2 )); then
    echo "SLURM_ARRAY_TASK_ID must be in 0..2, got ${IDX}" >&2
    exit 2
fi

export SUPERNEMO_ROPE_TOKENIZATION="${TOKENIZATIONS[$IDX]}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export SUPERNEMO_ROPE_DATA_DIR="${SUPERNEMO_ROPE_DATA_DIR:-/vast/pvenkata/vis224/SuperNEMO}"
export SUPERNEMO_ROPE_OUTPUT_ROOT="${1:-${SUPERNEMO_ROPE_OUTPUT_ROOT:-$PWD/supernemo_rope_classifier/results/supernemo_rope_classification_v1}}"
export SUPERNEMO_ROPE_DEVICE="${SUPERNEMO_ROPE_DEVICE:-auto}"
# 6 loader workers leave 2 of the 8 CPUs for the main process. One worker
# measured 5-19k events/s depending on the tokenizer (slowest: summary_features);
# scaling to 6 workers was not measured.
export SUPERNEMO_ROPE_NUM_WORKERS="${SUPERNEMO_ROPE_NUM_WORKERS:-6}"
export SUPERNEMO_ROPE_EPOCHS="${SUPERNEMO_ROPE_EPOCHS:-50}"
export SUPERNEMO_ROPE_BATCH_SIZE="${SUPERNEMO_ROPE_BATCH_SIZE:-64}"

module purge
module load anaconda3/2023.09-0-python_3.11.5
source activate /ix/pvenkata/vis224/envs/pyT

echo "host:  $(hostname)"
echo "job:   ${SLURM_ARRAY_JOB_ID:-?}_${IDX} on ${SLURM_JOB_PARTITION:-?}"
echo "cell:  ${SUPERNEMO_ROPE_TOKENIZATION} x ${SUPERNEMO_ROPE_POSITION_ENCODING:-rope}"
echo "out:   ${SUPERNEMO_ROPE_OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

srun python supernemo_rope_classifier/scripts/run_supernemo_rope_classification.py
