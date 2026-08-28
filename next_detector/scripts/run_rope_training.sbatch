#!/bin/bash
# Real (non-probe) headless training run for one NEXT Transformer model_id,
# via `jupyter nbconvert --execute` on next_energybench_train.ipynb.
#
# Meant to be run three times in parallel, once per new RoPE model_id, each
# with its own isolated NEXT_OUTPUT_ROOT so the jobs never share a file:
# next_energybench_train.ipynb's save_summary() rewrites the whole
# transformer_results.csv on every completed model, so concurrent jobs
# sharing one output root would race and drop each other's rows. Collate
# the three isolated results afterward with collate_results.py.
#
# QOS and walltime aren't baked in here -- size them from the probe run
# (probe_training.sbatch) and pass them at submit time.
#
# Note: use QOS "shared", not "gpu_shared" -- "gpu_shared" is a valid QOS
# name for this account but doesn't resolve to a real partition here
# ("Job request does not match any supported policy"); plain "shared"
# combined with -C gpu correctly resolves to partition shared_gpu_ss11.
#
# Usage:
#   sbatch --qos=shared -t <HH:MM:SS> \
#     run_rope_training.sbatch <model_id> <output_root>
#
# Example:
#   sbatch --qos=shared -t 04:00:00 run_rope_training.sbatch \
#     transformer_007_sampled_hits_rope \
#     /pscratch/sd/v/vsharma2/0vbb_benchmark/results/job_007
#
#SBATCH -A m2812_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o next_detector/scripts/logs/%x-%j.out

set -euo pipefail

MODEL_ID="$1"
OUTPUT_ROOT="$2"

module load conda
source activate pyT

export SIMPLE_ENERGYBENCH_DATA=/pscratch/sd/v/vsharma2/0vbb_benchmark/NEXT
export NEXT_OUTPUT_ROOT="$OUTPUT_ROOT"
export NEXT_RUN_MODEL_IDS="$MODEL_ID"

cd "$SLURM_SUBMIT_DIR"

# Convert to a plain .py script and run it directly, rather than
# `nbconvert --execute` on the notebook itself: the notebook form only
# writes captured cell output at the very end of a successful run, so a
# killed or crashed job would leave the log with nothing in it. Running
# as a script streams print() output to this log live. Each parallel
# job gets its own script filename (suffixed by model_id) so the three
# concurrent jobs don't overwrite each other's converted script.
jupyter nbconvert --to script \
  --output "next_energybench_train.${MODEL_ID}" \
  --output-dir next_detector/notebooks \
  next_detector/notebooks/next_energybench_train.ipynb
# The notebook's language_info has no file_extension, so nbconvert
# writes .txt instead of .py -- rename before running.
mv "next_detector/notebooks/next_energybench_train.${MODEL_ID}.txt" \
   "next_detector/notebooks/next_energybench_train.${MODEL_ID}.py"
python "next_detector/notebooks/next_energybench_train.${MODEL_ID}.py"
