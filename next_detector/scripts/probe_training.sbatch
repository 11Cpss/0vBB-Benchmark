#!/bin/bash
# One-off sanity/timing probe before the real RoPE training runs.
#
# Trains transformer_007_sampled_hits_rope against the real extracted NEXT
# dataset into an isolated, throwaway output root (results/probe), just to
# confirm the official split/count check passes and to read real per-epoch
# wall-clock timing from the log -- used to size the walltime for the real
# runs (run_rope_training.sbatch). The 1-hour walltime cap means this job
# will likely be killed mid-training; that's expected.
#
# Uses gpu_shared, not gpu_debug: gpu_debug requires whole-node allocation
# on Perlmutter (MaxTRES=node=8, no fractional-GPU support), which a
# single-GPU job doesn't match ("Job request does not match any supported
# policy"). gpu_shared supports a single-GPU request directly and still
# only bills for the GPU-time actually used.
#
# Cleanup after the probe: `rm -r` the whole results/probe directory --
# it's isolated from the real per-model output roots, so no reconciliation
# with transformer_results.csv is needed.
#
# Usage:
#   sbatch probe_training.sbatch
#
#SBATCH -A m2812_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o next_detector/scripts/logs/%x-%j.out

set -euo pipefail

module load conda
source activate pyT

export SIMPLE_ENERGYBENCH_DATA=/pscratch/sd/v/vsharma2/0vbb_benchmark/NEXT
export NEXT_OUTPUT_ROOT=/pscratch/sd/v/vsharma2/0vbb_benchmark/results/probe
export NEXT_RUN_MODEL_IDS=transformer_007_sampled_hits_rope

cd "$SLURM_SUBMIT_DIR"

# Convert to a plain .py script and run it directly, rather than
# `nbconvert --execute` on the notebook itself: the notebook form only
# writes captured cell output at the very end of a successful run, so a
# job killed by the walltime (expected for this probe) would leave the
# log with nothing in it -- no epoch timing, no confirmation the data
# loaded. Running as a script streams print() output to this log live,
# so partial progress survives a walltime kill.
jupyter nbconvert --to script \
  --output next_energybench_train.probe \
  --output-dir next_detector/notebooks \
  next_detector/notebooks/next_energybench_train.ipynb
# The notebook's language_info has no file_extension, so nbconvert
# writes .txt instead of .py -- rename before running.
mv next_detector/notebooks/next_energybench_train.probe.txt \
   next_detector/notebooks/next_energybench_train.probe.py
python next_detector/notebooks/next_energybench_train.probe.py
