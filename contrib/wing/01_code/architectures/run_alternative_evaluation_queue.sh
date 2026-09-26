#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wenyu/summer"
ENERGYBENCH_BIN="${PROJECT_ROOT}/.venv/bin/energybench"
CHECKPOINT_DIR="${PROJECT_ROOT}/02_models/checkpoints"
EVALUATION_ROOT="${PROJECT_ROOT}/04_evaluations"
LOG_DIR="${EVALUATION_ROOT}/logs"
COMPARISON_DIR="${EVALUATION_ROOT}/NEXTALT_all_models_comparison"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="${LOG_DIR}/alternative_evaluation_queue_${RUN_STAMP}.log"

architectures=(
  "cnn_004_multiview_late_fusion"
  "point_001_deepsets"
  "gnn_001_static_gine"
  "point_002_pointnetpp"
  "gnn_002_particlenet_edgeconv"
  "gnn_003_egnn"
  "gnn_004_gravnet"
  "cnn_005_multiscale_projection"
  "hybrid_001_cnn_gnn"
  "cnn_006_dense_3d_resnet"
)

# Conservative inference batches: each value is no larger than the batch used
# to train that architecture. This keeps the queue stable across the 2D, 3D,
# point-cloud, graph, and hybrid models without changing the test protocol.
declare -A batch_sizes=(
  [cnn_004_multiview_late_fusion]=16
  [point_001_deepsets]=64
  [gnn_001_static_gine]=16
  [point_002_pointnetpp]=16
  [gnn_002_particlenet_edgeconv]=12
  [gnn_003_egnn]=12
  [gnn_004_gravnet]=12
  [cnn_005_multiscale_projection]=8
  [hybrid_001_cnn_gnn]=8
  [cnn_006_dense_3d_resnet]=2
)

if [[ ! -x "${ENERGYBENCH_BIN}" ]]; then
  echo "EnergyBench executable is missing: ${ENERGYBENCH_BIN}" >&2
  exit 1
fi

# Check every immutable input and every strict output destination before doing
# any GPU work. EnergyBench itself also refuses to overwrite existing results.
for architecture_id in "${architectures[@]}"; do
  checkpoint="${CHECKPOINT_DIR}/NEXTALT_${architecture_id}_classification_best.pt"
  output_dir="${EVALUATION_ROOT}/NEXTALT_${architecture_id}_test"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing best checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "Evaluation output already exists: ${output_dir}" >&2
    exit 1
  fi
done

if [[ -e "${COMPARISON_DIR}" ]]; then
  echo "Comparison output already exists: ${COMPARISON_DIR}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "Alternative NEXT full-test evaluation queue"
echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "EnergyBench: ${ENERGYBENCH_BIN}"
echo "Split: test (all files)"
echo "Protocol: strict"
echo "Log: ${QUEUE_LOG}"

failures=()
evaluation_dirs=()

for architecture_id in "${architectures[@]}"; do
  checkpoint="${CHECKPOINT_DIR}/NEXTALT_${architecture_id}_classification_best.pt"
  output_dir="${EVALUATION_ROOT}/NEXTALT_${architecture_id}_test"
  evaluation_dir="${output_dir}/evaluation_test"
  batch_size="${batch_sizes[${architecture_id}]}"

  echo
  echo "[$(date --iso-8601=seconds)] START ${architecture_id} (batch=${batch_size})"
  if "${ENERGYBENCH_BIN}" next "${checkpoint}" \
    --output-dir "${output_dir}" \
    --model-id "${architecture_id}" \
    --device cuda:0 \
    --batch-size "${batch_size}" \
    --num-workers 0 \
    --split test; then
    echo "[$(date --iso-8601=seconds)] DONE  ${architecture_id}"
    evaluation_dirs+=("${evaluation_dir}")
  else
    status=$?
    echo "[$(date --iso-8601=seconds)] FAIL  ${architecture_id} (exit=${status})" >&2
    failures+=("${architecture_id}:${status}")
  fi
done

echo
if (( ${#failures[@]} > 0 )); then
  echo "All architectures were attempted, but ${#failures[@]} evaluation(s) failed:" >&2
  printf '  %s\n' "${failures[@]}" >&2
  echo "Comparison was skipped because strict ranking requires every requested result." >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] START unified comparison"
"${ENERGYBENCH_BIN}" compare \
  "${evaluation_dirs[@]}" \
  --output-dir "${COMPARISON_DIR}"
echo "[$(date --iso-8601=seconds)] DONE  unified comparison"

echo
echo "Completed: $(date --iso-8601=seconds)"
echo "Comparison: ${COMPARISON_DIR}/leaderboard.csv"
