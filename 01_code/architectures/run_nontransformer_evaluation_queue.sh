#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wenyu/summer"
ENERGYBENCH_BIN="${PROJECT_ROOT}/.venv/bin/energybench"
DATA_ROOT="/home/klz/Data/zeronu_benchmark/NEXT"
EVALUATION_ROOT="${PROJECT_ROOT}/04_evaluations"
LOG_DIR="${EVALUATION_ROOT}/logs"
RUN_ID="20260803_200356"
CHECKPOINT_ROOT="${PROJECT_ROOT}/02_models/checkpoints/${RUN_ID}"
COMPARISON_DIR="${EVALUATION_ROOT}/NEXTALT_nontransformer_v2_${RUN_ID}_comparison"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="${LOG_DIR}/nontransformer_v2_evaluation_queue_${RUN_STAMP}.log"

architectures=(
  "classic_001_topology_xgboost"
  "point_003_pointmlp"
  "seq_001_bigru"
  "seq_002_dilated_tcn"
  "mixer_001_projection_mlp_mixer"
  "gnn_005_dimenet_lite"
  "point_004_rigid_kpconv"
  "topo_001_persistence_perslay"
  "ssm_001_pointmamba"
  "sparse_001_submanifold_resnet"
)

declare -A batch_sizes=(
  [classic_001_topology_xgboost]=128
  [point_003_pointmlp]=12
  [seq_001_bigru]=16
  [seq_002_dilated_tcn]=12
  [mixer_001_projection_mlp_mixer]=16
  [gnn_005_dimenet_lite]=4
  [point_004_rigid_kpconv]=8
  [topo_001_persistence_perslay]=16
  [ssm_001_pointmamba]=4
  [sparse_001_submanifold_resnet]=8
)

checkpoint_for() {
  local architecture_id="$1"
  local extension="pt"
  if [[ "${architecture_id}" == "classic_001_topology_xgboost" ]]; then
    extension="json"
  fi
  printf '%s/%s/attempt_001/best.%s' \
    "${CHECKPOINT_ROOT}" "${architecture_id}" "${extension}"
}

if [[ ! -x "${ENERGYBENCH_BIN}" ]]; then
  echo "EnergyBench executable is missing: ${ENERGYBENCH_BIN}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "NEXT dataset is missing: ${DATA_ROOT}" >&2
  exit 1
fi

for architecture_id in "${architectures[@]}"; do
  checkpoint="$(checkpoint_for "${architecture_id}")"
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

echo "NEXT non-Transformer v2 full-test evaluation queue"
echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Training campaign: ${RUN_ID}"
echo "EnergyBench: ${ENERGYBENCH_BIN}"
echo "Dataset: ${DATA_ROOT}"
echo "Split: test (all files)"
echo "Protocol: strict"
echo "Log: ${QUEUE_LOG}"

failures=()
evaluation_dirs=()

for architecture_id in "${architectures[@]}"; do
  checkpoint="$(checkpoint_for "${architecture_id}")"
  output_dir="${EVALUATION_ROOT}/NEXTALT_${architecture_id}_test"
  evaluation_dir="${output_dir}/evaluation_test"
  batch_size="${batch_sizes[${architecture_id}]}"
  device="cuda:0"
  if [[ "${architecture_id}" == "classic_001_topology_xgboost" ]]; then
    device="cpu"
  fi

  echo
  echo "[$(date --iso-8601=seconds)] START ${architecture_id} (device=${device}, batch=${batch_size})"
  if "${ENERGYBENCH_BIN}" next "${checkpoint}" \
    --data "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --model-id "${architecture_id}" \
    --device "${device}" \
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
echo "Completed: $(date --iso-8601=seconds)"
echo "Comparison: ${COMPARISON_DIR}/leaderboard.csv"
