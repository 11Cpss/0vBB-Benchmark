#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wenyu/summer"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
LOG_DIR="${PROJECT_ROOT}/03_training_runs/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="${LOG_DIR}/alternative_training_queue_${RUN_STAMP}.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Project Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

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

echo "Alternative NEXT training queue"
echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Python: ${PYTHON_BIN}"
echo "Log: ${QUEUE_LOG}"

for architecture_id in "${architectures[@]}"; do
  entrypoint="${PROJECT_ROOT}/01_code/architectures/${architecture_id}/train_classification.py"
  if [[ ! -f "${entrypoint}" ]]; then
    echo "Missing training entry point: ${entrypoint}" >&2
    exit 1
  fi
  echo
  echo "[$(date --iso-8601=seconds)] START ${architecture_id}"
  "${PYTHON_BIN}" "${entrypoint}"
  echo "[$(date --iso-8601=seconds)] DONE  ${architecture_id}"
done

echo
echo "Completed: $(date --iso-8601=seconds)"
