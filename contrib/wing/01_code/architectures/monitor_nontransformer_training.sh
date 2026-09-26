#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/wenyu/summer"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
STATUS_SCRIPT="${PROJECT_ROOT}/01_code/architectures/nontransformer_campaign_status.py"
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  echo "--run-id is required" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
while true; do
  clear 2>/dev/null || true
  date --iso-8601=seconds
  "${PYTHON_BIN}" "${STATUS_SCRIPT}" --run-id "${RUN_ID}"
  STATUS=$?
  echo
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader 2>&1 || true
  if [[ ${STATUS} -eq 0 ]]; then
    echo
    echo "Campaign complete; monitor will close in 30 seconds."
    sleep 30
    exit 0
  fi
  sleep 15
done
