#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wenyu/summer"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
CAMPAIGN_RUNNER="${PROJECT_ROOT}/01_code/architectures/nontransformer_campaign.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Project Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${CAMPAIGN_RUNNER}" "$@"
