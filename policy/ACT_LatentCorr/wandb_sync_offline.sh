#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python}
SEARCH_ROOT=${SEARCH_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mapfile -t OFFLINE_RUNS < <(find "${SEARCH_ROOT}" -type d -name 'offline-run-*' | sort)

if [ "${#OFFLINE_RUNS[@]}" -eq 0 ]; then
  echo "no offline wandb runs found under ${SEARCH_ROOT}"
  exit 0
fi

for run_dir in "${OFFLINE_RUNS[@]}"; do
  echo "[wandb-sync] ${run_dir}"
  "${PYTHON_BIN}" -m wandb sync "${run_dir}"
done
