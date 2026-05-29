#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  "${PYTHON_BIN}" -m wandb login --relogin "${WANDB_API_KEY}"
else
  "${PYTHON_BIN}" -m wandb login
fi
