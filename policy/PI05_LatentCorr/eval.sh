#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
PYTHON_BIN=${PYTHON_BIN:-${UV_PROJECT}/.venv/bin/python}
POLICY_NAME=${POLICY_NAME:-PI05_LatentCorr}
TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
MODEL_NAME=${MODEL_NAME:-pi05_openloop_default}
CHECKPOINT_ID=${CHECKPOINT_ID:-latest}
CAMERA_MODE=${CAMERA_MODE:-head_only}
PI0_STEP=${PI0_STEP:-50}
SEED=${SEED:-0}
GPU_ID=${GPU_ID:-0}
ASSET_REPO_ID=${ASSET_REPO_ID:-}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${SCRIPT_DIR}/outputs/checkpoints}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR:-${SCRIPT_DIR}/outputs/assets}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${SCRIPT_DIR}/outputs/openpi_cache}
TMPDIR=${TMPDIR:-${SCRIPT_DIR}/outputs/tmp}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}
EVAL_TAG=${EVAL_TAG:-}
SEED_FILE=${SEED_FILE:-}
DEVICE=${DEVICE:-cuda:0}
CUROBO_SRC=${CUROBO_SRC:-/data/changguo/if-bench/envs/curobo/src}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export XLA_PYTHON_CLIENT_MEM_FRACTION
mkdir -p "${OPENPI_DATA_HOME}" "${TMPDIR}"
export OPENPI_DATA_HOME
export TMPDIR
if [ -d "${CUROBO_SRC}" ]; then
  if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${CUROBO_SRC}:${PYTHONPATH}"
  else
    export PYTHONPATH="${CUROBO_SRC}"
  fi
fi

TAG_ARGS=()
if [ -n "${EVAL_TAG}" ]; then
  TAG_ARGS+=(--eval_tag "${EVAL_TAG}")
fi

SEED_FILE_ARGS=()
if [ -n "${SEED_FILE}" ]; then
  SEED_FILE_ARGS+=(--seed_file "${SEED_FILE}")
fi

ASSET_ARGS=()
if [ -n "${ASSET_REPO_ID}" ]; then
  ASSET_ARGS+=(--asset_repo_id "${ASSET_REPO_ID}")
fi

RUN_CMD=()
if [ -x "${PYTHON_BIN}" ]; then
  RUN_CMD=("${PYTHON_BIN}")
else
  RUN_CMD=(uv run --project "${UV_PROJECT}" python)
fi

PYTHONWARNINGS=ignore::UserWarning \
"${RUN_CMD[@]}" script/eval_policy.py --config "policy/${POLICY_NAME}/deploy_policy.yml" \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --train_config_name "${TRAIN_CONFIG_NAME}" \
  --model_name "${MODEL_NAME}" \
  --checkpoint_id "${CHECKPOINT_ID}" \
  --camera_mode "${CAMERA_MODE}" \
  --pi0_step "${PI0_STEP}" \
  --ckpt_setting "${MODEL_NAME}" \
  --seed "${SEED}" \
  --policy_name "${POLICY_NAME}" \
  --checkpoint_base_dir "${CHECKPOINT_BASE_DIR}" \
  --assets_base_dir "${ASSETS_BASE_DIR}" \
  --device "${DEVICE}" \
  "${ASSET_ARGS[@]}" \
  "${SEED_FILE_ARGS[@]}" \
  "${TAG_ARGS[@]}"
