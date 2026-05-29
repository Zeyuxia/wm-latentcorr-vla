#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
PYTHON_BIN=${PYTHON_BIN:-${UV_PROJECT}/.venv/bin/python}
POLICY_NAME=PI05_LatentCorr
TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
CKPT_SETTING=${CKPT_SETTING:-pi0_latent_stage1}
STAGE1_CKPT=${STAGE1_CKPT:-}
INFERENCE_MODE=${INFERENCE_MODE:-bridge}
SEED=${SEED:-0}
SEED_FILE=${SEED_FILE:-}
GPU_ID=${GPU_ID:-0}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
CAMERA_MODE=${CAMERA_MODE:-head_only}
PI0_STEP=${PI0_STEP:-50}
NUM_STEPS=${NUM_STEPS:-10}
ASSET_REPO_ID=${ASSET_REPO_ID:-}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${SCRIPT_DIR}/outputs/checkpoints}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR:-${SCRIPT_DIR}/outputs/assets}
DEVICE=${DEVICE:-cuda:0}

EVAC_CKPT=${EVAC_CKPT:-}
EVAC_CONFIG=${EVAC_CONFIG:-}
URDF_PATH=${URDF_PATH:-}

if [ -z "${STAGE1_CKPT}" ]; then
  echo "STAGE1_CKPT is required" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}

TAG_ARGS=()
if [ -n "${EVAL_TAG:-}" ]; then
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

EXTRA_ARGS=()
if [ -n "${EVAC_CKPT}" ]; then
  EXTRA_ARGS+=(--evac_ckpt "${EVAC_CKPT}")
fi
if [ -n "${EVAC_CONFIG}" ]; then
  EXTRA_ARGS+=(--evac_config "${EVAC_CONFIG}")
fi
if [ -n "${URDF_PATH}" ]; then
  EXTRA_ARGS+=(--urdf_path "${URDF_PATH}")
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
  --ckpt_setting "${CKPT_SETTING}" \
  --policy_name "${POLICY_NAME}" \
  --train_config_name "${TRAIN_CONFIG_NAME}" \
  --stage1_ckpt "${STAGE1_CKPT}" \
  --latent_ckpt_path "${STAGE1_CKPT}" \
  --inference_mode "${INFERENCE_MODE}" \
  --camera_mode "${CAMERA_MODE}" \
  --pi0_step "${PI0_STEP}" \
  --num_steps "${NUM_STEPS}" \
  --seed "${SEED}" \
  --checkpoint_base_dir "${CHECKPOINT_BASE_DIR}" \
  --assets_base_dir "${ASSETS_BASE_DIR}" \
  --device "${DEVICE}" \
  "${ASSET_ARGS[@]}" \
  "${SEED_FILE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${TAG_ARGS[@]}"
