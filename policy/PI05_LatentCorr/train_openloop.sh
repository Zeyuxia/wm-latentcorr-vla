#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CAMERA_MODE=${CAMERA_MODE:-head_only}
PROXY_ON=${PROXY_ON:-false}
HTTP_PROXY_URL=${HTTP_PROXY_URL:-http://127.0.0.1:7890}
HTTPS_PROXY_URL=${HTTPS_PROXY_URL:-http://127.0.0.1:7890}
ALL_PROXY_URL=${ALL_PROXY_URL:-socks5://127.0.0.1:10808}
DEFAULT_REPO_SUFFIX=""
if [ "${CAMERA_MODE}" = "head_only" ]; then
  DEFAULT_REPO_SUFFIX="_headonly"
fi
REPO_ID=${REPO_ID:-robotwin/open_laptop_demo_clean_50${DEFAULT_REPO_SUFFIX}}
ASSET_ID=${ASSET_ID:-}
EXP_NAME=${EXP_NAME:-pi05_openloop_$(date +"%Y%m%d_%H%M%S")}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR:-${SCRIPT_DIR}/outputs/assets}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${SCRIPT_DIR}/outputs/checkpoints}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${SCRIPT_DIR}/outputs/openpi_cache}
TMPDIR=${TMPDIR:-${SCRIPT_DIR}/outputs/tmp}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
KEEP_PERIOD=${KEEP_PERIOD:-100}
FSDP_DEVICES=${FSDP_DEVICES:-8}
SEED=${SEED:-42}
OVERWRITE=${OVERWRITE:-false}
RESUME=${RESUME:-false}
WANDB_ENABLED=${WANDB_ENABLED:-true}
PROJECT_NAME=${PROJECT_NAME:-RoboTwin_PI05_LatentCorr}
SKIP_NORM_STATS=${SKIP_NORM_STATS:-false}
NORM_MAX_FRAMES=${NORM_MAX_FRAMES:-}
OPENLOOP_FRAMEWORK=${OPENLOOP_FRAMEWORK:-auto}
PREFER_OFFICIAL_JAX=${PREFER_OFFICIAL_JAX:-true}
DEFAULT_PYTORCH_WEIGHT_PATH="${SCRIPT_DIR}/lerobot/pi05_base/model.safetensors"
DEFAULT_JAX_PARAMS_PATH="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base/params"
JAX_PARAMS_PATH=${JAX_PARAMS_PATH:-}
PYTORCH_WEIGHT_PATH=${PYTORCH_WEIGHT_PATH:-}
PYTORCH_TRAINING_PRECISION=${PYTORCH_TRAINING_PRECISION:-}
PYTORCH_GRADIENT_CHECKPOINTING=${PYTORCH_GRADIENT_CHECKPOINTING:-true}
PYTORCH_FIND_UNUSED_PARAMETERS=${PYTORCH_FIND_UNUSED_PARAMETERS:-true}
PYTORCH_ENABLE_COMPILE=${PYTORCH_ENABLE_COMPILE:-false}

if [ -z "${PYTORCH_WEIGHT_PATH}" ] && [ -f "${DEFAULT_PYTORCH_WEIGHT_PATH}" ]; then
  PYTORCH_WEIGHT_PATH="${DEFAULT_PYTORCH_WEIGHT_PATH}"
fi

if [ -z "${JAX_PARAMS_PATH}" ] && [ -d "${DEFAULT_JAX_PARAMS_PATH}" ]; then
  JAX_PARAMS_PATH="${DEFAULT_JAX_PARAMS_PATH}"
fi

if [ "${OPENLOOP_FRAMEWORK}" = "auto" ]; then
  if [ -n "${JAX_PARAMS_PATH}" ] && [ -d "${JAX_PARAMS_PATH}" ]; then
    OPENLOOP_FRAMEWORK="jax"
  elif [ "${PREFER_OFFICIAL_JAX}" = "true" ]; then
    OPENLOOP_FRAMEWORK="jax"
  elif [ -n "${PYTORCH_WEIGHT_PATH}" ] && [ -f "${PYTORCH_WEIGHT_PATH}" ]; then
    OPENLOOP_FRAMEWORK="pytorch"
  else
    OPENLOOP_FRAMEWORK="jax"
  fi
fi

if [ "${PROXY_ON}" = "true" ]; then
  export http_proxy="${HTTP_PROXY_URL}"
  export https_proxy="${HTTPS_PROXY_URL}"
  export HTTP_PROXY="${HTTP_PROXY_URL}"
  export HTTPS_PROXY="${HTTPS_PROXY_URL}"
  export all_proxy="${ALL_PROXY_URL}"
  export ALL_PROXY="${ALL_PROXY_URL}"
  echo "[PI05_LatentCorr] proxy enabled: ${HTTP_PROXY_URL}"
fi

EXTRA_ARGS=()
if [ -n "${ASSET_ID}" ]; then
  EXTRA_ARGS+=(--asset-id "${ASSET_ID}")
fi
if [ -n "${BATCH_SIZE}" ]; then
  EXTRA_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [ -n "${NUM_WORKERS}" ]; then
  EXTRA_ARGS+=(--num-workers "${NUM_WORKERS}")
fi
if [ -n "${NUM_TRAIN_STEPS}" ]; then
  EXTRA_ARGS+=(--num-train-steps "${NUM_TRAIN_STEPS}")
fi
if [ -n "${LOG_INTERVAL}" ]; then
  EXTRA_ARGS+=(--log-interval "${LOG_INTERVAL}")
fi
if [ -n "${SAVE_INTERVAL}" ]; then
  EXTRA_ARGS+=(--save-interval "${SAVE_INTERVAL}")
fi
if [ -n "${KEEP_PERIOD}" ]; then
  EXTRA_ARGS+=(--keep-period "${KEEP_PERIOD}")
fi
if [ -n "${FSDP_DEVICES}" ]; then
  EXTRA_ARGS+=(--fsdp-devices "${FSDP_DEVICES}")
fi
if [ -n "${NORM_MAX_FRAMES}" ]; then
  EXTRA_ARGS+=(--norm-max-frames "${NORM_MAX_FRAMES}")
fi
if [ "${OPENLOOP_FRAMEWORK}" != "pytorch" ] && [ -n "${JAX_PARAMS_PATH}" ]; then
  EXTRA_ARGS+=(--jax-params-path "${JAX_PARAMS_PATH}")
fi
if [ -n "${PYTORCH_WEIGHT_PATH}" ]; then
  EXTRA_ARGS+=(--pytorch-weight-path "${PYTORCH_WEIGHT_PATH}")
fi
if [ -n "${PYTORCH_TRAINING_PRECISION}" ]; then
  EXTRA_ARGS+=(--pytorch-training-precision "${PYTORCH_TRAINING_PRECISION}")
fi
if [ -n "${PYTORCH_GRADIENT_CHECKPOINTING}" ]; then
  EXTRA_ARGS+=(--gradient-checkpointing "${PYTORCH_GRADIENT_CHECKPOINTING}")
fi
if [ -n "${PYTORCH_FIND_UNUSED_PARAMETERS}" ]; then
  EXTRA_ARGS+=(--find-unused-parameters "${PYTORCH_FIND_UNUSED_PARAMETERS}")
fi

mkdir -p "${OPENPI_DATA_HOME}" "${TMPDIR}"
export CUDA_VISIBLE_DEVICES
export OPENPI_DATA_HOME
export TMPDIR

if [ "${OPENLOOP_FRAMEWORK}" = "pytorch" ]; then
  if [ "${PYTORCH_ENABLE_COMPILE}" = "true" ]; then
    export OPENPI_DISABLE_TORCH_COMPILE=0
  else
    export OPENPI_DISABLE_TORCH_COMPILE=${OPENPI_DISABLE_TORCH_COMPILE:-1}
  fi
  bash "${SCRIPT_DIR}/fix_transformers_replace.sh"
fi

MODULE_ARGS=(
  --framework "${OPENLOOP_FRAMEWORK}"
  --train-config-name "${TRAIN_CONFIG_NAME}"
  --repo-id "${REPO_ID}"
  --camera-mode "${CAMERA_MODE}"
  --exp-name "${EXP_NAME}"
  --assets-base-dir "${ASSETS_BASE_DIR}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --seed "${SEED}"
  --overwrite "${OVERWRITE}"
  --resume "${RESUME}"
  --wandb-enabled "${WANDB_ENABLED}"
  --project-name "${PROJECT_NAME}"
  --skip-norm-stats "${SKIP_NORM_STATS}"
  "${EXTRA_ARGS[@]}"
)

if [ "${OPENLOOP_FRAMEWORK}" = "pytorch" ]; then
  IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_PROCS=${#GPU_LIST[@]}
  if [ "${NUM_PROCS}" -le 1 ]; then
    uv run --no-sync --project "${UV_PROJECT}" python -m policy.PI05_LatentCorr.train_openloop "${MODULE_ARGS[@]}"
  else
    uv run --no-sync --project "${UV_PROJECT}" torchrun --standalone --nproc_per_node "${NUM_PROCS}" -m policy.PI05_LatentCorr.train_openloop "${MODULE_ARGS[@]}"
  fi
else
  uv run --no-sync --project "${UV_PROJECT}" python -m policy.PI05_LatentCorr.train_openloop "${MODULE_ARGS[@]}"
fi
