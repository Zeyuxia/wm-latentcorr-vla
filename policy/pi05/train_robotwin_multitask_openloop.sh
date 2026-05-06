#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
CAMERA_MODE=${CAMERA_MODE:-head_only}
SECONDARY_CAMERA=${SECONDARY_CAMERA:-right_wrist}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-}
REPO_ID=${REPO_ID:-}
EXP_NAME=${EXP_NAME:-}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR:-${SCRIPT_DIR}/outputs/assets}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${SCRIPT_DIR}/outputs/checkpoints}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${SCRIPT_DIR}/outputs/openpi_cache}
TMPDIR=${TMPDIR:-${SCRIPT_DIR}/outputs/tmp}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
OPENLOOP_FRAMEWORK=${OPENLOOP_FRAMEWORK:-jax}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-20000}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
KEEP_PERIOD=${KEEP_PERIOD:-5000}
FSDP_DEVICES=${FSDP_DEVICES:-4}
SEED=${SEED:-42}
OVERWRITE=${OVERWRITE:-false}
RESUME=${RESUME:-false}
WANDB_ENABLED=${WANDB_ENABLED:-true}
PROJECT_NAME=${PROJECT_NAME:-RoboTwin_PI05_Base}
SKIP_NORM_STATS=${SKIP_NORM_STATS:-false}
NORM_MAX_FRAMES=${NORM_MAX_FRAMES:-}
JAX_PARAMS_PATH=${JAX_PARAMS_PATH:-}
PYTORCH_WEIGHT_PATH=${PYTORCH_WEIGHT_PATH:-}
PYTORCH_TRAINING_PRECISION=${PYTORCH_TRAINING_PRECISION:-}
PYTORCH_GRADIENT_CHECKPOINTING=${PYTORCH_GRADIENT_CHECKPOINTING:-true}
PYTORCH_FIND_UNUSED_PARAMETERS=${PYTORCH_FIND_UNUSED_PARAMETERS:-true}
PYTORCH_ENABLE_COMPILE=${PYTORCH_ENABLE_COMPILE:-false}
DRY_RUN=${DRY_RUN:-false}

case "${CAMERA_MODE}" in
  head_only)
    CAMERA_SUFFIX="headonly"
    ;;
  dual_view)
    case "${SECONDARY_CAMERA}" in
      left_wrist|right_wrist) ;;
      *)
        echo "[PI05_RobotWin] SECONDARY_CAMERA must be left_wrist or right_wrist for dual_view." >&2
        exit 1
        ;;
    esac
    CAMERA_SUFFIX="dualview_${SECONDARY_CAMERA}"
    ;;
  tri_view)
    CAMERA_SUFFIX="triview"
    ;;
  *)
    echo "[PI05_RobotWin] unsupported CAMERA_MODE=${CAMERA_MODE}. Expected head_only, dual_view, or tri_view." >&2
    exit 1
    ;;
esac

if [ -z "${TRAIN_CONFIG_NAME}" ]; then
  if [ "${CAMERA_MODE}" = "head_only" ]; then
    TRAIN_CONFIG_NAME="pi05_aloha_robotwin_singleview_base"
  else
    TRAIN_CONFIG_NAME="pi05_aloha_full_base"
  fi
fi

if [ -z "${REPO_ID}" ]; then
  REPO_ID="robotwin/multitask5_demo_clean_50_${CAMERA_SUFFIX}"
fi

if [ -z "${EXP_NAME}" ]; then
  EXP_NAME="pi05_robotwin_multitask5_${CAMERA_SUFFIX}_$(date +"%Y%m%d_%H%M%S")"
fi

norm_path="${ASSETS_BASE_DIR}/${TRAIN_CONFIG_NAME}/${REPO_ID}/norm_stats.json"
if [ ! -f "${norm_path}" ]; then
  echo "[PI05_RobotWin] missing norm stats: ${norm_path}" >&2
  echo "[PI05_RobotWin] run the matching prepare script first so processed data and norm stats exist." >&2
  exit 1
fi

echo "[PI05_RobotWin] base multitask open-loop config"
echo "  train_config: ${TRAIN_CONFIG_NAME}"
echo "  repo_id: ${REPO_ID}"
echo "  exp_name: ${EXP_NAME}"
echo "  camera_mode: ${CAMERA_MODE}"
echo "  secondary_camera: ${SECONDARY_CAMERA}"
echo "  checkpoint_base_dir: ${CHECKPOINT_BASE_DIR}"
echo "  framework: ${OPENLOOP_FRAMEWORK}"
echo "  gpus: ${CUDA_VISIBLE_DEVICES}"
if [ "${DRY_RUN}" = "true" ]; then
  echo "[PI05_RobotWin] DRY_RUN=true, config check passed; not launching training."
  exit 0
fi

export CUDA_VISIBLE_DEVICES
export OPENPI_DATA_HOME
export TMPDIR
mkdir -p "${OPENPI_DATA_HOME}" "${TMPDIR}"

EXTRA_ARGS=()
if [ -n "${NORM_MAX_FRAMES}" ]; then
  EXTRA_ARGS+=(--norm-max-frames "${NORM_MAX_FRAMES}")
fi
if [ -n "${JAX_PARAMS_PATH}" ]; then
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

if [ "${OPENLOOP_FRAMEWORK}" = "pytorch" ] && [ "${PYTORCH_ENABLE_COMPILE}" != "true" ]; then
  export OPENPI_DISABLE_TORCH_COMPILE=${OPENPI_DISABLE_TORCH_COMPILE:-1}
fi

MODULE_ARGS=(
  --framework "${OPENLOOP_FRAMEWORK}"
  --train-config-name "${TRAIN_CONFIG_NAME}"
  --repo-id "${REPO_ID}"
  --camera-mode "${CAMERA_MODE}"
  --secondary-camera "${SECONDARY_CAMERA}"
  --exp-name "${EXP_NAME}"
  --assets-base-dir "${ASSETS_BASE_DIR}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --log-interval "${LOG_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --keep-period "${KEEP_PERIOD}"
  --fsdp-devices "${FSDP_DEVICES}"
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
    uv run --no-sync --project "${UV_PROJECT}" python -m policy.pi05.train_robotwin_openloop "${MODULE_ARGS[@]}"
  else
    uv run --no-sync --project "${UV_PROJECT}" torchrun --standalone --nproc_per_node "${NUM_PROCS}" -m policy.pi05.train_robotwin_openloop "${MODULE_ARGS[@]}"
  fi
else
  uv run --no-sync --project "${UV_PROJECT}" python -m policy.pi05.train_robotwin_openloop "${MODULE_ARGS[@]}"
fi
