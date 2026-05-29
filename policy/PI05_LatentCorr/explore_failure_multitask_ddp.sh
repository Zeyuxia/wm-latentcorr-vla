#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PI05_BASE_DIR="${ROBOTWIN_ROOT}/policy/pi05"
cd "${ROBOTWIN_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_robotwin_singleview_base}
CAMERA_MODE=${CAMERA_MODE:-head_only}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR:-${PI05_BASE_DIR}/outputs/assets}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/pi05_failure_explore/$(date +%Y%m%d_%H%M%S)}
EVAC_CKPT=${EVAC_CKPT:-/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt}
EVAC_CONFIG=${EVAC_CONFIG:-/data/zhenyangfan/EVAC_cache/configs/robotwin/train_config.yaml}
PYTORCH_WEIGHT_PATH=${PYTORCH_WEIGHT_PATH:-}
TASK_CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR:-}
TASK_CHECKPOINT_ID=${TASK_CHECKPOINT_ID:-latest}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29641}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
NUM_EPOCHS=${NUM_EPOCHS:-1000}
PREFIX_STEPS=${PREFIX_STEPS:-16}
FUTURE_OFFSET=${FUTURE_OFFSET:-16}
ACTION_HORIZON=${ACTION_HORIZON:-50}
FAILURE_EXPLORE_K=${FAILURE_EXPLORE_K:-4}
FAILURE_FAIL_RECOVER_RATE_THRESH=${FAILURE_FAIL_RECOVER_RATE_THRESH:-0.5}
EXPLORE_DEBUG_MAX_SAMPLES=${EXPLORE_DEBUG_MAX_SAMPLES:-0}
RECOVER_EVAL_SAVE_VIDEO=${RECOVER_EVAL_SAVE_VIDEO:-false}
DRY_RUN=${DRY_RUN:-false}

DEFAULT_TASKS="pick_dual_bottles open_laptop place_burger_fries put_bottles_dustbin handover_block"
MULTI_TASK_NAMES=${MULTI_TASK_NAMES:-${DEFAULT_TASKS}}
PROCESSED_DIRS=${PROCESSED_DIRS:-}
REPO_IDS=${REPO_IDS:-}
RAW_DATA_DIRS=${RAW_DATA_DIRS:-}

URDF_PATH=${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/aloha_agilex_finegripper.xacro}
CUROBO_LEFT_YML=${CUROBO_LEFT_YML:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml}
CUROBO_RIGHT_YML=${CUROBO_RIGHT_YML:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[PI05_LatentCorr] python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

# shellcheck disable=SC2206
TASK_NAME_ARR=(${MULTI_TASK_NAMES})

if [ -z "${PROCESSED_DIRS}" ]; then
  PROCESSED_DIR_ARR=()
  for task in "${TASK_NAME_ARR[@]}"; do
    PROCESSED_DIR_ARR+=("${PI05_BASE_DIR}/outputs/processed_data/${task}-demo_clean-50-${CAMERA_MODE}")
  done
else
  # shellcheck disable=SC2206
  PROCESSED_DIR_ARR=(${PROCESSED_DIRS})
fi

if [ -z "${REPO_IDS}" ]; then
  REPO_ID_ARR=()
  for task in "${TASK_NAME_ARR[@]}"; do
    REPO_ID_ARR+=("robotwin/${task}_demo_clean_50_headonly")
  done
else
  # shellcheck disable=SC2206
  REPO_ID_ARR=(${REPO_IDS})
fi

if [ -z "${RAW_DATA_DIRS}" ]; then
  RAW_DATA_DIR_ARR=()
  for task in "${TASK_NAME_ARR[@]}"; do
    RAW_DATA_DIR_ARR+=("${ROBOTWIN_ROOT}/data/${task}/demo_clean/data")
  done
else
  # shellcheck disable=SC2206
  RAW_DATA_DIR_ARR=(${RAW_DATA_DIRS})
fi

if [ "${#TASK_NAME_ARR[@]}" -ne "${#PROCESSED_DIR_ARR[@]}" ] \
  || [ "${#TASK_NAME_ARR[@]}" -ne "${#REPO_ID_ARR[@]}" ] \
  || [ "${#TASK_NAME_ARR[@]}" -ne "${#RAW_DATA_DIR_ARR[@]}" ]; then
  echo "[PI05_LatentCorr] multitask arrays must have same length" >&2
  exit 1
fi

for path in "${PROCESSED_DIR_ARR[@]}" "${RAW_DATA_DIR_ARR[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "[PI05_LatentCorr] missing required path: ${path}" >&2
    echo "[PI05_LatentCorr] run prepare_multitask_headonly_assets.sh first if processed data is missing." >&2
    exit 1
  fi
done

for repo_id in "${REPO_ID_ARR[@]}"; do
  norm_path="${ASSETS_BASE_DIR}/${TRAIN_CONFIG_NAME}/${repo_id}/norm_stats.json"
  if [ ! -f "${norm_path}" ]; then
    echo "[PI05_LatentCorr] missing norm stats: ${norm_path}" >&2
    echo "[PI05_LatentCorr] run prepare_multitask_headonly_assets.sh first." >&2
    exit 1
  fi
done

echo "[PI05_LatentCorr] explore config"
echo "  tasks: ${TASK_NAME_ARR[*]}"
echo "  processed_dirs: ${PROCESSED_DIR_ARR[*]}"
echo "  repo_ids: ${REPO_ID_ARR[*]}"
echo "  raw_data_dirs: ${RAW_DATA_DIR_ARR[*]}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  gpus: ${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE}"
if [ "${DRY_RUN}" = "true" ]; then
  echo "[PI05_LatentCorr] DRY_RUN=true, config check passed; not launching explore."
  exit 0
fi

EXTRA_ARGS=()
if [ -n "${PYTORCH_WEIGHT_PATH}" ]; then
  EXTRA_ARGS+=(--pytorch-weight-path "${PYTORCH_WEIGHT_PATH}")
fi
if [ -n "${TASK_CHECKPOINT_DIR}" ]; then
  EXTRA_ARGS+=(--task-checkpoint-dir "${TASK_CHECKPOINT_DIR}" --task-checkpoint-id "${TASK_CHECKPOINT_ID}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m policy.PI05_LatentCorr.explore_failure_multitask \
  --output-dir "${OUTPUT_DIR}" \
  --evac-ckpt "${EVAC_CKPT}" \
  --evac-config "${EVAC_CONFIG}" \
  --train-config-name "${TRAIN_CONFIG_NAME}" \
  --camera-mode "${CAMERA_MODE}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --multi-task-names "${TASK_NAME_ARR[@]}" \
  --processed-dirs "${PROCESSED_DIR_ARR[@]}" \
  --repo-ids "${REPO_ID_ARR[@]}" \
  --raw-data-dirs "${RAW_DATA_DIR_ARR[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-epochs "${NUM_EPOCHS}" \
  --prefix-steps "${PREFIX_STEPS}" \
  --future-offset "${FUTURE_OFFSET}" \
  --action-horizon "${ACTION_HORIZON}" \
  --failure-explore-k "${FAILURE_EXPLORE_K}" \
  --failure-fail-recover-rate-thresh "${FAILURE_FAIL_RECOVER_RATE_THRESH}" \
  --explore-debug-max-samples "${EXPLORE_DEBUG_MAX_SAMPLES}" \
  --recover-eval-save-video "${RECOVER_EVAL_SAVE_VIDEO}" \
  --urdf-path "${URDF_PATH}" \
  --curobo-left-yml "${CUROBO_LEFT_YML}" \
  --curobo-right-yml "${CUROBO_RIGHT_YML}" \
  "${EXTRA_ARGS[@]}"
