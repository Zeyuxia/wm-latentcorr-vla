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
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/stage1_unified_multitask/$(date +%Y%m%d_%H%M%S)}
EVAC_CKPT=${EVAC_CKPT:-/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt}
EVAC_CONFIG=${EVAC_CONFIG:-/data/zhenyangfan/EVAC_cache/configs/robotwin/train_config.yaml}
PYTORCH_WEIGHT_PATH=${PYTORCH_WEIGHT_PATH:-}
TASK_CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR:-}
TASK_CHECKPOINT_ID=${TASK_CHECKPOINT_ID:-latest}
DEVICE=${DEVICE:-cuda:0}
NORMAL_BATCH_SIZE=${NORMAL_BATCH_SIZE:-4}
FAILURE_BATCH_SIZE=${FAILURE_BATCH_SIZE:-2}
REFERENCE_GLOBAL_BATCH_SIZE=${REFERENCE_GLOBAL_BATCH_SIZE:-6}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_EPOCHS=${NUM_EPOCHS:-1000}
MAX_STEPS=${MAX_STEPS:--1}
LR=${LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SAVE_EVERY=${SAVE_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-20}
PREFIX_STEPS=${PREFIX_STEPS:-16}
FUTURE_OFFSET=${FUTURE_OFFSET:-16}
ACTION_HORIZON=${ACTION_HORIZON:-50}
LAMBDA_ACTION_CONDITIONED=${LAMBDA_ACTION_CONDITIONED:-0.5}
SCHEDULE_ACTION_CONDITIONED=${SCHEDULE_ACTION_CONDITIONED:-true}
BETA_DYNAMICS_MAX=${BETA_DYNAMICS_MAX:-1.0}
FREEZE_BASE_PI0=${FREEZE_BASE_PI0:-false}
PROMPT_MODE=${PROMPT_MODE:-random}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29631}
DRY_RUN=${DRY_RUN:-false}

DEFAULT_TASKS="pick_dual_bottles open_laptop place_burger_fries put_bottles_dustbin handover_block"
MULTI_TASK_NAMES=${MULTI_TASK_NAMES:-${DEFAULT_TASKS}}
PROCESSED_DIRS=${PROCESSED_DIRS:-}
REPO_IDS=${REPO_IDS:-}
RAW_DATA_DIRS=${RAW_DATA_DIRS:-}
FAILURE_TABLE_ROOT=${FAILURE_TABLE_ROOT:-}
FAILURE_TABLE_PATHS=${FAILURE_TABLE_PATHS:-}

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

if [ -z "${FAILURE_TABLE_PATHS}" ]; then
  if [ -z "${FAILURE_TABLE_ROOT}" ]; then
    echo "[PI05_LatentCorr] FAILURE_TABLE_ROOT or FAILURE_TABLE_PATHS is required." >&2
    echo "[PI05_LatentCorr] Set FAILURE_TABLE_ROOT to <pi05_explore_output>/failure_explore after PI05 explore finishes." >&2
    exit 1
  fi
  FAILURE_TABLE_ARR=()
  for task in "${TASK_NAME_ARR[@]}"; do
    FAILURE_TABLE_ARR+=("${FAILURE_TABLE_ROOT}/${task}/failure_table.json")
  done
else
  # shellcheck disable=SC2206
  FAILURE_TABLE_ARR=(${FAILURE_TABLE_PATHS})
fi

# shellcheck disable=SC2206
RAW_DATA_DIR_ARR=(${RAW_DATA_DIRS})
if [ "${#RAW_DATA_DIR_ARR[@]}" -eq 0 ]; then
  for task in "${TASK_NAME_ARR[@]}"; do
    RAW_DATA_DIR_ARR+=("${ROBOTWIN_ROOT}/data/${task}/demo_clean/data")
  done
fi

if [ "${#TASK_NAME_ARR[@]}" -ne "${#PROCESSED_DIR_ARR[@]}" ] \
  || [ "${#TASK_NAME_ARR[@]}" -ne "${#REPO_ID_ARR[@]}" ] \
  || [ "${#TASK_NAME_ARR[@]}" -ne "${#FAILURE_TABLE_ARR[@]}" ] \
  || [ "${#TASK_NAME_ARR[@]}" -ne "${#RAW_DATA_DIR_ARR[@]}" ]; then
  echo "[PI05_LatentCorr] multitask arrays must have same length" >&2
  echo "  tasks=${#TASK_NAME_ARR[@]} processed=${#PROCESSED_DIR_ARR[@]} repo=${#REPO_ID_ARR[@]} failure=${#FAILURE_TABLE_ARR[@]} raw=${#RAW_DATA_DIR_ARR[@]}" >&2
  exit 1
fi

for path in "${PROCESSED_DIR_ARR[@]}" "${FAILURE_TABLE_ARR[@]}" "${RAW_DATA_DIR_ARR[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "[PI05_LatentCorr] missing required path: ${path}" >&2
    echo "[PI05_LatentCorr] run prepare_multitask_headonly_assets.sh first if processed data or norm stats are missing." >&2
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

echo "[PI05_LatentCorr] unified multitask config"
echo "  tasks: ${TASK_NAME_ARR[*]}"
echo "  processed_dirs: ${PROCESSED_DIR_ARR[*]}"
echo "  repo_ids: ${REPO_ID_ARR[*]}"
echo "  raw_data_dirs: ${RAW_DATA_DIR_ARR[*]}"
echo "  failure_tables: ${FAILURE_TABLE_ARR[*]}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  gpus: ${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE}"
if [ "${DRY_RUN}" = "true" ]; then
  echo "[PI05_LatentCorr] DRY_RUN=true, config check passed; not launching training."
  exit 0
fi

EXTRA_ARGS=()
if [ -n "${PYTORCH_WEIGHT_PATH}" ]; then
  EXTRA_ARGS+=(--pytorch-weight-path "${PYTORCH_WEIGHT_PATH}")
fi
if [ -n "${TASK_CHECKPOINT_DIR}" ]; then
  EXTRA_ARGS+=(--task-checkpoint-dir "${TASK_CHECKPOINT_DIR}" --task-checkpoint-id "${TASK_CHECKPOINT_ID}")
fi
EXTRA_ARGS+=(--raw-data-dirs "${RAW_DATA_DIR_ARR[@]}")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m policy.PI05_LatentCorr.train_stage1_unified_failure_multitask_latent \
  --output-dir "${OUTPUT_DIR}" \
  --evac-ckpt "${EVAC_CKPT}" \
  --evac-config "${EVAC_CONFIG}" \
  --train-config-name "${TRAIN_CONFIG_NAME}" \
  --camera-mode "${CAMERA_MODE}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --multi-task-names "${TASK_NAME_ARR[@]}" \
  --processed-dirs "${PROCESSED_DIR_ARR[@]}" \
  --repo-ids "${REPO_ID_ARR[@]}" \
  --failure-table-paths "${FAILURE_TABLE_ARR[@]}" \
  --device "${DEVICE}" \
  --normal-batch-size "${NORMAL_BATCH_SIZE}" \
  --failure-batch-size "${FAILURE_BATCH_SIZE}" \
  --reference-global-batch-size "${REFERENCE_GLOBAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-epochs "${NUM_EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --save-every "${SAVE_EVERY}" \
  --log-every "${LOG_EVERY}" \
  --prefix-steps "${PREFIX_STEPS}" \
  --future-offset "${FUTURE_OFFSET}" \
  --action-horizon "${ACTION_HORIZON}" \
  --lambda-action-conditioned "${LAMBDA_ACTION_CONDITIONED}" \
  --schedule-action-conditioned "${SCHEDULE_ACTION_CONDITIONED}" \
  --beta-dynamics-max "${BETA_DYNAMICS_MAX}" \
  --freeze-base-pi0 "${FREEZE_BASE_PI0}" \
  --prompt-mode "${PROMPT_MODE}" \
  --urdf-path "${URDF_PATH}" \
  --curobo-left-yml "${CUROBO_LEFT_YML}" \
  --curobo-right-yml "${CUROBO_RIGHT_YML}" \
  "${EXTRA_ARGS[@]}"
