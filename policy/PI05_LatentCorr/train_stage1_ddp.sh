#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
PYTHON_BIN=${PYTHON_BIN:-${UV_PROJECT}/.venv/bin/python}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-50}
CAMERA_MODE=${CAMERA_MODE:-head_only}
REPO_ID=${REPO_ID:-robotwin/${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}_headonly}
PROCESSED_DIR=${PROCESSED_DIR:-${SCRIPT_DIR}/outputs/processed_data/${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-${CAMERA_MODE}}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/stage1_ddp/${TRAIN_CONFIG_NAME}/${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}_${CAMERA_MODE}}
EVAC_CKPT=${EVAC_CKPT:-/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt}
EVAC_CONFIG=${EVAC_CONFIG:-/data/zhenyangfan/EVAC_cache/configs/robotwin/train_config.yaml}
PYTORCH_WEIGHT_PATH=${PYTORCH_WEIGHT_PATH:-}
TASK_CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR:-}
TASK_CHECKPOINT_ID=${TASK_CHECKPOINT_ID:-latest}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-1}
REFERENCE_GLOBAL_BATCH_SIZE=${REFERENCE_GLOBAL_BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
NUM_EPOCHS=${NUM_EPOCHS:-10}
MAX_STEPS=${MAX_STEPS:--1}
LR=${LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SAVE_EVERY=${SAVE_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-20}
PREFIX_STEPS=${PREFIX_STEPS:-16}
FUTURE_OFFSET=${FUTURE_OFFSET:-16}
LAMBDA_ACTION_CONDITIONED=${LAMBDA_ACTION_CONDITIONED:-1.0}
LAMBDA_ALIGN=${LAMBDA_ALIGN:-0.0}
BETA_DYNAMICS_MAX=${BETA_DYNAMICS_MAX:-1.0}
DYN_ZERO_STEPS=${DYN_ZERO_STEPS:-0}
DYN_RAMP_STEPS=${DYN_RAMP_STEPS:-2000}
DYN_WARMUP_CURVE=${DYN_WARMUP_CURVE:-cosine}
FREEZE_BASE_PI0=${FREEZE_BASE_PI0:-false}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-}
RESUME=${RESUME:-false}
RESUME_PATH=${RESUME_PATH:-}
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-RoboTwin_PI05_LatentCorr}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_MODE=${WANDB_MODE:-auto}
WANDB_TAGS=${WANDB_TAGS:-}
WANDB_SYNC_POLL_INTERVAL=${WANDB_SYNC_POLL_INTERVAL:-15}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29611}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [ -z "${EVAC_CKPT}" ] || [ -z "${EVAC_CONFIG}" ]; then
  echo "[PI05_LatentCorr] EVAC_CKPT and EVAC_CONFIG are required for Stage 1 DDP"
  exit 1
fi

bash "${SCRIPT_DIR}/fix_transformers_replace.sh"
bash "${SCRIPT_DIR}/fix_evac_deps.sh"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[PI05_LatentCorr] python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [ -n "${PYTORCH_WEIGHT_PATH}" ]; then
  EXTRA_ARGS+=(--pytorch-weight-path "${PYTORCH_WEIGHT_PATH}")
fi
if [ -n "${TASK_CHECKPOINT_DIR}" ]; then
  EXTRA_ARGS+=(--task-checkpoint-dir "${TASK_CHECKPOINT_DIR}")
  EXTRA_ARGS+=(--task-checkpoint-id "${TASK_CHECKPOINT_ID}")
fi
if [ -n "${SAMPLES_PER_EPOCH}" ]; then
  EXTRA_ARGS+=(--samples-per-epoch "${SAMPLES_PER_EPOCH}")
fi
if [ "${RESUME}" = "true" ]; then
  EXTRA_ARGS+=(--resume true)
fi
if [ -n "${RESUME_PATH}" ]; then
  EXTRA_ARGS+=(--resume-path "${RESUME_PATH}")
fi
EXTRA_ARGS+=(--use-wandb "${USE_WANDB}")
EXTRA_ARGS+=(--wandb-project "${WANDB_PROJECT}")
EXTRA_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
EXTRA_ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
EXTRA_ARGS+=(--wandb-group "${WANDB_GROUP}")
EXTRA_ARGS+=(--wandb-mode "${WANDB_MODE}")
if [ -n "${WANDB_TAGS}" ]; then
  # shellcheck disable=SC2206
  tag_array=(${WANDB_TAGS})
  EXTRA_ARGS+=(--wandb-tags "${tag_array[@]}")
fi

WANDB_SYNC_PID=""
cleanup() {
  local exit_code=$?
  if [ -n "${WANDB_SYNC_PID}" ] && kill -0 "${WANDB_SYNC_PID}" 2>/dev/null; then
    if [ "${exit_code}" -ne 0 ]; then
      kill "${WANDB_SYNC_PID}" 2>/dev/null || true
    fi
    wait "${WANDB_SYNC_PID}" || true
  fi
  trap - EXIT
  exit "${exit_code}"
}
trap cleanup EXIT

if [ "${USE_WANDB}" = "true" ] && [ "${WANDB_MODE}" != "disabled" ]; then
  mkdir -p "${OUTPUT_DIR}"
  rm -f "${OUTPUT_DIR}/stage1_wandb_manifest.json" \
        "${OUTPUT_DIR}/stage1_wandb_status.json" \
        "${OUTPUT_DIR}/stage1_wandb_summary.json" \
        "${OUTPUT_DIR}/stage1_wandb_state.json"
  "${PYTHON_BIN}" -m policy.PI05_LatentCorr.sync_stage1_wandb \
    --output-dir "${OUTPUT_DIR}" \
    --poll-interval "${WANDB_SYNC_POLL_INTERVAL}" \
    --start-timeout 900 \
    > "${OUTPUT_DIR}/wandb_sync.log" 2>&1 &
  WANDB_SYNC_PID=$!
  echo "[PI05_LatentCorr] started wandb sync sidecar: pid=${WANDB_SYNC_PID}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m policy.PI05_LatentCorr.train_stage1 \
  --processed-dir "${PROCESSED_DIR}" \
  --repo-id "${REPO_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --evac-ckpt "${EVAC_CKPT}" \
  --evac-config "${EVAC_CONFIG}" \
  --train-config-name "${TRAIN_CONFIG_NAME}" \
  --camera-mode "${CAMERA_MODE}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
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
  --lambda-action-conditioned "${LAMBDA_ACTION_CONDITIONED}" \
  --lambda-align "${LAMBDA_ALIGN}" \
  --beta-dynamics-max "${BETA_DYNAMICS_MAX}" \
  --dyn-zero-steps "${DYN_ZERO_STEPS}" \
  --dyn-ramp-steps "${DYN_RAMP_STEPS}" \
  --dyn-warmup-curve "${DYN_WARMUP_CURVE}" \
  --freeze-base-pi0 "${FREEZE_BASE_PI0}" \
  "${EXTRA_ARGS[@]}"
