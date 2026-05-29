#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}
TASK_NAME=${TASK_NAME:-open_laptop-demo_clean-50}
PROCESSED_DIR=${PROCESSED_DIR:-${SCRIPT_DIR}/outputs/processed_data/open_laptop-demo_clean-50-head_only}
REPO_ID=${REPO_ID:-robotwin/open_laptop_demo_clean_50_headonly}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/stage2_ddp/pi05_stage2_$(date +%Y%m%d_%H%M%S)}
STAGE1_CKPT=${STAGE1_CKPT:-}
BASE_ANCHOR_CKPT=${BASE_ANCHOR_CKPT:-}
EVAC_CKPT=${EVAC_CKPT:-/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt}
EVAC_CONFIG=${EVAC_CONFIG:-/data/zhenyangfan/EVAC_cache/configs/robotwin/train_config.yaml}
RAW_DATA_DIR=${RAW_DATA_DIR:-}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
CAMERA_MODE=${CAMERA_MODE:-head_only}
DEVICE=${DEVICE:-cuda:0}
NUM_EPISODES=${NUM_EPISODES:-50}
NUM_EPOCHS=${NUM_EPOCHS:-100}
MAX_STEPS=${MAX_STEPS:--1}
BATCH_SIZE=${BATCH_SIZE:-4}
CORRECTION_BATCH_SIZE=${CORRECTION_BATCH_SIZE:-2}
REFERENCE_GLOBAL_BATCH_SIZE=${REFERENCE_GLOBAL_BATCH_SIZE:-24}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SAVE_EVERY=${SAVE_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-20}
PREFIX_STEPS=${PREFIX_STEPS:-16}
FUTURE_OFFSET=${FUTURE_OFFSET:-16}
ACTION_HORIZON=${ACTION_HORIZON:-50}
DDIM_STEPS=${DDIM_STEPS:-27}
RETAIN_WEIGHT=${RETAIN_WEIGHT:-1.0}
RETAIN_WEIGHT_FINAL=${RETAIN_WEIGHT_FINAL:-0.1}
RETAIN_DECAY_START_EPOCH=${RETAIN_DECAY_START_EPOCH:-0}
RETAIN_DECAY_END_EPOCH=${RETAIN_DECAY_END_EPOCH:-100}
RETAIN_DECAY_CURVE=${RETAIN_DECAY_CURVE:-linear}
BRIDGE_WEIGHT=${BRIDGE_WEIGHT:-0.0}
USE_PI0_HEAD_CORRECTION=${USE_PI0_HEAD_CORRECTION:-true}
FAILURE_MODE=${FAILURE_MODE:-train}
FAILURE_TABLE_PATH=${FAILURE_TABLE_PATH:-}
URDF_PATH=${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/aloha_agilex.urdf}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29631}

if [ -z "${STAGE1_CKPT}" ]; then
  echo "[PI05_LatentCorr] STAGE1_CKPT is required" >&2
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[PI05_LatentCorr] python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [ -n "${BASE_ANCHOR_CKPT}" ]; then
  EXTRA_ARGS+=(--base-anchor-ckpt "${BASE_ANCHOR_CKPT}")
fi
if [ -n "${RAW_DATA_DIR}" ]; then
  EXTRA_ARGS+=(--raw-data-dir "${RAW_DATA_DIR}")
fi
if [ -n "${FAILURE_TABLE_PATH}" ]; then
  EXTRA_ARGS+=(--failure-table-path "${FAILURE_TABLE_PATH}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m policy.PI05_LatentCorr.train_stage2_latent \
  --task-name "${TASK_NAME}" \
  --processed-dir "${PROCESSED_DIR}" \
  --repo-id "${REPO_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --stage1-ckpt "${STAGE1_CKPT}" \
  --evac-ckpt "${EVAC_CKPT}" \
  --evac-config "${EVAC_CONFIG}" \
  --train-config-name "${TRAIN_CONFIG_NAME}" \
  --camera-mode "${CAMERA_MODE}" \
  --device "${DEVICE}" \
  --num-episodes "${NUM_EPISODES}" \
  --num-epochs "${NUM_EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --correction-batch-size "${CORRECTION_BATCH_SIZE}" \
  --reference-global-batch-size "${REFERENCE_GLOBAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --save-every "${SAVE_EVERY}" \
  --log-every "${LOG_EVERY}" \
  --prefix-steps "${PREFIX_STEPS}" \
  --future-offset "${FUTURE_OFFSET}" \
  --action-horizon "${ACTION_HORIZON}" \
  --ddim-steps "${DDIM_STEPS}" \
  --retain-weight "${RETAIN_WEIGHT}" \
  --retain-weight-final "${RETAIN_WEIGHT_FINAL}" \
  --retain-decay-start-epoch "${RETAIN_DECAY_START_EPOCH}" \
  --retain-decay-end-epoch "${RETAIN_DECAY_END_EPOCH}" \
  --retain-decay-curve "${RETAIN_DECAY_CURVE}" \
  --bridge-weight "${BRIDGE_WEIGHT}" \
  --use-pi0-head-correction "${USE_PI0_HEAD_CORRECTION}" \
  --failure-mode "${FAILURE_MODE}" \
  --urdf-path "${URDF_PATH}" \
  "${EXTRA_ARGS[@]}"
