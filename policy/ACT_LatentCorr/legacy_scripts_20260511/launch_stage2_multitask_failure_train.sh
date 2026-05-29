#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ROOT=${RUN_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure}
RUN_TAG=${RUN_TAG:-stage2_multitask_failure_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/${RUN_TAG}}
LOG_DIR=${LOG_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

STAGE1_CKPT=${STAGE1_CKPT:-}
BASE_ANCHOR_CKPT=${BASE_ANCHOR_CKPT:-${STAGE1_CKPT}}
if [ -z "${STAGE1_CKPT}" ]; then
  echo "STAGE1_CKPT is required" >&2
  exit 1
fi
if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "missing STAGE1_CKPT=${STAGE1_CKPT}" >&2
  exit 1
fi

MULTI_TASK_NAMES=${MULTI_TASK_NAMES:-"sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50"}
FAILURE_TABLE_PATHS=${FAILURE_TABLE_PATHS:-}
if [ -z "${FAILURE_TABLE_PATHS}" ]; then
  echo "FAILURE_TABLE_PATHS is required. Provide one table for all tasks or one path per task in MULTI_TASK_NAMES order." >&2
  exit 1
fi

GPU_LIST=${GPU_LIST:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29721}
NUM_EPOCHS=${NUM_EPOCHS:-250}
BATCH_SIZE=${BATCH_SIZE:-4}
CORRECTION_BATCH_SIZE=${CORRECTION_BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
SAVE_FREQ=${SAVE_FREQ:-50}

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TASK_NAME="robotwin_multitask_stage2" \
MULTI_TASK_NAMES="${MULTI_TASK_NAMES}" \
STAGE1_CKPT="${STAGE1_CKPT}" \
BASE_ANCHOR_CKPT="${BASE_ANCHOR_CKPT}" \
FAILURE_MODE=train \
FAILURE_TABLE_PATHS="${FAILURE_TABLE_PATHS}" \
CORRECTION_BUILDER_MODE=act_aligned \
USE_ACT_HEAD_CORRECTION=true \
DETACH_ACT_FEATURE_FOR_LATENT=true \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
BATCH_SIZE="${BATCH_SIZE}" \
CORRECTION_BATCH_SIZE="${CORRECTION_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
SAVE_FREQ="${SAVE_FREQ}" \
ACT_CHUNK_SIZE=50 \
PREFIX_STEPS=16 \
FUTURE_OFFSET=16 \
LR=${LR:-3e-5} \
RETAIN_WEIGHT=${RETAIN_WEIGHT:-1.0} \
RETAIN_WEIGHT_FINAL=${RETAIN_WEIGHT_FINAL:-0.1} \
RETAIN_DECAY_START_EPOCH=${RETAIN_DECAY_START_EPOCH:-0} \
RETAIN_DECAY_END_EPOCH=${RETAIN_DECAY_END_EPOCH:-100} \
RETAIN_DECAY_CURVE=${RETAIN_DECAY_CURVE:-cosine} \
BETA_DYNAMICS_MAX=${BETA_DYNAMICS_MAX:-1.0} \
DYN_ZERO_STEPS=${DYN_ZERO_STEPS:-0} \
DYN_RAMP_STEPS=${DYN_RAMP_STEPS:-100} \
DYN_WARMUP_CURVE=${DYN_WARMUP_CURVE:-cosine} \
DYN_SCHEDULE_UNIT=${DYN_SCHEDULE_UNIT:-epoch} \
REFERENCE_GLOBAL_BATCH_SIZE=${REFERENCE_GLOBAL_BATCH_SIZE:-24} \
LAMBDA_WM_ACTION_CURRENT=${LAMBDA_WM_ACTION_CURRENT:-0.0} \
LAMBDA_WM_ACTION_FUTURE=${LAMBDA_WM_ACTION_FUTURE:-0.0} \
LAMBDA_BRIDGE_FUTURE=${LAMBDA_BRIDGE_FUTURE:-0.0} \
BRIDGE_WEIGHT=${BRIDGE_WEIGHT:-0.0} \
EVAC_CKPT=${EVAC_CKPT:-/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt} \
EVAC_CONFIG=${EVAC_CONFIG:-/data/yujieyang/EVAC/configs/robotwin/train_config_mixed50p12.yaml} \
EVAC_REPO_ROOT=${EVAC_REPO_ROOT:-/data/zhenyangfan/EVAC} \
USE_WANDB=${USE_WANDB:-false} \
WANDB_LOG_MODE=${WANDB_LOG_MODE:-disabled} \
WANDB_RUN_NAME="${RUN_TAG}" \
WANDB_GROUP=stage2_multitask_failure \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_ddp.sh \
  2>&1 | tee -a "${LOG_FILE}"

echo "output_dir=${OUTPUT_DIR}"
echo "log_file=${LOG_FILE}"
