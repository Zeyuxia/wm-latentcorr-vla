#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORKFLOW_ROOT=${WORKFLOW_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/failure_workflow}
WORKFLOW_DIR=${WORKFLOW_DIR:-"${WORKFLOW_ROOT}/${TIMESTAMP}"}
EXPLORE_OUTPUT_DIR=${EXPLORE_OUTPUT_DIR:-"${WORKFLOW_DIR}/explore"}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-"${WORKFLOW_DIR}/train"}
FAILURE_TABLE_DIR=${FAILURE_TABLE_DIR:-"${EXPLORE_OUTPUT_DIR}/failure_explore"}

MODE=${MODE:-all}
EXPLORE_GPU=${EXPLORE_GPU:-1}
TRAIN_GPU=${TRAIN_GPU:-2}
USE_WANDB=${USE_WANDB:-false}
TRAIN_CORRECTION_BATCH_SIZE=${TRAIN_CORRECTION_BATCH_SIZE:-1}
CORRECTION_BUILDER_MODE=${CORRECTION_BUILDER_MODE:-act_aligned}
ACT_LIKE_LOSS_ONLY=${ACT_LIKE_LOSS_ONLY:-false}

if [[ "${MODE}" != "all" && "${MODE}" != "explore" && "${MODE}" != "train" ]]; then
  echo "Invalid MODE=${MODE}, expected all|explore|train" >&2
  exit 1
fi

mkdir -p "${EXPLORE_OUTPUT_DIR}" "${TRAIN_OUTPUT_DIR}"

TRAIN_SCRIPT=${TRAIN_SCRIPT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2.sh}
if [ ! -x "${TRAIN_SCRIPT}" ]; then
  echo "Launcher not executable: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

STAGE1_CKPT=${STAGE1_CKPT:-}
if [ -z "${STAGE1_CKPT}" ]; then
  STAGE1_CKPT=$(
    ls -1dt /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/*/*/stage1_epoch_*.pt 2>/dev/null \
    | head -n 1 \
    || true
  )
fi
if [ -z "${STAGE1_CKPT}" ]; then
  echo "STAGE1_CKPT is not set and no formal_runs stage1 checkpoint was found" >&2
  exit 1
fi

if [[ "${MODE}" == "all" || "${MODE}" == "explore" ]]; then
  echo "[workflow] phase=explore output=${EXPLORE_OUTPUT_DIR}"
  CUDA_VISIBLE_DEVICES="${EXPLORE_GPU}" \
  OUTPUT_DIR="${EXPLORE_OUTPUT_DIR}" \
  STAGE1_CKPT="${STAGE1_CKPT}" \
  FAILURE_MODE=explore \
  CORRECTION_BUILDER_MODE="${CORRECTION_BUILDER_MODE}" \
  FAILURE_TABLE_DIR="${FAILURE_TABLE_DIR}" \
  FAILURE_TABLE_PATH= \
  CORRECTION_BATCH_SIZE=0 \
  ACT_LIKE_LOSS_ONLY="${ACT_LIKE_LOSS_ONLY}" \
  USE_WANDB="${USE_WANDB}" \
  "${TRAIN_SCRIPT}"
fi

FAILURE_TABLE_PATH="${FAILURE_TABLE_DIR}/failure_table.json"
if [ ! -f "${FAILURE_TABLE_PATH}" ]; then
  echo "failure table not found: ${FAILURE_TABLE_PATH}" >&2
  exit 1
fi

if [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; then
  echo "[workflow] phase=train output=${TRAIN_OUTPUT_DIR}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
  OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
  STAGE1_CKPT="${STAGE1_CKPT}" \
  FAILURE_MODE=train \
  CORRECTION_BUILDER_MODE="${CORRECTION_BUILDER_MODE}" \
  FAILURE_TABLE_PATH="${FAILURE_TABLE_PATH}" \
  CORRECTION_BATCH_SIZE="${TRAIN_CORRECTION_BATCH_SIZE}" \
  ACT_LIKE_LOSS_ONLY="${ACT_LIKE_LOSS_ONLY}" \
  USE_WANDB="${USE_WANDB}" \
  "${TRAIN_SCRIPT}"
fi

echo "[workflow] done"
echo "WORKFLOW_DIR=${WORKFLOW_DIR}"
echo "FAILURE_TABLE_PATH=${FAILURE_TABLE_PATH}"
echo "ACT_LIKE_LOSS_ONLY=${ACT_LIKE_LOSS_ONLY}"
