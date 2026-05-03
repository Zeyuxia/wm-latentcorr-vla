#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

TRAIN_ROOT="${SCRIPT_DIR}/outputs/train/robotwin_multitask_5_cam_high_rgbfix"
TRAIN_RUN_TAG="rgbfix_from_smolvla_base_step70000"
TARGET_STEP="070000"
EVAL_TAG="rgbfix_from_smolvla_base_70000"
CKPT_SETTING="${EVAL_TAG}"
LOG_DIR="${SCRIPT_DIR}/outputs/eval_logs/${EVAL_TAG}_cards3210"

mkdir -p "${LOG_DIR}"

echo "Waiting for rgbfix 70000 checkpoint..."
echo "train_root=${TRAIN_ROOT}"
echo "train_run_tag=${TRAIN_RUN_TAG}"
echo "target_step=${TARGET_STEP}"
echo "eval_tag=${EVAL_TAG}"
echo "log_dir=${LOG_DIR}"

RUN_DIR=""
TARGET_CKPT=""

while true; do
  CANDIDATE_RUN_DIR=$(find "${TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d \
    -name "*-${TRAIN_RUN_TAG}" ! -name "*.pending" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -n 1 | cut -d' ' -f2- || true)

  if [ -n "${CANDIDATE_RUN_DIR}" ]; then
    RUN_DIR="${CANDIDATE_RUN_DIR}"
    TARGET_CKPT="${RUN_DIR}/checkpoints/${TARGET_STEP}/pretrained_model"
    if [ -d "${TARGET_CKPT}" ]; then
      break
    fi
    date
    echo "latest run dir: ${RUN_DIR}"
    echo "waiting checkpoint: ${TARGET_CKPT}"
  else
    date
    echo "waiting run dir for tag: ${TRAIN_RUN_TAG}"
  fi

  sleep 120
done

echo "Target checkpoint ready: ${TARGET_CKPT}"
sleep 30

cd "${ROOT_DIR}"

GPU_ID=3 TASK_NAME=handover_block CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${TARGET_CKPT}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/handover_block_g3.log" 2>&1 &
GPU_ID=2 TASK_NAME=open_laptop CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${TARGET_CKPT}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/open_laptop_g2.log" 2>&1 &
GPU_ID=1 TASK_NAME=pick_dual_bottles CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${TARGET_CKPT}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/pick_dual_bottles_g1.log" 2>&1 &
GPU_ID=0 TASK_NAME=place_burger_fries CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${TARGET_CKPT}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/place_burger_fries_g0.log" 2>&1 &

wait
