#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high
TRAIN_RUN_TAG=stage1_clean_only_from055000_cards3210_steps1500
EVAL_TAG=stage1_clean_only_from055000_cards3210_step1500
CKPT_SETTING="${EVAL_TAG}"
LOG_DIR=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/eval_logs/${EVAL_TAG}

mkdir -p "${LOG_DIR}"

while true; do
  RUN_DIR=$(ls -td "${TRAIN_ROOT}"/*-"${TRAIN_RUN_TAG}" 2>/dev/null | head -n 1 || true)
  if [ -n "${RUN_DIR}" ]; then
    break
  fi
  date
  echo "waiting run dir for tag: ${TRAIN_RUN_TAG}"
  sleep 60
done

MODEL_PATH="${RUN_DIR}/stage1_step_001500.pt"

while [ ! -s "${MODEL_PATH}" ]; do
  date
  echo "waiting checkpoint: ${MODEL_PATH}"
  sleep 60
done

sleep 30

cd /data/zhenyangfan/RoboTwin

GPU_ID=3 TASK_NAME=handover_block CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash policy/SmolVLA/eval.sh > "${LOG_DIR}/handover_block_g3.log" 2>&1 &
GPU_ID=2 TASK_NAME=open_laptop CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash policy/SmolVLA/eval.sh > "${LOG_DIR}/open_laptop_g2.log" 2>&1 &
GPU_ID=1 TASK_NAME=pick_dual_bottles CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash policy/SmolVLA/eval.sh > "${LOG_DIR}/pick_dual_bottles_g1.log" 2>&1 &
GPU_ID=0 TASK_NAME=place_burger_fries CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash policy/SmolVLA/eval.sh > "${LOG_DIR}/place_burger_fries_g0.log" 2>&1 &

wait
