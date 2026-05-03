#!/usr/bin/env bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TRAIN_ROOT=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high
TRAIN_RUN_TAG=stage1_clean_only_from055000_cards3210_steps1500
EVAL_TAG=stage1_clean_only_from055000_cards3210_step1500
CKPT_SETTING="${EVAL_TAG}"
LOG_DIR=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/eval_logs/${EVAL_TAG}

mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=3,2,1,0 \
MAIN_PROCESS_PORT=29612 \
RUN_TAG="${TRAIN_RUN_TAG}" \
MAX_STEPS=1500 \
FAILURE_MODE=off \
FAILURE_CORR_BATCH_RATIO=0.0 \
SAVE_CORRECTION_DATA=false \
DYN_MAX_WEIGHT=0.0 \
COND_MAX_WEIGHT=0.0 \
bash policy/SmolVLA/train_stage1.sh

RUN_DIR=$(ls -td "${TRAIN_ROOT}"/*-"${TRAIN_RUN_TAG}" 2>/dev/null | head -n 1 || true)
if [ -z "${RUN_DIR}" ]; then
  echo "Could not find training run dir for tag: ${TRAIN_RUN_TAG}" >&2
  exit 1
fi

MODEL_PATH="${RUN_DIR}/stage1_step_001500.pt"
if [ ! -s "${MODEL_PATH}" ]; then
  echo "Missing expected checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

sleep 30

GPU_ID=3 TASK_NAME=handover_block CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash policy/SmolVLA/eval.sh > "${LOG_DIR}/handover_block_g3.log" 2>&1 &
GPU_ID=2 TASK_NAME=open_laptop CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash policy/SmolVLA/eval.sh > "${LOG_DIR}/open_laptop_g2.log" 2>&1 &
GPU_ID=1 TASK_NAME=pick_dual_bottles CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash policy/SmolVLA/eval.sh > "${LOG_DIR}/pick_dual_bottles_g1.log" 2>&1 &
GPU_ID=0 TASK_NAME=place_burger_fries CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash policy/SmolVLA/eval.sh > "${LOG_DIR}/place_burger_fries_g0.log" 2>&1 &

wait
