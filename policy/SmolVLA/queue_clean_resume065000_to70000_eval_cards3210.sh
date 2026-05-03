#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

RUN_DIR="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random"
RESUME_CKPT="${RUN_DIR}/checkpoints/065000"
TARGET_CKPT="${RUN_DIR}/checkpoints/070000/pretrained_model"
EVAL_TAG="base_70000"
CKPT_SETTING="base_70000"
LOG_DIR="${SCRIPT_DIR}/outputs/eval_logs/${EVAL_TAG}"

mkdir -p "${LOG_DIR}"

if [ ! -d "${TARGET_CKPT}" ]; then
  echo "Target checkpoint missing, resume clean training from ${RESUME_CKPT} to 70000."
  CUDA_VISIBLE_DEVICES=3 \
  PRETRAINED_PATH="${RESUME_CKPT}/pretrained_model" \
  RESUME_FROM="${RESUME_CKPT}" \
  STEPS=70000 \
  SAVE_FREQ=2500 \
  RUN_TAG=rgb_seen_random \
  TRAIN_TAG=rgb_seen_random \
  bash "${SCRIPT_DIR}/train.sh"
else
  echo "Target checkpoint already exists: ${TARGET_CKPT}"
fi

if [ ! -d "${TARGET_CKPT}" ]; then
  echo "Target checkpoint was not created: ${TARGET_CKPT}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

GPU_ID=3 TASK_NAME=handover_block CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${TARGET_CKPT}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/handover_block_g3.log" 2>&1 &
GPU_ID=2 TASK_NAME=open_laptop CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${TARGET_CKPT}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/open_laptop_g2.log" 2>&1 &
GPU_ID=1 TASK_NAME=pick_dual_bottles CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${TARGET_CKPT}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/pick_dual_bottles_g1.log" 2>&1 &
GPU_ID=0 TASK_NAME=place_burger_fries CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${TARGET_CKPT}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/place_burger_fries_g0.log" 2>&1 &

wait
