#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/data/zhenyangfan/RoboTwin
SCRIPT_DIR="${ROOT_DIR}/policy/SmolVLA"
MODEL_PATH="${SCRIPT_DIR}/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/060000/pretrained_model"
EVAL_TAG=base_60000
CKPT_SETTING=base_60000
LOG_DIR="${SCRIPT_DIR}/outputs/eval_logs/base_60000_cards7654"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

GPU_ID=7 TASK_NAME=handover_block CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/handover_block_g7.log" 2>&1 &
GPU_ID=6 TASK_NAME=open_laptop CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/open_laptop_g6.log" 2>&1 &
GPU_ID=5 TASK_NAME=pick_dual_bottles CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/pick_dual_bottles_g5.log" 2>&1 &
GPU_ID=4 TASK_NAME=place_burger_fries CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/place_burger_fries_g4.log" 2>&1 &

wait
