#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/data/zhenyangfan/RoboTwin
SCRIPT_DIR="${ROOT_DIR}/policy/SmolVLA"
OUTPUT_ROOT="${SCRIPT_DIR}/outputs/stage1/robotwin_multitask_5_cam_high"

TARGET_CUDA_DEVICES="${TARGET_CUDA_DEVICES:-3,2,1,0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29614}"
MAX_USED_MIB="${MAX_USED_MIB:-35000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-stage1_corr025_fullchunk_recovery_cards3210_steps1500}"
EVAL_TAG="${EVAL_TAG:-stage1_corr025_fullchunk_recovery_cards3210_step1500}"
CKPT_SETTING="${CKPT_SETTING:-${EVAL_TAG}}"
LOG_DIR="${SCRIPT_DIR}/outputs/eval_logs/${EVAL_TAG}"
FAILURE_TASK_NAMES_OVERRIDE="${FAILURE_TASK_NAMES_OVERRIDE:-sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

wait_for_gpus() {
  IFS=',' read -r -a gpus <<< "${TARGET_CUDA_DEVICES}"
  while true; do
    local ready=1
    local status=()
    for gpu in "${gpus[@]}"; do
      local used
      used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      status+=("gpu${gpu}=${used}MiB")
      if (( used > MAX_USED_MIB )); then
        ready=0
      fi
    done
    echo "[$(timestamp)] GPU memory: ${status[*]} / threshold=${MAX_USED_MIB}MiB"
    if (( ready == 1 )); then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}/outputs/queue_logs"
cd "${ROOT_DIR}"

echo "[$(timestamp)] Waiting for GPUs ${TARGET_CUDA_DEVICES}"
wait_for_gpus

echo "[$(timestamp)] Starting stage1 full-chunk recovery training"
CUDA_VISIBLE_DEVICES="${TARGET_CUDA_DEVICES}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
RUN_TAG="${TRAIN_RUN_TAG}" \
MAX_STEPS=1500 \
SAVE_FREQ=250 \
FAILURE_MODE=train \
FAILURE_CORR_BATCH_RATIO=0.25 \
FAILURE_TASK_NAMES_OVERRIDE="${FAILURE_TASK_NAMES_OVERRIDE}" \
ACT_ALIGNED_FULL_CHUNK_RECOVERY=true \
DYN_MAX_WEIGHT=0.0 \
COND_MAX_WEIGHT=0.0 \
SAVE_CORRECTION_DATA=true \
bash "${SCRIPT_DIR}/train_stage1.sh"

MODEL_PATH=$(
  find "${OUTPUT_ROOT}" -maxdepth 2 -path "*-${TRAIN_RUN_TAG}/stage1_step_001500.pt" -printf '%T@ %p\n' |
    sort -nr |
    head -n 1 |
    cut -d' ' -f2-
)

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "Missing expected stage1 checkpoint for tag ${TRAIN_RUN_TAG}" >&2
  exit 1
fi

echo "[$(timestamp)] Training finished. Model path: ${MODEL_PATH}"
echo "[$(timestamp)] Starting evals with tag ${EVAL_TAG}"

sleep 30

GPU_ID=3 TASK_NAME=handover_block CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/handover_block_g3.log" 2>&1 &
GPU_ID=2 TASK_NAME=open_laptop CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/open_laptop_g2.log" 2>&1 &
GPU_ID=1 TASK_NAME=pick_dual_bottles CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/pick_dual_bottles_g1.log" 2>&1 &
GPU_ID=0 TASK_NAME=place_burger_fries CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/place_burger_fries_g0.log" 2>&1 &

wait

echo "[$(timestamp)] All evals finished. Logs: ${LOG_DIR}"
