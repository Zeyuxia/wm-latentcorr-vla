#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

RESUME_CKPT=${RESUME_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_act_mainline_8gpu_resume250_to500_wmf010/20260328_214615/stage1_epoch_0500.pt}
RUN_ROOT=${RUN_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_act_mainline_8gpu_resume500_to1000_clean}
RUN_TAG=${RUN_TAG:-stage1_act_mainline_8gpu_resume500_to1000_clean_$(date +"%Y%m%d_%H%M%S")}
LOG_FILE=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}.log
WATCH_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}_watch.log

TRAIN_SESSION=${TRAIN_SESSION:-stage1_resume500_to1000_clean}
WATCH_SESSION=${WATCH_SESSION:-stage1_resume500_to1000_clean_watch}

tmux new-session -d -s "${TRAIN_SESSION}" \
  "NPROC_PER_NODE=8 \
   RESUME_CKPT=${RESUME_CKPT} \
   OUTPUT_ROOT=${RUN_ROOT} \
   NUM_EPOCHS=1000 \
   SAVE_FREQ=25 \
   LAMBDA_ALIGN=0.25 \
   LAMBDA_WM_ACTION_CURRENT=0.0 \
   LAMBDA_WM_ACTION_FUTURE=0.0 \
   LAMBDA_BRIDGE_FUTURE=0.0 \
   DYN_ZERO_STEPS=1000 \
   DYN_RAMP_STEPS=2000 \
   WANDB_RUN_NAME=${RUN_TAG} \
   WANDB_GROUP=stage1_mainline_resume500_to1000_clean \
   /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_ddp.sh > ${LOG_FILE} 2>&1"

TARGET_DIR=$(find "${RUN_ROOT}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)
if [ -z "${TARGET_DIR}" ]; then
  sleep 5
  TARGET_DIR=$(find "${RUN_ROOT}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)
fi

if [ -z "${TARGET_DIR}" ]; then
  echo "failed to resolve target output directory under ${RUN_ROOT}" >&2
  exit 1
fi

tmux new-session -d -s "${WATCH_SESSION}" \
  "CKPT_PATH=${TARGET_DIR}/stage1_epoch_1000.pt \
   RUN_TAG=${RUN_TAG}_eval50 \
   /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/watch_stage1_checkpoint_and_eval.sh > ${WATCH_LOG} 2>&1"

echo "train_session=${TRAIN_SESSION}"
echo "watch_session=${WATCH_SESSION}"
echo "log_file=${LOG_FILE}"
echo "watch_log=${WATCH_LOG}"
echo "output_dir=${TARGET_DIR}"
