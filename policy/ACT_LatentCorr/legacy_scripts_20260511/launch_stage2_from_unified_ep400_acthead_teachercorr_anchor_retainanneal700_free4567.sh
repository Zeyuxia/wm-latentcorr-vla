#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

STAGE1_CKPT=${STAGE1_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/unified_stage1_acthead_open1000_trainable_base/20260330_100537/stage1_epoch_0400.pt}
BASE_ANCHOR_CKPT=${BASE_ANCHOR_CKPT:-${STAGE1_CKPT}}
RUN_ROOT=${RUN_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_retainanneal700_free4567}
RUN_TAG=${RUN_TAG:-stage2_from_unified_ep400_acthead_teachercorr_anchor_retainanneal700_free4567_$(date +"%Y%m%d_%H%M%S")}
LOG_FILE=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}.log
TRAIN_SESSION=${TRAIN_SESSION:-stage2_from_unified_ep400_acthead_teachercorr_anchor_retainanneal700_free4567}

mkdir -p "${RUN_ROOT}"

if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi

tmux new-session -d -s "${TRAIN_SESSION}" \
  "CUDA_VISIBLE_DEVICES=4,5,6,7 \
   STAGE1_CKPT=${STAGE1_CKPT} \
   BASE_ANCHOR_CKPT=${BASE_ANCHOR_CKPT} \
   OUTPUT_ROOT=${RUN_ROOT} \
   NPROC_PER_NODE=4 \
   NUM_EPOCHS=700 \
   SAVE_FREQ=25 \
   BATCH_SIZE=1 \
   CORRECTION_BATCH_SIZE=0 \
   LR=3e-5 \
   RETAIN_WEIGHT=1.0 \
   RETAIN_WEIGHT_FINAL=0.1 \
   RETAIN_DECAY_START_EPOCH=200 \
   RETAIN_DECAY_END_EPOCH=300 \
   RETAIN_DECAY_CURVE=linear \
   BRIDGE_WEIGHT=0.0 \
   FREEZE_BASE_ACT=false \
   DETACH_ACT_FEATURE_FOR_LATENT=true \
   USE_ACT_HEAD_CORRECTION=true \
   LAMBDA_WM_ACTION_CURRENT=0.0 \
   LAMBDA_WM_ACTION_FUTURE=0.0 \
   LAMBDA_BRIDGE_FUTURE=0.0 \
   DYN_ZERO_STEPS=0 \
   DYN_RAMP_STEPS=200 \
   REFERENCE_GLOBAL_BATCH_SIZE=4 \
   PLANNER_WARMUP=false \
   CORRECTION_INTERP_NEAREST_ENABLE=true \
   WANDB_RUN_NAME=${RUN_TAG} \
   WANDB_GROUP=stage2_from_unified_ep400_acthead_teachercorr_anchor_retainanneal700_free4567 \
   /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_ddp.sh > ${LOG_FILE} 2>&1"

TARGET_DIR=""
for _ in $(seq 1 20); do
  TARGET_DIR=$(find "${RUN_ROOT}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)
  [ -n "${TARGET_DIR}" ] && break
  sleep 3
done

echo "train_session=${TRAIN_SESSION}"
echo "log_file=${LOG_FILE}"
echo "output_dir=${TARGET_DIR}"
