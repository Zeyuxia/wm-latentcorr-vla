#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

STAGE1_CKPT=${STAGE1_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/unified_stage1_acthead_open1000_trainable_base/20260330_100537/stage1_epoch_0400.pt}
RUN_ROOT=${RUN_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_free0123}
RUN_TAG=${RUN_TAG:-stage2_from_unified_ep400_acthead_teachercorr_anchor_free0123_$(date +"%Y%m%d_%H%M%S")}
LOG_FILE=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}.log
WATCH_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}_watch.log
TRAIN_SESSION=${TRAIN_SESSION:-stage2_from_unified_ep400_acthead_teachercorr_anchor_free0123}
WATCH_SESSION=${WATCH_SESSION:-stage2_from_unified_ep400_acthead_teachercorr_anchor_free0123_watch}
SEED_FILE=${SEED_FILE:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_100.txt}

mkdir -p "${RUN_ROOT}"

if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi
if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${WATCH_SESSION}"
fi

tmux new-session -d -s "${TRAIN_SESSION}" \
  "CUDA_VISIBLE_DEVICES=0,1,2,3 \
   STAGE1_CKPT=${STAGE1_CKPT} \
   OUTPUT_ROOT=${RUN_ROOT} \
   NPROC_PER_NODE=4 \
   NUM_EPOCHS=100 \
   SAVE_FREQ=25 \
   BATCH_SIZE=1 \
   LR=3e-5 \
   RETAIN_WEIGHT=1.0 \
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
   WANDB_GROUP=stage2_from_unified_ep400_acthead_teachercorr_anchor_free0123 \
   /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_ddp.sh > ${LOG_FILE} 2>&1"

TARGET_DIR=""
for _ in $(seq 1 20); do
  TARGET_DIR=$(find "${RUN_ROOT}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)
  [ -n "${TARGET_DIR}" ] && break
  sleep 3
done

if [ -z "${TARGET_DIR}" ]; then
  echo "failed to resolve target output directory under ${RUN_ROOT}" >&2
  exit 1
fi

BASE_RUN_TAG=${RUN_TAG}_base100
TEACHER_RUN_TAG=${RUN_TAG}_teacher100
CKPT_PATH=${TARGET_DIR}/stage2_epoch_0100.pt

tmux new-session -d -s "${WATCH_SESSION}" \
  "while [ ! -f '${CKPT_PATH}' ]; do sleep 20; done; \
   LATENT_CKPT_PATH='${CKPT_PATH}' INFERENCE_MODE=base SEED_FILE='${SEED_FILE}' GPU_LIST='0 1' RUN_TAG='${BASE_RUN_TAG}' bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh; \
   LATENT_CKPT_PATH='${CKPT_PATH}' INFERENCE_MODE=teacher SEED_FILE='${SEED_FILE}' GPU_LIST='2 3' RUN_TAG='${TEACHER_RUN_TAG}' bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > ${WATCH_LOG} 2>&1"

echo "train_session=${TRAIN_SESSION}"
echo "watch_session=${WATCH_SESSION}"
echo "log_file=${LOG_FILE}"
echo "watch_log=${WATCH_LOG}"
echo "output_dir=${TARGET_DIR}"
