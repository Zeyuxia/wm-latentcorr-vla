#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

STAGE1_CKPT=${STAGE1_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/unified_stage1_acthead_open1000_trainable_base/20260330_100537/stage1_epoch_0400.pt}
BASE_ANCHOR_CKPT=${BASE_ANCHOR_CKPT:-${STAGE1_CKPT}}
RESUME_CKPT=${RESUME_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume200_to300_free0123/20260401_194728/stage2_epoch_0300.pt}
RUN_ROOT=${RUN_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567}
RUN_TAG=${RUN_TAG:-stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567_$(date +"%Y%m%d_%H%M%S")}
LOG_FILE=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}.log
WATCH_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}_watch.log
TRAIN_SESSION=${TRAIN_SESSION:-stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567}
WATCH_SESSION=${WATCH_SESSION:-stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567_watch}
SEED_FILE_G1=${SEED_FILE_G1:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group1.txt}
SEED_FILE_G2=${SEED_FILE_G2:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group2.txt}
SEED_FILE_G3=${SEED_FILE_G3:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group3.txt}

mkdir -p "${RUN_ROOT}"

if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi
if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${WATCH_SESSION}"
fi

tmux new-session -d -s "${TRAIN_SESSION}" \
  "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   STAGE1_CKPT=${STAGE1_CKPT} \
   BASE_ANCHOR_CKPT=${BASE_ANCHOR_CKPT} \
   RESUME_CKPT=${RESUME_CKPT} \
   OUTPUT_ROOT=${RUN_ROOT} \
   NPROC_PER_NODE=8 \
   NUM_EPOCHS=500 \
   SAVE_FREQ=25 \
   BATCH_SIZE=1 \
   CORRECTION_BATCH_SIZE=0 \
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
   REFERENCE_GLOBAL_BATCH_SIZE=8 \
   PLANNER_WARMUP=false \
   CORRECTION_INTERP_NEAREST_ENABLE=true \
   WANDB_RUN_NAME=${RUN_TAG} \
   WANDB_GROUP=stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567 \
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

CKPT_PATH=${TARGET_DIR}/stage2_epoch_0500.pt

tmux new-session -d -s "${WATCH_SESSION}" \
  "while [ ! -f '${CKPT_PATH}' ]; do sleep 20; done; \
   LATENT_CKPT_PATH='${CKPT_PATH}' INFERENCE_MODE=teacher SEED_FILE='${SEED_FILE_G1}' GPU_LIST='2 3' RUN_TAG='stage2_ep500_teacher150_g1_${RUN_TAG}' bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/stage2_ep500_teacher150_g1_${RUN_TAG}_launch.log 2>&1 & \
   LATENT_CKPT_PATH='${CKPT_PATH}' INFERENCE_MODE=teacher SEED_FILE='${SEED_FILE_G2}' GPU_LIST='4 5' RUN_TAG='stage2_ep500_teacher150_g2_${RUN_TAG}' bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/stage2_ep500_teacher150_g2_${RUN_TAG}_launch.log 2>&1 & \
   LATENT_CKPT_PATH='${CKPT_PATH}' INFERENCE_MODE=teacher SEED_FILE='${SEED_FILE_G3}' GPU_LIST='6 7' RUN_TAG='stage2_ep500_teacher150_g3_${RUN_TAG}' bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/stage2_ep500_teacher150_g3_${RUN_TAG}_launch.log 2>&1 & \
   wait > ${WATCH_LOG} 2>&1"

echo "train_session=${TRAIN_SESSION}"
echo "watch_session=${WATCH_SESSION}"
echo "log_file=${LOG_FILE}"
echo "watch_log=${WATCH_LOG}"
echo "output_dir=${TARGET_DIR}"
