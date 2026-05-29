#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

ACT_INIT_CKPT=${ACT_INIT_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-open_laptop/demo_clean-50/20260213_001933/policy_epoch_1000_seed_0.ckpt}
RUN_ROOT=${RUN_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_from_act1000_freezebase_latent100}
RUN_TAG=${RUN_TAG:-stage1_from_act1000_freezebase_latent100_$(date +"%Y%m%d_%H%M%S")}
LOG_FILE=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}.log
WATCH_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}_watch.log

TRAIN_SESSION=${TRAIN_SESSION:-stage1_from_act1000_freezebase_latent100}
WATCH_SESSION=${WATCH_SESSION:-stage1_from_act1000_freezebase_latent100_watch}

if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi
if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${WATCH_SESSION}"
fi

tmux new-session -d -s "${TRAIN_SESSION}" \
  "NPROC_PER_NODE=8 \
   ACT_INIT_CKPT=${ACT_INIT_CKPT} \
   OUTPUT_ROOT=${RUN_ROOT} \
   NUM_EPOCHS=100 \
   SAVE_FREQ=25 \
   BATCH_SIZE=1 \
   LR=3e-5 \
   BASE_ACT_LR_SCALE=0.0 \
   LAMBDA_ACTION=0.0 \
   LAMBDA_ALIGN=0.1 \
   BETA_DYNAMICS_MAX=0.5 \
   LAMBDA_WM_ACTION_CURRENT=0.0 \
   LAMBDA_WM_ACTION_FUTURE=0.0 \
   LAMBDA_BRIDGE_FUTURE=0.0 \
   FREEZE_BASE_ACT=true \
   FREEZE_READOUT_DECODER=true \
   DETACH_ACT_FEATURE_FOR_LATENT=true \
   USE_RAW_WM_TARGETS=true \
   DYN_ZERO_STEPS=0 \
   DYN_RAMP_STEPS=300 \
   FUTURE_TEACHER_SOURCE=sim \
   WANDB_RUN_NAME=${RUN_TAG} \
   WANDB_GROUP=stage1_from_act1000_freezebase_latent100 \
   /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_ddp.sh > ${LOG_FILE} 2>&1"

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

tmux new-session -d -s "${WATCH_SESSION}" \
  "CKPT_PATH=${TARGET_DIR}/stage1_epoch_0100.pt \
   RUN_TAG=${RUN_TAG}_eval50 \
   /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/watch_stage1_checkpoint_and_eval.sh > ${WATCH_LOG} 2>&1"

echo "train_session=${TRAIN_SESSION}"
echo "watch_session=${WATCH_SESSION}"
echo "log_file=${LOG_FILE}"
echo "watch_log=${WATCH_LOG}"
echo "output_dir=${TARGET_DIR}"
