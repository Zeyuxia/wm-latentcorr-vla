#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="stage1_unified_acthead_resume600_to1000_trainable_base_${TIMESTAMP}"
OUTPUT_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/unified_stage1_acthead_resume600_to1000_trainable_base"
OUTPUT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
LOG_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs"
TRAIN_LOG="${LOG_ROOT}/${RUN_TAG}.log"
WATCH_LOG="${LOG_ROOT}/${RUN_TAG}_watch.log"
SESSION_TRAIN="stage1_unified_acthead_resume600_to1000_trainable_base"
SESSION_WATCH="stage1_unified_acthead_resume600_to1000_trainable_base_watch"
SEED100="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_100.txt"
RESUME_CKPT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/unified_stage1_acthead_resume400_to1000_trainable_base/20260330_170838/stage1_epoch_0600.pt"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/launch_info.txt" <<INFO
run_tag=${RUN_TAG}
output_dir=${OUTPUT_DIR}
train_log=${TRAIN_LOG}
watch_log=${WATCH_LOG}
session_train=${SESSION_TRAIN}
session_watch=${SESSION_WATCH}
resume_ckpt=${RESUME_CKPT}
INFO

TRAIN_CMD="cd /data/zhenyangfan/RoboTwin && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR='${OUTPUT_DIR}' \
OUTPUT_ROOT='${OUTPUT_ROOT}' \
NPROC_PER_NODE=4 \
RESUME_CKPT='${RESUME_CKPT}' \
NUM_EPOCHS=1000 \
SAVE_FREQ=100 \
BATCH_SIZE=1 \
NUM_WORKERS=2 \
LR=3e-5 \
BASE_ACT_LR_SCALE=1.0 \
LAMBDA_ACTION=1.0 \
LAMBDA_ACTION_CONDITIONED=1.0 \
LAMBDA_ALIGN=0.0 \
BETA_DYNAMICS_MAX=1.0 \
LAMBDA_WM_ACTION_CURRENT=0.0 \
LAMBDA_WM_ACTION_FUTURE=0.0 \
LAMBDA_BRIDGE_FUTURE=0.0 \
FREEZE_BASE_ACT=false \
FREEZE_READOUT_DECODER=true \
DETACH_ACT_FEATURE_FOR_LATENT=true \
USE_ACT_HEAD_CONDITIONING=true \
USE_RAW_WM_TARGETS=false \
DYN_ZERO_STEPS=0 \
DYN_RAMP_STEPS=1000 \
REFERENCE_GLOBAL_BATCH_SIZE=4 \
FUTURE_TEACHER_SOURCE=sim \
WANDB_RUN_NAME='${RUN_TAG}' \
WANDB_GROUP='unified_stage1_acthead_resume600_to1000_trainable_base' \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_ddp.sh 2>&1 | tee -a '${TRAIN_LOG}'"

WATCH_CMD="cd /data/zhenyangfan/RoboTwin && \
echo '[watch] waiting for stage1_epoch_1000.pt' | tee -a '${WATCH_LOG}' && \
while [ ! -f '${OUTPUT_DIR}/stage1_epoch_1000.pt' ]; do sleep 60; done && \
echo '[watch] checkpoint ready, launching 100-seed base eval on 4 GPUs' | tee -a '${WATCH_LOG}' && \
LATENT_CKPT_PATH='${OUTPUT_DIR}/stage1_epoch_1000.pt' \
INFERENCE_MODE=base \
SEED_FILE='${SEED100}' \
GPU_LIST='0 1 2 3' \
RUN_TAG='${RUN_TAG}_eval100' \
CKPT_SETTING='${RUN_TAG}_eval100' \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh 2>&1 | tee -a '${WATCH_LOG}'"

tmux kill-session -t "${SESSION_TRAIN}" 2>/dev/null || true
tmux kill-session -t "${SESSION_WATCH}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_TRAIN}" "${TRAIN_CMD}"
tmux new-session -d -s "${SESSION_WATCH}" "${WATCH_CMD}"

echo "started ${RUN_TAG}"
echo "output_dir=${OUTPUT_DIR}"
echo "train_log=${TRAIN_LOG}"
echo "watch_log=${WATCH_LOG}"
