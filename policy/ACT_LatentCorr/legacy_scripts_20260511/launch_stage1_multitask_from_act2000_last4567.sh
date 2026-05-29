#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="stage1_multitask_from_act2000_last4567_${TIMESTAMP}"
OUTPUT_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_multitask_from_act2000_last4567"
OUTPUT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
LOG_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs"
TRAIN_LOG="${LOG_ROOT}/${RUN_TAG}.log"
SESSION_TRAIN="stage1_multitask_from_act2000_last4567"

ACT_INIT_CKPT="/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-robotwin_multitask_5/demo_clean-50/20260417_002800_robotwin_multitask_5_no_wm/policy_epoch_2000_seed_0.ckpt"
EVAC_CKPT="/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt"
EVAC_CONFIG="/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml"
MULTI_TASK_NAMES="sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/launch_info.txt" <<EOF
run_tag=${RUN_TAG}
output_dir=${OUTPUT_DIR}
train_log=${TRAIN_LOG}
session_train=${SESSION_TRAIN}
act_init_ckpt=${ACT_INIT_CKPT}
evac_ckpt=${EVAC_CKPT}
evac_config=${EVAC_CONFIG}
multi_task_names=${MULTI_TASK_NAMES}
EOF

TRAIN_CMD="cd /data/zhenyangfan/RoboTwin && \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
EVAC_REPO_ROOT='/data/zhenyangfan/EVAC' \
OUTPUT_DIR='${OUTPUT_DIR}' \
NPROC_PER_NODE=4 \
ACT_INIT_CKPT='${ACT_INIT_CKPT}' \
EVAC_CKPT='${EVAC_CKPT}' \
EVAC_CONFIG='${EVAC_CONFIG}' \
MULTI_TASK_NAMES='${MULTI_TASK_NAMES}' \
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
USE_WANDB=false \
WANDB_LOG_MODE=disabled \
WANDB_RUN_NAME='${RUN_TAG}' \
WANDB_GROUP='stage1_multitask_from_act2000_last4567' \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_multitask_ddp.sh 2>&1 | tee -a '${TRAIN_LOG}'"

tmux kill-session -t "${SESSION_TRAIN}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_TRAIN}" "${TRAIN_CMD}"

echo "started ${RUN_TAG}"
echo "output_dir=${OUTPUT_DIR}"
echo "train_log=${TRAIN_LOG}"
echo "session_train=${SESSION_TRAIN}"
