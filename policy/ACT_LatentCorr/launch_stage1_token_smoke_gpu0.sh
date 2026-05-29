#!/bin/bash
set -euo pipefail
cd /data/zhenyangfan/RoboTwin
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="stage1_token_smoke_gpu0_${TIMESTAMP}"
OUTPUT_DIR="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/diagnostics/token_smoke/${RUN_TAG}"
LOG_DIR="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}"
LOG_FILE="${LOG_DIR}/train.log"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR="${OUTPUT_DIR}" \
NPROC_PER_NODE=1 \
MASTER_PORT=29841 \
ACT_INIT_CKPT="/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-robotwin_multitask_5/demo_clean-50/20260417_002800_robotwin_multitask_5_no_wm/policy_epoch_2000_seed_0.ckpt" \
FAILURE_TABLE_PATHS="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore/stage2_multitask_failure_explore_new_evac_gpu0123_20260429_004924/failure_explore/failure_table.json" \
EVAC_CKPT="/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt" \
EVAC_CONFIG="/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml" \
MULTI_TASK_NAMES='sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50' \
NUM_EPOCHS=1 \
MAX_STEPS=2 \
SAVE_FREQ=1 \
DDIM_STEPS=2 \
NORMAL_BATCH_SIZE=2 \
FAILURE_BATCH_SIZE=1 \
NUM_WORKERS=0 \
LR=3e-5 \
BASE_ACT_LR_SCALE=1.0 \
LAMBDA_ACTION=1.0 \
LAMBDA_TEACHER_MAX=0.5 \
LAMBDA_PRED_MAX=0.5 \
LAMBDA_LATENT_MAX=0.3 \
LAMBDA_TOKEN_INIT=0.1 \
LAMBDA_TOKEN_LATE=0.02 \
NORMAL_CONDITION_KEEP_PROB=0.5 \
LATENT_LOSS_TYPE=normalized_mse \
TOKEN_LOSS_TYPE=mse \
BETA_DYNAMICS_MAX=1.0 \
LAMBDA_WM_ACTION_CURRENT=0.0 \
LAMBDA_WM_ACTION_FUTURE=0.0 \
LAMBDA_BRIDGE_FUTURE=0.0 \
FREEZE_BASE_ACT=false \
FREEZE_READOUT_DECODER=true \
DETACH_ACT_FEATURE_FOR_LATENT=false \
USE_RAW_WM_TARGETS=false \
DYN_ZERO_STEPS=0 \
DYN_RAMP_STEPS=1000 \
DYN_WARMUP_CURVE=cosine \
REFERENCE_GLOBAL_BATCH_SIZE=6 \
FAILURE_FUTURE_LATENT_MODE=rollout \
BALANCE_FAILURE_TASKS=true \
USE_WANDB=false \
WANDB_LOG_MODE=disabled \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_unified_failure_multitask_ddp.sh \
  2>&1 | tee "${LOG_FILE}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_file=${LOG_FILE}"
