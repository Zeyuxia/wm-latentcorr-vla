#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_RUN=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high_rgbfix/20260516_012823-stage1_actsync_losses_on_from070000_step20000
SRC_CKPT=$SRC_RUN/stage1_step_017000.pt
NEW_ROOT=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high_rgbfix
NEW_RUN=$NEW_ROOT/$(date +%Y%m%d_%H%M%S)-stage1_tokencouple_resume17000_to20000
mkdir -p "$NEW_RUN"
cp "$SRC_CKPT" "$NEW_RUN/stage1_step_017000.pt"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=29633
export RUN_TAG=stage1_tokencouple_resume17000_to20000
export TRAIN_TAG=stage1_tokencouple
export RESUME_FROM=$NEW_RUN/stage1_step_017000.pt
export MAX_STEPS=20000
export SAVE_FREQ=1000
export BATCH_SIZE=4
export NUM_WORKERS=8
export FAILURE_MODE=train
export STAGE1_CORR_SOURCE=offline_export_mixed_dataset
export OFFLINE_CORR_BALANCE_TASKS=true
export STAGE1_CORR_OUTLIER_FILTER=true
export STAGE1_CORR_OUTLIER_MAX_ACTION_LOSS=0.3
export FAILURE_CORR_BATCH_RATIO=0.25
export TEACHER_ACTION_WEIGHT=0.25
export MIXED_ACTION_WEIGHT=0.25
export TOKEN_MIX_TEACHER_PROB=0.5
export TOKEN_MIX_ZERO_PROB=0.1
export COND_MAX_WEIGHT=0.5
export DYN_MAX_WEIGHT=1.0
bash "$SCRIPT_DIR/train_stage1.sh"
