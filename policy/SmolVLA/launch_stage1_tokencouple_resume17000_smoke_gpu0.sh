#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=0
export NUM_PROCESSES=1
export MAIN_PROCESS_PORT=29631
export RUN_TAG=stage1_tokencouple_smoke_resume17000_step17001
export TRAIN_TAG=stage1_tokencouple_smoke
export RESUME_FROM=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high_rgbfix/20260516_012823-stage1_actsync_losses_on_from070000_step20000/stage1_step_017000.pt
export MAX_STEPS=17001
export SAVE_FREQ=1
export BATCH_SIZE=1
export NUM_WORKERS=0
export FAILURE_MODE=off
export STAGE1_CORR_SOURCE=offline_export_mixed_dataset
export TEACHER_ACTION_WEIGHT=0.25
export MIXED_ACTION_WEIGHT=0.25
export TOKEN_MIX_TEACHER_PROB=0.5
export TOKEN_MIX_ZERO_PROB=0.1
export COND_MAX_WEIGHT=0.5
export DYN_MAX_WEIGHT=1.0
bash "$SCRIPT_DIR/train_stage1.sh"
