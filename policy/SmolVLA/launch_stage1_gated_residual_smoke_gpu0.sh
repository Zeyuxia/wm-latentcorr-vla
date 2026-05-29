#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_RUN=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high_rgbfix/20260516_012823-stage1_actsync_losses_on_from070000_step20000
SMOKE_ROOT=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high_rgbfix/$(date +%Y%m%d_%H%M%S)-stage1_gated_residual_smoke
mkdir -p "$SMOKE_ROOT"
cp "$BASE_RUN/stage1_step_017000.pt" "$SMOKE_ROOT/stage1_step_017000.pt"
export CUDA_VISIBLE_DEVICES=0
export NUM_PROCESSES=1
export MAIN_PROCESS_PORT=29643
export RUN_TAG=stage1_gated_residual_smoke
export TRAIN_TAG=stage1_gated_residual_smoke
export RESUME_FROM="$SMOKE_ROOT/stage1_step_017000.pt"
export MAX_STEPS=17001
export SAVE_FREQ=1
export BATCH_SIZE=2
export NUM_WORKERS=0
export FAILURE_MODE=train
export STAGE1_CORR_SOURCE=offline_export_mixed_dataset
export OFFLINE_CORR_BALANCE_TASKS=true
export STAGE1_CORR_OUTLIER_FILTER=true
export STAGE1_CORR_OUTLIER_MAX_ACTION_LOSS=0.3
export TEACHER_ACTION_WEIGHT=0.0
export MIXED_ACTION_WEIGHT=0.0
export TOKEN_MIX_TEACHER_PROB=0.0
export TOKEN_MIX_ZERO_PROB=0.0
export RESIDUAL_CORRECT_WEIGHT=1.0
export RESIDUAL_RETAIN_WEIGHT=1.0
export RESIDUAL_GATE_CLEAN_WEIGHT=0.1
export RESIDUAL_GATE_CORR_WEIGHT=0.1
export RESIDUAL_DELTA_L1_WEIGHT=0.01
export COND_MAX_WEIGHT=0.0
export DYN_MAX_WEIGHT=1.0
bash "$SCRIPT_DIR/train_stage1.sh"
