#!/bin/bash
set -euo pipefail
cd /data/zhenyangfan/RoboTwin/policy/SmolVLA
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=29632
export BATCH_SIZE=4
export NUM_WORKERS=8
export MAX_STEPS=15000
export SAVE_FREQ=1000
export FAILURE_MODE=off
export STAGE1_LATENT_TARGET=wm_rollout
export STAGE1_ROLLOUT_DDIM_STEPS=27
export TRAIN_TAG=stage1_ablate_normal_only_wmrollout_from070000_step20000
export RUN_TAG=stage1_ablate_normal_only_wmrollout_resume14000_to15000_gpu0123
export RESUME_FROM=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high_rgbfix/20260517_000318-stage1_ablate_normal_only_wmrollout_from070000_step20000_20260517_000248/stage1_step_014000.pt
export RUNTIME_ROOT=/data/zhenyangfan/runtime_cache
export HF_HOME=/data/.cache/huggingface
bash train_stage1.sh
