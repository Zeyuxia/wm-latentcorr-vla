#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore
RUN_TAG=stage2_multitask_failure_explore_new_evac_gpu0123_${TIMESTAMP}
OUTPUT_DIR=${RUN_ROOT}/${RUN_TAG}
LOG_DIR=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs
LOG_FILE=${LOG_DIR}/${RUN_TAG}.log
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_ROOT="${RUN_ROOT}" \
RUN_TAG="${RUN_TAG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
LOG_FILE="${LOG_FILE}" \
GPU_LIST=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29731 \
EVAC_CKPT=/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt \
EVAC_CONFIG=/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml \
EVAC_REPO_ROOT=/data/yujieyang/EVAC_new \
RECOVER_EVAL_ENABLE=true \
RECOVER_EVAL_SAVE_VIDEO=false \
EXPLORE_DEBUG_MAX_SAMPLES=2 \
EXPLORE_DEBUG_DIR="${OUTPUT_DIR}/explore_debug" \
bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_stage2_multitask_failure_explore.sh
