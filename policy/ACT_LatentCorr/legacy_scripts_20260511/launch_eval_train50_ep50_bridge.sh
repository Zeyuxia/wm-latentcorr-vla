#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

LATENT_CKPT_PATH=${LATENT_CKPT_PATH:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_free0123/20260405_220904/stage2_epoch_0050.pt}
TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
INFERENCE_MODE=${INFERENCE_MODE:-bridge}
RUN_TAG=${RUN_TAG:-ep50_train50_bridge_$(date +"%Y%m%d_%H%M%S")}
GPU_LIST=${GPU_LIST:-"0 1 2 3 5 7"}
SEED_FILE=${SEED_FILE:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_demo_clean_train50.txt}

LATENT_CKPT_PATH="${LATENT_CKPT_PATH}" \
TASK_NAME="${TASK_NAME}" \
TASK_CONFIG="${TASK_CONFIG}" \
INFERENCE_MODE="${INFERENCE_MODE}" \
SEED_FILE="${SEED_FILE}" \
GPU_LIST="${GPU_LIST}" \
RUN_TAG="${RUN_TAG}" \
CKPT_SETTING="${RUN_TAG}" \
bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh
