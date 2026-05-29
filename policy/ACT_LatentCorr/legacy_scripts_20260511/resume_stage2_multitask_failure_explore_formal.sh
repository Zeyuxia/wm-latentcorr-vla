#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

# Resume the previous formal multitask explore run in-place by reusing its
# OUTPUT_DIR. train_stage2_latent.py will restore progress from
# OUTPUT_DIR/failure_explore/failure_trials_live_rank*.json.
OUTPUT_DIR=${OUTPUT_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore/stage2_multitask_failure_explore_formal_20260421_202855_gpu1234_full_tmux}
GPU_LIST=${GPU_LIST:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29721}
RUN_TAG=${RUN_TAG:-$(basename "${OUTPUT_DIR}")_resume_$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=${LOG_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}

mkdir -p "${LOG_DIR}"

RUN_ROOT=$(dirname "${OUTPUT_DIR}") \
OUTPUT_DIR="${OUTPUT_DIR}" \
RUN_TAG="${RUN_TAG}" \
LOG_FILE="${LOG_FILE}" \
GPU_LIST="${GPU_LIST}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT}" \
bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_stage2_multitask_failure_explore.sh
