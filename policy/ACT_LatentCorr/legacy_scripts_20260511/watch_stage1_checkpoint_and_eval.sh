#!/bin/bash
set -euo pipefail

CKPT_PATH=${CKPT_PATH:-}
if [ -z "${CKPT_PATH}" ]; then
  echo "CKPT_PATH is required" >&2
  exit 1
fi

POLL_SECONDS=${POLL_SECONDS:-60}
RUN_TAG=${RUN_TAG:-stage1_eval_watch_$(date +"%Y%m%d_%H%M%S")}

while [ ! -f "${CKPT_PATH}" ]; do
  sleep "${POLL_SECONDS}"
done

cd /data/zhenyangfan/RoboTwin

LATENT_CKPT_PATH="${CKPT_PATH}" \
RUN_TAG="${RUN_TAG}" \
CKPT_SETTING="${RUN_TAG}" \
SEED_FILE="${SEED_FILE:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_50.txt}" \
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}" \
INFERENCE_MODE="${INFERENCE_MODE:-base}" \
TASK_NAME="${TASK_NAME:-open_laptop}" \
TASK_CONFIG="${TASK_CONFIG:-demo_clean}" \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh
