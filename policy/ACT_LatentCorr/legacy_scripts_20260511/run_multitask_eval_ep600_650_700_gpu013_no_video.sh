#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

GPU_LIST=${GPU_LIST:-"0 1 3"}
EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG:-false}

for epoch in 600 650 700; do
  echo "[$(date '+%F %T')] launch epoch=${epoch} gpu_list=${GPU_LIST} eval_video_log=${EVAL_VIDEO_LOG}"
  GPU_LIST="${GPU_LIST}" EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG}" \
    /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_multitask_eval_generic_gpu0123_30.sh "${epoch}"
  echo "[$(date '+%F %T')] finished epoch=${epoch}"
done
