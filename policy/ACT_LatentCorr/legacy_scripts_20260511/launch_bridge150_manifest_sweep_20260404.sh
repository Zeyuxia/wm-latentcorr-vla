#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

MANIFEST=${MANIFEST:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/bridge_ckpt_manifest_sparse_20260404.txt}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-bridge150_manifest_sweep_20260404_$(date +"%Y%m%d_%H%M%S")}
SWEEP_SESSION=${SWEEP_SESSION:-bridge150_manifest_sweep_20260404}
LOG_FILE=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG_PREFIX}.log
GPU_LIST_G1=${GPU_LIST_G1:-"0 1 2"}
GPU_LIST_G2=${GPU_LIST_G2:-"3 4 5"}
GPU_LIST_G3=${GPU_LIST_G3:-"6 7"}

if tmux has-session -t "${SWEEP_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SWEEP_SESSION}"
fi

tmux new-session -d -s "${SWEEP_SESSION}" \
  "MANIFEST='${MANIFEST}' INFERENCE_MODE=bridge RUN_TAG_PREFIX='${RUN_TAG_PREFIX}' \
   SEED_FILE_G1='/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group1.txt' \
   SEED_FILE_G2='/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group2.txt' \
   SEED_FILE_G3='/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group3.txt' \
   GPU_LIST_G1='${GPU_LIST_G1}' GPU_LIST_G2='${GPU_LIST_G2}' GPU_LIST_G3='${GPU_LIST_G3}' \
   bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_latent150_manifest_sweep.sh > ${LOG_FILE} 2>&1"

echo "sweep_session=${SWEEP_SESSION}"
echo "log_file=${LOG_FILE}"
echo "manifest=${MANIFEST}"
