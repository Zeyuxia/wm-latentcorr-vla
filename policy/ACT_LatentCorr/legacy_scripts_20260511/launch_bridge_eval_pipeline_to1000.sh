#!/bin/bash
set -euo pipefail
cd /data/zhenyangfan/RoboTwin
SESSION=${SESSION:-bridge_eval_pipeline_to1000}
PIPELINE_TAG=${PIPELINE_TAG:-bridge_eval_pipeline_to1000_$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${PIPELINE_TAG}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/launcher.log"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi
tmux new-session -d -s "${SESSION}" "PIPELINE_TAG='${PIPELINE_TAG}' bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_bridge_eval_pipeline_to1000.sh > '${LOG_FILE}' 2>&1"
echo session=${SESSION}
echo pipeline_tag=${PIPELINE_TAG}
echo log_dir=${LOG_DIR}
echo log_file=${LOG_FILE}
