#!/bin/bash
set -euo pipefail

SESSION=${SESSION:-eval_train50_probe25_series}
LOG_FILE=${LOG_FILE:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${SESSION}_$(date +"%Y%m%d_%H%M%S").log}
mkdir -p /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
  "GPU_LIST='0 1 2 3 4 5 6 7' bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_eval_train50_probe25_series.sh > '$LOG_FILE' 2>&1"

echo "session=$SESSION"
echo "log_file=$LOG_FILE"
