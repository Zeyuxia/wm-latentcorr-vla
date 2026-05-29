#!/bin/bash
set -euo pipefail

HEARTBEAT=${HEARTBEAT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume250_to700_free012357/20260406_165413/heartbeat.json}
TRAIN_SESSION=${TRAIN_SESSION:-stage2_from_unified_ep400_actalignedcorr_resume250_to700_free012357}
TARGET_EPOCH=${TARGET_EPOCH:-400}
EVAL_LAUNCHER=${EVAL_LAUNCHER:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_eval_train50_ep50_bridge.sh}
STOP_LOG=${STOP_LOG:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/watch_stop_at_400_then_eval_ep50_train50.log}

mkdir -p "$(dirname "${STOP_LOG}")"
echo "[watcher] start $(date '+%F %T')" >> "${STOP_LOG}"
echo "[watcher] heartbeat=${HEARTBEAT}" >> "${STOP_LOG}"
echo "[watcher] train_session=${TRAIN_SESSION} target_epoch=${TARGET_EPOCH}" >> "${STOP_LOG}"

while true; do
  if [ -f "${HEARTBEAT}" ]; then
    epoch=$(python3 - <<PY
import json
with open(${HEARTBEAT@Q}, 'r', encoding='utf-8') as f:
    d=json.load(f)
print(int(d.get("epoch", -1)))
PY
)
    echo "[watcher] $(date '+%F %T') epoch=${epoch}" >> "${STOP_LOG}"
    if [ "${epoch}" -ge "${TARGET_EPOCH}" ]; then
      echo "[watcher] target reached, stopping ${TRAIN_SESSION}" >> "${STOP_LOG}"
      tmux kill-session -t "${TRAIN_SESSION}" 2>/dev/null || true
      sleep 5
      echo "[watcher] launching eval ${EVAL_LAUNCHER}" >> "${STOP_LOG}"
      bash "${EVAL_LAUNCHER}" >> "${STOP_LOG}" 2>&1
      echo "[watcher] done $(date '+%F %T')" >> "${STOP_LOG}"
      exit 0
    fi
  fi
  sleep 30
done
