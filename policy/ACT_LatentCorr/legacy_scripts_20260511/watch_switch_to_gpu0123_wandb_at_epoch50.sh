#!/bin/bash
set -euo pipefail

SRC_RUN_DIR="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_fulltable_lctok05_gpu567_20260510_151713"
SRC_LOG="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/stage1_unified_failure_multitask_fulltable_lctok05_gpu567_20260510_151713/train.log"
SRC_SESSION="stage1_unified_fulltable_lctok05_gpu567"
TARGET_CKPT="${SRC_RUN_DIR}/stage1_unified_epoch_0050.pt"
SWITCH_SCRIPT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_stage1_unified_failure_multitask_resume50_wandb_gpu0123.sh"

echo "[watch] source_run=${SRC_RUN_DIR}"
echo "[watch] waiting for checkpoint=${TARGET_CKPT}"

while [ ! -f "${TARGET_CKPT}" ]; do
  sleep 20
done

echo "[watch] checkpoint detected: ${TARGET_CKPT}"

if tmux has-session -t "${SRC_SESSION}" 2>/dev/null; then
  echo "[watch] stopping session ${SRC_SESSION}"
  tmux kill-session -t "${SRC_SESSION}"
else
  echo "[watch] source session ${SRC_SESSION} already stopped"
fi

echo "[watch] launching resume on gpu0123 with wandb"
"${SWITCH_SCRIPT}" "${TARGET_CKPT}"

echo "[watch] switch complete"
