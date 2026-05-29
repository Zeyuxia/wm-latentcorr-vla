#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

BASE_RUN_DIR=${BASE_RUN_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_probe25_free01234567/20260406_205807}
GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
SEED_FILE=${SEED_FILE:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_demo_clean_train50.txt}
TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
INFERENCE_MODE=${INFERENCE_MODE:-bridge}
RUN_PREFIX=${RUN_PREFIX:-probe25_train50_bridge}

for EPOCH in 5 10 15 20 25; do
  CKPT=$(printf "%s/stage2_epoch_%04d.pt" "$BASE_RUN_DIR" "$EPOCH")
  if [ ! -f "$CKPT" ]; then
    echo "missing checkpoint: $CKPT" >&2
    exit 1
  fi
  RUN_TAG=$(printf "%s_ep%03d_%s" "$RUN_PREFIX" "$EPOCH" "$(date +"%Y%m%d_%H%M%S")")
  echo "[series] launch epoch=$EPOCH ckpt=$CKPT run_tag=$RUN_TAG"
  LATENT_CKPT_PATH="$CKPT" \
  TASK_NAME="$TASK_NAME" \
  TASK_CONFIG="$TASK_CONFIG" \
  INFERENCE_MODE="$INFERENCE_MODE" \
  SEED_FILE="$SEED_FILE" \
  GPU_LIST="$GPU_LIST" \
  RUN_TAG="$RUN_TAG" \
  CKPT_SETTING="$RUN_TAG" \
  bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh

  LOG_DIR="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}"
  while [ ! -f "$LOG_DIR/summary.txt" ]; do
    sleep 10
  done
  echo "[series] completed epoch=$EPOCH"
  sed -n '1,20p' "$LOG_DIR/summary.txt"
  echo
 done
