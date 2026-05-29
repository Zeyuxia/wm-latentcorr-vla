#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

EPOCH=${1:?usage: run_multitask_eval_generic_gpu0123_30.sh <epoch>}
EPOCH_PAD=$(printf "%04d" "${EPOCH}")

CKPT_DIR=${CKPT_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_fulltable_lctok05_gpu0123_20260511_163455}
CKPT=${CKPT_DIR}/stage1_unified_epoch_${EPOCH_PAD}.pt
SEED30=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_30.txt
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
INFERENCE_MODE=${INFERENCE_MODE:-pred}
EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG:-false}
TASKS=(
  open_laptop
  pick_dual_bottles
  put_bottles_dustbin
  place_burger_fries
  handover_block
)
MASTER_TAG_PREFIX=${MASTER_TAG_PREFIX:-actmt${EPOCH}_gpu0123_${INFERENCE_MODE}_eval30}
MASTER_TAG=${MASTER_TAG_PREFIX}_$(date +"%Y%m%d_%H%M%S")
MASTER_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${MASTER_TAG}.log

if [ ! -f "${CKPT}" ]; then
  echo "checkpoint not found: ${CKPT}" >&2
  exit 1
fi

mkdir -p /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs
sed -n '1,30p' /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_50.txt > "${SEED30}"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "master_tag=$MASTER_TAG"
echo "epoch=$EPOCH"
echo "ckpt=$CKPT"
echo "seed_file=$SEED30"
echo "gpu_list=$GPU_LIST"
echo "inference_mode=$INFERENCE_MODE"
echo "eval_video_log=$EVAL_VIDEO_LOG"

for task in "${TASKS[@]}"; do
  run_tag=${MASTER_TAG}_${task}
  echo "[$(date '+%F %T')] launch task=$task run_tag=$run_tag"
  LATENT_CKPT_PATH="$CKPT" \
  TASK_NAME="$task" \
  TASK_CONFIG=demo_clean \
  INFERENCE_MODE="$INFERENCE_MODE" \
  SEED_FILE="$SEED30" \
  GPU_LIST="$GPU_LIST" \
  EVAL_VIDEO_LOG="$EVAL_VIDEO_LOG" \
  RUN_TAG="$run_tag" \
  /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh

  while true; do
    active=$( (tmux ls 2>/dev/null || true) | { grep -E "^(${run_tag}_shard_|${run_tag}_summary_watch)" || true; } | wc -l | tr -d ' ')
    if [ "$active" = "0" ]; then
      break
    fi
    sleep 20
  done

  log_root=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag}
  seed_dir=$(cat "${log_root}/seed_dir.txt")
  python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_parallel_eval_summary.py \
    --run-tag "$run_tag" \
    --log-root "$log_root" \
    --seed-dir "$seed_dir" || true
  python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/summarize_total_success.py \
    --run-tag "$run_tag" \
    --log-root "$log_root" \
    --seed-dir "$seed_dir"
  python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/summarize_first25_nonskip.py \
    --run-tag "$run_tag" \
    --log-root "$log_root" \
    --seed-dir "$seed_dir" \
    --take 25

  if [ -f "${log_root}/total_success.txt" ]; then
    echo "[$(date '+%F %T')] total success summary for $task"
    cat "${log_root}/total_success.txt"
  fi
  if [ -f "${log_root}/reference_first25_nonskip.txt" ]; then
    echo "[$(date '+%F %T')] reference summary for $task"
    cat "${log_root}/reference_first25_nonskip.txt"
  fi
  echo "[$(date '+%F %T')] finished task=$task run_tag=$run_tag"
done

python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_multitask_eval_overview.py \
  --master-tag "$MASTER_TAG"
echo "[$(date '+%F %T')] all epoch${EPOCH} multitask eval finished"
