#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

TASKS=(
  open_laptop
  pick_dual_bottles
  put_bottles_dustbin
  place_burger_fries
  handover_block
)

CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_fulltable_lctok05_gpu567_20260510_151713/stage1_unified_epoch_0050.pt
GPU_LIST=${GPU_LIST:-"5 6 7"}
INFERENCE_MODE=${INFERENCE_MODE:-pred}
EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG:-false}
EVAC_CKPT=${EVAC_CKPT:-/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt}
SEED_FILE=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_100.txt
MASTER_TAG=${MASTER_TAG:-actmt50_gpu567_pred_eval100_novideo_$(date +"%Y%m%d_%H%M%S")}
MASTER_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${MASTER_TAG}.log

if [ ! -f "${CKPT}" ]; then
  echo "checkpoint not found: ${CKPT}" >&2
  exit 1
fi

mkdir -p /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "master_tag=$MASTER_TAG"
echo "ckpt=$CKPT"
echo "seed_file=$SEED_FILE"
echo "gpu_list=$GPU_LIST"
echo "inference_mode=$INFERENCE_MODE"
echo "eval_video_log=$EVAL_VIDEO_LOG"
echo "evac_ckpt=$EVAC_CKPT"

for task in "${TASKS[@]}"; do
  run_tag=${MASTER_TAG}_${task}
  log_root=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag}
  if [ -f "${log_root}/summary.txt" ]; then
    echo "[$(date '+%F %T')] skip already completed task=$task run_tag=$run_tag"
    continue
  fi

  echo "[$(date '+%F %T')] launch task=$task run_tag=$run_tag"
  LATENT_CKPT_PATH="$CKPT" \
  TASK_NAME="$task" \
  TASK_CONFIG=demo_clean \
  INFERENCE_MODE="$INFERENCE_MODE" \
  SEED_FILE="$SEED_FILE" \
  GPU_LIST="$GPU_LIST" \
  EVAL_VIDEO_LOG="$EVAL_VIDEO_LOG" \
  EVAC_CKPT="$EVAC_CKPT" \
  RUN_TAG="$run_tag" \
  /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh

  while true; do
    active=$( (tmux ls 2>/dev/null || true) | { rg "^${run_tag}_shard_|^${run_tag}_summary_watch" || true; } | wc -l | tr -d ' ' )
    if [ "$active" = "0" ]; then
      break
    fi
    sleep 20
  done

  seed_dir=$(cat "${log_root}/seed_dir.txt")
  result_root=/data/zhenyangfan/RoboTwin/eval_result/${task}/ACT_LatentCorr/demo_clean
  python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_parallel_eval_summary.py \
    --run-tag "$run_tag" \
    --log-root "$log_root" \
    --seed-dir "$seed_dir" \
    --result-root "$result_root" || true
  python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/summarize_first25_nonskip.py \
    --run-tag "$run_tag" \
    --log-root "$log_root" \
    --seed-dir "$seed_dir" \
    --take 100

  if [ -f "${log_root}/reference_first25_nonskip.txt" ]; then
    echo "[$(date '+%F %T')] reference summary for $task"
    cat "${log_root}/reference_first25_nonskip.txt"
  fi
  echo "[$(date '+%F %T')] finished task=$task run_tag=$run_tag"
done

echo "[$(date '+%F %T')] all epoch50 multitask eval finished"
