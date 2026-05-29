#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_snapshot_gpu13_20260422_183001/stage1_unified_epoch_0250.pt
SEED_FILE=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_25.txt
GPU_LIST="0 2"
INFERENCE_MODE=base
TASKS=(
  pick_dual_bottles
  put_bottles_dustbin
  place_burger_fries
  handover_block
)
MASTER_TAG=actmt250_remain_gpu02_$(date +"%Y%m%d_%H%M%S")
MASTER_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${MASTER_TAG}.log
mkdir -p /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "master_tag=$MASTER_TAG"
echo "ckpt=$CKPT"
echo "seed_file=$SEED_FILE"
echo "gpu_list=$GPU_LIST"
echo "inference_mode=$INFERENCE_MODE"

for task in "${TASKS[@]}"; do
  run_tag=${MASTER_TAG}_${task}
  echo "[$(date '+%F %T')] launch task=$task run_tag=$run_tag"
  LATENT_CKPT_PATH="$CKPT" \
  TASK_NAME="$task" \
  TASK_CONFIG=demo_clean \
  INFERENCE_MODE="$INFERENCE_MODE" \
  SEED_FILE="$SEED_FILE" \
  GPU_LIST="$GPU_LIST" \
  RUN_TAG="$run_tag" \
  /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh

  while true; do
    active=$(tmux ls 2>/dev/null | rg -c "^${run_tag}_shard_|^${run_tag}_summary_watch" || true)
    if [ "$active" = "0" ]; then
      break
    fi
    sleep 15
  done

  summary=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag}/summary.txt
  if [ -f "$summary" ]; then
    echo "[$(date '+%F %T')] summary for $task"
    cat "$summary"
  else
    echo "[$(date '+%F %T')] missing summary for $task"
  fi
  echo "[$(date '+%F %T')] finished task=$task run_tag=$run_tag"
done

echo "[$(date '+%F %T')] all remaining epoch250 multitask eval finished"
