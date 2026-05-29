#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

EPOCHS=(700 650 600 550 500 450 400 350 300 250 200)
CKPT_DIR=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_partialclean_gpu456_resume150_20260429_210016
GPU_LIST=${GPU_LIST:-"4 5 6 7"}
INFERENCE_MODE=${INFERENCE_MODE:-pred}
EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG:-false}
MASTER_TAG=actmt700to200_gpu4567_pred_eval30_novideo_$(date +"%Y%m%d_%H%M%S")
MASTER_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${MASTER_TAG}.log

mkdir -p /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "master_tag=$MASTER_TAG"
echo "epochs=${EPOCHS[*]}"
echo "ckpt_dir=$CKPT_DIR"
echo "gpu_list=$GPU_LIST"
echo "inference_mode=$INFERENCE_MODE"
echo "eval_video_log=$EVAL_VIDEO_LOG"

for epoch in "${EPOCHS[@]}"; do
  epoch_pad=$(printf "%04d" "$epoch")
  ckpt="${CKPT_DIR}/stage1_unified_epoch_${epoch_pad}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[$(date '+%F %T')] skip missing checkpoint epoch=$epoch ckpt=$ckpt"
    continue
  fi

  echo "[$(date '+%F %T')] start epoch=$epoch ckpt=$ckpt"
  GPU_LIST="$GPU_LIST" \
  INFERENCE_MODE="$INFERENCE_MODE" \
  EVAL_VIDEO_LOG="$EVAL_VIDEO_LOG" \
  CKPT_DIR="$CKPT_DIR" \
  MASTER_TAG_PREFIX="actmt${epoch}_gpu4567_eval30" \
  /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_multitask_eval_generic_gpu0123_30.sh "$epoch"
  echo "[$(date '+%F %T')] finished epoch=$epoch"
done

echo "[$(date '+%F %T')] all requested epoch evals finished"
