#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

EPOCHS=(300 350 400 450 500 550 600 650)
MASTER_TAG=actmt300to650_gpu0123_eval30_$(date +"%Y%m%d_%H%M%S")
MASTER_LOG=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${MASTER_TAG}.log

mkdir -p /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "master_tag=$MASTER_TAG"
echo "epochs=${EPOCHS[*]}"

for epoch in "${EPOCHS[@]}"; do
  echo "[$(date '+%F %T')] start epoch=${epoch}"
  /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_multitask_eval_generic_gpu0123_30.sh "${epoch}"
  echo "[$(date '+%F %T')] finished epoch=${epoch}"
done

echo "[$(date '+%F %T')] all requested epoch evals finished"
