#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-zhenyangfan@111.56.136.157}"

RSYNC_OPTS=(
  -avh
  --info=progress2
  --partial
  --append-verify
)

sync_dir() {
  local src="$1"
  local dst="$2"
  rsync "${RSYNC_OPTS[@]}" "${HOST}:${src}" "${dst}"
}

sync_file() {
  local src="$1"
  local dst="$2"
  rsync "${RSYNC_OPTS[@]}" "${HOST}:${src}" "${dst}"
}

echo "[1/3] Creating local target directories"
mkdir -p /data/zhenyangfan/RoboTwin
mkdir -p /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr
mkdir -p /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs
mkdir -p /data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints
mkdir -p /data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints
mkdir -p /data/yujieyang/EVAC/configs/robotwin

echo "[2/3] Pulling missing RoboTwin directories from ${HOST}"
sync_dir /data/zhenyangfan/RoboTwin/script                                                 /data/zhenyangfan/RoboTwin/
sync_dir /data/zhenyangfan/RoboTwin/assets                                                 /data/zhenyangfan/RoboTwin/
sync_dir /data/zhenyangfan/RoboTwin/data                                                   /data/zhenyangfan/RoboTwin/
sync_dir /data/zhenyangfan/RoboTwin/data_eval                                              /data/zhenyangfan/RoboTwin/
sync_dir /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent                /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/
sync_dir /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_failure_workflow /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/
sync_dir /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs                    /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/

echo "[3/3] Pulling missing EVAC checkpoints/configs from ${HOST}"
sync_file /data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt \
          /data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/

sync_file /data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt \
          /data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/

sync_file /data/yujieyang/EVAC/configs/robotwin/train_config_mixed50p12.yaml \
          /data/yujieyang/EVAC/configs/robotwin/

echo
echo "Missing-item pull finished."
echo "Source host: ${HOST}"
