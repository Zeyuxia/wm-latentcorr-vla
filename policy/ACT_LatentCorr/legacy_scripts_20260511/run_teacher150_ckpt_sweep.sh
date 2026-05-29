#!/bin/bash
set -euo pipefail

CKPT_DIR=${CKPT_DIR:-}
if [ -z "${CKPT_DIR}" ]; then
  echo "CKPT_DIR is required" >&2
  exit 1
fi

RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-stage2_sweep}
START_EPOCH=${START_EPOCH:-700}
END_EPOCH=${END_EPOCH:-1000}
EPOCH_STEP=${EPOCH_STEP:-25}

SEED_FILE_G1=${SEED_FILE_G1:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group1.txt}
SEED_FILE_G2=${SEED_FILE_G2:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group2.txt}
SEED_FILE_G3=${SEED_FILE_G3:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group3.txt}

GPU_LIST_G1=${GPU_LIST_G1:-"2 3"}
GPU_LIST_G2=${GPU_LIST_G2:-"4 5"}
GPU_LIST_G3=${GPU_LIST_G3:-"6 7"}

cd /data/zhenyangfan/RoboTwin

for epoch in $(seq "${START_EPOCH}" "${EPOCH_STEP}" "${END_EPOCH}"); do
  epoch_str=$(printf "%04d" "${epoch}")
  ckpt_path="${CKPT_DIR}/stage2_epoch_${epoch_str}.pt"
  if [ ! -f "${ckpt_path}" ]; then
    echo "[sweep] missing checkpoint: ${ckpt_path}, skip"
    continue
  fi

  echo "[sweep] launch teacher150 for epoch ${epoch} using ${ckpt_path}"

  run_tag_g1="stage2_ep${epoch}_teacher150_g1_${RUN_TAG_PREFIX}"
  run_tag_g2="stage2_ep${epoch}_teacher150_g2_${RUN_TAG_PREFIX}"
  run_tag_g3="stage2_ep${epoch}_teacher150_g3_${RUN_TAG_PREFIX}"

  LATENT_CKPT_PATH="${ckpt_path}" INFERENCE_MODE=teacher SEED_FILE="${SEED_FILE_G1}" GPU_LIST="${GPU_LIST_G1}" RUN_TAG="${run_tag_g1}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh \
    > "/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag_g1}_launch.log" 2>&1

  LATENT_CKPT_PATH="${ckpt_path}" INFERENCE_MODE=teacher SEED_FILE="${SEED_FILE_G2}" GPU_LIST="${GPU_LIST_G2}" RUN_TAG="${run_tag_g2}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh \
    > "/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag_g2}_launch.log" 2>&1

  LATENT_CKPT_PATH="${ckpt_path}" INFERENCE_MODE=teacher SEED_FILE="${SEED_FILE_G3}" GPU_LIST="${GPU_LIST_G3}" RUN_TAG="${run_tag_g3}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh \
    > "/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag_g3}_launch.log" 2>&1

  while true; do
    active_sessions=$(tmux ls 2>/dev/null | rg "^(${run_tag_g1}|${run_tag_g2}|${run_tag_g3})_shard_" | wc -l | tr -d ' ')
    if [ "${active_sessions}" = "0" ]; then
      break
    fi
    echo "[sweep] epoch ${epoch} still running, active shard sessions=${active_sessions}"
    sleep 20
  done

  echo "[sweep] epoch ${epoch} completed"
done

echo "[sweep] all requested checkpoints completed"
