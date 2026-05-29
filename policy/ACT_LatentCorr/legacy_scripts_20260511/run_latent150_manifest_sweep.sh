#!/bin/bash
set -euo pipefail

MANIFEST=${MANIFEST:-}
if [ -z "${MANIFEST}" ]; then
  echo "MANIFEST is required" >&2
  exit 1
fi
if [ ! -f "${MANIFEST}" ]; then
  echo "manifest not found: ${MANIFEST}" >&2
  exit 1
fi

INFERENCE_MODE=${INFERENCE_MODE:-teacher}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-latent150_sweep}

SEED_FILE_G1=${SEED_FILE_G1:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group1.txt}
SEED_FILE_G2=${SEED_FILE_G2:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group2.txt}
SEED_FILE_G3=${SEED_FILE_G3:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group3.txt}

GPU_LIST_G1=${GPU_LIST_G1:-"2 3"}
GPU_LIST_G2=${GPU_LIST_G2:-"4 5"}
GPU_LIST_G3=${GPU_LIST_G3:-"6 7"}

cd /data/zhenyangfan/RoboTwin

while IFS='|' read -r label ckpt_path; do
  [ -z "${label}" ] && continue
  [ -z "${ckpt_path}" ] && continue
  if [ ! -f "${ckpt_path}" ]; then
    echo "[manifest-sweep] missing checkpoint for ${label}: ${ckpt_path}, skip"
    continue
  fi

  echo "[manifest-sweep] launch ${INFERENCE_MODE}150 for ${label} using ${ckpt_path}"

  run_tag_g1="${label}_${INFERENCE_MODE}150_g1_${RUN_TAG_PREFIX}"
  run_tag_g2="${label}_${INFERENCE_MODE}150_g2_${RUN_TAG_PREFIX}"
  run_tag_g3="${label}_${INFERENCE_MODE}150_g3_${RUN_TAG_PREFIX}"

  LATENT_CKPT_PATH="${ckpt_path}" INFERENCE_MODE="${INFERENCE_MODE}" SEED_FILE="${SEED_FILE_G1}" GPU_LIST="${GPU_LIST_G1}" RUN_TAG="${run_tag_g1}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh \
    > "/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag_g1}_launch.log" 2>&1

  LATENT_CKPT_PATH="${ckpt_path}" INFERENCE_MODE="${INFERENCE_MODE}" SEED_FILE="${SEED_FILE_G2}" GPU_LIST="${GPU_LIST_G2}" RUN_TAG="${run_tag_g2}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh \
    > "/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag_g2}_launch.log" 2>&1

  LATENT_CKPT_PATH="${ckpt_path}" INFERENCE_MODE="${INFERENCE_MODE}" SEED_FILE="${SEED_FILE_G3}" GPU_LIST="${GPU_LIST_G3}" RUN_TAG="${run_tag_g3}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh \
    > "/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag_g3}_launch.log" 2>&1

  while true; do
    active_sessions=$(tmux ls 2>/dev/null | rg "^(${run_tag_g1}|${run_tag_g2}|${run_tag_g3})_shard_" | wc -l | tr -d ' ')
    if [ "${active_sessions}" = "0" ]; then
      break
    fi
    echo "[manifest-sweep] ${label} still running, active shard sessions=${active_sessions}"
    sleep 20
  done

  echo "[manifest-sweep] ${label} completed"
done < "${MANIFEST}"

echo "[manifest-sweep] all requested checkpoints completed"
