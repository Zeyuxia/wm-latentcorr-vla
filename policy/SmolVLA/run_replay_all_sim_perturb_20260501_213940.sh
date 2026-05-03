#!/bin/bash
set -euo pipefail

ROOT=/data/zhenyangfan/RoboTwin
RUN_DIR=${ROOT}/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high/20260501_213940-stage1_corr025_all5_save_sim_perturb_concat
OUT_DIR=${RUN_DIR}/correction_replay_videos_sim
LOG_DIR=${RUN_DIR}/replay_logs
SCRIPT=${ROOT}/policy/SmolVLA/replay_correction_samples.py
PYTHONPATH_VALUE=${ROOT}/envs/curobo/src:${ROOT}/policy/SmolVLA/src:${ROOT}

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"
source /data/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

launch_rank() {
  local gpu=$1
  local rank=$2
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MPLCONFIGDIR=/tmp/matplotlib-codex \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PYTHONPATH_VALUE}" \
  python "${SCRIPT}" \
    --run-dir "${RUN_DIR}" \
    --output-dir "${OUT_DIR}" \
    --rank "${rank}" \
    --select first \
    --num-samples 0 \
    --fps 10 \
    --hold-frames 5 \
    --record-prefix \
    --prefix-frame-stride 20 \
    --skip-existing \
    > "${LOG_DIR}/replay_rank${rank}_gpu${gpu}.log" 2>&1 &
  echo "$!" > "${LOG_DIR}/replay_rank${rank}_gpu${gpu}.pid"
  echo "launched rank ${rank} on gpu ${gpu}, pid $(cat "${LOG_DIR}/replay_rank${rank}_gpu${gpu}.pid")"
}

launch_rank 4 0
launch_rank 5 1
launch_rank 6 2
launch_rank 7 3
wait

echo "all replay jobs finished at $(date)"
