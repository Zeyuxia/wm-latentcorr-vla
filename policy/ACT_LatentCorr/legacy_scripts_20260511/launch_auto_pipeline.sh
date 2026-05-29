#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

PYTHON_BIN=${PYTHON_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python}
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/teacher_mainline/stage1_ddp_schedfix/20260328_012145}
STAGE1_TARGET_EPOCH=${STAGE1_TARGET_EPOCH:-1000}
STAGE1_TMUX_SESSION=${STAGE1_TMUX_SESSION:-act_latent_stage1_ddp_8gpu_schedfix}
STAGE2_OUTPUT_ROOT=${STAGE2_OUTPUT_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/teacher_mainline/stage2_auto}
SUMMARY_ROOT=${SUMMARY_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/teacher_mainline/auto_pipeline_summary}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_DIR="${SUMMARY_ROOT}/${TIMESTAMP}"
mkdir -p "${SUMMARY_DIR}"

SEED_FILE=${SEED_FILE:-/data/zhenyangfan/RoboTwin/data_eval/open_laptop/demo_clean_seed100k/seed.txt}
TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
STAGE2_NUM_EPOCHS=${STAGE2_NUM_EPOCHS:-100}
STAGE2_SAVE_FREQ=${STAGE2_SAVE_FREQ:-10}
STAGE2_GPU=${STAGE2_GPU:-0}
EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}

exec "${PYTHON_BIN}" -m policy.ACT_LatentCorr.auto_pipeline_runner \
  --stage1-output-dir "${STAGE1_OUTPUT_DIR}" \
  --stage1-target-epoch "${STAGE1_TARGET_EPOCH}" \
  --stage1-tmux-session "${STAGE1_TMUX_SESSION}" \
  --stage2-output-root "${STAGE2_OUTPUT_ROOT}" \
  --summary-dir "${SUMMARY_DIR}" \
  --seed-file "${SEED_FILE}" \
  --task-name "${TASK_NAME}" \
  --task-config "${TASK_CONFIG}" \
  --stage2-num-epochs "${STAGE2_NUM_EPOCHS}" \
  --stage2-save-freq "${STAGE2_SAVE_FREQ}" \
  --stage2-gpu "${STAGE2_GPU}" \
  --eval-gpus "${EVAL_GPUS}" \
  "$@"
