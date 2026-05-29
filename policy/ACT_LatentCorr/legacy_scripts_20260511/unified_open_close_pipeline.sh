#!/bin/bash
set -euo pipefail
shopt -s nullglob

cd /data/zhenyangfan/RoboTwin

ACT_INIT_CKPT=${ACT_INIT_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-open_laptop/demo_clean-50/20260213_001933/policy_epoch_1000_seed_0.ckpt}
PIPE_TAG=${PIPE_TAG:-unified_open_close_$(date +"%Y%m%d_%H%M%S")}
LOG_ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs
STAGE1_ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/unified_stage1_acthead_open1000
STAGE2_ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/unified_stage2_acthead_close100
STAGE1_OUT=${STAGE1_OUT:-${STAGE1_ROOT}/${PIPE_TAG}}
STAGE2_OUT=${STAGE2_OUT:-${STAGE2_ROOT}/${PIPE_TAG}}
PIPE_LOG=${LOG_ROOT}/${PIPE_TAG}_pipeline.log
STAGE1_LOG=${LOG_ROOT}/${PIPE_TAG}_stage1.log
STAGE2_LOG=${LOG_ROOT}/${PIPE_TAG}_stage2.log
SUMMARY_FILE=${LOG_ROOT}/${PIPE_TAG}_summary.txt
SEED50=${SEED50:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_50.txt}
TRAIN_CUDA=${TRAIN_CUDA:-0,1,2,3,4,5,6,7}
EVAL_ALL_GPUS=${EVAL_ALL_GPUS:-"0 1 2 3 4 5 6 7"}
EVAL_BASE_GPUS=${EVAL_BASE_GPUS:-"0 1 2 3"}
EVAL_TEACHER_GPUS=${EVAL_TEACHER_GPUS:-"4 5 6 7"}

mkdir -p "${LOG_ROOT}" "${STAGE1_ROOT}" "${STAGE2_ROOT}"
: > "${PIPE_LOG}"
: > "${SUMMARY_FILE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PIPE_LOG}"
}

wait_for_file() {
  local path="$1"
  while [ ! -f "${path}" ]; do
    sleep 60
  done
}

wait_for_eval_results() {
  local run_tag="$1"
  local expected="$2"
  local count=0
  while true; do
    count=$(find /data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean \
      -path "*/${run_tag}_shard_*/*/_result.txt" 2>/dev/null | wc -l)
    if [ "${count}" -ge "${expected}" ]; then
      break
    fi
    sleep 60
  done
}

summarize_eval() {
  local run_tag="$1"
  local log_dir="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_tag}"
  local seed_dir
  seed_dir=$(cat "${log_dir}/seed_dir.txt")
  /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python - "$run_tag" "$seed_dir" <<'PY'
import glob
import os
import sys
run_tag = sys.argv[1]
seed_dir = sys.argv[2].strip()
base = "/data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean"
result_paths = sorted(glob.glob(os.path.join(base, f"{run_tag}_shard_*", "*", "_result.txt")))
if not result_paths:
    raise SystemExit(f"no results for {run_tag}")
success = 0
count = 0
parts = []
for path in result_paths:
    shard = os.path.basename(os.path.dirname(os.path.dirname(path)))
    shard_idx = int(shard.rsplit("_", 1)[-1])
    seed_path = os.path.join(seed_dir, f"shard_{shard_idx}.txt")
    with open(seed_path, "r", encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    with open(path, "r", encoding="utf-8") as f:
        ratio = float(f.read().strip())
    s = int(round(ratio * n))
    success += s
    count += n
    parts.append(f"shard{shard_idx}:{s}/{n}")
rate = 100.0 * success / max(1, count)
print(f"{success}/{count} = {rate:.1f}% | " + ", ".join(parts))
PY
}

log "pipeline=${PIPE_TAG}"
log "stage1_out=${STAGE1_OUT}"
log "stage2_out=${STAGE2_OUT}"
log "act_init_ckpt=${ACT_INIT_CKPT}"
log "train_cuda=${TRAIN_CUDA}"

log "[1/4] stage1 open-loop training start"
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" \
OUTPUT_DIR="${STAGE1_OUT}" \
OUTPUT_ROOT="${STAGE1_ROOT}" \
NPROC_PER_NODE=8 \
ACT_INIT_CKPT="${ACT_INIT_CKPT}" \
NUM_EPOCHS=1000 \
SAVE_FREQ=100 \
BATCH_SIZE=1 \
NUM_WORKERS=2 \
LR=3e-5 \
BASE_ACT_LR_SCALE=0.0 \
LAMBDA_ACTION=0.0 \
LAMBDA_ACTION_CONDITIONED=1.0 \
LAMBDA_ALIGN=0.0 \
BETA_DYNAMICS_MAX=1.0 \
LAMBDA_WM_ACTION_CURRENT=0.0 \
LAMBDA_WM_ACTION_FUTURE=0.0 \
LAMBDA_BRIDGE_FUTURE=0.0 \
FREEZE_BASE_ACT=true \
FREEZE_READOUT_DECODER=true \
DETACH_ACT_FEATURE_FOR_LATENT=true \
USE_ACT_HEAD_CONDITIONING=true \
USE_RAW_WM_TARGETS=false \
DYN_ZERO_STEPS=0 \
DYN_RAMP_STEPS=1000 \
REFERENCE_GLOBAL_BATCH_SIZE=8 \
FUTURE_TEACHER_SOURCE=sim \
WANDB_RUN_NAME="${PIPE_TAG}_stage1_open1000" \
WANDB_GROUP="unified_stage1_open1000" \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_ddp.sh > "${STAGE1_LOG}" 2>&1

STAGE1_CKPT="${STAGE1_OUT}/stage1_epoch_1000.pt"
wait_for_file "${STAGE1_CKPT}"
log "stage1 finished ckpt=${STAGE1_CKPT}"

RUN_TAG_STAGE1="${PIPE_TAG}_stage1_base50"
log "[2/4] stage1 open-loop eval start run_tag=${RUN_TAG_STAGE1}"
LATENT_CKPT_PATH="${STAGE1_CKPT}" \
INFERENCE_MODE=base \
SEED_FILE="${SEED50}" \
GPU_LIST="${EVAL_ALL_GPUS}" \
RUN_TAG="${RUN_TAG_STAGE1}" \
CKPT_SETTING="${RUN_TAG_STAGE1}" \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh | tee -a "${PIPE_LOG}"
wait_for_eval_results "${RUN_TAG_STAGE1}" 8
stage1_eval_summary=$(summarize_eval "${RUN_TAG_STAGE1}")
echo "stage1_base50 ${stage1_eval_summary}" | tee -a "${SUMMARY_FILE}" "${PIPE_LOG}"

log "[3/4] stage2 closed-loop training start"
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" \
STAGE1_CKPT="${STAGE1_CKPT}" \
OUTPUT_DIR="${STAGE2_OUT}" \
OUTPUT_ROOT="${STAGE2_ROOT}" \
NPROC_PER_NODE=8 \
NUM_EPOCHS=100 \
SAVE_FREQ=10 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
LR=3e-5 \
RETAIN_WEIGHT=0.0 \
BRIDGE_WEIGHT=0.0 \
FREEZE_BASE_ACT=true \
DETACH_ACT_FEATURE_FOR_LATENT=true \
USE_ACT_HEAD_CORRECTION=true \
LAMBDA_ALIGN=0.0 \
BETA_DYNAMICS_MAX=1.0 \
LAMBDA_WM_ACTION_CURRENT=0.0 \
LAMBDA_WM_ACTION_FUTURE=0.0 \
LAMBDA_BRIDGE_FUTURE=0.0 \
DYN_ZERO_STEPS=0 \
DYN_RAMP_STEPS=400 \
REFERENCE_GLOBAL_BATCH_SIZE=8 \
PLANNER_WARMUP=false \
CORRECTION_INTERP_NEAREST_ENABLE=true \
CORRECTION_INTERP_PREFIX_RATIO=0.4 \
WANDB_RUN_NAME="${PIPE_TAG}_stage2_close100" \
WANDB_GROUP="unified_stage2_close100" \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_ddp.sh > "${STAGE2_LOG}" 2>&1

STAGE2_CKPT="${STAGE2_OUT}/stage2_epoch_0100.pt"
wait_for_file "${STAGE2_CKPT}"
log "stage2 finished ckpt=${STAGE2_CKPT}"

RUN_TAG_STAGE2_BASE="${PIPE_TAG}_stage2_base50"
RUN_TAG_STAGE2_TEACHER="${PIPE_TAG}_stage2_teacher50"
log "[4/4] stage2 closed-loop eval start base=${RUN_TAG_STAGE2_BASE} teacher=${RUN_TAG_STAGE2_TEACHER}"
LATENT_CKPT_PATH="${STAGE2_CKPT}" \
INFERENCE_MODE=base \
SEED_FILE="${SEED50}" \
GPU_LIST="${EVAL_BASE_GPUS}" \
RUN_TAG="${RUN_TAG_STAGE2_BASE}" \
CKPT_SETTING="${RUN_TAG_STAGE2_BASE}" \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh | tee -a "${PIPE_LOG}"
LATENT_CKPT_PATH="${STAGE2_CKPT}" \
INFERENCE_MODE=teacher \
SEED_FILE="${SEED50}" \
GPU_LIST="${EVAL_TEACHER_GPUS}" \
RUN_TAG="${RUN_TAG_STAGE2_TEACHER}" \
CKPT_SETTING="${RUN_TAG_STAGE2_TEACHER}" \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh | tee -a "${PIPE_LOG}"

wait_for_eval_results "${RUN_TAG_STAGE2_BASE}" 4
wait_for_eval_results "${RUN_TAG_STAGE2_TEACHER}" 4
stage2_base_summary=$(summarize_eval "${RUN_TAG_STAGE2_BASE}")
stage2_teacher_summary=$(summarize_eval "${RUN_TAG_STAGE2_TEACHER}")
echo "stage2_base50 ${stage2_base_summary}" | tee -a "${SUMMARY_FILE}" "${PIPE_LOG}"
echo "stage2_teacher50 ${stage2_teacher_summary}" | tee -a "${SUMMARY_FILE}" "${PIPE_LOG}"

log "pipeline finished"
log "summary_file=${SUMMARY_FILE}"
