#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin
source /data/miniconda3/etc/profile.d/conda.sh
conda activate ACT

PY_BIN=${PY_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python}
LOG_DIR=${LOG_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/stage1_multitask_cached_$(date +"%Y%m%d_%H%M%S")}
CACHE_DIR=${CACHE_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/cache/multitask_future_latent_fo16_ps16_ddim27}
OUTPUT_DIR=${OUTPUT_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_multitask_cached_$(date +"%Y%m%d_%H%M%S")}
mkdir -p "${LOG_DIR}" "${CACHE_DIR}" "${OUTPUT_DIR}"

declare -a GPU_IDS=(${GPU_IDS:-4 5 6 7})
declare -a TASK_GROUPS=(
  "sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50"
  "sim-put_bottles_dustbin-demo_clean-50"
  "sim-place_burger_fries-demo_clean-50"
  "sim-handover_block-demo_clean-50"
)

for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  tasks="${TASK_GROUPS[$idx]}"
  [ -n "${tasks}" ] || continue
  log_file="${LOG_DIR}/precompute_gpu${gpu}.log"
  heartbeat_file="${LOG_DIR}/precompute_gpu${gpu}_heartbeat.json"
  EVAC_REPO_ROOT=${EVAC_REPO_ROOT:-/data/zhenyangfan/EVAC} \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PY_BIN}" -m policy.ACT_LatentCorr.precompute_multitask_future_latent_cache \
    --multi_task_names ${tasks} \
    --cache_dir "${CACHE_DIR}" \
    --evac_ckpt "${EVAC_CKPT:-/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt}" \
    --evac_config "${EVAC_CONFIG:-/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml}" \
    --urdf_path "${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}" \
    --device cuda:0 \
    --prefix_steps "${PREFIX_STEPS:-16}" \
    --future_offset "${FUTURE_OFFSET:-16}" \
    --ddim_steps "${DDIM_STEPS:-27}" \
    --batch_size "${PRECOMPUTE_BATCH_SIZE:-8}" \
    --heartbeat_path "${heartbeat_file}" \
    > "${log_file}" 2>&1 &
  echo $! > "${LOG_DIR}/precompute_gpu${gpu}.pid"
done

CUDA_VISIBLE_DEVICES=${TRAIN_VISIBLE_GPUS:-4,5,6,7} \
EVAC_REPO_ROOT=${EVAC_REPO_ROOT:-/data/zhenyangfan/EVAC} \
OUTPUT_DIR="${OUTPUT_DIR}" \
NPROC_PER_NODE=${NPROC_PER_NODE:-4} \
ACT_INIT_CKPT="${ACT_INIT_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-robotwin_multitask_5/demo_clean-50/20260417_002800_robotwin_multitask_5_no_wm/policy_epoch_2000_seed_0.ckpt}" \
EVAC_CKPT="${EVAC_CKPT:-/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt}" \
EVAC_CONFIG="${EVAC_CONFIG:-/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml}" \
MULTI_TASK_NAMES="${MULTI_TASK_NAMES:-sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50}" \
NUM_EPOCHS=${NUM_EPOCHS:-1000} \
SAVE_FREQ=${SAVE_FREQ:-50} \
BATCH_SIZE=${BATCH_SIZE:-6} \
NUM_WORKERS=${NUM_WORKERS:-4} \
LR=${LR:-3e-5} \
BASE_ACT_LR_SCALE=${BASE_ACT_LR_SCALE:-1.0} \
LAMBDA_ACTION=${LAMBDA_ACTION:-1.0} \
LAMBDA_ACTION_CONDITIONED=${LAMBDA_ACTION_CONDITIONED:-1.0} \
LAMBDA_ALIGN=${LAMBDA_ALIGN:-0.0} \
BETA_DYNAMICS_MAX=${BETA_DYNAMICS_MAX:-1.0} \
LAMBDA_WM_ACTION_CURRENT=${LAMBDA_WM_ACTION_CURRENT:-0.0} \
LAMBDA_WM_ACTION_FUTURE=${LAMBDA_WM_ACTION_FUTURE:-0.0} \
LAMBDA_BRIDGE_FUTURE=${LAMBDA_BRIDGE_FUTURE:-0.0} \
FREEZE_BASE_ACT=${FREEZE_BASE_ACT:-false} \
FREEZE_READOUT_DECODER=${FREEZE_READOUT_DECODER:-true} \
DETACH_ACT_FEATURE_FOR_LATENT=${DETACH_ACT_FEATURE_FOR_LATENT:-true} \
USE_ACT_HEAD_CONDITIONING=${USE_ACT_HEAD_CONDITIONING:-true} \
USE_RAW_WM_TARGETS=${USE_RAW_WM_TARGETS:-false} \
DYN_ZERO_STEPS=${DYN_ZERO_STEPS:-0} \
DYN_RAMP_STEPS=${DYN_RAMP_STEPS:-1000} \
REFERENCE_GLOBAL_BATCH_SIZE=${REFERENCE_GLOBAL_BATCH_SIZE:-24} \
FUTURE_TEACHER_SOURCE=${FUTURE_TEACHER_SOURCE:-sim} \
FUTURE_LATENT_CACHE_DIR="${CACHE_DIR}" \
FUTURE_LATENT_CACHE_STRICT=${FUTURE_LATENT_CACHE_STRICT:-false} \
FUTURE_LATENT_CACHE_WRITEBACK=${FUTURE_LATENT_CACHE_WRITEBACK:-true} \
USE_WANDB=${USE_WANDB:-false} \
WANDB_LOG_MODE=${WANDB_LOG_MODE:-disabled} \
WANDB_RUN_NAME=${WANDB_RUN_NAME:-stage1_multitask_cached} \
WANDB_GROUP=${WANDB_GROUP:-stage1_multitask_cached} \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_multitask_ddp.sh \
  2>&1 | tee -a "${LOG_DIR}/train.log"
