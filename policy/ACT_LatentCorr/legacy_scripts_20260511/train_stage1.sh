#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

PYTHON_BIN=${PYTHON_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python}
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

TASK_NAME=${TASK_NAME:-sim-open_laptop-demo_clean-50}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/stage1}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
DEVICE=${DEVICE:-cuda:0}
RAW_DATA_DIR=${RAW_DATA_DIR:-}
FUTURE_TEACHER_SOURCE=${FUTURE_TEACHER_SOURCE:-real}
URDF_PATH=${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}
NUM_EPOCHS=${NUM_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFIX_STEPS=${PREFIX_STEPS:-16}
ACT_CHUNK_SIZE=${ACT_CHUNK_SIZE:-50}
FUTURE_OFFSET=${FUTURE_OFFSET:-16}
MAX_STEPS=${MAX_STEPS:--1}
LR=${LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SAVE_FREQ=${SAVE_FREQ:-1}
LAMBDA_ALIGN=${LAMBDA_ALIGN:-0.1}
BETA_DYNAMICS_MAX=${BETA_DYNAMICS_MAX:-1.0}
LAMBDA_WM_ACTION_CURRENT=${LAMBDA_WM_ACTION_CURRENT:-0.0}
LAMBDA_WM_ACTION_FUTURE=${LAMBDA_WM_ACTION_FUTURE:-0.1}
LAMBDA_BRIDGE_FUTURE=${LAMBDA_BRIDGE_FUTURE:-0.0}
DYN_ZERO_STEPS=${DYN_ZERO_STEPS:-5000}
DYN_RAMP_STEPS=${DYN_RAMP_STEPS:-20000}
DYN_WARMUP_CURVE=${DYN_WARMUP_CURVE:-cosine}
REFERENCE_GLOBAL_BATCH_SIZE=${REFERENCE_GLOBAL_BATCH_SIZE:-4}
PREDICTOR_NUM_BLOCKS=${PREDICTOR_NUM_BLOCKS:-3}
PROJECTOR_MID_CHANNELS=${PROJECTOR_MID_CHANNELS:-256}
WM_ADAPTER_MID_CHANNELS=${WM_ADAPTER_MID_CHANNELS:-128}
READOUT_ADAPTER_MID_CHANNELS=${READOUT_ADAPTER_MID_CHANNELS:-128}
DDIM_STEPS=${DDIM_STEPS:-27}
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-RoboTwin_ACT_LatentCorr}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_LOG_MODE=${WANDB_LOG_MODE:-auto}

if [ -z "${WANDB_RUN_NAME}" ]; then
  WANDB_RUN_NAME="stage1_${TIMESTAMP}"
fi

if [ "${WANDB_LOG_MODE}" = "auto" ]; then
  unset WANDB_MODE
else
  export WANDB_MODE="${WANDB_LOG_MODE}"
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "${PYTHON_BIN}" -m policy.ACT_LatentCorr.train_stage1_latent \
  --task_name "${TASK_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --evac_ckpt /data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt \
  --evac_config /data/zhenyangfan/EVAC_cache/configs/robotwin/train_config.yaml \
  --raw_data_dir "${RAW_DATA_DIR}" \
  --future_teacher_source "${FUTURE_TEACHER_SOURCE}" \
  --urdf_path "${URDF_PATH}" \
  --device "${DEVICE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --save_freq "${SAVE_FREQ}" \
  --ddim_steps "${DDIM_STEPS}" \
  --prefix_steps "${PREFIX_STEPS}" \
  --act_chunk_size "${ACT_CHUNK_SIZE}" \
  --future_offset "${FUTURE_OFFSET}" \
  --lambda_align "${LAMBDA_ALIGN}" \
  --beta_dynamics_max "${BETA_DYNAMICS_MAX}" \
  --lambda_wm_action_current "${LAMBDA_WM_ACTION_CURRENT}" \
  --lambda_wm_action_future "${LAMBDA_WM_ACTION_FUTURE}" \
  --lambda_bridge_future "${LAMBDA_BRIDGE_FUTURE}" \
  --dyn_zero_steps "${DYN_ZERO_STEPS}" \
  --dyn_ramp_steps "${DYN_RAMP_STEPS}" \
  --dyn_warmup_curve "${DYN_WARMUP_CURVE}" \
  --reference_global_batch_size "${REFERENCE_GLOBAL_BATCH_SIZE}" \
  --predictor_num_blocks "${PREDICTOR_NUM_BLOCKS}" \
  --projector_mid_channels "${PROJECTOR_MID_CHANNELS}" \
  --wm_adapter_mid_channels "${WM_ADAPTER_MID_CHANNELS}" \
  --readout_adapter_mid_channels "${READOUT_ADAPTER_MID_CHANNELS}" \
  --use_wandb "${USE_WANDB}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_group "${WANDB_GROUP}" \
  --wandb_mode "${WANDB_LOG_MODE}"
