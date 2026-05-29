#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

VARIANT_NAME=${VARIANT_NAME:?VARIANT_NAME is required}
BASE_DIR=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/research_variants/${VARIANT_NAME}
STAGE1_ROOT=${STAGE1_ROOT:-${BASE_DIR}/stage1}
STAGE2_ROOT=${STAGE2_ROOT:-${BASE_DIR}/stage2}
ANALYSIS1_ROOT=${ANALYSIS1_ROOT:-${BASE_DIR}/analysis_stage1}
ANALYSIS2_ROOT=${ANALYSIS2_ROOT:-${BASE_DIR}/analysis_stage2}
mkdir -p "${STAGE1_ROOT}" "${STAGE2_ROOT}" "${ANALYSIS1_ROOT}" "${ANALYSIS2_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python}
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

URDF_PATH=${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}
RAW_DATA_DIR=${RAW_DATA_DIR:-}
FUTURE_TEACHER_SOURCE=${FUTURE_TEACHER_SOURCE:-real}
DDIM_STEPS=${DDIM_STEPS:-27}

echo "[variant] ${VARIANT_NAME}"
echo "[variant] stage1 root: ${STAGE1_ROOT}"
echo "[variant] stage2 root: ${STAGE2_ROOT}"

OUTPUT_ROOT="${STAGE1_ROOT}" \
RAW_DATA_DIR="${RAW_DATA_DIR}" \
FUTURE_TEACHER_SOURCE="${FUTURE_TEACHER_SOURCE}" \
URDF_PATH="${URDF_PATH}" \
DDIM_STEPS="${DDIM_STEPS}" \
MAX_STEPS=${STAGE1_MAX_STEPS:-300} \
NUM_EPOCHS=${STAGE1_NUM_EPOCHS:-20} \
BATCH_SIZE=${STAGE1_BATCH_SIZE:-4} \
NUM_WORKERS=${STAGE1_NUM_WORKERS:-2} \
LR=${STAGE1_LR:-3e-5} \
WEIGHT_DECAY=${STAGE1_WEIGHT_DECAY:-1e-4} \
DYN_ZERO_STEPS=${DYN_ZERO_STEPS:-50} \
DYN_RAMP_STEPS=${DYN_RAMP_STEPS:-250} \
LAMBDA_ALIGN=${LAMBDA_ALIGN:-1.0} \
LAMBDA_WM_ACTION_CURRENT=${LAMBDA_WM_ACTION_CURRENT:-0.5} \
LAMBDA_WM_ACTION_FUTURE=${LAMBDA_WM_ACTION_FUTURE:-1.0} \
LAMBDA_BRIDGE_FUTURE=${LAMBDA_BRIDGE_FUTURE:-0.25} \
PREDICTOR_NUM_BLOCKS=${PREDICTOR_NUM_BLOCKS:-3} \
PROJECTOR_MID_CHANNELS=${PROJECTOR_MID_CHANNELS:-256} \
WM_ADAPTER_MID_CHANNELS=${WM_ADAPTER_MID_CHANNELS:-128} \
READOUT_ADAPTER_MID_CHANNELS=${READOUT_ADAPTER_MID_CHANNELS:-128} \
bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1.sh

STAGE1_CKPT=$(ls -1dt "${STAGE1_ROOT}"/*/stage1_epoch_*.pt | head -n 1)
echo "[variant] stage1 ckpt: ${STAGE1_CKPT}"

OUTPUT_ROOT="${STAGE2_ROOT}" \
STAGE1_CKPT="${STAGE1_CKPT}" \
MAX_STEPS=${STAGE2_MAX_STEPS:-30} \
NUM_EPOCHS=${STAGE2_NUM_EPOCHS:-1} \
BATCH_SIZE=${STAGE2_BATCH_SIZE:-1} \
NUM_WORKERS=${STAGE2_NUM_WORKERS:-0} \
LR=${STAGE2_LR:-3e-5} \
WEIGHT_DECAY=${STAGE2_WEIGHT_DECAY:-1e-4} \
RETAIN_WEIGHT=${RETAIN_WEIGHT:-0.1} \
BRIDGE_WEIGHT=${STAGE2_BRIDGE_WEIGHT:-0.0} \
LAMBDA_ALIGN=${LAMBDA_ALIGN:-1.0} \
LAMBDA_WM_ACTION_CURRENT=${LAMBDA_WM_ACTION_CURRENT:-0.5} \
LAMBDA_WM_ACTION_FUTURE=${LAMBDA_WM_ACTION_FUTURE:-1.0} \
LAMBDA_BRIDGE_FUTURE=${LAMBDA_BRIDGE_FUTURE:-0.25} \
PREDICTOR_NUM_BLOCKS=${PREDICTOR_NUM_BLOCKS:-3} \
PROJECTOR_MID_CHANNELS=${PROJECTOR_MID_CHANNELS:-256} \
WM_ADAPTER_MID_CHANNELS=${WM_ADAPTER_MID_CHANNELS:-128} \
READOUT_ADAPTER_MID_CHANNELS=${READOUT_ADAPTER_MID_CHANNELS:-128} \
bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2.sh

STAGE2_CKPT=$(ls -1dt "${STAGE2_ROOT}"/*/stage2_epoch_*.pt | head -n 1)
echo "[variant] stage2 ckpt: ${STAGE2_CKPT}"

"${PYTHON_BIN}" -m policy.ACT_LatentCorr.analyze_stage1_latent \
  --ckpt_path "${STAGE1_CKPT}" \
  --output_dir "${ANALYSIS1_ROOT}" \
  --raw_data_dir "${RAW_DATA_DIR}" \
  --urdf_path "${URDF_PATH}" \
  --num_eval_episodes ${ANALYZE_STAGE1_EPISODES:-12} \
  --samples_per_episode ${ANALYZE_STAGE1_SAMPLES:-4}

"${PYTHON_BIN}" -m policy.ACT_LatentCorr.analyze_stage2_latent \
  --ckpt_path "${STAGE2_CKPT}" \
  --output_dir "${ANALYSIS2_ROOT}" \
  --urdf_path "${URDF_PATH}" \
  --num_eval_episodes ${ANALYZE_STAGE2_EPISODES:-6} \
  --samples_per_episode ${ANALYZE_STAGE2_SAMPLES:-2}

echo "[variant] finished ${VARIANT_NAME}"
