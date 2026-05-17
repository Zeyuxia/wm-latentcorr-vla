#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_SRC_DIR="${SCRIPT_DIR}/src"
CUROBO_SRC_DIR="/data/zhenyangfan/RoboTwin/envs/curobo/src"

source /data/miniconda3/etc/profile.d/conda.sh
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-smolvla}"
conda activate "${EVAL_CONDA_ENV}"
cd "${SCRIPT_DIR}"

# Edit the values in this block directly before launching the script.
POLICY_NAME="${POLICY_NAME:-SmolVLA}"
TASK_NAME="${TASK_NAME:-put_bottles_dustbin}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
# CKPT_SETTING="stage1_spatial_projector_step_001250_base"
CKPT_SETTING="${CKPT_SETTING:-evac_only_correction_ratio025_1500}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-7}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-seen}"
# MODEL_PATH="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high/20260420_130740-stage1_spatial_projector_0567/stage1_step_001250.pt"
MODEL_PATH="${MODEL_PATH:-/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high/20260428_235223-stage1_spatial_projector_no_aux_losses/stage1_step_001500.pt}"
# EVAL_TAG="stage1_spatial_projector_step_001250_base"
EVAL_TAG="${EVAL_TAG:-evac_only_correction_ratio025_1500}"
INFERENCE_MODE="${INFERENCE_MODE:-pred}"
SEED_FILE="${SEED_FILE:-}"
START_SEED="${START_SEED:-}"
INITIAL_SUCCESS_COUNT="${INITIAL_SUCCESS_COUNT:-}"
INITIAL_TEST_COUNT="${INITIAL_TEST_COUNT:-}"
EXPERT_CHECK="${EXPERT_CHECK:-}"
POLICY_CONDA_ENV="${POLICY_CONDA_ENV:-}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-}"
PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false

EFFECTIVE_PYTHONPATH="${CUROBO_SRC_DIR}:${LOCAL_SRC_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE}"
export PYTHONPATH="${EFFECTIVE_PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -t 1 ]]; then
  YELLOW=$'\033[33m'
  RESET=$'\033[0m'
else
  YELLOW=""
  RESET=""
fi

resolve_model_path() {
  local requested_path="$1"
  local candidate_path=""

  if [[ -d "${requested_path}" ]]; then
    printf '%s\n' "${requested_path}"
    return 0
  fi

  if [[ "${requested_path}" =~ ^(.*/checkpoints)/([0-9]{6})/pretrained_model$ ]]; then
    candidate_path="${BASH_REMATCH[1]}/checkpoints/${BASH_REMATCH[2]}/pretrained_model"
    if [[ -d "${candidate_path}" ]]; then
      printf '%s\n' "${candidate_path}"
      return 0
    fi
  fi

  if [[ -f "${requested_path}" && "${requested_path}" == *.pt ]]; then
    printf '%s\n' "${requested_path}"
    return 0
  fi

  return 1
}

if ! RESOLVED_MODEL_PATH=$(resolve_model_path "${MODEL_PATH}"); then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  echo "Expected a local pretrained_model directory or a stage1/stage2 .pt checkpoint." >&2
  exit 1
fi

if [[ "${RESOLVED_MODEL_PATH}" != "${MODEL_PATH}" ]]; then
  echo -e "${YELLOW}resolved model path: ${RESOLVED_MODEL_PATH}${RESET}"
fi

MODEL_PATH="${RESOLVED_MODEL_PATH}"

echo -e "${YELLOW}task: ${TASK_NAME}${RESET}"
echo -e "${YELLOW}task config: ${TASK_CONFIG}${RESET}"
echo -e "${YELLOW}gpu id: ${GPU_ID}${RESET}"
echo -e "${YELLOW}ckpt setting: ${CKPT_SETTING}${RESET}"
echo -e "${YELLOW}model path: ${MODEL_PATH}${RESET}"
echo -e "${YELLOW}instruction type: ${INSTRUCTION_TYPE}${RESET}"
echo -e "${YELLOW}eval tag: ${EVAL_TAG}${RESET}"
echo -e "${YELLOW}inference mode: ${INFERENCE_MODE}${RESET}"

cd "${ROOT_DIR}"

cmd=(
  python script/eval_policy.py
  --config "policy/${POLICY_NAME}/deploy_policy.yml"
  --overrides
  --task_name "${TASK_NAME}"
  --task_config "${TASK_CONFIG}"
  --ckpt_setting "${CKPT_SETTING}"
  --seed "${SEED}"
  --instruction_type "${INSTRUCTION_TYPE}"
  --model_path "${MODEL_PATH}"
  --policy_name "${POLICY_NAME}"
  --eval_tag "${EVAL_TAG}"
  --inference_mode "${INFERENCE_MODE}"
)

if [[ -n "${EVAL_VIDEO_LOG}" ]]; then
  cmd+=(--eval_video_log "${EVAL_VIDEO_LOG}")
fi

if [[ -n "${SEED_FILE}" ]]; then
  cmd+=(--seed_file "${SEED_FILE}")
fi

if [[ -n "${START_SEED}" ]]; then
  cmd+=(--start_seed "${START_SEED}")
fi

if [[ -n "${INITIAL_SUCCESS_COUNT}" ]]; then
  cmd+=(--initial_success_count "${INITIAL_SUCCESS_COUNT}")
fi

if [[ -n "${INITIAL_TEST_COUNT}" ]]; then
  cmd+=(--initial_test_count "${INITIAL_TEST_COUNT}")
fi

if [[ -n "${EXPERT_CHECK}" ]]; then
  cmd+=(--expert_check "${EXPERT_CHECK}")
fi

if [[ -n "${POLICY_CONDA_ENV}" ]]; then
  cmd+=(--policy_conda_env "${POLICY_CONDA_ENV}")
fi

PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
