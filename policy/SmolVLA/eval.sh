#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_SRC_DIR="${SCRIPT_DIR}/src"
CUROBO_SRC_DIR="/data/zhenyangfan/RoboTwin/envs/curobo/src"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate smolvla
cd "${SCRIPT_DIR}"

# Edit the values in this block directly before launching the script.
POLICY_NAME="SmolVLA"
TASK_NAME="put_bottles_dustbin"
TASK_CONFIG="demo_clean"
CKPT_SETTING="stage1_spatial_projector_step_001250_base"
SEED=0
GPU_ID=7
INSTRUCTION_TYPE="seen"
MODEL_PATH="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high/20260420_130740-stage1_spatial_projector_0567/stage1_step_001250.pt"
EVAL_TAG="stage1_spatial_projector_step_001250_base"
SEED_FILE=""
POLICY_CONDA_ENV=""
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
)

if [[ -n "${SEED_FILE}" ]]; then
  cmd+=(--seed_file "${SEED_FILE}")
fi

if [[ -n "${POLICY_CONDA_ENV}" ]]; then
  cmd+=(--policy_conda_env "${POLICY_CONDA_ENV}")
fi

PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
