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
POLICY_NAME="SmolVLA.deploy_policy_bridge"
TASK_NAME="put_bottles_dustbin"
TASK_CONFIG="demo_clean"
CKPT_SETTING="bridge"
SEED=0
GPU_ID=2
INSTRUCTION_TYPE="seen"
MODEL_PATH="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/055000/pretrained_model"
BRIDGE_CKPT="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high/20260420_005743-stage1_1345/stage1_step_000250.pt"
CONDITION_SCALE="1.0"
EVAL_TAG="bridge_seen"
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

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [ ! -f "${BRIDGE_CKPT}" ]; then
  echo "Bridge checkpoint does not exist: ${BRIDGE_CKPT}" >&2
  exit 1
fi

if [[ -t 1 ]]; then
  YELLOW=$'\033[33m'
  RESET=$'\033[0m'
else
  YELLOW=""
  RESET=""
fi

echo -e "${YELLOW}task: ${TASK_NAME}${RESET}"
echo -e "${YELLOW}task config: ${TASK_CONFIG}${RESET}"
echo -e "${YELLOW}gpu id: ${GPU_ID}${RESET}"
echo -e "${YELLOW}ckpt setting: ${CKPT_SETTING}${RESET}"
echo -e "${YELLOW}model path: ${MODEL_PATH}${RESET}"
echo -e "${YELLOW}bridge ckpt: ${BRIDGE_CKPT}${RESET}"
echo -e "${YELLOW}instruction type: ${INSTRUCTION_TYPE}${RESET}"
echo -e "${YELLOW}condition scale: ${CONDITION_SCALE}${RESET}"
echo -e "${YELLOW}eval tag: ${EVAL_TAG}${RESET}"

cd "${ROOT_DIR}"

cmd=(
  python script/eval_policy.py
  --config "policy/SmolVLA/deploy_policy_bridge.yml"
  --overrides
  --task_name "${TASK_NAME}"
  --task_config "${TASK_CONFIG}"
  --ckpt_setting "${CKPT_SETTING}"
  --seed "${SEED}"
  --instruction_type "${INSTRUCTION_TYPE}"
  --model_path "${MODEL_PATH}"
  --bridge_ckpt "${BRIDGE_CKPT}"
  --condition_scale "${CONDITION_SCALE}"
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
