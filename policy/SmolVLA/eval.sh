#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_SRC_DIR=${SCRIPT_DIR}/src

policy_name=SmolVLA
task_name=${1:-open_laptop}
task_config=${2:-demo_clean}
ckpt_setting=${3:-default}
seed=${4:-0}
gpu_id=${5:-6}

DEFAULT_MODEL_PATH=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260417_173023-base_cam_high/checkpoints/010000/pretrained_model
RUN_ROOT=${RUN_ROOT:-${SCRIPT_DIR}/outputs/train/robotwin_multitask_5_cam_high/latest}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-seen}
PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
PYTHONPATH=${PYTHONPATH:-}
MODEL_PATH_OVERRIDE=${MODEL_PATH:-}
CUROBO_SRC_DIR=/data/zhenyangfan/RoboTwin/envs/curobo/src

if [ -n "${PYTHONPATH}" ]; then
  EFFECTIVE_PYTHONPATH=${CUROBO_SRC_DIR}:${LOCAL_SRC_DIR}:${PYTHONPATH}
else
  EFFECTIVE_PYTHONPATH=${CUROBO_SRC_DIR}:${LOCAL_SRC_DIR}
fi

resolve_model_path() {
  local run_root="$1"
  local setting="$2"
  local latest_checkpoint=""
  local padded_setting=""
  local numeric_setting=""
  local candidates=()

  if [[ -n "${MODEL_PATH_OVERRIDE}" ]]; then
    echo "${MODEL_PATH_OVERRIDE}"
    return
  fi

  if [[ "${setting}" =~ ^ckpt([0-9]+)$ ]]; then
    numeric_setting="${BASH_REMATCH[1]}"
  elif [[ "${setting}" =~ ^[0-9]+$ ]]; then
    numeric_setting="${setting}"
  fi

  if [[ -n "${numeric_setting}" ]]; then
    padded_setting=$(printf "%06d" "${numeric_setting}")
    candidates+=(
      "${run_root}/checkpoints/${padded_setting}/pretrained_model"
      "${run_root}/checkpoints/${numeric_setting}/pretrained_model"
      "${run_root}/checkpoint-${numeric_setting}/pretrained_model"
      "${run_root}/checkpoint-${numeric_setting}"
    )
  fi

  candidates+=(
    "${run_root}/pretrained_model"
    "${DEFAULT_MODEL_PATH}"
  )

  for candidate in "${candidates[@]}"; do
    if [ -e "${candidate}" ]; then
      echo "${candidate}"
      return
    fi
  done

  if [ -d "${run_root}/checkpoints" ]; then
    latest_checkpoint=$(find "${run_root}/checkpoints" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -n 1)
    if [ -n "${latest_checkpoint}" ]; then
      if [ -d "${latest_checkpoint}/pretrained_model" ]; then
        echo "${latest_checkpoint}/pretrained_model"
        return
      fi
      echo "${latest_checkpoint}"
      return
    fi
  fi

  echo "${run_root}"
}

MODEL_PATH_RESOLVED=$(resolve_model_path "${RUN_ROOT}" "${ckpt_setting}")

export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE}
export PYTHONPATH=${EFFECTIVE_PYTHONPATH}

if [[ -t 1 ]]; then
  YELLOW=$'\033[33m'
  RESET=$'\033[0m'
else
  YELLOW=""
  RESET=""
fi

echo -e "${YELLOW}gpu id (to use): ${gpu_id}${RESET}"
echo -e "${YELLOW}run root: ${RUN_ROOT}${RESET}"
if [[ -n "${MODEL_PATH_OVERRIDE}" ]]; then
  echo -e "${YELLOW}model path override: ${MODEL_PATH_OVERRIDE}${RESET}"
fi
echo -e "${YELLOW}model path: ${MODEL_PATH_RESOLVED}${RESET}"

cd "${ROOT_DIR}"

cmd=(
  python script/eval_policy.py
  --config "policy/${policy_name}/deploy_policy.yml"
  --overrides
  --task_name "${task_name}"
  --task_config "${task_config}"
  --ckpt_setting "${ckpt_setting}"
  --seed "${seed}"
  --instruction_type "${INSTRUCTION_TYPE}"
  --model_path "${MODEL_PATH_RESOLVED}"
  --policy_name "${policy_name}"
)

if [[ -n "${EVAL_TAG:-}" ]]; then
  cmd+=(--eval_tag "${EVAL_TAG}")
fi

if [[ -n "${SEED_FILE:-}" ]]; then
  cmd+=(--seed_file "${SEED_FILE}")
fi

if [[ -n "${POLICY_CONDA_ENV:-}" ]]; then
  cmd+=(--policy_conda_env "${POLICY_CONDA_ENV}")
fi

PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
