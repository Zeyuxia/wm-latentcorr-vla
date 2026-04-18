#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

policy_name=TinyVLA
task_name=${1:-task_you_test}
task_config=${2:-demo_clean}
ckpt_setting=${3:-0}
expert_data_num=${4:-50}
seed=${5:-0}
gpu_id=${6:-0}

DEFAULT_RUN_ROOT="${SCRIPT_DIR}/unet_diffusion_policy_results/robotwin_multitask_5_cam_high/20260416_235212-base_multi_task"
run_root=${RUN_ROOT:-${DEFAULT_RUN_ROOT}}
model_base=${MODEL_BASE:-${SCRIPT_DIR}/model_param/InternVL3-1B}
state_path=${STATE_PATH:-${run_root}/dataset_stats.pkl}
instruction_type=${INSTRUCTION_TYPE:-unseen}

if [[ -n "${MODEL_PATH:-}" ]]; then
    model_path="${MODEL_PATH}"
elif [[ "${ckpt_setting}" =~ ^[0-9]+$ ]] && [[ -d "${run_root}/checkpoint-${ckpt_setting}" ]]; then
    model_path="${run_root}/checkpoint-${ckpt_setting}"
else
    model_path="${run_root}"
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTHONNOUSERSITE=1
if [[ -t 1 ]]; then
    YELLOW=$'\033[33m'
    RESET=$'\033[0m'
else
    YELLOW=""
    RESET=""
fi
echo -e "${YELLOW}gpu id (to use): ${gpu_id}${RESET}"
echo -e "${YELLOW}model path: ${model_path}${RESET}"

cd "${ROOT_DIR}"

cmd=(
    python script/eval_policy.py
    --config "policy/${policy_name}/deploy_policy.yml"
    --overrides
    --task_name "${task_name}"
    --task_config "${task_config}"
    --ckpt_setting "${ckpt_setting}"
    --expert_data_num "${expert_data_num}"
    --seed "${seed}"
    --instruction_type "${instruction_type}"
    --model_base "${model_base}"
    --state_path "${state_path}"
    --model_path "${model_path}"
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
