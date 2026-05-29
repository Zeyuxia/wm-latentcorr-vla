#!/bin/bash
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda activate ACT

PYTHON_BIN=${PYTHON_BIN:-python}
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/zhenyangfan/runtime_cache}
mkdir -p \
    "${RUNTIME_ROOT}/tmp" \
    "${RUNTIME_ROOT}/torch_extensions" \
    "${RUNTIME_ROOT}/mplconfig" \
    "${RUNTIME_ROOT}/hf_home" \
    "${RUNTIME_ROOT}/wandb" \
    "${RUNTIME_ROOT}/xdg_cache" \
    "${RUNTIME_ROOT}/xdg_config"

policy_name=ACT_LatentCorr
task_name=${TASK_NAME:-open_laptop}
task_config=${TASK_CONFIG:-demo_clean}
ckpt_setting=${CKPT_SETTING:-latentcorr_current}
latent_ckpt_path=${LATENT_CKPT_PATH:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_fulltable_lctok05_gpu0123_20260511_163455/stage1_unified_epoch_0200.pt}
inference_mode=${INFERENCE_MODE:-pred}
seed=${SEED:-0}
seed_file=${SEED_FILE:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_small.txt}
gpu_id=${GPU_ID:-0}
eval_video_log=${EVAL_VIDEO_LOG:-false}
disable_planner=${DISABLE_PLANNER:-auto}
expert_check=${EXPERT_CHECK:-auto}

if [[ "${seed_file}" != /* ]]; then
    seed_file="$(cd "$(dirname "${seed_file}")" && pwd)/$(basename "${seed_file}")"
fi

evac_ckpt=${EVAC_CKPT:-/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt}
evac_config=${EVAC_CONFIG:-/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml}
urdf_path=${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export PYTHONPATH="/data/zhenyangfan/RoboTwin:/data/zhenyangfan/RoboTwin/envs/curobo/src:/data/zhenyangfan/RoboTwin/envs/robot:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_ROOT}/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_ROOT}/mplconfig}"
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${RUNTIME_ROOT}/wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${RUNTIME_ROOT}/wandb/config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${RUNTIME_ROOT}/xdg_config}"
echo -e "\033[33m[gpu] CUDA_VISIBLE_DEVICES=${gpu_id}\033[0m"
echo -e "\033[33m[mode] ${inference_mode}\033[0m"
echo -e "\033[33m[ckpt] ${latent_ckpt_path}\033[0m"

if [ "${disable_planner}" = "auto" ]; then
    if [ "${inference_mode}" = "pred" ] || [ "${inference_mode}" = "base" ]; then
        disable_planner=true
    else
        disable_planner=false
    fi
fi

if [ "${expert_check}" = "auto" ]; then
    if [ "${inference_mode}" = "pred" ] || [ "${inference_mode}" = "base" ]; then
        expert_check=false
    else
        expert_check=true
    fi
fi

cd /data/zhenyangfan/RoboTwin

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
"${PYTHON_BIN}" -u script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --latent_ckpt_path ${latent_ckpt_path} \
    --seed ${seed} \
    --seed_file ${seed_file} \
    --device cuda:0 \
    --inference_mode ${inference_mode} \
    --eval_video_log ${eval_video_log} \
    --expert_check ${expert_check} \
    --disable_planner ${disable_planner} \
    --evac_ckpt ${evac_ckpt} \
    --evac_config ${evac_config} \
    --urdf_path ${urdf_path}
