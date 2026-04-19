#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
LOCAL_SRC_DIR="${SCRIPT_DIR}/src"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate smolvla
cd "${SCRIPT_DIR}"

# Edit the values in this block directly before launching the script.
DATASET_REPO_ID="robotwin_multitask_5_cam_high"
DATASET_ROOT="${SCRIPT_DIR}/data/${DATASET_REPO_ID}"
OUTPUT_ROOT="${SCRIPT_DIR}/outputs/train/${DATASET_REPO_ID}"
TRAIN_TAG="rgb_seen_random"
RUN_TAG="rgb_seen_random"

PRETRAINED_PATH="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/055000/pretrained_model"
RESUME_FROM="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/050000"
CUDA_VISIBLE_DEVICES="3"
POLICY_DEVICE="cuda"
FREEZE_VISION_ENCODER="false"
TRAIN_EXPERT_ONLY="false"
LOAD_VLM_WEIGHTS="true"

STEPS=80000
BATCH_SIZE=16
NUM_WORKERS=8
SAVE_FREQ=2500
LOG_FREQ=50
EVAL_FREQ=0
WANDB_ENABLE="false"
RANDOMIZE_SEEN_INSTRUCTIONS=1
OPTIMIZER_LR="5e-5"
WEIGHT_DECAY="1e-10"
SCHEDULER_WARMUP_STEPS=2000
SCHEDULER_DECAY_STEPS=50000
SCHEDULER_DECAY_LR="1e-5"
VIDEO_BACKEND="pyav"
PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false
PYTHONPATH="${LOCAL_SRC_DIR}"

if [ ! -d "${DATASET_ROOT}" ]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  echo "CUDA_VISIBLE_DEVICES is empty." >&2
  exit 1
fi

if [ -z "${PRETRAINED_PATH}" ]; then
  echo "PRETRAINED_PATH is required." >&2
  exit 1
fi

resolve_resume_run_dir() {
  local resume_path="$1"
  local latest_checkpoint=""
  local checkpoint_dir=""

  if [ ! -e "${resume_path}" ]; then
    echo "Resume path does not exist: ${resume_path}" >&2
    exit 1
  fi

  resume_path=$(readlink -f "${resume_path}")

  if [ -d "${resume_path}/training_state" ] && [ -d "${resume_path}/pretrained_model" ]; then
    checkpoint_dir="${resume_path}"
  elif [[ "$(basename "${resume_path}")" == checkpoint-* ]]; then
    if [ ! -d "${resume_path}/training_state" ] && [ ! -d "${resume_path}/pretrained_model" ]; then
      echo "Checkpoint directory does not look valid: ${resume_path}" >&2
      exit 1
    fi
    checkpoint_dir="${resume_path}"
  else
    latest_checkpoint=$(find "${resume_path}" -maxdepth 1 -mindepth 1 -type d \( -name 'checkpoint-*' -o -regex '.*/[0-9]+' \) | sort -V | tail -n 1)
    if [ -z "${latest_checkpoint}" ]; then
      echo "No checkpoint directory found under ${resume_path}" >&2
      exit 1
    fi

    echo "${resume_path}"
    return
  fi

  if [ "$(basename "$(dirname "${checkpoint_dir}")")" = "checkpoints" ]; then
    dirname "$(dirname "${checkpoint_dir}")"
  else
    dirname "${checkpoint_dir}"
  fi
}

resolve_resume_checkpoint_dir() {
  local resume_path="$1"
  local latest_checkpoint=""

  if [ ! -e "${resume_path}" ]; then
    echo "Resume path does not exist: ${resume_path}" >&2
    exit 1
  fi

  resume_path=$(readlink -f "${resume_path}")

  if [ -d "${resume_path}/training_state" ] && [ -d "${resume_path}/pretrained_model" ]; then
    echo "${resume_path}"
    return
  fi

  if [[ "$(basename "${resume_path}")" == checkpoint-* ]]; then
    if [ ! -d "${resume_path}/training_state" ] && [ ! -d "${resume_path}/pretrained_model" ]; then
      echo "Checkpoint directory does not look valid: ${resume_path}" >&2
      exit 1
    fi
    echo "${resume_path}"
    return
  fi

  latest_checkpoint=$(find "${resume_path}" -maxdepth 1 -mindepth 1 -type d \( -name 'checkpoint-*' -o -regex '.*/[0-9]+' \) | sort -V | tail -n 1)
  if [ -z "${latest_checkpoint}" ]; then
    echo "No checkpoint directory found under ${resume_path}" >&2
    exit 1
  fi

  echo "${latest_checkpoint}"
}

build_run_dir() {
  local root_dir="$1"
  local timestamp
  local base_name
  local target_dir
  timestamp=$(date +"%Y%m%d_%H%M%S")
  base_name="${timestamp}"
  if [ -n "${RUN_TAG}" ]; then
    base_name="${base_name}-${RUN_TAG}"
  fi
  target_dir="${root_dir}/${base_name}"
  if [ -e "${target_dir}" ]; then
    local suffix=1
    while [ -e "${root_dir}/${base_name}-${suffix}" ]; do
      suffix=$((suffix + 1))
    done
    target_dir="${root_dir}/${base_name}-${suffix}"
  fi
  echo "${target_dir}"
}

mkdir -p "${OUTPUT_ROOT}"

ARTIFACT_DIR=""
OUTPUT_DIR=""
RESUME_CHECKPOINT_DIR=""
if [ -n "${RESUME_FROM}" ]; then
  RESUME_CHECKPOINT_DIR=$(resolve_resume_checkpoint_dir "${RESUME_FROM}")
  OUTPUT_DIR=$(resolve_resume_run_dir "${RESUME_FROM}")
  RESUME_CHECKPOINT_DIR=$(readlink -f "${RESUME_CHECKPOINT_DIR}")
  OUTPUT_DIR=$(readlink -f "${OUTPUT_DIR}")
  ARTIFACT_DIR="${OUTPUT_DIR}"
  echo "Resuming existing run: ${OUTPUT_DIR}"
  echo "Resume checkpoint: ${RESUME_CHECKPOINT_DIR}"
else
  OUTPUT_DIR=$(build_run_dir "${OUTPUT_ROOT}")
  ARTIFACT_DIR="${OUTPUT_DIR}.pending"
  rm -rf "${ARTIFACT_DIR}"
  mkdir -p "${ARTIFACT_DIR}"
fi

cp "${SCRIPT_PATH}" "${ARTIFACT_DIR}/launch_train_robotwin_multitask.sh"

cat > "${ARTIFACT_DIR}/run_meta.txt" <<EOF
script=${SCRIPT_PATH}
script_dir=${SCRIPT_DIR}
train_mode=finetune
dataset_repo_id=${DATASET_REPO_ID}
dataset_root=${DATASET_ROOT}
pretrained_path=${PRETRAINED_PATH}
policy_device=${POLICY_DEVICE}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
freeze_vision_encoder=${FREEZE_VISION_ENCODER}
train_expert_only=${TRAIN_EXPERT_ONLY}
load_vlm_weights=${LOAD_VLM_WEIGHTS}
steps=${STEPS}
batch_size=${BATCH_SIZE}
num_workers=${NUM_WORKERS}
save_freq=${SAVE_FREQ}
log_freq=${LOG_FREQ}
eval_freq=${EVAL_FREQ}
wandb_enable=${WANDB_ENABLE}
randomize_seen_instructions=${RANDOMIZE_SEEN_INSTRUCTIONS}
optimizer_lr=${OPTIMIZER_LR}
weight_decay=${WEIGHT_DECAY}
scheduler_warmup_steps=${SCHEDULER_WARMUP_STEPS}
scheduler_decay_steps=${SCHEDULER_DECAY_STEPS}
scheduler_decay_lr=${SCHEDULER_DECAY_LR}
video_backend=${VIDEO_BACKEND}
run_tag=${RUN_TAG}
train_tag=${TRAIN_TAG}
output_dir=${OUTPUT_DIR}
resume_from=${RESUME_FROM}
pythonnousersite=${PYTHONNOUSERSITE}
pythonpath=${PYTHONPATH}
EOF

CMD=(
  python3
  "${SCRIPT_DIR}/src/lerobot/scripts/lerobot_train.py"
  --policy.type=smolvla
  --policy.pretrained_path="${PRETRAINED_PATH}"
  --policy.load_vlm_weights="${LOAD_VLM_WEIGHTS}"
  --policy.device="${POLICY_DEVICE}"
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}"
  --policy.train_expert_only="${TRAIN_EXPERT_ONLY}"
  --policy.optimizer_lr="${OPTIMIZER_LR}"
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --output_dir="${OUTPUT_DIR}"
  --steps="${STEPS}"
  --batch_size="${BATCH_SIZE}"
  --eval_freq="${EVAL_FREQ}"
  --wandb.enable="${WANDB_ENABLE}"
  --policy.push_to_hub=false
  --log_freq="${LOG_FREQ}"
  --save_freq="${SAVE_FREQ}"
  --num_workers="${NUM_WORKERS}"
)

if [ -n "${RESUME_FROM}" ]; then
  CMD+=(--resume=true --config_path="${RESUME_CHECKPOINT_DIR}/pretrained_model/train_config.json")
fi

printf '%q ' "${CMD[@]}" > "${ARTIFACT_DIR}/launch_command.sh"
printf '\n' >> "${ARTIFACT_DIR}/launch_command.sh"

echo "Training mode: finetune"
echo "Training output dir: ${OUTPUT_DIR}"
echo "Train tag: ${TRAIN_TAG}"
echo "Run tag: ${RUN_TAG}"

export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE
export PYTHONPATH
export TOKENIZERS_PARALLELISM
export LEROBOT_RANDOMIZE_TASK_FROM_EPISODE_INSTRUCTIONS="${RANDOMIZE_SEEN_INSTRUCTIONS}"

"${CMD[@]}" 2>&1 | tee "${ARTIFACT_DIR}/log.log"

if [ "${ARTIFACT_DIR}" != "${OUTPUT_DIR}" ] && [ -d "${OUTPUT_DIR}" ]; then
  cp "${ARTIFACT_DIR}/launch_train_robotwin_multitask.sh" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/run_meta.txt" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/launch_command.sh" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/log.log" "${OUTPUT_DIR}/"
  rm -rf "${ARTIFACT_DIR}"
fi

echo "${OUTPUT_DIR}"
