#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
cd "${SCRIPT_DIR}"
LOCAL_SRC_DIR=${SCRIPT_DIR}/src

DATASET_REPO_ID=${DATASET_REPO_ID:-robotwin_multitask_5_cam_high}
DATASET_ROOT=${DATASET_ROOT:-${SCRIPT_DIR}/data/${DATASET_REPO_ID}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/train/${DATASET_REPO_ID}}

TRAIN_TAG=${TRAIN_TAG:-}
RUN_TAG=${RUN_TAG:-${TRAIN_TAG:-base_cam_high}}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

PRETRAINED_PATH=${PRETRAINED_PATH:-/data/weights/smolvla_base}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES// /}
POLICY_DEVICE=${POLICY_DEVICE:-cuda}

STEPS=${STEPS:-50000}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-32}
SAVE_FREQ=${SAVE_FREQ:-2500}
LOG_FREQ=${LOG_FREQ:-50}
EVAL_FREQ=${EVAL_FREQ:-0}
WANDB_ENABLE=${WANDB_ENABLE:-false}
RANDOMIZE_SEEN_INSTRUCTIONS=${RANDOMIZE_SEEN_INSTRUCTIONS:-1}

RESUME_FROM=${RESUME_FROM:-}
PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
PYTHONPATH=${PYTHONPATH:-}

if [ -n "${PYTHONPATH}" ]; then
  EFFECTIVE_PYTHONPATH=${LOCAL_SRC_DIR}:${PYTHONPATH}
else
  EFFECTIVE_PYTHONPATH=${LOCAL_SRC_DIR}
fi

if [ ! -d "${DATASET_ROOT}" ]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  echo "CUDA_VISIBLE_DEVICES is empty." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

resolve_resume_run_dir() {
  local resume_path="$1"
  local latest_checkpoint=""

  if [ ! -e "${resume_path}" ]; then
    echo "Resume path does not exist: ${resume_path}" >&2
    exit 1
  fi

  resume_path=$(readlink -f "${resume_path}")

  if [[ "$(basename "${resume_path}")" == checkpoint-* ]]; then
    if [ ! -f "${resume_path}/training_state.safetensors" ] && [ ! -f "${resume_path}/optimizer_param_groups.json" ]; then
      echo "Checkpoint directory does not look valid: ${resume_path}" >&2
      exit 1
    fi
    dirname "${resume_path}"
    return
  fi

  latest_checkpoint=$(find "${resume_path}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
  if [ -z "${latest_checkpoint}" ]; then
    echo "No checkpoint-* found under ${resume_path}" >&2
    exit 1
  fi

  echo "${resume_path}"
}

if [ -n "${RESUME_FROM}" ]; then
  OUTPUT_DIR=$(resolve_resume_run_dir "${RESUME_FROM}")
  OUTPUT_DIR=$(readlink -f "${OUTPUT_DIR}")
  ARTIFACT_DIR="${OUTPUT_DIR}"
  echo "Resuming existing run: ${OUTPUT_DIR}"
else
  RUN_NAME="${RUN_TIMESTAMP}"
  if [ -n "${RUN_TAG}" ]; then
    RUN_NAME="${RUN_NAME}-${RUN_TAG}"
  fi
  OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}
  if [ -e "${OUTPUT_DIR}" ]; then
    echo "Output directory already exists: ${OUTPUT_DIR}" >&2
    exit 1
  fi
  ARTIFACT_DIR="${OUTPUT_DIR}.pending"
  rm -rf "${ARTIFACT_DIR}"
  mkdir -p "${ARTIFACT_DIR}"
fi

cp "${SCRIPT_PATH}" "${ARTIFACT_DIR}/launch_train_robotwin_multitask.sh"

cat > "${ARTIFACT_DIR}/run_meta.txt" <<EOF
script=${SCRIPT_PATH}
script_dir=${SCRIPT_DIR}
dataset_repo_id=${DATASET_REPO_ID}
dataset_root=${DATASET_ROOT}
pretrained_path=${PRETRAINED_PATH}
policy_device=${POLICY_DEVICE}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
steps=${STEPS}
batch_size=${BATCH_SIZE}
num_workers=${NUM_WORKERS}
save_freq=${SAVE_FREQ}
log_freq=${LOG_FREQ}
eval_freq=${EVAL_FREQ}
wandb_enable=${WANDB_ENABLE}
randomize_seen_instructions=${RANDOMIZE_SEEN_INSTRUCTIONS}
run_tag=${RUN_TAG}
train_tag=${TRAIN_TAG}
run_timestamp=${RUN_TIMESTAMP}
output_dir=${OUTPUT_DIR}
resume_from=${RESUME_FROM}
pythonnousersite=${PYTHONNOUSERSITE}
pythonpath=${EFFECTIVE_PYTHONPATH}
EOF

CMD=(
  python3
  "${SCRIPT_DIR}/src/lerobot/scripts/lerobot_train.py"
  --policy.type=smolvla
  --policy.pretrained_path="${PRETRAINED_PATH}"
  --policy.load_vlm_weights=true
  --policy.device="${POLICY_DEVICE}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.repo_id="${DATASET_REPO_ID}"
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
  CMD+=(--resume=true --config_path="${OUTPUT_DIR}/train_config.json")
fi

printf 'TRAIN_TAG=%q RUN_TAG=%q CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=%q PYTHONPATH=%q ' \
  "${TRAIN_TAG}" "${RUN_TAG}" "${CUDA_VISIBLE_DEVICES}" "${PYTHONNOUSERSITE}" "${EFFECTIVE_PYTHONPATH}" \
  > "${ARTIFACT_DIR}/launch_command.sh"
printf '%q ' "${CMD[@]}" >> "${ARTIFACT_DIR}/launch_command.sh"
printf '\n' >> "${ARTIFACT_DIR}/launch_command.sh"

echo "Training output dir: ${OUTPUT_DIR}"
echo "Train tag: ${TRAIN_TAG}"
echo "Run tag: ${RUN_TAG}"
echo "Checkpoint save frequency: every ${SAVE_FREQ} steps"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
LEROBOT_RANDOMIZE_TASK_FROM_EPISODE_INSTRUCTIONS="${RANDOMIZE_SEEN_INSTRUCTIONS}" \
PYTHONPATH="${EFFECTIVE_PYTHONPATH}" \
"${CMD[@]}" 2>&1 | tee "${ARTIFACT_DIR}/log.log"

if [ "${ARTIFACT_DIR}" != "${OUTPUT_DIR}" ] && [ -d "${OUTPUT_DIR}" ]; then
  cp "${ARTIFACT_DIR}/launch_train_robotwin_multitask.sh" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/run_meta.txt" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/launch_command.sh" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/log.log" "${OUTPUT_DIR}/"
  rm -rf "${ARTIFACT_DIR}"
fi

echo "${OUTPUT_DIR}"
