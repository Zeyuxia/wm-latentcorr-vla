#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/data/zhenyangfan/RoboTwin
SCRIPT_DIR="${ROOT_DIR}/policy/SmolVLA"
DATASET_REPO_ID=robotwin_multitask_5_cam_high
DATASET_ROOT="${SCRIPT_DIR}/data/${DATASET_REPO_ID}"
TRAIN_ROOT="${SCRIPT_DIR}/outputs/train/${DATASET_REPO_ID}"
RESUME_CHECKPOINT="${TRAIN_ROOT}/20260419_141110-rgb_seen_random/checkpoints/055000"
TRAIN_RUN_TAG=clean_optimizer_resume055000_cards3210_steps1500
EVAL_TAG=clean_optimizer_resume055000_cards3210_step56500
CKPT_SETTING="${EVAL_TAG}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29612}"
LOG_DIR="${SCRIPT_DIR}/outputs/eval_logs/${EVAL_TAG}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

timestamp=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${TRAIN_ROOT}/${timestamp}-${TRAIN_RUN_TAG}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

cat > "${RUN_DIR}/run_meta.txt" <<EOF
script=${BASH_SOURCE[0]}
train_mode=optimizer_resume_clean_finetune
dataset_repo_id=${DATASET_REPO_ID}
dataset_root=${DATASET_ROOT}
resume_checkpoint=${RESUME_CHECKPOINT}
resume_config=${RESUME_CHECKPOINT}/pretrained_model/train_config.json
resume_pretrained_model=${RESUME_CHECKPOINT}/pretrained_model
resume_training_step=55000
target_training_step=56500
cuda_visible_devices=3,2,1,0
main_process_port=${MAIN_PROCESS_PORT}
batch_size_per_process=4
effective_batch_size=16
num_workers=8
save_freq=1500
eval_tag=${EVAL_TAG}
output_dir=${RUN_DIR}
EOF

cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES=3,2,1,0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SCRIPT_DIR}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_RANDOMIZE_TASK_FROM_EPISODE_INSTRUCTIONS=1
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/zhenyangfan/runtime_cache}"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/torch_extensions" "${RUNTIME_ROOT}/mplconfig"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_ROOT}/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_ROOT}/mplconfig}"

cmd=(
  accelerate
  launch
  --multi_gpu
  --num_processes=4
  --main_process_port="${MAIN_PROCESS_PORT}"
  "${SCRIPT_DIR}/src/lerobot/scripts/lerobot_train.py"
  --config_path="${RESUME_CHECKPOINT}/pretrained_model/train_config.json"
  --resume=true
  --output_dir="${RUN_DIR}"
  --steps=56500
  --batch_size=4
  --num_workers=8
  --save_freq=1500
  --log_freq=50
  --eval_freq=0
  --wandb.enable=false
  --policy.push_to_hub=false
  --dataset.root="${DATASET_ROOT}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --policy.pretrained_path="${RESUME_CHECKPOINT}/pretrained_model"
  --policy.device=cuda
)

printf '%q ' "${cmd[@]}" > "${RUN_DIR}/launch_command.sh"
printf '\n' >> "${RUN_DIR}/launch_command.sh"

"${cmd[@]}" 2>&1 | tee "${RUN_DIR}/log.log"

MODEL_PATH="${RUN_DIR}/checkpoints/056500/pretrained_model"
if [ ! -d "${MODEL_PATH}" ]; then
  echo "Missing expected checkpoint directory: ${MODEL_PATH}" >&2
  exit 1
fi

sleep 30

GPU_ID=3 TASK_NAME=handover_block CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/handover_block_g3.log" 2>&1 &
GPU_ID=2 TASK_NAME=open_laptop CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/open_laptop_g2.log" 2>&1 &
GPU_ID=1 TASK_NAME=pick_dual_bottles CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/pick_dual_bottles_g1.log" 2>&1 &
GPU_ID=0 TASK_NAME=place_burger_fries CKPT_SETTING="${CKPT_SETTING}" EVAL_TAG="${EVAL_TAG}" MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/eval.sh" > "${LOG_DIR}/place_burger_fries_g0.log" 2>&1 &

wait
