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
OUTPUT_ROOT="${SCRIPT_DIR}/outputs/stage1/${DATASET_REPO_ID}"
TRAIN_TAG="stage1"
RUN_TAG="stage1_1345"

PRETRAINED_PATH="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/055000/pretrained_model"
RESUME_FROM=""
CUDA_VISIBLE_DEVICES="1,3,4,5"
POLICY_DEVICE="cuda"
FREEZE_VISION_ENCODER="false"
TRAIN_EXPERT_ONLY="false"
LOAD_VLM_WEIGHTS="true"

MULTI_TASK_NAMES=(
  sim-open_laptop-demo_clean-50
  sim-pick_dual_bottles-demo_clean-50
  sim-put_bottles_dustbin-demo_clean-50
  sim-place_burger_fries-demo_clean-50
  sim-handover_block-demo_clean-50
)
INSTRUCTION_TYPE="seen"
EVAC_CKPT="/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt"
EVAC_CONFIG="${SCRIPT_DIR}/evac/configs/robotwin/train_config.yaml"

SEED=0
BATCH_SIZE=4
NUM_WORKERS=8
MAX_STEPS=1500
SAVE_FREQ=250
OPTIMIZER_LR="5e-5"
WEIGHT_DECAY="1e-10"
SCHEDULER_WARMUP_STEPS=200
SCHEDULER_DECAY_STEPS=50000
SCHEDULER_DECAY_LR="1e-5"
FUTURE_OFFSET=16
ACT_CHUNK_SIZE=50
PREFIX_STEPS=16
ACTION_DIM=14
LATENT_DIM=4
ADAPTER_HIDDEN_DIM=512
PREDICTOR_HIDDEN_DIM=512
DYN_ZERO_STEPS=0
DYN_RAMP_STEPS=1000
DYN_MAX_WEIGHT=0.5
DYN_WARMUP_CURVE="cosine"
PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false
PYTHONPATH="/data/zhenyangfan/RoboTwin:${LOCAL_SRC_DIR}"

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  echo "CUDA_VISIBLE_DEVICES is empty." >&2
  exit 1
fi

if [ -z "${PRETRAINED_PATH}" ]; then
  echo "PRETRAINED_PATH is required." >&2
  exit 1
fi

if [ -z "${EVAC_CKPT}" ]; then
  echo "EVAC_CKPT is required for stage1." >&2
  exit 1
fi

if [ -n "${RESUME_FROM}" ] && [ ! -f "${RESUME_FROM}" ]; then
  echo "Resume checkpoint does not exist: ${RESUME_FROM}" >&2
  exit 1
fi

if [ ! -f "${EVAC_CKPT}" ]; then
  echo "EVAC checkpoint path is missing: ${EVAC_CKPT}" >&2
  exit 1
fi

if [ ! -f "${EVAC_CONFIG}" ]; then
  echo "EVAC config path is missing: ${EVAC_CONFIG}" >&2
  exit 1
fi

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
if [ -n "${RESUME_FROM}" ]; then
  OUTPUT_DIR=$(dirname "$(readlink -f "${RESUME_FROM}")")
  echo "Resuming existing stage1 run: ${OUTPUT_DIR}"
  echo "Resume checkpoint: ${RESUME_FROM}"
else
  OUTPUT_DIR=$(build_run_dir "${OUTPUT_ROOT}")
  mkdir -p "${OUTPUT_DIR}"
fi

cp "${SCRIPT_PATH}" "${OUTPUT_DIR}/launch_train_stage1.sh"

cat > "${OUTPUT_DIR}/run_meta.txt" <<EOF
script=${SCRIPT_PATH}
script_dir=${SCRIPT_DIR}
train_mode=stage1
dataset_repo_id=${DATASET_REPO_ID}
pretrained_path=${PRETRAINED_PATH}
resume_from=${RESUME_FROM}
policy_device=${POLICY_DEVICE}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
freeze_vision_encoder=${FREEZE_VISION_ENCODER}
train_expert_only=${TRAIN_EXPERT_ONLY}
load_vlm_weights=${LOAD_VLM_WEIGHTS}
batch_size=${BATCH_SIZE}
num_workers=${NUM_WORKERS}
max_steps=${MAX_STEPS}
save_freq=${SAVE_FREQ}
optimizer_lr=${OPTIMIZER_LR}
weight_decay=${WEIGHT_DECAY}
scheduler_warmup_steps=${SCHEDULER_WARMUP_STEPS}
scheduler_decay_steps=${SCHEDULER_DECAY_STEPS}
scheduler_decay_lr=${SCHEDULER_DECAY_LR}
run_tag=${RUN_TAG}
train_tag=${TRAIN_TAG}
output_dir=${OUTPUT_DIR}
pythonnousersite=${PYTHONNOUSERSITE}
pythonpath=${PYTHONPATH}
instruction_type=${INSTRUCTION_TYPE}
evac_ckpt=${EVAC_CKPT}
evac_config=${EVAC_CONFIG}
seed=${SEED}
EOF

CMD=(
  accelerate
  launch
  --multi_gpu
  --num_processes=4
  --main_process_port=29611
  "${SCRIPT_DIR}/latentcorr/train_smolvla.py"
  stage1
  --output_dir "${OUTPUT_DIR}"
  --smolvla_pretrained_path "${PRETRAINED_PATH}"
  --resume_ckpt "${RESUME_FROM}"
  --freeze_vision_encoder "${FREEZE_VISION_ENCODER}"
  --train_expert_only "${TRAIN_EXPERT_ONLY}"
  --load_vlm_weights "${LOAD_VLM_WEIGHTS}"
  --multi_task_names "${MULTI_TASK_NAMES[@]}"
  --instruction_type "${INSTRUCTION_TYPE}"
  --evac_ckpt "${EVAC_CKPT}"
  --evac_config "${EVAC_CONFIG}"
  --device "${POLICY_DEVICE}:0"
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_steps "${MAX_STEPS}"
  --save_freq "${SAVE_FREQ}"
  --learning_rate "${OPTIMIZER_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --scheduler_warmup_steps "${SCHEDULER_WARMUP_STEPS}"
  --scheduler_decay_steps "${SCHEDULER_DECAY_STEPS}"
  --scheduler_decay_lr "${SCHEDULER_DECAY_LR}"
  --future_offset "${FUTURE_OFFSET}"
  --act_chunk_size "${ACT_CHUNK_SIZE}"
  --prefix_steps "${PREFIX_STEPS}"
  --action_dim "${ACTION_DIM}"
  --latent_dim "${LATENT_DIM}"
  --adapter_hidden_dim "${ADAPTER_HIDDEN_DIM}"
  --predictor_hidden_dim "${PREDICTOR_HIDDEN_DIM}"
  --dyn_zero_steps "${DYN_ZERO_STEPS}"
  --dyn_ramp_steps "${DYN_RAMP_STEPS}"
  --dyn_max_weight "${DYN_MAX_WEIGHT}"
  --dyn_warmup_curve "${DYN_WARMUP_CURVE}"
)

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/launch_command.sh"
printf '\n' >> "${OUTPUT_DIR}/launch_command.sh"

echo "Training mode: stage1"
echo "Training output dir: ${OUTPUT_DIR}"
echo "Train tag: ${TRAIN_TAG}"
echo "Run tag: ${RUN_TAG}"

export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE
export PYTHONPATH
export TOKENIZERS_PARALLELISM
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export MPLCONFIGDIR=/tmp/mplconfig

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/log.log"

echo "${OUTPUT_DIR}"
