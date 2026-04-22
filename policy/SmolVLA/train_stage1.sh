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
TRAIN_TAG="stage1_no_object_contact_correction"
RUN_TAG="stage1_spatial_projector_only_condition"

PRETRAINED_PATH="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/055000/pretrained_model"
RESUME_FROM=""
CUDA_VISIBLE_DEVICES="0,5,6,7"
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
URDF_PATH="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
CUROBO_LEFT_YML="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml"
CUROBO_RIGHT_YML="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml"

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
DYN_RAMP_STEPS=0
DYN_MAX_WEIGHT=0.0
DYN_WARMUP_CURVE="cosine"
COND_ZERO_STEPS=0
COND_RAMP_STEPS=1000
COND_MAX_WEIGHT=0.5
COND_WARMUP_CURVE="cosine"
FAILURE_MODE="train"
FAILURE_TABLE_PATHS_JSON="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/explore/20260422_023320-explore_stage1_spatial_projector_accelerate_56/merged/multitask_failure_manifest.json"
FAILURE_CORR_BATCH_RATIO=0.5
SAMPLE_PHASE_WINDOW_LEN=20
START_MARGIN=0
FAILURE_PHASE_BINS=4
FAILURE_TRANSLATION_DIR_BINS=6
FAILURE_TRANSLATION_MAG_BINS=1
FAILURE_ROTATION_DIR_BINS=6
FAILURE_ROTATION_MAG_BINS=1
FAILURE_EXPLORE_K=4
ACT_ALIGNED_ROLLOUT_EXEC_STEPS=16
PLANNER_ORIENT_WEIGHT=0.0573
PLANNER_GRIPPER_PENALTY=1.0
PLANNER_NEAREST_WINDOW_RADIUS=12
PLANNER_ACTIVE_JOINT_DELTA_THRESH=0.01
PLANNER_ACTIVE_GRIPPER_DELTA_THRESH=0.05
RECOVER_EVAL_SAVE_VIDEO=false
RECOVER_EVAL_GRIPPER_OPEN_THRESH=0.3
RECOVER_EVAL_POS_THRESH_M=0.04
RECOVER_EVAL_ROT_THRESH_DEG=8.0
RECOVER_EVAL_NEAREST_WINDOW_RADIUS=16
RECOVER_EVAL_VIDEO_BRIDGE_STEPS=16
ACT_ALIGNED_ENABLE_PERTURB=true
ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN=0.03
ACT_ALIGNED_PERTURB_ROT_MAX_DEG=10.0
ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN=0.10
DEBUG_WM_CORRECTION=true
DEBUG_WM_ALL_RANKS=true
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
failure_mode=${FAILURE_MODE}
failure_table_paths_json=${FAILURE_TABLE_PATHS_JSON}
failure_corr_batch_ratio=${FAILURE_CORR_BATCH_RATIO}
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
  --cond_zero_steps "${COND_ZERO_STEPS}"
  --cond_ramp_steps "${COND_RAMP_STEPS}"
  --cond_max_weight "${COND_MAX_WEIGHT}"
  --cond_warmup_curve "${COND_WARMUP_CURVE}"
  --failure_mode "${FAILURE_MODE}"
  --failure_table_paths_json "${FAILURE_TABLE_PATHS_JSON}"
  --failure_corr_batch_ratio "${FAILURE_CORR_BATCH_RATIO}"
  --failure_phase_bins "${FAILURE_PHASE_BINS}"
  --failure_translation_dir_bins "${FAILURE_TRANSLATION_DIR_BINS}"
  --failure_translation_mag_bins "${FAILURE_TRANSLATION_MAG_BINS}"
  --failure_rotation_dir_bins "${FAILURE_ROTATION_DIR_BINS}"
  --failure_rotation_mag_bins "${FAILURE_ROTATION_MAG_BINS}"
  --failure_explore_k "${FAILURE_EXPLORE_K}"
  --sample_phase_window_len "${SAMPLE_PHASE_WINDOW_LEN}"
  --start_margin "${START_MARGIN}"
  --urdf_path "${URDF_PATH}"
  --curobo_left_yml "${CUROBO_LEFT_YML}"
  --curobo_right_yml "${CUROBO_RIGHT_YML}"
  --act_aligned_rollout_exec_steps "${ACT_ALIGNED_ROLLOUT_EXEC_STEPS}"
  --planner_orient_weight "${PLANNER_ORIENT_WEIGHT}"
  --planner_gripper_penalty "${PLANNER_GRIPPER_PENALTY}"
  --planner_nearest_window_radius "${PLANNER_NEAREST_WINDOW_RADIUS}"
  --planner_active_joint_delta_thresh "${PLANNER_ACTIVE_JOINT_DELTA_THRESH}"
  --planner_active_gripper_delta_thresh "${PLANNER_ACTIVE_GRIPPER_DELTA_THRESH}"
  --recover_eval_save_video "${RECOVER_EVAL_SAVE_VIDEO}"
  --recover_eval_gripper_open_thresh "${RECOVER_EVAL_GRIPPER_OPEN_THRESH}"
  --recover_eval_pos_thresh_m "${RECOVER_EVAL_POS_THRESH_M}"
  --recover_eval_rot_thresh_deg "${RECOVER_EVAL_ROT_THRESH_DEG}"
  --recover_eval_nearest_window_radius "${RECOVER_EVAL_NEAREST_WINDOW_RADIUS}"
  --recover_eval_video_bridge_steps "${RECOVER_EVAL_VIDEO_BRIDGE_STEPS}"
  --act_aligned_enable_perturb "${ACT_ALIGNED_ENABLE_PERTURB}"
  --act_aligned_perturb_eef_fail_gain "${ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN}"
  --act_aligned_perturb_rot_max_deg "${ACT_ALIGNED_PERTURB_ROT_MAX_DEG}"
  --act_aligned_perturb_gripper_close_min "${ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN}"
  --debug_wm_correction "${DEBUG_WM_CORRECTION}"
  --debug_wm_all_ranks "${DEBUG_WM_ALL_RANKS}"
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
