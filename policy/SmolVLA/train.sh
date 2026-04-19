#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
LOCAL_SRC_DIR="${SCRIPT_DIR}/src"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate smolvla
cd /data/zhenyangfan/RoboTwin/policy/SmolVLA
cd "${SCRIPT_DIR}"

# Edit the values in this block directly before launching the script.
TRAIN_MODE="finetune"
DATASET_REPO_ID="robotwin_multitask_5_cam_high"
DATASET_ROOT="${SCRIPT_DIR}/data/${DATASET_REPO_ID}"
TRAIN_OUTPUT_ROOT="${SCRIPT_DIR}/outputs/train/${DATASET_REPO_ID}"
STAGE1_OUTPUT_ROOT="${SCRIPT_DIR}/outputs/stage1"
STAGE2_OUTPUT_ROOT="${SCRIPT_DIR}/outputs/stage2"

TRAIN_TAG="rgb_seen_random"
RUN_TAG="rgb_seen_random"
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

PRETRAINED_PATH="/data/weights/smolvla_base"
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

RESUME_FROM="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/050000"
PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false
PYTHONPATH="${LOCAL_SRC_DIR}"

MULTI_TASK_NAMES=(
  sim-open_laptop-demo_clean-50
  sim-pick_dual_bottles-demo_clean-50
  sim-put_bottles_dustbin-demo_clean-50
  sim-place_burger_fries-demo_clean-50
  sim-handover_block-demo_clean-50
)
INSTRUCTION_TYPE="seen"
EVAC_CKPT=""
EVAC_CONFIG="${SCRIPT_DIR}/evac/configs/robotwin/train_config.yaml"
STAGE1_CKPT=""
FAILURE_TABLE_PATHS_JSON=""
SEED=0
NUM_EPOCHS=20
CORRECTION_BATCH_SIZE=4
FUTURE_OFFSET=16
ACT_CHUNK_SIZE=50
PREFIX_STEPS=16
ACTION_DIM=14
LATENT_DIM=4
ADAPTER_HIDDEN_DIM=512
PREDICTOR_HIDDEN_DIM=512
DYN_ZERO_STEPS=0
DYN_RAMP_STEPS=1000
DYN_MAX_WEIGHT=1.0
DYN_WARMUP_CURVE="cosine"

RETAIN_WEIGHT=1.0
RETAIN_WEIGHT_FINAL=0.1
RETAIN_DECAY_START_EPOCH=0
RETAIN_DECAY_END_EPOCH=100
RETAIN_DECAY_CURVE="cosine"
FAILURE_PHASE_BINS=3
FAILURE_TRANSLATION_DIR_BINS=6
FAILURE_TRANSLATION_MAG_BINS=3
FAILURE_ROTATION_DIR_BINS=6
FAILURE_ROTATION_MAG_BINS=3
FAILURE_EXPLORE_K=4
SAMPLE_PHASE_WINDOW_LEN=30
SAMPLE_SKIP_HEAD_RATIO=0.6
START_MARGIN=16
MAX_ROLLOUT_STEPS=1
ACT_ALIGNED_ROLLOUT_EXEC_STEPS=16
PLANNER_TARGET_MODE="backward"
PLANNER_TARGET_LOOKAHEAD_STEPS=6
PLANNER_ORIENT_WEIGHT=0.0573
PLANNER_GRIPPER_PENALTY=1.0
PLANNER_NEAREST_WINDOW_RADIUS=12
PLANNER_ACTIVE_JOINT_DELTA_THRESH=0.01
PLANNER_ACTIVE_GRIPPER_DELTA_THRESH=0.05
ACT_ALIGNED_MIN_DIST_FALLBACK_FORCE_CORRECTION=true
ACT_ALIGNED_MIN_DIST_RECOVER_RATIO=0.75
ACT_ALIGNED_REAL_ERROR_TRIGGER_ENABLE=true
ACT_ALIGNED_REAL_ERROR_MIN_DIST_THRESH=0.01
ACT_ALIGNED_REAL_ERROR_MIN_DIST_DELTA_THRESH=0.005
RECOVER_EVAL_ENABLE=false
RECOVER_EVAL_SAVE_VIDEO=false
RECOVER_EVAL_GRIPPER_OPEN_THRESH=0.8
RECOVER_EVAL_POS_THRESH_M=0.03
RECOVER_EVAL_ROT_THRESH_DEG=10.0
RECOVER_EVAL_NEAREST_WINDOW_RADIUS=16
RECOVER_EVAL_VIDEO_BRIDGE_STEPS=16
ACT_ALIGNED_CORRECTION_INTERP_NEAREST_ENABLE=false
ACT_ALIGNED_CORRECTION_INTERP_PREFIX_RATIO=0.6
ACT_ALIGNED_CORRECTION_PLANNER_PREFIX_RATIO=0.5
ACT_ALIGNED_CORRECTION_GRIPPER_CLOSE_PREFIX_RATIO=0.32
ACT_ALIGNED_CORRECTION_COMPOSE_GT_TAIL_ENABLE=true
ACT_ALIGNED_CORRECTION_GRIPPER_SWITCH_RATIO=0.5
ACT_ALIGNED_RECOVER_GRIPPER_PENALTY=0.0
ACT_ALIGNED_ENABLE_PERTURB=true
ACT_ALIGNED_PERTURB_PROB=1.0
ACT_ALIGNED_PERTURB_ERROR_MODE="open_laptop_pregrasp"
ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_CLOSE_PROB=0.5
ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_TRANSLATION_PROB=0.0
ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_ROTATION_PROB=0.0
ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN=0.10
ACT_ALIGNED_PERTURB_ROT_MAX_DEG=15.0
ACT_ALIGNED_PERTURB_MAG_RANDOM=false
ACT_ALIGNED_PERTURB_MAG_RAND_MIN=1.0
ACT_ALIGNED_PERTURB_MAG_RAND_MAX=1.4
ACT_ALIGNED_PERTURB_REJECT_SAMPLING_ENABLE=true
ACT_ALIGNED_PERTURB_REJECT_MAX_TRIALS=4
ACT_ALIGNED_PERTURB_REJECT_DIR_JITTER_EPS=0.2
ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN=0.10
ACT_ALIGNED_PERTURB_GRIPPER_OPEN_MAX=0.90
ACT_ALIGNED_PERTURB_GRIPPER_FAST_RATIO=0.20
ACT_ALIGNED_SAMPLE_PREGRASP_PHASE_WINDOW_LEN=30
ACT_ALIGNED_SAMPLE_TIMEOUT_SEC=30
URDF_PATH="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
CUROBO_LEFT_YML="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml"
CUROBO_RIGHT_YML="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml"

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
  local base_name="${RUN_TIMESTAMP}"
  if [ -n "${RUN_TAG}" ]; then
    base_name="${base_name}-${RUN_TAG}"
  fi
  local target_dir="${root_dir}/${base_name}"
  if [ -e "${target_dir}" ]; then
    local suffix=1
    while [ -e "${root_dir}/${base_name}-${suffix}" ]; do
      suffix=$((suffix + 1))
    done
    target_dir="${root_dir}/${base_name}-${suffix}"
  fi
  echo "${target_dir}"
}

MODE_LABEL=""
OUTPUT_ROOT=""
OUTPUT_DIR=""
ARTIFACT_DIR=""
LAUNCH_SCRIPT_BASENAME=""
RESUME_CHECKPOINT_DIR=""

case "${TRAIN_MODE}" in
  finetune)
    MODE_LABEL="finetune"
    OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT}"
    LAUNCH_SCRIPT_BASENAME="launch_train_robotwin_multitask.sh"
    mkdir -p "${OUTPUT_ROOT}"
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
    ;;
  stage1)
    MODE_LABEL="stage1"
    OUTPUT_ROOT="${STAGE1_OUTPUT_ROOT}"
    LAUNCH_SCRIPT_BASENAME="launch_train_stage1.sh"
    mkdir -p "${OUTPUT_ROOT}"
    OUTPUT_DIR=$(build_run_dir "${OUTPUT_ROOT}")
    ARTIFACT_DIR="${OUTPUT_DIR}"
    mkdir -p "${ARTIFACT_DIR}"
    ;;
  stage2)
    MODE_LABEL="stage2"
    OUTPUT_ROOT="${STAGE2_OUTPUT_ROOT}"
    LAUNCH_SCRIPT_BASENAME="launch_train_stage2.sh"
    mkdir -p "${OUTPUT_ROOT}"
    OUTPUT_DIR=$(build_run_dir "${OUTPUT_ROOT}")
    ARTIFACT_DIR="${OUTPUT_DIR}"
    mkdir -p "${ARTIFACT_DIR}"
    ;;
  *)
    echo "Unsupported TRAIN_MODE: ${TRAIN_MODE}" >&2
    echo "Expected one of: finetune, stage1, stage2" >&2
    exit 1
    ;;
esac

if [ "${TRAIN_MODE}" = "stage1" ] || [ "${TRAIN_MODE}" = "stage2" ]; then
  if [ -z "${EVAC_CKPT}" ]; then
    echo "EVAC_CKPT is required for ${TRAIN_MODE}." >&2
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
fi

if [ "${TRAIN_MODE}" = "stage2" ]; then
  if [ -z "${STAGE1_CKPT}" ]; then
    echo "STAGE1_CKPT is required for stage2." >&2
    exit 1
  fi
  if [ -z "${FAILURE_TABLE_PATHS_JSON}" ]; then
    echo "FAILURE_TABLE_PATHS_JSON is required for stage2." >&2
    exit 1
  fi
fi

cp "${SCRIPT_PATH}" "${ARTIFACT_DIR}/${LAUNCH_SCRIPT_BASENAME}"

cat > "${ARTIFACT_DIR}/run_meta.txt" <<EOF
script=${SCRIPT_PATH}
script_dir=${SCRIPT_DIR}
train_mode=${TRAIN_MODE}
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
correction_batch_size=${CORRECTION_BATCH_SIZE}
num_workers=${NUM_WORKERS}
num_epochs=${NUM_EPOCHS}
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
run_timestamp=${RUN_TIMESTAMP}
output_dir=${OUTPUT_DIR}
resume_from=${RESUME_FROM}
pythonnousersite=${PYTHONNOUSERSITE}
pythonpath=${PYTHONPATH}
instruction_type=${INSTRUCTION_TYPE}
evac_ckpt=${EVAC_CKPT}
evac_config=${EVAC_CONFIG}
stage1_ckpt=${STAGE1_CKPT}
failure_table_paths_json=${FAILURE_TABLE_PATHS_JSON}
seed=${SEED}
EOF

CMD=()

if [ "${TRAIN_MODE}" = "finetune" ]; then
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
fi

if [ "${TRAIN_MODE}" = "stage1" ]; then
  CMD=(
    python3
    "${SCRIPT_DIR}/train_smolvla.py"
    stage1
    --output_dir "${OUTPUT_DIR}"
    --smolvla_pretrained_path "${PRETRAINED_PATH}"
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
    --num_epochs "${NUM_EPOCHS}"
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
fi

if [ "${TRAIN_MODE}" = "stage2" ]; then
  CMD=(
    python3
    "${SCRIPT_DIR}/train_smolvla.py"
    stage2
    --output_dir "${OUTPUT_DIR}"
    --smolvla_pretrained_path "${PRETRAINED_PATH}"
    --freeze_vision_encoder "${FREEZE_VISION_ENCODER}"
    --train_expert_only "${TRAIN_EXPERT_ONLY}"
    --load_vlm_weights "${LOAD_VLM_WEIGHTS}"
    --stage1_ckpt "${STAGE1_CKPT}"
    --multi_task_names "${MULTI_TASK_NAMES[@]}"
    --instruction_type "${INSTRUCTION_TYPE}"
    --failure_table_paths_json "${FAILURE_TABLE_PATHS_JSON}"
    --evac_ckpt "${EVAC_CKPT}"
    --evac_config "${EVAC_CONFIG}"
    --urdf_path "${URDF_PATH}"
    --curobo_left_yml "${CUROBO_LEFT_YML}"
    --curobo_right_yml "${CUROBO_RIGHT_YML}"
    --device "${POLICY_DEVICE}:0"
    --seed "${SEED}"
    --batch_size "${BATCH_SIZE}"
    --correction_batch_size "${CORRECTION_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --num_epochs "${NUM_EPOCHS}"
    --save_freq "${SAVE_FREQ}"
    --learning_rate "${OPTIMIZER_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --scheduler_warmup_steps "${SCHEDULER_WARMUP_STEPS}"
    --scheduler_decay_steps "${SCHEDULER_DECAY_STEPS}"
    --scheduler_decay_lr "${SCHEDULER_DECAY_LR}"
    --future_offset "${FUTURE_OFFSET}"
    --prefix_steps "${PREFIX_STEPS}"
    --action_dim "${ACTION_DIM}"
    --latent_dim "${LATENT_DIM}"
    --adapter_hidden_dim "${ADAPTER_HIDDEN_DIM}"
    --predictor_hidden_dim "${PREDICTOR_HIDDEN_DIM}"
    --dyn_zero_steps "${DYN_ZERO_STEPS}"
    --dyn_ramp_steps "${DYN_RAMP_STEPS}"
    --dyn_max_weight "${DYN_MAX_WEIGHT}"
    --dyn_warmup_curve "${DYN_WARMUP_CURVE}"
    --retain_weight "${RETAIN_WEIGHT}"
    --retain_weight_final "${RETAIN_WEIGHT_FINAL}"
    --retain_decay_start_epoch "${RETAIN_DECAY_START_EPOCH}"
    --retain_decay_end_epoch "${RETAIN_DECAY_END_EPOCH}"
    --retain_decay_curve "${RETAIN_DECAY_CURVE}"
    --failure_phase_bins "${FAILURE_PHASE_BINS}"
    --failure_translation_dir_bins "${FAILURE_TRANSLATION_DIR_BINS}"
    --failure_translation_mag_bins "${FAILURE_TRANSLATION_MAG_BINS}"
    --failure_rotation_dir_bins "${FAILURE_ROTATION_DIR_BINS}"
    --failure_rotation_mag_bins "${FAILURE_ROTATION_MAG_BINS}"
    --failure_explore_k "${FAILURE_EXPLORE_K}"
    --sample_phase_window_len "${SAMPLE_PHASE_WINDOW_LEN}"
    --sample_skip_head_ratio "${SAMPLE_SKIP_HEAD_RATIO}"
    --start_margin "${START_MARGIN}"
    --max_rollout_steps "${MAX_ROLLOUT_STEPS}"
    --act_chunk_size "${ACT_CHUNK_SIZE}"
    --act_aligned_rollout_exec_steps "${ACT_ALIGNED_ROLLOUT_EXEC_STEPS}"
    --planner_target_mode "${PLANNER_TARGET_MODE}"
    --planner_target_lookahead_steps "${PLANNER_TARGET_LOOKAHEAD_STEPS}"
    --planner_orient_weight "${PLANNER_ORIENT_WEIGHT}"
    --planner_gripper_penalty "${PLANNER_GRIPPER_PENALTY}"
    --planner_nearest_window_radius "${PLANNER_NEAREST_WINDOW_RADIUS}"
    --planner_active_joint_delta_thresh "${PLANNER_ACTIVE_JOINT_DELTA_THRESH}"
    --planner_active_gripper_delta_thresh "${PLANNER_ACTIVE_GRIPPER_DELTA_THRESH}"
    --act_aligned_min_dist_fallback_force_correction "${ACT_ALIGNED_MIN_DIST_FALLBACK_FORCE_CORRECTION}"
    --act_aligned_min_dist_recover_ratio "${ACT_ALIGNED_MIN_DIST_RECOVER_RATIO}"
    --act_aligned_real_error_trigger_enable "${ACT_ALIGNED_REAL_ERROR_TRIGGER_ENABLE}"
    --act_aligned_real_error_min_dist_thresh "${ACT_ALIGNED_REAL_ERROR_MIN_DIST_THRESH}"
    --act_aligned_real_error_min_dist_delta_thresh "${ACT_ALIGNED_REAL_ERROR_MIN_DIST_DELTA_THRESH}"
    --recover_eval_enable "${RECOVER_EVAL_ENABLE}"
    --recover_eval_save_video "${RECOVER_EVAL_SAVE_VIDEO}"
    --recover_eval_gripper_open_thresh "${RECOVER_EVAL_GRIPPER_OPEN_THRESH}"
    --recover_eval_pos_thresh_m "${RECOVER_EVAL_POS_THRESH_M}"
    --recover_eval_rot_thresh_deg "${RECOVER_EVAL_ROT_THRESH_DEG}"
    --recover_eval_nearest_window_radius "${RECOVER_EVAL_NEAREST_WINDOW_RADIUS}"
    --recover_eval_video_bridge_steps "${RECOVER_EVAL_VIDEO_BRIDGE_STEPS}"
    --act_aligned_correction_interp_nearest_enable "${ACT_ALIGNED_CORRECTION_INTERP_NEAREST_ENABLE}"
    --act_aligned_correction_interp_prefix_ratio "${ACT_ALIGNED_CORRECTION_INTERP_PREFIX_RATIO}"
    --act_aligned_correction_planner_prefix_ratio "${ACT_ALIGNED_CORRECTION_PLANNER_PREFIX_RATIO}"
    --act_aligned_correction_gripper_close_prefix_ratio "${ACT_ALIGNED_CORRECTION_GRIPPER_CLOSE_PREFIX_RATIO}"
    --act_aligned_correction_compose_gt_tail_enable "${ACT_ALIGNED_CORRECTION_COMPOSE_GT_TAIL_ENABLE}"
    --act_aligned_correction_gripper_switch_ratio "${ACT_ALIGNED_CORRECTION_GRIPPER_SWITCH_RATIO}"
    --act_aligned_recover_gripper_penalty "${ACT_ALIGNED_RECOVER_GRIPPER_PENALTY}"
    --act_aligned_enable_perturb "${ACT_ALIGNED_ENABLE_PERTURB}"
    --act_aligned_perturb_prob "${ACT_ALIGNED_PERTURB_PROB}"
    --act_aligned_perturb_error_mode "${ACT_ALIGNED_PERTURB_ERROR_MODE}"
    --act_aligned_perturb_open_laptop_pregrasp_close_prob "${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_CLOSE_PROB}"
    --act_aligned_perturb_open_laptop_pregrasp_translation_prob "${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_TRANSLATION_PROB}"
    --act_aligned_perturb_open_laptop_pregrasp_rotation_prob "${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_ROTATION_PROB}"
    --act_aligned_perturb_eef_fail_gain "${ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN}"
    --act_aligned_perturb_rot_max_deg "${ACT_ALIGNED_PERTURB_ROT_MAX_DEG}"
    --act_aligned_perturb_mag_random "${ACT_ALIGNED_PERTURB_MAG_RANDOM}"
    --act_aligned_perturb_mag_rand_min "${ACT_ALIGNED_PERTURB_MAG_RAND_MIN}"
    --act_aligned_perturb_mag_rand_max "${ACT_ALIGNED_PERTURB_MAG_RAND_MAX}"
    --act_aligned_perturb_reject_sampling_enable "${ACT_ALIGNED_PERTURB_REJECT_SAMPLING_ENABLE}"
    --act_aligned_perturb_reject_max_trials "${ACT_ALIGNED_PERTURB_REJECT_MAX_TRIALS}"
    --act_aligned_perturb_reject_dir_jitter_eps "${ACT_ALIGNED_PERTURB_REJECT_DIR_JITTER_EPS}"
    --act_aligned_perturb_gripper_close_min "${ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN}"
    --act_aligned_perturb_gripper_open_max "${ACT_ALIGNED_PERTURB_GRIPPER_OPEN_MAX}"
    --act_aligned_perturb_gripper_fast_ratio "${ACT_ALIGNED_PERTURB_GRIPPER_FAST_RATIO}"
    --act_aligned_sample_pregrasp_phase_window_len "${ACT_ALIGNED_SAMPLE_PREGRASP_PHASE_WINDOW_LEN}"
    --act_aligned_sample_timeout_sec "${ACT_ALIGNED_SAMPLE_TIMEOUT_SEC}"
  )
fi

printf '%q ' "${CMD[@]}" > "${ARTIFACT_DIR}/launch_command.sh"
printf '\n' >> "${ARTIFACT_DIR}/launch_command.sh"

echo "Training mode: ${MODE_LABEL}"
echo "Training output dir: ${OUTPUT_DIR}"
echo "Train tag: ${TRAIN_TAG}"
echo "Run tag: ${RUN_TAG}"

export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE
export PYTHONPATH
export TOKENIZERS_PARALLELISM
export LEROBOT_RANDOMIZE_TASK_FROM_EPISODE_INSTRUCTIONS="${RANDOMIZE_SEEN_INSTRUCTIONS}"

"${CMD[@]}" 2>&1 | tee "${ARTIFACT_DIR}/log.log"

if [ "${TRAIN_MODE}" = "finetune" ] && [ "${ARTIFACT_DIR}" != "${OUTPUT_DIR}" ] && [ -d "${OUTPUT_DIR}" ]; then
  cp "${ARTIFACT_DIR}/${LAUNCH_SCRIPT_BASENAME}" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/run_meta.txt" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/launch_command.sh" "${OUTPUT_DIR}/"
  cp "${ARTIFACT_DIR}/log.log" "${OUTPUT_DIR}/"
  rm -rf "${ARTIFACT_DIR}"
fi

echo "${OUTPUT_DIR}"
