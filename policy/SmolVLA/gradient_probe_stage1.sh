#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
LOCAL_SRC_DIR="${SCRIPT_DIR}/src"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate smolvla
cd "${SCRIPT_DIR}"

DATASET_REPO_ID="robotwin_multitask_5_cam_high"
OUTPUT_ROOT="${SCRIPT_DIR}/outputs/gradient_probe/${DATASET_REPO_ID}"
RUN_TAG="stage1_gradient_probe"

PRETRAINED_PATH="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/055000/pretrained_model"
STAGE1_CKPT="${STAGE1_CKPT:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
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
EVAC_CKPT=/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt
EVAC_CONFIG=/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml
URDF_PATH="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
CUROBO_LEFT_YML="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml"
CUROBO_RIGHT_YML="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml"

SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PROBE_BATCHES="${PROBE_BATCHES:-4}"
PROBE_GLOBAL_STEP="${PROBE_GLOBAL_STEP:-1000}"
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
COND_ZERO_STEPS=0
COND_RAMP_STEPS=1000
COND_MAX_WEIGHT=0.5
COND_WARMUP_CURVE="cosine"
FAILURE_MODE="train"
FAILURE_TABLE_PATHS_JSON="/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/explore/20260427_235851-explore_stage1_spatial_projector_accelerate_45_vreuse_4_8_12_14/filtered_first_pregrasp_single_mode/multitask_failure_manifest.json"
FAILURE_CORR_BATCH_RATIO="${FAILURE_CORR_BATCH_RATIO:-1.0}"
PROBE_CORR_BATCH_SIZE="${PROBE_CORR_BATCH_SIZE:-${BATCH_SIZE}}"
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
SAVE_PERTURB_ROLLOUT_VIDEO=false
SAVE_CORRECTION_DEBUG=false
RECOVER_EVAL_GRIPPER_OPEN_THRESH=0.3
RECOVER_EVAL_POS_THRESH_M=0.04
RECOVER_EVAL_ROT_THRESH_DEG=8.0
RECOVER_EVAL_NEAREST_WINDOW_RADIUS=16
RECOVER_EVAL_VIDEO_BRIDGE_STEPS=16
ACT_ALIGNED_ENABLE_PERTURB=true
ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN=0.05
ACT_ALIGNED_PERTURB_ROT_MAX_DEG=15.0
ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN=0.10
EVAC_BLUR_FILTER_ENABLE=true
EVAC_BLUR_FILTER_MIN_RATIO=0.50
EVAC_BLUR_FILTER_PATCH_PAD_PX=24
EVAC_USE_DUAL_CACHE=true
EVAC_DC_V_BOUNDS=(4 8 12 14)
EVAC_DC_BUDGET=-1
EVAC_DC_ENC_START=999
EVAC_DC_REPLAY_STEP_NOISE=false
EVAC_DC_HF_METRIC=false
EVAC_DC_V_BLUR_ON_REUSE=false
EVAC_DC_V_BLUR_KERNEL=3
EVAC_DC_V_BLUR_STRENGTH=0.15
PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false
SMOLVLA_EVAC_PRINT_RUNTIME=false
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONPATH="/data/zhenyangfan/RoboTwin:${LOCAL_SRC_DIR}"

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  echo "CUDA_VISIBLE_DEVICES is empty." >&2
  exit 1
fi

if [ ! -d "${PRETRAINED_PATH}" ]; then
  echo "PRETRAINED_PATH does not exist: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [ -n "${STAGE1_CKPT}" ] && [ ! -f "${STAGE1_CKPT}" ]; then
  echo "STAGE1_CKPT does not exist: ${STAGE1_CKPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
timestamp=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_ROOT}/${timestamp}-${RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"
cp "${SCRIPT_PATH}" "${OUTPUT_DIR}/launch_gradient_probe_stage1.sh"

PROBE_OUTPUT_JSON="${OUTPUT_DIR}/gradient_probe.json"

CMD=(
  python "${SCRIPT_DIR}/latentcorr/gradient_probe_smolvla.py"
  --output_dir "${OUTPUT_DIR}"
  --smolvla_pretrained_path "${PRETRAINED_PATH}"
  --resume_ckpt ""
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
  --save_perturb_rollout_video "${SAVE_PERTURB_ROLLOUT_VIDEO}"
  --save_correction_debug "${SAVE_CORRECTION_DEBUG}"
  --recover_eval_gripper_open_thresh "${RECOVER_EVAL_GRIPPER_OPEN_THRESH}"
  --recover_eval_pos_thresh_m "${RECOVER_EVAL_POS_THRESH_M}"
  --recover_eval_rot_thresh_deg "${RECOVER_EVAL_ROT_THRESH_DEG}"
  --recover_eval_nearest_window_radius "${RECOVER_EVAL_NEAREST_WINDOW_RADIUS}"
  --recover_eval_video_bridge_steps "${RECOVER_EVAL_VIDEO_BRIDGE_STEPS}"
  --act_aligned_enable_perturb "${ACT_ALIGNED_ENABLE_PERTURB}"
  --act_aligned_perturb_eef_fail_gain "${ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN}"
  --act_aligned_perturb_rot_max_deg "${ACT_ALIGNED_PERTURB_ROT_MAX_DEG}"
  --act_aligned_perturb_gripper_close_min "${ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN}"
  --evac_blur_filter_enable "${EVAC_BLUR_FILTER_ENABLE}"
  --evac_blur_filter_min_ratio "${EVAC_BLUR_FILTER_MIN_RATIO}"
  --evac_blur_filter_patch_pad_px "${EVAC_BLUR_FILTER_PATCH_PAD_PX}"
  --evac_use_dual_cache "${EVAC_USE_DUAL_CACHE}"
  --evac_dc_v_bounds "${EVAC_DC_V_BOUNDS[@]}"
  --evac_dc_budget "${EVAC_DC_BUDGET}"
  --evac_dc_enc_start "${EVAC_DC_ENC_START}"
  --evac_dc_replay_step_noise "${EVAC_DC_REPLAY_STEP_NOISE}"
  --evac_dc_hf_metric "${EVAC_DC_HF_METRIC}"
  --evac_dc_v_blur_on_reuse "${EVAC_DC_V_BLUR_ON_REUSE}"
  --evac_dc_v_blur_kernel "${EVAC_DC_V_BLUR_KERNEL}"
  --evac_dc_v_blur_strength "${EVAC_DC_V_BLUR_STRENGTH}"
  --probe_batches "${PROBE_BATCHES}"
  --probe_corr_batch_size "${PROBE_CORR_BATCH_SIZE}"
  --probe_global_step "${PROBE_GLOBAL_STEP}"
  --stage1_ckpt "${STAGE1_CKPT}"
  --probe_output_json "${PROBE_OUTPUT_JSON}"
)

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/launch_command.sh"
printf '\n' >> "${OUTPUT_DIR}/launch_command.sh"

echo "Gradient probe output dir: ${OUTPUT_DIR}"
echo "Gradient probe json: ${PROBE_OUTPUT_JSON}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "BATCH_SIZE=${BATCH_SIZE}, PROBE_CORR_BATCH_SIZE=${PROBE_CORR_BATCH_SIZE}, PROBE_BATCHES=${PROBE_BATCHES}"

export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE
export PYTHONPATH
export TOKENIZERS_PARALLELISM
export SMOLVLA_EVAC_PRINT_RUNTIME
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/probe.log"
