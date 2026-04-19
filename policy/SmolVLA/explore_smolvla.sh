#!/bin/bash
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda activate smolvla
cd /data/zhenyangfan/RoboTwin

OUTPUT_DIR=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/explore/$(date +"%Y%m%d_%H%M%S")
SMOLVLA_PRETRAINED_PATH=
STAGE1_CKPT=
MULTI_TASK_NAMES=(
  sim-open_laptop-demo_clean-50
  sim-pick_dual_bottles-demo_clean-50
  sim-put_bottles_dustbin-demo_clean-50
  sim-place_burger_fries-demo_clean-50
  sim-handover_block-demo_clean-50
)
INSTRUCTION_TYPE=seen
EVAC_CKPT=
EVAC_CONFIG=/data/zhenyangfan/RoboTwin/policy/SmolVLA/evac/configs/robotwin/train_config.yaml
URDF_PATH=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
CUROBO_LEFT_YML=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
CUROBO_RIGHT_YML=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
DEVICE=cuda:0
GPU_IDS=(0)
SEED=0
NUM_WORKERS=0
FUTURE_OFFSET=16
PREFIX_STEPS=16
ACTION_DIM=14
LATENT_DIM=4
ADAPTER_HIDDEN_DIM=512
PREDICTOR_HIDDEN_DIM=512
ACT_CHUNK_SIZE=50
SAMPLE_PHASE_WINDOW_LEN=30
SAMPLE_SKIP_HEAD_RATIO=0.6
START_MARGIN=16
FAILURE_PHASE_BINS=3
FAILURE_TRANSLATION_DIR_BINS=6
FAILURE_TRANSLATION_MAG_BINS=3
FAILURE_ROTATION_DIR_BINS=6
FAILURE_ROTATION_MAG_BINS=3
FAILURE_EXPLORE_K=4
MAX_ROLLOUT_STEPS=1
ACT_ALIGNED_ROLLOUT_EXEC_STEPS=16
PLANNER_TARGET_MODE=backward
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
RECOVER_EVAL_ENABLE=true
RECOVER_EVAL_SAVE_VIDEO=true
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
ACT_ALIGNED_PERTURB_ERROR_MODE=open_laptop_pregrasp
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

if [ -z "${SMOLVLA_PRETRAINED_PATH}" ]; then
  echo "SMOLVLA_PRETRAINED_PATH is required" >&2
  exit 1
fi
if [ -z "${STAGE1_CKPT}" ]; then
  echo "STAGE1_CKPT is required" >&2
  exit 1
fi
if [ -z "${EVAC_CKPT}" ]; then
  echo "EVAC_CKPT is required" >&2
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
mkdir -p "${OUTPUT_DIR}"

WORLD_SIZE=${#GPU_IDS[@]}
PIDS=()
for RANK in "${!GPU_IDS[@]}"; do
  GPU_ID=${GPU_IDS[$RANK]}
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 /data/zhenyangfan/RoboTwin/policy/SmolVLA/explore_smolvla.py \
    --output_dir "${OUTPUT_DIR}" \
    --smolvla_pretrained_path "${SMOLVLA_PRETRAINED_PATH}" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --multi_task_names "${MULTI_TASK_NAMES[@]}" \
    --instruction_type "${INSTRUCTION_TYPE}" \
    --evac_ckpt "${EVAC_CKPT}" \
    --evac_config "${EVAC_CONFIG}" \
    --urdf_path "${URDF_PATH}" \
    --curobo_left_yml "${CUROBO_LEFT_YML}" \
    --curobo_right_yml "${CUROBO_RIGHT_YML}" \
    --device "${DEVICE}" \
    --rank "${RANK}" \
    --world_size "${WORLD_SIZE}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --future_offset "${FUTURE_OFFSET}" \
    --prefix_steps "${PREFIX_STEPS}" \
    --action_dim "${ACTION_DIM}" \
    --latent_dim "${LATENT_DIM}" \
    --adapter_hidden_dim "${ADAPTER_HIDDEN_DIM}" \
    --predictor_hidden_dim "${PREDICTOR_HIDDEN_DIM}" \
    --act_chunk_size "${ACT_CHUNK_SIZE}" \
    --sample_phase_window_len "${SAMPLE_PHASE_WINDOW_LEN}" \
    --sample_skip_head_ratio "${SAMPLE_SKIP_HEAD_RATIO}" \
    --start_margin "${START_MARGIN}" \
    --failure_phase_bins "${FAILURE_PHASE_BINS}" \
    --failure_translation_dir_bins "${FAILURE_TRANSLATION_DIR_BINS}" \
    --failure_translation_mag_bins "${FAILURE_TRANSLATION_MAG_BINS}" \
    --failure_rotation_dir_bins "${FAILURE_ROTATION_DIR_BINS}" \
    --failure_rotation_mag_bins "${FAILURE_ROTATION_MAG_BINS}" \
    --failure_explore_k "${FAILURE_EXPLORE_K}" \
    --max_rollout_steps "${MAX_ROLLOUT_STEPS}" \
    --act_aligned_rollout_exec_steps "${ACT_ALIGNED_ROLLOUT_EXEC_STEPS}" \
    --planner_target_mode "${PLANNER_TARGET_MODE}" \
    --planner_target_lookahead_steps "${PLANNER_TARGET_LOOKAHEAD_STEPS}" \
    --planner_orient_weight "${PLANNER_ORIENT_WEIGHT}" \
    --planner_gripper_penalty "${PLANNER_GRIPPER_PENALTY}" \
    --planner_nearest_window_radius "${PLANNER_NEAREST_WINDOW_RADIUS}" \
    --planner_active_joint_delta_thresh "${PLANNER_ACTIVE_JOINT_DELTA_THRESH}" \
    --planner_active_gripper_delta_thresh "${PLANNER_ACTIVE_GRIPPER_DELTA_THRESH}" \
    --act_aligned_min_dist_fallback_force_correction "${ACT_ALIGNED_MIN_DIST_FALLBACK_FORCE_CORRECTION}" \
    --act_aligned_min_dist_recover_ratio "${ACT_ALIGNED_MIN_DIST_RECOVER_RATIO}" \
    --act_aligned_real_error_trigger_enable "${ACT_ALIGNED_REAL_ERROR_TRIGGER_ENABLE}" \
    --act_aligned_real_error_min_dist_thresh "${ACT_ALIGNED_REAL_ERROR_MIN_DIST_THRESH}" \
    --act_aligned_real_error_min_dist_delta_thresh "${ACT_ALIGNED_REAL_ERROR_MIN_DIST_DELTA_THRESH}" \
    --recover_eval_enable "${RECOVER_EVAL_ENABLE}" \
    --recover_eval_save_video "${RECOVER_EVAL_SAVE_VIDEO}" \
    --recover_eval_gripper_open_thresh "${RECOVER_EVAL_GRIPPER_OPEN_THRESH}" \
    --recover_eval_pos_thresh_m "${RECOVER_EVAL_POS_THRESH_M}" \
    --recover_eval_rot_thresh_deg "${RECOVER_EVAL_ROT_THRESH_DEG}" \
    --recover_eval_nearest_window_radius "${RECOVER_EVAL_NEAREST_WINDOW_RADIUS}" \
    --recover_eval_video_bridge_steps "${RECOVER_EVAL_VIDEO_BRIDGE_STEPS}" \
    --act_aligned_correction_interp_nearest_enable "${ACT_ALIGNED_CORRECTION_INTERP_NEAREST_ENABLE}" \
    --act_aligned_correction_interp_prefix_ratio "${ACT_ALIGNED_CORRECTION_INTERP_PREFIX_RATIO}" \
    --act_aligned_correction_planner_prefix_ratio "${ACT_ALIGNED_CORRECTION_PLANNER_PREFIX_RATIO}" \
    --act_aligned_correction_gripper_close_prefix_ratio "${ACT_ALIGNED_CORRECTION_GRIPPER_CLOSE_PREFIX_RATIO}" \
    --act_aligned_correction_compose_gt_tail_enable "${ACT_ALIGNED_CORRECTION_COMPOSE_GT_TAIL_ENABLE}" \
    --act_aligned_correction_gripper_switch_ratio "${ACT_ALIGNED_CORRECTION_GRIPPER_SWITCH_RATIO}" \
    --act_aligned_recover_gripper_penalty "${ACT_ALIGNED_RECOVER_GRIPPER_PENALTY}" \
    --act_aligned_enable_perturb "${ACT_ALIGNED_ENABLE_PERTURB}" \
    --act_aligned_perturb_prob "${ACT_ALIGNED_PERTURB_PROB}" \
    --act_aligned_perturb_error_mode "${ACT_ALIGNED_PERTURB_ERROR_MODE}" \
    --act_aligned_perturb_open_laptop_pregrasp_close_prob "${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_CLOSE_PROB}" \
    --act_aligned_perturb_open_laptop_pregrasp_translation_prob "${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_TRANSLATION_PROB}" \
    --act_aligned_perturb_open_laptop_pregrasp_rotation_prob "${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_ROTATION_PROB}" \
    --act_aligned_perturb_eef_fail_gain "${ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN}" \
    --act_aligned_perturb_rot_max_deg "${ACT_ALIGNED_PERTURB_ROT_MAX_DEG}" \
    --act_aligned_perturb_mag_random "${ACT_ALIGNED_PERTURB_MAG_RANDOM}" \
    --act_aligned_perturb_mag_rand_min "${ACT_ALIGNED_PERTURB_MAG_RAND_MIN}" \
    --act_aligned_perturb_mag_rand_max "${ACT_ALIGNED_PERTURB_MAG_RAND_MAX}" \
    --act_aligned_perturb_reject_sampling_enable "${ACT_ALIGNED_PERTURB_REJECT_SAMPLING_ENABLE}" \
    --act_aligned_perturb_reject_max_trials "${ACT_ALIGNED_PERTURB_REJECT_MAX_TRIALS}" \
    --act_aligned_perturb_reject_dir_jitter_eps "${ACT_ALIGNED_PERTURB_REJECT_DIR_JITTER_EPS}" \
    --act_aligned_perturb_gripper_close_min "${ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN}" \
    --act_aligned_perturb_gripper_open_max "${ACT_ALIGNED_PERTURB_GRIPPER_OPEN_MAX}" \
    --act_aligned_perturb_gripper_fast_ratio "${ACT_ALIGNED_PERTURB_GRIPPER_FAST_RATIO}" \
    --act_aligned_sample_pregrasp_phase_window_len "${ACT_ALIGNED_SAMPLE_PREGRASP_PHASE_WINDOW_LEN}" \
    --act_aligned_sample_timeout_sec "${ACT_ALIGNED_SAMPLE_TIMEOUT_SEC}" &
  PIDS+=($!)
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    STATUS=1
  fi
done
exit "${STATUS}"
