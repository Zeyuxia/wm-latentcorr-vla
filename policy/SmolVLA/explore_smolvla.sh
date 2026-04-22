#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
LOCAL_SRC_DIR="${SCRIPT_DIR}/src"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate smolvla
cd /data/zhenyangfan/RoboTwin

RUN_TAG="explore_stage1_spatial_projector_accelerate_56"
OUTPUT_DIR=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/explore
SMOLVLA_PRETRAINED_PATH=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/055000/pretrained_model
STAGE1_CKPT=
MULTI_TASK_NAMES=(
  sim-handover_block-demo_clean-50
  sim-open_laptop-demo_clean-50
  sim-put_bottles_dustbin-demo_clean-50
  sim-pick_dual_bottles-demo_clean-50
  sim-place_burger_fries-demo_clean-50
)
INSTRUCTION_TYPE=seen
EVAC_CKPT=/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_plus_pi05_rollout_2026-04-20T12-52-47/checkpoints/epoch=666-step=20000.ckpt
EVAC_CONFIG=/data/yujieyang/EVAC/configs/robotwin/train_config_mixed50p12_plus_pi05_rollout_for_zhenyangfan.yaml
URDF_PATH=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
CUROBO_LEFT_YML=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
CUROBO_RIGHT_YML=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
DEVICE=cuda
GPU_IDS=1,5,6,7
MAIN_PROCESS_PORT=29501
SEED=0
NUM_WORKERS=0
FUTURE_OFFSET=16
PREFIX_STEPS=16
ACTION_DIM=14
LATENT_DIM=4
ADAPTER_HIDDEN_DIM=512
PREDICTOR_HIDDEN_DIM=512
ACT_CHUNK_SIZE=50
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
RECOVER_EVAL_GRIPPER_OPEN_THRESH=0.3
RECOVER_EVAL_POS_THRESH_M=0.04
RECOVER_EVAL_ROT_THRESH_DEG=8.0
RECOVER_EVAL_NEAREST_WINDOW_RADIUS=16
RECOVER_EVAL_VIDEO_BRIDGE_STEPS=16
ACT_ALIGNED_ENABLE_PERTURB=true
ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN=0.03
ACT_ALIGNED_PERTURB_ROT_MAX_DEG=10.0
ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN=0.10
EVAC_BLUR_FILTER_ENABLE=true
EVAC_BLUR_FILTER_MIN_RATIO=0.50
EVAC_BLUR_FILTER_PATCH_PAD_PX=24
PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false
PYTHONPATH="/data/zhenyangfan/RoboTwin:${LOCAL_SRC_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_NAME="${TIMESTAMP}"
if [ -n "${RUN_TAG}" ]; then
  BASE_NAME="${BASE_NAME}-${RUN_TAG}"
fi
OUTPUT_DIR="${OUTPUT_DIR}/${BASE_NAME}"

if [ -f "${SMOLVLA_PRETRAINED_PATH}" ]; then
  STAGE1_CKPT="${SMOLVLA_PRETRAINED_PATH}"
  SMOLVLA_PRETRAINED_PATH=$(python3 - "${STAGE1_CKPT}" <<'PY'
import sys
import torch

ckpt_path = sys.argv[1]
payload = torch.load(ckpt_path, map_location="cpu")
args = payload.get("args") or {}
path = args.get("smolvla_pretrained_path")
if not path:
    raise SystemExit(f"Missing args.smolvla_pretrained_path in checkpoint: {ckpt_path}")
print(path)
PY
)
fi
if [ -n "${STAGE1_CKPT}" ] && [ ! -f "${STAGE1_CKPT}" ]; then
  echo "Stage1 checkpoint path is missing: ${STAGE1_CKPT}" >&2
  exit 1
fi
if [ ! -d "${SMOLVLA_PRETRAINED_PATH}" ]; then
  echo "SmolVLA pretrained path is missing: ${SMOLVLA_PRETRAINED_PATH}" >&2
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
cp "${SCRIPT_PATH}" "${OUTPUT_DIR}/explore_smolvla.sh"

export PYTHONNOUSERSITE
export PYTHONPATH
export TOKENIZERS_PARALLELISM
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NUM_GPUS=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
MULTI_GPU_FLAG=""
if [ "${NUM_GPUS}" -gt 1 ]; then
  MULTI_GPU_FLAG="--multi_gpu"
fi

CMD=(
  accelerate launch
  ${MULTI_GPU_FLAG}
  --num_processes "${NUM_GPUS}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  /data/zhenyangfan/RoboTwin/policy/SmolVLA/latentcorr/explore_smolvla.py
    --output_dir "${OUTPUT_DIR}" \
    --smolvla_pretrained_path "${SMOLVLA_PRETRAINED_PATH}" \
    --multi_task_names "${MULTI_TASK_NAMES[@]}" \
    --instruction_type "${INSTRUCTION_TYPE}" \
    --evac_ckpt "${EVAC_CKPT}" \
    --evac_config "${EVAC_CONFIG}" \
    --urdf_path "${URDF_PATH}" \
    --curobo_left_yml "${CUROBO_LEFT_YML}" \
    --curobo_right_yml "${CUROBO_RIGHT_YML}" \
    --device "${DEVICE}" \
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
    --start_margin "${START_MARGIN}" \
    --failure_phase_bins "${FAILURE_PHASE_BINS}" \
    --failure_translation_dir_bins "${FAILURE_TRANSLATION_DIR_BINS}" \
    --failure_translation_mag_bins "${FAILURE_TRANSLATION_MAG_BINS}" \
    --failure_rotation_dir_bins "${FAILURE_ROTATION_DIR_BINS}" \
    --failure_rotation_mag_bins "${FAILURE_ROTATION_MAG_BINS}" \
    --failure_explore_k "${FAILURE_EXPLORE_K}" \
    --act_aligned_rollout_exec_steps "${ACT_ALIGNED_ROLLOUT_EXEC_STEPS}" \
    --planner_orient_weight "${PLANNER_ORIENT_WEIGHT}" \
    --planner_gripper_penalty "${PLANNER_GRIPPER_PENALTY}" \
    --planner_nearest_window_radius "${PLANNER_NEAREST_WINDOW_RADIUS}" \
    --planner_active_joint_delta_thresh "${PLANNER_ACTIVE_JOINT_DELTA_THRESH}" \
    --planner_active_gripper_delta_thresh "${PLANNER_ACTIVE_GRIPPER_DELTA_THRESH}" \
    --recover_eval_save_video "${RECOVER_EVAL_SAVE_VIDEO}" \
    --save_perturb_rollout_video "${SAVE_PERTURB_ROLLOUT_VIDEO}" \
    --recover_eval_gripper_open_thresh "${RECOVER_EVAL_GRIPPER_OPEN_THRESH}" \
    --recover_eval_pos_thresh_m "${RECOVER_EVAL_POS_THRESH_M}" \
    --recover_eval_rot_thresh_deg "${RECOVER_EVAL_ROT_THRESH_DEG}" \
    --recover_eval_nearest_window_radius "${RECOVER_EVAL_NEAREST_WINDOW_RADIUS}" \
    --recover_eval_video_bridge_steps "${RECOVER_EVAL_VIDEO_BRIDGE_STEPS}" \
    --act_aligned_enable_perturb "${ACT_ALIGNED_ENABLE_PERTURB}" \
    --act_aligned_perturb_eef_fail_gain "${ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN}" \
    --act_aligned_perturb_rot_max_deg "${ACT_ALIGNED_PERTURB_ROT_MAX_DEG}" \
    --act_aligned_perturb_gripper_close_min "${ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN}" \
    --evac_blur_filter_enable "${EVAC_BLUR_FILTER_ENABLE}" \
    --evac_blur_filter_min_ratio "${EVAC_BLUR_FILTER_MIN_RATIO}" \
    --evac_blur_filter_patch_pad_px "${EVAC_BLUR_FILTER_PATCH_PAD_PX}"
)

if [ -n "${STAGE1_CKPT}" ]; then
  CMD+=(--stage1_ckpt "${STAGE1_CKPT}")
fi

echo "Output dir: ${OUTPUT_DIR}"
echo "GPU ids: ${GPU_IDS}"
echo "Run tag: ${RUN_TAG}"
echo "Running: CUDA_VISIBLE_DEVICES=${GPU_IDS} ${CMD[*]}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
