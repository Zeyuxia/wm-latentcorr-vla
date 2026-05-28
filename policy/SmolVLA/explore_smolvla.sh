#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
LOCAL_SRC_DIR="${SCRIPT_DIR}/src"

source /data/miniconda3/etc/profile.d/conda.sh
SMOLVLA_CONDA_ENV=${SMOLVLA_CONDA_ENV:-smolvla_cosmos}
conda activate "${SMOLVLA_CONDA_ENV}"
cd /data/zhenyangfan/RoboTwin

RUN_TAG="explore_stage1_spatial_projector_cosmos_chunk12"
OUTPUT_DIR=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/explore
SMOLVLA_PRETRAINED_PATH=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/train/robotwin_multitask_5_cam_high/20260419_141110-rgb_seen_random/checkpoints/055000/pretrained_model
STAGE1_CKPT=
MULTI_TASK_NAMES=(
  sim-open_laptop-demo_clean-50
  sim-pick_dual_bottles-demo_clean-50
  sim-put_bottles_dustbin-demo_clean-50
  sim-place_burger_fries-demo_clean-50
  sim-handover_block-demo_clean-50
)
if [ -n "${MULTI_TASK_NAMES_OVERRIDE:-}" ]; then
  read -r -a MULTI_TASK_NAMES <<< "${MULTI_TASK_NAMES_OVERRIDE}"
fi
SMOLVLA_LATENT_DATA_SOURCE=${SMOLVLA_LATENT_DATA_SOURCE:-smolvla_rgbfix}
SMOLVLA_LATENT_DATA_ROOT=${SMOLVLA_LATENT_DATA_ROOT:-${SCRIPT_DIR}/data}
SMOLVLA_LATENT_DATA_SUFFIX=${SMOLVLA_LATENT_DATA_SUFFIX:-_rgbfix}
INSTRUCTION_TYPE=seen
EVAC_CKPT=/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt
EVAC_CONFIG=/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml
WORLD_MODEL_BACKEND=${WORLD_MODEL_BACKEND:-cosmos}
SIM_USE_SUBPROCESS=${SIM_USE_SUBPROCESS:-true}
SIM_TIMEOUT_S=${SIM_TIMEOUT_S:-180}
if [ -z "${WORLD_MODEL_QUALITY_RECORD+x}" ]; then
  WORLD_MODEL_QUALITY_RECORD=/data/zhenyangfan/RoboTwin/policy/SmolVLA/outputs/explore_analysis/world_model_quality_record.md
fi
case "${WORLD_MODEL_QUALITY_RECORD}" in
  none|None|NONE|off|Off|OFF|false|False|FALSE|disabled|Disabled|DISABLED)
    WORLD_MODEL_QUALITY_RECORD=""
    ;;
esac
# Keep trial filtering independent from the rollout backend.
# Default to EVAC so sim replays can reproduce the original EVAC-filtered trial
# schedule unless explicitly overridden.
WORLD_MODEL_QUALITY_BACKEND=${WORLD_MODEL_QUALITY_BACKEND:-evac}
COSMOS_EXECUTION_MODE=${COSMOS_EXECUTION_MODE:-direct}
COSMOS_ROOT=${COSMOS_ROOT:-${SCRIPT_DIR}/cosmos-predict2.5}
COSMOS_TARGET_ROOT=${COSMOS_TARGET_ROOT:-/data/zhenyangfan/cosmos-predict2.5}
COSMOS_PYTHON_BIN=${COSMOS_PYTHON_BIN:-${COSMOS_TARGET_ROOT}/.venv/bin/python}
COSMOS_CHECKPOINT_PATH=${COSMOS_CHECKPOINT_PATH:-${COSMOS_TARGET_ROOT}/outputs/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_robotwin_dualarm_action_conditioned_fullft_lr1e4_actioncond_fullft_gripperfix_4gpu_bsz8_openlaptop3x_from_iter10000_nooptim_20260502_171242/checkpoints/iter_000010000_pt/model_ema_bf16.pt}
COSMOS_EXPERIMENT=${COSMOS_EXPERIMENT:-robotwin_dualarm_actioncond_2b_256_320}
COSMOS_CONFIG_FILE=${COSMOS_CONFIG_FILE:-cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py}
COSMOS_CUDA_VISIBLE_DEVICES=${COSMOS_CUDA_VISIBLE_DEVICES:-}
COSMOS_CONTEXT_PARALLEL_SIZE=${COSMOS_CONTEXT_PARALLEL_SIZE:-1}
COSMOS_CHUNK_SIZE=${COSMOS_CHUNK_SIZE:-12}
COSMOS_GUIDANCE=${COSMOS_GUIDANCE:-7}
COSMOS_RESOLUTION=${COSMOS_RESOLUTION:-256,320}
COSMOS_FPS_DOWNSAMPLE_RATIO=${COSMOS_FPS_DOWNSAMPLE_RATIO:-1}
COSMOS_GRIPPER_SCALE=${COSMOS_GRIPPER_SCALE:-1.0}
COSMOS_INVERT_GRIPPER=${COSMOS_INVERT_GRIPPER:-true}
COSMOS_NUM_STEPS=${COSMOS_NUM_STEPS:-35}
COSMOS_SAVE_FPS=${COSMOS_SAVE_FPS:-30}
COSMOS_NUM_LATENT_CONDITIONAL_FRAMES=${COSMOS_NUM_LATENT_CONDITIONAL_FRAMES:-1}
COSMOS_ACTION_SCALER=${COSMOS_ACTION_SCALER:-20.0}
COSMOS_ACTION_STATS_PATH=${COSMOS_ACTION_STATS_PATH:-}
COSMOS_ACTION_NORMALIZATION_CLIP=${COSMOS_ACTION_NORMALIZATION_CLIP:-}
COSMOS_USE_QUAT=${COSMOS_USE_QUAT:-false}
# The current RoboTwin Cosmos checkpoint was trained with the converter that
# read endpose quaternions as xyzw. Keep online FK action conditioning aligned.
COSMOS_QUAT_INPUT_ORDER=${COSMOS_QUAT_INPUT_ORDER:-wxyz}
COSMOS_PROMPT=${COSMOS_PROMPT:-}
COSMOS_NEGATIVE_PROMPT=${COSMOS_NEGATIVE_PROMPT:-}
COSMOS_SEED=${COSMOS_SEED:-0}
COSMOS_WORK_DIR=${COSMOS_WORK_DIR:-}
COSMOS_STARTUP_TIMEOUT_S=${COSMOS_STARTUP_TIMEOUT_S:-600}
COSMOS_REQUEST_TIMEOUT_S=${COSMOS_REQUEST_TIMEOUT_S:-900}
WORLD_MODEL_COMPARE_MODE=${WORLD_MODEL_COMPARE_MODE:-false}
WORLD_MODEL_COMPARE_BACKENDS=(${WORLD_MODEL_COMPARE_BACKENDS:-evac cosmos})
WORLD_MODEL_COMPARE_SIM=${WORLD_MODEL_COMPARE_SIM:-true}
WORLD_MODEL_COMPARE_SIM_AUTORUN=${WORLD_MODEL_COMPARE_SIM_AUTORUN:-false}
WORLD_MODEL_COMPARE_SIM_TIMEOUT_S=${WORLD_MODEL_COMPARE_SIM_TIMEOUT_S:-600}
WORLD_MODEL_COMPARE_COSMOS_AUTORUN=${WORLD_MODEL_COMPARE_COSMOS_AUTORUN:-false}
CORR_EXPORT_DATASET=${CORR_EXPORT_DATASET:-true}
CORR_EXPORT_ROOT=${CORR_EXPORT_ROOT:-/data/zhenyangfan/RoboTwin/data}
CORR_EXPORT_TASK_CONFIG=${CORR_EXPORT_TASK_CONFIG:-demo_clean_corr_export}
CORR_EXPORT_FORMAT=${CORR_EXPORT_FORMAT:-raw_episode}
CORR_EXPORT_MAX_LOOPS=${CORR_EXPORT_MAX_LOOPS:-1}
CORR_EXPORT_DEBUG_VIDEO=${CORR_EXPORT_DEBUG_VIDEO:-true}
CORR_EXPORT_FPS=${CORR_EXPORT_FPS:-30}
EXPLORE_TRIAL_SOURCE_DIR=${EXPLORE_TRIAL_SOURCE_DIR:-}
EXPLORE_RESUME_FROM_LIVE_DIR=${EXPLORE_RESUME_FROM_LIVE_DIR:-}
EXPLORE_DISABLE_DISTRIBUTED_PROGRESS_SYNC=${EXPLORE_DISABLE_DISTRIBUTED_PROGRESS_SYNC:-}
EXPLORE_DEBUG_MAX_SAMPLES=${EXPLORE_DEBUG_MAX_SAMPLES:-0}
URDF_PATH=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
CUROBO_LEFT_YML=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
CUROBO_RIGHT_YML=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
DEVICE=${DEVICE:-cuda}
GPU_IDS=${GPU_IDS:-4,5,6,7}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}
SEED=${SEED:-100}
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
FAILURE_PHASE_BINS=${FAILURE_PHASE_BINS:-4}
FAILURE_TRANSLATION_DIR_BINS=${FAILURE_TRANSLATION_DIR_BINS:-6}
FAILURE_TRANSLATION_MAG_BINS=${FAILURE_TRANSLATION_MAG_BINS:-1}
FAILURE_ROTATION_DIR_BINS=${FAILURE_ROTATION_DIR_BINS:-6}
FAILURE_ROTATION_MAG_BINS=${FAILURE_ROTATION_MAG_BINS:-1}
FAILURE_EXPLORE_K=${FAILURE_EXPLORE_K:-4}
EXPLORE_PHASE_KEYS=(
  transport
  pregrasp
  approach
  place
)
if [ -n "${EXPLORE_PHASE_KEYS_OVERRIDE:-}" ]; then
  read -r -a EXPLORE_PHASE_KEYS <<< "${EXPLORE_PHASE_KEYS_OVERRIDE}"
fi
EXPLORE_PHASE_INSTANCE_IDXS=()
if [ -n "${EXPLORE_PHASE_INSTANCE_IDXS_OVERRIDE:-}" ]; then
  read -r -a EXPLORE_PHASE_INSTANCE_IDXS <<< "${EXPLORE_PHASE_INSTANCE_IDXS_OVERRIDE}"
fi
EXPLORE_ERROR_MODES=(
  gripper_close
  rotation
  translation
)
if [ -n "${EXPLORE_ERROR_MODES_OVERRIDE:-}" ]; then
  read -r -a EXPLORE_ERROR_MODES <<< "${EXPLORE_ERROR_MODES_OVERRIDE}"
fi
# Demo-direction defaults: translation left/right (local +/-Y) and planar yaw
# rotation around local +/-Z. Override with empty string arrays from the env if
# a full direction sweep is needed.
EXPLORE_TRANSLATION_DIR_BINS=(2 3)
if [ -n "${EXPLORE_TRANSLATION_DIR_BINS_OVERRIDE:-}" ]; then
  read -r -a EXPLORE_TRANSLATION_DIR_BINS <<< "${EXPLORE_TRANSLATION_DIR_BINS_OVERRIDE}"
fi
EXPLORE_ROTATION_DIR_BINS=(2 3)
if [ -n "${EXPLORE_ROTATION_DIR_BINS_OVERRIDE:-}" ]; then
  read -r -a EXPLORE_ROTATION_DIR_BINS <<< "${EXPLORE_ROTATION_DIR_BINS_OVERRIDE}"
fi
EXPLORE_DISABLE_PHASE_BIN_SKIP=${EXPLORE_DISABLE_PHASE_BIN_SKIP:-false}
EXPLORE_SKIP_OPEN_LAPTOP_TRANSPORT=${EXPLORE_SKIP_OPEN_LAPTOP_TRANSPORT:-false}
ACT_ALIGNED_ROLLOUT_EXEC_STEPS=${ACT_ALIGNED_ROLLOUT_EXEC_STEPS:-12}
PLANNER_ORIENT_WEIGHT=0.0573
PLANNER_GRIPPER_PENALTY=1.0
PLANNER_NEAREST_WINDOW_RADIUS=12
PLANNER_ACTIVE_JOINT_DELTA_THRESH=0.01
PLANNER_ACTIVE_GRIPPER_DELTA_THRESH=0.05
RECOVER_EVAL_SAVE_VIDEO=${RECOVER_EVAL_SAVE_VIDEO:-true}
SAVE_PERTURB_ROLLOUT_VIDEO=${SAVE_PERTURB_ROLLOUT_VIDEO:-true}
SAVE_CORRECTION_DEBUG=${SAVE_CORRECTION_DEBUG:-true}
RECOVER_EVAL_GRIPPER_OPEN_THRESH=${RECOVER_EVAL_GRIPPER_OPEN_THRESH:-0.4}
RECOVER_EVAL_POS_THRESH_M=0.04
RECOVER_EVAL_ROT_THRESH_DEG=8.0
RECOVER_EVAL_NEAREST_WINDOW_RADIUS=16
RECOVER_EVAL_VIDEO_BRIDGE_STEPS=16
ACT_ALIGNED_ENABLE_PERTURB=true
ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN=${ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN:-0.08}
ACT_ALIGNED_PERTURB_ROT_MAX_DEG=${ACT_ALIGNED_PERTURB_ROT_MAX_DEG:-25.0}
ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN=${ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN:-0.02}
EVAC_BLUR_FILTER_ENABLE=${EVAC_BLUR_FILTER_ENABLE:-false}
EVAC_BLUR_FILTER_METRIC=${EVAC_BLUR_FILTER_METRIC:-sharpness_ratio}
EVAC_BLUR_FILTER_MIN_RATIO=${EVAC_BLUR_FILTER_MIN_RATIO:-0.75}
EVAC_BLUR_FILTER_REGION=${EVAC_BLUR_FILTER_REGION:-active_gripper_patch}
EVAC_BLUR_FILTER_PATCH_PAD_PX=${EVAC_BLUR_FILTER_PATCH_PAD_PX:-12}
EVAC_BLUR_FILTER_GRIPPER_AXIS_M=${EVAC_BLUR_FILTER_GRIPPER_AXIS_M:-0.04}
EVAC_USE_DUAL_CACHE=${EVAC_USE_DUAL_CACHE:-true}
# bounds=(4 8 12 14): refresh v every few steps, then run full after step 14.
EVAC_DC_V_BOUNDS=(${EVAC_DC_V_BOUNDS:-4 8 12 14})
EVAC_DC_BUDGET=${EVAC_DC_BUDGET:--1}
EVAC_DC_ENC_START=${EVAC_DC_ENC_START:-999}
EVAC_DC_REPLAY_STEP_NOISE=${EVAC_DC_REPLAY_STEP_NOISE:-false}
EVAC_DC_HF_METRIC=${EVAC_DC_HF_METRIC:-false}
EVAC_DC_V_BLUR_ON_REUSE=${EVAC_DC_V_BLUR_ON_REUSE:-false}
EVAC_DC_V_BLUR_KERNEL=${EVAC_DC_V_BLUR_KERNEL:-3}
EVAC_DC_V_BLUR_STRENGTH=${EVAC_DC_V_BLUR_STRENGTH:-0.15}
PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false
SMOLVLA_EVAC_PRINT_RUNTIME=false
TORCHVISION_VIDEO_WARNING_FILTER="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning:torchvision.io._video_deprecation_warning"
PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${TORCHVISION_VIDEO_WARNING_FILTER}"
PYTHONPATH="/data/zhenyangfan/RoboTwin:${LOCAL_SRC_DIR}"
HF_HOME=${HF_HOME:-${SCRIPT_DIR}/.hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}

if [ -L "${SCRIPT_DIR}/cosmos-predict2.5" ]; then
  :
elif [ ! -e "${SCRIPT_DIR}/cosmos-predict2.5" ]; then
  ln -s "${COSMOS_TARGET_ROOT}" "${SCRIPT_DIR}/cosmos-predict2.5"
else
  echo "Cosmos link path exists but is not a symlink: ${SCRIPT_DIR}/cosmos-predict2.5" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_NAME="${TIMESTAMP}"
if [ -n "${RUN_TAG}" ]; then
  BASE_NAME="${BASE_NAME}-${RUN_TAG}"
fi
if [ -n "${EXPLORE_OUTPUT_DIR_EXACT:-}" ]; then
  OUTPUT_DIR="${EXPLORE_OUTPUT_DIR_EXACT}"
else
  OUTPUT_DIR="${OUTPUT_DIR}/${BASE_NAME}"
fi

if [ -f "${SMOLVLA_PRETRAINED_PATH}" ]; then
  STAGE1_CKPT="${SMOLVLA_PRETRAINED_PATH}"
  SMOLVLA_PRETRAINED_PATH=$(python3 - "${STAGE1_CKPT}" <<'PY'
import sys
import torch

ckpt_path = sys.argv[1]
payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
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
if [ "${WORLD_MODEL_BACKEND}" = "cosmos" ] || [ "${WORLD_MODEL_COMPARE_COSMOS_AUTORUN}" = "true" ]; then
  if [ ! -d "${COSMOS_TARGET_ROOT}" ]; then
    echo "Cosmos root is missing: ${COSMOS_TARGET_ROOT}" >&2
    exit 1
  fi
  if [ ! -d "${COSMOS_ROOT}" ]; then
    echo "Cosmos root/link is missing: ${COSMOS_ROOT}" >&2
    exit 1
  fi
  if [ "${COSMOS_EXECUTION_MODE}" = "worker" ] && [ ! -x "${COSMOS_PYTHON_BIN}" ]; then
    echo "Cosmos python is missing or not executable: ${COSMOS_PYTHON_BIN}" >&2
    exit 1
  fi
  if [ "${COSMOS_EXECUTION_MODE}" != "worker" ] && [ "${COSMOS_EXECUTION_MODE}" != "direct" ]; then
    echo "Unsupported COSMOS_EXECUTION_MODE=${COSMOS_EXECUTION_MODE}; expected worker or direct" >&2
    exit 1
  fi
  if [ ! -e "${COSMOS_CHECKPOINT_PATH}" ]; then
    echo "Cosmos checkpoint path is missing: ${COSMOS_CHECKPOINT_PATH}" >&2
    exit 1
  fi
elif [ "${WORLD_MODEL_BACKEND}" != "evac" ] && [ "${WORLD_MODEL_BACKEND}" != "sim" ]; then
  echo "Unsupported WORLD_MODEL_BACKEND=${WORLD_MODEL_BACKEND}; expected evac, sim, or cosmos" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"
cp "${SCRIPT_PATH}" "${OUTPUT_DIR}/explore_smolvla.sh"

export PYTHONNOUSERSITE
export PYTHONPATH
export TOKENIZERS_PARALLELISM
export SMOLVLA_EVAC_PRINT_RUNTIME
export PYTHONWARNINGS
export SMOLVLA_LATENT_DATA_SOURCE
export SMOLVLA_LATENT_DATA_ROOT
export SMOLVLA_LATENT_DATA_SUFFIX
export HF_HOME
export HF_DATASETS_CACHE
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
    --latent_dataset_source "${SMOLVLA_LATENT_DATA_SOURCE}" \
    --latent_dataset_root "${SMOLVLA_LATENT_DATA_ROOT}" \
    --latent_dataset_suffix "${SMOLVLA_LATENT_DATA_SUFFIX}" \
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
    --explore_phase_keys "${EXPLORE_PHASE_KEYS[@]}" \
    --explore_phase_instance_idxs "${EXPLORE_PHASE_INSTANCE_IDXS[@]}" \
    --explore_error_modes "${EXPLORE_ERROR_MODES[@]}" \
    --explore_translation_dir_bins "${EXPLORE_TRANSLATION_DIR_BINS[@]}" \
    --explore_rotation_dir_bins "${EXPLORE_ROTATION_DIR_BINS[@]}" \
    --explore_disable_phase_bin_skip "${EXPLORE_DISABLE_PHASE_BIN_SKIP}" \
    --explore_skip_open_laptop_transport "${EXPLORE_SKIP_OPEN_LAPTOP_TRANSPORT}" \
    --act_aligned_rollout_exec_steps "${ACT_ALIGNED_ROLLOUT_EXEC_STEPS}" \
    --planner_orient_weight "${PLANNER_ORIENT_WEIGHT}" \
    --planner_gripper_penalty "${PLANNER_GRIPPER_PENALTY}" \
    --planner_nearest_window_radius "${PLANNER_NEAREST_WINDOW_RADIUS}" \
    --planner_active_joint_delta_thresh "${PLANNER_ACTIVE_JOINT_DELTA_THRESH}" \
    --planner_active_gripper_delta_thresh "${PLANNER_ACTIVE_GRIPPER_DELTA_THRESH}" \
    --recover_eval_save_video "${RECOVER_EVAL_SAVE_VIDEO}" \
    --save_perturb_rollout_video "${SAVE_PERTURB_ROLLOUT_VIDEO}" \
    --save_correction_debug "${SAVE_CORRECTION_DEBUG}" \
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
    --evac_blur_filter_metric "${EVAC_BLUR_FILTER_METRIC}" \
    --evac_blur_filter_min_ratio "${EVAC_BLUR_FILTER_MIN_RATIO}" \
    --evac_blur_filter_region "${EVAC_BLUR_FILTER_REGION}" \
    --evac_blur_filter_patch_pad_px "${EVAC_BLUR_FILTER_PATCH_PAD_PX}" \
    --evac_blur_filter_gripper_axis_m "${EVAC_BLUR_FILTER_GRIPPER_AXIS_M}" \
    --evac_use_dual_cache "${EVAC_USE_DUAL_CACHE}" \
    --evac_dc_v_bounds "${EVAC_DC_V_BOUNDS[@]}" \
    --evac_dc_budget "${EVAC_DC_BUDGET}" \
    --evac_dc_enc_start "${EVAC_DC_ENC_START}" \
    --evac_dc_replay_step_noise "${EVAC_DC_REPLAY_STEP_NOISE}" \
    --evac_dc_hf_metric "${EVAC_DC_HF_METRIC}" \
    --evac_dc_v_blur_on_reuse "${EVAC_DC_V_BLUR_ON_REUSE}" \
    --evac_dc_v_blur_kernel "${EVAC_DC_V_BLUR_KERNEL}" \
    --evac_dc_v_blur_strength "${EVAC_DC_V_BLUR_STRENGTH}" \
    --world_model_backend "${WORLD_MODEL_BACKEND}" \
    --sim_use_subprocess "${SIM_USE_SUBPROCESS}" \
    --sim_timeout_s "${SIM_TIMEOUT_S}" \
    --world_model_quality_record "${WORLD_MODEL_QUALITY_RECORD}" \
    --world_model_quality_backend "${WORLD_MODEL_QUALITY_BACKEND}" \
    --cosmos_execution_mode "${COSMOS_EXECUTION_MODE}" \
    --cosmos_root "${COSMOS_ROOT}" \
    --cosmos_python_bin "${COSMOS_PYTHON_BIN}" \
    --cosmos_checkpoint_path "${COSMOS_CHECKPOINT_PATH}" \
    --cosmos_experiment "${COSMOS_EXPERIMENT}" \
    --cosmos_config_file "${COSMOS_CONFIG_FILE}" \
    --cosmos_cuda_visible_devices "${COSMOS_CUDA_VISIBLE_DEVICES}" \
    --cosmos_context_parallel_size "${COSMOS_CONTEXT_PARALLEL_SIZE}" \
    --cosmos_chunk_size "${COSMOS_CHUNK_SIZE}" \
    --cosmos_guidance "${COSMOS_GUIDANCE}" \
    --cosmos_resolution "${COSMOS_RESOLUTION}" \
    --cosmos_fps_downsample_ratio "${COSMOS_FPS_DOWNSAMPLE_RATIO}" \
    --cosmos_gripper_scale "${COSMOS_GRIPPER_SCALE}" \
    --cosmos_invert_gripper "${COSMOS_INVERT_GRIPPER}" \
    --cosmos_num_steps "${COSMOS_NUM_STEPS}" \
    --cosmos_save_fps "${COSMOS_SAVE_FPS}" \
    --cosmos_num_latent_conditional_frames "${COSMOS_NUM_LATENT_CONDITIONAL_FRAMES}" \
    --cosmos_action_scaler "${COSMOS_ACTION_SCALER}" \
    --cosmos_action_stats_path "${COSMOS_ACTION_STATS_PATH}" \
    --cosmos_action_normalization_clip "${COSMOS_ACTION_NORMALIZATION_CLIP}" \
    --cosmos_use_quat "${COSMOS_USE_QUAT}" \
    --cosmos_quat_input_order "${COSMOS_QUAT_INPUT_ORDER}" \
    --cosmos_prompt "${COSMOS_PROMPT}" \
    --cosmos_negative_prompt "${COSMOS_NEGATIVE_PROMPT}" \
    --cosmos_seed "${COSMOS_SEED}" \
    --cosmos_work_dir "${COSMOS_WORK_DIR}" \
    --cosmos_startup_timeout_s "${COSMOS_STARTUP_TIMEOUT_S}" \
    --cosmos_request_timeout_s "${COSMOS_REQUEST_TIMEOUT_S}" \
    --world_model_compare_mode "${WORLD_MODEL_COMPARE_MODE}" \
    --world_model_compare_backends "${WORLD_MODEL_COMPARE_BACKENDS[@]}" \
    --world_model_compare_sim "${WORLD_MODEL_COMPARE_SIM}" \
    --world_model_compare_sim_autorun "${WORLD_MODEL_COMPARE_SIM_AUTORUN}" \
    --world_model_compare_sim_timeout_s "${WORLD_MODEL_COMPARE_SIM_TIMEOUT_S}" \
    --world_model_compare_cosmos_autorun "${WORLD_MODEL_COMPARE_COSMOS_AUTORUN}" \
    --corr_export_dataset "${CORR_EXPORT_DATASET}" \
    --corr_export_root "${CORR_EXPORT_ROOT}" \
    --corr_export_task_config "${CORR_EXPORT_TASK_CONFIG}" \
    --corr_export_format "${CORR_EXPORT_FORMAT}" \
    --corr_export_max_loops "${CORR_EXPORT_MAX_LOOPS}" \
    --corr_export_debug_video "${CORR_EXPORT_DEBUG_VIDEO}" \
    --corr_export_fps "${CORR_EXPORT_FPS}" \
    --explore_trial_source_dir "${EXPLORE_TRIAL_SOURCE_DIR}" \
    --resume_from_live_dir "${EXPLORE_RESUME_FROM_LIVE_DIR}" \
    --debug_max_samples_per_rank "${EXPLORE_DEBUG_MAX_SAMPLES}"
)

if [ -n "${STAGE1_CKPT}" ]; then
  CMD+=(--stage1_ckpt "${STAGE1_CKPT}")
fi
if [ -n "${EXPLORE_DISABLE_DISTRIBUTED_PROGRESS_SYNC}" ]; then
  CMD+=(--explore_disable_distributed_progress_sync "${EXPLORE_DISABLE_DISTRIBUTED_PROGRESS_SYNC}")
fi

echo "Output dir: ${OUTPUT_DIR}"
echo "GPU ids: ${GPU_IDS}"
echo "Run tag: ${RUN_TAG}"
echo "Conda env: ${SMOLVLA_CONDA_ENV}"
echo "Latent dataset source: ${SMOLVLA_LATENT_DATA_SOURCE}"
echo "Latent dataset root: ${SMOLVLA_LATENT_DATA_ROOT}"
echo "Latent dataset suffix: ${SMOLVLA_LATENT_DATA_SUFFIX}"
echo "World model backend: ${WORLD_MODEL_BACKEND}"
echo "World model quality record: ${WORLD_MODEL_QUALITY_RECORD:-<disabled>}"
echo "World model quality backend: ${WORLD_MODEL_QUALITY_BACKEND}"
echo "World model compare mode: ${WORLD_MODEL_COMPARE_MODE}"
echo "Correction export dataset: ${CORR_EXPORT_DATASET}"
if [ "${CORR_EXPORT_DATASET}" = "true" ]; then
  echo "Correction export root: ${CORR_EXPORT_ROOT}"
  echo "Correction export task config: ${CORR_EXPORT_TASK_CONFIG}"
  echo "Correction export format: ${CORR_EXPORT_FORMAT}"
  echo "Correction export max loops: ${CORR_EXPORT_MAX_LOOPS}"
fi
if [ -n "${EXPLORE_RESUME_FROM_LIVE_DIR}" ]; then
  echo "Explore resume from live dir: ${EXPLORE_RESUME_FROM_LIVE_DIR}"
fi
if [ -n "${EXPLORE_TRIAL_SOURCE_DIR}" ]; then
  echo "Explore trial source dir: ${EXPLORE_TRIAL_SOURCE_DIR}"
fi
if [ "${WORLD_MODEL_COMPARE_MODE}" = "true" ]; then
  echo "World model compare backends: ${WORLD_MODEL_COMPARE_BACKENDS[*]}"
  echo "World model compare sim: ${WORLD_MODEL_COMPARE_SIM} autorun=${WORLD_MODEL_COMPARE_SIM_AUTORUN}"
  echo "World model compare cosmos autorun: ${WORLD_MODEL_COMPARE_COSMOS_AUTORUN}"
fi
if [ "${EXPLORE_DEBUG_MAX_SAMPLES}" != "0" ]; then
  echo "Explore debug max samples per rank: ${EXPLORE_DEBUG_MAX_SAMPLES}"
fi
if [ "${WORLD_MODEL_BACKEND}" = "cosmos" ] || [ "${WORLD_MODEL_COMPARE_COSMOS_AUTORUN}" = "true" ]; then
  echo "Cosmos execution mode: ${COSMOS_EXECUTION_MODE}"
  echo "Cosmos checkpoint: ${COSMOS_CHECKPOINT_PATH}"
fi
echo "Running: CUDA_VISIBLE_DEVICES=${GPU_IDS} ${CMD[*]}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
