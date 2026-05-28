#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

PYTHON_BIN=${PYTHON_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/torchrun}
if [ ! -x "${TORCHRUN_BIN}" ]; then
  echo "torchrun not found: ${TORCHRUN_BIN}" >&2
  exit 1
fi
VENV_BIN_DIR=$(dirname "${PYTHON_BIN}")
export PATH="${VENV_BIN_DIR}:${PATH}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export PYTHONPATH="/data/zhenyangfan/RoboTwin:/data/zhenyangfan/RoboTwin/envs/curobo/src:/data/zhenyangfan/RoboTwin/envs/robot:${PYTHONPATH:-}"
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/zhenyangfan/runtime_cache}
mkdir -p \
  "${RUNTIME_ROOT}/tmp" \
  "${RUNTIME_ROOT}/torch_extensions" \
  "${RUNTIME_ROOT}/mplconfig" \
  "${RUNTIME_ROOT}/hf_home" \
  "${RUNTIME_ROOT}/wandb" \
  "${RUNTIME_ROOT}/xdg_cache" \
  "${RUNTIME_ROOT}/xdg_config"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_ROOT}/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_ROOT}/mplconfig}"
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${RUNTIME_ROOT}/wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${RUNTIME_ROOT}/wandb/config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${RUNTIME_ROOT}/xdg_config}"

OUTPUT_ROOT=${OUTPUT_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${TIMESTAMP}"}
mkdir -p "${OUTPUT_DIR}"

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29731}

ACT_INIT_CKPT=${ACT_INIT_CKPT:-}
RESUME_CKPT=${RESUME_CKPT:-}
if [ -z "${ACT_INIT_CKPT}" ] && [ -z "${RESUME_CKPT}" ]; then
  echo "Either ACT_INIT_CKPT or RESUME_CKPT is required" >&2
  exit 1
fi

MULTI_TASK_NAMES=${MULTI_TASK_NAMES:-"sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50"}
FAILURE_TABLE_PATHS=${FAILURE_TABLE_PATHS:-}
if [ -z "${FAILURE_TABLE_PATHS}" ]; then
  echo "FAILURE_TABLE_PATHS is required. Pass one shared table or one table per task." >&2
  exit 1
fi

EVAC_CKPT=${EVAC_CKPT:-/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt}
EVAC_CONFIG=${EVAC_CONFIG:-/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml}
URDF_PATH=${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}
CUROBO_LEFT_YML=${CUROBO_LEFT_YML:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml}
CUROBO_RIGHT_YML=${CUROBO_RIGHT_YML:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml}

NUM_EPOCHS=${NUM_EPOCHS:-1000}
MAX_STEPS=${MAX_STEPS:--1}
NORMAL_BATCH_SIZE=${NORMAL_BATCH_SIZE:-4}
FAILURE_BATCH_SIZE=${FAILURE_BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-2}
PREFIX_STEPS=${PREFIX_STEPS:-16}
ACT_CHUNK_SIZE=${ACT_CHUNK_SIZE:-50}
FUTURE_OFFSET=${FUTURE_OFFSET:-16}
SAVE_FREQ=${SAVE_FREQ:-50}
LR=${LR:-3e-5}
BASE_ACT_LR_SCALE=${BASE_ACT_LR_SCALE:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
DDIM_STEPS=${DDIM_STEPS:-27}

LAMBDA_ACTION=${LAMBDA_ACTION:-1.0}
NORMAL_CONDITION_KEEP_PROB=${NORMAL_CONDITION_KEEP_PROB:-1.0}
BETA_DYNAMICS_MAX=${BETA_DYNAMICS_MAX:-1.0}
LAMBDA_WM_ACTION_CURRENT=${LAMBDA_WM_ACTION_CURRENT:-0.0}
LAMBDA_WM_ACTION_FUTURE=${LAMBDA_WM_ACTION_FUTURE:-0.0}
LAMBDA_BRIDGE_FUTURE=${LAMBDA_BRIDGE_FUTURE:-0.0}
LAMBDA_TEACHER_MAX=${LAMBDA_TEACHER_MAX:-0.5}
LAMBDA_PRED_MAX=${LAMBDA_PRED_MAX:-0.5}
LAMBDA_LATENT_MAX=${LAMBDA_LATENT_MAX:-0.3}
LAMBDA_TOKEN_INIT=${LAMBDA_TOKEN_INIT:-0.1}
LAMBDA_TOKEN_LATE=${LAMBDA_TOKEN_LATE:-0.02}
LATENT_LOSS_TYPE=${LATENT_LOSS_TYPE:-normalized_mse}
TOKEN_LOSS_TYPE=${TOKEN_LOSS_TYPE:-mse}
TEACHER_DECAY_START_RATIO=${TEACHER_DECAY_START_RATIO:-0.20}
TEACHER_DECAY_END_RATIO=${TEACHER_DECAY_END_RATIO:-0.90}
PRED_WARMUP_START_RATIO=${PRED_WARMUP_START_RATIO:-0.10}
PRED_WARMUP_END_RATIO=${PRED_WARMUP_END_RATIO:-0.70}
LATENT_WARMUP_END_RATIO=${LATENT_WARMUP_END_RATIO:-0.20}
TOKEN_DECAY_START_RATIO=${TOKEN_DECAY_START_RATIO:-0.20}
TOKEN_DECAY_END_RATIO=${TOKEN_DECAY_END_RATIO:-0.90}
PRED_ONLY_FINETUNE_START_RATIO=${PRED_ONLY_FINETUNE_START_RATIO:-0.90}
STOPGRAD_WM_TEACHER=${STOPGRAD_WM_TEACHER:-true}
STOPGRAD_TEACHER_TOKEN=${STOPGRAD_TEACHER_TOKEN:-true}
STOPGRAD_ADAPTER_INPUT_FOR_TOKEN_LOSS=${STOPGRAD_ADAPTER_INPUT_FOR_TOKEN_LOSS:-false}
DYN_ZERO_STEPS=${DYN_ZERO_STEPS:-0}
DYN_RAMP_STEPS=${DYN_RAMP_STEPS:-1000}
DYN_WARMUP_CURVE=${DYN_WARMUP_CURVE:-cosine}
REFERENCE_GLOBAL_BATCH_SIZE=${REFERENCE_GLOBAL_BATCH_SIZE:-6}

FREEZE_BASE_ACT=${FREEZE_BASE_ACT:-false}
FREEZE_READOUT_DECODER=${FREEZE_READOUT_DECODER:-true}
DETACH_ACT_FEATURE_FOR_LATENT=${DETACH_ACT_FEATURE_FOR_LATENT:-false}
USE_RAW_WM_TARGETS=${USE_RAW_WM_TARGETS:-false}

PROJECTOR_MID_CHANNELS=${PROJECTOR_MID_CHANNELS:-256}
WM_ADAPTER_MID_CHANNELS=${WM_ADAPTER_MID_CHANNELS:-128}
READOUT_ADAPTER_MID_CHANNELS=${READOUT_ADAPTER_MID_CHANNELS:-128}
PREDICTOR_NUM_BLOCKS=${PREDICTOR_NUM_BLOCKS:-3}
PREDICTOR_MLP_HIDDEN=${PREDICTOR_MLP_HIDDEN:-512}
ACTION_DECODER_HIDDEN=${ACTION_DECODER_HIDDEN:-512}
TOKEN_ADAPTER_HIDDEN_DIM=${TOKEN_ADAPTER_HIDDEN_DIM:-512}
TOKEN_ADAPTER_NUM_LAYERS=${TOKEN_ADAPTER_NUM_LAYERS:-2}
TOKEN_ADAPTER_DROPOUT=${TOKEN_ADAPTER_DROPOUT:-0.1}
BACKBONE=${BACKBONE:-resnet18}
HIDDEN_DIM=${HIDDEN_DIM:-512}
STATE_DIM=${STATE_DIM:-14}
ACTION_DIM=${ACTION_DIM:-14}

MAX_ROLLOUT_STEPS=${MAX_ROLLOUT_STEPS:-1}
PLANNER_TARGET_MODE=${PLANNER_TARGET_MODE:-backward}
PLANNER_TARGET_LOOKAHEAD_STEPS=${PLANNER_TARGET_LOOKAHEAD_STEPS:-6}
PLANNER_ORIENT_WEIGHT=${PLANNER_ORIENT_WEIGHT:-0.0573}
PLANNER_GRIPPER_PENALTY=${PLANNER_GRIPPER_PENALTY:-1.0}
PLANNER_NEAREST_WINDOW_RADIUS=${PLANNER_NEAREST_WINDOW_RADIUS:-12}
PLANNER_ACTIVE_JOINT_DELTA_THRESH=${PLANNER_ACTIVE_JOINT_DELTA_THRESH:-0.01}
PLANNER_ACTIVE_GRIPPER_DELTA_THRESH=${PLANNER_ACTIVE_GRIPPER_DELTA_THRESH:-0.05}
ACT_ALIGNED_ROLLOUT_EXEC_STEPS=${ACT_ALIGNED_ROLLOUT_EXEC_STEPS:-16}
ACT_ALIGNED_SAMPLE_PREGRASP_PHASE_WINDOW_LEN=${ACT_ALIGNED_SAMPLE_PREGRASP_PHASE_WINDOW_LEN:-30}
ACT_ALIGNED_SAMPLE_TIMEOUT_SEC=${ACT_ALIGNED_SAMPLE_TIMEOUT_SEC:-30}

ACT_ALIGNED_MIN_DIST_FALLBACK_FORCE_CORRECTION=${ACT_ALIGNED_MIN_DIST_FALLBACK_FORCE_CORRECTION:-true}
ACT_ALIGNED_MIN_DIST_RECOVER_RATIO=${ACT_ALIGNED_MIN_DIST_RECOVER_RATIO:-0.75}
ACT_ALIGNED_REAL_ERROR_TRIGGER_ENABLE=${ACT_ALIGNED_REAL_ERROR_TRIGGER_ENABLE:-true}
ACT_ALIGNED_REAL_ERROR_MIN_DIST_THRESH=${ACT_ALIGNED_REAL_ERROR_MIN_DIST_THRESH:-0.01}
ACT_ALIGNED_REAL_ERROR_MIN_DIST_DELTA_THRESH=${ACT_ALIGNED_REAL_ERROR_MIN_DIST_DELTA_THRESH:-0.005}
ACT_ALIGNED_DEBUG_RECOVER_EVAL_ROLLOUT=${ACT_ALIGNED_DEBUG_RECOVER_EVAL_ROLLOUT:-false}
ACT_ALIGNED_DEBUG_CORRECTION_EVAC_ROLLOUT=${ACT_ALIGNED_DEBUG_CORRECTION_EVAC_ROLLOUT:-false}
RECOVER_EVAL_ENABLE=${RECOVER_EVAL_ENABLE:-false}
RECOVER_EVAL_SAVE_VIDEO=${RECOVER_EVAL_SAVE_VIDEO:-false}
RECOVER_EVAL_GRIPPER_OPEN_THRESH=${RECOVER_EVAL_GRIPPER_OPEN_THRESH:-0.8}
RECOVER_EVAL_POS_THRESH_M=${RECOVER_EVAL_POS_THRESH_M:-0.03}
RECOVER_EVAL_ROT_THRESH_DEG=${RECOVER_EVAL_ROT_THRESH_DEG:-10.0}
RECOVER_EVAL_NEAREST_WINDOW_RADIUS=${RECOVER_EVAL_NEAREST_WINDOW_RADIUS:-16}
RECOVER_EVAL_VIDEO_BRIDGE_STEPS=${RECOVER_EVAL_VIDEO_BRIDGE_STEPS:-16}
ACT_ALIGNED_CORRECTION_INTERP_NEAREST_ENABLE=${ACT_ALIGNED_CORRECTION_INTERP_NEAREST_ENABLE:-false}
ACT_ALIGNED_CORRECTION_INTERP_PREFIX_RATIO=${ACT_ALIGNED_CORRECTION_INTERP_PREFIX_RATIO:-0.6}
ACT_ALIGNED_CORRECTION_PLANNER_PREFIX_RATIO=${ACT_ALIGNED_CORRECTION_PLANNER_PREFIX_RATIO:-0.5}
ACT_ALIGNED_CORRECTION_GRIPPER_CLOSE_PREFIX_RATIO=${ACT_ALIGNED_CORRECTION_GRIPPER_CLOSE_PREFIX_RATIO:-0.32}
ACT_ALIGNED_CORRECTION_COMPOSE_GT_TAIL_ENABLE=${ACT_ALIGNED_CORRECTION_COMPOSE_GT_TAIL_ENABLE:-true}
ACT_ALIGNED_CORRECTION_GRIPPER_SWITCH_RATIO=${ACT_ALIGNED_CORRECTION_GRIPPER_SWITCH_RATIO:-0.5}
ACT_ALIGNED_RECOVER_GRIPPER_PENALTY=${ACT_ALIGNED_RECOVER_GRIPPER_PENALTY:-0.0}
ACT_ALIGNED_ENABLE_PERTURB=${ACT_ALIGNED_ENABLE_PERTURB:-true}
ACT_ALIGNED_PERTURB_PROB=${ACT_ALIGNED_PERTURB_PROB:-1.0}
ACT_ALIGNED_PERTURB_ERROR_MODE=${ACT_ALIGNED_PERTURB_ERROR_MODE:-open_laptop_pregrasp}
ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_CLOSE_PROB=${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_CLOSE_PROB:-0.5}
ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_TRANSLATION_PROB=${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_TRANSLATION_PROB:-0.0}
ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_ROTATION_PROB=${ACT_ALIGNED_PERTURB_OPEN_LAPTOP_PREGRASP_ROTATION_PROB:-0.0}
ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN=${ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN:-0.10}
ACT_ALIGNED_PERTURB_ROT_MAX_DEG=${ACT_ALIGNED_PERTURB_ROT_MAX_DEG:-15.0}
ACT_ALIGNED_PERTURB_MAG_RANDOM=${ACT_ALIGNED_PERTURB_MAG_RANDOM:-false}
ACT_ALIGNED_PERTURB_MAG_RAND_MIN=${ACT_ALIGNED_PERTURB_MAG_RAND_MIN:-1.0}
ACT_ALIGNED_PERTURB_MAG_RAND_MAX=${ACT_ALIGNED_PERTURB_MAG_RAND_MAX:-1.4}
ACT_ALIGNED_PERTURB_REJECT_SAMPLING_ENABLE=${ACT_ALIGNED_PERTURB_REJECT_SAMPLING_ENABLE:-true}
ACT_ALIGNED_PERTURB_REJECT_MAX_TRIALS=${ACT_ALIGNED_PERTURB_REJECT_MAX_TRIALS:-4}
ACT_ALIGNED_PERTURB_REJECT_DIR_JITTER_EPS=${ACT_ALIGNED_PERTURB_REJECT_DIR_JITTER_EPS:-0.2}
ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN=${ACT_ALIGNED_PERTURB_GRIPPER_CLOSE_MIN:-0.10}
ACT_ALIGNED_PERTURB_GRIPPER_OPEN_MAX=${ACT_ALIGNED_PERTURB_GRIPPER_OPEN_MAX:-0.90}
ACT_ALIGNED_PERTURB_GRIPPER_FAST_RATIO=${ACT_ALIGNED_PERTURB_GRIPPER_FAST_RATIO:-0.20}

FAILURE_PHASE_BINS=${FAILURE_PHASE_BINS:-3}
FAILURE_TRANSLATION_DIR_BINS=${FAILURE_TRANSLATION_DIR_BINS:-6}
FAILURE_TRANSLATION_MAG_BINS=${FAILURE_TRANSLATION_MAG_BINS:-3}
FAILURE_ROTATION_DIR_BINS=${FAILURE_ROTATION_DIR_BINS:-6}
FAILURE_ROTATION_MAG_BINS=${FAILURE_ROTATION_MAG_BINS:-3}
FAILURE_EXPLORE_K=${FAILURE_EXPLORE_K:-4}
FAILURE_SAMPLE_SKIP_HEAD_RATIO=${FAILURE_SAMPLE_SKIP_HEAD_RATIO:-0.6}
FAILURE_FUTURE_LATENT_MODE=${FAILURE_FUTURE_LATENT_MODE:-rollout}
BALANCE_FAILURE_TASKS=${BALANCE_FAILURE_TASKS:-false}
REBALANCE_FAILURE_GROUPS=${REBALANCE_FAILURE_GROUPS:-false}
FAILURE_PREPARE_OWNER_RANK=${FAILURE_PREPARE_OWNER_RANK:--1}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-false}

USE_WANDB=${USE_WANDB:-false}
WANDB_PROJECT=${WANDB_PROJECT:-RoboTwin_ACT_LatentCorr}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-stage1_unified_failure_${TIMESTAMP}}
WANDB_GROUP=${WANDB_GROUP:-robotwin_multitask_stage1_unified_failure}
WANDB_LOG_MODE=${WANDB_LOG_MODE:-auto}

if [ "${WANDB_LOG_MODE}" = "auto" ]; then
  unset WANDB_MODE
else
  export WANDB_MODE="${WANDB_LOG_MODE}"
fi

TORCH_EXT_ROOT=${TORCH_EXTENSIONS_DIR}
SKIP_CUROBO_PREWARM=${SKIP_CUROBO_PREWARM:-false}

cleanup_stale_curobo_locks() {
  local ext_root="${TORCH_EXT_ROOT}/py310_cu121"
  local stale_files=(
    "${ext_root}/geom_cu/lock"
    "${ext_root}/geom_cu/.ninja_lock"
    "${ext_root}/kinematics_fused_cu/lock"
    "${ext_root}/kinematics_fused_cu/.ninja_lock"
    "${ext_root}/tensor_step_cu/lock"
    "${ext_root}/tensor_step_cu/.ninja_lock"
    "${ext_root}/lbfgs_step_cu/lock"
    "${ext_root}/lbfgs_step_cu/.ninja_lock"
    "${ext_root}/line_search_cu/lock"
    "${ext_root}/line_search_cu/.ninja_lock"
  )
  for file in "${stale_files[@]}"; do
    if [ -f "${file}" ]; then
      rm -f "${file}"
      echo "[curobo-prewarm] removed stale lock: ${file}"
    fi
  done
}

prewarm_curobo_extensions() {
  echo "[curobo-prewarm] starting single-process extension warmup"
  "${PYTHON_BIN}" - <<'PY'
import importlib
mods = [
    "curobo.curobolib.kinematics",
    "curobo.curobolib.geom",
    "curobo.curobolib.tensor_step",
    "curobo.curobolib.opt",
    "curobo.curobolib.ls",
]
for name in mods:
    importlib.import_module(name)
    print(f"[curobo-prewarm] imported {name}")
PY
  echo "[curobo-prewarm] finished"
}

cleanup_stale_curobo_locks
if [ "${SKIP_CUROBO_PREWARM}" != "true" ]; then
  prewarm_curobo_extensions
else
  echo "[curobo-prewarm] skipped by SKIP_CUROBO_PREWARM=true"
fi

read -r -a MULTI_TASK_NAMES_ARR <<< "${MULTI_TASK_NAMES}"
read -r -a FAILURE_TABLE_PATHS_ARR <<< "${FAILURE_TABLE_PATHS}"

TORCHRUN_ARGS=(
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
)
if [ "${NNODES}" -gt 1 ]; then
  TORCHRUN_ARGS+=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
  )
fi

EXTRA_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
  EXTRA_ARGS+=(--resume_ckpt "${RESUME_CKPT}")
fi
if [ -n "${ACT_INIT_CKPT}" ]; then
  EXTRA_ARGS+=(--act_init_ckpt "${ACT_INIT_CKPT}")
fi

"${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" \
  -m policy.ACT_LatentCorr.train_stage1_unified_failure_multitask_latent \
  --output_dir "${OUTPUT_DIR}" \
  --evac_ckpt "${EVAC_CKPT}" \
  --evac_config "${EVAC_CONFIG}" \
  "${EXTRA_ARGS[@]}" \
  --multi_task_names "${MULTI_TASK_NAMES_ARR[@]}" \
  --failure_table_paths "${FAILURE_TABLE_PATHS_ARR[@]}" \
  --urdf_path "${URDF_PATH}" \
  --curobo_left_yml "${CUROBO_LEFT_YML}" \
  --curobo_right_yml "${CUROBO_RIGHT_YML}" \
  --device cuda:0 \
  --num_epochs "${NUM_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --normal_batch_size "${NORMAL_BATCH_SIZE}" \
  --failure_batch_size "${FAILURE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --base_act_lr_scale "${BASE_ACT_LR_SCALE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --save_freq "${SAVE_FREQ}" \
  --ddim_steps "${DDIM_STEPS}" \
  --prefix_steps "${PREFIX_STEPS}" \
  --act_chunk_size "${ACT_CHUNK_SIZE}" \
  --future_offset "${FUTURE_OFFSET}" \
  --lambda_action "${LAMBDA_ACTION}" \
  --normal_condition_keep_prob "${NORMAL_CONDITION_KEEP_PROB}" \
  --beta_dynamics_max "${BETA_DYNAMICS_MAX}" \
  --lambda_wm_action_current "${LAMBDA_WM_ACTION_CURRENT}" \
  --lambda_wm_action_future "${LAMBDA_WM_ACTION_FUTURE}" \
  --lambda_bridge_future "${LAMBDA_BRIDGE_FUTURE}" \
  --lambda_teacher_max "${LAMBDA_TEACHER_MAX}" \
  --lambda_pred_max "${LAMBDA_PRED_MAX}" \
  --lambda_latent_max "${LAMBDA_LATENT_MAX}" \
  --lambda_token_init "${LAMBDA_TOKEN_INIT}" \
  --lambda_token_late "${LAMBDA_TOKEN_LATE}" \
  --latent_loss_type "${LATENT_LOSS_TYPE}" \
  --token_loss_type "${TOKEN_LOSS_TYPE}" \
  --teacher_decay_start_ratio "${TEACHER_DECAY_START_RATIO}" \
  --teacher_decay_end_ratio "${TEACHER_DECAY_END_RATIO}" \
  --pred_warmup_start_ratio "${PRED_WARMUP_START_RATIO}" \
  --pred_warmup_end_ratio "${PRED_WARMUP_END_RATIO}" \
  --latent_warmup_end_ratio "${LATENT_WARMUP_END_RATIO}" \
  --token_decay_start_ratio "${TOKEN_DECAY_START_RATIO}" \
  --token_decay_end_ratio "${TOKEN_DECAY_END_RATIO}" \
  --pred_only_finetune_start_ratio "${PRED_ONLY_FINETUNE_START_RATIO}" \
  --stopgrad_wm_teacher "${STOPGRAD_WM_TEACHER}" \
  --stopgrad_teacher_token "${STOPGRAD_TEACHER_TOKEN}" \
  --stopgrad_adapter_input_for_token_loss "${STOPGRAD_ADAPTER_INPUT_FOR_TOKEN_LOSS}" \
  --freeze_base_act "${FREEZE_BASE_ACT}" \
  --freeze_readout_decoder "${FREEZE_READOUT_DECODER}" \
  --detach_act_feature_for_latent "${DETACH_ACT_FEATURE_FOR_LATENT}" \
  --use_raw_wm_targets "${USE_RAW_WM_TARGETS}" \
  --dyn_zero_steps "${DYN_ZERO_STEPS}" \
  --dyn_ramp_steps "${DYN_RAMP_STEPS}" \
  --dyn_warmup_curve "${DYN_WARMUP_CURVE}" \
  --reference_global_batch_size "${REFERENCE_GLOBAL_BATCH_SIZE}" \
  --predictor_num_blocks "${PREDICTOR_NUM_BLOCKS}" \
  --projector_mid_channels "${PROJECTOR_MID_CHANNELS}" \
  --wm_adapter_mid_channels "${WM_ADAPTER_MID_CHANNELS}" \
  --readout_adapter_mid_channels "${READOUT_ADAPTER_MID_CHANNELS}" \
  --predictor_mlp_hidden "${PREDICTOR_MLP_HIDDEN}" \
  --action_decoder_hidden "${ACTION_DECODER_HIDDEN}" \
  --token_adapter_hidden_dim "${TOKEN_ADAPTER_HIDDEN_DIM}" \
  --token_adapter_num_layers "${TOKEN_ADAPTER_NUM_LAYERS}" \
  --token_adapter_dropout "${TOKEN_ADAPTER_DROPOUT}" \
  --backbone "${BACKBONE}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --state_dim "${STATE_DIM}" \
  --action_dim "${ACTION_DIM}" \
  --max_rollout_steps "${MAX_ROLLOUT_STEPS}" \
  --planner_target_mode "${PLANNER_TARGET_MODE}" \
  --planner_target_lookahead_steps "${PLANNER_TARGET_LOOKAHEAD_STEPS}" \
  --planner_orient_weight "${PLANNER_ORIENT_WEIGHT}" \
  --planner_gripper_penalty "${PLANNER_GRIPPER_PENALTY}" \
  --planner_nearest_window_radius "${PLANNER_NEAREST_WINDOW_RADIUS}" \
  --planner_active_joint_delta_thresh "${PLANNER_ACTIVE_JOINT_DELTA_THRESH}" \
  --planner_active_gripper_delta_thresh "${PLANNER_ACTIVE_GRIPPER_DELTA_THRESH}" \
  --act_aligned_rollout_exec_steps "${ACT_ALIGNED_ROLLOUT_EXEC_STEPS}" \
  --act_aligned_min_dist_fallback_force_correction "${ACT_ALIGNED_MIN_DIST_FALLBACK_FORCE_CORRECTION}" \
  --act_aligned_min_dist_recover_ratio "${ACT_ALIGNED_MIN_DIST_RECOVER_RATIO}" \
  --act_aligned_real_error_trigger_enable "${ACT_ALIGNED_REAL_ERROR_TRIGGER_ENABLE}" \
  --act_aligned_real_error_min_dist_thresh "${ACT_ALIGNED_REAL_ERROR_MIN_DIST_THRESH}" \
  --act_aligned_real_error_min_dist_delta_thresh "${ACT_ALIGNED_REAL_ERROR_MIN_DIST_DELTA_THRESH}" \
  --act_aligned_debug_recover_eval_rollout "${ACT_ALIGNED_DEBUG_RECOVER_EVAL_ROLLOUT}" \
  --act_aligned_debug_correction_evac_rollout "${ACT_ALIGNED_DEBUG_CORRECTION_EVAC_ROLLOUT}" \
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
  --act_aligned_sample_timeout_sec "${ACT_ALIGNED_SAMPLE_TIMEOUT_SEC}" \
  --failure_phase_bins "${FAILURE_PHASE_BINS}" \
  --failure_translation_dir_bins "${FAILURE_TRANSLATION_DIR_BINS}" \
  --failure_translation_mag_bins "${FAILURE_TRANSLATION_MAG_BINS}" \
  --failure_rotation_dir_bins "${FAILURE_ROTATION_DIR_BINS}" \
  --failure_rotation_mag_bins "${FAILURE_ROTATION_MAG_BINS}" \
  --failure_explore_k "${FAILURE_EXPLORE_K}" \
  --failure_sample_skip_head_ratio "${FAILURE_SAMPLE_SKIP_HEAD_RATIO}" \
  --failure_future_latent_mode "${FAILURE_FUTURE_LATENT_MODE}" \
  --balance_failure_tasks "${BALANCE_FAILURE_TASKS}" \
  --rebalance_failure_groups "${REBALANCE_FAILURE_GROUPS}" \
  --failure_prepare_owner_rank "${FAILURE_PREPARE_OWNER_RANK}" \
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}" \
  --use_wandb "${USE_WANDB}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_group "${WANDB_GROUP}" \
  --wandb_mode "${WANDB_LOG_MODE}"
