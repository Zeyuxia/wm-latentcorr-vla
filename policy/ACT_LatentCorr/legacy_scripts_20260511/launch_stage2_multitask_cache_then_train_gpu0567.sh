#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG=${RUN_TAG:-stage2_multitask_cache_then_train_ep600_relaxed090_gpu0567_${TIMESTAMP}}
RUN_ROOT=${RUN_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/${RUN_TAG}}
LOG_ROOT=${LOG_ROOT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs}
LOG_DIR=${LOG_DIR:-${LOG_ROOT}/${RUN_TAG}}
CACHE_DIR=${CACHE_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/cache/stage2_multitask_ep600_relaxed090_ps16_ddim27}
SESSION=${SESSION:-stage2_mt_cache_train_0567}

STAGE1_CKPT=${STAGE1_CKPT:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_multitask_ddp/stage1_multitask_resume0050_to1000_fixcond_4gpu_20260420_0129/stage1_epoch_0600.pt}
FAILURE_TABLE_PATH=${FAILURE_TABLE_PATH:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore/debug_stage2_explore_retry2_20260420_150117/failure_explore_relaxed090/failure_table.json}
EVAC_CKPT=${EVAC_CKPT:-/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt}
EVAC_CONFIG=${EVAC_CONFIG:-/data/yujieyang/EVAC/configs/robotwin/train_config_mixed50p12.yaml}
URDF_PATH=${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}
PY_BIN=${PY_BIN:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python}
TRAIN_LAUNCH=${TRAIN_LAUNCH:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_stage2_multitask_failure_train.sh}
GPU_LIST=${GPU_LIST:-0,5,6,7}
PRECOMPUTE_NUM_BATCHES=${PRECOMPUTE_NUM_BATCHES:-1200}
PRECOMPUTE_BATCH_SIZE=${PRECOMPUTE_BATCH_SIZE:-4}
PRECOMPUTE_NUM_WORKERS=${PRECOMPUTE_NUM_WORKERS:-2}
NUM_EPOCHS=${NUM_EPOCHS:-250}
SAVE_FREQ=${SAVE_FREQ:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
CORRECTION_BATCH_SIZE=${CORRECTION_BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
MASTER_PORT=${MASTER_PORT:-29721}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
ACT_ALIGNED_ERROR_POLICY_CKPT=${ACT_ALIGNED_ERROR_POLICY_CKPT:-${STAGE1_CKPT}}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${CACHE_DIR}"

cat > "${OUTPUT_DIR}/launch_info.txt" <<INFO
run_tag=${RUN_TAG}
output_dir=${OUTPUT_DIR}
log_dir=${LOG_DIR}
cache_dir=${CACHE_DIR}
session=${SESSION}
gpus=${GPU_LIST}
stage1_ckpt=${STAGE1_CKPT}
act_aligned_error_policy_ckpt=${ACT_ALIGNED_ERROR_POLICY_CKPT}
failure_table_path=${FAILURE_TABLE_PATH}
mode=offline_cache_then_train
precompute_num_batches=${PRECOMPUTE_NUM_BATCHES}
precompute_batch_size=${PRECOMPUTE_BATCH_SIZE}
train_batch_size=${BATCH_SIZE}
train_correction_batch_size=${CORRECTION_BATCH_SIZE}
INFO

cat > "${LOG_DIR}/run_cache_then_train.sh" <<'INNER'
#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin
source /data/miniconda3/etc/profile.d/conda.sh
conda activate ACT

TASKS=(
  sim-open_laptop-demo_clean-50
  sim-pick_dual_bottles-demo_clean-50
  sim-put_bottles_dustbin-demo_clean-50
  sim-place_burger_fries-demo_clean-50
  sim-handover_block-demo_clean-50
)
GPU_IDS=(0 5 6 7)
TASK_GROUPS=(
  "sim-open_laptop-demo_clean-50 sim-handover_block-demo_clean-50"
  "sim-pick_dual_bottles-demo_clean-50"
  "sim-put_bottles_dustbin-demo_clean-50"
  "sim-place_burger_fries-demo_clean-50"
)

echo "[pipeline] start $(date '+%F %T')"
echo "[pipeline] cache_dir=${CACHE_DIR}"
echo "[pipeline] output_dir=${OUTPUT_DIR}"

for idx in 0 1 2 3; do
  gpu="${GPU_IDS[$idx]}"
  tasks="${TASK_GROUPS[$idx]}"
  log_file="${LOG_DIR}/precompute_gpu${gpu}.log"
  (
    set -euo pipefail
    for task in ${tasks}; do
      echo "[precompute][gpu=${gpu}] task=${task} start $(date '+%F %T')" | tee -a "${log_file}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
      EVAC_REPO_ROOT='/data/zhenyangfan/EVAC' \
      "${PY_BIN}" -m policy.ACT_LatentCorr.precompute_stage2_correction_latent_cache \
        --task_name "${task}" \
        --cache_dir "${CACHE_DIR}" \
        --evac_ckpt "${EVAC_CKPT}" \
        --evac_config "${EVAC_CONFIG}" \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --act_aligned_error_policy_ckpt "${ACT_ALIGNED_ERROR_POLICY_CKPT}" \
        --urdf_path "${URDF_PATH}" \
        --device cuda:0 \
        --prefix_steps 16 \
        --act_chunk_size 50 \
        --future_offset 16 \
        --ddim_steps 27 \
        --batch_size "${PRECOMPUTE_BATCH_SIZE}" \
        --num_workers "${PRECOMPUTE_NUM_WORKERS}" \
        --num_batches "${PRECOMPUTE_NUM_BATCHES}" \
        --failure_table_path "${FAILURE_TABLE_PATH}" \
        --failure_mode train \
        --failure_phase_bins 3 \
        --failure_translation_dir_bins 6 \
        --failure_translation_mag_bins 3 \
        --failure_rotation_dir_bins 6 \
        --failure_rotation_mag_bins 3 \
        --failure_explore_k 4 \
        --failure_sample_skip_head_ratio 0.6 \
        --planner_target_mode backward \
        --planner_target_lookahead_steps 6 \
        --planner_orient_weight 0.0573 \
        --planner_gripper_penalty 1.0 \
        --planner_nearest_window_radius 12 \
        --planner_active_joint_delta_thresh 0.01 \
        --planner_active_gripper_delta_thresh 0.05 \
        --act_aligned_rollout_exec_steps 16 \
        --act_aligned_min_dist_fallback_force_correction true \
        --act_aligned_min_dist_recover_ratio 0.75 \
        --act_aligned_real_error_trigger_enable true \
        --act_aligned_real_error_min_dist_thresh 0.01 \
        --act_aligned_real_error_min_dist_delta_thresh 0.005 \
        --act_aligned_debug_recover_eval_rollout false \
        --act_aligned_debug_correction_evac_rollout false \
        --recover_eval_enable false \
        --recover_eval_save_video false \
        --recover_eval_gripper_open_thresh 0.8 \
        --recover_eval_pos_thresh_m 0.03 \
        --recover_eval_rot_thresh_deg 10.0 \
        --recover_eval_nearest_window_radius 16 \
        --recover_eval_video_bridge_steps 16 \
        --act_aligned_correction_interp_nearest_enable false \
        --act_aligned_correction_interp_prefix_ratio 0.6 \
        --act_aligned_correction_planner_prefix_ratio 0.5 \
        --act_aligned_correction_gripper_close_prefix_ratio 0.32 \
        --act_aligned_correction_compose_gt_tail_enable true \
        --act_aligned_correction_gripper_switch_ratio 0.5 \
        --act_aligned_recover_gripper_penalty 0.0 \
        --act_aligned_enable_perturb true \
        --act_aligned_perturb_prob 1.0 \
        --act_aligned_perturb_error_mode open_laptop_pregrasp \
        --act_aligned_perturb_open_laptop_pregrasp_close_prob 0.5 \
        --act_aligned_perturb_open_laptop_pregrasp_translation_prob 0.0 \
        --act_aligned_perturb_open_laptop_pregrasp_rotation_prob 0.0 \
        --act_aligned_perturb_eef_fail_gain 0.10 \
        --act_aligned_perturb_rot_max_deg 15.0 \
        --act_aligned_perturb_mag_random false \
        --act_aligned_perturb_mag_rand_min 1.0 \
        --act_aligned_perturb_mag_rand_max 1.4 \
        --act_aligned_perturb_reject_sampling_enable true \
        --act_aligned_perturb_reject_max_trials 4 \
        --act_aligned_perturb_reject_dir_jitter_eps 0.2 \
        --act_aligned_perturb_gripper_close_min 0.10 \
        --act_aligned_perturb_gripper_open_max 0.90 \
        --act_aligned_perturb_gripper_fast_ratio 0.20 \
        --act_aligned_sample_pregrasp_phase_window_len 30 \
        --act_aligned_sample_timeout_sec 30 \
        2>&1 | tee -a "${log_file}"
      echo "[precompute][gpu=${gpu}] task=${task} done $(date '+%F %T')" | tee -a "${log_file}"
    done
  ) &
  echo $! > "${LOG_DIR}/precompute_gpu${gpu}.pid"
done

wait

echo "[pipeline] precompute finished $(date '+%F %T')" | tee -a "${LOG_DIR}/pipeline.log"

env \
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  RUN_TAG="${RUN_TAG}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LOG_DIR="${LOG_DIR}" \
  LOG_FILE="${LOG_DIR}/train.log" \
  GPU_LIST="${GPU_LIST}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  MASTER_PORT="${MASTER_PORT}" \
  STAGE1_CKPT="${STAGE1_CKPT}" \
  BASE_ANCHOR_CKPT="${STAGE1_CKPT}" \
  ACT_ALIGNED_ERROR_POLICY_CKPT="${ACT_ALIGNED_ERROR_POLICY_CKPT}" \
  FAILURE_TABLE_PATHS="${FAILURE_TABLE_PATH}" \
  STAGE2_LATENT_CACHE_DIR="${CACHE_DIR}" \
  STAGE2_LATENT_CACHE_STRICT=false \
  STAGE2_LATENT_CACHE_WRITEBACK=false \
  NUM_EPOCHS="${NUM_EPOCHS}" \
  SAVE_FREQ="${SAVE_FREQ}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  CORRECTION_BATCH_SIZE="${CORRECTION_BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  LR=3e-5 \
  RETAIN_WEIGHT=1.0 \
  RETAIN_WEIGHT_FINAL=0.1 \
  RETAIN_DECAY_START_EPOCH=0 \
  RETAIN_DECAY_END_EPOCH=100 \
  RETAIN_DECAY_CURVE=cosine \
  BETA_DYNAMICS_MAX=1.0 \
  DYN_ZERO_STEPS=0 \
  DYN_RAMP_STEPS=100 \
  DYN_WARMUP_CURVE=cosine \
  DYN_SCHEDULE_UNIT=epoch \
  REFERENCE_GLOBAL_BATCH_SIZE=24 \
  LAMBDA_WM_ACTION_CURRENT=0.0 \
  LAMBDA_WM_ACTION_FUTURE=0.0 \
  LAMBDA_BRIDGE_FUTURE=0.0 \
  BRIDGE_WEIGHT=0.0 \
  EVAC_CKPT="${EVAC_CKPT}" \
  EVAC_CONFIG="${EVAC_CONFIG}" \
  EVAC_REPO_ROOT='/data/zhenyangfan/EVAC' \
  USE_WANDB=false \
  WANDB_LOG_MODE=disabled \
  WANDB_RUN_NAME="${RUN_TAG}" \
  WANDB_GROUP=stage2_multitask_failure \
  "${TRAIN_LAUNCH}" 2>&1 | tee -a "${LOG_DIR}/pipeline.log"
INNER

chmod +x "${LOG_DIR}/run_cache_then_train.sh"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "export RUN_TAG='${RUN_TAG}' OUTPUT_DIR='${OUTPUT_DIR}' LOG_DIR='${LOG_DIR}' CACHE_DIR='${CACHE_DIR}' GPU_LIST='${GPU_LIST}' STAGE1_CKPT='${STAGE1_CKPT}' ACT_ALIGNED_ERROR_POLICY_CKPT='${ACT_ALIGNED_ERROR_POLICY_CKPT}' FAILURE_TABLE_PATH='${FAILURE_TABLE_PATH}' EVAC_CKPT='${EVAC_CKPT}' EVAC_CONFIG='${EVAC_CONFIG}' URDF_PATH='${URDF_PATH}' PY_BIN='${PY_BIN}' TRAIN_LAUNCH='${TRAIN_LAUNCH}' PRECOMPUTE_NUM_BATCHES='${PRECOMPUTE_NUM_BATCHES}' PRECOMPUTE_BATCH_SIZE='${PRECOMPUTE_BATCH_SIZE}' PRECOMPUTE_NUM_WORKERS='${PRECOMPUTE_NUM_WORKERS}' NUM_EPOCHS='${NUM_EPOCHS}' SAVE_FREQ='${SAVE_FREQ}' BATCH_SIZE='${BATCH_SIZE}' CORRECTION_BATCH_SIZE='${CORRECTION_BATCH_SIZE}' NUM_WORKERS='${NUM_WORKERS}' MASTER_PORT='${MASTER_PORT}' NPROC_PER_NODE='${NPROC_PER_NODE}'; bash '${LOG_DIR}/run_cache_then_train.sh'"

echo "started ${RUN_TAG}"
echo "session=${SESSION}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_dir=${LOG_DIR}"
echo "cache_dir=${CACHE_DIR}"
