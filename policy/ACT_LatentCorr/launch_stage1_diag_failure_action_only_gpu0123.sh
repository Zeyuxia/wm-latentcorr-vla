#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="stage1_diag_failure_action_only_gpu0123_${TIMESTAMP}"
OUTPUT_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/diagnostics/stage1_ablation"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
LOG_FILE="${LOG_DIR}/train.log"
SESSION="${RUN_TAG}"

RESUME_CKPT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_resume450_gpu0123_20260512_134044/stage1_unified_epoch_0650.pt"
FAILURE_TABLE_PATH="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore/stage2_multitask_failure_explore_new_evac_gpu0123_20260429_004924/failure_explore/failure_table.json"
EVAC_CKPT="/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt"
EVAC_CONFIG="/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${LOG_ROOT}" "${LOG_DIR}"

cat > "${LOG_DIR}/run_train.sh" <<'INNER'
#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

CUDA_VISIBLE_DEVICES=0,1,2,3 \
TORCH_EXTENSIONS_DIR="/data/zhenyangfan/.cache/torch_extensions" \
SKIP_CUROBO_PREWARM=true \
OUTPUT_DIR="${OUTPUT_DIR}" \
NPROC_PER_NODE=4 \
MASTER_PORT=29811 \
RESUME_CKPT="${RESUME_CKPT}" \
FAILURE_TABLE_PATHS="${FAILURE_TABLE_PATH}" \
EVAC_CKPT="${EVAC_CKPT}" \
EVAC_CONFIG="${EVAC_CONFIG}" \
MULTI_TASK_NAMES='sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50' \
NUM_EPOCHS=900 \
SAVE_FREQ=50 \
DDIM_STEPS=4 \
NORMAL_BATCH_SIZE=4 \
FAILURE_BATCH_SIZE=2 \
NUM_WORKERS=4 \
LR=3e-5 \
BASE_ACT_LR_SCALE=1.0 \
LAMBDA_ACTION=1.0 \
LAMBDA_ACTION_CONDITIONED=0.0 \
SCHEDULE_ACTION_CONDITIONED=false \
LAMBDA_CONDITION_TOKEN=0.0 \
NORMAL_CONDITION_KEEP_PROB=1.0 \
LAMBDA_ALIGN=0.0 \
BETA_DYNAMICS_MAX=0.0 \
BALANCE_FAILURE_TASKS=true \
REBALANCE_FAILURE_GROUPS=true \
FAILURE_FUTURE_LATENT_MODE=rollout \
FAILURE_PREPARE_OWNER_RANK=0 \
DDP_FIND_UNUSED_PARAMETERS=true \
USE_WANDB=false \
WANDB_LOG_MODE=disabled \
WANDB_RUN_NAME="${RUN_TAG}" \
WANDB_GROUP='stage1_diag_ablation' \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_unified_failure_multitask_ddp.sh \
  2>&1 | tee -a "${LOG_FILE}"
INNER

chmod +x "${LOG_DIR}/run_train.sh"
tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}"
tmux new-session -d -s "${SESSION}" "export OUTPUT_DIR='${OUTPUT_DIR}' LOG_FILE='${LOG_FILE}' RUN_TAG='${RUN_TAG}' RESUME_CKPT='${RESUME_CKPT}' FAILURE_TABLE_PATH='${FAILURE_TABLE_PATH}' EVAC_CKPT='${EVAC_CKPT}' EVAC_CONFIG='${EVAC_CONFIG}'; bash '${LOG_DIR}/run_train.sh'"

echo "started ${RUN_TAG}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_file=${LOG_FILE}"
echo "session=${SESSION}"
