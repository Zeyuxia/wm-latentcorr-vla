#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="stage1_unified_failure_multitask_resume650_short8_smoke1tmux_${TIMESTAMP}"
OUTPUT_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
LOG_FILE="${LOG_DIR}/train.log"
SESSION="stage1_unified_smoke1tmux_8"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cat > "${LOG_DIR}/run_train.sh" <<'INNER'
#!/bin/bash
set -euo pipefail
cd /data/zhenyangfan/RoboTwin
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OUTPUT_DIR="${OUTPUT_DIR}" \
NPROC_PER_NODE=8 \
MASTER_PORT=29783 \
RESUME_CKPT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_resume450_gpu0123_20260512_134044/stage1_unified_epoch_0650.pt" \
FAILURE_TABLE_PATHS="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore/stage2_multitask_failure_explore_new_evac_gpu0123_20260429_004924/failure_explore/failure_table.json" \
EVAC_CKPT="/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt" \
EVAC_CONFIG="/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml" \
MULTI_TASK_NAMES='sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50' \
NUM_EPOCHS=700 \
MAX_STEPS=1 \
SAVE_FREQ=50 \
DDIM_STEPS=4 \
NORMAL_BATCH_SIZE=2 \
FAILURE_BATCH_SIZE=1 \
NUM_WORKERS=4 \
LR=3e-5 \
BASE_ACT_LR_SCALE=1.0 \
LAMBDA_ACTION=1.0 \
LAMBDA_TEACHER_MAX=0.5 \
LAMBDA_PRED_MAX=0.5 \
LAMBDA_LATENT_MAX=0.3 \
LAMBDA_TOKEN_INIT=0.1 \
LAMBDA_TOKEN_LATE=0.02 \
NORMAL_CONDITION_KEEP_PROB=0.5 \
LATENT_LOSS_TYPE=normalized_mse \
TOKEN_LOSS_TYPE=mse \
BETA_DYNAMICS_MAX=1.0 \
LAMBDA_WM_ACTION_CURRENT=0.0 \
LAMBDA_WM_ACTION_FUTURE=0.0 \
LAMBDA_BRIDGE_FUTURE=0.0 \
FREEZE_BASE_ACT=false \
FREEZE_READOUT_DECODER=true \
DETACH_ACT_FEATURE_FOR_LATENT=false \
USE_RAW_WM_TARGETS=false \
DYN_ZERO_STEPS=0 \
DYN_RAMP_STEPS=1000 \
DYN_WARMUP_CURVE=cosine \
REFERENCE_GLOBAL_BATCH_SIZE=24 \
FAILURE_FUTURE_LATENT_MODE=rollout \
BALANCE_FAILURE_TASKS=true \
REBALANCE_FAILURE_GROUPS=true \
FAILURE_PREPARE_OWNER_RANK=0 \
USE_WANDB=false \
WANDB_LOG_MODE=disabled \
WANDB_RUN_NAME="${RUN_TAG}" \
WANDB_GROUP='stage1_unified_failure_multitask_resume650_short8_smoke1tmux' \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_unified_failure_multitask_ddp.sh \
  2>&1 | tee -a "${LOG_FILE}"
INNER
chmod +x "${LOG_DIR}/run_train.sh"
tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}"
tmux new-session -d -s "${SESSION}" "export OUTPUT_DIR='${OUTPUT_DIR}' LOG_FILE='${LOG_FILE}' RUN_TAG='${RUN_TAG}'; bash '${LOG_DIR}/run_train.sh'"
echo "started ${RUN_TAG}"
echo "log_file=${LOG_FILE}"
echo "session=${SESSION}"
