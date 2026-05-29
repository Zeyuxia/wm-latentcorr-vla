#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="stage1_unified_failure_multitask_fulltable_lctok05_gpu567_${TIMESTAMP}"
OUTPUT_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
LOG_FILE="${LOG_DIR}/train.log"
SESSION="stage1_unified_fulltable_lctok05_gpu567"

GPU_IDS="5,6,7"
ACT_INIT_CKPT="/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-robotwin_multitask_5/demo_clean-50/20260417_002800_robotwin_multitask_5_no_wm/policy_epoch_2000_seed_0.ckpt"
FAILURE_TABLE_PATH="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore/stage2_multitask_failure_explore_new_evac_gpu0123_20260429_004924/failure_explore/failure_table.json"
EVAC_CKPT="/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt"
EVAC_CONFIG="/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${LOG_ROOT}" "${LOG_DIR}"

cat > "${OUTPUT_DIR}/launch_info.txt" <<INFO
run_tag=${RUN_TAG}
output_dir=${OUTPUT_DIR}
log_dir=${LOG_DIR}
log_file=${LOG_FILE}
session=${SESSION}
gpus=${GPU_IDS}
act_init_ckpt=${ACT_INIT_CKPT}
failure_table_path=${FAILURE_TABLE_PATH}
evac_ckpt=${EVAC_CKPT}
evac_config=${EVAC_CONFIG}
normal_batch_size=4
failure_batch_size=2
global_batch_size=18
reference_global_batch_size=24
num_epochs=1000
save_freq=50
ddim_steps=4
lr=3e-5
lambda_action=1.0
lambda_action_conditioned=0.5
schedule_action_conditioned=true
lambda_condition_token=0.5
lambda_align=0.0
beta_dynamics_max=1.0
failure_future_latent_mode=rollout
note=restart_from_act2000_with_full_explore_table_and_lower_condition_token
INFO

cat > "${LOG_DIR}/run_train.sh" <<'INNER'
#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

CUDA_VISIBLE_DEVICES=5,6,7 \
OUTPUT_DIR="${OUTPUT_DIR}" \
NPROC_PER_NODE=3 \
MASTER_PORT=29769 \
ACT_INIT_CKPT="${ACT_INIT_CKPT}" \
FAILURE_TABLE_PATHS="${FAILURE_TABLE_PATH}" \
EVAC_CKPT="${EVAC_CKPT}" \
EVAC_CONFIG="${EVAC_CONFIG}" \
MULTI_TASK_NAMES='sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50' \
NUM_EPOCHS=1000 \
SAVE_FREQ=50 \
DDIM_STEPS=4 \
NORMAL_BATCH_SIZE=4 \
FAILURE_BATCH_SIZE=2 \
NUM_WORKERS=4 \
LR=3e-5 \
BASE_ACT_LR_SCALE=1.0 \
LAMBDA_ACTION=1.0 \
LAMBDA_ACTION_CONDITIONED=0.5 \
SCHEDULE_ACTION_CONDITIONED=true \
LAMBDA_CONDITION_TOKEN=0.5 \
LAMBDA_ALIGN=0.0 \
BETA_DYNAMICS_MAX=1.0 \
LAMBDA_WM_ACTION_CURRENT=0.0 \
LAMBDA_WM_ACTION_FUTURE=0.0 \
LAMBDA_BRIDGE_FUTURE=0.0 \
FREEZE_BASE_ACT=false \
FREEZE_READOUT_DECODER=true \
DETACH_ACT_FEATURE_FOR_LATENT=true \
USE_RAW_WM_TARGETS=false \
DYN_ZERO_STEPS=0 \
DYN_RAMP_STEPS=1000 \
DYN_WARMUP_CURVE=cosine \
REFERENCE_GLOBAL_BATCH_SIZE=24 \
FAILURE_FUTURE_LATENT_MODE=rollout \
USE_WANDB=false \
WANDB_LOG_MODE=disabled \
WANDB_RUN_NAME="${RUN_TAG}" \
WANDB_GROUP='stage1_unified_failure_multitask_fulltable_lctok05' \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_unified_failure_multitask_ddp.sh \
  2>&1 | tee -a "${LOG_FILE}"
INNER

chmod +x "${LOG_DIR}/run_train.sh"

tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}"
tmux new-session -d -s "${SESSION}" "export OUTPUT_DIR='${OUTPUT_DIR}' LOG_FILE='${LOG_FILE}' RUN_TAG='${RUN_TAG}' ACT_INIT_CKPT='${ACT_INIT_CKPT}' FAILURE_TABLE_PATH='${FAILURE_TABLE_PATH}' EVAC_CKPT='${EVAC_CKPT}' EVAC_CONFIG='${EVAC_CONFIG}'; bash '${LOG_DIR}/run_train.sh'"

echo "started ${RUN_TAG}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_file=${LOG_FILE}"
echo "session=${SESSION}"
