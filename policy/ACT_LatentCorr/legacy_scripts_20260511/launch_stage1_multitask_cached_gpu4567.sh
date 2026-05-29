#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="stage1_multitask_cached_gpu4567_${TIMESTAMP}"
OUTPUT_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_multitask_cached_gpu4567"
OUTPUT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
LOG_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
CACHE_DIR="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/cache/multitask_future_latent_fo16_ps16_ddim27"
SESSION="stage1_multitask_cached_gpu4567"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${LOG_ROOT}" "${LOG_DIR}" "${CACHE_DIR}"

cat > "${OUTPUT_DIR}/launch_info.txt" <<EOF
run_tag=${RUN_TAG}
output_dir=${OUTPUT_DIR}
log_dir=${LOG_DIR}
cache_dir=${CACHE_DIR}
session=${SESSION}
gpus=4,5,6,7
per_gpu_batch_size=6
global_batch_size=24
reference_global_batch_size=24
save_freq=50
num_workers=4
mode=offline_cache_then_train
EOF

cat > "${LOG_DIR}/run_cached_pipeline.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin
source /data/miniconda3/etc/profile.d/conda.sh
conda activate ACT

PY_BIN=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python

declare -a GPU_IDS=(4 5 6 7)
declare -a TASK_GROUPS=(
  "sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50"
  "sim-put_bottles_dustbin-demo_clean-50"
  "sim-place_burger_fries-demo_clean-50"
  "sim-handover_block-demo_clean-50"
)

for idx in 0 1 2 3; do
  gpu="${GPU_IDS[$idx]}"
  tasks="${TASK_GROUPS[$idx]}"
  log_file="${LOG_DIR}/precompute_gpu${gpu}.log"
  EVAC_REPO_ROOT='/data/zhenyangfan/EVAC' \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PY_BIN}" -m policy.ACT_LatentCorr.precompute_multitask_future_latent_cache \
    --multi_task_names ${tasks} \
    --cache_dir "${CACHE_DIR}" \
    --evac_ckpt /data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt \
    --evac_config /data/yujieyang/EVAC/configs/robotwin/train_config_mixed50p12.yaml \
    --urdf_path /data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf \
    --device cuda:0 \
    --prefix_steps 16 \
    --future_offset 16 \
    --ddim_steps 27 \
    --batch_size 8 \
    > "${log_file}" 2>&1 &
  echo $! > "${LOG_DIR}/precompute_gpu${gpu}.pid"
done

wait

CUDA_VISIBLE_DEVICES=4,5,6,7 \
EVAC_REPO_ROOT='/data/zhenyangfan/EVAC' \
OUTPUT_DIR="${OUTPUT_DIR}" \
NPROC_PER_NODE=4 \
ACT_INIT_CKPT='/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-robotwin_multitask_5/demo_clean-50/20260417_002800_robotwin_multitask_5_no_wm/policy_epoch_2000_seed_0.ckpt' \
EVAC_CKPT='/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt' \
EVAC_CONFIG='/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml' \
MULTI_TASK_NAMES='sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50' \
NUM_EPOCHS=1000 \
SAVE_FREQ=50 \
BATCH_SIZE=6 \
NUM_WORKERS=4 \
LR=3e-5 \
BASE_ACT_LR_SCALE=1.0 \
LAMBDA_ACTION=1.0 \
LAMBDA_ACTION_CONDITIONED=1.0 \
LAMBDA_ALIGN=0.0 \
BETA_DYNAMICS_MAX=1.0 \
LAMBDA_WM_ACTION_CURRENT=0.0 \
LAMBDA_WM_ACTION_FUTURE=0.0 \
LAMBDA_BRIDGE_FUTURE=0.0 \
FREEZE_BASE_ACT=false \
FREEZE_READOUT_DECODER=true \
DETACH_ACT_FEATURE_FOR_LATENT=true \
USE_ACT_HEAD_CONDITIONING=true \
USE_RAW_WM_TARGETS=false \
DYN_ZERO_STEPS=0 \
DYN_RAMP_STEPS=1000 \
REFERENCE_GLOBAL_BATCH_SIZE=24 \
FUTURE_TEACHER_SOURCE=sim \
FUTURE_LATENT_CACHE_DIR="${CACHE_DIR}" \
FUTURE_LATENT_CACHE_STRICT=true \
FUTURE_LATENT_CACHE_WRITEBACK=false \
USE_WANDB=false \
WANDB_LOG_MODE=disabled \
WANDB_RUN_NAME="${RUN_TAG}" \
WANDB_GROUP='stage1_multitask_cached_gpu4567' \
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_multitask_ddp.sh \
  2>&1 | tee -a "${LOG_DIR}/train.log"
EOF

chmod +x "${LOG_DIR}/run_cached_pipeline.sh"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "export LOG_DIR='${LOG_DIR}' OUTPUT_DIR='${OUTPUT_DIR}' CACHE_DIR='${CACHE_DIR}' RUN_TAG='${RUN_TAG}'; bash '${LOG_DIR}/run_cached_pipeline.sh'"

echo "started ${RUN_TAG}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_dir=${LOG_DIR}"
echo "cache_dir=${CACHE_DIR}"
echo "session=${SESSION}"
+
