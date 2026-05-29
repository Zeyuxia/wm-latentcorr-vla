#!/bin/bash
set -euo pipefail

ROOT=/data/zhenyangfan/RoboTwin
POLICY_ROOT=$ROOT/policy/ACT_LatentCorr
CKPT=$POLICY_ROOT/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_snapshot_gpu13_20260422_183001/stage1_unified_epoch_0250.pt
SEED_FILE=$POLICY_ROOT/jobs/seed_success_25.txt
GPU0=0
GPU1=2
TASKS=(pick_dual_bottles put_bottles_dustbin place_burger_fries handover_block)
MASTER_TAG=actmt250_robust_$(date +"%Y%m%d_%H%M%S")
LOG_ROOT=$POLICY_ROOT/outputs/logs/$MASTER_TAG
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/master.log") 2>&1

echo "master_tag=$MASTER_TAG"
echo "ckpt=$CKPT"

split_seeds() {
  local tag=$1
  local seed_dir=/tmp/${tag}_seeds
  mkdir -p "$seed_dir"
  : > "$seed_dir/shard_0.txt"
  : > "$seed_dir/shard_1.txt"
  local idx=0
  while IFS= read -r seed; do
    [ -z "$seed" ] && continue
    if (( idx % 2 == 0 )); then
      echo "$seed" >> "$seed_dir/shard_0.txt"
    else
      echo "$seed" >> "$seed_dir/shard_1.txt"
    fi
    idx=$((idx+1))
  done < "$SEED_FILE"
  echo "$seed_dir"
}

parse_success() {
  local log=$1
  local line
  line=$(rg 'Success rate:' "$log" | tail -n 1 || true)
  if [ -z "$line" ]; then
    echo "NA"
    return
  fi
  echo "$line" | sed -E 's/.*Success rate: ([0-9]+)\/([0-9]+).*/\1 \2/'
}

for task in "${TASKS[@]}"; do
  run_tag=${MASTER_TAG}_${task}
  task_log_dir=$LOG_ROOT/$task
  mkdir -p "$task_log_dir"
  seed_dir=$(split_seeds "$run_tag")
  echo "[$(date '+%F %T')] launch task=$task run_tag=$run_tag seed_dir=$seed_dir"

  tmux new-session -d -s ${run_tag}_s0 \
    "GPU_ID=$GPU0 TASK_NAME=$task TASK_CONFIG=demo_clean CKPT_SETTING=${run_tag}_shard_0 LATENT_CKPT_PATH=$CKPT INFERENCE_MODE=base SEED_FILE=$seed_dir/shard_0.txt $POLICY_ROOT/eval_success.sh > $task_log_dir/shard_0.log 2>&1"
  tmux new-session -d -s ${run_tag}_s1 \
    "GPU_ID=$GPU1 TASK_NAME=$task TASK_CONFIG=demo_clean CKPT_SETTING=${run_tag}_shard_1 LATENT_CKPT_PATH=$CKPT INFERENCE_MODE=base SEED_FILE=$seed_dir/shard_1.txt $POLICY_ROOT/eval_success.sh > $task_log_dir/shard_1.log 2>&1"

  while tmux has-session -t ${run_tag}_s0 2>/dev/null || tmux has-session -t ${run_tag}_s1 2>/dev/null; do
    sleep 15
  done

  r0=$(parse_success "$task_log_dir/shard_0.log")
  r1=$(parse_success "$task_log_dir/shard_1.log")
  echo "[$(date '+%F %T')] final $task shard0=$r0 shard1=$r1"

  if [ "$r0" != "NA" ] && [ "$r1" != "NA" ]; then
    s0=$(echo "$r0" | awk '{print $1}')
    n0=$(echo "$r0" | awk '{print $2}')
    s1=$(echo "$r1" | awk '{print $1}')
    n1=$(echo "$r1" | awk '{print $2}')
    st=$((s0+s1))
    nt=$((n0+n1))
    python3 - <<PY
s=$st
n=$nt
print(f"[$(date '+%F %T')] total {s}/{n} = {100.0*s/n:.2f}%")
PY
  fi

done

echo "[$(date '+%F %T')] all tasks done"
