#!/bin/bash
set -euo pipefail

ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr
RUN_ROOT=${RUN_ROOT:-$ROOT/outputs/logs/multitask_stage1_eval20}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR="$RUN_ROOT/$TS"
mkdir -p "$OUT_DIR"

CKPT=${CKPT:-$ROOT/outputs/formal_runs/stage1_multitask_ddp/stage1_multitask_partialcache_align02_teachercond_4gpu_20260418_224028/stage1_epoch_0650.pt}
SEED_FILE=${SEED_FILE:-$ROOT/outputs/logs/multitask_stage1_eval_seeds20.txt}
CKPT_SETTING=${CKPT_SETTING:-multitask_stage1_ep650_smoke20}
INFERENCE_MODE=${INFERENCE_MODE:-bridge}
EVAC_CKPT=${EVAC_CKPT:-/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-04-16T19-26-12/checkpoints/epoch=562-step=18000.ckpt}
EVAC_CONFIG=${EVAC_CONFIG:-/data/yujieyang/EVAC/configs/robotwin/train_config_mixed50p12.yaml}
EVAC_REPO_ROOT=${EVAC_REPO_ROOT:-/data/zhenyangfan/EVAC}
URDF_PATH=${URDF_PATH:-/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}

TASKS=(open_laptop pick_dual_bottles put_bottles_dustbin place_burger_fries handover_block)
GPUS=(4 5)

echo -e "label\ttask\tgpu\tlog" > "$OUT_DIR/summary.tsv"
echo "[start] ts=$TS ckpt=$CKPT" | tee "$OUT_DIR/series.log"

run_one() {
  local task="$1"
  local gpu="$2"
  local log="$OUT_DIR/${task}_g${gpu}.log"
  echo "[launch] task=$task gpu=$gpu log=$log" | tee -a "$OUT_DIR/series.log"
  (
    export EVAC_REPO_ROOT="$EVAC_REPO_ROOT"
    TASK_NAME="$task" \
    TASK_CONFIG=demo_clean \
    CKPT_SETTING="$CKPT_SETTING" \
    LATENT_CKPT_PATH="$CKPT" \
    INFERENCE_MODE="$INFERENCE_MODE" \
    SEED_FILE="$SEED_FILE" \
    GPU_ID="$gpu" \
    EVAC_CKPT="$EVAC_CKPT" \
    EVAC_CONFIG="$EVAC_CONFIG" \
    URDF_PATH="$URDF_PATH" \
    bash "$ROOT/eval_success.sh"
  ) > "$log" 2>&1

  local rate_line
  rate_line=$(rg "Success rate:" "$log" | tail -n 1 || true)
  echo -e "${task}\t${task}\t${gpu}\t${log}" >> "$OUT_DIR/summary.tsv"
  echo "[done] task=$task gpu=$gpu ${rate_line:-NO_RATE_FOUND}" | tee -a "$OUT_DIR/series.log"
}

idx=0
while [ $idx -lt ${#TASKS[@]} ]; do
  pids=()
  labels=()
  for slot in 0 1; do
    if [ $idx -ge ${#TASKS[@]} ]; then
      break
    fi
    task="${TASKS[$idx]}"
    gpu="${GPUS[$slot]}"
    run_one "$task" "$gpu" &
    pids+=("$!")
    labels+=("$task")
    idx=$((idx+1))
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
  done
  echo "[batch_done] remaining=$(( ${#TASKS[@]} - idx ))" | tee -a "$OUT_DIR/series.log"
done

echo "[finished] out_dir=$OUT_DIR" | tee -a "$OUT_DIR/series.log"
