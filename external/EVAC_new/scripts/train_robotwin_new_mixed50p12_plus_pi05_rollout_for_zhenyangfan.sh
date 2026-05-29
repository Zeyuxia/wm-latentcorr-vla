#!/usr/bin/env bash
set -euo pipefail

EVAC_REPO="${EVAC_REPO:-/data/yujieyang/EVAC_new}"
CFG="${CFG:-/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml}"
RUN_ROOT="${RUN_ROOT:-/data/yujieyang/EVAC_new/runs}"
GPU_NUMS="${GPU_NUMS:-2}"
NUM_NODES="${NUM_NODES:-1}"
RESUME_CKPT="${RESUME_CKPT:-}"
NAME_BASE="${NAME_BASE:-evac_robotwin_new_mixed50p12_plus_pi05_rollout}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_CKPT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--resume /path/to/checkpoint.ckpt]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$RUN_ROOT"
TIMESTAMP=$(date +%Y-%m-%dT%H-%M-%S)
RUN_NAME="${NAME_BASE}_${TIMESTAMP}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training.log"
MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"
SESSION_NAME="$RUN_NAME"

echo "RUN_NAME=$RUN_NAME" | tee -a "$LOG_FILE"
echo "RUN_DIR=$RUN_DIR" | tee -a "$LOG_FILE"
echo "CONFIG=$CFG" | tee -a "$LOG_FILE"
echo "EVAC_REPO=$EVAC_REPO" | tee -a "$LOG_FILE"
echo "GPU_NUMS=$GPU_NUMS" | tee -a "$LOG_FILE"
echo "NUM_NODES=$NUM_NODES" | tee -a "$LOG_FILE"
echo "TMUX_SESSION=$SESSION_NAME" | tee -a "$LOG_FILE"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
fi
if [[ -n "$RESUME_CKPT" ]]; then
  echo "RESUME_CKPT=$RESUME_CKPT" | tee -a "$LOG_FILE"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found in PATH" >&2
  exit 1
fi

CMD=(
  torchrun
  --standalone
  --nproc_per_node="${GPU_NUMS}"
  trainer/trainer.py
  --base "$CFG"
  --train
  --devices "${GPU_NUMS}"
  --name "$RUN_NAME"
  --logdir "$RUN_ROOT"
  lightning.trainer.num_nodes="${NUM_NODES}"
  lightning.strategy=ddp
)

if [[ -n "$RESUME_CKPT" ]]; then
  CMD+=(--resume "$RESUME_CKPT")
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

printf -v CMD_STR '%q ' "${CMD[@]}"

TMUX_CMD=(
  env
  "EVAC_REPO=$EVAC_REPO"
  "RUN_ROOT=$RUN_ROOT"
  "RUN_NAME=$RUN_NAME"
  "LOG_FILE=$LOG_FILE"
  "MPLCONFIGDIR=$MPLCONFIGDIR"
  "PYTHONNOUSERSITE=1"
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  TMUX_CMD+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

TMUX_CMD+=(
  bash
  -lc
  "source /data/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_EVAC && cd \"\$EVAC_REPO\" && MPLCONFIGDIR=\"\$MPLCONFIGDIR\" ${CMD_STR} 2>&1 | tee -a \"\$LOG_FILE\""
)

tmux new-session -d -s "$SESSION_NAME" "${TMUX_CMD[@]}"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Log file: $LOG_FILE"
