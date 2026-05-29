#!/bin/bash
set -euo pipefail

ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr
cd "$ROOT"

SEED_GPU2=${SEED_GPU2:-$ROOT/outputs/logs/eval_series_no_skip_2gpu/seed_valid100_gpu2.txt}
SEED_GPU3=${SEED_GPU3:-$ROOT/outputs/logs/eval_series_no_skip_2gpu/seed_valid100_gpu3.txt}
LOG_ROOT=${LOG_ROOT:-$ROOT/outputs/logs/eval_series_no_skip_2gpu/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG_ROOT"
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SERIES_LOG="$LOG_ROOT/series.log"

echo -e "label\tckpt\tsucc_gpu2\ttotal_gpu2\tsucc_gpu3\ttotal_gpu3\tsucc_total\ttotal\trate" > "$SUMMARY_TSV"

run_one() {
  local label="$1"
  local ckpt="$2"
  local log2="$LOG_ROOT/${label}_gpu2.log"
  local log3="$LOG_ROOT/${label}_gpu3.log"

  echo "[$(date '+%F %T')] start label=$label ckpt=$ckpt" | tee -a "$SERIES_LOG"

  (
    GPU_ID=2 \
    LATENT_CKPT_PATH="$ckpt" \
    SEED_FILE="$SEED_GPU2" \
    TASK_NAME=open_laptop \
    TASK_CONFIG=demo_clean \
    CKPT_SETTING=latentcorr_stage2 \
    INFERENCE_MODE=bridge \
    bash "$ROOT/eval_success.sh"
  ) > "$log2" 2>&1 &
  local pid2=$!

  (
    GPU_ID=3 \
    LATENT_CKPT_PATH="$ckpt" \
    SEED_FILE="$SEED_GPU3" \
    TASK_NAME=open_laptop \
    TASK_CONFIG=demo_clean \
    CKPT_SETTING=latentcorr_stage2 \
    INFERENCE_MODE=bridge \
    bash "$ROOT/eval_success.sh"
  ) > "$log3" 2>&1 &
  local pid3=$!

  wait "$pid2"
  wait "$pid3"

  python3 - "$label" "$ckpt" "$log2" "$log3" "$SUMMARY_TSV" <<'PY'
import re, sys
from pathlib import Path
label, ckpt, log2, log3, summary = sys.argv[1:]
pat = re.compile(r'Success rate:\s*(\d+)/(\d+)')
rows = []
for p in [log2, log3]:
    text = Path(p).read_text(errors='ignore')
    ms = pat.findall(text)
    if not ms:
        raise SystemExit(f'no success rate found in {p}')
    s, t = map(int, ms[-1])
    rows.append((s, t))
(s2, t2), (s3, t3) = rows
st = s2 + s3
tt = t2 + t3
rate = st / tt if tt else 0.0
line = f"{label}\t{ckpt}\t{s2}\t{t2}\t{s3}\t{t3}\t{st}\t{tt}\t{rate:.4f}\n"
with open(summary, 'a') as f:
    f.write(line)
print(line, end='')
PY

  echo "[$(date '+%F %T')] finished label=$label" | tee -a "$SERIES_LOG"
}

run_one ep300 "$ROOT/outputs/formal_runs/stage2_ddp/stage2_openlaptop_nocache_resume0250_to0500_4gpu_4p2_20260419_010318/stage2_epoch_0300.pt"
run_one ep350 "$ROOT/outputs/formal_runs/stage2_ddp/stage2_openlaptop_nocache_resume0250_to0500_4gpu_4p2_20260419_010318/stage2_epoch_0350.pt"
run_one ep400 "$ROOT/outputs/formal_runs/stage2_ddp/stage2_openlaptop_nocache_resume0250_to0500_4gpu_4p2_20260419_010318/stage2_epoch_0400.pt"
run_one ep450 "$ROOT/outputs/formal_runs/stage2_ddp/stage2_openlaptop_nocache_resume0250_to0500_4gpu_4p2_20260419_010318/stage2_epoch_0450.pt"
run_one ep500 "$ROOT/outputs/formal_runs/stage2_ddp/stage2_openlaptop_nocache_resume0250_to0500_4gpu_4p2_20260419_010318/stage2_epoch_0500.pt"

echo "done: $SUMMARY_TSV" | tee -a "$SERIES_LOG"
