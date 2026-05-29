#!/bin/bash
set -euo pipefail

SERIES_DIR="$1"
SERIES_LOG="$2"
SUMMARY_TSV="$3"
SUMMARY_TXT="$4"

mkdir -p "${SERIES_DIR}"
echo -e "epoch\tstatus\tnum\tden\trate\tsummary_dir" > "${SUMMARY_TSV}"

run_one() {
  local epoch="$1"
  local ckpt="$2"
  local run_prefix="series_bridge100_ep${epoch}_$(date +%Y%m%d_%H%M%S)"
  local summary_dir="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${run_prefix}_summary"

  echo "[$(date '+%F %T')] launch epoch=${epoch} ckpt=${ckpt}" | tee -a "${SERIES_LOG}"
  LATENT_CKPT_PATH="${ckpt}" \
  INFERENCE_MODE=bridge \
  RUN_PREFIX="${run_prefix}" \
  GPU_LIST_G1="0 1 2 3" \
  GPU_LIST_G2="4 5 6 7" \
  bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_eval_g1g2_100.sh | tee -a "${SERIES_LOG}"

  local watch_session="${run_prefix}_summary_watch"
  while tmux has-session -t "${watch_session}" 2>/dev/null; do
    sleep 15
  done

  if [ -f "${summary_dir}/summary.tsv" ]; then
    awk -F '\t' 'NR>1 && $1=="TOTAL" {printf "%s\t%s\t%s\t%s\t%s\t%s\n", EPOCH, $2, $3, $4, $5, SDIR}' \
      EPOCH="${epoch}" SDIR="${summary_dir}" "${summary_dir}/summary.tsv" >> "${SUMMARY_TSV}"
    echo "[$(date '+%F %T')] finished epoch=${epoch} summary=${summary_dir}" | tee -a "${SERIES_LOG}"
  else
    echo -e "${epoch}\tmissing\t\t\t\t${summary_dir}" >> "${SUMMARY_TSV}"
    echo "[$(date '+%F %T')] missing summary for epoch=${epoch}" | tee -a "${SERIES_LOG}"
  fi

  python3 - "${SUMMARY_TSV}" "${SUMMARY_TXT}" <<'PY'
from pathlib import Path
import sys

summary_tsv = Path(sys.argv[1])
summary_txt = Path(sys.argv[2])
lines = [l.strip().split('\t') for l in summary_tsv.read_text().splitlines() if l.strip()]
rows = lines[1:]
with summary_txt.open('w') as f:
    for row in rows:
        epoch, status, num, den, rate, sdir = row
        if status == 'missing':
            f.write(f'ep{epoch}: missing | {sdir}\n')
        else:
            pct = float(rate) * 100.0 if rate else 0.0
            f.write(f'ep{epoch}: {num}/{den} = {pct:.1f}% | {sdir}\n')
PY
}

run_one 300 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume250_to700_free012357/20260406_165413/stage2_epoch_0300.pt
run_one 400 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume250_to700_free012357/20260406_165413/stage2_epoch_0400.pt
run_one 500 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume400_to1000_free01234567/20260407_005658/stage2_epoch_0500.pt
run_one 600 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume400_to1000_free01234567/20260407_005658/stage2_epoch_0600.pt
run_one 700 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume400_to1000_free01234567/20260407_005658/stage2_epoch_0700.pt
run_one 800 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume400_to1000_free01234567/20260407_005658/stage2_epoch_0800.pt

echo "[$(date '+%F %T')] series done" | tee -a "${SERIES_LOG}"
