#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

PIPELINE_TAG=${PIPELINE_TAG:-bridge_eval_resume_from_ep400_$(date +"%Y%m%d_%H%M%S")}
LOG_ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${PIPELINE_TAG}
mkdir -p "${LOG_ROOT}"
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
MAIN_LOG="${LOG_ROOT}/pipeline.log"

SEED_G1=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group1.txt
SEED_G2=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group2.txt
SEED_G3=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group3.txt
SEED_FULL=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_bridge150_full.txt
SEED_EP300_REMAIN=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_ep300_bridge_remaining1.txt

EP300_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume200_to300_free0123/20260401_194728/stage2_epoch_0300.pt
EP400_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567/20260402_112353/stage2_epoch_0400.pt
EP500_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567/20260402_112353/stage2_epoch_0500.pt
EP600_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_0600.pt
EP700_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_0700.pt
EP800_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_0800.pt
EP900_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_0900.pt
EP1000_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_1000.pt

EP300_REM1_MANUAL_TAG=${EP300_REM1_MANUAL_TAG:-ep300_bridge_remaining1_manual_20260405_112123}

printf 'label\tmode\trun_tag\tstatus\tnum\tden\tnotes\n' > "${SUMMARY_FILE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MAIN_LOG}"
}

ensure_full_seed_file() {
  cat "${SEED_G1}" "${SEED_G2}" "${SEED_G3}" > "${SEED_FULL}"
}

free_gpus() {
  python3 - <<'PY'
import subprocess
out = subprocess.check_output([
    'nvidia-smi',
    '--query-gpu=index,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits'
], text=True)
free = []
for line in out.strip().splitlines():
    idx, mem, util = [x.strip() for x in line.split(',')]
    if int(mem) < 1200 and int(util) < 15:
        free.append(idx)
print(' '.join(free))
PY
}

wait_for_free_gpus() {
  local need=$1
  while true; do
    local gpus
    gpus="$(free_gpus)"
    local count
    count=$(echo "${gpus}" | awk '{print NF}')
    log "free_gpus=${gpus:-none} need=${need}" >&2
    if [ "${count}" -ge "${need}" ]; then
      echo "${gpus}"
      return 0
    fi
    sleep 60
  done
}

wait_for_result_files() {
  local run_tag=$1
  local expected=$2
  local count
  while true; do
    count=$(find /data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean -path "*${run_tag}*" -name '_result.txt' 2>/dev/null | wc -l | tr -d ' ')
    log "wait run_tag=${run_tag} result_files=${count}/${expected}"
    if [ "${count}" -ge "${expected}" ]; then
      return 0
    fi
    sleep 20
  done
}

summarize_run_from_seed_dir() {
  local run_tag=$1
  python3 - <<PY
import glob, pathlib, re
run_tag = ${run_tag@Q}
log_root = pathlib.Path('/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs') / run_tag
seed_dir = pathlib.Path((log_root / 'seed_dir.txt').read_text().strip())
base = pathlib.Path('/data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean')
files = sorted(base.glob(f'*{run_tag}*/**/_result.txt'))
num = 0
den = 0
for f in files:
    m = re.search(r'_shard_(\\d+)', str(f))
    shard = int(m.group(1))
    shard_seed_file = seed_dir / f'shard_{shard}.txt'
    d = sum(1 for x in shard_seed_file.read_text().splitlines() if x.strip())
    val = float(f.read_text().strip().splitlines()[-1])
    num += round(val * d)
    den += d
print(f'{len(files)}\\t{num}\\t{den}')
PY
}

write_ep200_final_summary() {
  printf 'ep200\tbridge\tep200_bridge_final_reconstructed\tcompleted\t123\t150\treconstructed_from_sparse6+partial_logs+remaining12_patch\n' >> "${SUMMARY_FILE}"
}

wait_or_launch_ep300_remaining1() {
  local base="/data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean"
  local count
  count=$(find "${base}" -path "*${EP300_REM1_MANUAL_TAG}*" -name '_result.txt' 2>/dev/null | wc -l | tr -d ' ')
  if [ "${count}" -ge 1 ]; then
    log "ep300 remaining1 result already exists under ${EP300_REM1_MANUAL_TAG}"
    return 0
  fi

  if tmux has-session -t "${EP300_REM1_MANUAL_TAG}_shard_0" 2>/dev/null; then
    log "ep300 remaining1 manual session still running; waiting for result"
    wait_for_result_files "${EP300_REM1_MANUAL_TAG}" 1
    return 0
  fi

  local gpus run_tag
  gpus="$(wait_for_free_gpus 1)"
  gpus=$(echo "${gpus}" | awk '{print $1}')
  run_tag="ep300_bridge_remaining1_resume_${PIPELINE_TAG}"
  log "launch ${run_tag} gpu=${gpus}"
  LATENT_CKPT_PATH="${EP300_CKPT}" INFERENCE_MODE=bridge SEED_FILE="${SEED_EP300_REMAIN}" GPU_LIST="${gpus}" RUN_TAG="${run_tag}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > "${LOG_ROOT}/${run_tag}_launch.log" 2>&1
  wait_for_result_files "${run_tag}" 1
  EP300_REM1_MANUAL_TAG="${run_tag}"
}

write_ep300_final_summary() {
  local rem_tag=${EP300_REM1_MANUAL_TAG}
  python3 - <<PY >> "${SUMMARY_FILE}"
from pathlib import Path
import re

rem_tag = ${rem_tag@Q}
base = Path('/data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean')
log_path = Path('/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/ep300_bridge150_g1_bridge150_ep300_only_20260404_20260404_235552/shard_0.log')

pat_succ = re.compile(r'Success rate:\\s*\\x1b\\[96m(\\d+)/(\\d+)\\x1b\\[0m.*current seed: \\x1b\\[90m(\\d+)')
pat_skip = re.compile(r'skip seed due to exception: (\\d+)')

prev_num = 0
prev_den = 0
succ = 0
for line in log_path.read_text(errors='ignore').splitlines():
    m = pat_skip.search(line)
    if m:
        continue
    m = pat_succ.search(line)
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        if den == prev_den + 1 and num == prev_num + 1:
            succ += 1
        prev_num, prev_den = num, den

files = sorted(base.glob(f'*{rem_tag}*/**/_result.txt'))
extra = 0
for f in files:
    val = float(f.read_text().strip().splitlines()[-1])
    extra += round(val * 1)

fixed = []
for f in sorted(base.glob('ep300_bridge150_*_bridge150_ep300_only_20260404_20260404_235552_shard_*/**/_result.txt')):
    s = str(f)
    if 'g1' in s and 'shard_0' in s:
        continue
    if '_g1_' in s or '_g2_' in s:
        d = 17 if ('shard_0' in s or 'shard_1' in s) else 16
    else:
        d = 25
    fixed.append((f, d))

num = succ + extra
den = 17
for f, d in fixed:
    val = float(Path(f).read_text().strip().splitlines()[-1])
    num += round(val * d)
    den += d
print(f'ep300\\tbridge\\tep300_bridge_final_reconstructed\\tcompleted\\t{num}\\t{den}\\treconstructed_from_7_result_files+shard0_partial+remaining1_patch')
PY
}

run_full_150() {
  local label=$1
  local ckpt=$2
  local all_gpus use_gpus count run_tag launch_log files num den
  all_gpus="$(wait_for_free_gpus 4)"
  count=$(echo "${all_gpus}" | awk '{print NF}')
  if [ "${count}" -gt 8 ]; then
    use_gpus=$(echo "${all_gpus}" | awk '{for(i=1;i<=8;i++) printf (i==1?"":" ") $i; printf "\\n"}')
    count=8
  else
    use_gpus="${all_gpus}"
  fi
  run_tag="${label}_bridge150_${PIPELINE_TAG}"
  launch_log="${LOG_ROOT}/${run_tag}_launch.log"
  log "launch ${run_tag} gpus='${use_gpus}' shards=${count}"
  LATENT_CKPT_PATH="${ckpt}" INFERENCE_MODE=bridge SEED_FILE="${SEED_FULL}" GPU_LIST="${use_gpus}" RUN_TAG="${run_tag}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > "${launch_log}" 2>&1
  wait_for_result_files "${run_tag}" "${count}"
  read files num den <<<"$(summarize_run_from_seed_dir "${run_tag}")"
  log "done ${label} result_files=${files} result=${num}/${den}"
  printf '%s\tbridge\t%s\tcompleted\t%s\t%s\tresult_files=%s\n' "${label}" "${run_tag}" "${num}" "${den}" "${files}" >> "${SUMMARY_FILE}"
}

ensure_full_seed_file
log "pipeline start tag=${PIPELINE_TAG}"
write_ep200_final_summary
wait_or_launch_ep300_remaining1
write_ep300_final_summary
run_full_150 ep400 "${EP400_CKPT}"
run_full_150 ep500 "${EP500_CKPT}"
run_full_150 ep600 "${EP600_CKPT}"
run_full_150 ep700 "${EP700_CKPT}"
run_full_150 ep800 "${EP800_CKPT}"
run_full_150 ep900 "${EP900_CKPT}"
run_full_150 ep1000 "${EP1000_CKPT}"
log "pipeline complete"
