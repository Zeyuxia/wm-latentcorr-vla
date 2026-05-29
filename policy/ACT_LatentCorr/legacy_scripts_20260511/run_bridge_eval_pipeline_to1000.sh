#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

PIPELINE_TAG=${PIPELINE_TAG:-bridge_eval_pipeline_to1000_$(date +"%Y%m%d_%H%M%S")}
LOG_ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${PIPELINE_TAG}
mkdir -p "${LOG_ROOT}"
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
MAIN_LOG="${LOG_ROOT}/pipeline.log"

SEED_G1=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group1.txt
SEED_G2=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group2.txt
SEED_G3=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group3.txt
SEED_EP200_REMAIN=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_ep200_bridge_remaining12.txt
SEED_EP300_REMAIN=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_ep300_bridge_remaining1.txt

EP200_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume100_to200_free0123/20260401_151035/stage2_epoch_0200.pt
EP300_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume200_to300_free0123/20260401_194728/stage2_epoch_0300.pt
EP400_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567/20260402_112353/stage2_epoch_0400.pt
EP500_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume300_to500_free01234567/20260402_112353/stage2_epoch_0500.pt
EP600_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_0600.pt
EP700_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_0700.pt
EP800_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_0800.pt
EP900_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_0900.pt
EP1000_CKPT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_acthead_teachercorr_anchor_resume500_to1000_free01234567/20260402_150253/stage2_epoch_1000.pt

printf 'label\tmode\trun_tag\tstatus\tnum\tden\tnotes\n' > "${SUMMARY_FILE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MAIN_LOG}"
}

free_gpus() {
  python3 - <<'PY'
import subprocess
out = subprocess.check_output([
    'nvidia-smi',
    '--query-gpu=index,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits'
], text=True)
free=[]
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
      echo "${gpus}" | awk -v n="${need}" '{for (i=1; i<=n; ++i) printf (i==1?"":" ") $i; printf "\n"}'
      return 0
    fi
    sleep 60
  done
}

wait_run_done() {
  local pattern=$1
  while true; do
    local active
    active=$(tmux ls 2>/dev/null | rg "^(${pattern})_shard_" | wc -l | tr -d ' ')
    log "wait pattern=${pattern} active=${active}"
    if [ "${active}" = "0" ]; then
      return 0
    fi
    sleep 20
  done
}

summarize_full_150() {
  local run_tag=$1
  python3 - <<PY
import glob, pathlib, re, sys
run_tag = ${run_tag@Q}
base = '/data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean'
files = sorted(glob.glob(f'{base}/*{run_tag}*/**/_result.txt', recursive=True))
num=0
den=0
for f in files:
    txt = pathlib.Path(f).read_text().strip().splitlines()
    val = float(txt[-1])
    m = re.search(r'_shard_(\d+)', f)
    shard = int(m.group(1)) if m else -1
    if '_g1_' in f or '_g2_' in f:
        d = 17 if shard in (0,1) else 16
    else:
        d = 25
    num += round(val*d)
    den += d
print(f'{len(files)}\t{num}\t{den}')
PY
}

summarize_seed_file_run() {
  local run_tag=$1
  local seed_file=$2
  python3 - <<PY
import glob, pathlib, re
run_tag = ${run_tag@Q}
seed_file = ${seed_file@Q}
base = '/data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean'
files = sorted(glob.glob(f'{base}/*{run_tag}*/**/_result.txt', recursive=True))
# launch_parallel_eval round-robin split by GPU count
seed_count = sum(1 for x in pathlib.Path(seed_file).read_text().splitlines() if x.strip())
# we will run this on 4 GPUs => 3,3,3,3 for 12 seeds
num=0
den=0
for f in files:
    txt = pathlib.Path(f).read_text().strip().splitlines()
    val = float(txt[-1])
    m = re.search(r'_shard_(\d+)', f)
    shard = int(m.group(1)) if m else -1
    d = seed_count // 4 + (1 if shard < seed_count % 4 else 0)
    num += round(val*d)
    den += d
print(f'{len(files)}\t{num}\t{den}')
PY
}

run_remaining12() {
  local gpus run_tag launch_log files num den
  gpus="$(wait_for_free_gpus 4)"
  run_tag="ep200_bridge_remaining12_${PIPELINE_TAG}"
  launch_log="${LOG_ROOT}/${run_tag}_launch.log"
  log "launch ${run_tag} gpus=${gpus}"
  LATENT_CKPT_PATH="${EP200_CKPT}" INFERENCE_MODE=bridge SEED_FILE="${SEED_EP200_REMAIN}" GPU_LIST="${gpus}" RUN_TAG="${run_tag}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > "${launch_log}" 2>&1
  wait_run_done "${run_tag}"
  read files num den <<<"$(summarize_seed_file_run "${run_tag}" "${SEED_EP200_REMAIN}")"
  log "done ${run_tag} result_files=${files} partial=${num}/${den}"
  printf 'ep200_remaining12\tbridge\t%s\tcompleted\t%s\t%s\tresult_files=%s\n' "${run_tag}" "${num}" "${den}" "${files}" >> "${SUMMARY_FILE}"
}

run_ep300_remaining1() {
  local gpus run_tag launch_log files num den
  gpus="$(wait_for_free_gpus 1)"
  run_tag="ep300_bridge_remaining1_${PIPELINE_TAG}"
  launch_log="${LOG_ROOT}/${run_tag}_launch.log"
  log "launch ${run_tag} gpus=${gpus}"
  LATENT_CKPT_PATH="${EP300_CKPT}" INFERENCE_MODE=bridge SEED_FILE="${SEED_EP300_REMAIN}" GPU_LIST="${gpus}" RUN_TAG="${run_tag}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > "${launch_log}" 2>&1
  wait_run_done "${run_tag}"
  read files num den <<<"$(summarize_seed_file_run "${run_tag}" "${SEED_EP300_REMAIN}")"
  log "done ${run_tag} result_files=${files} partial=${num}/${den}"
  printf 'ep300_remaining1\tbridge\t%s\tcompleted\t%s\t%s\tresult_files=%s\n' "${run_tag}" "${num}" "${den}" "${files}" >> "${SUMMARY_FILE}"
}

run_full_150() {
  local label=$1
  local ckpt=$2
  local gpus g1 g2 g3 arr run_tag1 run_tag2 run_tag3 files num den
  gpus="$(wait_for_free_gpus 8)"
  read -r -a arr <<< "${gpus}"
  g1="${arr[0]} ${arr[1]} ${arr[2]}"
  g2="${arr[3]} ${arr[4]} ${arr[5]}"
  g3="${arr[6]} ${arr[7]}"
  run_tag1="${label}_bridge150_g1_${PIPELINE_TAG}"
  run_tag2="${label}_bridge150_g2_${PIPELINE_TAG}"
  run_tag3="${label}_bridge150_g3_${PIPELINE_TAG}"
  log "launch ${label} bridge150 g1='${g1}' g2='${g2}' g3='${g3}'"
  LATENT_CKPT_PATH="${ckpt}" INFERENCE_MODE=bridge SEED_FILE="${SEED_G1}" GPU_LIST="${g1}" RUN_TAG="${run_tag1}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > "${LOG_ROOT}/${run_tag1}_launch.log" 2>&1
  LATENT_CKPT_PATH="${ckpt}" INFERENCE_MODE=bridge SEED_FILE="${SEED_G2}" GPU_LIST="${g2}" RUN_TAG="${run_tag2}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > "${LOG_ROOT}/${run_tag2}_launch.log" 2>&1
  LATENT_CKPT_PATH="${ckpt}" INFERENCE_MODE=bridge SEED_FILE="${SEED_G3}" GPU_LIST="${g3}" RUN_TAG="${run_tag3}" \
    bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh > "${LOG_ROOT}/${run_tag3}_launch.log" 2>&1
  wait_run_done "${label}_bridge150_g1_${PIPELINE_TAG}|${label}_bridge150_g2_${PIPELINE_TAG}|${label}_bridge150_g3_${PIPELINE_TAG}"
  read files num den <<<"$(summarize_full_150 "${PIPELINE_TAG}")"
  # summarize_full_150(run_tag) expects exact run tag; aggregate via label-specific glob instead
  read files num den <<<"$(python3 - <<PY
import glob, pathlib, re
label=${label@Q}
pipeline=${PIPELINE_TAG@Q}
base='/data/zhenyangfan/RoboTwin/eval_result/open_laptop/ACT_LatentCorr/demo_clean'
files=sorted(glob.glob(f'{base}/*{label}_bridge150_g*_{pipeline}*/**/_result.txt', recursive=True))
num=0; den=0
for f in files:
    txt=pathlib.Path(f).read_text().strip().splitlines(); val=float(txt[-1])
    m=re.search(r'_shard_(\d+)', f); shard=int(m.group(1)) if m else -1
    if '_g1_' in f or '_g2_' in f:
        d=17 if shard in (0,1) else 16
    else:
        d=25
    num += round(val*d); den += d
print(f'{len(files)}\t{num}\t{den}')
PY
)"
  log "done ${label} result_files=${files} result=${num}/${den}"
  printf '%s\tbridge\t%s\tcompleted\t%s\t%s\tresult_files=%s\n' "${label}" "${PIPELINE_TAG}" "${num}" "${den}" "${files}" >> "${SUMMARY_FILE}"
}

log "pipeline start tag=${PIPELINE_TAG}"
run_remaining12
run_ep300_remaining1
run_full_150 ep400 "${EP400_CKPT}"
run_full_150 ep500 "${EP500_CKPT}"
run_full_150 ep600 "${EP600_CKPT}"
run_full_150 ep700 "${EP700_CKPT}"
run_full_150 ep800 "${EP800_CKPT}"
run_full_150 ep900 "${EP900_CKPT}"
run_full_150 ep1000 "${EP1000_CKPT}"
log "pipeline complete"
