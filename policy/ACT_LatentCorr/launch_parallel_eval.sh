#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

LATENT_CKPT_PATH=${LATENT_CKPT_PATH:-}
if [ -z "${LATENT_CKPT_PATH}" ]; then
  echo "LATENT_CKPT_PATH is required" >&2
  exit 1
fi

TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
INFERENCE_MODE=${INFERENCE_MODE:-pred}
SEED_FILE=${SEED_FILE:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_50.txt}
GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG:-false}
RUN_TAG=${RUN_TAG:-stage1_openloop_base_50seed_parallel_$(date +"%Y%m%d_%H%M%S")}
CKPT_SETTING=${CKPT_SETTING:-${RUN_TAG}}
EVAC_CKPT=${EVAC_CKPT:-}
EVAC_CONFIG=${EVAC_CONFIG:-}
URDF_PATH=${URDF_PATH:-}
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/zhenyangfan/runtime_cache}
LOG_ROOT=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_TAG}
TMP_SEED_DIR=${TMP_SEED_DIR:-${RUNTIME_ROOT}/tmp/${RUN_TAG}_seeds}
mkdir -p \
  "${LOG_ROOT}" \
  "${TMP_SEED_DIR}" \
  "${RUNTIME_ROOT}/tmp" \
  "${RUNTIME_ROOT}/torch_extensions" \
  "${RUNTIME_ROOT}/mplconfig" \
  "${RUNTIME_ROOT}/hf_home" \
  "${RUNTIME_ROOT}/wandb" \
  "${RUNTIME_ROOT}/xdg_cache" \
  "${RUNTIME_ROOT}/xdg_config"
SUMMARY_WATCH_SESSION="${RUN_TAG}_summary_watch"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_ROOT}/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_ROOT}/mplconfig}"
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${RUNTIME_ROOT}/wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${RUNTIME_ROOT}/wandb/config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${RUNTIME_ROOT}/xdg_config}"
TORCH_EXT_ROOT=${TORCH_EXTENSIONS_DIR}

mapfile -t GPUS < <(echo "${GPU_LIST}" | tr ' ' '\n' | sed '/^$/d')
NUM_SHARDS=${#GPUS[@]}
if [ "${NUM_SHARDS}" -eq 0 ]; then
  echo "GPU_LIST produced zero shards" >&2
  exit 1
fi

cleanup_stale_curobo_locks() {
  local ext_root="${TORCH_EXT_ROOT}/py310_cu121"
  local stale_files=(
    "${ext_root}/geom_cu/lock"
    "${ext_root}/geom_cu/.ninja_lock"
    "${ext_root}/kinematics_fused_cu/lock"
    "${ext_root}/kinematics_fused_cu/.ninja_lock"
  )
  for file in "${stale_files[@]}"; do
    if [ -f "${file}" ]; then
      rm -f "${file}"
      echo "[eval-prewarm] removed stale lock: ${file}"
    fi
  done
}

prewarm_eval_curobo_extensions() {
  local gpu="${GPUS[0]}"
  echo "[eval-prewarm] starting single-process warmup on gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" \
  TMPDIR="${TMPDIR}" \
  TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
  MPLCONFIGDIR="${MPLCONFIGDIR}" \
  HF_HOME="${HF_HOME}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
  PYTHONPATH="/data/zhenyangfan/RoboTwin:/data/zhenyangfan/RoboTwin/envs/curobo/src:/data/zhenyangfan/RoboTwin/envs/robot:${PYTHONPATH:-}" \
  /bin/bash -lc '
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate ACT
    python - <<'"'"'PY'"'"'
import importlib
mods = [
    "curobo.curobolib.kinematics",
    "curobo.curobolib.geom",
]
for name in mods:
    importlib.import_module(name)
    print(f"[eval-prewarm] imported {name}")
PY
  '
  echo "[eval-prewarm] finished"
}

cleanup_stale_curobo_locks
prewarm_eval_curobo_extensions

# Split seed file round-robin for stable shard sizes.
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  : > "${TMP_SEED_DIR}/shard_${i}.txt"
done

idx=0
while IFS= read -r seed; do
  [ -z "${seed}" ] && continue
  shard=$((idx % NUM_SHARDS))
  echo "${seed}" >> "${TMP_SEED_DIR}/shard_${shard}.txt"
  idx=$((idx + 1))
done < "${SEED_FILE}"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$i]}"
  shard_file="${TMP_SEED_DIR}/shard_${i}.txt"
  [ -s "${shard_file}" ] || continue
  session="${RUN_TAG}_shard_${i}"
  log_file="${LOG_ROOT}/shard_${i}.log"
  tmux new-session -d -s "${session}" \
    "GPU_ID=${gpu} \
     TASK_NAME=${TASK_NAME} \
     TASK_CONFIG=${TASK_CONFIG} \
     CKPT_SETTING=${CKPT_SETTING}_shard_${i} \
     LATENT_CKPT_PATH=${LATENT_CKPT_PATH} \
     INFERENCE_MODE=${INFERENCE_MODE} \
     EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG} \
     EVAC_CKPT=${EVAC_CKPT} \
     EVAC_CONFIG=${EVAC_CONFIG} \
     URDF_PATH=${URDF_PATH} \
     RUNTIME_ROOT=${RUNTIME_ROOT} \
     TMPDIR=${TMPDIR} \
     TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR} \
     MPLCONFIGDIR=${MPLCONFIGDIR} \
     HF_HOME=${HF_HOME} \
     HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
     TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
     WANDB_DIR=${WANDB_DIR} \
     WANDB_CACHE_DIR=${WANDB_CACHE_DIR} \
     WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} \
     XDG_CACHE_HOME=${XDG_CACHE_HOME} \
     XDG_CONFIG_HOME=${XDG_CONFIG_HOME} \
     SEED_FILE=${shard_file} \
     /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/eval_success.sh > ${log_file} 2>&1"
done

echo "${RUN_TAG}" > "${LOG_ROOT}/run_tag.txt"
echo "${TMP_SEED_DIR}" > "${LOG_ROOT}/seed_dir.txt"
cat > "${LOG_ROOT}/summarize_when_done.sh" <<EOF
#!/bin/bash
set -euo pipefail
while true; do
  active=\$( (tmux ls 2>/dev/null || true) | { grep -F "${RUN_TAG}_shard_" || true; } | wc -l | tr -d ' ' )
  if [ "\${active}" = "0" ]; then
    break
  fi
  sleep 10
done
python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_parallel_eval_summary.py \\
  --run-tag ${RUN_TAG@Q} \\
  --log-root ${LOG_ROOT@Q} \\
  --seed-dir ${TMP_SEED_DIR@Q}
python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/summarize_total_success.py \\
  --run-tag ${RUN_TAG@Q} \\
  --log-root ${LOG_ROOT@Q} \\
  --seed-dir ${TMP_SEED_DIR@Q}
EOF
chmod +x "${LOG_ROOT}/summarize_when_done.sh"
tmux new-session -d -s "${SUMMARY_WATCH_SESSION}" "${LOG_ROOT}/summarize_when_done.sh"
echo "launched ${NUM_SHARDS} eval shards under ${RUN_TAG}"
