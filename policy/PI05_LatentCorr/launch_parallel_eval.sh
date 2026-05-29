#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
MODEL_NAME=${MODEL_NAME:-pi05_openloop_default}
CHECKPOINT_ID=${CHECKPOINT_ID:-latest}
CAMERA_MODE=${CAMERA_MODE:-head_only}
PI0_STEP=${PI0_STEP:-50}
ASSET_REPO_ID=${ASSET_REPO_ID:-}
SEED_FILE=${SEED_FILE:-${ROBOTWIN_ROOT}/policy/ACT_LatentCorr/jobs/seed_success_50.txt}
GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
RUN_TAG=${RUN_TAG:-pi05_openloop_parallel_$(date +"%Y%m%d_%H%M%S")}
LOG_ROOT=${LOG_ROOT:-${SCRIPT_DIR}/outputs/logs/${RUN_TAG}}
RESULT_ROOT=${RESULT_ROOT:-${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/PI05_LatentCorr/${TASK_CONFIG}}
TMP_SEED_DIR=/tmp/${RUN_TAG}_seeds
SUMMARY_WATCH_SESSION="${RUN_TAG}_summary_watch"
mkdir -p "${LOG_ROOT}" "${TMP_SEED_DIR}"

mapfile -t GPUS < <(echo "${GPU_LIST}" | tr ' ' '\n' | sed '/^$/d')
NUM_SHARDS=${#GPUS[@]}
if [ "${NUM_SHARDS}" -eq 0 ]; then
  echo "GPU_LIST produced zero shards" >&2
  exit 1
fi

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
     TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME} \
     MODEL_NAME=${MODEL_NAME} \
     CHECKPOINT_ID=${CHECKPOINT_ID} \
     CAMERA_MODE=${CAMERA_MODE} \
     PI0_STEP=${PI0_STEP} \
     ASSET_REPO_ID=${ASSET_REPO_ID} \
     SEED_FILE=${shard_file} \
     EVAL_TAG=${RUN_TAG}_shard_${i} \
     ${SCRIPT_DIR}/eval_success.sh > ${log_file} 2>&1"
done

echo "${RUN_TAG}" > "${LOG_ROOT}/run_tag.txt"
echo "${TMP_SEED_DIR}" > "${LOG_ROOT}/seed_dir.txt"
cat > "${LOG_ROOT}/summarize_when_done.sh" <<EOF
#!/bin/bash
set -euo pipefail
while true; do
  active=\$(tmux ls 2>/dev/null | rg "^${RUN_TAG}_shard_" | wc -l | tr -d ' ' || true)
  if [ "\${active}" = "0" ]; then
    break
  fi
  sleep 10
done
python3 ${SCRIPT_DIR@Q}/write_parallel_eval_summary.py \\
  --run-tag ${RUN_TAG@Q} \\
  --log-root ${LOG_ROOT@Q} \\
  --seed-dir ${TMP_SEED_DIR@Q} \\
  --result-root ${RESULT_ROOT@Q}
EOF
chmod +x "${LOG_ROOT}/summarize_when_done.sh"
tmux new-session -d -s "${SUMMARY_WATCH_SESSION}" "${LOG_ROOT}/summarize_when_done.sh"

echo "launched ${NUM_SHARDS} eval shards under ${RUN_TAG}"
