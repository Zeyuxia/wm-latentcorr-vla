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
RUN_PREFIX=${RUN_PREFIX:-pi05_eval100_$(date +"%Y%m%d_%H%M%S")}
GROUP1_TAG=${GROUP1_TAG:-${RUN_PREFIX}_g1}
GROUP2_TAG=${GROUP2_TAG:-${RUN_PREFIX}_g2}
GPU_LIST_G1=${GPU_LIST_G1:-"0 1 2"}
GPU_LIST_G2=${GPU_LIST_G2:-"3 4 5"}
SEED_G1=${SEED_G1:-${ROBOTWIN_ROOT}/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group1.txt}
SEED_G2=${SEED_G2:-${ROBOTWIN_ROOT}/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group2.txt}
SUMMARY_DIR=${SUMMARY_DIR:-${SCRIPT_DIR}/outputs/logs/${RUN_PREFIX}_summary}
RESULT_ROOT=${RESULT_ROOT:-${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/PI05_LatentCorr/${TASK_CONFIG}}
SUMMARY_WATCH_SESSION=${RUN_PREFIX}_summary_watch
GROUP1_LOG_ROOT=${SCRIPT_DIR}/outputs/logs/${GROUP1_TAG}
GROUP2_LOG_ROOT=${SCRIPT_DIR}/outputs/logs/${GROUP2_TAG}

TASK_NAME="${TASK_NAME}" TASK_CONFIG="${TASK_CONFIG}" TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME}" MODEL_NAME="${MODEL_NAME}" CHECKPOINT_ID="${CHECKPOINT_ID}" CAMERA_MODE="${CAMERA_MODE}" PI0_STEP="${PI0_STEP}" ASSET_REPO_ID="${ASSET_REPO_ID}" SEED_FILE="${SEED_G1}" GPU_LIST="${GPU_LIST_G1}" RUN_TAG="${GROUP1_TAG}" RESULT_ROOT="${RESULT_ROOT}" \
  bash "${SCRIPT_DIR}/launch_parallel_eval.sh"

TASK_NAME="${TASK_NAME}" TASK_CONFIG="${TASK_CONFIG}" TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME}" MODEL_NAME="${MODEL_NAME}" CHECKPOINT_ID="${CHECKPOINT_ID}" CAMERA_MODE="${CAMERA_MODE}" PI0_STEP="${PI0_STEP}" ASSET_REPO_ID="${ASSET_REPO_ID}" SEED_FILE="${SEED_G2}" GPU_LIST="${GPU_LIST_G2}" RUN_TAG="${GROUP2_TAG}" RESULT_ROOT="${RESULT_ROOT}" \
  bash "${SCRIPT_DIR}/launch_parallel_eval.sh"

cat > "${SUMMARY_DIR}.watch.sh" <<EOF
#!/bin/bash
set -euo pipefail
while true; do
  active=\$(tmux ls 2>/dev/null | rg "^(${GROUP1_TAG}|${GROUP2_TAG})_shard_" | wc -l | tr -d ' ' || true)
  if [ "\${active}" = "0" ]; then
    break
  fi
  sleep 10
done
python3 ${SCRIPT_DIR@Q}/write_grouped_eval_summary.py \\
  --output-dir ${SUMMARY_DIR@Q} \\
  --log-root ${GROUP1_LOG_ROOT@Q} \\
  --log-root ${GROUP2_LOG_ROOT@Q} \\
  --result-root ${RESULT_ROOT@Q}
EOF
chmod +x "${SUMMARY_DIR}.watch.sh"
tmux new-session -d -s "${SUMMARY_WATCH_SESSION}" "${SUMMARY_DIR}.watch.sh"

echo "launched ${GROUP1_TAG} + ${GROUP2_TAG}"
echo "summary will be written to ${SUMMARY_DIR}"
