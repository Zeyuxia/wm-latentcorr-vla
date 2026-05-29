#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

LATENT_CKPT_PATH=${LATENT_CKPT_PATH:-}
if [ -z "${LATENT_CKPT_PATH}" ]; then
  echo "LATENT_CKPT_PATH is required" >&2
  exit 1
fi

TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
INFERENCE_MODE=${INFERENCE_MODE:-bridge}
RUN_PREFIX=${RUN_PREFIX:-eval100_$(date +"%Y%m%d_%H%M%S")}
GROUP1_TAG=${GROUP1_TAG:-${RUN_PREFIX}_g1}
GROUP2_TAG=${GROUP2_TAG:-${RUN_PREFIX}_g2}
GPU_LIST_G1=${GPU_LIST_G1:-"0 1 2"}
GPU_LIST_G2=${GPU_LIST_G2:-"3 4 5"}
SEED_G1=${SEED_G1:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group1.txt}
SEED_G2=${SEED_G2:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/jobs/seed_success_150_teacher_group2.txt}
SUMMARY_DIR=${SUMMARY_DIR:-/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${RUN_PREFIX}_summary}
SUMMARY_WATCH_SESSION=${RUN_PREFIX}_summary_watch

LATENT_CKPT_PATH="${LATENT_CKPT_PATH}" INFERENCE_MODE="${INFERENCE_MODE}" SEED_FILE="${SEED_G1}" GPU_LIST="${GPU_LIST_G1}" RUN_TAG="${GROUP1_TAG}" CKPT_SETTING="${GROUP1_TAG}" \
  bash launch_parallel_eval.sh

LATENT_CKPT_PATH="${LATENT_CKPT_PATH}" INFERENCE_MODE="${INFERENCE_MODE}" SEED_FILE="${SEED_G2}" GPU_LIST="${GPU_LIST_G2}" RUN_TAG="${GROUP2_TAG}" CKPT_SETTING="${GROUP2_TAG}" \
  bash launch_parallel_eval.sh

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
python3 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_grouped_eval_summary.py \\
  --output-dir ${SUMMARY_DIR@Q} \\
  --log-root /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${GROUP1_TAG@Q} \\
  --log-root /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${GROUP2_TAG@Q}
EOF
chmod +x "${SUMMARY_DIR}.watch.sh"
tmux new-session -d -s "${SUMMARY_WATCH_SESSION}" "${SUMMARY_DIR}.watch.sh"

echo "launched ${GROUP1_TAG} + ${GROUP2_TAG}"
echo "summary will be written to ${SUMMARY_DIR}"
