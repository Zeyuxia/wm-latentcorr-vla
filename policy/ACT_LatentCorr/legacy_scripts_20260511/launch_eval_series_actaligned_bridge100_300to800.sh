#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr

STAMP=$(date +"%Y%m%d_%H%M%S")
SERIES_TAG=${SERIES_TAG:-actaligned_bridge100_300to800_${STAMP}}
SERIES_DIR=/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/${SERIES_TAG}
SERIES_LOG=${SERIES_DIR}/series.log
SUMMARY_TSV=${SERIES_DIR}/summary.tsv
SUMMARY_TXT=${SERIES_DIR}/summary.txt
SESSION=${SERIES_TAG}

mkdir -p "${SERIES_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" \
  "/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_eval_series_actaligned_bridge100_300to800.sh ${SERIES_DIR@Q} ${SERIES_LOG@Q} ${SUMMARY_TSV@Q} ${SUMMARY_TXT@Q}"

echo "session=${SESSION}"
echo "series_dir=${SERIES_DIR}"
echo "series_log=${SERIES_LOG}"
echo "summary_tsv=${SUMMARY_TSV}"
echo "summary_txt=${SUMMARY_TXT}"
