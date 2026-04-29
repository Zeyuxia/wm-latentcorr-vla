#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)

CURRENT_RUN_DIR="${CURRENT_RUN_DIR:-${SCRIPT_DIR}/outputs/stage1/robotwin_multitask_5_cam_high/20260428_235223-stage1_spatial_projector_no_aux_losses}"
CURRENT_FINAL_CKPT="${CURRENT_FINAL_CKPT:-${CURRENT_RUN_DIR}/stage1_step_001500.pt}"
QUEUE_LOG="${QUEUE_LOG:-${CURRENT_RUN_DIR}/queue_ratio05_after_current.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

NEXT_FAILURE_CORR_BATCH_RATIO="${NEXT_FAILURE_CORR_BATCH_RATIO:-0.5}"
NEXT_RUN_TAG="${NEXT_RUN_TAG:-stage1_spatial_projector_no_aux_losses_ratio05}"

log() {
  local ts
  ts=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[${ts}] $*" | tee -a "${QUEUE_LOG}"
}

mkdir -p "$(dirname "${QUEUE_LOG}")"
log "queue script: ${SCRIPT_PATH}"
log "waiting for current final checkpoint: ${CURRENT_FINAL_CKPT}"
log "next ratio=${NEXT_FAILURE_CORR_BATCH_RATIO}, next run_tag=${NEXT_RUN_TAG}"

while [ ! -f "${CURRENT_FINAL_CKPT}" ]; do
  if [ -f "${CURRENT_RUN_DIR}/log.log" ] && grep -qiE "Traceback|RuntimeError|CUDA out of memory|ChildFailedError" "${CURRENT_RUN_DIR}/log.log"; then
    log "detected failure marker in current log; not launching next run."
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

log "current final checkpoint found; launching next run."
(
  cd "${SCRIPT_DIR}"
  export FAILURE_CORR_BATCH_RATIO="${NEXT_FAILURE_CORR_BATCH_RATIO}"
  export RUN_TAG="${NEXT_RUN_TAG}"
  bash "${SCRIPT_DIR}/train_stage1.sh"
) >> "${QUEUE_LOG}" 2>&1
log "next run finished."
