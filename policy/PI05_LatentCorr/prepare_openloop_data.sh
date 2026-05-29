#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-50}
CAMERA_MODE=${CAMERA_MODE:-head_only}
DEFAULT_REPO_SUFFIX=""
if [ "${CAMERA_MODE}" = "head_only" ]; then
  DEFAULT_REPO_SUFFIX="_headonly"
fi
REPO_ID=${REPO_ID:-robotwin/${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}${DEFAULT_REPO_SUFFIX}}
PROCESSED_DIR=${PROCESSED_DIR:-}
DESCRIPTION_TYPE=${DESCRIPTION_TYPE:-seen}
MODE=${MODE:-image}
OVERWRITE=${OVERWRITE:-false}

EXTRA_ARGS=()
if [ -n "${PROCESSED_DIR}" ]; then
  EXTRA_ARGS+=(--processed-dir "${PROCESSED_DIR}")
fi
if [ "${OVERWRITE}" = "true" ]; then
  EXTRA_ARGS+=(--overwrite)
fi

uv run --project "${UV_PROJECT}" python -m policy.PI05_LatentCorr.prepare_openloop_data \
  --task-name "${TASK_NAME}" \
  --task-config "${TASK_CONFIG}" \
  --expert-data-num "${EXPERT_DATA_NUM}" \
  --repo-id "${REPO_ID}" \
  --description-type "${DESCRIPTION_TYPE}" \
  --camera-mode "${CAMERA_MODE}" \
  --mode "${MODE}" \
  "${EXTRA_ARGS[@]}"
