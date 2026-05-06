#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
PYTHON_BIN=${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
CAMERA_MODE=${CAMERA_MODE:-dual_view}
SECONDARY_CAMERA=${SECONDARY_CAMERA:-right_wrist}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-50}
TASKS=${TASKS:-"pick_dual_bottles open_laptop place_burger_fries put_bottles_dustbin handover_block"}
REPO_ID=${REPO_ID:-robotwin/multitask5_${TASK_CONFIG}_${EXPERT_DATA_NUM}_dualview_${SECONDARY_CAMERA}}
PROCESSED_DIR=${PROCESSED_DIR:-${SCRIPT_DIR}/outputs/processed_data/multitask5-${TASK_CONFIG}-${EXPERT_DATA_NUM}-dual_view_${SECONDARY_CAMERA}}
PREPARE_PER_TASK_ASSETS=${PREPARE_PER_TASK_ASSETS:-true}
DESCRIPTION_TYPE=${DESCRIPTION_TYPE:-seen}
MODE=${MODE:-image}
OVERWRITE=${OVERWRITE:-false}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR:-${SCRIPT_DIR}/outputs/assets}
NORM_MAX_FRAMES=${NORM_MAX_FRAMES:-}
DRY_RUN=${DRY_RUN:-false}

if [ "${CAMERA_MODE}" != "dual_view" ]; then
  echo "[PI05_RobotWin] this helper is intended for CAMERA_MODE=dual_view." >&2
  exit 1
fi

if [ "${SECONDARY_CAMERA}" != "left_wrist" ] && [ "${SECONDARY_CAMERA}" != "right_wrist" ]; then
  echo "[PI05_RobotWin] SECONDARY_CAMERA must be left_wrist or right_wrist." >&2
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[PI05_RobotWin] python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

# shellcheck disable=SC2206
TASK_ARR=(${TASKS})
for task in "${TASK_ARR[@]}"; do
  raw_root="${ROBOTWIN_ROOT}/data/${task}/${TASK_CONFIG}"
  if [ ! -d "${raw_root}/data" ]; then
    echo "[PI05_RobotWin] missing raw data for ${task}: ${raw_root}/data" >&2
    exit 1
  fi
done

echo "[PI05_RobotWin] multitask dual-view data plan"
echo "  tasks: ${TASK_ARR[*]}"
echo "  secondary_camera: ${SECONDARY_CAMERA}"
echo "  repo_id: ${REPO_ID}"
echo "  processed_dir: ${PROCESSED_DIR}"
echo "  assets_base_dir: ${ASSETS_BASE_DIR}"
echo "  prepare_per_task_assets: ${PREPARE_PER_TASK_ASSETS}"
if [ "${DRY_RUN}" = "true" ]; then
  echo "[PI05_RobotWin] DRY_RUN=true, path check passed; no files written."
  exit 0
fi

EXTRA_ARGS=()
if [ "${OVERWRITE}" = "true" ]; then
  EXTRA_ARGS+=(--overwrite)
fi

uv run --project "${UV_PROJECT}" python -m policy.pi05.prepare_robotwin_multitask_data \
  --task-names "${TASK_ARR[@]}" \
  --task-config "${TASK_CONFIG}" \
  --expert-data-num "${EXPERT_DATA_NUM}" \
  --repo-id "${REPO_ID}" \
  --processed-dir "${PROCESSED_DIR}" \
  --description-type "${DESCRIPTION_TYPE}" \
  --camera-mode "${CAMERA_MODE}" \
  --secondary-camera "${SECONDARY_CAMERA}" \
  --mode "${MODE}" \
  "${EXTRA_ARGS[@]}"

echo "[PI05_RobotWin] computing norm stats for ${REPO_ID}"
TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME}" \
REPO_ID="${REPO_ID}" \
ASSETS_BASE_DIR="${ASSETS_BASE_DIR}" \
NORM_MAX_FRAMES="${NORM_MAX_FRAMES}" \
CAMERA_MODE="${CAMERA_MODE}" \
SECONDARY_CAMERA="${SECONDARY_CAMERA}" \
"${PYTHON_BIN}" - <<'PY'
import os

from policy.pi05.robotwin_common import build_train_config, ensure_norm_stats

norm_max_frames_raw = os.environ.get("NORM_MAX_FRAMES", "").strip()
cfg = build_train_config(
    train_config_name=os.environ["TRAIN_CONFIG_NAME"],
    repo_id=os.environ["REPO_ID"],
    exp_name="prepare_robotwin_multitask_dualview",
    camera_mode=os.environ["CAMERA_MODE"],
    secondary_camera=os.environ["SECONDARY_CAMERA"],
    asset_id=os.environ["REPO_ID"],
    assets_base_dir=os.environ["ASSETS_BASE_DIR"],
)
ensure_norm_stats(cfg, max_frames=(int(norm_max_frames_raw) if norm_max_frames_raw else None))
PY

if [ "${PREPARE_PER_TASK_ASSETS}" = "true" ]; then
  for task in "${TASK_ARR[@]}"; do
    task_repo_id="robotwin/${task}_${TASK_CONFIG}_${EXPERT_DATA_NUM}_dualview_${SECONDARY_CAMERA}"
    task_processed_dir="${SCRIPT_DIR}/outputs/processed_data/${task}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-dual_view_${SECONDARY_CAMERA}"
    echo "[PI05_RobotWin] preparing per-task assets for ${task}"
    uv run --project "${UV_PROJECT}" python -m policy.pi05.prepare_robotwin_data \
      --task-name "${task}" \
      --task-config "${TASK_CONFIG}" \
      --expert-data-num "${EXPERT_DATA_NUM}" \
      --repo-id "${task_repo_id}" \
      --processed-dir "${task_processed_dir}" \
      --description-type "${DESCRIPTION_TYPE}" \
      --camera-mode "${CAMERA_MODE}" \
      --secondary-camera "${SECONDARY_CAMERA}" \
      --mode "${MODE}" \
      "${EXTRA_ARGS[@]}"

    TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME}" \
    REPO_ID="${task_repo_id}" \
    ASSETS_BASE_DIR="${ASSETS_BASE_DIR}" \
    NORM_MAX_FRAMES="${NORM_MAX_FRAMES}" \
    CAMERA_MODE="${CAMERA_MODE}" \
    SECONDARY_CAMERA="${SECONDARY_CAMERA}" \
    "${PYTHON_BIN}" - <<'PY'
import os

from policy.pi05.robotwin_common import build_train_config, ensure_norm_stats

norm_max_frames_raw = os.environ.get("NORM_MAX_FRAMES", "").strip()
cfg = build_train_config(
    train_config_name=os.environ["TRAIN_CONFIG_NAME"],
    repo_id=os.environ["REPO_ID"],
    exp_name="prepare_robotwin_per_task_dualview",
    camera_mode=os.environ["CAMERA_MODE"],
    secondary_camera=os.environ["SECONDARY_CAMERA"],
    asset_id=os.environ["REPO_ID"],
    assets_base_dir=os.environ["ASSETS_BASE_DIR"],
)
ensure_norm_stats(cfg, max_frames=(int(norm_max_frames_raw) if norm_max_frames_raw else None))
PY
  done
fi

echo "[PI05_RobotWin] finished dual-view multitask data and norm stats"
