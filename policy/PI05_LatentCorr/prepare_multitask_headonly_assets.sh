#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
PYTHON_BIN=${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
CAMERA_MODE=${CAMERA_MODE:-head_only}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-50}
DESCRIPTION_TYPE=${DESCRIPTION_TYPE:-seen}
MODE=${MODE:-image}
OVERWRITE=${OVERWRITE:-false}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR:-${SCRIPT_DIR}/outputs/assets}
NORM_MAX_FRAMES=${NORM_MAX_FRAMES:-}
TASKS=${TASKS:-"pick_dual_bottles open_laptop place_burger_fries put_bottles_dustbin handover_block"}
DRY_RUN=${DRY_RUN:-false}

if [ "${CAMERA_MODE}" != "head_only" ]; then
  echo "[PI05_LatentCorr] this helper is intended for single-view/head_only PI0.5 runs" >&2
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[PI05_LatentCorr] python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

# shellcheck disable=SC2206
TASK_ARR=(${TASKS})

for task in "${TASK_ARR[@]}"; do
  repo_id="robotwin/${task}_${TASK_CONFIG}_${EXPERT_DATA_NUM}_headonly"
  processed_dir="${SCRIPT_DIR}/outputs/processed_data/${task}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-${CAMERA_MODE}"
  raw_root="${ROBOTWIN_ROOT}/data/${task}/${TASK_CONFIG}"
  if [ ! -d "${raw_root}/data" ]; then
    echo "[PI05_LatentCorr] missing raw data for ${task}: ${raw_root}/data" >&2
    exit 1
  fi

  echo "[PI05_LatentCorr] asset plan | task=${task} repo_id=${repo_id} processed_dir=${processed_dir} raw_root=${raw_root}"
  if [ "${DRY_RUN}" = "true" ]; then
    continue
  fi

  echo "[PI05_LatentCorr] preparing ${task}"
  uv run --project "${UV_PROJECT}" python -m policy.PI05_LatentCorr.prepare_openloop_data \
    --task-name "${task}" \
    --task-config "${TASK_CONFIG}" \
    --expert-data-num "${EXPERT_DATA_NUM}" \
    --repo-id "${repo_id}" \
    --processed-dir "${processed_dir}" \
    --description-type "${DESCRIPTION_TYPE}" \
    --camera-mode "${CAMERA_MODE}" \
    --mode "${MODE}" \
    $([ "${OVERWRITE}" = "true" ] && echo "--overwrite")

  echo "[PI05_LatentCorr] computing norm stats for ${repo_id}"
  TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME}" \
  REPO_ID="${repo_id}" \
  ASSETS_BASE_DIR="${ASSETS_BASE_DIR}" \
  NORM_MAX_FRAMES="${NORM_MAX_FRAMES}" \
  CAMERA_MODE="${CAMERA_MODE}" \
  "${PYTHON_BIN}" - <<'PY'
import os

from policy.PI05_LatentCorr.common import build_train_config, ensure_norm_stats

train_config_name = os.environ["TRAIN_CONFIG_NAME"]
repo_id = os.environ["REPO_ID"]
assets_base_dir = os.environ["ASSETS_BASE_DIR"]
camera_mode = os.environ["CAMERA_MODE"]
norm_max_frames_raw = os.environ.get("NORM_MAX_FRAMES", "").strip()
norm_max_frames = int(norm_max_frames_raw) if norm_max_frames_raw else None

cfg = build_train_config(
    train_config_name=train_config_name,
    repo_id=repo_id,
    exp_name="prepare_multitask_headonly_assets",
    camera_mode=camera_mode,
    asset_id=repo_id,
    assets_base_dir=assets_base_dir,
)
ensure_norm_stats(cfg, max_frames=norm_max_frames)
PY
done

if [ "${DRY_RUN}" = "true" ]; then
  echo "[PI05_LatentCorr] DRY_RUN=true, asset path check passed; no files written."
  exit 0
fi

echo "[PI05_LatentCorr] finished preparing ${#TASK_ARR[@]} task assets"
