#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CAMERA_MODE=${CAMERA_MODE:-dual_view}
export SECONDARY_CAMERA=${SECONDARY_CAMERA:-right_wrist}
export TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
export REPO_ID=${REPO_ID:-robotwin/multitask5_demo_clean_50_dualview_${SECONDARY_CAMERA}}
export EXP_NAME=${EXP_NAME:-pi05_robotwin_multitask5_dualview_${SECONDARY_CAMERA}_$(date +"%Y%m%d_%H%M%S")}

exec bash "${SCRIPT_DIR}/train_robotwin_multitask_openloop.sh"
