#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

SOURCE_ROOT=${SOURCE_ROOT:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-}
TASK_NAMES=(
  sim-open_laptop-demo_clean-50
  sim-pick_dual_bottles-demo_clean-50
  sim-put_bottles_dustbin-demo_clean-50
  sim-place_burger_fries-demo_clean-50
  sim-handover_block-demo_clean-50
)
FAILURE_FAIL_RECOVER_RATE_THRESH=0.5

if [ -z "${SOURCE_ROOT}" ]; then
  echo "SOURCE_ROOT is required" >&2
  exit 1
fi
if [ -z "${OUTPUT_ROOT}" ]; then
  echo "OUTPUT_ROOT is required" >&2
  exit 1
fi

python3 /data/zhenyangfan/RoboTwin/policy/SmolVLA/latentcorr/merge_multitask_failure_tables.py \
  --source_root "${SOURCE_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --task_names "${TASK_NAMES[@]}" \
  --failure_fail_recover_rate_thresh "${FAILURE_FAIL_RECOVER_RATE_THRESH}"
