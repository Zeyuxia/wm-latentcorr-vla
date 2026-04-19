#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

SOURCE_ROOT=
OUTPUT_ROOT=
TASK_NAMES=(
  open_laptop
  pick_dual_bottles
  put_bottles_dustbin
  place_burger_fries
  handover_block
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

python3 /data/zhenyangfan/RoboTwin/policy/SmolVLA/merge_multitask_failure_tables.py \
  --source_root "${SOURCE_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --task_names "${TASK_NAMES[@]}" \
  --failure_fail_recover_rate_thresh "${FAILURE_FAIL_RECOVER_RATE_THRESH}"

