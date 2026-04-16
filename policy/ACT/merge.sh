#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_PY="${SCRIPT_DIR}/imitate_episodes_pkg/merge_failure_tables.py"

failure_explore_dir="./act_ckpt/act-open_laptop/demo_clean-50/20260412_112518_open_laptop_closed_loop_exploration_multigpu_test/failure_explore"
out_dir=""
failure_fail_recover_rate_thresh="0.5"

if [ ! -f "${MERGE_PY}" ]; then
  echo "merge script not found: ${MERGE_PY}"
  exit 1
fi

CMD=(
  python3 "${MERGE_PY}"
  --failure_dir "${failure_explore_dir}"
  --failure_fail_recover_rate_thresh "${failure_fail_recover_rate_thresh}"
)

if [ -n "${out_dir}" ]; then
  CMD+=(--out_dir "${out_dir}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "Done."
if [ -n "${out_dir}" ]; then
  echo "failure_table_path=${out_dir}"
else
  echo "failure_table_path=${failure_explore_dir}"
fi
