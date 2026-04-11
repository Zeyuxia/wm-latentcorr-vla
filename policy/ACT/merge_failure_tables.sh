#!/bin/bash
set -euo pipefail

# Usage:
#   ./merge_failure_tables.sh <failure_explore_dir> [epoch] [out_dir]
#
# Examples:
#   ./merge_failure_tables.sh ./act_ckpt/act-open_laptop/demo_clean-50/20260411_xxx/failure_explore
#   ./merge_failure_tables.sh ./act_ckpt/.../failure_explore 3
#   ./merge_failure_tables.sh ./act_ckpt/.../failure_explore 3 ./act_ckpt/.../failure_explore/merged

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_PY="${SCRIPT_DIR}/imitate_episodes_pkg/merge_failure_tables.py"

if [ ! -f "${MERGE_PY}" ]; then
  echo "merge script not found: ${MERGE_PY}"
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 <failure_explore_dir> [epoch] [out_dir]"
  exit 1
fi

FAILURE_DIR="$1"
EPOCH="${2:-}"
OUT_DIR="${3:-}"

CMD=(python3 "${MERGE_PY}" --failure_dir "${FAILURE_DIR}")
if [ -n "${EPOCH}" ]; then
  CMD+=(--epoch "${EPOCH}")
fi
if [ -n "${OUT_DIR}" ]; then
  CMD+=(--out_dir "${OUT_DIR}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "Done."
echo "You can now use:"
echo "  failure_mode=train"
if [ -n "${OUT_DIR}" ]; then
  echo "  failure_table_path=${OUT_DIR}"
else
  echo "  failure_table_path=${FAILURE_DIR}"
fi

