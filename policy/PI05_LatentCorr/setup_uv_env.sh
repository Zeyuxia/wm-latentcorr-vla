#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${SCRIPT_DIR}/outputs/openpi_cache}
TMPDIR=${TMPDIR:-${SCRIPT_DIR}/outputs/tmp}

mkdir -p "${OPENPI_DATA_HOME}" "${TMPDIR}"
export OPENPI_DATA_HOME
export TMPDIR

uv sync --project "${UV_PROJECT}"
bash "${SCRIPT_DIR}/fix_transformers_replace.sh"
bash "${SCRIPT_DIR}/fix_evac_deps.sh"
