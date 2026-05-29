#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
PYTHON_BIN="${UV_PROJECT}/.venv/bin/python"
REPLACEMENT_SRC="${ROBOTWIN_ROOT}/policy/pi05/src/openpi/models_pytorch/transformers_replace"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[PI05_LatentCorr] Python env not found: ${PYTHON_BIN}" >&2
  echo "[PI05_LatentCorr] Run ./setup_uv_env.sh first." >&2
  exit 1
fi

if [ ! -d "${REPLACEMENT_SRC}" ]; then
  echo "[PI05_LatentCorr] transformers_replace source not found: ${REPLACEMENT_SRC}" >&2
  exit 1
fi

if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
try:
    import transformers
    from transformers.models.siglip import check
    ok = transformers.__version__ == "4.53.2" and check.check_whether_transformers_replace_is_installed_correctly()
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
then
  echo "[PI05_LatentCorr] transformers_replace already configured"
  exit 0
fi

uv pip install --python "${PYTHON_BIN}" "transformers==4.53.2"

TRANSFORMERS_DIR="$("${PYTHON_BIN}" - <<'PY'
import pathlib
import transformers
print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)"

cp -r "${REPLACEMENT_SRC}/"* "${TRANSFORMERS_DIR}/"

"${PYTHON_BIN}" - <<'PY'
import transformers
from transformers.models.siglip import check

print(f"[PI05_LatentCorr] transformers={transformers.__version__}")
print(f"[PI05_LatentCorr] transformers_replace_ok={check.check_whether_transformers_replace_is_installed_correctly()}")
PY
