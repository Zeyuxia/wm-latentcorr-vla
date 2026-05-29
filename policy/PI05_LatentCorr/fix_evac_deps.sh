#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_PROJECT=${UV_PROJECT:-${SCRIPT_DIR}}
PYTHON_BIN="${UV_PROJECT}/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[PI05_LatentCorr] Python env not found: ${PYTHON_BIN}" >&2
  echo "[PI05_LatentCorr] Run ./setup_uv_env.sh first." >&2
  exit 1
fi

if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
required = {
    "pytorch_lightning": None,
    "open_clip": None,
    "kornia": None,
    "xformers": None,
    "moviepy": None,
}
ok = True
for module_name in required:
    try:
        __import__(module_name)
    except Exception:
        ok = False
        break
raise SystemExit(0 if ok else 1)
PY
then
  echo "[PI05_LatentCorr] EVAC deps already configured"
  exit 0
fi

"${PYTHON_BIN}" -m pip install \
  "setuptools<81" \
  "wandb==0.17.9" \
  "pytorch-lightning==1.9.5" \
  "open-clip-torch==2.22.0" \
  "kornia" \
  "moviepy==1.0.3" \
  "xformers==0.0.29.post3"

"${PYTHON_BIN}" - <<'PY'
mods = ["pytorch_lightning", "open_clip", "kornia", "xformers", "moviepy"]
for name in mods:
    mod = __import__(name)
    print(f"[PI05_LatentCorr] {name}={getattr(mod, '__version__', 'n/a')}")
PY
