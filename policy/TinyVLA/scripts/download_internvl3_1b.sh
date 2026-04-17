#!/usr/bin/env bash

REPO_ID="OpenGVLab/InternVL3-1B"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/model_param/InternVL3-1B"
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_DELAY="${RETRY_DELAY:-5}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf command not found. Please install huggingface_hub first."
  exit 1
fi

mkdir -p "$LOCAL_DIR"

echo "Start downloading ${REPO_ID}"
echo "Target directory: ${LOCAL_DIR}"

attempt=1
while [ "$attempt" -le "$MAX_RETRIES" ]; do
  echo "----------------------------------------"
  echo "Attempt ${attempt}/${MAX_RETRIES}"

  hf download "$REPO_ID" \
    --repo-type model \
    --local-dir "$LOCAL_DIR"

  if [ $? -eq 0 ]; then
    echo "Download finished."
    exit 0
  fi

  if [ "$attempt" -lt "$MAX_RETRIES" ]; then
    echo "Download failed. Retry in ${RETRY_DELAY}s..."
    sleep "$RETRY_DELAY"
  else
    echo "Download failed after ${MAX_RETRIES} attempts."
  fi

  attempt=$((attempt + 1))
done

exit 1
