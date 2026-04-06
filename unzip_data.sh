#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/data/zhenyangfan/RoboTwin/data"
DATASET_ROOT="${DATA_ROOT}/dataset"

if [ ! -d "${DATASET_ROOT}" ]; then
  echo "dataset dir not found: ${DATASET_ROOT}"
  exit 1
fi

found_any=0

while IFS= read -r -d '' archive; do
  found_any=1
  task_dir="$(dirname "${archive}")"
  task_name="$(basename "${task_dir}")"
  out_dir="${DATA_ROOT}/${task_name}/demo_clean"
  tmp_dir="$(mktemp -d "/tmp/rt_unpack_${task_name}_XXXXXX")"

  echo "[unpack] task=${task_name} archive=${archive}"

  case "${archive}" in
    *.zip) unzip -q -o "${archive}" -d "${tmp_dir}" ;;
    *.tar) tar -xf "${archive}" -C "${tmp_dir}" ;;
    *.tar.gz|*.tgz) tar -xzf "${archive}" -C "${tmp_dir}" ;;
    *.tar.bz2|*.tbz2) tar -xjf "${archive}" -C "${tmp_dir}" ;;
    *.tar.xz) tar -xJf "${archive}" -C "${tmp_dir}" ;;
    *)
      echo "  skip unsupported format: ${archive}"
      rm -rf "${tmp_dir}"
      continue
      ;;
  esac

  mapfile -t top_dirs < <(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
  if [ "${#top_dirs[@]}" -eq 1 ]; then
    src_dir="${top_dirs[0]}"
  else
    src_dir="${tmp_dir}"
  fi

  mkdir -p "${out_dir}"
  rsync -a --delete "${src_dir}/" "${out_dir}/"
  rm -rf "${tmp_dir}"

  echo "  -> synced to ${out_dir}"
done < <(
  find "${DATASET_ROOT}" -mindepth 2 -maxdepth 2 -type f \
    \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.tar.bz2" -o -name "*.tbz2" -o -name "*.tar.xz" \) \
    -print0
)

if [ "${found_any}" -eq 0 ]; then
  echo "no archives found under ${DATASET_ROOT}/<task>/"
  exit 1
fi

echo "[done] all archives processed."
