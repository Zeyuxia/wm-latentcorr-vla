#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${ROOT}"

LLM=${LLM:-InternVL3}
ACTION_HEAD=${ACTION_HEAD:-unet_diffusion_policy}
TASK=${TASK:-robotwin_multitask_5_cam_high}
MODEL_PATH=${MNOP:-${ROOT}/model_param/InternVL3-1B}

BS=${BS:-64}
LR=${LR:-2e-5}
noise_samples=${noise_samples:-8}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-50}
LOGGING_STEPS=${LOGGING_STEPS:-5}
MASTER_PORT=${MASTER_PORT:-29604}
NUM_NODES=${NUM_NODES:-1}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
LORA_ENABLE=${LORA_ENABLE:-True}
LORA_MODULE=${LORA_MODULE:-vit llm}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-256}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
RUN_TAG=${RUN_TAG:-base_multi_task_lora}
RESUME_FROM=${RESUME_FROM:-}
PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES// /}

port_is_available() {
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    print("0")
else:
    print("1")
finally:
    s.close()
PY
}

find_free_port() {
  python3 - "$1" <<'PY'
import socket
import sys

start_port = int(sys.argv[1])
for port in range(start_port, start_port + 1000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        s.close()
        continue
    s.close()
    print(port)
    sys.exit(0)

raise SystemExit(f"Could not find a free port in [{start_port}, {start_port + 999}]")
PY
}

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  echo "CUDA_VISIBLE_DEVICES is empty." >&2
  exit 1
fi

if [ "$(port_is_available "${MASTER_PORT}")" != "1" ]; then
  OLD_MASTER_PORT="${MASTER_PORT}"
  MASTER_PORT=$(find_free_port "${MASTER_PORT}")
  echo "MASTER_PORT ${OLD_MASTER_PORT} is already in use; switching to ${MASTER_PORT}."
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
DETECTED_NUM_GPUS=${#GPU_IDS[@]}
if [ -n "${NUM_GPUS:-}" ] && [ "${NUM_GPUS}" != "${DETECTED_NUM_GPUS}" ]; then
  echo "NUM_GPUS=${NUM_GPUS} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; using detected value ${DETECTED_NUM_GPUS}."
fi
NUM_GPUS=${DETECTED_NUM_GPUS}

RUNS_ROOT=${ROOT}/${ACTION_HEAD}_results/${TASK}

resolve_resume_checkpoint() {
  local resume_path="$1"
  local latest_checkpoint=""

  if [ ! -e "${resume_path}" ]; then
    echo "Resume path does not exist: ${resume_path}" >&2
    exit 1
  fi

  resume_path=$(readlink -f "${resume_path}")

  if [[ "$(basename "${resume_path}")" == checkpoint-* ]]; then
    if [ ! -f "${resume_path}/trainer_state.json" ]; then
      echo "Invalid checkpoint directory: ${resume_path}" >&2
      exit 1
    fi
    echo "${resume_path}"
    return
  fi

  latest_checkpoint=$(find "${resume_path}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
  if [ -z "${latest_checkpoint}" ]; then
    echo "No checkpoint-* found under ${resume_path}" >&2
    exit 1
  fi
  if [ ! -f "${latest_checkpoint}/trainer_state.json" ]; then
    echo "Latest checkpoint is missing trainer_state.json: ${latest_checkpoint}" >&2
    exit 1
  fi

  echo "${latest_checkpoint}"
}

if [ -n "${RESUME_FROM}" ]; then
  RESUME_CKPT=$(resolve_resume_checkpoint "${RESUME_FROM}")
  OUTPUT=$(dirname "${RESUME_CKPT}")
  echo "Resuming from checkpoint: ${RESUME_CKPT}"
else
  mkdir -p "${RUNS_ROOT}"
  RUN_NAME="${RUN_TIMESTAMP}"
  if [ -n "${RUN_TAG}" ]; then
    RUN_NAME="${RUN_NAME}-${RUN_TAG}"
  fi
  OUTPUT="${RUNS_ROOT}/${RUN_NAME}"
  if [ -e "${OUTPUT}" ]; then
    echo "Output directory already exists: ${OUTPUT}" >&2
    exit 1
  fi
  mkdir -p "${OUTPUT}"
  RESUME_CKPT=""
fi

RUNS_ROOT=$(dirname "${OUTPUT}")
mkdir -p "${RUNS_ROOT}"
mkdir -p "${OUTPUT}/src"
cp -rT "${ROOT}/aloha_scripts" "${OUTPUT}/src/aloha_scripts"
cp -rT "${ROOT}/scripts" "${OUTPUT}/scripts"
cp -rT "${ROOT}/data_utils" "${OUTPUT}/src/data_utils"
cp -rT "${ROOT}/vla" "${OUTPUT}/src/vla"
cp -rT "${ROOT}/policy_heads" "${OUTPUT}/src/policy_heads"
cp "${SCRIPT_PATH}" "${OUTPUT}/launch_train_robotwin_aloha.sh"
ln -sfn "${OUTPUT}" "${RUNS_ROOT}/latest"

cat > "${OUTPUT}/run_meta.txt" <<EOF
script=${SCRIPT_PATH}
root=${ROOT}
task=${TASK}
action_head=${ACTION_HEAD}
model_path=${MODEL_PATH}
batch_size=${BS}
learning_rate=${LR}
noise_samples=${noise_samples}
max_steps=${MAX_STEPS}
save_steps=${SAVE_STEPS}
save_total_limit=${SAVE_TOTAL_LIMIT}
logging_steps=${LOGGING_STEPS}
num_gpus=${NUM_GPUS}
detected_num_gpus=${DETECTED_NUM_GPUS}
num_nodes=${NUM_NODES}
master_port=${MASTER_PORT}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
pythonnousersite=${PYTHONNOUSERSITE}
run_tag=${RUN_TAG}
run_timestamp=${RUN_TIMESTAMP}
output_dir=${OUTPUT}
resume_from_checkpoint=${RESUME_CKPT}
lora_enable=${LORA_ENABLE}
lora_module=${LORA_MODULE}
lora_r=${LORA_R}
lora_alpha=${LORA_ALPHA}
lora_dropout=${LORA_DROPOUT}
EOF

CMD=(
  deepspeed
  --master_port "${MASTER_PORT}"
  --num_gpus="${NUM_GPUS}"
  --num_nodes="${NUM_NODES}"
  ./train_vla.py
  --deepspeed scripts/zero2.json
  --action_dim 14
  --state_dim 14
  --flash_attn True
  --chunk_size 16
  --noise_samples "${noise_samples}"
  --policy_head_type "${ACTION_HEAD}"
  --episode_first False
  --task_name "${TASK}"
  --model_name_or_path "${MODEL_PATH}"
  --lora_enable "${LORA_ENABLE}"
  --lora_module "${LORA_MODULE}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --freeze_vision_tower False
  --freeze_backbone False
  --bf16 True
  --output_dir "${OUTPUT}"
  --max_steps "${MAX_STEPS}"
  --per_device_train_batch_size "${BS}"
  --gradient_accumulation_steps 1
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --save_only_model False
  --learning_rate "${LR}"
  --weight_decay 0.
  --warmup_ratio 0.
  --lr_scheduler_type cosine
  --logging_steps "${LOGGING_STEPS}"
  --tf32 True
  --model_max_length 2048
  --gradient_checkpointing True
  --dataloader_num_workers 8
  --report_to tensorboard
  --logging_dir "${OUTPUT}/log"
)

if [ -n "${RESUME_CKPT}" ]; then
  CMD+=(--resume_from_checkpoint "${RESUME_CKPT}")
fi

printf 'CUDA_VISIBLE_DEVICES=%q PYTHONNOUSERSITE=%q ' "${CUDA_VISIBLE_DEVICES}" "${PYTHONNOUSERSITE}" > "${OUTPUT}/launch_command.sh"
printf '%q ' "${CMD[@]}" >> "${OUTPUT}/launch_command.sh"
printf '\n' >> "${OUTPUT}/launch_command.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" "${CMD[@]}" 2>&1 | tee "${OUTPUT}/log.log"

echo "${OUTPUT}"
