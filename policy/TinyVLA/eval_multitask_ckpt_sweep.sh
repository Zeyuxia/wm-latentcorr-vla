#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TASKS=(
    open_laptop
    pick_dual_bottles
    put_bottles_dustbin
    place_burger_fries
    handover_block
)

CKPTS=(1000 2000 3000 4000)
GPUS=(0 1 2 3 6)

task_config=${TASK_CONFIG:-demo_clean}
expert_data_num=${EXPERT_DATA_NUM:-50}
seed=${SEED:-0}
run_root=${RUN_ROOT:-${SCRIPT_DIR}/unet_diffusion_policy_results/robotwin_multitask_5_cam_high/20260416_235212-base_multi_task}
log_dir="${run_root}/log/eval_multitask_ckpt_sweep"
mkdir -p "${log_dir}"

job_idx=0
for ckpt in "${CKPTS[@]}"; do
    for task_name in "${TASKS[@]}"; do
        gpu=${GPUS[$((job_idx % ${#GPUS[@]}))]}
        log_file="${log_dir}/ckpt${ckpt}_${task_name}_gpu${gpu}.log"

        (
            export RUN_ROOT="${run_root}"
            export EVAL_TAG="ckpt${ckpt}_${task_name}"
            bash "${SCRIPT_DIR}/eval.sh" \
                "${task_name}" \
                "${task_config}" \
                "${ckpt}" \
                "${expert_data_num}" \
                "${seed}" \
                "${gpu}"
        ) > "${log_file}" 2>&1 &

        echo "launched checkpoint ${ckpt} task ${task_name} on gpu ${gpu}, log: ${log_file}"
        job_idx=$((job_idx + 1))
    done
done

wait
echo "all checkpoint sweep jobs finished"
