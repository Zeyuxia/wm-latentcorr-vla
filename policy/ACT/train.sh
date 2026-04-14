#!/bin/bash
cd /data/zhenyangfan/RoboTwin/policy/ACT
source /data/miniconda3/etc/profile.d/conda.sh
conda activate ACT

# Base training parameters
train_tag="open_laptop_0413_001944_train_test"
gpu_ids=0,1,2,3
main_process_port=29500
seed=0

# Single-task settings (used when task_names is empty)
task_name=open_laptop
task_config=demo_clean
expert_data_num=50
# Multi-task settings (task_names non-empty -> override single-task data source)
# Format example:
# task_names="sim-open_laptop-demo_clean-50,sim-blocks_ranking_rgb-demo_clean-50"
# task_weights="1.0,1.0"
task_names=
task_weights=

# ACT Parameters
policy_class=ACT
kl_weight=10
chunk_size=50
hidden_dim=512
batch_size=4
dim_feedforward=3200
num_epochs=2000
lr=4e-5
save_freq=10
state_dim=14
act_init_ckpt=./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/20260213_001933_1000epoch_baseline/policy_epoch_1000_seed_0.ckpt

# EVAC parameters
enable_wm=true
# evac_ckpt=/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt
evac_ckpt=/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-03-16T21-19-22/logs/evac_robotwin_mixed50p12_2026-03-16T21-19-22/checkpoints/epoch=3499-step=14000.ckpt
evac_config=./evac/configs/robotwin/train_config.yaml
urdf_path=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
curobo_left_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
curobo_right_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
raw_data_dir=/data/zhenyangfan/RoboTwin/data/${task_name}/${task_config}/data
max_rollout_steps=1

# Online rollout perturbation options (effective only when enable_wm=true).
# Perturbation on/off is controlled by failure_mode:
# off/explore -> disabled, train -> enabled.
sample_phase_window_len=20
sample_skip_head_ratio=0.6
perturb_active_joint_delta_thresh=0.01
perturb_active_gripper_delta_thresh=0.05
rollout_exec_steps=16
failure_mode=train
failure_table_path=/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-open_laptop/demo_clean-50/20260413_001944_open_laptop_closed_loop_exploration_multigpu_test_thresh04_recovervideo/failure_explore/failure_table.json
failure_table_dir=
failure_phase_bins=3
failure_translation_dir_bins=6
failure_translation_mag_bins=3
failure_rotation_dir_bins=6
failure_rotation_mag_bins=3
failure_corr_batch_ratio=0.5
failure_explore_k=4
perturb_eef_fail_gain=0.12
perturb_rot_max_deg=24
recover_eval_enable=true
recover_eval_save_video=false
recover_eval_gripper_open_thresh=0.8
recover_eval_pos_thresh_m=0.04
recover_eval_rot_thresh_deg=10.0
recover_eval_nearest_window_radius=16
orient_weight=0.0573
gripper_penalty=1.0

# Correction Parameters
debug_wm=true
debug_wm_all_ranks=true
debug_loss_batch_projection=true
debug_correction_evac_rollout=false
export_correction_dataset=true
export_correction_dir=

# Build saving dirs
timestamp=$(date +"%Y%m%d_%H%M%S")
if [ -n "${train_tag}" ]; then
    safe_train_tag=$(echo "${train_tag}" | sed 's/[^0-9A-Za-z._-]/_/g')
    timestamp="${timestamp}_${safe_train_tag}"
fi
ckpt_dir=./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/${timestamp}
mkdir -p ${ckpt_dir}
cp "$0" ${ckpt_dir}/train.sh

# Bulid gpu flags
num_gpus=$(echo ${gpu_ids} | awk -F',' '{print NF}')
if [ ${num_gpus} -gt 1 ]; then
    multi_gpu_flag="--multi_gpu"
else
    multi_gpu_flag=""
fi
gpu_flags="${multi_gpu_flag} \
--num_processes ${num_gpus} \
--main_process_port ${main_process_port} \
"

# Build world model flags
wm_flags=""
if [ "${enable_wm}" = "true" ]; then
    wm_flags="--enable_wm_correction \
    --evac_ckpt ${evac_ckpt} \
    --evac_config ${evac_config} \
    --urdf_path ${urdf_path} \
    --curobo_left_yml ${curobo_left_yml} \
    --curobo_right_yml ${curobo_right_yml} \
    --raw_data_dir ${raw_data_dir} \
    --act_init_ckpt ${act_init_ckpt} \
    --max_rollout_steps ${max_rollout_steps} \
    --debug_correction_evac_rollout ${debug_correction_evac_rollout} \
    --rollout_exec_steps ${rollout_exec_steps} \
    --recover_eval_enable ${recover_eval_enable} \
    --recover_eval_save_video ${recover_eval_save_video} \
    --recover_eval_gripper_open_thresh ${recover_eval_gripper_open_thresh} \
    --recover_eval_pos_thresh_m ${recover_eval_pos_thresh_m} \
    --recover_eval_rot_thresh_deg ${recover_eval_rot_thresh_deg} \
    --recover_eval_nearest_window_radius ${recover_eval_nearest_window_radius} \
    --orient_weight ${orient_weight} \
    --gripper_penalty ${gripper_penalty} \
    --export_correction_dataset ${export_correction_dataset} \
    "
    if [ -n "${export_correction_dir}" ]; then
        wm_flags="${wm_flags} --export_correction_dir ${export_correction_dir}"
    fi
    if [ "${debug_wm}" = "true" ]; then
        wm_flags="${wm_flags} --debug_wm_correction"
        wm_flags="${wm_flags} --debug_wm_all_ranks ${debug_wm_all_ranks}"
    fi
    wm_flags="${wm_flags} --debug_loss_batch_projection ${debug_loss_batch_projection}"
fi

# Build perturb flags
perturb_flags="--sample_skip_head_ratio ${sample_skip_head_ratio} \
--perturb_eef_fail_gain ${perturb_eef_fail_gain} \
--perturb_rot_max_deg ${perturb_rot_max_deg} \
--perturb_active_joint_delta_thresh ${perturb_active_joint_delta_thresh} \
--perturb_active_gripper_delta_thresh ${perturb_active_gripper_delta_thresh} \
--sample_phase_window_len ${sample_phase_window_len} \
"

# Build failure-mode flags
failure_flags="--failure_mode ${failure_mode} \
--failure_phase_bins ${failure_phase_bins} \
--failure_translation_dir_bins ${failure_translation_dir_bins} \
--failure_translation_mag_bins ${failure_translation_mag_bins} \
--failure_rotation_dir_bins ${failure_rotation_dir_bins} \
--failure_rotation_mag_bins ${failure_rotation_mag_bins} \
--failure_corr_batch_ratio ${failure_corr_batch_ratio} \
--failure_explore_k ${failure_explore_k} \
"
if [ -n "${failure_table_path}" ]; then
    failure_flags="${failure_flags} --failure_table_path ${failure_table_path}"
fi
if [ -n "${failure_table_dir}" ]; then
    failure_flags="${failure_flags} --failure_table_dir ${failure_table_dir}"
fi

# Build ACT training flags
train_flags="--task_name sim-${task_name}-${task_config}-${expert_data_num} \
--ckpt_dir ${ckpt_dir} \
--policy_class ${policy_class} \
--kl_weight ${kl_weight} \
--chunk_size ${chunk_size} \
--hidden_dim ${hidden_dim} \
--batch_size ${batch_size} \
--dim_feedforward ${dim_feedforward} \
--num_epochs ${num_epochs} \
--lr ${lr} \
--save_freq ${save_freq} \
--state_dim ${state_dim} \
--seed ${seed} \
"
if [ -n "${task_names}" ]; then
    train_flags="${train_flags} --multi_task_names ${task_names}"
fi
if [ -n "${task_weights}" ]; then
    train_flags="${train_flags} --multi_task_weights ${task_weights}"
fi
if [ -n "${act_init_ckpt}" ]; then
    train_flags="${train_flags} --act_init_ckpt ${act_init_ckpt}"
fi

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${gpu_ids} accelerate launch \
    ${gpu_flags} \
    imitate_episodes.py \
    ${train_flags} \
    ${perturb_flags} \
    ${failure_flags} \
    ${wm_flags}
