#!/bin/bash
cd /data/zhenyangfan/RoboTwin/policy/ACT

# Base training parameters
train_tag="open_laptop_recover_transaltion_test"
gpu_ids=4
main_process_port=29500

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

# Common training hyper-parameters
seed=0
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

# World model parameters
enable_wm=true
# evac_ckpt=/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt
evac_ckpt=/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-03-16T21-19-22/logs/evac_robotwin_mixed50p12_2026-03-16T21-19-22/checkpoints/epoch=3124-step=12500.ckpt
evac_config=./evac/configs/robotwin/train_config.yaml
urdf_path=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
curobo_left_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
curobo_right_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
raw_data_dir=/data/zhenyangfan/RoboTwin/data/${task_name}/${task_config}/data
act_init_ckpt=./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/20260213_001933_1000epoch_baseline/policy_epoch_1000_seed_0.ckpt
max_rollout_steps=1
correction_force_generate=true
debug_correction_evac_rollout=true
rollout_exec_steps=16
recover_eval_enable=true
recover_eval_use_for_trigger=false
recover_eval_save_video=true
recover_eval_gripper_open_thresh=0.8
recover_eval_pos_thresh_m=0.03
recover_eval_rot_thresh_deg=10.0
recover_eval_nearest_window_radius=16
orient_weight=0.0573
gripper_penalty=1.0
debug_wm=true
debug_loss_batch_projection=true
export_correction_dataset=true
export_correction_dir=

# Online rollout perturbation options (effective only when enable_wm=true)
enable_perturb=true
perturb_prob=1.0
perturb_error_mode=open_laptop_pregrasp
perturb_open_laptop_pregrasp_close_prob=0.0
perturb_open_laptop_pregrasp_translation_prob=1.0
perturb_open_laptop_pregrasp_rotation_prob=0.0
sample_pregrasp_bias_enable=true
sample_pregrasp_prob=1.0
sample_phase_window_len=30
sample_pregrasp_keep_start_ratio=0.3
sample_pregrasp_keep_end_ratio=0.5
sample_skip_head_ratio=0.25
wm_corr_pregrasp_extra_enable=true
wm_corr_pregrasp_extra_ratio=0.5
perturb_eef_fail_gain=0.10
perturb_rot_max_deg=15
perturb_mag_random=true
perturb_mag_rand_min=1.00
perturb_mag_rand_max=1.40
perturb_active_joint_delta_thresh=0.01
perturb_active_gripper_delta_thresh=0.05

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
    --correction_force_generate ${correction_force_generate} \
    --debug_correction_evac_rollout ${debug_correction_evac_rollout} \
    --rollout_exec_steps ${rollout_exec_steps} \
    --recover_eval_enable ${recover_eval_enable} \
    --recover_eval_use_for_trigger ${recover_eval_use_for_trigger} \
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
    fi
    wm_flags="${wm_flags} --debug_loss_batch_projection ${debug_loss_batch_projection}"
fi

# Build perturb flags
perturb_flags="--enable_perturb ${enable_perturb} \
--perturb_prob ${perturb_prob} \
--perturb_error_mode ${perturb_error_mode} \
--sample_skip_head_ratio ${sample_skip_head_ratio} \
--perturb_open_laptop_pregrasp_close_prob ${perturb_open_laptop_pregrasp_close_prob} \
--perturb_open_laptop_pregrasp_translation_prob ${perturb_open_laptop_pregrasp_translation_prob} \
--perturb_open_laptop_pregrasp_rotation_prob ${perturb_open_laptop_pregrasp_rotation_prob} \
--perturb_eef_fail_gain ${perturb_eef_fail_gain} \
--perturb_rot_max_deg ${perturb_rot_max_deg} \
--perturb_mag_random ${perturb_mag_random} \
--perturb_mag_rand_min ${perturb_mag_rand_min} \
--perturb_mag_rand_max ${perturb_mag_rand_max} \
--perturb_active_joint_delta_thresh ${perturb_active_joint_delta_thresh} \
--perturb_active_gripper_delta_thresh ${perturb_active_gripper_delta_thresh} \
--sample_pregrasp_bias_enable ${sample_pregrasp_bias_enable} \
--sample_pregrasp_prob ${sample_pregrasp_prob} \
--sample_phase_window_len ${sample_phase_window_len} \
--sample_pregrasp_keep_start_ratio ${sample_pregrasp_keep_start_ratio} \
--sample_pregrasp_keep_end_ratio ${sample_pregrasp_keep_end_ratio} \
--wm_corr_pregrasp_extra_enable ${wm_corr_pregrasp_extra_enable} \
--wm_corr_pregrasp_extra_ratio ${wm_corr_pregrasp_extra_ratio} \
"

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

CUDA_VISIBLE_DEVICES=${gpu_ids} accelerate launch \
    ${gpu_flags} \
    imitate_episodes.py \
    ${train_flags} \
    ${perturb_flags} \
    ${wm_flags}
