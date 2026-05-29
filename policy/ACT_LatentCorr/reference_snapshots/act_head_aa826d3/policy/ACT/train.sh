#!/bin/bash
cd /data/zhenyangfan/RoboTwin/policy/ACT

# Base training parameters
gpu_ids=0
main_process_port=29500
task_name=open_laptop
task_config=demo_clean
expert_data_num=50
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
evac_ckpt=/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-03-16T21-19-22/logs/evac_robotwin_mixed50p12_2026-03-16T21-19-22/checkpoints/epoch=5624-step=22500.ckpt
evac_config=./evac/configs/robotwin/train_config.yaml
urdf_path=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
curobo_left_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
curobo_right_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
raw_data_dir=/data/zhenyangfan/RoboTwin/data/${task_name}/${task_config}/data
act_init_ckpt=./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/20260213_001933/policy_epoch_1000_seed_0.ckpt
max_rollout_steps=1
target_mode=backward
target_lookahead_steps=0
min_dist_fallback_force_correction=true
min_dist_recover_ratio=0.75
debug_recover_eval_rollout=true
debug_correction_evac_rollout=true
correction_interp_nearest_enable=false
correction_interp_prefix_ratio=0.6
correction_planner_prefix_ratio=0.5
correction_compose_gt_tail_enable=true
rollout_exec_steps=16
correction_weight=2.0
orient_weight=0.0573
gripper_penalty=1.0
recover_gripper_penalty=0.0
debug_wm=true
export_correction_dataset=true
export_correction_dir=

# Online rollout perturbation options (effective only when enable_wm=true)
enable_perturb=true
perturb_prob=1.0
perturb_error_mode=open_laptop_pregrasp
perturb_open_laptop_pregrasp_close_prob=0.5
perturb_open_laptop_pregrasp_translation_prob=0.5
perturb_open_laptop_pregrasp_rotation_prob=0.0
perturb_eef_fail_gain=0.05
perturb_rot_max_deg=15
perturb_mag_random=true
perturb_mag_rand_min=1.00
perturb_mag_rand_max=1.40
perturb_reject_sampling_enable=true
perturb_reject_max_trials=4
perturb_reject_dir_jitter_eps=0.2
perturb_active_joint_delta_thresh=0.01
perturb_active_gripper_delta_thresh=0.05
nearest_window_radius=12
perturb_gripper_close_min=0.10
perturb_gripper_open_max=0.90
perturb_gripper_fast_ratio=0.20
lr_sched_enable=true
lr_warmup_steps=30
lr_min_ratio=0.1
sp_reg_enable=true
sp_reg_lambda=1e-5
sample_pregrasp_bias_enable=true
sample_pregrasp_prob=1.0
sample_pregrasp_phase_window_len=24
sample_pregrasp_keep_start_ratio=0.0
sample_pregrasp_keep_end_ratio=0.5
sample_skip_head_ratio=0.25
wm_corr_pregrasp_extra_enable=true
wm_corr_pregrasp_extra_ratio=0.5



# Build saving dirs
timestamp=$(date +"%Y%m%d_%H%M%S")
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
    --target_mode ${target_mode} \
    --target_lookahead_steps ${target_lookahead_steps} \
    --min_dist_fallback_force_correction ${min_dist_fallback_force_correction} \
    --min_dist_recover_ratio ${min_dist_recover_ratio} \
    --debug_recover_eval_rollout ${debug_recover_eval_rollout} \
    --debug_correction_evac_rollout ${debug_correction_evac_rollout} \
    --correction_interp_nearest_enable ${correction_interp_nearest_enable} \
    --correction_interp_prefix_ratio ${correction_interp_prefix_ratio} \
    --correction_planner_prefix_ratio ${correction_planner_prefix_ratio} \
    --correction_compose_gt_tail_enable ${correction_compose_gt_tail_enable} \
    --rollout_exec_steps ${rollout_exec_steps} \
    --correction_weight ${correction_weight} \
    --orient_weight ${orient_weight} \
    --gripper_penalty ${gripper_penalty} \
    --recover_gripper_penalty ${recover_gripper_penalty} \
    --export_correction_dataset ${export_correction_dataset} \
    "
    if [ -n "${export_correction_dir}" ]; then
        wm_flags="${wm_flags} --export_correction_dir ${export_correction_dir}"
    fi
    if [ "${debug_wm}" = "true" ]; then
        wm_flags="${wm_flags} --debug_wm_correction"
    fi
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
--perturb_reject_sampling_enable ${perturb_reject_sampling_enable} \
--perturb_reject_max_trials ${perturb_reject_max_trials} \
--perturb_reject_dir_jitter_eps ${perturb_reject_dir_jitter_eps} \
--perturb_active_joint_delta_thresh ${perturb_active_joint_delta_thresh} \
--perturb_active_gripper_delta_thresh ${perturb_active_gripper_delta_thresh} \
--nearest_window_radius ${nearest_window_radius} \
--perturb_gripper_close_min ${perturb_gripper_close_min} \
--perturb_gripper_open_max ${perturb_gripper_open_max} \
--perturb_gripper_fast_ratio ${perturb_gripper_fast_ratio} \
--lr_sched_enable ${lr_sched_enable} \
--lr_warmup_steps ${lr_warmup_steps} \
--lr_min_ratio ${lr_min_ratio} \
--sp_reg_enable ${sp_reg_enable} \
--sp_reg_lambda ${sp_reg_lambda} \
--sample_pregrasp_bias_enable ${sample_pregrasp_bias_enable} \
--sample_pregrasp_prob ${sample_pregrasp_prob} \
--sample_pregrasp_phase_window_len ${sample_pregrasp_phase_window_len} \
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
if [ -n "${act_init_ckpt}" ]; then
    train_flags="${train_flags} --act_init_ckpt ${act_init_ckpt}"
fi

CUDA_VISIBLE_DEVICES=${gpu_ids} accelerate launch \
    ${gpu_flags} \
    imitate_episodes.py \
    ${train_flags} \
    ${perturb_flags} \
    ${wm_flags}
