#!/bin/bash
cd /data/zhenyangfan/RoboTwin/policy/ACT

task_name=open_laptop
task_config=demo_clean
expert_data_num=50
seed=0

# Online rollout perturbation options (effective when enable_wm=true)
enable_perturb=true
perturb_prob=1.0
perturb_space=eef
perturb_lp_alpha=0.6
perturb_noise_std=0.04
perturb_noise_rho=0.85
perturb_vel_limit=0.2
perturb_acc_limit=0.1
perturb_bias_prob=0.5
perturb_bias_std=0.08
perturb_joint_limit_abs=3.14
perturb_mode=directional_fail
perturb_fail_gain=0.12
perturb_fail_direction=1,1,1,1,1,1,-1,-1,-1,-1,-1,-1
perturb_eef_mode=directional_fail
perturb_eef_pos_std=0.01
perturb_eef_fail_gain=0.028
perturb_eef_tcp_offset_x=0.085
perturb_eef_ramp=true
perturb_eef_ramp_min=0.0
perturb_eef_ramp_power=1.5
perturb_eef_ramp_apply_eps=1e-4
perturb_eef_joint_delta_cap=0.08
perturb_eef_fail_dir_left=1,0,0
perturb_eef_fail_dir_right=-1,0,0
vla_input_noise_enable=false
vla_img_noise_std=0.0
vla_qpos_noise_std=0.0

# World-model correction options (set enable_wm=true to activate)
enable_wm=true
evac_ckpt=/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt
evac_config=./evac/configs/robotwin/train_config.yaml
urdf_path=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
curobo_left_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
curobo_right_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
raw_data_dir=/data/zhenyangfan/RoboTwin/data/${task_name}/${task_config}/data
act_init_ckpt=./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/20260213_001933/policy_epoch_1000_seed_0.ckpt
correction_threshold=0.05
max_rollout_steps=2
single_rollout_correction=true
target_lookahead_steps=5
rollout_exec_steps=16
correction_freq=1
correction_weight=1.0
orient_weight=0.01
gripper_penalty=1.0
always_correction=true
debug_wm=true

# EVAC acceleration (same semantics as eval_evac.sh)
evac_budget_accel=false
evac_rank_transfer=false
evac_ddim_eta=None
evac_dc_budget=0.6
evac_rt_full_chunks=3
evac_rt_per_channel=true

# Append timestamp to ckpt_dir
timestamp=$(date +"%Y%m%d_%H%M%S")
ckpt_dir=./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/${timestamp}
mkdir -p ${ckpt_dir}

# Save a copy of this script for reproducibility
cp "$0" ${ckpt_dir}/train.sh

gpu_ids=0
num_gpus=$(echo ${gpu_ids} | awk -F',' '{print NF}')

if [ ${num_gpus} -gt 1 ]; then
    multi_gpu_flag="--multi_gpu"
else
    multi_gpu_flag=""
fi

# Build WM correction flags
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
    --correction_threshold ${correction_threshold} \
    --max_rollout_steps ${max_rollout_steps} \
    --single_rollout_correction ${single_rollout_correction} \
    --target_lookahead_steps ${target_lookahead_steps} \
    --rollout_exec_steps ${rollout_exec_steps} \
    --correction_freq ${correction_freq} \
    --correction_weight ${correction_weight} \
    --orient_weight ${orient_weight} \
    --gripper_penalty ${gripper_penalty} \
    --always_correction ${always_correction} \
    --evac_budget_accel ${evac_budget_accel} \
    --evac_rank_transfer ${evac_rank_transfer} \
    --evac_ddim_eta ${evac_ddim_eta} \
    --evac_dc_budget ${evac_dc_budget} \
    --evac_rt_full_chunks ${evac_rt_full_chunks} \
    --evac_rt_per_channel ${evac_rt_per_channel}"
    if [ "${debug_wm}" = "true" ]; then
        wm_flags="${wm_flags} --debug_wm_correction"
    fi
fi

perturb_flags="--enable_perturb ${enable_perturb} \
--perturb_prob ${perturb_prob} \
--perturb_space ${perturb_space} \
--perturb_lp_alpha ${perturb_lp_alpha} \
--perturb_noise_std ${perturb_noise_std} \
--perturb_noise_rho ${perturb_noise_rho} \
--perturb_vel_limit ${perturb_vel_limit} \
--perturb_acc_limit ${perturb_acc_limit} \
--perturb_bias_prob ${perturb_bias_prob} \
--perturb_bias_std ${perturb_bias_std} \
--perturb_joint_limit_abs ${perturb_joint_limit_abs} \
--perturb_mode ${perturb_mode} \
--perturb_fail_gain ${perturb_fail_gain} \
--perturb_fail_direction ${perturb_fail_direction} \
--perturb_eef_mode ${perturb_eef_mode} \
--perturb_eef_pos_std ${perturb_eef_pos_std} \
--perturb_eef_fail_gain ${perturb_eef_fail_gain} \
--perturb_eef_tcp_offset_x ${perturb_eef_tcp_offset_x} \
--perturb_eef_ramp ${perturb_eef_ramp} \
--perturb_eef_ramp_min ${perturb_eef_ramp_min} \
--perturb_eef_ramp_power ${perturb_eef_ramp_power} \
--perturb_eef_ramp_apply_eps ${perturb_eef_ramp_apply_eps} \
--perturb_eef_joint_delta_cap ${perturb_eef_joint_delta_cap} \
--perturb_eef_fail_dir_left=${perturb_eef_fail_dir_left} \
--perturb_eef_fail_dir_right=${perturb_eef_fail_dir_right} \
--vla_input_noise_enable ${vla_input_noise_enable} \
--vla_img_noise_std ${vla_img_noise_std} \
--vla_qpos_noise_std ${vla_qpos_noise_std}"

CUDA_VISIBLE_DEVICES=${gpu_ids} accelerate launch \
    ${multi_gpu_flag} \
    --num_processes ${num_gpus} \
    --main_process_port 29600 \
    imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ${ckpt_dir} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 4 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 4e-5 \
    --save_freq 10 \
    --state_dim 14 \
    --seed ${seed} \
    ${perturb_flags} \
    ${wm_flags}
