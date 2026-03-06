#!/bin/bash
cd /data/zhenyangfan/RoboTwin/policy/ACT

task_name=open_laptop
task_config=demo_clean
expert_data_num=50
seed=0

# World-model correction options (set enable_wm=true to activate)
enable_wm=true
evac_ckpt=/data/zhenyangfan/EVAC_cache/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt
evac_config=./evac/configs/robotwin/train_config.yaml
urdf_path=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
curobo_left_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
curobo_right_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
raw_data_dir=/data/zhenyangfan/RoboTwin/data/${task_name}/${task_config}/data
act_init_ckpt=./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/20260213_001933/policy_epoch_100_seed_0.ckpt
correction_threshold=0.05
max_rollout_steps=5
correction_freq=1
correction_weight=1.0
orient_weight=0.01
gripper_penalty=1.0
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

gpu_ids=0,1,2,3
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
    --correction_freq ${correction_freq} \
    --correction_weight ${correction_weight} \
    --orient_weight ${orient_weight} \
    --gripper_penalty ${gripper_penalty} \
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

CUDA_VISIBLE_DEVICES=${gpu_ids} accelerate launch \
    ${multi_gpu_flag} \
    --num_processes ${num_gpus} \
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
    ${wm_flags}