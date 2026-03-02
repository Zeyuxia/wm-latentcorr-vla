#!/bin/bash

# == Policy config (same as eval.sh) ==
policy_name=ACT
task_name=open_laptop
task_config=demo_clean_seed100k
ckpt_dir=/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-open_laptop/demo_clean-50/20260213_001933
ckpt_name=policy_epoch_1000_seed_0.ckpt
seed=0
gpu_id=5

# == EVAC config (same as train.sh) ==
evac_ckpt=/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt
evac_config=policy/ACT/evac/configs/robotwin/train_config.yaml
urdf_path=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
curobo_left_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml
curobo_right_yml=/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml
raw_data_dir=data_eval/${task_name}/${task_config}/data

# == Eval config ==
max_steps=700
orient_weight=0.01
gripper_penalty=1.0
success_threshold=1e6
num_episodes=1         # uncomment to limit number of episodes
save_video=true            # save predicted video per episode
debug_evac_eval=true       # save EVAC debug outputs (frames + traj video)

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id: ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_evac.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --ckpt_dir ${ckpt_dir} \
    --ckpt_name ${ckpt_name} \
    --seed ${seed} \
    --temporal_agg false \
    --evac_ckpt ${evac_ckpt} \
    --evac_config ${evac_config} \
    --urdf_path ${urdf_path} \
    --curobo_left_yml ${curobo_left_yml} \
    --curobo_right_yml ${curobo_right_yml} \
    --raw_data_dir ${raw_data_dir} \
    --num_episodes ${num_episodes} \
    --max_steps ${max_steps} \
    --orient_weight ${orient_weight} \
    --gripper_penalty ${gripper_penalty} \
    --success_threshold ${success_threshold} \
    --save_video ${save_video} \
    --debug_evac_eval ${debug_evac_eval}