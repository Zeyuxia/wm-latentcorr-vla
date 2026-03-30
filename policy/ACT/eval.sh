#!/bin/bash

# Base eval paramters
policy_name=ACT
task_name=open_laptop
task_config=demo_clean
ckpt_setting=demo_clean
ckpt_dir=/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-open_laptop/demo_clean-50/20260330_220553
ckpt_name=policy_epoch_700_seed_0.ckpt
seed=0
seed_file=/data/zhenyangfan/RoboTwin/data/open_laptop/demo_clean/seed.txt
gpu_id=3

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_dir ${ckpt_dir} \
    --ckpt_name ${ckpt_name} \
    --seed ${seed} \
    # --temporal_agg true
