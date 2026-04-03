#!/bin/bash
cd /data/zhenyangfan/RoboTwin/policy/ACT

# Base eval paramters
policy_name=ACT
task_name=open_laptop
task_config=demo_clean
ckpt_setting=demo_clean
# ckpt_dir=/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-open_laptop/demo_clean-50/20260331_103042
ckpt_dir=/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-open_laptop/demo_clean-50/20260330_220553
ckpt_name=policy_epoch_200_seed_0.ckpt
seed=0
seed_file=/data/zhenyangfan/RoboTwin/data/open_laptop/demo_clean/seed.txt
gpu_id=1
# eval_tag="bsz4_cor2_gripperonly_200epoch"
eval_tag="bsz6_200epoch"

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

TAG_ARGS=()
if [ -n "${eval_tag}" ]; then
    TAG_ARGS+=(--eval_tag "${eval_tag}")
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_dir ${ckpt_dir} \
    --ckpt_name ${ckpt_name} \
    --seed ${seed} \
    "${TAG_ARGS[@]}" \
    # --temporal_agg true
