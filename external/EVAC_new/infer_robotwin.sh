#!/bin/bash

export PYTHONPATH=/data/zhenyangfan/EVAC:$PYTHONPATH

input_root=/data/datasets/robotwin_agibotworld_format/test_10_converted
save_root=/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/eval_results/test_speed
ckp_path=/data/zhenyangfan/EVAC/logs/evac_robotwin_finetune_2026-02-07T21-13-51/checkpoints/epoch=2499-step=10000.ckpt
config_path=/data/zhenyangfan/EVAC/configs/robotwin/train_config.yaml
n_pred=1
CUDA_VISIBLE_DEVICES=0 python /data/zhenyangfan/EVAC/evac/main/infer_all.py -i $input_root -s $save_root --ckp_path $ckp_path --config_path $config_path --n_pred $n_pred