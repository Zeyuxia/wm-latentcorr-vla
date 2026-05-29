#!/bin/bash

export PYTHONPATH=/data/zhenyangfan/EVAC:$PYTHONPATH

input_root=/data/datasets/libero_agibotworld_format/unseen_structured
save_root=/data/zhenyangfan/EVAC/logs/evac_libero_failure_2026-01-13T01-16-45/eval_results/10000/useen
ckp_path=/data/zhenyangfan/EVAC/logs/evac_libero_failure_2026-01-13T01-16-45/checkpoints/epoch=249-step=10000.ckpt
config_path=/data/zhenyangfan/EVAC/configs/libero/train_config.yaml
n_pred=2
CUDA_VISIBLE_DEVICES=4 python /data/zhenyangfan/EVAC/evac/main/infer_all.py -i $input_root -s $save_root --ckp_path $ckp_path --config_path $config_path --n_pred $n_pred