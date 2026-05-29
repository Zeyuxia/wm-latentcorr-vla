#!/bin/bash

# 将 LIBERO 数据集转换为 AgiBotWorld 格式

INPUT_DIR="/data/zhenyangfan/smolvla/outputs/libero_inference_data_merged"
OUTPUT_DIR="/data/datasets/libero_agibotworld_format"
CAMERA_PARAMS="/data/datasets/libero/libero_camera_params.json"
SPLIT="unseen"

python /data/zhenyangfan/EVAC/utils/convert_libero_to_agibotworld.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --camera_params "$CAMERA_PARAMS" \
    --split "$SPLIT"
