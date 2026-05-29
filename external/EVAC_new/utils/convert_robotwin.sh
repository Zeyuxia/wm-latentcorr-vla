cd /data/zhenyangfan/EVAC && conda run -n enerverse python utils/convert_robotwin_to_agibotworld.py \
    --input_dir /data/zhenyangfan/RoboTwin/data_self_collect/ \
    --output_dir /data/datasets/robotwin_agibotworld_format \
    --split train_500 \
    --tasks open_laptop 