#!/usr/bin/env python3
"""
从libero_agibotworld_format训练数据中选择20个episode，
转换成inference格式，放到test文件夹中。

Inference格式要求:
PATH_TO_YOUR_DATASET/
├── task_0/
│   ├── episode_0/
│   │   ├── frame.png
│   │   ├── head_intrinsic_params.json
│   │   ├── head_extrinsic_params_aligned.json
│   │   └── proprio_stats.h5
│   ├── episode_1/
│   └── ...
├── task_1/
└── ...
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

import cv2


def extract_first_frame(video_path: str, output_path: str) -> bool:
    """使用OpenCV从视频中提取第一帧"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  警告: 无法打开视频文件 {video_path}")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print(f"  警告: 无法读取第一帧 {video_path}")
            return False
        
        # 保存为PNG格式
        cv2.imwrite(output_path, frame)
        return True
    except Exception as e:
        print(f"  警告: 提取第一帧失败: {e}")
        return False


def parse_episode_name(name: str):
    """解析episode目录名，返回task_id和episode_id"""
    # 格式: task00-ep000-step001
    parts = name.split('-')
    task_id = int(parts[0].replace('task', ''))
    ep_id = int(parts[1].replace('ep', ''))
    return task_id, ep_id


def main():
    parser = argparse.ArgumentParser(description='准备inference测试数据')
    parser.add_argument(
        '--src_dir',
        type=str,
        default='/data/datasets/libero_agibotworld_format/train',
        help='源数据目录'
    )
    parser.add_argument(
        '--dst_dir',
        type=str,
        default='/data/datasets/libero_agibotworld_format/test',
        help='目标输出目录'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=40,
        help='要选择的episode数量'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='balanced',
        choices=['random', 'balanced'],
        help='选择策略: random=随机选择, balanced=每个task均匀选择'
    )
    args = parser.parse_args()

    random.seed(args.seed)
    
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    
    # 获取所有episode目录
    all_episodes = [d for d in os.listdir(src_dir) if os.path.isdir(src_dir / d)]
    print(f"源目录共有 {len(all_episodes)} 个episode")

    # 按task分组
    task_episodes = defaultdict(list)
    for ep_name in all_episodes:
        task_id, ep_id = parse_episode_name(ep_name)
        task_episodes[task_id].append((ep_name, ep_id))
    
    print(f"共有 {len(task_episodes)} 个task")
    
    # 选择episode
    selected_episodes = []
    
    if args.strategy == 'balanced':
        # 每个task均匀选择
        num_tasks = len(task_episodes)
        eps_per_task = max(1, args.num_episodes // num_tasks)
        remaining = args.num_episodes - eps_per_task * num_tasks
        
        for task_id in sorted(task_episodes.keys()):
            eps = task_episodes[task_id]
            # 随机选择
            n_select = eps_per_task
            if remaining > 0 and task_id < remaining:
                n_select += 1
            
            selected = random.sample(eps, min(n_select, len(eps)))
            for ep_name, ep_id in selected:
                selected_episodes.append((task_id, ep_name, ep_id))
    else:
        # 完全随机选择
        all_eps_with_task = []
        for task_id, eps in task_episodes.items():
            for ep_name, ep_id in eps:
                all_eps_with_task.append((task_id, ep_name, ep_id))
        selected_episodes = random.sample(all_eps_with_task, min(args.num_episodes, len(all_eps_with_task)))
    
    print(f"选择了 {len(selected_episodes)} 个episode")
    
    # 创建目标目录
    
    # 处理每个选中的episode
    for task_id, ep_name, orig_ep_id in sorted(selected_episodes):
        src_ep_dir = src_dir / ep_name
        dst_task_dir = dst_dir / f"task_{task_id}"
        dst_ep_dir = dst_task_dir / f"episode_{orig_ep_id}"  # 使用原始episode编号
        
        dst_ep_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"处理: {ep_name} -> task_{task_id}/episode_{orig_ep_id}")
        
        # 复制json和h5文件
        for filename in ['head_intrinsic_params.json', 'head_extrinsic_params_aligned.json', 'proprio_stats.h5']:
            src_file = src_ep_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, dst_ep_dir / filename)
            else:
                print(f"  警告: 文件不存在 {src_file}")
        
        # 从视频提取第一帧
        video_file = src_ep_dir / 'head_color.mp4'
        frame_file = dst_ep_dir / 'frame.png'
        
        if video_file.exists():
            success = extract_first_frame(str(video_file), str(frame_file))
            if not success:
                print(f"  错误: 无法提取第一帧 {video_file}")
        else:
            print(f"  警告: 视频文件不存在 {video_file}")
        
        # 复制GT视频文件
        gt_video_src = src_ep_dir / 'head_color.mp4'
        gt_video_dst = dst_ep_dir / 'head_color.mp4'
        
        if gt_video_src.exists():
            shutil.copy2(gt_video_src, gt_video_dst)
        else:
            print(f"  警告: GT视频文件不存在 {gt_video_src}")
    
    print(f"\n完成! 输出目录: {dst_dir}")
    print(f"共处理 {len(selected_episodes)} 个episode")
    
    # 打印统计
    print("\n各task的episode:")
    task_eps = defaultdict(list)
    for task_id, ep_name, orig_ep_id in selected_episodes:
        task_eps[task_id].append(orig_ep_id)
    for task_id in sorted(task_eps.keys()):
        eps = sorted(task_eps[task_id])
        print(f"  task_{task_id}: {len(eps)} 个episode - {eps}")


if __name__ == '__main__':
    main()
