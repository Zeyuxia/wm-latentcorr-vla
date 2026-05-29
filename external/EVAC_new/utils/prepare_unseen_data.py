#!/usr/bin/env python3
"""
将扁平的episode目录转换为两层结构(task/episode)，并提取第一帧
输入: task00-ep000-step001/head_color.mp4
输出: task_0/episode_0/frame.png + 复制其他文件
"""
import os
import shutil
import argparse
from pathlib import Path
import cv2


def parse_episode_name(name: str):
    """解析episode目录名，返回task_id和episode_id"""
    # 格式: task00-ep000-step001
    parts = name.split('-')
    task_id = int(parts[0].replace('task', ''))
    ep_id = int(parts[1].replace('ep', ''))
    return task_id, ep_id


def main():
    parser = argparse.ArgumentParser(description='转换为两层目录结构并提取第一帧')
    parser.add_argument('--src_dir', type=str, default='/data/datasets/libero_agibotworld_format/unseen', help='源数据目录')
    parser.add_argument('--dst_dir', type=str, default='/data/datasets/libero_agibotworld_format/unseen_structured', help='目标目录')
    args = parser.parse_args()
    
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    
    # 获取所有episode目录
    all_episodes = [d for d in os.listdir(src_dir) if os.path.isdir(src_dir / d)]
    print(f"共有 {len(all_episodes)} 个episode")
    
    success_count = 0
    for ep_name in sorted(all_episodes):
        task_id, ep_id = parse_episode_name(ep_name)
        
        src_ep_dir = src_dir / ep_name
        dst_ep_dir = dst_dir / f"task_{task_id}" / f"episode_{ep_id}"
        dst_ep_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制所需文件
        for filename in ['head_intrinsic_params.json', 'head_extrinsic_params_aligned.json', 'proprio_stats.h5', 'head_color.mp4']:
            src_file = src_ep_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, dst_ep_dir / filename)
        
        # 提取第一帧
        video_file = src_ep_dir / 'head_color.mp4'
        frame_file = dst_ep_dir / 'frame.png'
        
        if video_file.exists():
            cap = cv2.VideoCapture(str(video_file))
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                cv2.imwrite(str(frame_file), frame)
                success_count += 1
            else:
                print(f"  失败: {ep_name}")
        else:
            print(f"  跳过: {ep_name} (无视频)")
    
    print(f"\n完成! 成功处理 {success_count}/{len(all_episodes)} 个episode")
    print(f"输出目录: {dst_dir}")


if __name__ == '__main__':
    main()