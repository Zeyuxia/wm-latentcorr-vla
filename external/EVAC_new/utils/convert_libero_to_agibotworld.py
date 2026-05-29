"""
将 LIBERO 数据集转换为 AgiBotWorld 格式
使得可以直接使用 AgiBotWorldIROSChallenge 数据集类进行训练

LIBERO 格式:
- parquet 文件: images(bytes), action(7), state(8)
- action: [dx, dy, dz, droll, dpitch, dyaw, gripper] (delta action, gripper in [-1, 1])

AgiBotWorld 格式:
- head_color.mp4: 视频文件
- head_intrinsic_params.json: 相机内参
- head_extrinsic_params_aligned.json: 每帧外参列表
- proprio_stats.h5: 动作数据 (state/effector/position, state/end/position, state/end/orientation)
"""

import os
import json
import argparse
import numpy as np
import h5py
import imageio
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def states_to_absolute_pose(states):
    """
    将 LIBERO 的 observation.state 转换为绝对位姿
    
    LIBERO observation.state: [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    输出: positions (T, 3), quaternions (T, 4), grippers (T,)
    
    gripper_qpos: 原始是关节角度，需要映射到 0-120
    """
    T = len(states)
    
    positions = np.zeros((T, 3), dtype=np.float32)
    quaternions = np.zeros((T, 4), dtype=np.float32)  # xyzw 格式
    grippers = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        state = np.array(states[t])
        
        # 位置
        positions[t] = state[:3]
        
        # axis-angle 转 quaternion
        axis_angle = state[3:6]
        rot = Rotation.from_rotvec(axis_angle)
        quaternions[t] = rot.as_quat()  # xyzw 格式
        
        # gripper: gripper_qpos[0] 通常在 [0.0, 0.04] 范围
        # 映射到 AgiBotWorld 的 [0, 120]
        # Panda gripper: 0.04 = fully open, 0.0 = closed
        # 注意: 实际数据可能略超出范围，需要 clip
        gripper_qpos = np.clip(state[6], 0.0, 0.04)  # 只用第一个手指
        grippers[t] = (1 - gripper_qpos / 0.04) * 120  # 反转: open=0, closed=120
    
    return positions, quaternions, grippers


def save_h5_file(filepath, positions, quaternions, grippers):
    """
    保存为 AgiBotWorld 格式的 H5 文件
    
    格式:
    - state/effector/position: (T, 2) - 左右 gripper 值
    - state/end/position: (T, 2, 3) - 左右末端位置
    - state/end/orientation: (T, 2, 4) - 左右末端四元数 (xyzw)
    """
    T = len(positions)
    
    with h5py.File(filepath, 'w') as f:
        # Gripper values (复制给左右臂)
        effector_pos = np.stack([grippers, grippers], axis=1)  # (T, 2)
        f.create_dataset('state/effector/position', data=effector_pos)
        
        # End effector positions (复制给左右臂)
        end_pos = np.stack([positions, positions], axis=1)  # (T, 2, 3)
        f.create_dataset('state/end/position', data=end_pos)
        
        # End effector orientations (复制给左右臂)
        end_orient = np.stack([quaternions, quaternions], axis=1)  # (T, 2, 4)
        f.create_dataset('state/end/orientation', data=end_orient)


def save_intrinsic_json(filepath, intrinsic):
    """保存相机内参"""
    data = {
        "intrinsic": {
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "ppx": float(intrinsic[0, 2]),
            "ppy": float(intrinsic[1, 2])
        }
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_extrinsic_json(filepath, extrinsic_c2w, num_frames):
    """
    保存相机外参 (每帧相同，因为相机固定)
    """
    rotation_matrix = extrinsic_c2w[:3, :3].tolist()
    translation_vector = extrinsic_c2w[:3, 3].tolist()
    
    data = []
    for _ in range(num_frames):
        data.append({
            "extrinsic": {
                "rotation_matrix": rotation_matrix,
                "translation_vector": translation_vector
            }
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_video(filepath, images, fps=10):
    """使用 imageio 保存视频为 MP4 格式"""
    # imageio-ffmpeg 自动使用 H.264 编码
    imageio.mimwrite(filepath, images, fps=fps, codec='libx264', quality=8)


def process_episode(df, episode_idx, output_dir, task_idx, extrinsic_c2w, intrinsic, fps=10):
    """处理单个 episode"""
    
    # 筛选当前 episode 的数据
    ep_data = df[df['episode_index'] == episode_idx].sort_values('frame_index')
        
    # 创建输出目录 (格式: task-episode-step)
    ep_dir = os.path.join(output_dir, f"task{task_idx:02d}-ep{episode_idx:03d}-step001")
    os.makedirs(ep_dir, exist_ok=True)
    
    # 1. 提取图像并保存为视频
    images = []
    for _, row in ep_data.iterrows():
        img_data = row['observation.images.image']
        if isinstance(img_data, dict):
            img_bytes = img_data['bytes']
            img = np.array(Image.open(BytesIO(img_bytes)))
            # MuJoCo 使用 OpenGL 坐标系 (y轴向上)，需要垂直翻转图像
            # img = np.flipud(img)
            images.append(img)
    
    if len(images) == 0:
        return False
    
    save_video(os.path.join(ep_dir, 'head_color.mp4'), images, fps=fps)
    
    # 2. 直接从 observation.state 获取绝对位姿 (不需要积分 delta action)
    states = ep_data['observation.state'].values.tolist()
    positions, quaternions, grippers = states_to_absolute_pose(states)
    
    # 3. 保存 H5 文件
    save_h5_file(os.path.join(ep_dir, 'proprio_stats.h5'), positions, quaternions, grippers)
    
    # 4. 保存相机参数
    save_intrinsic_json(os.path.join(ep_dir, 'head_intrinsic_params.json'), intrinsic)
    save_extrinsic_json(os.path.join(ep_dir, 'head_extrinsic_params_aligned.json'), extrinsic_c2w, len(images))
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert LIBERO dataset to AgiBotWorld format')
    parser.add_argument('--input_dir', type=str, default='',
                        help='LIBERO dataset directory')
    parser.add_argument('--output_dir', type=str, default='/data/datasets/libero_agibotworld_format',
                        help='Output directory')
    parser.add_argument('--camera_params', type=str, default='/data/datasets/libero/libero_camera_params.json',
                        help='Pre-computed camera parameters file (run precompute_libero_camera_params.py first)')
    parser.add_argument('--split', type=str, default='train', help='Split name')
    parser.add_argument('--max_episodes', type=int, default=None, help='Max episodes to process')
    args = parser.parse_args()

    # 创建输出目录
    output_split_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # 加载预计算的相机参数 (key 是 task_name)
    if not os.path.exists(args.camera_params):
        raise FileNotFoundError(f"Camera parameters file not found: {args.camera_params}")
    
    print(f"Loading camera parameters from {args.camera_params}...")
    with open(args.camera_params, 'r') as f:
        raw_params = json.load(f)
    
    # 按 task_name 索引的相机参数
    task_name_to_camera_params = {}
    for task_name, params in raw_params.items():
        task_name_to_camera_params[task_name] = {
            'extrinsic': np.array(params['extrinsic'], dtype=np.float32),
            'intrinsic': np.array(params['intrinsic'], dtype=np.float32),
            'benchmark': params['benchmark']
        }
    print(f"Loaded camera parameters for {len(task_name_to_camera_params)} tasks")
    
    # 读取 tasks.parquet 获取 task_index -> task_name 的映射
    tasks_parquet_path = os.path.join(args.input_dir, 'meta', 'tasks.parquet')
    if not os.path.exists(tasks_parquet_path):
        raise FileNotFoundError(f"Tasks parquet file not found: {tasks_parquet_path}")
    
    tasks_df = pq.read_table(tasks_parquet_path).to_pandas()
    tasks_df = tasks_df.reset_index()
    tasks_df.columns = ['task_name', 'task_index']
    
    # 建立 task_index -> task_name 的映射
    task_idx_to_name = {}
    for _, row in tasks_df.iterrows():
        task_idx_to_name[int(row['task_index'])] = row['task_name'].lower()
    print(f"Loaded {len(task_idx_to_name)} task names from tasks.parquet")
    
    # 读取所有 parquet 文件
    data_dir = os.path.join(args.input_dir, 'data', 'chunk-000')
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    
    print(f"\nFound {len(parquet_files)} parquet files")
    
    episode_count = 0
    
    for pq_file in tqdm(parquet_files, desc="Processing parquet files"):
        df = pq.read_table(os.path.join(data_dir, pq_file)).to_pandas()
        
        # 获取所有 episode
        episodes = df['episode_index'].unique()
        
        for ep_idx in episodes:
            task_idx = int(df[df['episode_index'] == ep_idx]['task_index'].iloc[0])
            
            # 通过 task_index 获取 task_name，再查找相机参数
            if task_idx not in task_idx_to_name:
                print(f"Warning: task_idx {task_idx} not in tasks.parquet, skipping...")
                continue
            
            task_name = task_idx_to_name[task_idx]
            if task_name not in task_name_to_camera_params:
                print(f"Warning: task_name '{task_name}' not in camera params, skipping...")
                continue
            
            cam_params = task_name_to_camera_params[task_name]
            
            success = process_episode(
                df, ep_idx, output_split_dir, task_idx,
                cam_params['extrinsic'], cam_params['intrinsic']
            )
            
            if success:
                episode_count += 1
            
            if args.max_episodes and episode_count >= args.max_episodes:
                break
        
        if args.max_episodes and episode_count >= args.max_episodes:
            break
    
    print(f"\nDone! Processed {episode_count} episodes")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()