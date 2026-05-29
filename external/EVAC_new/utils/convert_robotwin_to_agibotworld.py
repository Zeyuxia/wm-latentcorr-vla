"""
将 RoboTwin 数据集转换为 AgiBotWorld 格式
使得可以直接使用 AgiBotWorldIROSChallenge 数据集类进行训练

RoboTwin 格式:
- data/episodeX.hdf5:
  - endpose/left_endpose: (T, 7) [x, y, z, qw, qx, qy, qz] - 位置 + 四元数
  - endpose/left_gripper: (T,) - gripper 值 (0-1)
  - endpose/right_endpose: (T, 7)
  - endpose/right_gripper: (T,)
  - observation/head_camera/rgb: JPEG 编码的图像字节串
  - observation/head_camera/intrinsic_cv: (T, 3, 3) 相机内参
  - observation/head_camera/extrinsic_cv: (T, 3, 4) OpenCV 坐标系外参 (world-to-camera)

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
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def convert_endpose_to_agibotworld(left_endpose, right_endpose, left_gripper, right_gripper):
    """
    将 RoboTwin 的 endpose 转换为 AgiBotWorld 格式
    
    RoboTwin endpose: [x, y, z, qw, qx, qy, qz] (quaternion: wxyz 格式)
    AgiBotWorld:
    - positions: (T, 3)
    - quaternions: (T, 4) xyzw 格式
    - grippers: (T,) 范围 0-120
    
    返回: left_positions, left_quaternions, left_grippers, right_positions, right_quaternions, right_grippers
    """
    T = len(left_endpose)
    
    # 左臂
    left_positions = left_endpose[:, :3].astype(np.float32)
    # 四元数: wxyz -> xyzw
    left_quaternions = np.concatenate([
        left_endpose[:, 4:7],  # xyz
        left_endpose[:, 3:4]   # w
    ], axis=1).astype(np.float32)
    # gripper: 0-1 -> 0-120 (0=open, 1=closed -> 0=open, 120=closed)
    left_grippers = (left_gripper * 120).astype(np.float32)
    
    # 右臂
    right_positions = right_endpose[:, :3].astype(np.float32)
    right_quaternions = np.concatenate([
        right_endpose[:, 4:7],
        right_endpose[:, 3:4]
    ], axis=1).astype(np.float32)
    right_grippers = (right_gripper * 120).astype(np.float32)
    
    return left_positions, left_quaternions, left_grippers, right_positions, right_quaternions, right_grippers


def save_h5_file(filepath, left_positions, left_quaternions, left_grippers, 
                 right_positions, right_quaternions, right_grippers):
    """
    保存为 AgiBotWorld 格式的 H5 文件
    
    格式:
    - state/effector/position: (T, 2) - 左右 gripper 值
    - state/end/position: (T, 2, 3) - 左右末端位置
    - state/end/orientation: (T, 2, 4) - 左右末端四元数 (xyzw)
    """
    T = len(left_positions)
    
    with h5py.File(filepath, 'w') as f:
        # Gripper values (左, 右)
        effector_pos = np.stack([left_grippers, right_grippers], axis=1)  # (T, 2)
        f.create_dataset('state/effector/position', data=effector_pos)
        
        # End effector positions (左, 右)
        end_pos = np.stack([left_positions, right_positions], axis=1)  # (T, 2, 3)
        f.create_dataset('state/end/position', data=end_pos)
        
        # End effector orientations (左, 右)
        end_orient = np.stack([left_quaternions, right_quaternions], axis=1)  # (T, 2, 4)
        f.create_dataset('state/end/orientation', data=end_orient)


def save_intrinsic_json(filepath, intrinsic):
    """
    保存相机内参
    
    intrinsic: (3, 3) 内参矩阵
    """
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


def extrinsic_cv_to_c2w(extrinsic_cv):
    """
    将 OpenCV 坐标系的 world2cam 外参 (3x4) 转换为 camera2world (4x4)
    
    extrinsic_cv: (3, 4) [R|t] 格式，world-to-camera
    返回: (4, 4) camera-to-world 矩阵
    """
    # 补全为 4x4
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = extrinsic_cv
    # 求逆得到 c2w
    c2w = np.linalg.inv(w2c)
    return c2w


def save_extrinsic_json(filepath, extrinsic_cv_matrices):
    """
    保存相机外参 (每帧可能不同，但通常相机是固定的)
    
    extrinsic_cv_matrices: (T, 3, 4) 每帧的 extrinsic_cv 矩阵 (world-to-camera)
    """
    data = []
    for i in range(len(extrinsic_cv_matrices)):
        c2w = extrinsic_cv_to_c2w(extrinsic_cv_matrices[i])
        rotation_matrix = c2w[:3, :3].tolist()
        translation_vector = c2w[:3, 3].tolist()
        
        data.append({
            "extrinsic": {
                "rotation_matrix": rotation_matrix,
                "translation_vector": translation_vector
            }
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def decode_images(rgb_data):
    """
    解码 JPEG 格式的 RGB 图像数据
    
    rgb_data: (T,) 每个元素是 JPEG 编码的字节串
    """
    images = []
    for i in range(len(rgb_data)):
        img_bytes = bytes(rgb_data[i])
        img = np.array(Image.open(BytesIO(img_bytes)))
        images.append(img)
    return images


def save_video(filepath, images, fps=30):
    """使用 imageio 保存视频为 MP4 格式"""
    imageio.mimwrite(filepath, images, fps=fps, codec='libx264', quality=8)


def process_episode(hdf5_path, output_dir, task_name, episode_idx, camera_name='head_camera', fps=30):
    """处理单个 episode"""
    
    with h5py.File(hdf5_path, 'r') as f:
        # 获取帧数
        T = len(f['endpose/left_endpose'])
        
        # 1. 提取 endpose 数据
        left_endpose = np.array(f['endpose/left_endpose'])
        right_endpose = np.array(f['endpose/right_endpose'])
        left_gripper = np.array(f['endpose/left_gripper'])
        right_gripper = np.array(f['endpose/right_gripper'])
        
        # 2. 提取相机数据
        camera_key = f'observation/{camera_name}'
        rgb_data = f[f'{camera_key}/rgb'][:]
        intrinsic = f[f'{camera_key}/intrinsic_cv'][0]  # 假设内参不变，取第一帧
        extrinsic_cv = f[f'{camera_key}/extrinsic_cv'][:]  # 使用 OpenCV 坐标系外参
    
    # 创建输出目录 (格式: task-episode-step)
    ep_dir = os.path.join(output_dir, f"{task_name}-ep{episode_idx:03d}-step001")
    os.makedirs(ep_dir, exist_ok=True)
    
    # 3. 解码图像并保存为视频
    images = decode_images(rgb_data)
    if len(images) == 0:
        print(f"Warning: No images found in {hdf5_path}")
        return False
    
    save_video(os.path.join(ep_dir, 'head_color.mp4'), images, fps=fps)
    
    # 4. 转换并保存 endpose 数据
    left_pos, left_quat, left_grip, right_pos, right_quat, right_grip = \
        convert_endpose_to_agibotworld(left_endpose, right_endpose, left_gripper, right_gripper)
    
    save_h5_file(
        os.path.join(ep_dir, 'proprio_stats.h5'),
        left_pos, left_quat, left_grip,
        right_pos, right_quat, right_grip
    )
    
    # 5. 保存相机参数
    save_intrinsic_json(os.path.join(ep_dir, 'head_intrinsic_params.json'), intrinsic)
    save_extrinsic_json(os.path.join(ep_dir, 'head_extrinsic_params_aligned.json'), extrinsic_cv)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert RoboTwin dataset to AgiBotWorld format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='RoboTwin dataset directory (e.g., /data/zhenyangfan/RoboTwin/data)')
    parser.add_argument('--output_dir', type=str, default='/data/datasets/robotwin_agibotworld_format',
                        help='Output directory')
    parser.add_argument('--split', type=str, default='train', help='Split name')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Task names to process (default: all tasks)')
    parser.add_argument('--demo_type', type=str, default='demo_clean',
                        help='Demo type to process (e.g., demo_clean, demo_aug)')
    parser.add_argument('--camera', type=str, default='head_camera',
                        choices=['head_camera', 'front_camera', 'left_camera', 'right_camera'],
                        help='Camera to use for video')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Max episodes to process per task')
    args = parser.parse_args()

    # 创建输出目录
    output_split_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # 获取所有任务
    if args.tasks:
        tasks = args.tasks
    else:
        # 自动发现任务 (排除非目录项)
        tasks = [d for d in os.listdir(args.input_dir) 
                 if os.path.isdir(os.path.join(args.input_dir, d))]
    
    print(f"Found {len(tasks)} tasks: {tasks}")
    
    total_episodes = 0
    
    for task_name in tasks:
        task_dir = os.path.join(args.input_dir, task_name, args.demo_type)
        data_dir = os.path.join(task_dir, 'data')
        
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory not found: {data_dir}, skipping...")
            continue
        
        # 获取所有 episode 文件
        episode_files = sorted([
            f for f in os.listdir(data_dir) 
            if f.startswith('episode') and f.endswith('.hdf5')
        ], key=lambda x: int(x.replace('episode', '').replace('.hdf5', '')))
        
        if args.max_episodes:
            episode_files = episode_files[:args.max_episodes]
        
        print(f"\nProcessing task: {task_name} ({len(episode_files)} episodes)")
        
        for ep_file in tqdm(episode_files, desc=f"Processing {task_name}"):
            episode_idx = int(ep_file.replace('episode', '').replace('.hdf5', ''))
            hdf5_path = os.path.join(data_dir, ep_file)
            
            success = process_episode(
                hdf5_path, output_split_dir, task_name, episode_idx,
                camera_name=args.camera, fps=args.fps
            )
            
            if success:
                total_episodes += 1
    
    print(f"\nDone! Processed {total_episodes} episodes from {len(tasks)} tasks")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
