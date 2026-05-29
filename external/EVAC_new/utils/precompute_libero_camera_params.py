"""
预计算 LIBERO 所有任务的相机内外参并保存为文件
这样后续转换数据时可以直接加载，不需要每次都创建环境
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


def get_camera_params_from_env(env, camera_name="agentview", image_height=256, image_width=256):
    """
    从 LIBERO 环境获取相机参数
    使用 robosuite 官方工具函数确保正确性
    返回: extrinsic_c2w (4x4), intrinsic (3x3)
    """
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
    
    sim = env.sim
    
    # 使用 robosuite 官方函数获取外参（已包含 MuJoCo 坐标系校正）
    extrinsic_c2w = get_camera_extrinsic_matrix(sim, camera_name).astype(np.float32)
    
    # 使用 robosuite 官方函数获取内参
    intrinsic = get_camera_intrinsic_matrix(sim, camera_name, image_height, image_width).astype(np.float32)
    
    return extrinsic_c2w, intrinsic


def create_libero_env(task, benchmark_name):
    """
    根据 task 创建 LIBERO 环境
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    
    # 获取 BDDL 文件的完整路径
    bddl_path = os.path.join(get_libero_path('bddl_files'), benchmark_name, task.bddl_file)
    
    env_args = {
        'bddl_file_name': bddl_path,
        'camera_heights': 256,
        'camera_widths': 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    return env


def main():
    parser = argparse.ArgumentParser(description='Precompute LIBERO camera parameters')
    parser.add_argument('--output', type=str, default='/data/datasets/libero/libero_camera_params.json',
                        help='Output JSON file path')
    args = parser.parse_args()
    
    from libero.libero import benchmark
    
    benchmark_names = ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal']
    
    print("Computing camera parameters from LIBERO environments...")
    print("This will take a few minutes but only needs to be done once.\n")
    
    def extract_task_description(full_name):
        """从完整任务名中提取任务描述，去掉场景前缀
        格式: SCENE_NAME_task_description -> task description
        """
        parts = full_name.split('_')
        for i, part in enumerate(parts):
            if part.islower() or part in ['put', 'pick', 'turn', 'open', 'push']:
                return ' '.join(parts[i:]).lower()
        return full_name.lower()
    
    all_params = {}
    
    for bm_name in benchmark_names:
        bm = benchmark.get_benchmark(bm_name)()
        print(f"\nBenchmark: {bm_name}, Tasks: {bm.n_tasks}")
        
        for local_task_idx in tqdm(range(bm.n_tasks), desc=f"Loading {bm_name}"):
            task = bm.get_task(local_task_idx)
            env = create_libero_env(task, bm_name)
            extrinsic_c2w, intrinsic = get_camera_params_from_env(env)

            # 用任务描述作为 key（与 parquet 中的 task_name 一致）
            task_desc = extract_task_description(task.name)
            
            all_params[task_desc] = {
                'extrinsic': extrinsic_c2w.tolist(),
                'intrinsic': intrinsic.tolist(),
                'task_name': task.name,
                'benchmark': bm_name,
            }
            
            env.close()
            print(f"  ({bm_name}) {task_desc}")
    
    print(f"\nTotal tasks: {len(all_params)}")
    
    # 保存到文件
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print(f"\nSaved to: {args.output}")
    print("You can now run convert_libero_to_agibotworld.py which will load this file.")


if __name__ == '__main__':
    main()