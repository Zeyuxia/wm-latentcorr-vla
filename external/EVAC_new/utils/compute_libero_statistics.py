"""
计算 AgiBotWorld-format 数据集的统计数据（用于训练时的归一化）

统计 delta actions 的 mean, std, min, max, q01, q99。
delta actions 格式: [dx, dy, dz, droll, dpitch, dyaw, gripper] * 2 (左右臂)。
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import numpy as np
from tqdm import tqdm

# 直接复用训练时使用的函数，确保统计口径与训练一致
from evac.lvdm.data.get_actions import parse_h5


def compute_statistics(data_dir: str, split: str = 'train'):
    """
    遍历所有 episode 计算 delta actions 的统计数据
    
    返回格式与 StatisticInfo 兼容:
    - mean: 12 维 (左右臂各 6 维位姿差分，不含 gripper)
    - std, min, max, q01, q99: 同上
    """
    split_dir = os.path.join(data_dir, split)
    episodes = sorted(os.listdir(split_dir))
    
    print(f"Found {len(episodes)} episodes in {split_dir}")
    
    all_delta_actions = []
    
    for ep_name in tqdm(episodes, desc="Processing episodes"):
        ep_dir = os.path.join(split_dir, ep_name)
        h5_path = os.path.join(ep_dir, 'proprio_stats.h5')
        
        if not os.path.exists(h5_path):
            print(f"Warning: {h5_path} not found, skipping...")
            continue
        
        try:
            # 直接复用 parse_h5 函数
            _, delta_actions = parse_h5(h5_path, slices=None, delta_act_sidx=1)
            # delta_actions: (T-1, 14) - [dx, dy, dz, droll, dpitch, dyaw, gripper] * 2
            # 我们只需要位姿部分，不需要 gripper (共 12 维)
            pose_delta = np.concatenate([
                delta_actions[:, :6],   # 左臂 xyz, rpy
                delta_actions[:, 7:13]  # 右臂 xyz, rpy
            ], axis=1)  # (T-1, 12)
            
            all_delta_actions.append(pose_delta)
        except Exception as e:
            print(f"Error processing {ep_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_delta_actions) == 0:
        raise ValueError("No valid episodes found!")
    
    # 合并所有数据
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)  # (N, 12)
    print(f"Total samples: {len(all_delta_actions)}")
    
    # 计算统计数据
    stats = {
        'mean': all_delta_actions.mean(axis=0).tolist(),
        'std': all_delta_actions.std(axis=0).tolist(),
        'min': all_delta_actions.min(axis=0).tolist(),
        'max': all_delta_actions.max(axis=0).tolist(),
        'q01': np.percentile(all_delta_actions, 1, axis=0).tolist(),
        'q99': np.percentile(all_delta_actions, 99, axis=0).tolist(),
    }
    
    return stats


def format_as_python_dict(stats: dict, name: str = 'robotwin_new') -> str:
    """格式化为可直接粘贴到 statistics.py 的 Python 代码"""
    lines = [f'    "{name}": {{']
    
    for key in ['mean', 'std', 'max', 'min', 'q01', 'q99']:
        lines.append(f'        "{key}": [')
        for i, val in enumerate(stats[key]):
            comma = ',' if i < len(stats[key]) - 1 else ''
            lines.append(f'            {val}{comma}')
        lines.append('        ],')
    
    lines.append('    },')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Compute dataset statistics for training')
    parser.add_argument('--data_dir', type=str, default='/data/yujieyang/datasets/robotwin_agibotworld_format_mixed5tasks_clean50_plus_pi05_rollout',
                        help='AgiBotWorld format dataset directory')
    parser.add_argument('--split', type=str, default='train', help='Split name')
    parser.add_argument('--output', type=str, default='/data/yujieyang/datasets/robotwin_agibotworld_format_mixed5tasks_clean50_plus_pi05_rollout/robotwin_new_statistics.txt',
                        help='Output file path (default: print to stdout)')
    parser.add_argument('--domain-name', type=str, default='robotwin_new',
                        help='Domain name used in statistics.py output')
    args = parser.parse_args()
    
    print(f"Computing statistics for {args.data_dir}/{args.split}...")
    stats = compute_statistics(args.data_dir, args.split)
    
    print("\n" + "="*60)
    print("Statistics computed successfully!")
    print("="*60 + "\n")
    
    # 打印简要统计
    print("Summary:")
    print(f"  Mean (first 6): {stats['mean'][:6]}")
    print(f"  Std (first 6):  {stats['std'][:6]}")
    print(f"  Min (first 6):  {stats['min'][:6]}")
    print(f"  Max (first 6):  {stats['max'][:6]}")
    
    # 生成可粘贴的代码
    code = format_as_python_dict(stats, args.domain_name)
    
    print("\n" + "="*60)
    print("Add the following to evac/lvdm/data/statistics.py:")
    print("="*60)
    print(code)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(code)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
