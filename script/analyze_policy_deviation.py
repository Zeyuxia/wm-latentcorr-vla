"""
分析 ACT policy 输出与 GT 轨迹的偏差，用于确定在线纠错的阈值

用法: python script/analyze_policy_deviation.py open_laptop demo_clean --num_episodes 50
"""

import sys
sys.path.append("./")
sys.path.append("./policy")

import os
import argparse
import numpy as np
import torch
import json
import yaml
import h5py
import cv2
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# 复用已有逻辑
from eval_policy import class_decorator, get_embodiment_config, eval_function_decorator
from envs import CONFIGS_PATH
from ACT.deploy_policy import encode_obs, reset_model


def load_config_and_model(task_name, task_config, ckpt_dir):
    """加载配置和模型"""
    with open("policy/ACT/deploy_policy.yml", "r") as f:
        usr_args = yaml.safe_load(f)
    
    usr_args.update({
        "task_name": task_name,
        "task_config": task_config,
        "ckpt_dir": ckpt_dir,
        "ckpt_setting": os.path.basename(ckpt_dir).replace("-50", ""),
    })
    
    with open(f"./task_config/{task_config}.yml", "r") as f:
        args = yaml.safe_load(f)
    
    args['task_name'] = task_name
    args['task_config'] = task_config
    
    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        _embodiment_types = yaml.safe_load(f)
    
    get_file = lambda t: _embodiment_types[t]["file_path"]
    
    if len(embodiment_type) == 1:
        args["left_robot_file"] = args["right_robot_file"] = get_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    else:
        args["left_robot_file"], args["right_robot_file"] = get_file(embodiment_type[0]), get_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    
    model = eval_function_decorator("ACT", "get_model")(usr_args)
    return args, model


def load_gt_trajectory(data_dir, episode_idx):
    """从 hdf5 加载双臂 GT 轨迹（arm + gripper + 末端位姿）"""
    hdf5_path = os.path.join(data_dir, f"episode{episode_idx}.hdf5")
    with h5py.File(hdf5_path, 'r') as f:
        left_arm = f['joint_action/left_arm'][:]    # (T, 6)
        right_arm = f['joint_action/right_arm'][:]  # (T, 6)
        left_gripper = f['joint_action/left_gripper'][:]
        right_gripper = f['joint_action/right_gripper'][:]
        # 末端位姿 (T, 7): pos(3) + quat(4)
        left_endpose = f['endpose/left_endpose'][:]   # (T, 7)
        right_endpose = f['endpose/right_endpose'][:]  # (T, 7)
    return {
        'left_arm': left_arm,
        'right_arm': right_arm,
        'left_gripper': left_gripper,
        'right_gripper': right_gripper,
        'left_endpose': left_endpose,    # 直接用数据集里的末端位姿
        'right_endpose': right_endpose,
    }


def project_point_to_image(point_3d, intrinsic, extrinsic):
    """将3D点投影到图像平面"""
    # point_3d: (3,) 世界坐标系下的点
    # extrinsic: (4,4) world to camera
    # intrinsic: (3,3) camera to image
    
    point_homo = np.append(point_3d, 1.0)
    point_cam = extrinsic @ point_homo  # (4,)
    point_cam = point_cam[:3]
    
    if point_cam[2] <= 0:
        return None  # 点在相机后面
    
    point_img = intrinsic @ point_cam
    u, v = point_img[0] / point_img[2], point_img[1] / point_img[2]
    return int(u), int(v)


def draw_marker(img, pos, color, size=8):
    """在图像上绘制标记"""
    if pos is None:
        return
    u, v = pos
    h, w = img.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(img, (u, v), size, color, -1)
        cv2.circle(img, (u, v), size+2, (255, 255, 255), 2)


def analyze_episode(task_env, model, gt_data, episode_idx, max_steps=500, save_video=True, output_dir=None, open_loop_chunk=False):
    """分析单个 episode 的双臂偏差
    
    Args:
        open_loop_chunk: 如果 True，每次推理后执行整个 chunk 不更新观测（纯开环）
                        如果 False，每步都获取观测（默认，和 eval 一致）
    """
    reset_model(model)
    
    left_gt = gt_data['left_arm']
    right_gt = gt_data['right_arm']
    left_gripper_gt = gt_data['left_gripper']
    right_gripper_gt = gt_data['right_gripper']
    left_endpose_gt = gt_data['left_endpose']    # (T, 7): pos(3) + quat(4)
    right_endpose_gt = gt_data['right_endpose']
    
    gt_len = len(left_gt)
    deviations = []
    frames = []
    
    chunk_size = model.num_queries  # 通常是 50
    step = 0
    
    pbar = tqdm(total=max_steps, desc=f"Episode {episode_idx}", leave=False)
    
    while step < max_steps:
        # 获取观测并推理
        obs = encode_obs(task_env.get_obs())
        
        if open_loop_chunk:
            # 开环模式：直接获取整个 chunk
            with torch.no_grad():
                qpos_numpy = np.array(obs["qpos"])
                qpos_normalized = model.pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos_normalized).float().to(model.device).unsqueeze(0)
                
                curr_images = []
                for cam_name in ["head_cam", "left_cam", "right_cam"]:
                    curr_images.append(obs[cam_name])
                curr_image = np.stack(curr_images, axis=0)
                curr_image = torch.from_numpy(curr_image).float().to(model.device).unsqueeze(0)
                
                all_actions = model.policy(qpos, curr_image)  # (1, chunk_size, 14)
                all_actions = all_actions[0].cpu().numpy()  # (chunk_size, 14)
                all_actions = model.post_process(all_actions)  # 反归一化
            
            # 执行整个 chunk
            for chunk_step in range(chunk_size):
                if step >= max_steps:
                    break
                    
                action = all_actions[chunk_step]
                policy_left_arm = action[:6]
                policy_right_arm = action[7:13]
                policy_left_gripper = action[6]
                policy_right_gripper = action[13]
                
                # 计算偏差
                if step < gt_len:
                    gt_left_arm = left_gt[step]
                    gt_right_arm = right_gt[step]
                    left_diff = policy_left_arm - gt_left_arm
                    right_diff = policy_right_arm - gt_right_arm
                    total_diff = np.concatenate([left_diff, right_diff])
                    
                    deviations.append({
                        'step': step,
                        'chunk_step': chunk_step,
                        'total_l2': float(np.linalg.norm(total_diff)),
                        'left_l2': float(np.linalg.norm(left_diff)),
                        'right_l2': float(np.linalg.norm(right_diff)),
                        'max_abs': float(np.max(np.abs(total_diff))),
                        'left_diff': left_diff.tolist(),
                        'right_diff': right_diff.tolist(),
                    })
                    
                    if save_video:
                        gt_left_tcp = left_endpose_gt[step][:3]
                        gt_right_tcp = right_endpose_gt[step][:3]
                        frame = capture_frame_with_projection(task_env, gt_left_tcp, gt_right_tcp, deviations[-1])
                        if frame is not None:
                            frames.append(frame)
                else:
                    if save_video:
                        frame = capture_frame_with_projection(task_env, None, None, 
                            {'step': step, 'total_l2': 0, 'left_l2': 0, 'right_l2': 0})
                        if frame is not None:
                            frames.append(frame)
                
                # 执行 action（和 eval 一致）
                task_env.take_action(action)
                
                step += 1
                pbar.update(1)
                
                if task_env.eval_success:
                    print(f"  任务在 step {step} 完成!")
                    break
            
            if task_env.eval_success:
                break
        else:
            # 正常模式：每步获取观测（和 eval 一致）
            action = model.get_action(obs)[0]
            
            policy_left_arm = action[:6]
            policy_right_arm = action[7:13]
            policy_left_gripper = action[6]
            policy_right_gripper = action[13]
            
            if step < gt_len:
                gt_left_arm = left_gt[step]
                gt_right_arm = right_gt[step]
                left_diff = policy_left_arm - gt_left_arm
                right_diff = policy_right_arm - gt_right_arm
                total_diff = np.concatenate([left_diff, right_diff])
                
                deviations.append({
                    'step': step,
                    'total_l2': float(np.linalg.norm(total_diff)),
                    'left_l2': float(np.linalg.norm(left_diff)),
                    'right_l2': float(np.linalg.norm(right_diff)),
                    'max_abs': float(np.max(np.abs(total_diff))),
                    'left_diff': left_diff.tolist(),
                    'right_diff': right_diff.tolist(),
                })
                
                if save_video:
                    gt_left_tcp = left_endpose_gt[step][:3]
                    gt_right_tcp = right_endpose_gt[step][:3]
                    frame = capture_frame_with_projection(task_env, gt_left_tcp, gt_right_tcp, deviations[-1])
                    if frame is not None:
                        frames.append(frame)
            else:
                if save_video:
                    frame = capture_frame_with_projection(task_env, None, None,
                        {'step': step, 'total_l2': 0, 'left_l2': 0, 'right_l2': 0})
                    if frame is not None:
                        frames.append(frame)
            
            # 执行 action（和 eval 一致）
            task_env.take_action(action)
            
            step += 1
            pbar.update(1)
            
            if task_env.eval_success:
                print(f"  任务在 step {step} 完成!")
                break
    
    pbar.close()
    
    # 保存视频
    if save_video and frames and output_dir:
        video_path = os.path.join(output_dir, f"episode_{episode_idx}_deviation.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"视频已保存: {video_path}")
    
    return deviations


def capture_frame_with_projection(task_env, gt_left_tcp, gt_right_tcp, deviation):
    """捕获帧并可视化偏差（GT 为 None 时不投影 GT）"""
    task_env.cameras.update_picture()
    rgb = task_env.cameras.get_rgb()
    config = task_env.cameras.get_config()
    
    if 'head_camera' not in rgb:
        return None
    
    frame = rgb['head_camera']['rgb'].copy()
    cam_config = config.get('head_camera', {})
    
    intrinsic = cam_config.get('intrinsic_cv')
    extrinsic = cam_config.get('extrinsic_cv')
    
    if intrinsic is not None and extrinsic is not None:
        # 获取当前末端位置（policy执行后的位置）
        left_tcp = task_env.robot.get_left_tcp_pose()[:3]
        right_tcp = task_env.robot.get_right_tcp_pose()[:3]
        
        # 投影当前位置（绿色 - policy）
        left_pos = project_point_to_image(left_tcp, intrinsic, extrinsic)
        right_pos = project_point_to_image(right_tcp, intrinsic, extrinsic)
        
        draw_marker(frame, left_pos, (0, 255, 0), 5)   # 绿色 - 左臂 policy
        draw_marker(frame, right_pos, (0, 255, 0), 5)  # 绿色 - 右臂 policy
        
        # 投影 GT 位置（红色）- 仅当 GT 不为 None 时
        if gt_left_tcp is not None and gt_right_tcp is not None:
            gt_left_pos = project_point_to_image(gt_left_tcp, intrinsic, extrinsic)
            gt_right_pos = project_point_to_image(gt_right_tcp, intrinsic, extrinsic)
            
            draw_marker(frame, gt_left_pos, (255, 0, 0), 4)   # 红色 - 左臂 GT
            draw_marker(frame, gt_right_pos, (255, 0, 0), 4)  # 红色 - 右臂 GT
            
            # 画连线
            if left_pos and gt_left_pos:
                cv2.line(frame, left_pos, gt_left_pos, (255, 255, 0), 1)
            if right_pos and gt_right_pos:
                cv2.line(frame, right_pos, gt_right_pos, (255, 255, 0), 1)
    
    # 添加文字信息（黑色描边 + 彩色文字）
    def put_text_outlined(img, text, pos, scale, color):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)  # 黑色描边
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    
    put_text_outlined(frame, f"Step: {deviation['step']}", (10, 20), 0.4, (255, 255, 0))  # 黄色
    put_text_outlined(frame, f"Total L2: {deviation['total_l2']:.4f}", (10, 38), 0.4, (255, 255, 0))
    put_text_outlined(frame, f"Left L2: {deviation['left_l2']:.4f}", (10, 56), 0.4, (0, 255, 0))  # 绿色
    put_text_outlined(frame, f"Right L2: {deviation['right_l2']:.4f}", (10, 74), 0.4, (0, 255, 0))
    
    # 图例
    put_text_outlined(frame, "Green: Policy  Red: GT", (10, frame.shape[0] - 6), 0.35, (255, 255, 0))
    
    return frame


def plot_deviation_curves(all_deviations, output_dir, open_loop=False):
    """绘制偏差曲线"""
    mode = "open_loop" if open_loop else "eval"
    
    # 绘制每个 episode 的曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Policy vs GT Deviation ({mode} mode)', fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_deviations)))
    
    for ep_idx, deviations in enumerate(all_deviations):
        steps = [d['step'] for d in deviations]
        total_l2 = [d['total_l2'] for d in deviations]
        left_l2 = [d['left_l2'] for d in deviations]
        right_l2 = [d['right_l2'] for d in deviations]
        max_abs = [d['max_abs'] for d in deviations]
        
        axes[0, 0].plot(steps, total_l2, color=colors[ep_idx], alpha=0.7, label=f'ep{ep_idx}')
        axes[0, 1].plot(steps, left_l2, color=colors[ep_idx], alpha=0.7, label=f'ep{ep_idx}')
        axes[1, 0].plot(steps, right_l2, color=colors[ep_idx], alpha=0.7, label=f'ep{ep_idx}')
        axes[1, 1].plot(steps, max_abs, color=colors[ep_idx], alpha=0.7, label=f'ep{ep_idx}')
    
    axes[0, 0].set_title('Total L2 (12 joints)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('L2 Norm (rad)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Left Arm L2 (6 joints)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('L2 Norm (rad)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Right Arm L2 (6 joints)')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('L2 Norm (rad)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Max Abs (single joint)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Abs Deviation (rad)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'deviation_curves_{mode}.png'), dpi=150)
    plt.close()
    print(f"曲线图已保存: {os.path.join(output_dir, f'deviation_curves_{mode}.png')}")
    
    # 绘制每个关节的偏差曲线（使用第一个 episode）
    if all_deviations:
        fig, axes = plt.subplots(2, 6, figsize=(18, 8))
        fig.suptitle(f'Per-Joint Deviation (Episode 0, {mode} mode)', fontsize=14)
        
        deviations = all_deviations[0]
        steps = [d['step'] for d in deviations]
        
        for j in range(6):
            left_j = [d['left_diff'][j] for d in deviations]
            right_j = [d['right_diff'][j] for d in deviations]
            
            axes[0, j].plot(steps, left_j, 'b-', alpha=0.7)
            axes[0, j].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0, j].set_title(f'Left Joint {j+1}')
            axes[0, j].set_xlabel('Step')
            axes[0, j].set_ylabel('Δθ (rad)')
            axes[0, j].grid(True, alpha=0.3)
            
            axes[1, j].plot(steps, right_j, 'r-', alpha=0.7)
            axes[1, j].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1, j].set_title(f'Right Joint {j+1}')
            axes[1, j].set_xlabel('Step')
            axes[1, j].set_ylabel('Δθ (rad)')
            axes[1, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'per_joint_deviation_{mode}.png'), dpi=150)
        plt.close()
        print(f"关节曲线图已保存: {os.path.join(output_dir, f'per_joint_deviation_{mode}.png')}")


def compute_statistics(all_deviations):
    """计算统计数据和建议阈值"""
    all_total_l2 = [d['total_l2'] for ep in all_deviations for d in ep]
    all_left_l2 = [d['left_l2'] for ep in all_deviations for d in ep]
    all_right_l2 = [d['right_l2'] for ep in all_deviations for d in ep]
    all_max = [d['max_abs'] for ep in all_deviations for d in ep]
    
    def calc_stats(arr):
        arr = np.array(arr)
        return {k: float(f(arr)) for k, f in [
            ('mean', np.mean), ('std', np.std), ('median', np.median),
            ('max', np.max), ('p90', lambda x: np.percentile(x, 90)),
            ('p95', lambda x: np.percentile(x, 95)), ('p99', lambda x: np.percentile(x, 99)),
        ]}
    
    # 计算每个关节的偏差（左臂6个 + 右臂6个）
    per_joint_left = defaultdict(list)
    per_joint_right = defaultdict(list)
    for ep in all_deviations:
        for d in ep:
            for j, v in enumerate(d['left_diff']):
                per_joint_left[j].append(abs(v))
            for j, v in enumerate(d['right_diff']):
                per_joint_right[j].append(abs(v))
    
    all_total_l2 = np.array(all_total_l2)
    all_left_l2 = np.array(all_left_l2)
    all_right_l2 = np.array(all_right_l2)
    all_max = np.array(all_max)
    
    stats = {
        'total_l2': calc_stats(all_total_l2),
        'left_l2': calc_stats(all_left_l2),
        'right_l2': calc_stats(all_right_l2),
        'max_abs': calc_stats(all_max),
        'per_joint_left': {f'joint_{j}': {'mean': float(np.mean(v)), 'max': float(np.max(v)), 
                                           'p95': float(np.percentile(v, 95))} 
                           for j, v in per_joint_left.items()},
        'per_joint_right': {f'joint_{j}': {'mean': float(np.mean(v)), 'max': float(np.max(v)), 
                                            'p95': float(np.percentile(v, 95))} 
                            for j, v in per_joint_right.items()},
        'recommended_thresholds': {
            'conservative_p99': {
                'total_l2': float(np.percentile(all_total_l2, 99)),
                'left_l2': float(np.percentile(all_left_l2, 99)),
                'right_l2': float(np.percentile(all_right_l2, 99)),
                'max_abs': float(np.percentile(all_max, 99)),
            },
            'moderate_p95': {
                'total_l2': float(np.percentile(all_total_l2, 95)),
                'left_l2': float(np.percentile(all_left_l2, 95)),
                'right_l2': float(np.percentile(all_right_l2, 95)),
                'max_abs': float(np.percentile(all_max, 95)),
            },
            'aggressive_p90': {
                'total_l2': float(np.percentile(all_total_l2, 90)),
                'left_l2': float(np.percentile(all_left_l2, 90)),
                'right_l2': float(np.percentile(all_right_l2, 90)),
                'max_abs': float(np.percentile(all_max, 90)),
            },
        }
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_name', type=str)
    parser.add_argument('task_config', type=str)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=500, help='每个episode最大步数，默认500')
    parser.add_argument('--output_dir', type=str, default='eval_result/deviation_analysis')
    parser.add_argument('--data_dir', type=str, default=None, help='hdf5数据目录')
    parser.add_argument('--no_video', action='store_true', help='不生成可视化视频')
    parser.add_argument('--open_loop_chunk', action='store_true', help='开环模式：每次推理后执行整个 chunk 不更新观测')
    args = parser.parse_args()
    
    if args.ckpt_dir is None:
        args.ckpt_dir = f"policy/ACT/act_ckpt/act-{args.task_name}/{args.task_config}-50"
    if args.data_dir is None:
        args.data_dir = f"data/{args.task_name}/{args.task_config}/data"
    
    print(f"=== Policy 偏差分析 ===\n任务: {args.task_name}\n模型: {args.ckpt_dir}\n数据: {args.data_dir}")
    
    # 加载配置和模型
    config, model = load_config_and_model(args.task_name, args.task_config, args.ckpt_dir)
    
    # 获取 episode 数量
    episode_files = sorted([f for f in os.listdir(args.data_dir) if f.startswith('episode') and f.endswith('.hdf5')])
    num_episodes = min(args.num_episodes, len(episode_files))
    
    # 加载 seeds
    seed_file = os.path.join(os.path.dirname(args.data_dir), "seed.txt")
    if os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            seeds = [int(s) for s in f.read().split()]
    else:
        seeds = list(range(num_episodes))
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.task_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析每个 episode
    all_deviations = []
    # 基础配置（用于 play_once 规划）
    base_config = config.copy()
    base_config.update({"render_freq": 0, "save_data": False, "eval_mode": True})
    # 执行配置（用于 policy 执行，不需要规划）
    exec_config = base_config.copy()
    exec_config.update({"need_plan": False})
    save_video = not args.no_video
    
    for ep_idx in range(num_episodes):
        print(f"\n=== Episode {ep_idx} (seed={seeds[ep_idx]}) ===")
        
        # 从 hdf5 加载 GT 轨迹
        gt_data = load_gt_trajectory(args.data_dir, ep_idx)
        print(f"GT 轨迹长度: {len(gt_data['left_arm'])}")
        
        # 创建环境（和 eval_policy 保持一致：先 play_once 初始化任务属性如 arm_tag）
        task_env = class_decorator(args.task_name)
        task_env.setup_demo(now_ep_num=ep_idx, seed=seeds[ep_idx], is_test=True, **base_config)
        try:
            task_env.play_once()  # 初始化 arm_tag 等任务属性
        except Exception as e:
            print(f"  Warning: play_once failed: {e}")
        task_env.close_env()
        
        # 重新初始化环境用于 policy 评估（不需要规划）
        task_env.setup_demo(now_ep_num=ep_idx, seed=seeds[ep_idx], is_test=True, **exec_config)
        
        # 使用任务实际的 step_lim（和 eval 一致），如果未设置则用默认值
        actual_max_steps = task_env.step_lim if task_env.step_lim else args.max_steps
        if ep_idx == 0:
            print(f"使用 step_lim: {actual_max_steps}")
        
        deviations = analyze_episode(task_env, model, gt_data, ep_idx, actual_max_steps,
                                      save_video=save_video, output_dir=output_dir,
                                      open_loop_chunk=args.open_loop_chunk)
        all_deviations.append(deviations)
        
        # 保存 episode 结果
        with open(os.path.join(output_dir, f"episode_{ep_idx}.json"), 'w') as f:
            json.dump({'episode': ep_idx, 'seed': seeds[ep_idx], 'deviations': deviations}, f, indent=2)
        
        task_env.close_env()
    
    # 绘制偏差曲线
    plot_deviation_curves(all_deviations, output_dir, open_loop=args.open_loop_chunk)
    
    # 计算统计并保存
    stats = compute_statistics(all_deviations)
    stats['config'] = {'task': args.task_name, 'ckpt': args.ckpt_dir, 'num_episodes': len(all_deviations)}
    
    with open(os.path.join(output_dir, "statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印结果
    print("\n" + "="*60 + "\n📊 偏差统计 (双臂 arm only)\n" + "="*60)
    print(f"Total L2: mean={stats['total_l2']['mean']:.4f}, p95={stats['total_l2']['p95']:.4f}, max={stats['total_l2']['max']:.4f}")
    print(f"Left L2:  mean={stats['left_l2']['mean']:.4f}, p95={stats['left_l2']['p95']:.4f}, max={stats['left_l2']['max']:.4f}")
    print(f"Right L2: mean={stats['right_l2']['mean']:.4f}, p95={stats['right_l2']['p95']:.4f}, max={stats['right_l2']['max']:.4f}")
    print(f"Max Abs:  mean={stats['max_abs']['mean']:.4f}, p95={stats['max_abs']['p95']:.4f}, max={stats['max_abs']['max']:.4f}")
    
    print("\n🎯 建议阈值 (arm only):")
    for name, thresh in stats['recommended_thresholds'].items():
        print(f"  {name}: Total_L2={thresh['total_l2']:.4f}, Left_L2={thresh['left_l2']:.4f}, Right_L2={thresh['right_l2']:.4f}")
    
    print(f"\n✅ 结果已保存到: {output_dir}")


if __name__ == '__main__':
    from test_render import Sapien_TEST
    Sapien_TEST()
    main()
