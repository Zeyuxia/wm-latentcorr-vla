"""
测试避障纠错轨迹生成效果
用法: python script/test_correction_collision_free.py open_laptop demo_clean --episode 0 --perturb_step 1000 --perturb_scale 0.5 2>&1
python script/test_correction_collision_free.py open_laptop demo_clean --episode 0 --perturb_step 400 --perturb_scale 0.5 --no_correction 2>&1
python script/test_correction_collision_free.py open_laptop demo_clean --episode 0 --perturb_step 300 --perturb_scale 0.5 --optimal_step 2>&1
"""

import sys
sys.path.append("./")

import argparse
import pickle
import numpy as np
import imageio
import os
from tqdm import tqdm

from collect_data import class_decorator, get_embodiment_config
from correction_trajectory import CorrectionTrajectoryGenerator
from envs import CONFIGS_PATH
import yaml


def load_config(task_name, task_config):
    """加载配置（复用collect_data逻辑）"""
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    embodiment_type = args.get("embodiment")
    
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
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
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    return args


def extract_joint_positions(traj_data, arm='left'):
    """提取关节位置序列"""
    positions = [seg['position'] for seg in traj_data[f'{arm}_joint_path'] if seg['status'] == 'Success']
    return np.concatenate(positions, axis=0) if positions else None


def run_test(task_name, task_config, episode_idx=0, perturb_step=100, perturb_scale=0.1, 
             correction_steps=50, output_dir='eval_result/correction_test', collision_free=True,
             no_correction=False, optimal_steps=False):
    """运行纠错测试
    
    Args:
        no_correction: 如果为True，不使用规划器，直接跳到最近GT状态执行
        optimal_steps: 如果为True，使用动力学最优步数而不是固定correction_steps
    """
    
    # 加载配置和seed
    args = load_config(task_name, task_config)
    with open(os.path.join(args["save_path"], "seed.txt"), "r") as f:
        seeds = [int(s) for s in f.read().split()]
    seed = seeds[episode_idx]
    print(f"Seed: {seed}, Episode: {episode_idx}")
    
    # 加载轨迹
    traj_file = os.path.join(args["save_path"], "_traj_data", f"episode{episode_idx}.pkl")
    with open(traj_file, 'rb') as f:
        traj_data = pickle.load(f)
    left_traj = extract_joint_positions(traj_data, 'left')
    print(f"左臂轨迹长度: {len(left_traj)}")
    
    # 创建环境
    args.update({"render_freq": 0, "need_plan": False, "save_data": False})
    task_env = class_decorator(task_name)
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
    
    # 初始化纠错生成器
    generator = CorrectionTrajectoryGenerator(task_env=task_env if collision_free else None)
    frames = []
    
    def capture_frame():
        """捕获一帧图像"""
        task_env.cameras.update_picture()
        rgb = task_env.cameras.get_rgb()
        if 'head_camera' in rgb:
            return rgb['head_camera']['rgb']
        # 返回第一个可用的相机图像
        for cam_name in rgb:
            return rgb[cam_name]['rgb']
        return None
    
    # 阶段1: 执行正常轨迹
    print(f"\n=== 阶段1: 执行正常轨迹到步骤 {perturb_step} ===")
    for step in tqdm(range(min(perturb_step, len(left_traj))), desc="正常轨迹"):
        task_env.robot.set_arm_joints(left_traj[step], np.zeros(6), "left")
        task_env.scene.step()
        task_env.scene.update_render()
        if step % 5 == 0:
            frame = capture_frame()
            if frame is not None:
                frames.append(frame)
    
    # 阶段2: 添加扰动
    print(f"\n=== 阶段2: 添加扰动 (scale={perturb_scale}) ===")
    current_qpos = left_traj[perturb_step - 1].copy()
    np.random.seed(42)
    perturbed_qpos = current_qpos + np.clip(np.random.randn(6) * perturb_scale, -0.2, 0.2)
    print(f"扰动量: {perturbed_qpos - current_qpos}")
    
    # 先保存扰动前的几帧（用于对比）
    for _ in range(5):
        frame = capture_frame()
        if frame is not None:
            frames.append(frame)
    
    # 应用扰动
    task_env.robot.set_arm_joints(perturbed_qpos, np.zeros(6), "left")
    task_env.scene.step()
    task_env.scene.update_render()
    
    # 红框标记扰动后的帧（持续30帧，约1秒@30fps）
    for _ in range(30):
        task_env.scene.step()
        task_env.scene.update_render()
        frame = capture_frame()
        if frame is not None:
            frame = frame.copy()
            # 红色边框
            frame[:10, :] = [255, 0, 0]
            frame[-10:, :] = [255, 0, 0]
            frame[:, :10] = [255, 0, 0]
            frame[:, -10:] = [255, 0, 0]
            frames.append(frame)
    
    # 阶段3: 找最近GT状态
    print(f"\n=== 阶段3: 寻找最近GT状态 ===")
    remaining = left_traj[perturb_step:]
    nearest_idx = perturb_step + np.argmin(np.linalg.norm(remaining - perturbed_qpos, axis=1))
    target_idx = min(nearest_idx + 10, len(left_traj) - 1)
    target_qpos = left_traj[target_idx]
    print(f"目标索引: {target_idx}")
    
    # 阶段4: 生成纠错轨迹（或直接跳转）
    if no_correction:
        print(f"\n=== 阶段4: 直接跳转模式（无纠错轨迹）===")
        # 直接设置到目标状态
        task_env.robot.set_arm_joints(target_qpos, np.zeros(6), "left")
        task_env.scene.step()
        task_env.scene.update_render()
        
        # 用黄色边框标记跳转帧
        for _ in range(10):
            task_env.scene.step()
            task_env.scene.update_render()
            frame = capture_frame()
            if frame is not None:
                frame = frame.copy()
                frame[:10, :] = [255, 255, 0]
                frame[-10:, :] = [255, 255, 0]
                frame[:, :10] = [255, 255, 0]
                frame[:, -10:] = [255, 255, 0]
                frames.append(frame)
        print("跳转完成（黄色边框）")
    else:
        # 如果 optimal_steps=True，num_steps=None 让生成器返回最优步数
        steps_param = None if optimal_steps else correction_steps
        mode_str = '避障' if collision_free else ''
        steps_str = '最优步数' if optimal_steps else f'{correction_steps}步'
        print(f"\n=== 阶段4: 生成{mode_str}纠错轨迹 ({steps_str}) ===")
        try:
            correction_traj = generator.generate(
                start_qpos=perturbed_qpos, 
                end_qpos=target_qpos, 
                num_steps=steps_param, 
                collision_free=collision_free
            )
        except Exception as e:
            print(f"避障失败({e}), 使用toppra回退")
            correction_traj = generator.generate(
                start_qpos=perturbed_qpos, 
                end_qpos=target_qpos, 
                num_steps=correction_steps, 
                method='toppra',
                collision_free=False
            )
        print(f"纠错轨迹: {correction_traj.shape}")
        
        # 阶段5: 执行纠错轨迹
        print(f"\n=== 阶段5: 执行纠错轨迹 ===")
        for step in tqdm(range(len(correction_traj)), desc="纠错轨迹"):
            task_env.robot.set_arm_joints(correction_traj[step], np.zeros(6), "left")
            task_env.scene.step()
            task_env.scene.update_render()
            if step % 2 == 0:
                frame = capture_frame()
                if frame is not None:
                    frame = frame.copy()
                    frame[:10, :], frame[-10:, :], frame[:, :10], frame[:, -10:] = [0,255,0], [0,255,0], [0,255,0], [0,255,0]
                    frames.append(frame)
    
    # 阶段6: 继续执行剩余轨迹
    print(f"\n=== 阶段6: 执行剩余轨迹 ===")
    for step in tqdm(range(min(300, len(left_traj) - target_idx)), desc="剩余轨迹"):
        idx = target_idx + step
        task_env.robot.set_arm_joints(left_traj[idx], np.zeros(6), "left")
        task_env.scene.step()
        task_env.scene.update_render()
        if step % 5 == 0:
            frame = capture_frame()
            if frame is not None:
                frames.append(frame)
    
    # 保存视频
    os.makedirs(output_dir, exist_ok=True)
    mode_suffix = "nocorr" if no_correction else ("cf" if collision_free else "toppra")
    video_file = os.path.join(output_dir, f'{task_name}_ep{episode_idx}_{mode_suffix}.mp4')
    imageio.mimsave(video_file, frames, fps=30)
    print(f"\n✅ 视频已保存: {video_file} ({len(frames)} 帧)")
    
    task_env.close_env()
    return video_file


if __name__ == '__main__':
    from test_render import Sapien_TEST
    Sapien_TEST()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('task_name', type=str, help='任务名称')
    parser.add_argument('task_config', type=str, help='配置名称')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--perturb_step', type=int, default=100)
    parser.add_argument('--perturb_scale', type=float, default=0.1)
    parser.add_argument('--no_collision_free', action='store_true')
    parser.add_argument('--no_correction', action='store_true', help='不使用规划器，直接跳转到最近GT状态')
    parser.add_argument('--optimal_steps', action='store_true', help='使用动力学最优步数而不是固定步数')
    args = parser.parse_args()
    
    run_test(args.task_name, args.task_config, args.episode, args.perturb_step, 
             args.perturb_scale, collision_free=not args.no_collision_free,
             no_correction=args.no_correction, optimal_steps=args.optimal_steps)
