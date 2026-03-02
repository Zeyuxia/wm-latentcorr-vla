"""
通过 seed 重现场景并获取所有物体的初始位姿和第一帧图片
用法: python script/get_scene_poses.py <task_name> <task_config> <episode_idx>
示例: python script/get_scene_poses.py open_laptop demo_clean 0
"""

import sys
sys.path.append("./")

import json
import os
from argparse import ArgumentParser

# 从 collect_data.py 复用函数
from collect_data import class_decorator, get_embodiment_config
from envs import CONFIGS_PATH
import yaml


def get_all_actor_poses(task_env):
    """获取场景中所有物体的位姿"""
    actor_poses = {}
    for actor in task_env.scene.get_all_actors():
        name = actor.get_name()
        pose = actor.get_pose()
        actor_poses[name] = {
            "position": pose.p.tolist(),  # [x, y, z]
            "quaternion": pose.q.tolist(),  # [w, x, y, z]
        }
    return actor_poses


def load_config(task_name, task_config):
    """加载任务配置（复用 collect_data.py 中的逻辑）"""
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    # 加载 embodiment 配置
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type):
        robot_file = _embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise ValueError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    args["embodiment_name"] = embodiment_name

    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])

    return args


def main(task_name, task_config, episode_idx):
    # 加载配置
    args = load_config(task_name, task_config)

    # 读取 seed
    seed_file_path = os.path.join(args["save_path"], "seed.txt")
    with open(seed_file_path, "r") as f:
        seed_list = [int(s) for s in f.read().split()]

    if episode_idx >= len(seed_list):
        raise ValueError(f"Episode {episode_idx} not found. Available: 0-{len(seed_list)-1}")

    seed = seed_list[episode_idx]
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"Config: {task_config}")
    print(f"Episode: {episode_idx}")
    print(f"Seed: {seed}")
    print(f"{'='*50}\n")

    # 创建任务环境（使用从 collect_data 导入的 class_decorator）
    task_env = class_decorator(task_name)
    
    # 设置参数，不渲染 viewer
    args["render_freq"] = 0
    args["need_plan"] = False
    args["save_data"] = False

    # 使用 seed 重现场景
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)

    # 获取所有物体位姿
    actor_poses = get_all_actor_poses(task_env)

    print("=" * 50)
    print("All Actor Poses (Initial State):")
    print("=" * 50)
    
    for name, pose_info in actor_poses.items():
        print(f"\n{name}:")
        print(f"  Position: {pose_info['position']}")
        print(f"  Quaternion (wxyz): {pose_info['quaternion']}")

    # 保存位姿到文件
    output_dir = os.path.join(args["save_path"], "scene_snapshots")
    os.makedirs(output_dir, exist_ok=True)
    
    poses_output_path = os.path.join(output_dir, f"actor_poses_episode{episode_idx}.json")
    with open(poses_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "task_name": task_name,
            "task_config": task_config,
            "episode_idx": episode_idx,
            "seed": seed,
            "actor_poses": actor_poses
        }, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ Poses saved to: {poses_output_path}")

    # 保存第一帧图片（使用 task_env 内置的 save_camera_rgb 方法）
    img_output_path = os.path.join(output_dir, f"first_frame_episode{episode_idx}.png")
    task_env.save_camera_rgb(img_output_path, camera_name='head_camera')
    print(f"✅ First frame image saved to: {img_output_path}")

    # 如果有手腕相机，也保存
    try:
        left_wrist_path = os.path.join(output_dir, f"first_frame_episode{episode_idx}_left_wrist.png")
        task_env.save_camera_rgb(left_wrist_path, camera_name='left_wrist_camera')
        print(f"✅ Left wrist camera image saved to: {left_wrist_path}")
        
        right_wrist_path = os.path.join(output_dir, f"first_frame_episode{episode_idx}_right_wrist.png")
        task_env.save_camera_rgb(right_wrist_path, camera_name='right_wrist_camera')
        print(f"✅ Right wrist camera image saved to: {right_wrist_path}")
    except Exception as e:
        print(f"⚠️ Could not save wrist camera images: {e}")

    # 关闭环境
    task_env.close_env()
    
    print(f"\n{'='*50}")
    print("Done!")
    print(f"{'='*50}")


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str, help="Task name, e.g., open_laptop")
    parser.add_argument("task_config", type=str, help="Task config, e.g., demo_clean")
    parser.add_argument("episode_idx", type=int, help="Episode index, e.g., 0")
    args = parser.parse_args()

    main(args.task_name, args.task_config, args.episode_idx)
