#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset


FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parents[1]
RAW_DATA_ROOT = REPO_ROOT / "data"
PROCESSED_ROOT = FILE_DIR / "data"

DEFAULT_TASKS = [
    "open_laptop",
    "pick_dual_bottles",
    "put_bottles_dustbin",
    "place_burger_fries",
    "handover_block",
]

EPISODE_INSTRUCTIONS_PATH = Path("meta") / "episode_instructions.json"

ROBOTWIN_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": None,
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": None,
    },
    "observation.images.camera1": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
}


def load_seen_instructions(task_dir: Path, episode_idx: int) -> list[str]:
    instruction_path = task_dir / "instructions" / f"episode{episode_idx}.json"
    if not instruction_path.is_file():
        raise FileNotFoundError(f"Missing instruction file: {instruction_path}")

    with instruction_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    seen_instructions = payload.get("seen", [])
    if not seen_instructions:
        raise ValueError(f"No usable instruction found in {instruction_path}")

    return [str(instruction) for instruction in seen_instructions]


def decode_head_camera(image_bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode head_camera frame")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)


def convert_episode(dataset: LeRobotDataset, episode_path: Path, task_name: str, instruction: str) -> None:
    with h5py.File(episode_path, "r") as root:
        left_arm = root["/joint_action/left_arm"][()]
        left_gripper = root["/joint_action/left_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        head_frames = root["/observation/head_camera/rgb"][()]

        num_steps = left_arm.shape[0]
        if num_steps < 2:
            raise ValueError(f"Episode too short: {episode_path}")

        for step_idx in range(num_steps - 1):
            state = np.concatenate(
                (
                    left_arm[step_idx],
                    [left_gripper[step_idx]],
                    right_arm[step_idx],
                    [right_gripper[step_idx]],
                ),
                axis=0,
            ).astype(np.float32)

            action = np.concatenate(
                (
                    left_arm[step_idx + 1],
                    [left_gripper[step_idx + 1]],
                    right_arm[step_idx + 1],
                    [right_gripper[step_idx + 1]],
                ),
                axis=0,
            ).astype(np.float32)

            frame = {
                "observation.state": state,
                "action": action,
                "observation.images.camera1": decode_head_camera(head_frames[step_idx]),
                "task": instruction,
            }
            dataset.add_frame(frame)

    dataset.save_episode()
    print(f"Converted {task_name}: {episode_path.name}")


def convert_task(task_name: str, task_config: str, expert_data_num: int, fps: int) -> Path:
    task_dir = RAW_DATA_ROOT / task_name / task_config
    raw_data_dir = task_dir / "data"
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Missing raw data directory: {raw_data_dir}")

    output_repo_id = f"robotwin_{task_name}_{task_config}_{expert_data_num}_cam_high"
    output_dir = PROCESSED_ROOT / output_repo_id
    if output_dir.exists():
        raise FileExistsError(f"Output dataset already exists: {output_dir}")

    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_dir,
        robot_type="aloha",
        fps=fps,
        features=ROBOTWIN_FEATURES,
        use_videos=True,
        image_writer_threads=4,
        batch_encoding_size=1,
    )
    episode_instructions = []

    try:
        for episode_idx in range(expert_data_num):
            episode_path = raw_data_dir / f"episode{episode_idx}.hdf5"
            if not episode_path.is_file():
                raise FileNotFoundError(f"Missing episode file: {episode_path}")
            instructions = load_seen_instructions(task_dir, episode_idx)
            convert_episode(dataset, episode_path, task_name, instructions[0])
            episode_instructions.append(
                {
                    "episode_index": episode_idx,
                    "instructions": instructions,
                }
            )
    finally:
        dataset.finalize()

    metadata_path = output_dir / EPISODE_INSTRUCTIONS_PATH
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(episode_instructions, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to {output_dir}")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RoboTwin raw data to a LeRobot dataset for SmolVLA.")
    parser.add_argument("task_name", nargs="?", default="all", help="Task name or 'all'")
    parser.add_argument("task_config", nargs="?", default="demo_clean")
    parser.add_argument("expert_data_num", nargs="?", type=int, default=50)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = DEFAULT_TASKS if args.task_name == "all" else [args.task_name]
    for task_name in tasks:
        convert_task(task_name, args.task_config, args.expert_data_num, args.fps)


if __name__ == "__main__":
    main()
