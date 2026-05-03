#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

FILE_DIR = Path(__file__).resolve().parent
DATA_ROOT = FILE_DIR / "data"
os.environ["HF_HOME"] = str((FILE_DIR / ".hf_home").resolve())
os.environ["HF_DATASETS_CACHE"] = str((FILE_DIR / ".hf_home" / "datasets").resolve())

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_TASKS = [
    "open_laptop",
    "pick_dual_bottles",
    "put_bottles_dustbin",
    "place_burger_fries",
    "handover_block",
]

EPISODE_INSTRUCTIONS_PATH = Path("meta") / "episode_instructions.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild a 5-task RoboTwin multitask LeRobot dataset from single-task datasets."
    )
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument(
        "--source-repo-suffix",
        default="",
        help="Suffix appended to source single-task repo ids, e.g. '_rgbfix'.",
    )
    parser.add_argument("--output-repo-id", default="robotwin_multitask_5_cam_high")
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser.parse_args()


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def normalize_image(image):
    image = to_numpy(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return np.ascontiguousarray(image)


def build_source_datasets(
    task_config: str,
    expert_data_num: int,
    source_repo_suffix: str,
) -> list[tuple[str, LeRobotDataset]]:
    datasets = []
    for task_name in DEFAULT_TASKS:
        repo_id = f"robotwin_{task_name}_{task_config}_{expert_data_num}_cam_high{source_repo_suffix}"
        root = DATA_ROOT / repo_id
        if not root.is_dir():
            raise FileNotFoundError(f"Missing processed dataset: {root}")
        datasets.append((task_name, LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")))
    return datasets


def load_episode_instructions(dataset_root: Path, expected_episodes: int) -> list[dict]:
    metadata_path = dataset_root / EPISODE_INSTRUCTIONS_PATH
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing episode instructions metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        episode_instructions = json.load(f)

    if len(episode_instructions) != expected_episodes:
        raise ValueError(
            f"Episode instruction metadata length mismatch for {dataset_root}: "
            f"{len(episode_instructions)} vs {expected_episodes}"
        )

    return episode_instructions


def validate_compatible_sources(datasets: list[tuple[str, LeRobotDataset]]) -> tuple[dict, int, str]:
    _, first_dataset = datasets[0]
    first_features = first_dataset.meta.features
    first_fps = first_dataset.fps
    first_robot_type = first_dataset.meta.robot_type

    for task_name, dataset in datasets[1:]:
        if dataset.meta.features != first_features:
            raise ValueError(f"Feature mismatch for {task_name}")
        if dataset.fps != first_fps:
            raise ValueError(f"FPS mismatch for {task_name}: {dataset.fps} vs {first_fps}")
        if dataset.meta.robot_type != first_robot_type:
            raise ValueError(
                f"Robot type mismatch for {task_name}: {dataset.meta.robot_type} vs {first_robot_type}"
            )

    return first_features, first_fps, first_robot_type


def rebuild_episode(target_dataset: LeRobotDataset, source_dataset: LeRobotDataset, episode_index: int) -> None:
    episode_meta = source_dataset.meta.episodes[episode_index]
    start_idx = int(episode_meta["dataset_from_index"])
    end_idx = int(episode_meta["dataset_to_index"])

    for frame_idx in range(start_idx, end_idx):
        item = source_dataset[frame_idx]
        frame = {
            "observation.state": to_numpy(item["observation.state"]).astype(np.float32),
            "action": to_numpy(item["action"]).astype(np.float32),
            "observation.images.camera1": normalize_image(item["observation.images.camera1"]),
            "task": str(item["task"]),
        }
        target_dataset.add_frame(frame)

    target_dataset.save_episode()


def main():
    args = parse_args()

    datasets = build_source_datasets(args.task_config, args.expert_data_num, args.source_repo_suffix)
    features, fps, robot_type = validate_compatible_sources(datasets)
    merged_episode_instructions = []

    output_dir = DATA_ROOT / args.output_repo_id
    if output_dir.exists():
        raise FileExistsError(f"Merged dataset already exists: {output_dir}")

    merged_dataset = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=output_dir,
        robot_type=robot_type,
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
    )

    try:
        for task_name, dataset in datasets:
            print(f"Rebuilding task: {task_name}")
            task_episode_instructions = load_episode_instructions(dataset.root, dataset.meta.total_episodes)
            for episode_index in range(dataset.meta.total_episodes):
                rebuild_episode(merged_dataset, dataset, episode_index)
                merged_episode_instructions.append(
                    {
                        "episode_index": len(merged_episode_instructions),
                        "instructions": task_episode_instructions[episode_index]["instructions"],
                    }
                )
                print(
                    f"  merged {task_name} episode {episode_index + 1}/{dataset.meta.total_episodes}",
                    flush=True,
                )
    finally:
        merged_dataset.finalize()

    metadata_path = output_dir / EPISODE_INSTRUCTIONS_PATH
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(merged_episode_instructions, f, indent=2, ensure_ascii=False)

    print(f"Saved merged dataset to {output_dir}")


if __name__ == "__main__":
    main()
