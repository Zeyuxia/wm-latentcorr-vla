from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from .robotwin_common import (
    DEFAULT_CAMERA_MODE,
    DEFAULT_SECONDARY_CAMERA,
    DEFAULT_PROCESSED_DATA_ROOT,
    ROBOTWIN_ROOT,
    camera_mode_dir_suffix,
    prepare_openpi_imports,
    selected_cameras,
)


def _images_encoding(images: list[np.ndarray]) -> tuple[list[bytes], int]:
    encoded: list[bytes] = []
    max_len = 0
    for image in images:
        success, jpg = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("Failed to encode image as jpg")
        data = jpg.tobytes()
        encoded.append(data)
        max_len = max(max_len, len(data))
    padded = [data.ljust(max_len, b"\0") for data in encoded]
    return padded, max_len


def _load_robotwin_episode(
    episode_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    with h5py.File(episode_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        joint_vector = root["/joint_action/vector"][()] if "/joint_action/vector" in root else None
        image_dict = {}
        for cam_name in root["/observation"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
    return left_gripper, left_arm, right_gripper, right_arm, joint_vector, image_dict


def _decode_and_resize(image_bytes: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode jpeg frame from raw episode")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def convert_raw_episode_to_pi0_format(
    raw_root: Path,
    episode_id: int,
    output_root: Path,
    *,
    output_episode_id: int | None = None,
    description_type: str = "seen",
    camera_mode: str = DEFAULT_CAMERA_MODE,
    secondary_camera: str = DEFAULT_SECONDARY_CAMERA,
) -> None:
    episode_path = raw_root / "data" / f"episode{episode_id}.hdf5"
    if not episode_path.is_file():
        raise FileNotFoundError(f"Raw episode not found: {episode_path}")

    instruction_path = raw_root / "instructions" / f"episode{episode_id}.json"
    if not instruction_path.is_file():
        raise FileNotFoundError(f"Instruction json not found: {instruction_path}")

    with instruction_path.open("r", encoding="utf-8") as f:
        instruction_dict = json.load(f)
    instructions = instruction_dict[description_type]

    dst_episode_id = int(episode_id if output_episode_id is None else output_episode_id)
    episode_dir = output_root / f"episode_{dst_episode_id}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "instructions.json").open("w", encoding="utf-8") as f:
        json.dump({"instructions": instructions}, f, indent=2)

    left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, joint_vector_all, image_dict = _load_robotwin_episode(
        episode_path
    )

    qpos = []
    actions = []
    resolved_cameras = selected_cameras(camera_mode, secondary_camera)
    camera_buffers = {camera_name: [] for camera_name, _ in resolved_cameras}
    left_arm_dims = []
    right_arm_dims = []

    for step in range(left_gripper_all.shape[0]):
        left_gripper = left_gripper_all[step]
        left_arm = left_arm_all[step]
        right_gripper = right_gripper_all[step]
        right_arm = right_arm_all[step]

        if joint_vector_all is not None:
            state = np.asarray(joint_vector_all[step], dtype=np.float32)
        else:
            state = np.asarray(
                left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper],
                dtype=np.float32,
            )

        if step != left_gripper_all.shape[0] - 1:
            qpos.append(state)
            for camera_name, robotwin_key in resolved_cameras:
                camera_buffers[camera_name].append(_decode_and_resize(image_dict[robotwin_key][step]))

        if step != 0:
            actions.append(state)
            left_arm_dims.append(int(left_arm.shape[0]))
            right_arm_dims.append(int(right_arm.shape[0]))

    hdf5_path = episode_dir / f"episode_{dst_episode_id}.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
        observations = f.create_group("observations")
        observations.create_dataset("qpos", data=np.asarray(qpos, dtype=np.float32))
        observations.create_dataset("left_arm_dim", data=np.asarray(left_arm_dims, dtype=np.int32))
        observations.create_dataset("right_arm_dim", data=np.asarray(right_arm_dims, dtype=np.int32))
        images = observations.create_group("images")

        for camera_name, _ in resolved_cameras:
            encoded, encoded_len = _images_encoding(camera_buffers[camera_name])
            images.create_dataset(camera_name, data=encoded, dtype=f"S{encoded_len}")


def create_lerobot_dataset(
    processed_dir: Path,
    repo_id: str,
    task_name: str,
    episodes: list[int],
    *,
    mode: str,
    camera_mode: str,
    secondary_camera: str = DEFAULT_SECONDARY_CAMERA,
) -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    import torch
    import tqdm

    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    camera_names = [camera_name for camera_name, _ in selected_cameras(camera_mode, secondary_camera)]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [motors],
        },
    }
    for camera_name in camera_names:
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    dataset_root = HF_LEROBOT_HOME / repo_id
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type="aloha",
        features=features,
        use_videos=(mode == "video"),
    )

    def _episode_sort_key(path: Path) -> int:
        try:
            return int(path.parent.name.split("_")[-1])
        except Exception:
            return 0

    hdf5_files = sorted(processed_dir.glob("episode_*/episode_*.hdf5"), key=_episode_sort_key)
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        with h5py.File(ep_path, "r") as ep:
            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])
            images_group = ep["/observations/images"]
            imgs_per_cam: dict[str, np.ndarray] = {}
            for camera_name in camera_names:
                imgs_array = []
                for data in images_group[camera_name]:
                    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if decoded is None:
                        raise RuntimeError(f"Failed to decode frame for {camera_name} in {ep_path}")
                    imgs_array.append(decoded)
                imgs_per_cam[camera_name] = np.asarray(imgs_array)

        instruction_path = ep_path.parent / "instructions.json"
        with instruction_path.open("r", encoding="utf-8") as f_instr:
            instruction_dict = json.load(f_instr)
        instruction = np.random.choice(instruction_dict["instructions"])

        for frame_idx in range(state.shape[0]):
            frame = {
                "observation.state": state[frame_idx],
                "action": action[frame_idx],
                "task": instruction,
            }
            for camera_name in camera_names:
                frame[f"observation.images.{camera_name}"] = imgs_per_cam[camera_name][frame_idx]
            dataset.add_frame(frame)
        dataset.save_episode()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Prepare RobotWin data for PI0 open-loop training")
    parser.add_argument("--task-name", default="open_laptop")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id written under HF_LEROBOT_HOME")
    parser.add_argument("--processed-dir", default=None, help="Intermediate processed directory for pi0 conversion")
    parser.add_argument("--description-type", default="seen", choices=["seen", "unseen"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--camera-mode", default=DEFAULT_CAMERA_MODE, choices=["head_only", "dual_view", "tri_view"])
    parser.add_argument(
        "--secondary-camera",
        default=DEFAULT_SECONDARY_CAMERA,
        choices=["left_wrist", "right_wrist"],
        help="Used only when camera-mode=dual_view.",
    )
    parser.add_argument("--mode", default="image", choices=["image", "video"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    prepare_openpi_imports()

    raw_root = ROBOTWIN_ROOT / "data" / args.task_name / args.task_config
    processed_dir = Path(
        args.processed_dir
        or (
            DEFAULT_PROCESSED_DATA_ROOT
            / f"{args.task_name}-{args.task_config}-{args.expert_data_num}-"
              f"{camera_mode_dir_suffix(args.camera_mode, args.secondary_camera)}"
        )
    ).expanduser().resolve()

    if processed_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Processed dir already exists: {processed_dir}. Use --overwrite to rebuild.")
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PI05_RobotWin] raw_root={raw_root}")
    print(f"[PI05_RobotWin] processed_dir={processed_dir}")
    for episode_id in range(args.expert_data_num):
        convert_raw_episode_to_pi0_format(
            raw_root=raw_root,
            episode_id=episode_id,
            output_root=processed_dir,
            description_type=args.description_type,
            camera_mode=args.camera_mode,
            secondary_camera=args.secondary_camera,
        )
        print(f"[PI05_RobotWin] converted episode_{episode_id}")

    create_lerobot_dataset(
        processed_dir=processed_dir,
        repo_id=args.repo_id,
        task_name=args.task_name,
        episodes=list(range(args.expert_data_num)),
        mode=args.mode,
        camera_mode=args.camera_mode,
        secondary_camera=args.secondary_camera,
    )

    print(f"[PI05_RobotWin] finished LeRobot export for repo_id={args.repo_id}")


if __name__ == "__main__":
    main()
