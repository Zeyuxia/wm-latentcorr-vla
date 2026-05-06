from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .prepare_robotwin_data import convert_raw_episode_to_pi0_format, create_lerobot_dataset
from .robotwin_common import (
    DEFAULT_CAMERA_MODE,
    DEFAULT_PROCESSED_DATA_ROOT,
    DEFAULT_SECONDARY_CAMERA,
    ROBOTWIN_ROOT,
    camera_mode_dir_suffix,
    prepare_openpi_imports,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Prepare combined RobotWin multitask data for PI0.5 base open-loop training")
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=["pick_dual_bottles", "open_laptop", "place_burger_fries", "put_bottles_dustbin", "handover_block"],
    )
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--processed-dir", default=None)
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

    processed_dir = Path(
        args.processed_dir
        or (
            DEFAULT_PROCESSED_DATA_ROOT
            / f"multitask5-{args.task_config}-{args.expert_data_num}-"
              f"{camera_mode_dir_suffix(args.camera_mode, args.secondary_camera)}"
        )
    ).expanduser().resolve()
    if processed_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Processed dir already exists: {processed_dir}. Use --overwrite to rebuild.")
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    global_episode_id = 0
    for task_name in args.task_names:
        raw_root = ROBOTWIN_ROOT / "data" / task_name / args.task_config
        if not (raw_root / "data").is_dir():
            raise FileNotFoundError(f"Raw data not found for {task_name}: {raw_root / 'data'}")
        print(f"[PI05_RobotWin] converting task={task_name} raw_root={raw_root}")
        for episode_id in range(int(args.expert_data_num)):
            convert_raw_episode_to_pi0_format(
                raw_root=raw_root,
                episode_id=episode_id,
                output_root=processed_dir,
                output_episode_id=global_episode_id,
                description_type=args.description_type,
                camera_mode=args.camera_mode,
                secondary_camera=args.secondary_camera,
            )
            global_episode_id += 1

    create_lerobot_dataset(
        processed_dir=processed_dir,
        repo_id=args.repo_id,
        task_name=",".join(args.task_names),
        episodes=list(range(global_episode_id)),
        mode=args.mode,
        camera_mode=args.camera_mode,
        secondary_camera=args.secondary_camera,
    )
    print(
        f"[PI05_RobotWin] finished multitask LeRobot export repo_id={args.repo_id} "
        f"episodes={global_episode_id} processed_dir={processed_dir}"
    )


if __name__ == "__main__":
    main()
