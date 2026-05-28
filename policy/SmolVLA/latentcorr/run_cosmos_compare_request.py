#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy.SmolVLA.latentcorr.cosmos_rollout_worker import CosmosActionConditionedRunner


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run a saved Cosmos compare request.")
    parser.add_argument("--request_npz", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--cosmos_root", default="/data/zhenyangfan/cosmos-predict2.5")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--experiment", default="robotwin_dualarm_actioncond_2b_256_320")
    parser.add_argument(
        "--config_file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=12)
    parser.add_argument("--guidance", type=int, default=7)
    parser.add_argument("--resolution", default="256,320")
    parser.add_argument("--fps_downsample_ratio", type=int, default=1)
    parser.add_argument("--gripper_scale", type=float, default=1.0)
    parser.add_argument("--invert_gripper", dest="invert_gripper", action="store_true", default=True)
    parser.add_argument("--no-invert_gripper", dest="invert_gripper", action="store_false")
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--save_fps", type=int, default=30)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=1)
    parser.add_argument("--action_scaler", type=float, default=20.0)
    parser.add_argument("--action_stats_path", default="")
    parser.add_argument("--action_normalization_clip", default="")
    parser.add_argument("--use_quat", dest="use_quat", action="store_true", default=False)
    parser.add_argument("--no-use_quat", dest="use_quat", action="store_false")
    parser.add_argument("--quat_input_order", choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("HF_HOME", "/data/zhenyangfan/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    payload = np.load(str(args.request_npz))
    curr_image = np.asarray(payload["curr_image"], dtype=np.float32)
    fk_poses = np.asarray(payload["fk_poses"], dtype=np.float32)
    grippers = np.asarray(payload["grippers"], dtype=np.float32)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    runner = CosmosActionConditionedRunner(args)
    try:
        result = runner._run_impl(
            curr_image=curr_image,
            fk_poses=fk_poses,
            grippers=grippers,
            save_dir=str(args.save_dir),
        )
        out_img = result["out_img"]
        np.save(str(Path(args.save_dir) / "output_last_frame.npy"), out_img.astype(np.float32))
    finally:
        close_fn = getattr(runner, "close", None)
        if callable(close_fn):
            close_fn()

    video_path = Path(args.save_dir) / "outputs.mp4"
    print(f"Saved Cosmos compare video: {video_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
