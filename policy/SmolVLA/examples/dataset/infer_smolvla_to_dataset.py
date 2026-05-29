#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SmolVLA inference over a training dataset and save a new LeRobot dataset "
            "with per-step inputs, GT action, and predicted action."
        )
    )
    # Input dataset (training set used for finetuning)
    parser.add_argument("--train_repo_id", type=str, required=True, help="HF dataset repo id of the training set")
    parser.add_argument(
        "--train_root",
        type=str,
        default=None,
        help="Optional local root for the training dataset (defaults to ~/.cache/huggingface/lerobot/<repo>)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help=(
            "Optional episode range or list for evaluation. Examples: '0:10' for first 10 episodes, "
            "'[0,2,5]' for explicit list. Defaults to all episodes."
        ),
    )

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "SmolVLA model path or hub repo id. E.g. 'lerobot/smolvla_base' or a local checkpoints directory."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference: 'cuda', 'cpu', or 'mps'. Defaults to auto-detect.",
    )

    # Output dataset
    parser.add_argument("--out_repo_id", type=str, required=True, help="Repo id for the new dataset")
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Optional local root for the new dataset (defaults to ~/.cache/huggingface/lerobot/<repo>)",
    )
    parser.add_argument(
        "--use_videos",
        action="store_true",
        help="Store visual inputs as mp4 videos (faster/lighter than png images)",
    )
    parser.add_argument(
        "--chunks_size",
        type=int,
        default=None,
        help="Max files per chunk directory for the output dataset (optional)",
    )
    parser.add_argument(
        "--data_file_mb",
        type=int,
        default=None,
        help="Max parquet file size in MB for the output dataset (optional)",
    )
    parser.add_argument(
        "--video_file_mb",
        type=int,
        default=None,
        help="Max video file size in MB for the output dataset (optional)",
    )

    parser.add_argument(
        "--rename_map",
        type=str,
        default=None,
        help=(
            "Optional JSON mapping to rename observation keys to match model expectations. "
            "Example: '{\"observation.images.left\": \"observation.images.camera1\"}'"
        ),
    )

    parser.add_argument(
        "--batch_encoding_size",
        type=int,
        default=1,
        help=(
            "If >1, defer video encoding and batch multiple episodes together to speed up ffmpeg work."
        ),
    )

    return parser.parse_args()


def parse_episodes_arg(episodes_arg: str | None) -> list[int] | None:
    if not episodes_arg:
        return None
    episodes_arg = episodes_arg.strip()
    if episodes_arg.startswith("["):
        return json.loads(episodes_arg)
    if ":" in episodes_arg:
        start, end = episodes_arg.split(":", 1)
        s = int(start) if start else 0
        e = int(end) if end else s
        return list(range(s, e))
    # single integer
    return [int(episodes_arg)]


def filter_input_output_features(ds_features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Keep observation.* and action. Drop unrelated keys (like rewards, next.*).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for k, ft in ds_features.items():
        if k.startswith(f"{OBS_STR}.") or k == ACTION:
            out[k] = ft
    return out


def add_pred_action_feature(features: Dict[str, Dict[str, Any]], pred_key: str = "action_pred") -> Dict[str, Dict[str, Any]]:
    if ACTION not in features:
        raise ValueError("Input dataset features do not contain 'action' vector.")
    act_ft = features[ACTION]
    pred_ft = {
        "dtype": act_ft["dtype"],
        "shape": act_ft["shape"],
        "names": list(act_ft.get("names", [])),
    }
    if pred_key in features:
        raise ValueError(f"predicted action key '{pred_key}' already exists in features")
    features[pred_key] = pred_ft
    return features


def move_to_device(d: dict, device: torch.device) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main() -> None:
    args = parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    # Load training dataset (read-only)
    episodes = parse_episodes_arg(args.episodes)
    train_ds = LeRobotDataset(
        repo_id=args.train_repo_id,
        root=args.train_root,
        episodes=episodes,
        download_videos=True,
    )

    # Build policy from pretrained with dataset metadata (shapes/stats)
    cfg = make_policy_config(
        "smolvla",
        pretrained_path=args.model_path,
        device=str(device),
    )
    policy = make_policy(cfg, ds_meta=train_ds.meta)

    # Pre/Post processors (use saved processors if present; otherwise construct)
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=policy.config.pretrained_path, dataset_stats=train_ds.meta.stats
    )

    # Optional rename map for observations
    rename_map = json.loads(args.rename_map) if args.rename_map else {}
    if rename_map:
        # When a rename map is provided, we accept that visual feature names differ from config
        pass  # The RenameObservationsProcessor is already at the head of smolvla preprocess with empty map.
        # Advanced: You could re-instantiate preprocess with overrides, but kept simple to avoid breaking saved cfgs.

    # Prepare output dataset features: copy inputs + GT action and add predicted action
    base_features = filter_input_output_features(train_ds.meta.features)
    out_features = add_pred_action_feature(dict(base_features))

    # Create destination dataset
    out_ds = LeRobotDataset.create(
        repo_id=args.out_repo_id,
        fps=train_ds.meta.fps,
        features=out_features,
        root=args.out_root,
        use_videos=bool(args.use_videos),
        batch_encoding_size=args.batch_encoding_size,
    )

    # Inherit chunk/file size preferences if provided
    out_ds.meta.update_chunk_settings(
        chunks_size=args.chunks_size,
        data_files_size_in_mb=args.data_file_mb,
        video_files_size_in_mb=args.video_file_mb,
    )

    breakpoint()

    # Iterate frames and save
    current_ep = None
    num_written = 0

    try:

        for idx in range(len(train_ds)):
            item = train_ds[idx]
            ep_idx = int(item["episode_index"].item())

            # If new episode, save the previous one
            if current_ep is None:
                current_ep = ep_idx
            elif ep_idx != current_ep:
                out_ds.save_episode()
                current_ep = ep_idx

            # Build per-step model input: keep observation.* keys + task
            obs_for_model: dict[str, Any] = {k: item[k] for k in item.keys() if k.startswith(f"{OBS_STR}.")}
            obs_for_model["task"] = item.get("task", "")

            # Run preprocess -> policy -> postprocess
            model_in = preprocess(move_to_device(obs_for_model, device))
            with torch.inference_mode():
                action_pred = policy.select_action(model_in)
            action_pred = postprocess(action_pred)

            # Convert action_pred to tensor if wrapped
            if isinstance(action_pred, dict) and ACTION in action_pred:
                pred_tensor = action_pred[ACTION]
            else:
                pred_tensor = action_pred

            # Prepare frame for output dataset
            frame: dict[str, Any] = {"task": obs_for_model.get("task", "")}

            # Copy observation inputs ensuring visual shapes match features (H, W, C)
            for k, ft in out_features.items():
                if not k.startswith(f"{OBS_STR}."):
                    continue
                if k not in item:
                    continue
                val = item[k]
                if ft.get("dtype") in ["image", "video"]:
                    # Convert torch CHW tensor -> numpy HWC strictly to satisfy validator
                    if isinstance(val, torch.Tensor):
                        arr = val.detach().cpu().numpy()
                    else:
                        arr = val
                    if hasattr(arr, "shape") and len(arr.shape) == 3:
                        # If channel-first (C,H,W), convert to channel-last (H,W,C)
                        if arr.shape[0] in (1, 3):
                            arr = arr.transpose(1, 2, 0)
                    frame[k] = arr
                else:
                    frame[k] = val

            # Ground-truth action
            gt = item[ACTION]
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().numpy()
            frame[ACTION] = gt

            # Predicted action (1D np array)
            if isinstance(pred_tensor, torch.Tensor):
                # shape may be (1, D)
                pred_np = pred_tensor.detach().cpu().squeeze(0).numpy()
            else:
                pred_np = pred_tensor  # assume numpy-like
            frame["action_pred"] = pred_np
            

            out_ds.add_frame(frame)
            num_written += 1

        # Save last episode
        if out_ds.episode_buffer and out_ds.episode_buffer.get("size", 0) > 0:
            out_ds.save_episode()

    finally:
        out_ds.finalize()

    print(
        f"Saved {num_written} frames into new dataset at {out_ds.root} (repo_id={args.out_repo_id})."
    )


if __name__ == "__main__":
    main()