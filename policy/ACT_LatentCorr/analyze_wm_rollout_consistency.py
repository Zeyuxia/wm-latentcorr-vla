#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.ACT.util.fk_sapien import SapienFK

from .deploy_policy import _build_model_from_ckpt_args
from .evac_interface import EvacLatentTeacher
from .utils_latent import load_raw_episode
from .utils_multitask_latent import build_multitask_stage1_dataset, resolve_multitask_specs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_task(task_names: list[str], task_name_arg: str) -> tuple[str, int]:
    if task_name_arg:
        if task_name_arg not in task_names:
            raise ValueError(f"task_name={task_name_arg!r} not found in checkpoint tasks: {task_names}")
        return task_name_arg, task_names.index(task_name_arg)
    if len(task_names) == 1:
        return task_names[0], 0
    return task_names[0], 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Analyze WM rollout latent consistency on normal samples")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    ckpt_args = dict(ckpt["args"])
    task_names = list(ckpt_args.get("multi_task_names") or [ckpt_args["task_name"]])
    task_name, task_idx = _resolve_task(task_names, args.task_name)
    task_specs = resolve_multitask_specs(task_names, SIM_TASK_CONFIGS)
    shared_camera_names = list(task_specs[0].camera_names)
    dataset, _ = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=int(ckpt_args.get("act_chunk_size", ckpt_args["prefix_steps"])),
        prefix_steps=int(ckpt_args["prefix_steps"]),
        future_offset=int(ckpt_args.get("future_offset", ckpt_args["prefix_steps"])),
    )
    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=str(device))
    model = _build_model_from_ckpt_args(ckpt_args, shared_camera_names, str(device), ",".join(task_names))
    first = dataset[0]
    model.initialize_latent_heads(first["image_t"].unsqueeze(0).to(device), teacher)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    fk = SapienFK(str(ckpt_args["urdf_path"]))
    raw_cache: dict[int, dict] = {}

    local_indices = [i for i, (ds_task_idx, _) in enumerate(dataset.samples) if int(ds_task_idx) == int(task_idx)]
    if not local_indices:
        raise RuntimeError(f"No normal dataset entries found for task {task_name}")

    raw_latent_mse = []
    shared_latent_mse = []
    token_mse = []

    with torch.no_grad():
        for _ in range(max(1, int(args.num_samples))):
            item = dataset[int(random.choice(local_indices))]
            ep_id = int(item["episode_id"])
            if ep_id not in raw_cache:
                raw_cache[ep_id] = load_raw_episode(str(item["raw_data_dir"]), ep_id)

            gt_future = teacher.encode_image(item["image_t_future"].unsqueeze(0).to(device)[:, 0])
            rollout_future = teacher.rollout_latent_from_actions(
                curr_image=item["image_t"][0].to(device),
                curr_qpos_raw=item["qpos_raw"].to(device),
                action_prefix_raw=item["action_prefix_raw"].to(device),
                raw_data=raw_cache[ep_id],
                fk=fk,
                ddim_steps=int(ckpt_args.get("ddim_steps", 27)),
            ).to(device)
            raw_latent_mse.append(float(torch.mean((rollout_future - gt_future) ** 2).item()))

            if model.latent_loss_cfg.use_raw_wm_targets:
                gt_shared = gt_future
                rollout_shared = rollout_future
            else:
                gt_shared = model._shared_wm_latent(gt_future)
                rollout_shared = model._shared_wm_latent(rollout_future)
            shared_latent_mse.append(float(torch.mean((rollout_shared - gt_shared) ** 2).item()))

            gt_token = model._latent_to_act_token(gt_shared, scale=1.0)
            rollout_token = model._latent_to_act_token(rollout_shared, scale=1.0)
            token_mse.append(float(torch.mean((rollout_token - gt_token) ** 2).item()))

    results = {
        "ckpt_path": os.path.realpath(args.ckpt_path),
        "task_name": task_name,
        "num_samples": int(len(raw_latent_mse)),
        "raw_latent_mse_mean": float(np.mean(raw_latent_mse)),
        "raw_latent_mse_min": float(np.min(raw_latent_mse)),
        "raw_latent_mse_max": float(np.max(raw_latent_mse)),
        "shared_latent_mse_mean": float(np.mean(shared_latent_mse)),
        "shared_latent_mse_min": float(np.min(shared_latent_mse)),
        "shared_latent_mse_max": float(np.max(shared_latent_mse)),
        "token_mse_mean": float(np.mean(token_mse)),
        "token_mse_min": float(np.min(token_mse)),
        "token_mse_max": float(np.max(token_mse)),
    }
    print(json.dumps(results, indent=2), flush=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
