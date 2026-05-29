#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import torch
from tqdm import tqdm

from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.ACT.util.fk_sapien import SapienFK

from .evac_interface import EvacLatentTeacher
from .utils_latent import load_raw_episode
from .utils_multitask_latent import build_future_latent_cache_relpath, resolve_multitask_specs


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Precompute multitask future teacher latent cache")
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prefix_steps", type=int, default=16)
    parser.add_argument("--future_offset", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=27)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--end_episode", type=int, default=-1)
    parser.add_argument("--heartbeat_path", type=str, default="")
    parser.add_argument("--heartbeat_every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _pad_action_window(actions, start_ts: int, horizon: int):
    out = torch.zeros((horizon, int(actions.shape[1])), dtype=torch.float32)
    if start_ts < int(actions.shape[0]):
        valid = torch.as_tensor(actions[start_ts : start_ts + horizon], dtype=torch.float32)
        out[: valid.shape[0]] = valid
    return out


def _write_heartbeat(path: str, payload: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main() -> None:
    args = build_argparser().parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    cache_dir = os.path.realpath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    heartbeat_path = os.path.realpath(args.heartbeat_path) if args.heartbeat_path else ""

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS)
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
    )
    fk = SapienFK(args.urdf_path)

    total_written = 0
    total_skipped = 0
    total_pending = 0
    total_seen = 0
    started_at = time.time()

    for spec in task_specs:
        ep_start = max(0, int(args.start_episode))
        ep_end = spec.num_episodes if int(args.end_episode) < 0 else min(spec.num_episodes, int(args.end_episode))
        task_written = 0
        task_skipped = 0
        task_pending = 0
        raw_data_cache: dict[int, dict] = {}
        episode_iter = tqdm(range(ep_start, ep_end), desc=f"{spec.task_name}", leave=True)
        for ep_id in episode_iter:
            h5_path = os.path.join(spec.dataset_dir, f"episode_{ep_id}.hdf5")
            if not os.path.exists(h5_path):
                continue
            with h5py.File(h5_path, "r") as root:
                actions = root["/action"][()]
                qpos_seq = root["/observations/qpos"][()]
                images = root[f"/observations/images/{spec.camera_names[0]}"][()]
                max_start = int(actions.shape[0]) - args.future_offset - 1
                if max_start < 0:
                    continue

                pending = []
                for start_ts in range(max_start + 1):
                    total_seen += 1
                    relpath = build_future_latent_cache_relpath(
                        task_name=spec.task_name,
                        episode_id=ep_id,
                        start_ts=start_ts,
                        future_offset=args.future_offset,
                        prefix_steps=args.prefix_steps,
                        ddim_steps=args.ddim_steps,
                    )
                    cache_path = os.path.join(cache_dir, relpath)
                    if os.path.exists(cache_path) and not args.overwrite:
                        total_skipped += 1
                        task_skipped += 1
                        continue
                    pending.append((start_ts, cache_path))
                total_pending += len(pending)
                task_pending += len(pending)
                if not pending:
                    episode_iter.set_postfix(
                        write=task_written,
                        skip=task_skipped,
                        pending=task_pending,
                    )
                    continue

                if ep_id not in raw_data_cache:
                    raw_data_cache[ep_id] = load_raw_episode(spec.raw_data_dir, ep_id)
                raw_data = raw_data_cache[ep_id]

                for chunk_start in range(0, len(pending), args.batch_size):
                    chunk = pending[chunk_start : chunk_start + args.batch_size]
                    curr_images = []
                    curr_qpos_raw = []
                    action_prefix_raw = []
                    for start_ts, _ in chunk:
                        img = torch.from_numpy(images[start_ts]).permute(2, 0, 1).float() / 255.0
                        curr_images.append(img)
                        curr_qpos_raw.append(torch.as_tensor(qpos_seq[start_ts], dtype=torch.float32))
                        action_prefix_raw.append(_pad_action_window(actions, start_ts, args.prefix_steps))

                    with torch.inference_mode():
                        latent_batch = teacher.rollout_latent_from_actions_batch(
                            curr_image=torch.stack(curr_images, dim=0),
                            curr_qpos_raw=torch.stack(curr_qpos_raw, dim=0),
                            action_prefix_raw=torch.stack(action_prefix_raw, dim=0),
                            raw_data=[raw_data] * len(chunk),
                            fk=fk,
                            ddim_steps=args.ddim_steps,
                        ).cpu()

                    for idx, (_, cache_path) in enumerate(chunk):
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        torch.save(latent_batch[idx].half(), cache_path)
                        total_written += 1
                        task_written += 1

                    processed = total_written + total_skipped
                    if args.heartbeat_every > 0 and (processed % args.heartbeat_every == 0):
                        elapsed = max(1e-6, time.time() - started_at)
                        rate = total_written / elapsed
                        _write_heartbeat(
                            heartbeat_path,
                            {
                                "cache_dir": cache_dir,
                                "task_name": spec.task_name,
                                "episode_id": int(ep_id),
                                "elapsed_sec": elapsed,
                                "total_seen": int(total_seen),
                                "total_pending": int(total_pending),
                                "total_written": int(total_written),
                                "total_skipped": int(total_skipped),
                                "write_per_sec": rate,
                                "batch_size": int(args.batch_size),
                                "prefix_steps": int(args.prefix_steps),
                                "future_offset": int(args.future_offset),
                                "ddim_steps": int(args.ddim_steps),
                            },
                        )
                    episode_iter.set_postfix(
                        write=task_written,
                        skip=task_skipped,
                        pending=task_pending,
                    )

        print(
            f"[future_latent_cache] task={spec.task_name} "
            f"written={task_written} skipped={task_skipped} pending={task_pending}",
            flush=True,
        )

    print(
        f"[future_latent_cache] done | cache_dir={cache_dir} "
        f"written={total_written} skipped={total_skipped} pending={total_pending}",
        flush=True,
    )


if __name__ == "__main__":
    main()
