from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SMOLVLA_ROOT = THIS_DIR.parent
SMOLVLA_SRC_DIR = SMOLVLA_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.SmolVLA.latentcorr.act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from policy.SmolVLA.latentcorr.evac_interface import EvacLatentTeacher
from policy.SmolVLA.latentcorr.correction_policy_adapter import SampleBoundSmolVLAAdapter
from policy.SmolVLA.latentcorr.latent_dataset_utils import load_raw_episode
from policy.SmolVLA.latentcorr.multitask_latent_utils import resolve_multitask_specs
from policy.SmolVLA.latentcorr.smolvla_data_utils import make_smolvla_processors
from policy.SmolVLA.latentcorr.smolvla_latent_policy import SmolVLALatentBridgeConfig, SmolVLALatentPolicy
from policy.SmolVLA.latentcorr.stage2_failure_dataset import build_failure_table_dataset
from policy.SmolVLA.latentcorr.latent_config import DynamicsWarmupConfig


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_accelerator() -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(kwargs_handlers=[ddp_kwargs])


def parse_task_parts(task_name: str) -> tuple[str, str]:
    if not task_name.startswith("sim-"):
        raise ValueError(f"Expected sim task name, got {task_name}")
    parts = task_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected multitask name format: {task_name}")
    return parts[1], parts[2]


def get_explore_num_units(dataset) -> int:
    if not hasattr(dataset, "_explore_units"):
        raise ValueError("Explore dataset is missing _explore_units")
    return int(len(dataset._explore_units))


def get_current_explore_unit_idx(dataset) -> int:
    if not hasattr(dataset, "_explore_curr_unit_idx"):
        raise ValueError("Explore dataset is missing _explore_curr_unit_idx")
    return int(dataset._explore_curr_unit_idx)


def get_current_explore_trial_count(dataset) -> int:
    if not hasattr(dataset, "_explore_trial_count_local"):
        raise ValueError("Explore dataset is missing _explore_trial_count_local")
    return int(dataset._explore_trial_count_local)


def get_explore_completed_unit_count(dataset) -> int:
    if not hasattr(dataset, "_explore_completed_unit_count"):
        raise ValueError("Explore dataset is missing _explore_completed_unit_count")
    return int(dataset._explore_completed_unit_count)


def get_explore_completed_sample_count(dataset, k_per_unit: int) -> int:
    completed_units = get_explore_completed_unit_count(dataset)
    current_trials = get_current_explore_trial_count(dataset)
    return int(completed_units) * int(k_per_unit) + int(current_trials)


def sync_rank_sample_totals(local_total_samples: int, device: torch.device) -> list[int]:
    total_t = torch.tensor([int(local_total_samples)], device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized():
        gathered = [torch.zeros_like(total_t) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, total_t)
        return [int(item.item()) for item in gathered]
    return [int(local_total_samples)]


def sync_all_explore_status(
    local_current_samples: int,
    local_total_samples: int,
    local_done: bool,
    local_completed_units: int,
    local_total_units: int,
    local_unit_idx: int,
    local_trial_count: int,
    device: torch.device,
) -> list[dict]:
    status_t = torch.tensor(
        [
            int(local_current_samples),
            int(local_total_samples),
            1 if bool(local_done) else 0,
            int(local_completed_units),
            int(local_total_units),
            int(local_unit_idx),
            int(local_trial_count),
        ],
        device=device,
        dtype=torch.int64,
    )
    if dist.is_available() and dist.is_initialized():
        gathered = [torch.zeros_like(status_t) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, status_t)
        progress = []
        for ridx, item in enumerate(gathered):
            vals = [int(x) for x in item.tolist()]
            progress.append(
                {
                    "rank": int(ridx),
                    "current_samples": int(vals[0]),
                    "total_samples": int(vals[1]),
                    "done": bool(vals[2]),
                    "completed_units": int(vals[3]),
                    "total_units": int(vals[4]),
                    "unit_idx": int(vals[5]),
                    "trial_count": int(vals[6]),
                }
            )
        return progress
    return [
        {
            "rank": 0,
            "current_samples": int(local_current_samples),
            "total_samples": int(local_total_samples),
            "done": bool(local_done),
            "completed_units": int(local_completed_units),
            "total_units": int(local_total_units),
            "unit_idx": int(local_unit_idx),
            "trial_count": int(local_trial_count),
        }
    ]


def format_rank_progress(progress: list[dict]) -> str:
    parts = []
    for item in progress:
        total_i = max(0, int(item["total_units"]))
        comp_i = max(0, int(item["completed_units"]))
        unit_i = max(0, int(item["unit_idx"])) + 1 if total_i > 0 else 0
        trial_i = max(0, int(item["trial_count"]))
        wait_tag = "w" if bool(item.get("done", False)) else ""
        parts.append(
            f"r{int(item['rank'])}:{comp_i}/{total_i} u{unit_i} t{trial_i}{wait_tag}"
        )
    return " | ".join(parts)


def write_failure_live_files(task_output_dir: Path, args, epoch: int, rank: int, failure_trials: list[dict]) -> None:
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = task_output_dir / f"failure_trials_live_rank{int(rank):02d}.json"
    with open(trials_path, "w", encoding="utf-8") as f:
        json.dump(failure_trials, f, indent=2, ensure_ascii=False)
    meta = {
        "version": 1,
        "mode": "explore_meta_live",
        "epoch": int(epoch),
        "failure_phase_bins": int(args.failure_phase_bins),
        "failure_translation_dir_bins": int(args.failure_translation_dir_bins),
        "failure_translation_mag_bins": int(args.failure_translation_mag_bins),
        "failure_rotation_dir_bins": int(args.failure_rotation_dir_bins),
        "failure_rotation_mag_bins": int(args.failure_rotation_mag_bins),
        "failure_explore_k": int(args.failure_explore_k),
        "rank": int(rank),
        "world_size": int(args.world_size),
    }
    with open(task_output_dir / f"failure_meta_rank{int(rank):02d}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def sanitize_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or "unknown"


def shard_episode_ids(episode_ids: list[int], rank: int, world_size: int) -> list[int]:
    if world_size <= 1:
        return [int(ep) for ep in episode_ids]
    return [int(ep) for idx, ep in enumerate(episode_ids) if idx % int(world_size) == int(rank)]


def shuffle_episode_ids_for_rank(episode_ids: list[int], seed: int, rank: int, task_name: str) -> list[int]:
    ids = [int(ep) for ep in episode_ids]
    if len(ids) <= 1:
        return ids
    task_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(task_name)))
    rng = np.random.default_rng(int(seed) + 1009 * int(rank) + 9176 * int(task_hash))
    perm = rng.permutation(len(ids))
    return [ids[int(idx)] for idx in perm]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("explore smolvla latentcorr")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smolvla_pretrained_path", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--instruction_type", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--curobo_left_yml", type=str, required=True)
    parser.add_argument("--curobo_right_yml", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--future_offset", type=int, required=True)
    parser.add_argument("--prefix_steps", type=int, required=True)
    parser.add_argument("--action_dim", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--adapter_hidden_dim", type=int, required=True)
    parser.add_argument("--predictor_hidden_dim", type=int, required=True)
    parser.add_argument("--act_chunk_size", type=int, required=True)
    parser.add_argument("--sample_phase_window_len", type=int, required=True)
    parser.add_argument("--start_margin", type=int, required=True)
    parser.add_argument("--failure_phase_bins", type=int, required=True)
    parser.add_argument("--failure_translation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_translation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_explore_k", type=int, required=True)
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, required=True)
    parser.add_argument("--planner_orient_weight", type=float, required=True)
    parser.add_argument("--planner_gripper_penalty", type=float, required=True)
    parser.add_argument("--planner_nearest_window_radius", type=int, required=True)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, required=True)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_save_video", type=str2bool, required=True)
    parser.add_argument("--save_perturb_rollout_video", type=str2bool, required=True)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, required=True)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, required=True)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, required=True)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, required=True)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, required=True)
    parser.add_argument("--evac_blur_filter_enable", type=str2bool, default=False)
    parser.add_argument("--evac_blur_filter_min_ratio", type=float, default=0.25)
    parser.add_argument("--evac_blur_filter_patch_pad_px", type=int, default=24)
    parser.add_argument("--fail_fast_on_error", type=str2bool, default=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    accelerator = build_accelerator()
    if args.rank is None:
        args.rank = int(accelerator.process_index)
    if args.world_size is None:
        args.world_size = int(accelerator.num_processes)
    if str(args.device).startswith("cuda"):
        args.device = str(accelerator.device)
    device_obj = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    evac_cfg_for_dataset = OmegaConf.load(args.evac_config)
    evac_sample_size = tuple(int(x) for x in evac_cfg_for_dataset.data.params.train.params.sample_size)
    interrupted = {"flag": False}

    def _handle_signal(signum, frame):
        interrupted["flag"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    base_policy = SmolVLAPolicy.from_pretrained(args.smolvla_pretrained_path)
    base_policy.to(device_obj)
    preprocess, postprocess = make_smolvla_processors(base_policy, args.smolvla_pretrained_path)

    bridge_cfg = SmolVLALatentBridgeConfig(
        action_dim=int(args.action_dim),
        prefix_steps=int(args.prefix_steps),
        latent_dim=int(args.latent_dim),
        adapter_hidden_dim=int(args.adapter_hidden_dim),
        predictor_hidden_dim=int(args.predictor_hidden_dim),
    )
    warmup_cfg = DynamicsWarmupConfig(
        zero_steps=0,
        ramp_steps=0,
        max_weight=0.0,
        curve="linear",
    )
    model = SmolVLALatentPolicy(base_policy=base_policy, bridge_cfg=bridge_cfg, warmup_cfg=warmup_cfg)
    init_spec = task_specs[0]
    init_dataset, _ = build_failure_table_dataset(
        dataset_dir=init_spec.dataset_dir,
        num_episodes=int(init_spec.num_episodes),
        camera_names=list(init_spec.camera_names),
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
        sample_phase_window_len=int(args.sample_phase_window_len),
        start_margin=int(args.start_margin),
        failure_mode="explore",
        failure_table_path="",
        failure_phase_bins=int(args.failure_phase_bins),
        failure_translation_dir_bins=int(args.failure_translation_dir_bins),
        failure_translation_mag_bins=int(args.failure_translation_mag_bins),
        failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
        failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
        failure_explore_k=int(args.failure_explore_k),
        raw_data_dir=init_spec.raw_data_dir,
        perturb_eef_fail_gain=float(args.act_aligned_perturb_eef_fail_gain),
        perturb_rot_max_deg=float(args.act_aligned_perturb_rot_max_deg),
        evac_sample_size=evac_sample_size,
    )
    init_sample = init_dataset[0]
    init_task_only, init_task_config = parse_task_parts(init_spec.task_name)
    init_batch = SampleBoundSmolVLAAdapter(
        latent_policy=model,
        preprocess=preprocess,
        postprocess=postprocess,
        task_name=init_task_only,
        task_config=init_task_config,
        episode_id=int(init_sample["episode_id"]),
        instruction_type=args.instruction_type,
    ).build_batch(
        image_t=init_sample["image_t"],
        qpos_raw=init_sample["qpos_raw"],
        action_chunk_raw=init_sample["act_action_chunk_raw"],
    )
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
        load_rollout_model=True,
    )
    init_teacher_latent = teacher.encode_image(
        init_sample["image_t"][0].unsqueeze(0).to(device=device_obj, dtype=torch.float32)
    )
    model.initialize_from_batch(init_batch, teacher_latent=init_teacher_latent)
    if str(args.stage1_ckpt).strip():
        stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        model.load_state_dict(stage1_ckpt["model"], strict=True)
    model.to(device_obj)
    model.eval()
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=args.device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )

    if accelerator.is_main_process:
        with open(output_dir / "explore_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    world_size = int(max(1, int(args.world_size)))
    failure_explore_k_global = int(max(1, int(args.failure_explore_k)))
    failure_explore_k_local = int(max(1, (failure_explore_k_global + world_size - 1) // world_size))

    task_run_stats: dict[str, dict] = {}
    task_order: list[str] = []
    total_rank_samples = 0
    for spec in task_specs:
        dataset, _ = build_failure_table_dataset(
            dataset_dir=spec.dataset_dir,
            num_episodes=int(spec.num_episodes),
            camera_names=list(spec.camera_names),
            act_chunk_size=int(args.act_chunk_size),
            prefix_steps=int(args.prefix_steps),
            future_offset=int(args.future_offset),
            sample_phase_window_len=int(args.sample_phase_window_len),
            start_margin=int(args.start_margin),
            failure_mode="explore",
            failure_table_path="",
            failure_phase_bins=int(args.failure_phase_bins),
            failure_translation_dir_bins=int(args.failure_translation_dir_bins),
            failure_translation_mag_bins=int(args.failure_translation_mag_bins),
            failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
            failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
            failure_explore_k=int(args.failure_explore_k),
            raw_data_dir=spec.raw_data_dir,
            perturb_eef_fail_gain=float(args.act_aligned_perturb_eef_fail_gain),
            perturb_rot_max_deg=float(args.act_aligned_perturb_rot_max_deg),
            evac_sample_size=evac_sample_size,
        )
        full_episode_ids = [int(ep) for ep in dataset.episode_ids]
        local_episode_ids = shuffle_episode_ids_for_rank(
            full_episode_ids, seed=int(args.seed), rank=int(args.rank), task_name=spec.task_name
        )
        dataset.set_episode_ids(local_episode_ids)
        total_units = get_explore_num_units(dataset)
        task_only, _ = parse_task_parts(spec.task_name)
        task_order.append(task_only)
        task_samples = int(total_units) * int(failure_explore_k_local)
        total_rank_samples += task_samples
        task_run_stats[task_only] = {
            "num_explore_units_local": int(total_units),
            "num_explore_units_global": int(len(dataset._collect_local_explore_phase_units())),
            "num_episode_ids_local": int(len(local_episode_ids)),
            "num_episode_ids_global": int(len(full_episode_ids)),
            "total_samples_local": int(task_samples),
        }

    rank_totals = sync_rank_sample_totals(int(total_rank_samples), device_obj)
    overall_progress = None
    if accelerator.is_main_process:
        overall_progress = tqdm(
            total=max(1, int(sum(rank_totals))),
            desc="Explore Samples",
            dynamic_ncols=True,
            leave=True,
        )

    explore_manifest = {
        "version": 1,
        "output_dir": str(output_dir),
        "multi_task_names": list(args.multi_task_names),
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "total_samples_local": int(total_rank_samples),
        "task_outputs": {},
    }

    for spec in task_specs:
        task_only, _ = parse_task_parts(spec.task_name)
        _, task_config = parse_task_parts(spec.task_name)
        task_output_dir = output_dir / "failure_explore" / task_only
        raw_cache: dict[int, dict] = {}
        dataset, norm_stats = build_failure_table_dataset(
            dataset_dir=spec.dataset_dir,
            num_episodes=int(spec.num_episodes),
            camera_names=list(spec.camera_names),
            act_chunk_size=int(args.act_chunk_size),
            prefix_steps=int(args.prefix_steps),
            future_offset=int(args.future_offset),
            sample_phase_window_len=int(args.sample_phase_window_len),
            start_margin=int(args.start_margin),
            failure_mode="explore",
            failure_table_path="",
            failure_phase_bins=int(args.failure_phase_bins),
            failure_translation_dir_bins=int(args.failure_translation_dir_bins),
            failure_translation_mag_bins=int(args.failure_translation_mag_bins),
            failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
            failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
            failure_explore_k=int(args.failure_explore_k),
            raw_data_dir=spec.raw_data_dir,
            perturb_eef_fail_gain=float(args.act_aligned_perturb_eef_fail_gain),
            perturb_rot_max_deg=float(args.act_aligned_perturb_rot_max_deg),
            evac_sample_size=evac_sample_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        full_episode_ids = [int(ep) for ep in dataset.episode_ids]
        local_episode_ids = shuffle_episode_ids_for_rank(
            full_episode_ids, seed=int(args.seed), rank=int(args.rank), task_name=spec.task_name
        )
        dataset.set_episode_ids(local_episode_ids)
        dataset.set_explore_local_k(int(failure_explore_k_local))
        full_units_global = [dict(item) for item in dataset._collect_local_explore_phase_units()]
        total_units = int(task_run_stats[task_only]["num_explore_units_local"])
        task_total_samples = int(total_units) * int(failure_explore_k_local)
        failure_trials: list[dict] = []
        epoch = 0
        processed_samples = 0
        local_task_done = bool(total_units <= 0)

        if total_units == 0:
            write_failure_live_files(task_output_dir, args, epoch, int(args.rank), failure_trials)
            explore_manifest["task_outputs"][task_only] = {
                "task_name": spec.task_name,
                "task_output_dir": str(task_output_dir),
                "num_failure_trials": 0,
                "num_explore_units_local": 0,
                "num_explore_units_global": int(task_run_stats[task_only]["num_explore_units_global"]),
                "num_episode_ids_local": int(task_run_stats[task_only]["num_episode_ids_local"]),
                "num_episode_ids_global": int(task_run_stats[task_only]["num_episode_ids_global"]),
                "total_samples_local": 0,
            }
            continue

        all_task_done = False
        while not all_task_done:
            if local_task_done:
                task_prefix_samples = 0
                for prev_task in task_order:
                    if prev_task == task_only:
                        break
                    task_prefix_samples += int(task_run_stats[prev_task]["total_samples_local"])
                local_current_samples = int(task_prefix_samples) + int(task_total_samples)
                explore_progress = sync_all_explore_status(
                    local_current_samples=int(local_current_samples),
                    local_total_samples=int(total_rank_samples),
                    local_done=True,
                    local_completed_units=int(total_units),
                    local_total_units=int(total_units),
                    local_unit_idx=int(max(0, total_units - 1)),
                    local_trial_count=int(failure_explore_k_local),
                    device=device_obj,
                )
                all_task_done = bool(len(explore_progress) > 0 and all(bool(item["done"]) for item in explore_progress))
                if overall_progress is not None:
                    target_done = int(sum(int(item["current_samples"]) for item in explore_progress))
                    if target_done > int(overall_progress.n):
                        overall_progress.update(target_done - int(overall_progress.n))
                    overall_progress.set_postfix_str(
                        f"task={task_only} {format_rank_progress(explore_progress)}"
                    )
                if all_task_done:
                    break
                accelerator.wait_for_everyone()
                continue
            for batch in dataloader:
                if interrupted["flag"]:
                    raise KeyboardInterrupt
                processed_samples += 1
                episode_id = int(batch["episode_id"][0].item())
                start_ts = int(batch["start_ts"][0].item())
                adapter = SampleBoundSmolVLAAdapter(
                    latent_policy=model,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    task_name=task_only,
                    task_config=task_config,
                    episode_id=episode_id,
                    instruction_type=args.instruction_type,
                )
                if episode_id not in raw_cache:
                    raw_cache[episode_id] = load_raw_episode(spec.raw_data_dir, episode_id)
                sample_debug_dir = None
                if bool(args.save_perturb_rollout_video) or bool(args.recover_eval_save_video):
                    phase_bin_id = int(batch["sampled_phase_bin_id"][0].item())
                    sample_debug_dir = (
                        task_output_dir
                        / "debug_wm"
                        / f"rank{int(args.rank):02d}"
                        / f"step_{processed_samples - 1:06d}"
                        / f"bi{phase_bin_id}"
                    )
                correction = correction_builder.build(
                    latent_model=adapter,
                    image_t=batch["image_t"][0].to(args.device),
                    qpos_t=batch["qpos_t"][0].to(args.device),
                    raw_data=raw_cache[episode_id],
                    norm_stats=norm_stats,
                    start_ts=start_ts,
                    failure_mode_override="explore",
                    sampled_phase_id=int(batch["sampled_phase_id"][0].item()),
                    sampled_phase_bin_id=int(batch["sampled_phase_bin_id"][0].item()),
                    sampled_phase_instance_id=int(batch["sampled_phase_instance_id"][0].item()),
                    forced_error_mode_id=int(batch["forced_error_mode_id"][0].item()),
                    sampled_active_arm_pattern_id=int(batch["sampled_active_arm_pattern_id"][0].item()),
                    forced_dir_bin_id=int(batch["forced_dir_bin_id"][0].item()),
                    forced_mag_bin_id=int(batch["forced_mag_bin_id"][0].item()),
                    sampled_mode_prob=float(batch["sampled_mode_prob"][0].item()),
                    sampled_entry_prob_within_mode=float(batch["sampled_entry_prob_within_mode"][0].item()),
                    sampled_unit_prob=float(batch["sampled_unit_prob"][0].item()),
                    debug_dir=(None if sample_debug_dir is None else str(sample_debug_dir)),
                    precomputed_action_chunk_norm=batch["act_action_chunk"][0],
                )
                corr_meta = None
                if isinstance(correction, dict):
                    corr_meta = correction.get("corr_meta")
                if corr_meta is None:
                    corr_meta = correction_builder.pop_last_skip_meta()
                if isinstance(corr_meta, dict):
                    recover_eval_last = corr_meta.get("recover_eval_last", {}) or {}
                    recoverable_raw = recover_eval_last.get("recoverable", None)
                    error_mode_key = str(corr_meta.get("sampled_error_mode", "")).strip().lower()
                    invalid_trial = bool(corr_meta.get("invalid_trial", False))
                    invalid_reason = str(corr_meta.get("invalid_reason", "")).strip()
                    if (
                        (recoverable_raw is not None or invalid_trial)
                        and error_mode_key in {"translation", "rotation", "gripper_close"}
                    ):
                        phase_key = str(corr_meta.get("sampled_phase_key", "")).strip().lower()
                        active_arm_pattern = str(corr_meta.get("sampled_active_arm_pattern", "both")).strip().lower()
                        if not phase_key:
                            raise ValueError("corr_meta missing sampled_phase_key during explore")
                        failure_trials.append(
                            {
                                "epoch": int(epoch),
                                "global_step": int(len(failure_trials)),
                                "episode_id": episode_id,
                                "start_ts": start_ts,
                                "phase_key": phase_key,
                                "phase_instance_idx": int(batch["sampled_phase_instance_id"][0].item()),
                                "phase_bin_id": int(batch["sampled_phase_bin_id"][0].item()),
                                "error_mode": error_mode_key,
                                "active_arm_pattern": active_arm_pattern,
                                "dir_bin_id": int(batch["forced_dir_bin_id"][0].item()),
                                "mag_bin_id": int(batch["forced_mag_bin_id"][0].item()),
                                "recoverable": bool(recoverable_raw) if recoverable_raw is not None else False,
                                "invalid_trial": bool(invalid_trial),
                                "invalid_reason": invalid_reason if invalid_trial else None,
                                "recover_eval_mode": recover_eval_last.get("mode"),
                                "recover_eval_metric_name": recover_eval_last.get("metric_name"),
                                "recover_eval_metric": recover_eval_last.get("metric"),
                                "recover_eval_threshold": recover_eval_last.get("threshold"),
                                "evac_blur_filter": corr_meta.get("evac_blur_filter"),
                            }
                        )

                dataset.record_explore_trial(
                    int(batch["sampled_explore_unit_idx"][0].item()),
                    episode_id,
                    start_ts,
                )
                completed_units = get_explore_completed_unit_count(dataset)
                task_sample_progress = get_explore_completed_sample_count(dataset, int(failure_explore_k_local))
                task_prefix_samples = 0
                for prev_task in task_order:
                    if prev_task == task_only:
                        break
                    task_prefix_samples += int(task_run_stats[prev_task]["total_samples_local"])
                overall_current_samples_local = int(task_prefix_samples) + int(task_sample_progress)
                explore_progress = sync_all_explore_status(
                    local_current_samples=int(overall_current_samples_local),
                    local_total_samples=int(total_rank_samples),
                    local_done=bool(completed_units >= total_units),
                    local_completed_units=int(completed_units),
                    local_total_units=int(total_units),
                    local_unit_idx=int(get_current_explore_unit_idx(dataset)),
                    local_trial_count=int(get_current_explore_trial_count(dataset)),
                    device=device_obj,
                )
                if overall_progress is not None:
                    target_done = int(sum(int(item["current_samples"]) for item in explore_progress))
                    if target_done > int(overall_progress.n):
                        overall_progress.update(target_done - int(overall_progress.n))
                    overall_progress.set_postfix_str(
                        f"task={task_only} {format_rank_progress(explore_progress)}"
                    )
                write_failure_live_files(task_output_dir, args, epoch, int(args.rank), failure_trials)
                if completed_units >= total_units:
                    local_task_done = True
                    break
                all_task_done = bool(len(explore_progress) > 0 and all(bool(item["done"]) for item in explore_progress))
                if all_task_done:
                    local_task_done = True
                    break
            epoch += 1
        explore_manifest["task_outputs"][task_only] = {
            "task_name": spec.task_name,
            "task_output_dir": str(task_output_dir),
            "num_failure_trials": len(failure_trials),
            "num_explore_units_local": int(total_units),
            "num_explore_units_global": int(task_run_stats[task_only]["num_explore_units_global"]),
            "num_episode_ids_local": int(task_run_stats[task_only]["num_episode_ids_local"]),
            "num_episode_ids_global": int(task_run_stats[task_only]["num_episode_ids_global"]),
            "total_samples_local": int(task_total_samples),
        }

    if overall_progress is not None:
        overall_progress.close()
    with open(output_dir / f"explore_manifest_rank{int(args.rank):02d}.json", "w", encoding="utf-8") as f:
        json.dump(explore_manifest, f, indent=2, ensure_ascii=False)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
