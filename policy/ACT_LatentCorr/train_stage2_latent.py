#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .correction_planner import PlannerCorrectionBuilder, PlannerCorrectionConfig
from .act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .evac_interface import EvacLatentTeacher
from .failure_utils import active_arm_pattern_id_to_key, error_mode_id_to_key, phase_id_to_key
from .latent_policy import ACTLatentStage1
from .merge_failure_tables import merge_failure_dir
from .stage2_latent_cache import build_stage2_latent_cache_relpath
from .train_stage1_latent import (
    _build_act_args,
    _resolve_dataset_info,
    cleanup_distributed,
    init_distributed_if_needed,
    is_main_process,
)
from .stage2_failure_dataset import build_failure_table_dataset
from .utils_latent import build_stage1_dataset, load_raw_episode, resolve_raw_data_dir
from .wandb_utils import finish_wandb, init_wandb_run, log_wandb, update_wandb_summary


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _empty_correction_stats() -> Counter:
    return Counter(
        {
            "generated": 0,
            "triggered": 0,
            "fallback": 0,
            "branch_interp_nearest": 0,
            "branch_gripper_close": 0,
            "branch_translation": 0,
            "branch_rotation": 0,
            "branch_planner": 0,
            "branch_other": 0,
            "branch_timeout": 0,
            "branch_error": 0,
        }
    )


def _update_correction_stats(counter: Counter, corr_meta: dict | None) -> None:
    if not isinstance(corr_meta, dict):
        return
    counter["generated"] += int(bool(corr_meta.get("correction_generated", True)))
    counter["triggered"] += int(bool(corr_meta.get("min_dist_triggered", False)))
    counter["fallback"] += int(bool(corr_meta.get("closed_loop_fallback_used", False)))
    branch = str(corr_meta.get("correction_branch", "other")).strip().lower()
    branch_key = {
        "interp_nearest": "branch_interp_nearest",
        "gripper_close": "branch_gripper_close",
        "translation": "branch_translation",
        "rotation": "branch_rotation",
        "planner": "branch_planner",
        "timeout": "branch_timeout",
        "error": "branch_error",
    }.get(branch, "branch_other")
    counter[branch_key] += 1


class _FrozenACTChunkPolicy:
    def __init__(self, base_act):
        self.base_act = base_act

    @property
    def training(self) -> bool:
        return bool(self.base_act.training)

    def eval(self):
        self.base_act.eval()
        return self

    def train(self, mode: bool = True):
        self.base_act.train(mode)
        return self

    def predict_act_chunk(self, qpos, image):
        return self.base_act(qpos, image)


def _load_base_act_from_ckpt(base_act, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    base_state = {
        key[len("base_act.") :]: value
        for key, value in state.items()
        if key.startswith("base_act.")
    }
    if not base_state:
        raise ValueError(f"Checkpoint does not contain base_act weights: {ckpt_path}")
    missing, unexpected = base_act.load_state_dict(base_state, strict=False)
    return missing, unexpected


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("ACT latent correction stage-2 minimal closed loop")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--raw_data_dir", type=str, default=None)
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--base_anchor_ckpt", type=str, default=None)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--camera_names", nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--correction_batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--prefix_steps", type=int, default=16)
    parser.add_argument("--act_chunk_size", type=int, default=None)
    parser.add_argument("--future_offset", type=int, default=None)
    parser.add_argument("--max_rollout_steps", type=int, default=1)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--beta_dynamics_max", type=float, default=1.0)
    parser.add_argument("--lambda_wm_action_current", type=float, default=0.5)
    parser.add_argument("--lambda_wm_action_future", type=float, default=1.0)
    parser.add_argument("--lambda_bridge_future", type=float, default=0.25)
    parser.add_argument("--dyn_zero_steps", type=int, default=0)
    parser.add_argument("--dyn_ramp_steps", type=int, default=1000)
    parser.add_argument("--dyn_warmup_curve", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--dyn_schedule_unit", type=str, default="step", choices=["step", "epoch"])
    parser.add_argument("--reference_global_batch_size", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--predictor_num_blocks", type=int, default=3)
    parser.add_argument("--projector_mid_channels", type=int, default=256)
    parser.add_argument("--wm_adapter_mid_channels", type=int, default=128)
    parser.add_argument("--readout_adapter_mid_channels", type=int, default=128)
    parser.add_argument("--predictor_mlp_hidden", type=int, default=512)
    parser.add_argument("--action_decoder_hidden", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=27)
    parser.add_argument("--retain_weight", type=float, default=0.1)
    parser.add_argument("--retain_weight_final", type=float, default=None)
    parser.add_argument("--retain_decay_start_epoch", type=float, default=-1.0)
    parser.add_argument("--retain_decay_end_epoch", type=float, default=-1.0)
    parser.add_argument("--retain_decay_curve", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--bridge_weight", type=float, default=0.0)
    parser.add_argument("--freeze_base_act", type=str2bool, default=False)
    parser.add_argument("--detach_act_feature_for_latent", type=str2bool, default=False)
    parser.add_argument("--use_act_head_correction", type=str2bool, default=False)
    parser.add_argument(
        "--curobo_left_yml",
        type=str,
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml",
    )
    parser.add_argument(
        "--curobo_right_yml",
        type=str,
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml",
    )
    parser.add_argument("--planner_target_mode", type=str, default="backward", choices=["forward", "backward"])
    parser.add_argument("--planner_target_lookahead_steps", type=int, default=6)
    parser.add_argument("--planner_orient_weight", type=float, default=0.0573)
    parser.add_argument("--planner_gripper_penalty", type=float, default=1.0)
    parser.add_argument("--planner_nearest_window_radius", type=int, default=12)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, default=0.01)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, default=0.05)
    parser.add_argument("--planner_gripper_switch_ratio", type=float, default=0.8)
    parser.add_argument("--planner_interp_fallback", type=str2bool, default=True)
    parser.add_argument("--planner_warmup", type=str2bool, default=True)
    parser.add_argument("--correction_interp_nearest_enable", type=str2bool, default=False)
    parser.add_argument("--correction_interp_prefix_ratio", type=float, default=0.4)
    parser.add_argument("--correction_builder_mode", type=str, default="legacy", choices=["legacy", "act_aligned"])
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, default=16)
    parser.add_argument("--act_aligned_min_dist_fallback_force_correction", type=str2bool, default=True)
    parser.add_argument("--act_aligned_min_dist_recover_ratio", type=float, default=0.75)
    parser.add_argument("--act_aligned_real_error_trigger_enable", type=str2bool, default=True)
    parser.add_argument("--act_aligned_real_error_min_dist_thresh", type=float, default=0.01)
    parser.add_argument("--act_aligned_real_error_min_dist_delta_thresh", type=float, default=0.005)
    parser.add_argument("--act_aligned_debug_recover_eval_rollout", type=str2bool, default=False)
    parser.add_argument("--act_aligned_debug_correction_evac_rollout", type=str2bool, default=False)
    parser.add_argument("--recover_eval_enable", type=str2bool, default=False)
    parser.add_argument("--recover_eval_save_video", type=str2bool, default=False)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, default=0.8)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, default=0.03)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, default=10.0)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, default=16)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, default=16)
    parser.add_argument("--act_aligned_correction_interp_nearest_enable", type=str2bool, default=False)
    parser.add_argument("--act_aligned_correction_interp_prefix_ratio", type=float, default=0.6)
    parser.add_argument("--act_aligned_correction_planner_prefix_ratio", type=float, default=0.5)
    parser.add_argument("--act_aligned_correction_gripper_close_prefix_ratio", type=float, default=0.32)
    parser.add_argument("--act_aligned_correction_compose_gt_tail_enable", type=str2bool, default=True)
    parser.add_argument("--act_aligned_correction_gripper_switch_ratio", type=float, default=0.5)
    parser.add_argument("--act_aligned_recover_gripper_penalty", type=float, default=0.0)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, default=True)
    parser.add_argument("--act_aligned_perturb_prob", type=float, default=1.0)
    parser.add_argument("--act_aligned_perturb_error_mode", type=str, default="open_laptop_pregrasp")
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_close_prob", type=float, default=0.5)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_translation_prob", type=float, default=0.0)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_rotation_prob", type=float, default=0.0)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, default=0.10)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, default=15.0)
    parser.add_argument("--act_aligned_perturb_mag_random", type=str2bool, default=False)
    parser.add_argument("--act_aligned_perturb_mag_rand_min", type=float, default=1.0)
    parser.add_argument("--act_aligned_perturb_mag_rand_max", type=float, default=1.4)
    parser.add_argument("--act_aligned_perturb_reject_sampling_enable", type=str2bool, default=True)
    parser.add_argument("--act_aligned_perturb_reject_max_trials", type=int, default=4)
    parser.add_argument("--act_aligned_perturb_reject_dir_jitter_eps", type=float, default=0.2)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, default=0.10)
    parser.add_argument("--act_aligned_perturb_gripper_open_max", type=float, default=0.90)
    parser.add_argument("--act_aligned_perturb_gripper_fast_ratio", type=float, default=0.20)
    parser.add_argument("--act_aligned_sample_pregrasp_phase_window_len", type=int, default=30)
    parser.add_argument("--act_aligned_error_policy_ckpt", type=str, default="")
    parser.add_argument("--act_aligned_sample_timeout_sec", type=float, default=30.0)
    parser.add_argument("--failure_mode", type=str, default="off", choices=["off", "train", "explore"])
    parser.add_argument("--failure_table_path", type=str, default="")
    parser.add_argument("--failure_table_dir", type=str, default="")
    parser.add_argument("--failure_phase_bins", type=int, default=3)
    parser.add_argument("--failure_translation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_translation_mag_bins", type=int, default=3)
    parser.add_argument("--failure_rotation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_rotation_mag_bins", type=int, default=3)
    parser.add_argument("--failure_explore_k", type=int, default=4)
    parser.add_argument("--failure_fail_recover_rate_thresh", type=float, default=0.5)
    parser.add_argument("--failure_sample_skip_head_ratio", type=float, default=0.6)
    parser.add_argument("--stage2_latent_cache_dir", type=str, default="")
    parser.add_argument("--stage2_latent_cache_strict", type=str2bool, default=False)
    parser.add_argument("--stage2_latent_cache_writeback", type=str2bool, default=True)
    parser.add_argument(
        "--act_like_loss_only",
        type=str2bool,
        default=False,
        help="Train stage2 with ACT-style correction loss only, skipping latent rollout/dynamics inside each step.",
    )
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="RoboTwin_ACT_LatentCorr")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="auto", choices=["auto", "online", "offline", "disabled"])
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    return parser


def compute_retain_weight(args, epoch_progress: float) -> float:
    start = float(args.retain_decay_start_epoch)
    end = float(args.retain_decay_end_epoch)
    initial = float(args.retain_weight)
    final = initial if args.retain_weight_final is None else float(args.retain_weight_final)
    if start < 0.0 or end <= start:
        return initial
    if epoch_progress <= start:
        return initial
    if epoch_progress >= end:
        return final
    ratio = (epoch_progress - start) / max(1e-8, end - start)
    ratio = max(0.0, min(1.0, ratio))
    if args.retain_decay_curve == "cosine":
        ratio = 0.5 * (1.0 - math.cos(math.pi * ratio))
    return initial + (final - initial) * ratio


def write_heartbeat(output_dir: str, payload: dict) -> None:
    path = os.path.join(output_dir, "heartbeat.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _iter_leaf_datasets(ds):
    if ds is None:
        return
    if hasattr(ds, "datasets"):
        for sub in ds.datasets:
            yield from _iter_leaf_datasets(sub)
        return
    if hasattr(ds, "dataset"):
        yield from _iter_leaf_datasets(ds.dataset)
        return
    yield ds


def _set_explore_unit_idx_for_loader(loader, unit_idx: int) -> None:
    if loader is None:
        return
    ds = getattr(loader, "dataset", None)
    for leaf in _iter_leaf_datasets(ds):
        if hasattr(leaf, "set_explore_unit_idx"):
            leaf.set_explore_unit_idx(int(unit_idx))


def _set_explore_local_k_for_loader(loader, k_local: int) -> None:
    if loader is None:
        return
    ds = getattr(loader, "dataset", None)
    for leaf in _iter_leaf_datasets(ds):
        if hasattr(leaf, "set_explore_local_k"):
            leaf.set_explore_local_k(int(k_local))


def _record_explore_trial_for_loader(loader, unit_idx: int, episode_id: int, start_ts: int) -> None:
    if loader is None:
        return
    ds = getattr(loader, "dataset", None)
    for leaf in _iter_leaf_datasets(ds):
        if hasattr(leaf, "record_explore_trial"):
            leaf.record_explore_trial(int(unit_idx), int(episode_id), int(start_ts))


def _get_explore_num_units(loader) -> int:
    ds = getattr(loader, "dataset", None)
    for leaf in _iter_leaf_datasets(ds):
        if hasattr(leaf, "_explore_units"):
            return int(len(getattr(leaf, "_explore_units")))
    return 0


def _get_current_explore_unit_idx(loader) -> int:
    ds = getattr(loader, "dataset", None)
    for leaf in _iter_leaf_datasets(ds):
        if hasattr(leaf, "_explore_curr_unit_idx"):
            return int(getattr(leaf, "_explore_curr_unit_idx"))
    return -1


def _get_current_explore_trial_count(loader) -> int:
    ds = getattr(loader, "dataset", None)
    for leaf in _iter_leaf_datasets(ds):
        if hasattr(leaf, "_explore_trial_count_local"):
            return int(getattr(leaf, "_explore_trial_count_local"))
    return 0


def _get_explore_completed_unit_count(loader) -> int:
    ds = getattr(loader, "dataset", None)
    for leaf in _iter_leaf_datasets(ds):
        if hasattr(leaf, "_explore_completed_unit_count"):
            return int(getattr(leaf, "_explore_completed_unit_count"))
    return 0


def _sync_all_explore_status(
    device: str,
    local_done: bool,
    local_completed: int,
    local_total: int,
    local_unit_idx: int,
    local_trial_count: int,
):
    status_t = torch.tensor(
        [
            1 if bool(local_done) else 0,
            int(local_completed),
            int(local_total),
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
                    "done": bool(vals[0]),
                    "completed": int(vals[1]),
                    "total": int(vals[2]),
                    "unit_idx": int(vals[3]),
                    "trial_count": int(vals[4]),
                }
            )
    else:
        progress = [
            {
                "rank": 0,
                "done": bool(local_done),
                "completed": int(local_completed),
                "total": int(local_total),
                "unit_idx": int(local_unit_idx),
                "trial_count": int(local_trial_count),
            }
        ]
    all_done = bool(progress and all(bool(x["done"]) for x in progress))
    return all_done, progress


def _format_explore_progress(progress: list[dict]) -> str:
    parts = []
    for item in progress:
        total_i = max(0, int(item["total"]))
        comp_i = max(0, int(item["completed"]))
        unit_i = max(0, int(item["unit_idx"])) + 1 if total_i > 0 else 0
        trial_i = max(0, int(item["trial_count"]))
        wait_tag = "w" if bool(item["done"]) else ""
        parts.append(f"r{int(item['rank'])}:{comp_i}/{total_i} u{unit_i} t{trial_i}{wait_tag}")
    return " | ".join(parts)


def main():
    args = build_argparser().parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    args.failure_mode = str(args.failure_mode).strip().lower()
    if args.failure_mode in {"train", "explore"} and args.correction_builder_mode != "act_aligned":
        raise ValueError("failure_mode=train/explore requires --correction_builder_mode act_aligned")
    if args.failure_mode == "train" and int(args.correction_batch_size) <= 0:
        raise ValueError("failure_mode=train requires correction_batch_size > 0")
    if args.failure_mode == "explore":
        args.recover_eval_enable = True
    distributed, rank, world_size, runtime_device = init_distributed_if_needed(args)
    args.device = runtime_device
    device_multiplier = world_size if distributed else 1
    base_global_batch_size = args.batch_size * device_multiplier
    corr_global_batch_size = args.correction_batch_size * device_multiplier
    current_global_batch_size = (args.batch_size + args.correction_batch_size) * device_multiplier
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        if args.stage2_latent_cache_dir:
            args.stage2_latent_cache_dir = os.path.realpath(args.stage2_latent_cache_dir)
            os.makedirs(args.stage2_latent_cache_dir, exist_ok=True)
    if distributed:
        dist.barrier()
    if args.stage2_latent_cache_dir:
        args.stage2_latent_cache_dir = os.path.realpath(args.stage2_latent_cache_dir)

    def _heartbeat_status(status: str, **extra) -> None:
        if not is_main_process(rank):
            return
        payload = {
            "timestamp": time.time(),
            "status": status,
            "epoch": 0,
            "global_step": 0,
            "sample_count": 0,
            "resume_ckpt": args.resume_ckpt,
            "stage1_ckpt": args.stage1_ckpt,
        }
        payload.update(extra)
        write_heartbeat(args.output_dir, payload)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dir, num_episodes, camera_names = _resolve_dataset_info(args.task_name)
    if args.dataset_dir is not None:
        dataset_dir = os.path.realpath(args.dataset_dir)
    if args.num_episodes is not None:
        num_episodes = int(args.num_episodes)
    if args.camera_names is not None:
        camera_names = list(args.camera_names)
    future_offset = args.future_offset if args.future_offset is not None else args.prefix_steps
    raw_data_dir = resolve_raw_data_dir(args.task_name, args.raw_data_dir)
    source_ckpt_args = {}
    source_ckpt_path = args.resume_ckpt or args.stage1_ckpt
    source_ckpt = None
    if source_ckpt_path:
        source_ckpt = torch.load(source_ckpt_path, map_location="cpu")
        source_ckpt_args = dict(source_ckpt.get("args", {}))
    act_chunk_size = int(args.act_chunk_size or source_ckpt_args.get("act_chunk_size") or args.prefix_steps)
    if act_chunk_size < args.prefix_steps:
        raise ValueError(f"act_chunk_size ({act_chunk_size}) must be >= prefix_steps ({args.prefix_steps})")

    print(f"[stage2] dataset_dir={dataset_dir}")
    print(f"[stage2] raw_data_dir={raw_data_dir}")
    print(f"[stage2] num_episodes={num_episodes}, cameras={camera_names}")
    print(
        f"[stage2] act_chunk_size={act_chunk_size}, "
        f"prefix_steps={args.prefix_steps}, future_offset={future_offset}"
    )
    print(
        f"[stage2] base_batch_size={args.batch_size}, "
        f"correction_batch_size={args.correction_batch_size}",
    )
    print(
        "[stage2] training_mode="
        + ("act_style_supervision_only" if args.act_like_loss_only else "latent_rollout_with_dynamics")
    )
    if args.correction_batch_size > 0:
        print(
            f"[stage2] correction_failure_mode={args.failure_mode}, "
            f"failure_table_path={args.failure_table_path or '<none>'}",
        )
    if args.stage2_latent_cache_dir:
        print(
            f"[stage2] stage2_latent_cache_dir={args.stage2_latent_cache_dir} "
            f"strict={args.stage2_latent_cache_strict} writeback={args.stage2_latent_cache_writeback}"
        )
    _heartbeat_status(
        "init_dataset_ready",
        failure_mode=args.failure_mode,
        dataset_dir=dataset_dir,
        raw_data_dir=raw_data_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        act_chunk_size=act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=future_offset,
    )

    dataset, norm_stats = build_stage1_dataset(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        act_chunk_size=act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=future_offset,
    )
    use_failure_explore = str(args.failure_mode).strip().lower() == "explore"
    use_failure_train = str(args.failure_mode).strip().lower() == "train"
    corr_dataset = None
    if use_failure_explore or (args.correction_batch_size > 0 and args.failure_mode in {"train", "explore"}):
        corr_start_margin = int(max(0, args.max_rollout_steps)) * int(max(1, args.act_aligned_rollout_exec_steps))
        corr_dataset, corr_norm_stats = build_failure_table_dataset(
            dataset_dir=dataset_dir,
            num_episodes=num_episodes,
            camera_names=camera_names,
            act_chunk_size=act_chunk_size,
            prefix_steps=args.prefix_steps,
            future_offset=future_offset,
            sample_phase_window_len=args.act_aligned_sample_pregrasp_phase_window_len,
            sample_skip_head_ratio=args.failure_sample_skip_head_ratio,
            start_margin=corr_start_margin,
            failure_mode=args.failure_mode,
            failure_table_path=args.failure_table_path,
            failure_phase_bins=args.failure_phase_bins,
            failure_translation_dir_bins=args.failure_translation_dir_bins,
            failure_translation_mag_bins=args.failure_translation_mag_bins,
            failure_rotation_dir_bins=args.failure_rotation_dir_bins,
            failure_rotation_mag_bins=args.failure_rotation_mag_bins,
            failure_explore_k=args.failure_explore_k,
        )
    sampler = None
    corr_sampler = None
    corr_dataloader = None
    explore_sampler = None
    explore_dataloader = None
    loader_workers = int(max(0, args.num_workers))
    loader_kwargs = {
        "num_workers": loader_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            **loader_kwargs,
        )
        if use_failure_explore:
            if corr_dataset is None:
                raise RuntimeError("failure_mode=explore requires corr_dataset")
            explore_sampler = DistributedSampler(
                corr_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed + 1000,
            )
            explore_dataloader = DataLoader(
                corr_dataset,
                batch_size=args.batch_size,
                sampler=explore_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        elif args.correction_batch_size > 0:
            corr_source_dataset = corr_dataset if corr_dataset is not None else dataset
            corr_sampler = DistributedSampler(
                corr_source_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed + 1000,
            )
            corr_dataloader = DataLoader(
                corr_source_dataset,
                batch_size=args.correction_batch_size,
                sampler=corr_sampler,
                shuffle=False,
                **loader_kwargs,
            )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        if use_failure_explore:
            if corr_dataset is None:
                raise RuntimeError("failure_mode=explore requires corr_dataset")
            explore_dataloader = DataLoader(
                corr_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                **loader_kwargs,
            )
        elif args.correction_batch_size > 0:
            corr_source_dataset = corr_dataset if corr_dataset is not None else dataset
            corr_dataloader = DataLoader(
                corr_source_dataset,
                batch_size=args.correction_batch_size,
                shuffle=True,
                **loader_kwargs,
            )
    if source_ckpt is not None and source_ckpt.get("norm_stats") is not None:
        norm_stats = source_ckpt["norm_stats"]
        if hasattr(dataset, "stats"):
            dataset.stats = norm_stats
        if corr_dataset is not None and hasattr(corr_dataset, "norm_stats"):
            corr_dataset.norm_stats = norm_stats

    latent_model_cfg = LatentModelConfig(
        projector_mid_channels=args.projector_mid_channels,
        wm_adapter_mid_channels=args.wm_adapter_mid_channels,
        readout_adapter_mid_channels=args.readout_adapter_mid_channels,
        predictor_num_blocks=args.predictor_num_blocks,
        predictor_mlp_hidden=args.predictor_mlp_hidden,
        action_decoder_hidden=args.action_decoder_hidden,
        action_dim=args.action_dim,
        prefix_steps=args.prefix_steps,
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_align=args.lambda_align,
        beta_dynamics_max=args.beta_dynamics_max,
        lambda_wm_action_current=args.lambda_wm_action_current,
        lambda_wm_action_future=args.lambda_wm_action_future,
        lambda_bridge_future=args.lambda_bridge_future,
        use_projector_detach_for_predictor=True,
        use_projector_detach_for_action_decoder=True,
        detach_act_feature_for_latent=args.detach_act_feature_for_latent,
    )
    warmup_cfg = DynamicsWarmupConfig(
        zero_steps=args.dyn_zero_steps,
        ramp_steps=args.dyn_ramp_steps,
        max_weight=1.0,
        curve=args.dyn_warmup_curve,
        unit=args.dyn_schedule_unit,
    )

    if is_main_process(rank):
        print(f"[stage2] init: loading EVAC teacher from {args.evac_ckpt}", flush=True)
    _heartbeat_status("init_loading_teacher", evac_ckpt=args.evac_ckpt)
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
    )
    if is_main_process(rank):
        print("[stage2] init: EVAC teacher ready", flush=True)
    _heartbeat_status("init_teacher_ready")
    act_args = _build_act_args(camera_names, argparse.Namespace(**{**vars(args), "act_chunk_size": act_chunk_size}))
    if is_main_process(rank):
        print("[stage2] init: building ACT latent model", flush=True)
    model = ACTLatentStage1(
        act_args=act_args,
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
    ).to(args.device)
    _heartbeat_status("init_model_ready")

    init_loader = explore_dataloader if use_failure_explore and explore_dataloader is not None else dataloader
    if is_main_process(rank):
        print("[stage2] init: running latent head bootstrap batch", flush=True)
    first_batch = next(iter(init_loader))
    model.initialize_latent_heads(first_batch["image_t"][:1].to(args.device), teacher)
    if is_main_process(rank):
        print("[stage2] init: latent heads ready", flush=True)
    _heartbeat_status("init_latent_heads_ready")

    start_epoch = 0
    global_step = 0
    sample_count = 0
    optimizer_state = None

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        optimizer_state = ckpt.get("optimizer")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        sample_count = int(ckpt.get("sample_count", global_step * current_global_batch_size))
        if is_main_process(rank):
            print(f"[stage2] resumed stage2 checkpoint: {args.resume_ckpt}", flush=True)
            print(f"[stage2] resume start_epoch={start_epoch} global_step={global_step}", flush=True)
            print(f"[stage2] sample_count={sample_count}", flush=True)
            print(f"[stage2] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    elif args.stage1_ckpt:
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if is_main_process(rank):
            print(f"[stage2] loaded stage1 checkpoint: {args.stage1_ckpt}", flush=True)
            print(f"[stage2] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if args.freeze_base_act:
        model.set_base_act_frozen(True)
        if is_main_process(rank):
            print("[stage2] base_act frozen", flush=True)

    base_anchor_act = copy.deepcopy(model.base_act).to(args.device)
    if args.base_anchor_ckpt:
        anchor_ckpt = torch.load(args.base_anchor_ckpt, map_location="cpu")
        anchor_model_state = anchor_ckpt.get("model", anchor_ckpt)
        anchor_state = {
            key[len("base_act.") :]: value
            for key, value in anchor_model_state.items()
            if key.startswith("base_act.")
        }
        if not anchor_state:
            raise ValueError(f"base_anchor_ckpt does not contain base_act weights: {args.base_anchor_ckpt}")
        missing, unexpected = base_anchor_act.load_state_dict(anchor_state, strict=False)
        if is_main_process(rank):
            print(f"[stage2] explicit base anchor loaded from: {args.base_anchor_ckpt}", flush=True)
            print(f"[stage2] base anchor missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    for param in base_anchor_act.parameters():
        param.requires_grad = False
    base_anchor_act.eval()
    if is_main_process(rank):
        print("[stage2] initialized explicit base anchor from stage1 baseline", flush=True)

    error_policy_model = None
    if args.act_aligned_error_policy_ckpt:
        error_base_act = copy.deepcopy(model.base_act).to(args.device)
        missing, unexpected = _load_base_act_from_ckpt(
            error_base_act,
            args.act_aligned_error_policy_ckpt,
            args.device,
        )
        for param in error_base_act.parameters():
            param.requires_grad = False
        error_base_act.eval()
        error_policy_model = _FrozenACTChunkPolicy(error_base_act)
        if is_main_process(rank):
            print(
                f"[stage2] explicit act_aligned error policy loaded from: {args.act_aligned_error_policy_ckpt}",
                flush=True,
            )
            print(
                f"[stage2] error policy base_act missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            if is_main_process(rank):
                print("[stage2] optimizer state restored", flush=True)
        except ValueError as exc:
            if is_main_process(rank):
                print(f"[stage2] optimizer state restore skipped: {exc}", flush=True)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if is_main_process(rank):
        print(f"[stage2] distributed={distributed} rank={rank} world_size={world_size} device={args.device}", flush=True)
        print(
            f"[stage2] current_global_batch_size={current_global_batch_size} "
            f"(base={base_global_batch_size}, corr={corr_global_batch_size}) "
            f"reference_global_batch_size={args.reference_global_batch_size}",
            flush=True,
        )
        print(f"[stage2] trainable_params={trainable_count} frozen_params={frozen_count}", flush=True)

    if distributed:
        model = DDP(
            model,
            device_ids=[int(args.device.split(":")[-1])],
            output_device=int(args.device.split(":")[-1]),
            find_unused_parameters=True,
        )

    raw_cache: dict[int, dict] = {}
    if args.correction_builder_mode == "legacy":
        planner_cfg = PlannerCorrectionConfig(
            correction_horizon=args.prefix_steps,
            target_mode=args.planner_target_mode,
            target_lookahead_steps=args.planner_target_lookahead_steps,
            correction_interp_nearest_enable=args.correction_interp_nearest_enable,
            correction_interp_prefix_ratio=args.correction_interp_prefix_ratio,
            correction_gripper_switch_ratio=args.planner_gripper_switch_ratio,
            use_interp_fallback_on_planner_fail=args.planner_interp_fallback,
            orient_weight=args.planner_orient_weight,
            gripper_penalty=args.planner_gripper_penalty,
            nearest_window_radius=args.planner_nearest_window_radius,
            active_joint_delta_thresh=args.planner_active_joint_delta_thresh,
            active_gripper_delta_thresh=args.planner_active_gripper_delta_thresh,
        )
        correction_builder = PlannerCorrectionBuilder(
            urdf_path=args.urdf_path,
            curobo_left_yml=args.curobo_left_yml,
            curobo_right_yml=args.curobo_right_yml,
            cfg=planner_cfg,
            device=args.device,
            planner_warmup=args.planner_warmup,
        )
    else:
        act_aligned_cfg = build_act_aligned_cfg_from_args(args, max_action_len=act_chunk_size)
        if is_main_process(rank):
            print("[stage2] init: building ACT-aligned correction builder", flush=True)
        correction_builder = ACTAlignedCorrectionBuilder(
            cfg=act_aligned_cfg,
            urdf_path=args.urdf_path,
            curobo_left_yml=args.curobo_left_yml,
            curobo_right_yml=args.curobo_right_yml,
            device=args.device,
            shared_evac_model=teacher.model,
            shared_evac_config=teacher.cfg,
        )
    _heartbeat_status("init_correction_builder_ready")
    config_path = os.path.join(args.output_dir, "stage2_config.txt")
    if is_main_process(rank):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(str(vars(args)))
            f.write("\n")
            f.write(str(asdict(latent_model_cfg)))
            f.write("\n")
            f.write(str(asdict(latent_loss_cfg)))
            f.write("\n")
            f.write(str(asdict(warmup_cfg)))

    qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=args.device)
    qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=args.device)

    wandb_run = init_wandb_run(
        enabled=is_main_process(rank) and args.use_wandb and args.wandb_mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or f"stage2_{os.path.basename(args.output_dir)}",
        group=args.wandb_group or f"{args.task_name}_stage2",
        tags=args.wandb_tags or ["stage2", args.task_name, "planner_teacher"],
        mode=args.wandb_mode,
        output_dir=args.output_dir,
        config={
            "stage": "stage2",
            "args": vars(args),
            "latent_model_cfg": asdict(latent_model_cfg),
            "latent_loss_cfg": asdict(latent_loss_cfg),
            "warmup_cfg": asdict(warmup_cfg),
        },
    )

    failure_table_dir = str(args.failure_table_dir).strip()
    if failure_table_dir == "":
        failure_table_dir = os.path.join(args.output_dir, "failure_explore")
    failure_live_dump_interval = 20
    failure_trials: list[dict] = []
    failure_stats: dict[tuple, dict] = {}
    explore_first_debug_path = os.path.join(args.output_dir, "explore_first_sample_debug.json")

    def _write_failure_tables(epoch_idx: int) -> None:
        if not use_failure_explore:
            return
        os.makedirs(failure_table_dir, exist_ok=True)
        rank_suffix = f"_rank{int(rank):02d}"
        trials_path = os.path.join(failure_table_dir, f"failure_trials_live{rank_suffix}.json")
        with open(trials_path, "w", encoding="utf-8") as f:
            json.dump(list(failure_trials), f, indent=2, ensure_ascii=False)
        meta = {
            "version": 1,
            "mode": "explore_meta_live",
            "epoch": int(epoch_idx + 1),
            "failure_phase_bins": int(args.failure_phase_bins),
            "failure_translation_dir_bins": int(args.failure_translation_dir_bins),
            "failure_translation_mag_bins": int(args.failure_translation_mag_bins),
            "failure_rotation_dir_bins": int(args.failure_rotation_dir_bins),
            "failure_rotation_mag_bins": int(args.failure_rotation_mag_bins),
            "failure_explore_k": int(args.failure_explore_k),
        }
        meta_path = os.path.join(failure_table_dir, f"failure_meta{rank_suffix}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    try:
        if is_main_process(rank):
            write_heartbeat(
                args.output_dir,
                {
                    "timestamp": time.time(),
                    "status": "started",
                    "epoch": start_epoch,
                    "global_step": global_step,
                    "sample_count": sample_count,
                    "resume_ckpt": args.resume_ckpt,
                    "stage1_ckpt": args.stage1_ckpt,
                },
            )
        if use_failure_explore:
            if explore_dataloader is None:
                raise RuntimeError("failure_mode=explore requires explore_dataloader")
            explore_num_units = int(_get_explore_num_units(explore_dataloader))
            failure_explore_k_global = int(max(1, int(args.failure_explore_k)))
            failure_explore_k_local = int(max(1, (failure_explore_k_global + world_size - 1) // world_size))
            if explore_num_units > 0:
                _set_explore_unit_idx_for_loader(explore_dataloader, 0)
                _set_explore_local_k_for_loader(explore_dataloader, failure_explore_k_local)
            init_unit_idx = _get_current_explore_unit_idx(explore_dataloader)
            init_trial_count = _get_current_explore_trial_count(explore_dataloader)
            init_completed_units = int(_get_explore_completed_unit_count(explore_dataloader))
            local_explore_done = bool(init_completed_units >= explore_num_units)
            all_explore_done, explore_progress = _sync_all_explore_status(
                args.device,
                local_explore_done,
                init_completed_units,
                explore_num_units,
                init_unit_idx,
                init_trial_count,
            )
            explore_pbar = None
            if is_main_process(rank):
                total_all = int(sum(int(x["total"]) for x in explore_progress))
                done_all = int(sum(int(x["completed"]) for x in explore_progress))
                explore_pbar = tqdm(total=max(1, total_all), desc="Explore Units", dynamic_ncols=True)
                if done_all > 0:
                    explore_pbar.update(done_all)
                explore_pbar.set_postfix_str(_format_explore_progress(explore_progress))
            for epoch in range(start_epoch, args.num_epochs):
                if all_explore_done:
                    break
                if explore_sampler is not None:
                    explore_sampler.set_epoch(epoch + 1000)
                model.train()
                raw_model_for_mode = model.module if isinstance(model, DDP) else model
                if args.freeze_base_act:
                    raw_model_for_mode.base_act.eval()
                pbar = tqdm(
                    explore_dataloader,
                    desc=f"stage2 explore {epoch}",
                    leave=True,
                    disable=not is_main_process(rank),
                )
                for batch_idx, batch in enumerate(pbar):
                    image_t = batch["image_t"].to(args.device, non_blocking=True)
                    qpos_t = batch["qpos_t"].to(args.device, non_blocking=True)
                    episode_ids = batch["episode_id"].tolist()
                    start_ts_list = batch["start_ts"].tolist()
                    sampled_phase_ids = batch["sampled_phase_id"].tolist()
                    pregrasp_seg_starts = batch["pregrasp_seg_start"].tolist()
                    pregrasp_seg_ends = batch["pregrasp_seg_end"].tolist()
                    sampled_phase_bin_ids = batch["sampled_phase_bin_id"].tolist()
                    sampled_phase_instance_ids = batch["sampled_phase_instance_id"].tolist()
                    forced_error_mode_ids = batch["forced_error_mode_id"].tolist()
                    sampled_active_arm_pattern_ids = batch["sampled_active_arm_pattern_id"].tolist()
                    forced_dir_bin_ids = batch["forced_dir_bin_id"].tolist()
                    forced_mag_bin_ids = batch["forced_mag_bin_id"].tolist()
                    sampled_explore_unit_ids = (
                        batch["sampled_explore_unit_idx"].tolist()
                        if "sampled_explore_unit_idx" in batch
                        else [-1] * image_t.shape[0]
                    )

                    for i in range(image_t.shape[0]):
                        ep_id = int(episode_ids[i])
                        if ep_id not in raw_cache:
                            raw_cache[ep_id] = load_raw_episode(raw_data_dir, ep_id)
                        raw_data = raw_cache[ep_id]
                        correction_policy_model = error_policy_model if error_policy_model is not None else raw_model_for_mode
                        corr = correction_builder.build(
                            latent_model=correction_policy_model,
                            image_t=image_t[i],
                            qpos_t=qpos_t[i],
                            raw_data=raw_data,
                            norm_stats=norm_stats,
                            start_ts=int(start_ts_list[i]),
                            failure_mode_override="explore",
                            sampled_phase_id=int(sampled_phase_ids[i]),
                            pregrasp_seg_start=int(pregrasp_seg_starts[i]),
                            pregrasp_seg_end=int(pregrasp_seg_ends[i]),
                            sampled_phase_bin_id=int(sampled_phase_bin_ids[i]),
                            sampled_phase_instance_id=int(sampled_phase_instance_ids[i]),
                            forced_error_mode_id=int(forced_error_mode_ids[i]),
                            sampled_active_arm_pattern_id=int(sampled_active_arm_pattern_ids[i]),
                            forced_dir_bin_id=int(forced_dir_bin_ids[i]),
                            forced_mag_bin_id=int(forced_mag_bin_ids[i]),
                        )
                        corr_meta = None
                        skip_meta = None
                        if isinstance(corr, dict):
                            corr_meta = corr.get("corr_meta")
                        if corr_meta is None:
                            skip_meta = correction_builder.pop_last_skip_meta()
                            corr_meta = skip_meta
                        if use_failure_explore and (not os.path.exists(explore_first_debug_path)):
                            debug_payload = {
                                "episode_id": int(ep_id),
                                "start_ts": int(start_ts_list[i]),
                                "forced_error_mode_id": int(forced_error_mode_ids[i]),
                                "sampled_phase_id": int(sampled_phase_ids[i]),
                                "corr_is_none": bool(corr is None),
                                "corr_keys": (list(corr.keys()) if isinstance(corr, dict) else None),
                                "skip_meta_used": bool(skip_meta is not None),
                                "corr_meta": corr_meta,
                                "skip_meta": skip_meta,
                            }
                            with open(explore_first_debug_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    debug_payload,
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                    default=lambda o: (o.tolist() if hasattr(o, "tolist") else str(o)),
                                )
                        if not isinstance(corr_meta, dict):
                            continue

                        forced_mode_id = int(forced_error_mode_ids[i])
                        forced_mode_key = None
                        if forced_mode_id >= 0:
                            forced_mode_key = error_mode_id_to_key(forced_mode_id)
                        error_mode_key = str(forced_mode_key or corr_meta.get("sampled_error_mode") or "").strip().lower()
                        recover_eval_last = corr_meta.get("recover_eval_last", {}) or {}
                        recoverable_raw = recover_eval_last.get("recoverable", None)
                        if recoverable_raw is None or error_mode_key not in {"translation", "rotation", "gripper_close"}:
                            continue

                        phase_key = phase_id_to_key(int(sampled_phase_ids[i]))
                        active_pattern_key = "both"
                        active_pattern_id = int(sampled_active_arm_pattern_ids[i])
                        if active_pattern_id >= 0:
                            active_pattern_key = active_arm_pattern_id_to_key(active_pattern_id)
                        sample_unit_idx = int(sampled_explore_unit_ids[i])
                        recoverable = bool(recoverable_raw)
                        trial = {
                            "epoch": int(epoch),
                            "global_step": int(global_step),
                            "episode_id": int(ep_id),
                            "start_ts": int(start_ts_list[i]),
                            "phase_key": str(phase_key),
                            "phase_instance_idx": int(sampled_phase_instance_ids[i]),
                            "phase_bin_id": int(sampled_phase_bin_ids[i]),
                            "error_mode": str(error_mode_key),
                            "active_arm_pattern": str(active_pattern_key),
                            "dir_bin_id": int(forced_dir_bin_ids[i]),
                            "mag_bin_id": int(forced_mag_bin_ids[i]),
                            "recoverable": bool(recoverable),
                            "recover_eval_mode": recover_eval_last.get("mode"),
                            "recover_eval_metric_name": recover_eval_last.get("metric_name"),
                            "recover_eval_metric": recover_eval_last.get("metric"),
                            "recover_eval_threshold": recover_eval_last.get("threshold"),
                        }
                        key = (
                            str(trial["phase_key"]),
                            int(trial["phase_instance_idx"]),
                            int(trial["phase_bin_id"]),
                            str(trial["error_mode"]),
                            str(trial["active_arm_pattern"]),
                            int(trial["dir_bin_id"]),
                            int(trial["mag_bin_id"]),
                        )
                        if key not in failure_stats:
                            failure_stats[key] = {"n": 0, "n_recover": 0, "seen_samples": set()}
                        sample_uid = (int(ep_id), int(start_ts_list[i]))
                        if sample_uid in failure_stats[key]["seen_samples"]:
                            continue
                        failure_stats[key]["seen_samples"].add(sample_uid)
                        failure_stats[key]["n"] += 1
                        failure_stats[key]["n_recover"] += int(recoverable)
                        failure_trials.append(trial)
                        _record_explore_trial_for_loader(
                            explore_dataloader,
                            sample_unit_idx,
                            int(ep_id),
                            int(start_ts_list[i]),
                        )

                    curr_unit_idx = _get_current_explore_unit_idx(explore_dataloader)
                    curr_trial_count = _get_current_explore_trial_count(explore_dataloader)
                    curr_completed_units = int(_get_explore_completed_unit_count(explore_dataloader))
                    local_explore_done = bool(curr_completed_units >= explore_num_units)
                    all_explore_done, explore_progress = _sync_all_explore_status(
                        args.device,
                        local_explore_done,
                        curr_completed_units,
                        explore_num_units,
                        curr_unit_idx,
                        curr_trial_count,
                    )
                    if is_main_process(rank) and explore_pbar is not None:
                        done_all = int(sum(int(x["completed"]) for x in explore_progress))
                        if done_all > int(explore_pbar.n):
                            explore_pbar.update(done_all - int(explore_pbar.n))
                        explore_pbar.set_postfix_str(_format_explore_progress(explore_progress))
                    if global_step % failure_live_dump_interval == 0:
                        _write_failure_tables(epoch)
                    log_wandb(
                        wandb_run,
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "failure_explore_trials": len(failure_trials),
                            "failure_explore_completed_units_local": curr_completed_units,
                            "failure_explore_total_units_local": explore_num_units,
                            "failure_explore_trial_count_local": curr_trial_count,
                        },
                        step=global_step,
                    )
                    global_step += 1
                    if all_explore_done:
                        break
                _write_failure_tables(epoch)
                if distributed:
                    dist.barrier()
                if all_explore_done:
                    break

            if distributed:
                dist.barrier()
            merged_failure_table = None
            if is_main_process(rank):
                merged_failure_table = merge_failure_dir(
                    failure_dir=failure_table_dir,
                    out_dir=failure_table_dir,
                    epoch=None,
                    failure_fail_recover_rate_thresh=float(args.failure_fail_recover_rate_thresh),
                )
                write_heartbeat(
                    args.output_dir,
                    {
                        "timestamp": time.time(),
                        "status": "finished",
                        "epoch": epoch + 1 if "epoch" in locals() else start_epoch,
                        "global_step": global_step,
                        "sample_count": sample_count,
                        "failure_table_dir": failure_table_dir,
                        "failure_table_path": merged_failure_table,
                        "failure_trials": len(failure_trials),
                    },
                )
            if distributed:
                dist.barrier()
            return
        for epoch in range(start_epoch, args.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            if corr_sampler is not None:
                corr_sampler.set_epoch(epoch + 1000)
            model.train()
            raw_model_for_mode = model.module if isinstance(model, DDP) else model
            if args.freeze_base_act:
                raw_model_for_mode.base_act.eval()
            start = time.time()
            pbar = tqdm(dataloader, desc=f"stage2 epoch {epoch}", leave=True, disable=not is_main_process(rank))
            corr_iter = iter(corr_dataloader) if corr_dataloader is not None else None
            meter_loss = 0.0
            meter_correct = 0.0
            meter_dyn = 0.0
            meter_retain = 0.0
            meter_bridge = 0.0
            meter_skipped = 0.0
            meter_retain_weight = 0.0
            meter_t_raw = 0.0
            meter_t_builder = 0.0
            meter_t_rollout = 0.0
            meter_t_model = 0.0
            meter_cache_hit = 0.0
            meter_cache_miss = 0.0
            corr_stats = _empty_correction_stats()
            steps_this_epoch = 0

            total_batches = max(1, len(dataloader))
            for batch_idx, batch in enumerate(pbar):
                batch_groups: list[tuple[str, dict]] = [("off", batch)]
                if corr_iter is not None:
                    try:
                        corr_batch = next(corr_iter)
                    except StopIteration:
                        corr_iter = iter(corr_dataloader)
                        corr_batch = next(corr_iter)
                    batch_groups.append((args.failure_mode, corr_batch))

                epoch_progress = epoch + (batch_idx / float(total_batches))
                retain_weight_cur = compute_retain_weight(args, epoch_progress)
                schedule_step = sample_count / float(max(1, args.reference_global_batch_size))
                dyn_schedule_value = epoch_progress if args.dyn_schedule_unit == "epoch" else schedule_step
                prepared_samples: list[dict] = []
                n_skipped = 0
                step_t_raw = 0.0
                step_t_builder = 0.0
                step_t_rollout = 0.0
                step_t_model = 0.0
                step_cache_hit = 0
                step_cache_miss = 0
                for batch_failure_mode, batch_src in batch_groups:
                    image_t = batch_src["image_t"].to(args.device, non_blocking=True)
                    qpos_t = batch_src["qpos_t"].to(args.device, non_blocking=True)
                    act_action_chunk_batch = batch_src["act_action_chunk"].to(args.device, non_blocking=True)
                    act_is_pad_batch = batch_src["act_is_pad"].to(args.device, non_blocking=True)
                    episode_ids = batch_src["episode_id"].tolist()
                    start_ts_list = batch_src["start_ts"].tolist()
                    sampled_phase_ids = (
                        batch_src["sampled_phase_id"].tolist()
                        if "sampled_phase_id" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    pregrasp_seg_starts = (
                        batch_src["pregrasp_seg_start"].tolist()
                        if "pregrasp_seg_start" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    pregrasp_seg_ends = (
                        batch_src["pregrasp_seg_end"].tolist()
                        if "pregrasp_seg_end" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    sampled_phase_bin_ids = (
                        batch_src["sampled_phase_bin_id"].tolist()
                        if "sampled_phase_bin_id" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    sampled_phase_instance_ids = (
                        batch_src["sampled_phase_instance_id"].tolist()
                        if "sampled_phase_instance_id" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    forced_error_mode_ids = (
                        batch_src["forced_error_mode_id"].tolist()
                        if "forced_error_mode_id" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    sampled_active_arm_pattern_ids = (
                        batch_src["sampled_active_arm_pattern_id"].tolist()
                        if "sampled_active_arm_pattern_id" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    forced_dir_bin_ids = (
                        batch_src["forced_dir_bin_id"].tolist()
                        if "forced_dir_bin_id" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    forced_mag_bin_ids = (
                        batch_src["forced_mag_bin_id"].tolist()
                        if "forced_mag_bin_id" in batch_src
                        else [-1] * image_t.shape[0]
                    )
                    sampled_mode_probs = (
                        batch_src["sampled_mode_prob"].tolist()
                        if "sampled_mode_prob" in batch_src
                        else [float("nan")] * image_t.shape[0]
                    )
                    sampled_entry_probs = (
                        batch_src["sampled_entry_prob_within_mode"].tolist()
                        if "sampled_entry_prob_within_mode" in batch_src
                        else [float("nan")] * image_t.shape[0]
                    )
                    sampled_unit_probs = (
                        batch_src["sampled_unit_prob"].tolist()
                        if "sampled_unit_prob" in batch_src
                        else [float("nan")] * image_t.shape[0]
                    )

                    with torch.no_grad():
                        base_anchor_chunk_batch = base_anchor_act(qpos_t, image_t)
                        pred_act_action_chunk_batch = None
                        correction_policy_pred_chunk_batch = None
                        if batch_failure_mode != "off":
                            if args.correction_builder_mode == "legacy":
                                pred_act_action_chunk_batch = raw_model_for_mode.predict_act_chunk(qpos_t, image_t)
                            else:
                                correction_policy_model = (
                                    error_policy_model if error_policy_model is not None else raw_model_for_mode
                                )
                                correction_policy_pred_chunk_batch = correction_policy_model.predict_act_chunk(
                                    qpos_t,
                                    image_t,
                                )

                    for i in range(image_t.shape[0]):
                        ep_id = int(episode_ids[i])
                        t_raw0 = time.perf_counter()
                        if ep_id not in raw_cache:
                            raw_cache[ep_id] = load_raw_episode(raw_data_dir, ep_id)
                        step_t_raw += time.perf_counter() - t_raw0
                        raw_data = raw_cache[ep_id]
                        qpos_raw = qpos_t[i : i + 1] * qpos_std.view(1, -1) + qpos_mean.view(1, -1)
                        if batch_failure_mode == "off":
                            correction_target_chunk = act_action_chunk_batch[i].detach().clone()
                            correction_is_pad = act_is_pad_batch[i].detach().clone()
                            correction_target_prefix = correction_target_chunk[: args.prefix_steps]
                            correction_is_pad_prefix = correction_is_pad[: args.prefix_steps]
                            prepared_samples.append(
                                {
                                    "image_t": image_t[i].detach(),
                                    "qpos_t": qpos_t[i].detach(),
                                    "qpos_raw": qpos_raw[0].detach(),
                                    "act_action_chunk": act_action_chunk_batch[i].detach(),
                                    "act_is_pad": act_is_pad_batch[i].detach(),
                                    "base_anchor_chunk": base_anchor_chunk_batch[i].detach(),
                                    "correction_target_chunk": correction_target_chunk,
                                    "correction_is_pad": correction_is_pad,
                                    "correction_target_prefix": correction_target_prefix,
                                    "correction_is_pad_prefix": correction_is_pad_prefix,
                                    "external_action_dev_norm": None,
                                    "external_action_dev_raw": None,
                                    "external_qpos_err_norm": None,
                                    "external_action_is_pad_prefix": correction_is_pad_prefix,
                                    "raw_data": raw_data,
                                }
                            )
                            continue

                        t_builder0 = time.perf_counter()
                        with torch.no_grad():
                            if args.correction_builder_mode == "legacy":
                                pred_act_action_chunk = pred_act_action_chunk_batch[i : i + 1]
                                action_dev_norm = pred_act_action_chunk[:, : args.prefix_steps]
                                action_mean = torch.as_tensor(
                                    norm_stats["action_mean"],
                                    dtype=action_dev_norm.dtype,
                                    device=action_dev_norm.device,
                                )
                                action_std = torch.as_tensor(
                                    norm_stats["action_std"],
                                    dtype=action_dev_norm.dtype,
                                    device=action_dev_norm.device,
                                )
                                action_dev_raw = action_dev_norm * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
                                action_dev_raw = action_dev_raw.clone()
                                action_dev_raw[..., 6] = action_dev_raw[..., 6].clamp(0.0, 1.0)
                                action_dev_raw[..., 13] = action_dev_raw[..., 13].clamp(0.0, 1.0)
                                corr = correction_builder.build(
                                    action_dev_raw=action_dev_raw[0],
                                    raw_data=raw_data,
                                    norm_stats=norm_stats,
                                    start_ts=int(start_ts_list[i]),
                                )
                            else:
                                correction_policy_model = error_policy_model if error_policy_model is not None else raw_model_for_mode
                                corr = correction_builder.build(
                                    latent_model=correction_policy_model,
                                    image_t=image_t[i],
                                    qpos_t=qpos_t[i],
                                    raw_data=raw_data,
                                    norm_stats=norm_stats,
                                    start_ts=int(start_ts_list[i]),
                                    failure_mode_override=batch_failure_mode,
                                    sampled_phase_id=int(sampled_phase_ids[i]),
                                    pregrasp_seg_start=int(pregrasp_seg_starts[i]),
                                    pregrasp_seg_end=int(pregrasp_seg_ends[i]),
                                    sampled_phase_bin_id=int(sampled_phase_bin_ids[i]),
                                    sampled_phase_instance_id=int(sampled_phase_instance_ids[i]),
                                    forced_error_mode_id=int(forced_error_mode_ids[i]),
                                    sampled_active_arm_pattern_id=int(sampled_active_arm_pattern_ids[i]),
                                    forced_dir_bin_id=int(forced_dir_bin_ids[i]),
                                    forced_mag_bin_id=int(forced_mag_bin_ids[i]),
                                    sampled_mode_prob=float(sampled_mode_probs[i]),
                                    sampled_entry_prob_within_mode=float(sampled_entry_probs[i]),
                                    sampled_unit_prob=float(sampled_unit_probs[i]),
                                    precomputed_action_chunk_norm=(
                                        None
                                        if correction_policy_pred_chunk_batch is None
                                        else correction_policy_pred_chunk_batch[i]
                                    ),
                                )
                        step_t_builder += time.perf_counter() - t_builder0
                        if corr is None:
                            if args.correction_builder_mode == "act_aligned":
                                _update_correction_stats(corr_stats, correction_builder.pop_last_skip_meta())
                            n_skipped += 1
                            continue
                        if args.correction_builder_mode == "act_aligned":
                            _update_correction_stats(corr_stats, corr.get("corr_meta"))
                        external_action_dev_norm = None
                        external_action_dev_raw = None
                        external_qpos_err_norm = None
                        external_action_is_pad_prefix = None
                        if "corr_action_chunk_norm" in corr:
                            correction_target_chunk = corr["corr_action_chunk_norm"].to(args.device)
                            correction_is_pad = corr["corr_is_pad"].to(args.device)
                            correction_target_prefix = correction_target_chunk[: args.prefix_steps]
                            correction_is_pad_prefix = correction_is_pad[: args.prefix_steps]
                            if corr.get("error_action_prefix_norm") is not None:
                                external_action_dev_norm = corr["error_action_prefix_norm"].to(args.device)
                            if corr.get("error_action_prefix_raw") is not None:
                                external_action_dev_raw = corr["error_action_prefix_raw"].to(args.device)
                            if corr.get("corr_qpos_norm") is not None:
                                external_qpos_err_norm = corr["corr_qpos_norm"].to(args.device)
                            if corr.get("error_is_pad_prefix") is not None:
                                external_action_is_pad_prefix = corr["error_is_pad_prefix"].to(args.device)
                        else:
                            correction_target_prefix = corr["correction_target_norm"].to(args.device)
                            correction_is_pad_prefix = corr["is_pad"].to(args.device)
                            correction_target_chunk = act_action_chunk_batch[i].detach().clone()
                            correction_is_pad = torch.ones_like(act_is_pad_batch[i], dtype=torch.bool)
                            correction_target_chunk[: args.prefix_steps] = correction_target_prefix
                            correction_is_pad[: args.prefix_steps] = correction_is_pad_prefix
                        if external_action_is_pad_prefix is None:
                            external_action_is_pad_prefix = correction_is_pad_prefix
                        cache_relpath = None
                        if args.stage2_latent_cache_dir and external_action_dev_raw is not None:
                            cache_relpath = build_stage2_latent_cache_relpath(
                                task_name=args.task_name,
                                episode_id=ep_id,
                                start_ts=int(start_ts_list[i]),
                                prefix_steps=args.prefix_steps,
                                ddim_steps=args.ddim_steps,
                                error_action_prefix_raw=external_action_dev_raw,
                            )
                        prepared_samples.append(
                            {
                                "image_t": image_t[i].detach(),
                                "qpos_t": qpos_t[i].detach(),
                                "qpos_raw": qpos_raw[0].detach(),
                                "act_action_chunk": act_action_chunk_batch[i].detach(),
                                "act_is_pad": act_is_pad_batch[i].detach(),
                                "base_anchor_chunk": base_anchor_chunk_batch[i].detach(),
                                "correction_target_chunk": correction_target_chunk.detach(),
                                "correction_is_pad": correction_is_pad.detach(),
                                "correction_target_prefix": correction_target_prefix.detach(),
                                "correction_is_pad_prefix": correction_is_pad_prefix.detach(),
                                "external_action_dev_norm": (
                                    None if external_action_dev_norm is None else external_action_dev_norm.detach()
                                ),
                                "external_action_dev_raw": (
                                    None if external_action_dev_raw is None else external_action_dev_raw.detach()
                                ),
                                "external_qpos_err_norm": (
                                    None if external_qpos_err_norm is None else external_qpos_err_norm.detach()
                                ),
                                "external_action_is_pad_prefix": external_action_is_pad_prefix.detach(),
                                "cache_relpath": cache_relpath,
                                "raw_data": raw_data,
                            }
                        )

                local_valid_count = len(prepared_samples)
                if distributed:
                    local_valid_tensor = torch.tensor([local_valid_count], device=args.device, dtype=torch.int64)
                    min_valid_tensor = local_valid_tensor.clone()
                    total_valid_tensor = local_valid_tensor.clone()
                    dist.all_reduce(min_valid_tensor, op=dist.ReduceOp.MIN)
                    dist.all_reduce(total_valid_tensor, op=dist.ReduceOp.SUM)
                    if min_valid_tensor.item() == 0:
                        if is_main_process(rank):
                            print(
                                "[stage2] skipped batch globally at "
                                f"step={global_step} because at least one rank produced no valid correction target "
                                f"(global_valid_sum={int(total_valid_tensor.item())})",
                                flush=True,
                            )
                        meter_skipped += n_skipped
                        continue

                if not prepared_samples:
                    print(f"[stage2] skipped batch at step={global_step} because planner returned no valid correction target")
                    meter_skipped += n_skipped
                    continue

                image_batch = torch.stack([item["image_t"] for item in prepared_samples], dim=0)
                qpos_batch = torch.stack([item["qpos_t"] for item in prepared_samples], dim=0)
                qpos_raw_batch = torch.stack([item["qpos_raw"] for item in prepared_samples], dim=0)
                act_action_chunk_train = torch.stack([item["act_action_chunk"] for item in prepared_samples], dim=0)
                act_is_pad_train = torch.stack([item["act_is_pad"] for item in prepared_samples], dim=0)
                base_anchor_chunk_train = torch.stack([item["base_anchor_chunk"] for item in prepared_samples], dim=0)
                correction_target_chunk_train = torch.stack([item["correction_target_chunk"] for item in prepared_samples], dim=0)
                correction_is_pad_train = torch.stack([item["correction_is_pad"] for item in prepared_samples], dim=0)
                correction_target_prefix_train = torch.stack(
                    [item["correction_target_prefix"] for item in prepared_samples],
                    dim=0,
                )
                correction_is_pad_prefix_train = torch.stack(
                    [item["correction_is_pad_prefix"] for item in prepared_samples],
                    dim=0,
                )

                action_dev_norm_batch = None
                action_dev_raw_batch = None
                qpos_err_norm_batch = None
                action_is_pad_prefix_batch = None
                rollout_latent_batch = None
                if not args.act_like_loss_only:
                    with torch.no_grad():
                        action_mean = torch.as_tensor(
                            norm_stats["action_mean"],
                            dtype=act_action_chunk_train.dtype,
                            device=act_action_chunk_train.device,
                        )
                        action_std = torch.as_tensor(
                            norm_stats["action_std"],
                            dtype=act_action_chunk_train.dtype,
                            device=act_action_chunk_train.device,
                        )
                        qpos_mean_local = torch.as_tensor(
                            norm_stats["qpos_mean"],
                            dtype=act_action_chunk_train.dtype,
                            device=act_action_chunk_train.device,
                        )
                        qpos_std_local = torch.as_tensor(
                            norm_stats["qpos_std"],
                            dtype=act_action_chunk_train.dtype,
                            device=act_action_chunk_train.device,
                        )
                        num_prepared = len(prepared_samples)
                        action_dev_norm_batch = torch.zeros(
                            (num_prepared, args.prefix_steps, act_action_chunk_train.shape[-1]),
                            dtype=act_action_chunk_train.dtype,
                            device=args.device,
                        )
                        action_dev_raw_batch = torch.zeros_like(action_dev_norm_batch)
                        qpos_err_norm_batch = torch.zeros(
                            (num_prepared, qpos_raw_batch.shape[-1]),
                            dtype=act_action_chunk_train.dtype,
                            device=args.device,
                        )
                        action_is_pad_prefix_batch = correction_is_pad_prefix_train.detach().clone()
                        missing_pred_indices = [
                            i
                            for i, item in enumerate(prepared_samples)
                            if item["external_action_dev_norm"] is None
                            or item["external_action_dev_raw"] is None
                            or item["external_qpos_err_norm"] is None
                        ]
                        if missing_pred_indices:
                            pred_index_tensor = torch.as_tensor(missing_pred_indices, dtype=torch.long, device=args.device)
                            rollout_pred_chunk = raw_model_for_mode.predict_act_chunk(
                                qpos_batch.index_select(0, pred_index_tensor),
                                image_batch.index_select(0, pred_index_tensor),
                            )
                            rollout_pred_chunk = rollout_pred_chunk[:, : args.prefix_steps].detach().clone()
                            rollout_pred_raw = rollout_pred_chunk * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
                            rollout_pred_raw = rollout_pred_raw.clone()
                            rollout_pred_raw[..., 6] = rollout_pred_raw[..., 6].clamp(0.0, 1.0)
                            rollout_pred_raw[..., 13] = rollout_pred_raw[..., 13].clamp(0.0, 1.0)
                            qpos_err_norm_pred = (
                                rollout_pred_raw[:, -1, :] - qpos_mean_local.view(1, -1)
                            ) / qpos_std_local.view(1, -1)
                            action_dev_norm_batch.index_copy_(0, pred_index_tensor, rollout_pred_chunk)
                            action_dev_raw_batch.index_copy_(0, pred_index_tensor, rollout_pred_raw)
                            qpos_err_norm_batch.index_copy_(0, pred_index_tensor, qpos_err_norm_pred)

                        for i, item in enumerate(prepared_samples):
                            if item["external_action_dev_norm"] is not None:
                                action_dev_norm_batch[i] = item["external_action_dev_norm"]
                            if item["external_action_dev_raw"] is not None:
                                action_dev_raw_batch[i] = item["external_action_dev_raw"]
                            if item["external_qpos_err_norm"] is not None:
                                qpos_err_norm_batch[i] = item["external_qpos_err_norm"]
                            if item["external_action_is_pad_prefix"] is not None:
                                action_is_pad_prefix_batch[i] = item["external_action_is_pad_prefix"]

                        loaded_rollout_latents: list[torch.Tensor | None] = [None] * num_prepared
                        missing_rollout_indices: list[int] = []
                        if args.stage2_latent_cache_dir:
                            for i, item in enumerate(prepared_samples):
                                cache_relpath = item.get("cache_relpath")
                                if not cache_relpath:
                                    missing_rollout_indices.append(i)
                                    continue
                                cache_path = os.path.join(args.stage2_latent_cache_dir, cache_relpath)
                                if os.path.exists(cache_path):
                                    latent_i = torch.load(cache_path, map_location="cpu")
                                    if latent_i.ndim == 4 and latent_i.shape[0] == 1:
                                        latent_i = latent_i[0]
                                    loaded_rollout_latents[i] = latent_i.float()
                                    step_cache_hit += 1
                                else:
                                    if args.stage2_latent_cache_strict:
                                        raise FileNotFoundError(
                                            f"Missing stage2 latent cache entry: {cache_path}"
                                        )
                                    missing_rollout_indices.append(i)
                                    step_cache_miss += 1
                        else:
                            missing_rollout_indices = list(range(num_prepared))

                        if missing_rollout_indices:
                            miss_index_tensor = torch.as_tensor(
                                missing_rollout_indices, dtype=torch.long, device=args.device
                            )
                            t_roll0 = time.perf_counter()
                            rollout_latent_missing = teacher.rollout_latent_from_actions_batch(
                                curr_image=image_batch[:, 0].index_select(0, miss_index_tensor),
                                curr_qpos_raw=qpos_raw_batch.index_select(0, miss_index_tensor),
                                action_prefix_raw=action_dev_raw_batch.index_select(0, miss_index_tensor),
                                raw_data=[prepared_samples[i]["raw_data"] for i in missing_rollout_indices],
                                fk=correction_builder.fk,
                                ddim_steps=args.ddim_steps,
                            ).detach().cpu()
                            step_t_rollout += time.perf_counter() - t_roll0
                            for local_idx, batch_idx in enumerate(missing_rollout_indices):
                                latent_i = rollout_latent_missing[local_idx].float()
                                loaded_rollout_latents[batch_idx] = latent_i
                                cache_relpath = prepared_samples[batch_idx].get("cache_relpath")
                                if (
                                    args.stage2_latent_cache_dir
                                    and args.stage2_latent_cache_writeback
                                    and cache_relpath
                                ):
                                    cache_path = os.path.join(args.stage2_latent_cache_dir, cache_relpath)
                                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                                    torch.save(latent_i.half(), cache_path)

                        rollout_latent_batch = torch.stack(
                            [latent_i for latent_i in loaded_rollout_latents],
                            dim=0,
                        ).to(args.device)

                t_model0 = time.perf_counter()
                out = model(
                    image_t=image_batch,
                    qpos_t=qpos_batch,
                    qpos_raw=qpos_raw_batch,
                    act_action_chunk=act_action_chunk_train,
                    act_is_pad=act_is_pad_train,
                    correction_target_prefix=correction_target_prefix_train,
                    is_pad_prefix=correction_is_pad_prefix_train,
                    wm_teacher=teacher,
                    raw_data=None,
                    fk=correction_builder.fk,
                    norm_stats=norm_stats,
                    global_step=dyn_schedule_value,
                    ddim_steps=args.ddim_steps,
                    retain_weight=retain_weight_cur,
                    bridge_weight=args.bridge_weight,
                    use_act_head_correction=args.use_act_head_correction,
                    base_anchor_chunk=base_anchor_chunk_train,
                    correction_target_chunk=correction_target_chunk_train,
                    correction_is_pad=correction_is_pad_train,
                    external_action_dev_norm=action_dev_norm_batch,
                    external_action_dev_raw=action_dev_raw_batch,
                    external_qpos_err_norm=qpos_err_norm_batch,
                    external_action_is_pad_prefix=action_is_pad_prefix_batch,
                    external_z_wm_sim=rollout_latent_batch,
                    act_like_loss_only=args.act_like_loss_only,
                    mode="stage2",
                )
                step_t_model += time.perf_counter() - t_model0

                loss = out.loss
                loss_correct = out.loss_correct
                loss_dyn = out.loss_dynamics
                loss_retain = out.loss_retain
                loss_bridge = out.loss_bridge
                beta_dyn = float(out.beta_dynamics)
                alpha_latent = float(out.alpha_latent)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                global_step += 1
                sample_count += current_global_batch_size
                steps_this_epoch += 1
                meter_loss += loss.item()
                meter_correct += loss_correct.item()
                meter_dyn += loss_dyn.item()
                meter_retain += loss_retain.item()
                meter_bridge += loss_bridge.item()
                meter_skipped += n_skipped
                meter_retain_weight += retain_weight_cur
                meter_t_raw += step_t_raw
                meter_t_builder += step_t_builder
                meter_t_rollout += step_t_rollout
                meter_t_model += step_t_model
                meter_cache_hit += step_cache_hit
                meter_cache_miss += step_cache_miss

                if is_main_process(rank):
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        correct=f"{loss_correct.item():.4f}",
                        dyn=f"{loss_dyn.item():.4f}",
                        retain=f"{loss_retain.item():.4f}",
                        bridge=f"{loss_bridge.item():.4f}",
                        rw=f"{retain_weight_cur:.3f}",
                        beta=f"{beta_dyn:.4f}",
                        alpha=f"{alpha_latent:.4f}",
                        skipped=n_skipped,
                        step=global_step,
                        sched=f"{sample_count / float(max(1, args.reference_global_batch_size)):.0f}",
                        t_raw=f"{step_t_raw:.2f}s",
                        t_build=f"{step_t_builder:.2f}s",
                        t_roll=f"{step_t_rollout:.2f}s",
                        t_model=f"{step_t_model:.2f}s",
                        cache=f"{step_cache_hit}/{step_cache_hit + step_cache_miss}",
                    )
                log_wandb(
                    wandb_run,
                    {
                        "global_step": global_step,
                        "schedule_step": sample_count / float(max(1, args.reference_global_batch_size)),
                        "sample_count": sample_count,
                        "base_global_batch_size": base_global_batch_size,
                        "correction_global_batch_size": corr_global_batch_size,
                        "global_batch_size": current_global_batch_size,
                        "epoch": epoch,
                        "train_loss_step": loss.item(),
                        "train_correct_step": loss_correct.item(),
                        "train_dynamics_step": loss_dyn.item(),
                        "train_retain_step": loss_retain.item(),
                        "train_bridge_step": loss_bridge.item(),
                        "retain_weight_current": retain_weight_cur,
                        "beta_dynamics": beta_dyn,
                        "alpha_latent": alpha_latent,
                        "batch_skipped_samples": n_skipped,
                        "step_time_raw_sec": step_t_raw,
                        "step_time_builder_sec": step_t_builder,
                        "step_time_rollout_sec": step_t_rollout,
                        "step_time_model_sec": step_t_model,
                        "step_rollout_cache_hit": step_cache_hit,
                        "step_rollout_cache_miss": step_cache_miss,
                    },
                    step=global_step,
                )
                if is_main_process(rank):
                    write_heartbeat(
                        args.output_dir,
                        {
                            "timestamp": time.time(),
                            "status": "running",
                            "epoch": epoch,
                            "global_step": global_step,
                            "sample_count": sample_count,
                            "loss": float(loss.item()),
                            "correct": float(loss_correct.item()),
                            "dynamics": float(loss_dyn.item()),
                            "retain": float(loss_retain.item()),
                            "bridge": float(loss_bridge.item()),
                            "retain_weight_current": float(retain_weight_cur),
                            "beta_dynamics": float(beta_dyn),
                            "alpha_latent": float(alpha_latent),
                            "skipped": int(n_skipped),
                            "step_time_raw_sec": float(step_t_raw),
                            "step_time_builder_sec": float(step_t_builder),
                            "step_time_rollout_sec": float(step_t_rollout),
                            "step_time_model_sec": float(step_t_model),
                            "step_rollout_cache_hit": int(step_cache_hit),
                            "step_rollout_cache_miss": int(step_cache_miss),
                        },
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            n = max(1, steps_this_epoch)
            epoch_time = time.time() - start
            if distributed:
                stats_tensor = torch.tensor(
                    [
                        meter_loss,
                        meter_correct,
                        meter_dyn,
                        meter_retain,
                        meter_bridge,
                        meter_skipped,
                        meter_retain_weight,
                        meter_t_raw,
                        meter_t_builder,
                        meter_t_rollout,
                        meter_t_model,
                        meter_cache_hit,
                        meter_cache_miss,
                        float(corr_stats["generated"]),
                        float(corr_stats["triggered"]),
                        float(corr_stats["fallback"]),
                        float(corr_stats["branch_interp_nearest"]),
                        float(corr_stats["branch_gripper_close"]),
                        float(corr_stats["branch_translation"]),
                        float(corr_stats["branch_rotation"]),
                        float(corr_stats["branch_planner"]),
                        float(corr_stats["branch_other"]),
                        float(corr_stats["branch_timeout"]),
                        float(corr_stats["branch_error"]),
                        float(steps_this_epoch),
                    ],
                    device=args.device,
                    dtype=torch.float64,
                )
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                meter_loss = float(stats_tensor[0].item())
                meter_correct = float(stats_tensor[1].item())
                meter_dyn = float(stats_tensor[2].item())
                meter_retain = float(stats_tensor[3].item())
                meter_bridge = float(stats_tensor[4].item())
                meter_skipped = float(stats_tensor[5].item())
                meter_retain_weight = float(stats_tensor[6].item())
                meter_t_raw = float(stats_tensor[7].item())
                meter_t_builder = float(stats_tensor[8].item())
                meter_t_rollout = float(stats_tensor[9].item())
                meter_t_model = float(stats_tensor[10].item())
                meter_cache_hit = float(stats_tensor[11].item())
                meter_cache_miss = float(stats_tensor[12].item())
                corr_stats["generated"] = int(stats_tensor[13].item())
                corr_stats["triggered"] = int(stats_tensor[14].item())
                corr_stats["fallback"] = int(stats_tensor[15].item())
                corr_stats["branch_interp_nearest"] = int(stats_tensor[16].item())
                corr_stats["branch_gripper_close"] = int(stats_tensor[17].item())
                corr_stats["branch_translation"] = int(stats_tensor[18].item())
                corr_stats["branch_rotation"] = int(stats_tensor[19].item())
                corr_stats["branch_planner"] = int(stats_tensor[20].item())
                corr_stats["branch_other"] = int(stats_tensor[21].item())
                corr_stats["branch_timeout"] = int(stats_tensor[22].item())
                corr_stats["branch_error"] = int(stats_tensor[23].item())
                n = max(1, int(stats_tensor[24].item()))
            corr_generated = int(corr_stats["generated"])
            corr_triggered = int(corr_stats["triggered"])
            corr_fallback = int(corr_stats["fallback"])
            cache_total = int(meter_cache_hit + meter_cache_miss)
            cache_hit_rate = (meter_cache_hit / cache_total) if cache_total > 0 else 0.0
            corr_trigger_rate = (corr_triggered / corr_generated) if corr_generated > 0 else 0.0
            corr_fallback_ratio = (corr_fallback / corr_generated) if corr_generated > 0 else 0.0
            corr_branch_counts = {
                "interp_nearest": int(corr_stats["branch_interp_nearest"]),
                "gripper_close": int(corr_stats["branch_gripper_close"]),
                "translation": int(corr_stats["branch_translation"]),
                "rotation": int(corr_stats["branch_rotation"]),
                "planner": int(corr_stats["branch_planner"]),
                "other": int(corr_stats["branch_other"]),
                "timeout": int(corr_stats["branch_timeout"]),
                "error": int(corr_stats["branch_error"]),
            }
            if is_main_process(rank):
                print(
                    f"[stage2 epoch {epoch}] "
                    f"loss={meter_loss/n:.4f} correct={meter_correct/n:.4f} "
                    f"dyn={meter_dyn/n:.4f} retain={meter_retain/n:.4f} bridge={meter_bridge/n:.4f} "
                    f"retain_w={meter_retain_weight/n:.4f} "
                    f"t_raw={meter_t_raw/n:.2f}s t_build={meter_t_builder/n:.2f}s "
                    f"t_roll={meter_t_rollout/n:.2f}s t_model={meter_t_model/n:.2f}s "
                    f"cache={int(meter_cache_hit)}/{cache_total} ({cache_hit_rate:.1%}) "
                    f"corr={corr_generated} trig={corr_triggered} ({corr_trigger_rate:.1%}) "
                    f"fb={corr_fallback} ({corr_fallback_ratio:.1%}) "
                    f"branch={json.dumps(corr_branch_counts, ensure_ascii=True)} "
                    f"time={epoch_time:.1f}s",
                    flush=True,
                )
            log_wandb(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "epoch_loss": meter_loss / n,
                    "epoch_correct": meter_correct / n,
                    "epoch_dynamics": meter_dyn / n,
                    "epoch_retain": meter_retain / n,
                    "epoch_bridge": meter_bridge / n,
                    "epoch_retain_weight": meter_retain_weight / n,
                    "epoch_time_raw_sec": meter_t_raw / n,
                    "epoch_time_builder_sec": meter_t_builder / n,
                    "epoch_time_rollout_sec": meter_t_rollout / n,
                    "epoch_time_model_sec": meter_t_model / n,
                    "epoch_rollout_cache_hit": meter_cache_hit,
                    "epoch_rollout_cache_miss": meter_cache_miss,
                    "epoch_rollout_cache_hit_rate": cache_hit_rate,
                    "epoch_skipped_samples": meter_skipped,
                    "epoch_correction_generated": corr_generated,
                    "epoch_correction_triggered": corr_triggered,
                    "epoch_correction_trigger_rate": corr_trigger_rate,
                    "epoch_correction_fallback": corr_fallback,
                    "epoch_correction_fallback_ratio": corr_fallback_ratio,
                    "epoch_correction_branch_interp_nearest": corr_branch_counts["interp_nearest"],
                    "epoch_correction_branch_gripper_close": corr_branch_counts["gripper_close"],
                    "epoch_correction_branch_translation": corr_branch_counts["translation"],
                    "epoch_correction_branch_rotation": corr_branch_counts["rotation"],
                    "epoch_correction_branch_planner": corr_branch_counts["planner"],
                    "epoch_correction_branch_other": corr_branch_counts["other"],
                    "epoch_correction_branch_timeout": corr_branch_counts["timeout"],
                    "epoch_correction_branch_error": corr_branch_counts["error"],
                    "epoch_time_sec": epoch_time,
                    "global_step": global_step,
                    "schedule_step": sample_count / float(max(1, args.reference_global_batch_size)),
                    "sample_count": sample_count,
                },
                step=global_step,
            )
            if is_main_process(rank):
                write_heartbeat(
                    args.output_dir,
                    {
                        "timestamp": time.time(),
                        "status": "epoch_end",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "sample_count": sample_count,
                        "epoch_loss": meter_loss / n,
                        "epoch_correct": meter_correct / n,
                        "epoch_dynamics": meter_dyn / n,
                        "epoch_retain": meter_retain / n,
                        "epoch_bridge": meter_bridge / n,
                        "epoch_retain_weight": meter_retain_weight / n,
                        "epoch_time_raw_sec": meter_t_raw / n,
                        "epoch_time_builder_sec": meter_t_builder / n,
                        "epoch_time_rollout_sec": meter_t_rollout / n,
                        "epoch_time_model_sec": meter_t_model / n,
                        "epoch_skipped_samples": meter_skipped,
                        "epoch_correction_generated": corr_generated,
                        "epoch_correction_triggered": corr_triggered,
                        "epoch_correction_trigger_rate": corr_trigger_rate,
                        "epoch_correction_fallback": corr_fallback,
                        "epoch_correction_fallback_ratio": corr_fallback_ratio,
                        "epoch_correction_branch_counts": corr_branch_counts,
                        "epoch_time_sec": epoch_time,
                    },
                )

            if is_main_process(rank) and (epoch + 1) % args.save_freq == 0:
                raw_model = model.module if isinstance(model, DDP) else model
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "sample_count": sample_count,
                    "schedule_step": sample_count / float(max(1, args.reference_global_batch_size)),
                    "global_batch_size": current_global_batch_size,
                    "world_size": world_size,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "norm_stats": norm_stats,
                    "args": vars(args),
                }
                save_path = os.path.join(args.output_dir, f"stage2_epoch_{epoch+1:04d}.pt")
                torch.save(ckpt, save_path)
                print(f"[stage2] saved: {save_path}", flush=True)
                update_wandb_summary(
                    wandb_run,
                    {
                        "last_checkpoint": save_path,
                        "last_epoch": epoch + 1,
                        "last_global_step": global_step,
                    },
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"[stage2] reached max_steps={args.max_steps}, stopping early", flush=True)
                break
        if is_main_process(rank):
            write_heartbeat(
                args.output_dir,
                {
                    "timestamp": time.time(),
                    "status": "finished",
                    "epoch": epoch + 1 if "epoch" in locals() else start_epoch,
                    "global_step": global_step,
                    "sample_count": sample_count,
                },
            )
    finally:
        finish_wandb(wandb_run)
        cleanup_distributed()


if __name__ == "__main__":
    main()
