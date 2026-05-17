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
from policy.SmolVLA.latentcorr.correction_episode_io import (
    export_correction_raw_episode,
    export_correction_state_action_sample,
    init_export_episode_id,
)
from policy.SmolVLA.latentcorr.evac_interface import EvacLatentTeacher
from policy.SmolVLA.latentcorr.correction_policy_adapter import SampleBoundSmolVLAAdapter
from policy.SmolVLA.latentcorr.failure_utils import active_arm_pattern_id_to_key, error_mode_id_to_key, phase_id_to_key
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
    if hasattr(dataset, "_explore_unit_target_trials"):
        target_trials = list(getattr(dataset, "_explore_unit_target_trials", []))
        if target_trials:
            num_units = int(len(target_trials))
            full_loops = int(completed_units) // num_units
            rem_units = int(completed_units) % num_units
            return (
                int(full_loops) * int(sum(int(x) for x in target_trials))
                + int(sum(int(x) for x in target_trials[:rem_units]))
                + int(current_trials)
            )
    return int(completed_units) * int(k_per_unit) + int(current_trials)


def get_explore_total_target_samples(dataset, k_per_unit: int) -> int:
    if hasattr(dataset, "_explore_total_target_samples"):
        return int(getattr(dataset, "_explore_total_target_samples"))
    return int(get_explore_num_units(dataset)) * int(k_per_unit)


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
    trials_tmp = trials_path.with_name(trials_path.name + ".tmp")
    with open(trials_tmp, "w", encoding="utf-8") as f:
        json.dump(failure_trials, f, indent=2, ensure_ascii=False)
    os.replace(trials_tmp, trials_path)
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
    meta_path = task_output_dir / f"failure_meta_rank{int(rank):02d}.json"
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(meta_tmp, meta_path)


def write_corr_export_live_files(task_output_dir: Path, args, epoch: int, rank: int, export_records: list[dict]) -> None:
    task_output_dir.mkdir(parents=True, exist_ok=True)
    records_path = task_output_dir / f"correction_export_live_rank{int(rank):02d}.json"
    tmp_path = records_path.with_name(records_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(export_records, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, records_path)
    shard_suffix = (
        f"{args.corr_export_task_config}_state_action_shards"
        if str(args.corr_export_format) == "state_action"
        else f"{args.corr_export_task_config}_shards"
    )
    meta = {
        "version": 1,
        "mode": "corr_export_meta_live",
        "epoch": int(epoch),
        "rank": int(rank),
        "world_size": int(args.world_size),
        "corr_export_root": str(args.corr_export_root),
        "corr_export_task_config": str(args.corr_export_task_config),
        "corr_export_format": str(args.corr_export_format),
        "corr_export_shard_suffix": str(shard_suffix),
        "corr_export_max_loops": int(args.corr_export_max_loops),
        "failure_explore_k": int(args.failure_explore_k),
        "explore_trial_source_dir": str(getattr(args, "explore_trial_source_dir", "")).strip(),
    }
    meta_path = task_output_dir / f"correction_export_meta_rank{int(rank):02d}.json"
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(meta_tmp, meta_path)


def _load_json_list(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in resume live file: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def load_resume_live_records(resume_root: Path, task_only: str, rank: int) -> tuple[list[dict], list[dict]]:
    task_dir = Path(resume_root) / "failure_explore" / str(task_only)
    failure_trials = _load_json_list(task_dir / f"failure_trials_live_rank{int(rank):02d}.json")
    export_records = _load_json_list(task_dir / f"correction_export_live_rank{int(rank):02d}.json")
    return failure_trials, export_records


def load_trial_source_records(source_root: Path, task_only: str, rank: int) -> list[dict]:
    task_dir = Path(source_root) / "failure_explore" / str(task_only)
    source_path = task_dir / f"failure_trials_live_rank{int(rank):02d}.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Trial source file not found: {source_path}")
    records = _load_json_list(source_path)
    valid_records = []
    for idx, record in enumerate(records):
        if "episode_id" not in record or "start_ts" not in record:
            raise ValueError(f"Trial source record #{idx} missing episode_id/start_ts in {source_path}")
        valid_records.append(dict(record))
    return valid_records


def apply_explore_resume_progress(dataset, completed_samples: int) -> tuple[int, int]:
    """Advance explore unit counters using live trial count from a previous run."""
    completed_samples = int(max(0, int(completed_samples)))
    target_trials = list(getattr(dataset, "_explore_unit_target_trials", []))
    num_units = int(len(target_trials))
    if num_units <= 0 or completed_samples <= 0:
        return 0, 0

    completed_units = 0
    remaining = int(completed_samples)
    guard = 0
    max_guard = max(1, num_units * max(1, completed_samples + 1))
    while remaining > 0 and guard < max_guard:
        unit_idx = int(completed_units) % num_units
        target = int(max(0, int(target_trials[unit_idx])))
        if target <= 0:
            completed_units += 1
            guard += 1
            continue
        if remaining >= target:
            remaining -= target
            completed_units += 1
            guard += 1
            continue
        break

    dataset._explore_completed_unit_count = int(completed_units)
    dataset._explore_curr_unit_idx = int(completed_units % num_units)
    dataset._explore_trial_count_local = int(max(0, remaining))
    dataset._explore_seen_samples = set()
    return int(completed_units), int(max(0, remaining))


def sanitize_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or "unknown"


def shuffle_episode_ids_for_rank(episode_ids: list[int], seed: int, rank: int, task_name: str) -> list[int]:
    ids = [int(ep) for ep in episode_ids]
    if len(ids) <= 1:
        return ids
    task_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(task_name)))
    rng = np.random.default_rng(int(seed) + 9176 * int(task_hash) + 1009 * int(rank))
    perm = rng.permutation(len(ids))
    return [ids[int(idx)] for idx in perm]


def parse_compare_backends(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = []
        for item in list(value):
            raw_items.extend(str(item).replace(";", ",").split(","))
    out: set[str] = set()
    for item in raw_items:
        backend = str(item).strip().lower()
        if not backend:
            continue
        if backend not in {"evac", "cosmos"}:
            raise ValueError(f"Unsupported compare backend={backend!r}; expected evac/cosmos")
        out.add(backend)
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("explore smolvla latentcorr")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smolvla_pretrained_path", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--latent_dataset_source", type=str, default=os.environ.get("SMOLVLA_LATENT_DATA_SOURCE", ""))
    parser.add_argument(
        "--latent_dataset_root",
        type=str,
        default=os.environ.get("SMOLVLA_LATENT_DATA_ROOT", "/data/zhenyangfan/RoboTwin/policy/SmolVLA/data"),
    )
    parser.add_argument("--latent_dataset_suffix", type=str, default=os.environ.get("SMOLVLA_LATENT_DATA_SUFFIX", ""))
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
    parser.add_argument("--explore_phase_keys", nargs="*", default=[])
    parser.add_argument("--explore_error_modes", nargs="*", default=[])
    parser.add_argument("--explore_disable_phase_bin_skip", type=str2bool, default=False)
    parser.add_argument("--explore_skip_open_laptop_transport", type=str2bool, default=False)
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, required=True)
    parser.add_argument("--planner_orient_weight", type=float, required=True)
    parser.add_argument("--planner_gripper_penalty", type=float, required=True)
    parser.add_argument("--planner_nearest_window_radius", type=int, required=True)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, required=True)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_save_video", type=str2bool, required=True)
    parser.add_argument("--save_perturb_rollout_video", type=str2bool, required=True)
    parser.add_argument("--save_correction_debug", type=str2bool, default=False)
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
    parser.add_argument("--evac_blur_filter_metric", type=str, choices=["sharpness_ratio", "grad_cosine", "mode_aware"], default="sharpness_ratio")
    parser.add_argument("--evac_blur_filter_min_ratio", type=float, default=0.75)
    parser.add_argument("--evac_blur_filter_region", type=str, choices=["full_image", "active_gripper_patch"], default="active_gripper_patch")
    parser.add_argument("--evac_blur_filter_patch_pad_px", type=int, default=12)
    parser.add_argument("--evac_blur_filter_gripper_axis_m", type=float, default=0.04)
    parser.add_argument("--evac_use_dual_cache", type=str2bool, default=False)
    parser.add_argument("--evac_dc_v_bounds", nargs="*", type=int, default=[])
    parser.add_argument("--evac_dc_budget", type=float, default=-1.0)
    parser.add_argument("--evac_dc_enc_start", type=int, default=999)
    parser.add_argument("--evac_dc_replay_step_noise", type=str2bool, default=False)
    parser.add_argument("--evac_dc_hf_metric", type=str2bool, default=False)
    parser.add_argument("--evac_dc_v_blur_on_reuse", type=str2bool, default=False)
    parser.add_argument("--evac_dc_v_blur_kernel", type=int, default=3)
    parser.add_argument("--evac_dc_v_blur_strength", type=float, default=0.15)
    parser.add_argument("--world_model_backend", type=str, choices=["evac", "cosmos", "sim"], default="evac")
    parser.add_argument("--sim_use_subprocess", type=str2bool, default=True)
    parser.add_argument("--sim_timeout_s", type=float, default=180.0)
    parser.add_argument("--world_model_quality_record", type=str, default="")
    parser.add_argument("--world_model_quality_backend", type=str, choices=["evac", "cosmos", "sim"], default="")
    parser.add_argument("--cosmos_execution_mode", type=str, choices=["worker", "direct"], default="direct")
    parser.add_argument("--cosmos_root", type=str, default="/data/zhenyangfan/cosmos-predict2.5")
    parser.add_argument("--cosmos_python_bin", type=str, default="/data/zhenyangfan/cosmos-predict2.5/.venv/bin/python")
    parser.add_argument("--cosmos_checkpoint_path", type=str, default="")
    parser.add_argument("--cosmos_experiment", type=str, default="robotwin_dualarm_actioncond_2b_256_320")
    parser.add_argument(
        "--cosmos_config_file",
        type=str,
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--cosmos_cuda_visible_devices", type=str, default="")
    parser.add_argument("--cosmos_context_parallel_size", type=int, default=1)
    parser.add_argument("--cosmos_chunk_size", type=int, default=12)
    parser.add_argument("--cosmos_guidance", type=int, default=7)
    parser.add_argument("--cosmos_resolution", type=str, default="256,320")
    parser.add_argument("--cosmos_fps_downsample_ratio", type=int, default=1)
    parser.add_argument("--cosmos_gripper_scale", type=float, default=1.0)
    parser.add_argument("--cosmos_invert_gripper", type=str2bool, default=True)
    parser.add_argument("--cosmos_num_steps", type=int, default=35)
    parser.add_argument("--cosmos_save_fps", type=int, default=30)
    parser.add_argument("--cosmos_num_latent_conditional_frames", type=int, default=1)
    parser.add_argument("--cosmos_action_scaler", type=float, default=20.0)
    parser.add_argument("--cosmos_action_stats_path", type=str, default="")
    parser.add_argument("--cosmos_action_normalization_clip", type=str, default="")
    parser.add_argument("--cosmos_use_quat", type=str2bool, default=False)
    parser.add_argument("--cosmos_quat_input_order", type=str, choices=["wxyz", "xyzw"], default="xyzw")
    parser.add_argument("--cosmos_prompt", type=str, default="")
    parser.add_argument("--cosmos_negative_prompt", type=str, default="")
    parser.add_argument("--cosmos_seed", type=int, default=0)
    parser.add_argument("--cosmos_work_dir", type=str, default="")
    parser.add_argument("--cosmos_startup_timeout_s", type=float, default=600.0)
    parser.add_argument("--cosmos_request_timeout_s", type=float, default=900.0)
    parser.add_argument("--world_model_compare_mode", type=str2bool, default=False)
    parser.add_argument("--world_model_compare_backends", nargs="*", default=[])
    parser.add_argument("--world_model_compare_sim", type=str2bool, default=True)
    parser.add_argument("--world_model_compare_sim_autorun", type=str2bool, default=False)
    parser.add_argument("--world_model_compare_sim_timeout_s", type=float, default=600.0)
    parser.add_argument("--world_model_compare_cosmos_autorun", type=str2bool, default=False)
    parser.add_argument("--corr_export_dataset", type=str2bool, default=False)
    parser.add_argument("--corr_export_root", type=str, default="/data/zhenyangfan/RoboTwin/data")
    parser.add_argument("--corr_export_task_config", type=str, default="demo_clean_corr_export")
    parser.add_argument("--corr_export_format", type=str, choices=["raw_episode", "state_action"], default="raw_episode")
    parser.add_argument("--corr_export_max_loops", type=int, default=1)
    parser.add_argument("--corr_export_debug_video", type=str2bool, default=True)
    parser.add_argument("--corr_export_fps", type=int, default=30)
    parser.add_argument("--explore_trial_source_dir", type=str, default="")
    parser.add_argument("--resume_from_live_dir", type=str, default="")
    parser.add_argument(
        "--explore_disable_distributed_progress_sync",
        type=str2bool,
        default=None,
        help="Disable NCCL all_gather progress sync. Defaults to true for sim backend.",
    )
    parser.add_argument(
        "--debug_max_samples_per_rank",
        type=int,
        default=int(os.environ.get("EXPLORE_DEBUG_MAX_SAMPLES_PER_RANK", os.environ.get("EXPLORE_DEBUG_MAX_SAMPLES", "0"))),
        help="Stop each rank after this many explore samples. 0 disables the debug limit.",
    )
    parser.add_argument("--fail_fast_on_error", type=str2bool, default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not str(args.world_model_quality_backend).strip():
        backend = str(args.world_model_backend).strip().lower()
        # Keep trial filtering aligned with the original EVAC-filtered runs by
        # default.  The rollout backend can still be switched to sim/cosmos
        # independently.
        args.world_model_quality_backend = "evac" if backend == "sim" else backend
    if args.explore_disable_distributed_progress_sync is None:
        args.explore_disable_distributed_progress_sync = bool(
            str(args.world_model_backend).strip().lower() == "sim"
        )
    trial_source_dir = str(getattr(args, "explore_trial_source_dir", "")).strip()
    accelerator = build_accelerator()
    if args.rank is None:
        args.rank = int(accelerator.process_index)
    if args.world_size is None:
        args.world_size = int(accelerator.num_processes)
    if str(args.device).startswith("cuda"):
        args.device = str(accelerator.device)
    compare_backends = (
        parse_compare_backends(args.world_model_compare_backends)
        if bool(args.world_model_compare_mode)
        else set()
    )
    if bool(args.world_model_compare_mode) and not compare_backends:
        compare_backends = {"evac", "cosmos"}
    need_cosmos_backend = bool(
        str(args.world_model_backend).strip().lower() == "cosmos"
        or ("cosmos" in compare_backends and bool(args.world_model_compare_cosmos_autorun))
    )
    need_evac_backend = bool(
        str(args.world_model_backend).strip().lower() == "evac"
        or "evac" in compare_backends
    )
    if need_cosmos_backend and not str(args.cosmos_cuda_visible_devices).strip():
        local_cuda_index = int(getattr(args, "rank", 0) or 0)
        if local_cuda_index < 0:
            local_cuda_index = 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
        args.cosmos_cuda_visible_devices = (
            visible_ids[local_cuda_index] if 0 <= local_cuda_index < len(visible_ids) else str(local_cuda_index)
        )
    device_obj = torch.device(args.device)
    if torch.cuda.is_available() and device_obj.type == "cuda":
        if device_obj.index is None:
            local_rank = int(
                os.environ.get(
                    "LOCAL_RANK",
                    getattr(accelerator.state, "local_process_index", accelerator.local_process_index),
                )
            )
            device_obj = torch.device("cuda", local_rank)
            args.device = str(device_obj)
        torch.cuda.set_device(device_obj)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    if str(args.latent_dataset_source).strip():
        os.environ["SMOLVLA_LATENT_DATA_SOURCE"] = str(args.latent_dataset_source).strip()
    if str(args.latent_dataset_root).strip():
        os.environ["SMOLVLA_LATENT_DATA_ROOT"] = str(Path(args.latent_dataset_root).resolve())
    if str(args.latent_dataset_suffix).strip():
        os.environ["SMOLVLA_LATENT_DATA_SUFFIX"] = str(args.latent_dataset_suffix).strip()

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    if (
        need_cosmos_backend
        and str(args.cosmos_execution_mode).strip().lower() != "direct"
        and not str(args.cosmos_work_dir).strip()
    ):
        mode_dir = "cosmos_workers"
        args.cosmos_work_dir = str(output_dir / mode_dir / f"rank{int(args.rank):02d}")
    if need_cosmos_backend and not str(args.cosmos_checkpoint_path).strip():
        raise ValueError("--cosmos_checkpoint_path is required when using Cosmos as main or compare backend")
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
        failure_mode="off",
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
        explore_phase_keys=list(args.explore_phase_keys),
        explore_error_modes=list(args.explore_error_modes),
        explore_disable_phase_bin_skip=bool(args.explore_disable_phase_bin_skip),
        explore_skip_open_laptop_transport=bool(args.explore_skip_open_laptop_transport),
        world_model_quality_record=str(args.world_model_quality_record),
        world_model_quality_backend=str(args.world_model_quality_backend),
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
        load_rollout_model=need_evac_backend,
    )
    init_teacher_latent = teacher.encode_image(
        init_sample["image_t"][0].unsqueeze(0).to(device=device_obj, dtype=torch.float32)
    )
    model.initialize_from_batch(init_batch, teacher_latent=init_teacher_latent)
    if str(args.stage1_ckpt).strip():
        stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(stage1_ckpt["model"], strict=True)
    model.to(device_obj)
    model.eval()
    shared_evac_model = teacher.model
    shared_evac_config = teacher.cfg
    if str(args.world_model_backend).strip().lower() in {"cosmos", "sim"} and not need_evac_backend:
        del teacher
        shared_evac_model = None
        shared_evac_config = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=args.device,
        shared_evac_model=shared_evac_model,
        shared_evac_config=shared_evac_config,
    )

    if accelerator.is_main_process:
        with open(output_dir / "explore_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    world_size = int(max(1, int(args.world_size)))
    failure_explore_k_global = int(max(1, int(args.failure_explore_k)))
    failure_explore_k_local = int(max(1, (failure_explore_k_global + world_size - 1) // world_size))
    corr_export_max_loops = int(max(1, int(args.corr_export_max_loops)))
    debug_max_samples_per_rank = int(max(0, int(args.debug_max_samples_per_rank)))

    task_run_stats: dict[str, dict] = {}
    task_order: list[str] = []
    total_rank_samples = 0
    for spec in task_specs:
        task_only, _ = parse_task_parts(spec.task_name)
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
            explore_phase_keys=list(args.explore_phase_keys),
            explore_error_modes=list(args.explore_error_modes),
            explore_disable_phase_bin_skip=bool(args.explore_disable_phase_bin_skip),
            explore_skip_open_laptop_transport=bool(args.explore_skip_open_laptop_transport),
            world_model_quality_record=str(args.world_model_quality_record),
            world_model_quality_backend=str(args.world_model_quality_backend),
        )
        full_episode_ids = [int(ep) for ep in dataset.episode_ids]
        global_units = int(get_explore_num_units(dataset))
        global_target_samples = int(get_explore_total_target_samples(dataset, int(failure_explore_k_local)))
        local_episode_ids = shuffle_episode_ids_for_rank(
            full_episode_ids,
            seed=int(args.seed),
            rank=int(args.rank),
            task_name=spec.task_name,
        )
        task_only, _ = parse_task_parts(spec.task_name)
        dataset.set_episode_ids(local_episode_ids)
        dataset.set_explore_local_k(int(failure_explore_k_local))
        trial_source_records: list[dict] = []
        if trial_source_dir:
            trial_source_records = load_trial_source_records(Path(trial_source_dir), task_only, int(args.rank))
            dataset.set_explore_replay_trials(trial_source_records)
            dataset.set_explore_local_k(1)
        total_units = get_explore_num_units(dataset)
        task_order.append(task_only)
        task_samples = int(get_explore_total_target_samples(dataset, int(failure_explore_k_local)))
        if trial_source_dir:
            total_units = int(len(trial_source_records))
            task_samples = int(len(trial_source_records))
        if bool(args.corr_export_dataset) and not trial_source_records:
            task_samples *= int(corr_export_max_loops)
        total_rank_samples += task_samples
        task_run_stats[task_only] = {
            "num_explore_units_local": int(total_units),
            "num_explore_units_global": int(global_units),
            "num_episode_ids_local": int(len(local_episode_ids)),
            "num_episode_ids_global": int(len(full_episode_ids)),
            "total_samples_local": int(task_samples),
            "total_samples_global": int(global_target_samples),
        }

    disable_progress_sync = bool(getattr(args, "explore_disable_distributed_progress_sync", False))
    rank_totals = (
        [int(total_rank_samples)]
        if disable_progress_sync
        else sync_rank_sample_totals(int(total_rank_samples), device_obj)
    )
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
        "corr_export_dataset": bool(args.corr_export_dataset),
        "corr_export_root": str(args.corr_export_root),
        "corr_export_task_config": str(args.corr_export_task_config),
        "corr_export_format": str(args.corr_export_format),
        "corr_export_shard_suffix": (
            f"{args.corr_export_task_config}_state_action_shards"
            if str(args.corr_export_format) == "state_action"
            else f"{args.corr_export_task_config}_shards"
        ),
        "corr_export_max_loops": int(corr_export_max_loops),
        "explore_trial_source_dir": trial_source_dir,
        "world_model_quality_record": str(args.world_model_quality_record),
        "world_model_quality_backend": str(args.world_model_quality_backend),
        "total_samples_local": int(total_rank_samples),
        "task_outputs": {},
    }

    task_contexts: list[dict] = []
    total_units_all = int(sum(int(task_run_stats[task_only]["num_explore_units_local"]) for task_only in task_order))

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
            explore_phase_keys=list(args.explore_phase_keys),
            explore_error_modes=list(args.explore_error_modes),
            explore_disable_phase_bin_skip=bool(args.explore_disable_phase_bin_skip),
            explore_skip_open_laptop_transport=bool(args.explore_skip_open_laptop_transport),
            world_model_quality_record=str(args.world_model_quality_record),
            world_model_quality_backend=str(args.world_model_quality_backend),
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
            full_episode_ids,
            seed=int(args.seed),
            rank=int(args.rank),
            task_name=spec.task_name,
        )
        dataset.set_episode_ids(local_episode_ids)
        dataset.set_explore_local_k(int(failure_explore_k_local))
        trial_source_records: list[dict] = []
        if trial_source_dir:
            trial_source_records = load_trial_source_records(Path(trial_source_dir), task_only, int(args.rank))
            dataset.set_explore_replay_trials(trial_source_records)
            dataset.set_explore_local_k(1)
        total_units = int(task_run_stats[task_only]["num_explore_units_local"])
        task_total_samples = int(task_run_stats[task_only]["total_samples_local"])
        if trial_source_dir:
            total_units = int(len(trial_source_records))
        failure_trials: list[dict] = []
        export_records: list[dict] = []
        epoch = 0
        processed_samples = 0
        local_task_done = bool(total_units <= 0)
        if str(args.resume_from_live_dir).strip():
            loaded_failure_trials, loaded_export_records = load_resume_live_records(
                Path(args.resume_from_live_dir),
                task_only,
                int(args.rank),
            )
            if loaded_failure_trials:
                failure_trials = loaded_failure_trials
                export_records = loaded_export_records
                processed_samples = int(len(failure_trials))
                completed_units, partial_trials = apply_explore_resume_progress(dataset, processed_samples)
                if int(partial_trials) > 0:
                    # Avoid reusing the same unit with an empty seen set. This resume path is intended
                    # for the common k_global == world_size case, but stays conservative otherwise.
                    dataset._advance_explore_unit()
                    completed_units = int(get_explore_completed_unit_count(dataset))
                if completed_units >= int(total_units) * int(corr_export_max_loops):
                    local_task_done = True

        if total_units == 0:
            write_failure_live_files(task_output_dir, args, epoch, int(args.rank), failure_trials)
            explore_manifest["task_outputs"][task_only] = {
                "task_name": spec.task_name,
                "task_output_dir": str(task_output_dir),
                "num_failure_trials": 0,
                "num_correction_exports": 0,
                "num_explore_units_local": 0,
                "num_explore_units_global": int(task_run_stats[task_only]["num_explore_units_global"]),
                "num_episode_ids_local": int(task_run_stats[task_only]["num_episode_ids_local"]),
                "num_episode_ids_global": int(task_run_stats[task_only]["num_episode_ids_global"]),
                "total_samples_local": 0,
                "total_samples_global": int(task_run_stats[task_only]["total_samples_global"]),
            }
            continue

        task_contexts.append(
            {
                "spec": spec,
                "task_only": task_only,
                "task_config": task_config,
                "task_output_dir": task_output_dir,
                "raw_cache": raw_cache,
                "dataset": dataset,
                "norm_stats": norm_stats,
                "dataloader": dataloader,
                "dataloader_iter": iter(dataloader),
                "failure_trials": failure_trials,
                "export_records": export_records,
                "export_episode_id_next": int(
                    init_export_episode_id(
                        Path(args.corr_export_root)
                        / task_only
                        / (
                            f"{args.corr_export_task_config}_state_action_shards"
                            if str(args.corr_export_format) == "state_action"
                            else f"{args.corr_export_task_config}_shards"
                        )
                        / f"rank{int(args.rank):02d}"
                        / ("correction_data" if str(args.corr_export_format) == "state_action" else "data")
                    )
                ),
                "epoch": int(epoch),
                "processed_samples": int(processed_samples),
                "local_task_done": bool(local_task_done),
                "base_total_units": int(total_units),
                "total_units": (
                    int(total_units)
                    if bool(trial_source_records)
                    else int(total_units) * int(corr_export_max_loops)
                ),
                "task_total_samples": int(task_total_samples),
                "trial_source_replay": bool(trial_source_records),
            }
        )

    def _compute_local_task_progress() -> tuple[int, int]:
        current_samples = 0
        completed_units = 0
        for ctx in task_contexts:
            if bool(ctx["local_task_done"]):
                current_samples += int(ctx["task_total_samples"])
                completed_units += int(ctx["total_units"])
            else:
                current_samples += int(
                    get_explore_completed_sample_count(ctx["dataset"], int(failure_explore_k_local))
                )
                completed_units += int(get_explore_completed_unit_count(ctx["dataset"]))
        return int(current_samples), int(completed_units)

    all_task_done = bool(len(task_contexts) == 0)
    while not all_task_done:
        all_task_done = True
        for ctx in task_contexts:
            if bool(ctx["local_task_done"]):
                continue
            all_task_done = False
            if interrupted["flag"]:
                raise KeyboardInterrupt
            try:
                batch = next(ctx["dataloader_iter"])
            except StopIteration:
                ctx["epoch"] = int(ctx["epoch"]) + 1
                ctx["dataloader_iter"] = iter(ctx["dataloader"])
                try:
                    batch = next(ctx["dataloader_iter"])
                except StopIteration:
                    ctx["local_task_done"] = True
                    continue
            task_only = str(ctx["task_only"])
            task_config = str(ctx["task_config"])
            spec = ctx["spec"]
            task_output_dir = ctx["task_output_dir"]
            raw_cache = ctx["raw_cache"]
            dataset = ctx["dataset"]
            norm_stats = ctx["norm_stats"]
            failure_trials = ctx["failure_trials"]
            export_records = ctx["export_records"]
            ctx["processed_samples"] = int(ctx["processed_samples"]) + 1
            processed_samples = int(ctx["processed_samples"])
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
            if (
                bool(args.save_correction_debug)
                or bool(args.save_perturb_rollout_video)
                or bool(args.recover_eval_save_video)
                or bool(args.world_model_compare_mode)
            ):
                phase_bin_id = int(batch["sampled_phase_bin_id"][0].item())
                sample_debug_dir = (
                    task_output_dir
                    / "debug_wm"
                    / f"rank{int(args.rank):02d}"
                    / f"step_{processed_samples - 1:06d}"
                    / f"bi{phase_bin_id}"
                )
            correction_builder.cfg.compare_task_name_full = str(spec.task_name)
            correction_builder.cfg.compare_episode_id = int(episode_id)
            correction_builder.cfg.compare_rank = int(args.rank)
            correction_builder.cfg.compare_global_step = int(processed_samples - 1)
            correction_builder.cfg.compare_raw_data_dir = str(spec.raw_data_dir)
            correction_builder.cfg.compare_repo_root = str(PROJECT_ROOT)
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
                original_active_arm_pattern_id=int(batch["original_active_arm_pattern_id"][0].item()),
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
                if not error_mode_key:
                    error_mode_key = error_mode_id_to_key(int(batch["forced_error_mode_id"][0].item()))
                invalid_trial = bool(corr_meta.get("invalid_trial", False))
                if corr_meta.get("skip_reason") and recoverable_raw is None:
                    invalid_trial = True
                recover_eval_error = str(recover_eval_last.get("error", "")).strip()
                if recover_eval_error and recoverable_raw is None:
                    invalid_trial = True
                invalid_reason = str(corr_meta.get("invalid_reason", "")).strip()
                if invalid_trial and not invalid_reason:
                    invalid_reason = str(corr_meta.get("skip_reason", "")).strip()
                    if not invalid_reason and recover_eval_error:
                        invalid_reason = "recover_eval_error"
                if (
                    (recoverable_raw is not None or invalid_trial)
                    and error_mode_key in {"translation", "rotation", "gripper_close"}
                ):
                    phase_key = str(corr_meta.get("sampled_phase_key", "")).strip().lower()
                    if not phase_key:
                        phase_key = phase_id_to_key(int(batch["sampled_phase_id"][0].item()))
                    active_arm_pattern = str(corr_meta.get("sampled_active_arm_pattern", "")).strip().lower()
                    if active_arm_pattern not in {"left_arm", "right_arm"}:
                        active_arm_pattern = active_arm_pattern_id_to_key(
                            int(batch["sampled_active_arm_pattern_id"][0].item())
                        )
                    original_active_arm_pattern = active_arm_pattern_id_to_key(
                        int(batch["original_active_arm_pattern_id"][0].item())
                    )
                    if not phase_key:
                        raise ValueError("corr_meta missing sampled_phase_key during explore")
                    failure_trials.append(
                        {
                            "epoch": int(ctx["epoch"]),
                            "global_step": int(len(failure_trials)),
                            "episode_id": episode_id,
                            "start_ts": start_ts,
                            "phase_key": phase_key,
                            "phase_instance_idx": int(batch["sampled_phase_instance_id"][0].item()),
                            "phase_bin_id": int(batch["sampled_phase_bin_id"][0].item()),
                            "error_mode": error_mode_key,
                            "active_arm_pattern": active_arm_pattern,
                            "original_active_arm_pattern": original_active_arm_pattern,
                            "dir_bin_id": int(batch["forced_dir_bin_id"][0].item()),
                            "mag_bin_id": int(batch["forced_mag_bin_id"][0].item()),
                            "recoverable": bool(recoverable_raw) if recoverable_raw is not None else False,
                            "invalid_trial": bool(invalid_trial),
                            "invalid_reason": invalid_reason if invalid_trial else None,
                            "recover_eval_mode": recover_eval_last.get("mode"),
                            "recover_eval_metric_name": recover_eval_last.get("metric_name"),
                            "recover_eval_metric": recover_eval_last.get("metric"),
                            "recover_eval_threshold": recover_eval_last.get("threshold"),
                            "recover_eval_metrics": recover_eval_last.get("metrics"),
                            "recover_eval_thresholds": recover_eval_last.get("thresholds"),
                            "recover_eval_passes": recover_eval_last.get("passes"),
                            "recover_eval_failed_thresholds": recover_eval_last.get("failed_thresholds"),
                            "skip_reason": corr_meta.get("skip_reason"),
                            "skip_error": corr_meta.get("skip_error"),
                            "skip_traceback": corr_meta.get("skip_traceback"),
                            "evac_blur_filter": corr_meta.get("evac_blur_filter"),
                        }
                    )
                if (
                    bool(args.corr_export_dataset)
                    and isinstance(correction, dict)
                    and correction.get("corr_image") is not None
                    and correction.get("corr_qpos_norm") is not None
                    and correction.get("corr_action_chunk_norm") is not None
                    and bool(corr_meta.get("correction_generated", False))
                    and recoverable_raw is False
                ):
                    corr_action_raw = corr_meta.get("corr_action_chunk_raw")
                    if corr_action_raw is None:
                        action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32)
                        action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32)
                        corr_action_raw = (
                            correction["corr_action_chunk_norm"].detach().cpu() * action_std.view(1, -1)
                            + action_mean.view(1, -1)
                        ).numpy().astype(np.float32)
                    corr_qpos_raw = corr_meta.get("corr_qpos_raw")
                    if corr_qpos_raw is None:
                        qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32)
                        qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32)
                        corr_qpos_raw = (
                            correction["corr_qpos_norm"].detach().cpu() * qpos_std + qpos_mean
                        ).numpy().astype(np.float32)
                    attach_idx = corr_meta.get("attach_idx")
                    if attach_idx is None:
                        attach_idx = int(start_ts)
                    if str(args.corr_export_format) == "state_action":
                        export_record = export_correction_state_action_sample(
                            export_root=Path(args.corr_export_root),
                            task_name=task_only,
                            task_config=str(args.corr_export_task_config),
                            rank=int(args.rank),
                            sample_id=int(ctx["export_episode_id_next"]),
                            source_raw_data_dir=Path(spec.raw_data_dir),
                            source_episode_id=int(episode_id),
                            start_ts=int(start_ts),
                            corr_action_raw=np.asarray(corr_action_raw, dtype=np.float32),
                            corr_qpos_raw=np.asarray(corr_qpos_raw, dtype=np.float32),
                            corr_image=correction["corr_image"][0].detach().cpu(),
                            corr_meta=corr_meta,
                            corr_action_norm=correction["corr_action_chunk_norm"].detach().cpu().numpy(),
                            corr_qpos_norm=correction["corr_qpos_norm"].detach().cpu().numpy(),
                        )
                    else:
                        export_record = export_correction_raw_episode(
                            export_root=Path(args.corr_export_root),
                            task_name=task_only,
                            task_config=str(args.corr_export_task_config),
                            rank=int(args.rank),
                            episode_id=int(ctx["export_episode_id_next"]),
                            source_raw_data_dir=Path(spec.raw_data_dir),
                            source_episode_id=int(episode_id),
                            start_ts=int(start_ts),
                            attach_idx=int(attach_idx),
                            corr_action_raw=np.asarray(corr_action_raw, dtype=np.float32),
                            corr_qpos_raw=np.asarray(corr_qpos_raw, dtype=np.float32),
                            corr_image=correction["corr_image"][0].detach().cpu(),
                            corr_meta=corr_meta,
                            fk=correction_builder.fk,
                            fps=int(args.corr_export_fps),
                            write_debug_video=bool(args.corr_export_debug_video),
                            debug_video_dir=(
                                None
                                if sample_debug_dir is None
                                else Path(sample_debug_dir) / "correction_export_debug_video"
                            ),
                        )
                    export_record["loop_id"] = int(get_explore_completed_unit_count(dataset)) // max(
                        1, int(ctx["base_total_units"])
                    )
                    export_record["explore_unit_idx"] = int(batch["sampled_explore_unit_idx"][0].item())
                    export_records.append(export_record)
                    ctx["export_episode_id_next"] = int(ctx["export_episode_id_next"]) + 1

            dataset.record_explore_trial(
                int(batch["sampled_explore_unit_idx"][0].item()),
                episode_id,
                start_ts,
            )
            completed_units = int(get_explore_completed_unit_count(dataset))
            if completed_units >= int(ctx["total_units"]):
                ctx["local_task_done"] = True
            overall_current_samples_local, overall_completed_units_local = _compute_local_task_progress()
            if debug_max_samples_per_rank > 0 and int(overall_current_samples_local) >= debug_max_samples_per_rank:
                for stop_ctx in task_contexts:
                    stop_ctx["local_task_done"] = True
                overall_current_samples_local, overall_completed_units_local = _compute_local_task_progress()
            local_all_done = bool(all(bool(item["local_task_done"]) for item in task_contexts))
            if disable_progress_sync:
                explore_progress = [
                    {
                        "rank": int(args.rank),
                        "current_samples": int(overall_current_samples_local),
                        "total_samples": int(total_rank_samples),
                        "done": bool(local_all_done),
                        "completed_units": int(overall_completed_units_local),
                        "total_units": int(total_units_all),
                        "unit_idx": int(get_current_explore_unit_idx(dataset)),
                        "trial_count": int(get_current_explore_trial_count(dataset)),
                    }
                ]
            else:
                explore_progress = sync_all_explore_status(
                    local_current_samples=int(overall_current_samples_local),
                    local_total_samples=int(total_rank_samples),
                    local_done=bool(local_all_done),
                    local_completed_units=int(overall_completed_units_local),
                    local_total_units=int(total_units_all),
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
            write_failure_live_files(task_output_dir, args, int(ctx["epoch"]), int(args.rank), failure_trials)
            if bool(args.corr_export_dataset):
                write_corr_export_live_files(
                    task_output_dir,
                    args,
                    int(ctx["epoch"]),
                    int(args.rank),
                    export_records,
                )

        all_task_done = bool(all(bool(item["local_task_done"]) for item in task_contexts))

    for ctx in task_contexts:
        task_only = str(ctx["task_only"])
        spec = ctx["spec"]
        failure_trials = ctx["failure_trials"]
        export_records = ctx["export_records"]
        corr_export_shard_suffix = (
            f"{args.corr_export_task_config}_state_action_shards"
            if str(args.corr_export_format) == "state_action"
            else f"{args.corr_export_task_config}_shards"
        )
        explore_manifest["task_outputs"][task_only] = {
            "task_name": spec.task_name,
            "task_output_dir": str(ctx["task_output_dir"]),
            "num_failure_trials": len(failure_trials),
            "num_correction_exports": len(export_records),
            "correction_export_shard_dir": str(
                Path(args.corr_export_root)
                / task_only
                / corr_export_shard_suffix
                / f"rank{int(args.rank):02d}"
            ),
            "num_explore_units_local": int(ctx["total_units"]),
            "num_explore_units_global": int(task_run_stats[task_only]["num_explore_units_global"]),
            "num_episode_ids_local": int(task_run_stats[task_only]["num_episode_ids_local"]),
            "num_episode_ids_global": int(task_run_stats[task_only]["num_episode_ids_global"]),
            "total_samples_local": int(ctx["task_total_samples"]),
            "total_samples_global": int(task_run_stats[task_only]["total_samples_global"]),
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
