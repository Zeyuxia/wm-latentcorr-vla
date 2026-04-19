from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SMOLVLA_SRC_DIR = THIS_DIR / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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


def shard_explore_units(units: list[dict], rank: int, world_size: int) -> list[dict]:
    if world_size <= 1:
        return [dict(item) for item in units]
    return [dict(item) for idx, item in enumerate(units) if idx % int(world_size) == int(rank)]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("explore smolvla latentcorr")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smolvla_pretrained_path", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--instruction_type", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--curobo_left_yml", type=str, required=True)
    parser.add_argument("--curobo_right_yml", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
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
    parser.add_argument("--sample_skip_head_ratio", type=float, required=True)
    parser.add_argument("--start_margin", type=int, required=True)
    parser.add_argument("--failure_phase_bins", type=int, required=True)
    parser.add_argument("--failure_translation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_translation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_explore_k", type=int, required=True)
    parser.add_argument("--max_rollout_steps", type=int, required=True)
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, required=True)
    parser.add_argument("--planner_target_mode", type=str, required=True)
    parser.add_argument("--planner_target_lookahead_steps", type=int, required=True)
    parser.add_argument("--planner_orient_weight", type=float, required=True)
    parser.add_argument("--planner_gripper_penalty", type=float, required=True)
    parser.add_argument("--planner_nearest_window_radius", type=int, required=True)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, required=True)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, required=True)
    parser.add_argument("--act_aligned_min_dist_fallback_force_correction", type=str2bool, required=True)
    parser.add_argument("--act_aligned_min_dist_recover_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_real_error_trigger_enable", type=str2bool, required=True)
    parser.add_argument("--act_aligned_real_error_min_dist_thresh", type=float, required=True)
    parser.add_argument("--act_aligned_real_error_min_dist_delta_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_enable", type=str2bool, required=True)
    parser.add_argument("--recover_eval_save_video", type=str2bool, required=True)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, required=True)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, required=True)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, required=True)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, required=True)
    parser.add_argument("--act_aligned_correction_interp_nearest_enable", type=str2bool, required=True)
    parser.add_argument("--act_aligned_correction_interp_prefix_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_correction_planner_prefix_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_correction_gripper_close_prefix_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_correction_compose_gt_tail_enable", type=str2bool, required=True)
    parser.add_argument("--act_aligned_correction_gripper_switch_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_recover_gripper_penalty", type=float, required=True)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_prob", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_error_mode", type=str, required=True)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_close_prob", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_translation_prob", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_rotation_prob", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_mag_random", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_mag_rand_min", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_mag_rand_max", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_reject_sampling_enable", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_reject_max_trials", type=int, required=True)
    parser.add_argument("--act_aligned_perturb_reject_dir_jitter_eps", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_open_max", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_fast_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_sample_pregrasp_phase_window_len", type=int, required=True)
    parser.add_argument("--act_aligned_sample_timeout_sec", type=float, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    base_policy = SmolVLAPolicy.from_pretrained(args.smolvla_pretrained_path)
    base_policy.to(torch.device(args.device))
    preprocess, _ = make_smolvla_processors(base_policy, args.smolvla_pretrained_path)

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
        sample_skip_head_ratio=float(args.sample_skip_head_ratio),
        start_margin=int(args.start_margin),
        failure_mode="explore",
        failure_table_path="",
        failure_phase_bins=int(args.failure_phase_bins),
        failure_translation_dir_bins=int(args.failure_translation_dir_bins),
        failure_translation_mag_bins=int(args.failure_translation_mag_bins),
        failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
        failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
        failure_explore_k=int(args.failure_explore_k),
    )
    init_sample = init_dataset[0]
    init_task_only, init_task_config = parse_task_parts(init_spec.task_name)
    model.initialize_from_batch(
        SampleBoundSmolVLAAdapter(
            latent_policy=model,
            preprocess=preprocess,
            task_name=init_task_only,
            task_config=init_task_config,
            episode_id=int(init_sample["episode_id"]),
            instruction_type=args.instruction_type,
        ).build_batch(
            image_t=init_sample["image_t"],
            qpos_raw=init_sample["qpos_raw"],
            action_chunk_raw=init_sample["act_action_chunk_raw"],
        )
    )
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    model.load_state_dict(stage1_ckpt["model"], strict=True)
    model.to(torch.device(args.device))
    model.eval()

    teacher = EvacLatentTeacher(evac_ckpt=args.evac_ckpt, evac_config=args.evac_config, device=args.device)
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=args.device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )

    if int(args.rank) == 0:
        with open(output_dir / "explore_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    explore_manifest = {
        "version": 1,
        "output_dir": str(output_dir),
        "multi_task_names": list(args.multi_task_names),
        "rank": int(args.rank),
        "world_size": int(args.world_size),
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
            sample_skip_head_ratio=float(args.sample_skip_head_ratio),
            start_margin=int(args.start_margin),
            failure_mode="explore",
            failure_table_path="",
            failure_phase_bins=int(args.failure_phase_bins),
            failure_translation_dir_bins=int(args.failure_translation_dir_bins),
            failure_translation_mag_bins=int(args.failure_translation_mag_bins),
            failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
            failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
            failure_explore_k=int(args.failure_explore_k),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        full_units = [dict(item) for item in dataset._explore_units]
        local_units = shard_explore_units(full_units, rank=int(args.rank), world_size=int(args.world_size))
        dataset.set_explore_units(local_units)
        failure_explore_k_local = int(max(1, (int(args.failure_explore_k) + int(args.world_size) - 1) // int(args.world_size)))
        dataset.set_explore_local_k(failure_explore_k_local)
        total_units = len(local_units)
        failure_trials: list[dict] = []
        progress = tqdm(total=max(1, total_units), desc=f"explore {spec.task_name}", leave=True)
        seen_completed = 0
        epoch = 0

        if total_units == 0:
            write_failure_live_files(task_output_dir, args, epoch, int(args.rank), failure_trials)
            progress.close()
            explore_manifest["task_outputs"][task_only] = {
                "task_name": spec.task_name,
                "task_output_dir": str(task_output_dir),
                "num_failure_trials": 0,
                "num_explore_units_local": 0,
                "num_explore_units_global": int(len(full_units)),
            }
            continue

        while get_explore_completed_unit_count(dataset) < total_units:
            for batch in dataloader:
                episode_id = int(batch["episode_id"][0].item())
                start_ts = int(batch["start_ts"][0].item())
                adapter = SampleBoundSmolVLAAdapter(
                    latent_policy=model,
                    preprocess=preprocess,
                    task_name=task_only,
                    task_config=task_config,
                    episode_id=episode_id,
                    instruction_type=args.instruction_type,
                )
                if episode_id not in raw_cache:
                    raw_cache[episode_id] = load_raw_episode(spec.raw_data_dir, episode_id)
                correction = correction_builder.build(
                    latent_model=adapter,
                    image_t=batch["image_t"][0].to(args.device),
                    qpos_t=batch["qpos_t"][0].to(args.device),
                    raw_data=raw_cache[episode_id],
                    norm_stats=norm_stats,
                    start_ts=start_ts,
                    failure_mode_override="explore",
                    sampled_phase_id=int(batch["sampled_phase_id"][0].item()),
                    pregrasp_seg_start=int(batch["pregrasp_seg_start"][0].item()),
                    pregrasp_seg_end=int(batch["pregrasp_seg_end"][0].item()),
                    sampled_phase_bin_id=int(batch["sampled_phase_bin_id"][0].item()),
                    sampled_phase_instance_id=int(batch["sampled_phase_instance_id"][0].item()),
                    forced_error_mode_id=int(batch["forced_error_mode_id"][0].item()),
                    sampled_active_arm_pattern_id=int(batch["sampled_active_arm_pattern_id"][0].item()),
                    forced_dir_bin_id=int(batch["forced_dir_bin_id"][0].item()),
                    forced_mag_bin_id=int(batch["forced_mag_bin_id"][0].item()),
                    sampled_mode_prob=float(batch["sampled_mode_prob"][0].item()),
                    sampled_entry_prob_within_mode=float(batch["sampled_entry_prob_within_mode"][0].item()),
                    sampled_unit_prob=float(batch["sampled_unit_prob"][0].item()),
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
                    if recoverable_raw is not None and error_mode_key in {"translation", "rotation", "gripper_close"}:
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
                                "recoverable": bool(recoverable_raw),
                                "recover_eval_mode": recover_eval_last.get("mode"),
                                "recover_eval_metric_name": recover_eval_last.get("metric_name"),
                                "recover_eval_metric": recover_eval_last.get("metric"),
                                "recover_eval_threshold": recover_eval_last.get("threshold"),
                            }
                        )

                dataset.record_explore_trial(
                    int(batch["sampled_explore_unit_idx"][0].item()),
                    episode_id,
                    start_ts,
                )
                completed_units = get_explore_completed_unit_count(dataset)
                if completed_units > seen_completed:
                    progress.update(completed_units - seen_completed)
                    seen_completed = completed_units
                progress.set_postfix_str(
                    f"unit={get_current_explore_unit_idx(dataset)+1}/{total_units} "
                    f"trial={get_current_explore_trial_count(dataset)} "
                    f"records={len(failure_trials)}"
                )
                write_failure_live_files(task_output_dir, args, epoch, int(args.rank), failure_trials)
                if completed_units >= total_units:
                    break
            epoch += 1
        progress.close()
        explore_manifest["task_outputs"][task_only] = {
            "task_name": spec.task_name,
            "task_output_dir": str(task_output_dir),
            "num_failure_trials": len(failure_trials),
            "num_explore_units_local": int(total_units),
            "num_explore_units_global": int(len(full_units)),
        }

    with open(output_dir / f"explore_manifest_rank{int(args.rank):02d}.json", "w", encoding="utf-8") as f:
        json.dump(explore_manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
