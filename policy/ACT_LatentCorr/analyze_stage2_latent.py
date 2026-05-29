#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import h5py
import torch

from .correction_planner import PlannerCorrectionBuilder, PlannerCorrectionConfig
from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .evac_interface import EvacLatentTeacher
from .latent_policy import ACTLatentStage1
from .train_stage1_latent import _build_act_args, _resolve_dataset_info
from .utils_latent import (
    build_stage1_dataloader,
    choose_eval_start_indices,
    list_valid_episode_ids,
    load_processed_episode_window,
    load_raw_episode,
    resolve_raw_data_dir,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Analyze ACT latent stage-2 minimal closed loop")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--camera_names", nargs="+", default=None)
    parser.add_argument("--evac_ckpt", type=str, default=None)
    parser.add_argument("--evac_config", type=str, default=None)
    parser.add_argument("--raw_data_dir", type=str, default=None)
    parser.add_argument("--urdf_path", type=str, required=True)
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
    parser.add_argument("--correction_interp_nearest_enable", type=str2bool, default=False)
    parser.add_argument("--correction_interp_prefix_ratio", type=float, default=0.4)
    parser.add_argument("--num_eval_episodes", type=int, default=4)
    parser.add_argument("--samples_per_episode", type=int, default=2)
    parser.add_argument("--episode_ids", nargs="+", type=int, default=None)
    parser.add_argument("--ddim_steps", type=int, default=None)
    return parser


def _make_output_dir(output_dir: str | None) -> str:
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/analysis_stage2/{ts}"
    os.makedirs(path, exist_ok=True)
    return path


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    mask = (~is_pad).unsqueeze(-1).float()
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    mask = (~is_pad).unsqueeze(-1).float().expand_as(pred)
    diff = (pred - target).pow(2) * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def _build_model_from_ckpt(
    ckpt_args: dict,
    camera_names: list[str],
    device: str,
) -> ACTLatentStage1:
    latent_model_cfg = LatentModelConfig(
        projector_mid_channels=ckpt_args["projector_mid_channels"],
        wm_adapter_mid_channels=ckpt_args.get("wm_adapter_mid_channels", 128),
        readout_adapter_mid_channels=ckpt_args.get("readout_adapter_mid_channels", 128),
        predictor_num_blocks=ckpt_args["predictor_num_blocks"],
        predictor_mlp_hidden=ckpt_args["predictor_mlp_hidden"],
        action_decoder_hidden=ckpt_args["action_decoder_hidden"],
        action_dim=ckpt_args["action_dim"],
        prefix_steps=ckpt_args["prefix_steps"],
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_align=ckpt_args["lambda_align"],
        beta_dynamics_max=ckpt_args["beta_dynamics_max"],
        lambda_wm_action_current=ckpt_args.get("lambda_wm_action_current", 0.5),
        lambda_wm_action_future=ckpt_args.get("lambda_wm_action_future", 1.0),
        lambda_bridge_future=ckpt_args.get("lambda_bridge_future", 0.25),
        use_projector_detach_for_predictor=True,
        use_projector_detach_for_action_decoder=True,
    )
    warmup_cfg = DynamicsWarmupConfig(
        zero_steps=ckpt_args["dyn_zero_steps"],
        ramp_steps=ckpt_args["dyn_ramp_steps"],
        max_weight=1.0,
        curve=ckpt_args["dyn_warmup_curve"],
    )
    act_args = _build_act_args(camera_names, argparse.Namespace(**ckpt_args))
    model = ACTLatentStage1(
        act_args=act_args,
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
    ).to(device)
    return model


def _summarize(records: list[dict]) -> dict:
    def _stats(key: str) -> dict[str, float]:
        vals = [float(item[key]) for item in records]
        return {
            "mean": float(sum(vals) / max(1, len(vals))),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    teacher_wins = sum(1 for item in records if item["teacher_action_l1"] < item["base_action_l1"])
    bridge_wins = sum(1 for item in records if item["bridge_action_l1"] < item["base_action_l1"])
    source_counts = {}
    for item in records:
        key = str(item.get("correction_source", "unknown"))
        source_counts[key] = int(source_counts.get(key, 0) + 1)
    return {
        "num_samples": len(records),
        "base_action_l1": _stats("base_action_l1"),
        "teacher_action_l1": _stats("teacher_action_l1"),
        "bridge_action_l1": _stats("bridge_action_l1"),
        "teacher_action_mse": _stats("teacher_action_mse"),
        "bridge_action_mse": _stats("bridge_action_mse"),
        "dynamics_mse": _stats("dynamics_mse"),
        "align_mse": _stats("align_mse"),
        "teacher_improvement_l1": _stats("teacher_improvement_l1"),
        "bridge_improvement_l1": _stats("bridge_improvement_l1"),
        "teacher_better_ratio": float(teacher_wins / max(1, len(records))),
        "bridge_better_ratio": float(bridge_wins / max(1, len(records))),
        "correction_source_counts": source_counts,
    }


def main():
    args = build_argparser().parse_args()
    output_dir = _make_output_dir(args.output_dir)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    ckpt_args = dict(ckpt["args"])

    task_name = args.task_name or ckpt_args["task_name"]
    dataset_dir, num_episodes, camera_names = _resolve_dataset_info(task_name)
    if args.dataset_dir is not None:
        dataset_dir = os.path.realpath(args.dataset_dir)
    elif ckpt_args.get("dataset_dir") is not None:
        dataset_dir = os.path.realpath(ckpt_args["dataset_dir"])

    if args.num_episodes is not None:
        num_episodes = int(args.num_episodes)
    elif ckpt_args.get("num_episodes") is not None:
        num_episodes = int(ckpt_args["num_episodes"])

    if args.camera_names is not None:
        camera_names = list(args.camera_names)

    evac_ckpt = args.evac_ckpt or ckpt_args["evac_ckpt"]
    evac_config = args.evac_config or ckpt_args["evac_config"]
    raw_data_dir = resolve_raw_data_dir(task_name, args.raw_data_dir or ckpt_args.get("raw_data_dir"))
    prefix_steps = int(ckpt_args["prefix_steps"])
    act_chunk_size = int(ckpt_args.get("act_chunk_size", prefix_steps))
    future_offset = int(ckpt_args.get("future_offset") or prefix_steps)
    ddim_steps = int(args.ddim_steps or ckpt_args.get("ddim_steps") or 27)

    _, norm_stats = build_stage1_dataloader(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        act_chunk_size=act_chunk_size,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
        batch_size=1,
        num_workers=0,
    )
    if ckpt.get("norm_stats") is not None:
        norm_stats = ckpt["norm_stats"]
    valid_ids = list_valid_episode_ids(dataset_dir, num_episodes, prefix_steps, future_offset)
    if args.episode_ids is not None:
        selected_ids = [int(ep) for ep in args.episode_ids if int(ep) in valid_ids]
    else:
        selected_ids = valid_ids[: args.num_eval_episodes]
    if not selected_ids:
        raise RuntimeError("No valid episodes selected for stage2 analysis")

    teacher = EvacLatentTeacher(evac_ckpt=evac_ckpt, evac_config=evac_config, device=args.device)
    model = _build_model_from_ckpt(ckpt_args, camera_names, args.device)

    with h5py.File(os.path.join(dataset_dir, f"episode_{selected_ids[0]}.hdf5"), "r") as root:
        ep_len = int(root["/action"].shape[0])
    first_start = choose_eval_start_indices(ep_len, prefix_steps, future_offset, args.samples_per_episode)[0]
    first_sample = load_processed_episode_window(
        dataset_dir=dataset_dir,
        episode_id=selected_ids[0],
        camera_names=camera_names,
        norm_stats=norm_stats,
        act_chunk_size=act_chunk_size,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
        start_ts=first_start,
    )
    model.initialize_latent_heads(first_sample["image_t"].unsqueeze(0).to(args.device), teacher)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=args.device)
    action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=args.device)

    planner_cfg = PlannerCorrectionConfig(
        correction_horizon=prefix_steps,
        target_mode=args.planner_target_mode,
        target_lookahead_steps=args.planner_target_lookahead_steps,
        correction_interp_nearest_enable=bool(args.correction_interp_nearest_enable),
        correction_interp_prefix_ratio=args.correction_interp_prefix_ratio,
        correction_gripper_switch_ratio=args.planner_gripper_switch_ratio,
        use_interp_fallback_on_planner_fail=bool(args.planner_interp_fallback),
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
    )
    raw_cache: dict[int, dict] = {}
    records = []
    started = time.time()

    with torch.no_grad():
        for episode_id in selected_ids:
            if episode_id not in raw_cache:
                raw_cache[episode_id] = load_raw_episode(raw_data_dir, episode_id)
            raw_data = raw_cache[episode_id]

            with h5py.File(os.path.join(dataset_dir, f"episode_{episode_id}.hdf5"), "r") as root:
                ep_len = int(root["/action"].shape[0])
            for start_ts in choose_eval_start_indices(ep_len, prefix_steps, future_offset, args.samples_per_episode):
                sample = load_processed_episode_window(
                    dataset_dir=dataset_dir,
                    episode_id=episode_id,
                    camera_names=camera_names,
                    norm_stats=norm_stats,
                    act_chunk_size=act_chunk_size,
                    prefix_steps=prefix_steps,
                    future_offset=future_offset,
                    start_ts=start_ts,
                )

                image_t = sample["image_t"].unsqueeze(0).to(args.device)
                qpos_t = sample["qpos_t"].unsqueeze(0).to(args.device)
                qpos_raw = sample["qpos_raw"].unsqueeze(0).to(args.device)
                is_pad = sample["is_pad_prefix"].unsqueeze(0).to(args.device)
                stage2_ctx = model.prepare_stage2_context(
                    image_t=image_t,
                    qpos_t=qpos_t,
                    qpos_raw=qpos_raw,
                    wm_teacher=teacher,
                    raw_data=raw_data,
                    fk=correction_builder.fk,
                    norm_stats=norm_stats,
                    ddim_steps=ddim_steps,
                    is_pad_prefix=is_pad,
                )
                corr = correction_builder.build(
                    action_dev_raw=stage2_ctx.action_dev_raw[0],
                    raw_data=raw_data,
                    norm_stats=norm_stats,
                    start_ts=start_ts,
                )
                if corr is None:
                    continue

                corr_target = corr["correction_target_norm"].unsqueeze(0).to(args.device)
                corr_is_pad = corr["is_pad"].unsqueeze(0).to(args.device)
                action_base = stage2_ctx.action_dev_norm
                action_teacher = model.decode_action_latent(stage2_ctx.z_wm_sim_shared.detach())
                action_bridge = model.decode_action_latent(stage2_ctx.z_hat_next.detach())

                base_action_l1 = _masked_l1(action_base, corr_target, corr_is_pad)
                teacher_action_l1 = _masked_l1(action_teacher, corr_target, corr_is_pad)
                bridge_action_l1 = _masked_l1(action_bridge, corr_target, corr_is_pad)
                teacher_action_mse = _masked_mse(action_teacher, corr_target, corr_is_pad)
                bridge_action_mse = _masked_mse(action_bridge, corr_target, corr_is_pad)
                dynamics_mse = torch.nn.functional.mse_loss(stage2_ctx.z_hat_next, stage2_ctx.z_wm_sim_shared.detach())
                align_mse = torch.nn.functional.mse_loss(stage2_ctx.z_proj, model._shared_wm_latent(teacher.encode_image(image_t[:, 0]).detach()).detach())

                records.append(
                    {
                        "episode_id": int(episode_id),
                        "start_ts": int(start_ts),
                        "base_action_l1": float(base_action_l1.item()),
                        "teacher_action_l1": float(teacher_action_l1.item()),
                        "bridge_action_l1": float(bridge_action_l1.item()),
                        "teacher_action_mse": float(teacher_action_mse.item()),
                        "bridge_action_mse": float(bridge_action_mse.item()),
                        "dynamics_mse": float(dynamics_mse.item()),
                        "align_mse": float(align_mse.item()),
                        "teacher_improvement_l1": float(base_action_l1.item() - teacher_action_l1.item()),
                        "bridge_improvement_l1": float(base_action_l1.item() - bridge_action_l1.item()),
                        "correction_source": str(corr["meta"]["source"]),
                        "t_star": int(corr["meta"]["t_star"]),
                        "t_target": int(corr["meta"]["t_target"]),
                        "planner_left_status": str(corr["meta"]["planner_left_status"]),
                        "planner_right_status": str(corr["meta"]["planner_right_status"]),
                    }
                )

    summary = _summarize(records)
    summary.update(
        {
            "task_name": task_name,
            "dataset_dir": dataset_dir,
            "raw_data_dir": raw_data_dir,
            "camera_names": camera_names,
            "prefix_steps": prefix_steps,
            "act_chunk_size": act_chunk_size,
            "future_offset": future_offset,
            "ddim_steps": ddim_steps,
            "selected_episode_ids": selected_ids,
            "samples_per_episode": int(args.samples_per_episode),
            "ckpt_path": os.path.realpath(args.ckpt_path),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "elapsed_sec": round(time.time() - started, 3),
        }
    )

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "per_sample.jsonl"), "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Stage-2 最小闭环纠错分析\n")
        f.write(f"任务: {task_name}\n")
        f.write(f"样本数: {summary['num_samples']}\n")
        f.write(f"评估 episode: {selected_ids}\n")
        f.write(f"每个 episode 取样数: {args.samples_per_episode}\n")
        f.write(f"base_action_l1 平均值: {summary['base_action_l1']['mean']:.6f}\n")
        f.write(f"teacher_action_l1 平均值: {summary['teacher_action_l1']['mean']:.6f}\n")
        f.write(f"bridge_action_l1 平均值: {summary['bridge_action_l1']['mean']:.6f}\n")
        f.write(f"dynamics_mse 平均值: {summary['dynamics_mse']['mean']:.6f}\n")
        f.write(f"teacher 改善比例: {summary['teacher_better_ratio']:.4f}\n")
        f.write(f"bridge 改善比例: {summary['bridge_better_ratio']:.4f}\n")
        f.write(f"输出目录: {output_dir}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
