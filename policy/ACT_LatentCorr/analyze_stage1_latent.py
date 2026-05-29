#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import h5py
import torch

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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Analyze ACT latent stage-1 warmup")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--camera_names", nargs="+", default=None)
    parser.add_argument("--raw_data_dir", type=str, default=None)
    parser.add_argument("--urdf_path", type=str, default=None)
    parser.add_argument("--future_teacher_source_override", type=str, default=None, choices=["real", "sim"])
    parser.add_argument("--evac_ckpt", type=str, default=None)
    parser.add_argument("--evac_config", type=str, default=None)
    parser.add_argument("--num_eval_episodes", type=int, default=8)
    parser.add_argument("--samples_per_episode", type=int, default=2)
    parser.add_argument("--episode_ids", nargs="+", type=int, default=None)
    return parser


def _make_output_dir(output_dir: str | None) -> str:
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/analysis_stage1/{ts}"
    os.makedirs(path, exist_ok=True)
    return path


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

    return {
        "num_samples": len(records),
        "loss": _stats("loss"),
        "loss_action": _stats("loss_action"),
        "loss_align": _stats("loss_align"),
        "loss_dynamics": _stats("loss_dynamics"),
        "loss_wm_action_current": _stats("loss_wm_action_current"),
        "loss_wm_action_future": _stats("loss_wm_action_future"),
        "loss_bridge_future": _stats("loss_bridge_future"),
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
    prefix_steps = int(ckpt_args["prefix_steps"])
    act_chunk_size = int(ckpt_args.get("act_chunk_size", prefix_steps))
    future_offset = int(ckpt_args.get("future_offset") or prefix_steps)
    future_teacher_source = str(args.future_teacher_source_override or ckpt_args.get("future_teacher_source", "real"))
    raw_data_dir = None
    fk = None
    raw_cache: dict[int, dict] = {}
    if future_teacher_source == "sim":
        raw_data_dir = resolve_raw_data_dir(task_name, (args.raw_data_dir or ckpt_args.get("raw_data_dir") or None))
        urdf_path = args.urdf_path or ckpt_args.get("urdf_path")
        if not urdf_path:
            raise ValueError("urdf_path is required to analyze a sim-future-teacher checkpoint")
        from policy.ACT.util.fk_sapien import SapienFK

        fk = SapienFK(urdf_path)

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
        raise RuntimeError("No valid episodes selected for stage1 analysis")

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

    records = []
    started = time.time()
    with torch.no_grad():
        for episode_id in selected_ids:
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
                future_teacher_latent = None
                if future_teacher_source == "sim":
                    assert fk is not None
                    if episode_id not in raw_cache:
                        raw_cache[episode_id] = load_raw_episode(raw_data_dir, episode_id)
                    future_teacher_latent = teacher.rollout_latent_from_actions(
                        curr_image=sample["image_t"][0],
                        curr_qpos_raw=sample["qpos_raw"],
                        action_prefix_raw=sample["action_prefix_raw"],
                        raw_data=raw_cache[episode_id],
                        fk=fk,
                        ddim_steps=int(ckpt_args.get("ddim_steps", 27)),
                    ).to(args.device)
                out = model.forward_stage1(
                    image_t=sample["image_t"].unsqueeze(0).to(args.device),
                    image_t1=sample["image_t_future"].unsqueeze(0).to(args.device),
                    qpos_t=sample["qpos_t"].unsqueeze(0).to(args.device),
                    act_action_chunk=sample["act_action_chunk"].unsqueeze(0).to(args.device),
                    act_is_pad=sample["act_is_pad"].unsqueeze(0).to(args.device),
                    action_prefix=sample["action_prefix"].unsqueeze(0).to(args.device),
                    action_future_prefix=sample["action_future_prefix"].unsqueeze(0).to(args.device),
                    is_pad_prefix=sample["is_pad_prefix"].unsqueeze(0).to(args.device),
                    is_pad_future_prefix=sample["is_pad_future_prefix"].unsqueeze(0).to(args.device),
                    wm_teacher=teacher,
                    global_step=int(ckpt.get("global_step", 0)),
                    future_teacher_latent=future_teacher_latent,
                )
                records.append(
                    {
                        "episode_id": int(episode_id),
                        "start_ts": int(start_ts),
                        "loss": float(out.loss.item()),
                        "loss_action": float(out.loss_action.item()),
                        "loss_align": float(out.loss_align.item()),
                        "loss_dynamics": float(out.loss_dynamics.item()),
                        "loss_wm_action_current": float(out.loss_wm_action_current.item()),
                        "loss_wm_action_future": float(out.loss_wm_action_future.item()),
                        "loss_bridge_future": float(out.loss_bridge_future.item()),
                        "beta_dynamics": float(out.beta_dynamics),
                    }
                )

    summary = _summarize(records)
    summary.update(
        {
            "task_name": task_name,
            "dataset_dir": dataset_dir,
            "camera_names": camera_names,
            "prefix_steps": prefix_steps,
            "act_chunk_size": act_chunk_size,
            "future_offset": future_offset,
            "selected_episode_ids": selected_ids,
            "samples_per_episode": int(args.samples_per_episode),
            "ckpt_path": os.path.realpath(args.ckpt_path),
            "future_teacher_source": future_teacher_source,
            "raw_data_dir": raw_data_dir,
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
        f.write("Stage-1 开环潜空间预热分析\n")
        f.write(f"任务: {task_name}\n")
        f.write(f"样本数: {summary['num_samples']}\n")
        f.write(f"评估 episode: {selected_ids}\n")
        f.write(f"每个 episode 取样数: {args.samples_per_episode}\n")
        f.write(f"future teacher 来源: {future_teacher_source}\n")
        f.write(f"loss 平均值: {summary['loss']['mean']:.6f}\n")
        f.write(f"loss_action 平均值: {summary['loss_action']['mean']:.6f}\n")
        f.write(f"loss_align 平均值: {summary['loss_align']['mean']:.6f}\n")
        f.write(f"loss_dynamics 平均值: {summary['loss_dynamics']['mean']:.6f}\n")
        f.write(f"loss_wm_action_current 平均值: {summary['loss_wm_action_current']['mean']:.6f}\n")
        f.write(f"loss_wm_action_future 平均值: {summary['loss_wm_action_future']['mean']:.6f}\n")
        f.write(f"loss_bridge_future 平均值: {summary['loss_bridge_future']['mean']:.6f}\n")
        f.write(f"输出目录: {output_dir}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
