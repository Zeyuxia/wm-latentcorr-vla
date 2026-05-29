#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from policy.ACT.constants import SIM_TASK_CONFIGS

from .act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from .deploy_policy import _build_model_from_ckpt_args
from .evac_interface import EvacLatentTeacher
from .latent_policy import ACTLatentStage1
from .stage2_failure_dataset import build_multitask_failure_table_dataset
from .train_stage1_unified_failure_multitask_latent import _prepare_failure_samples
from .utils_multitask_latent import build_multitask_stage1_dataset, resolve_multitask_specs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_failure_table_paths(ckpt_args: dict, task_names: list[str], cli_paths: list[str] | None) -> list[str]:
    paths = list(cli_paths or ckpt_args.get("failure_table_paths") or [])
    if not paths:
        raise ValueError("failure_table_paths not provided and not found in checkpoint args")
    if len(paths) == 1 and len(task_names) > 1:
        paths = paths * len(task_names)
    if len(paths) != len(task_names):
        raise ValueError(f"Expected 1 or {len(task_names)} failure_table_paths, got {len(paths)}")
    return [os.path.realpath(p) for p in paths]


def _resolve_task(task_names: list[str], task_name_arg: str) -> tuple[str, int]:
    if task_name_arg:
        if task_name_arg not in task_names:
            raise ValueError(f"task_name={task_name_arg!r} not found in checkpoint tasks: {task_names}")
        return task_name_arg, task_names.index(task_name_arg)
    if len(task_names) == 1:
        return task_names[0], 0
    return task_names[0], 0


def _normal_local_indices(dataset, task_idx: int) -> list[int]:
    return [i for i, (ds_task_idx, _) in enumerate(dataset.samples) if int(ds_task_idx) == int(task_idx)]


def _sample_normal_batch(dataset, local_idx: int, device: torch.device) -> dict[str, torch.Tensor]:
    item = dataset[int(local_idx)]
    out: dict[str, torch.Tensor] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0).to(device)
    out["task_name"] = [str(item["task_name"])]
    out["episode_id"] = [int(item["episode_id"])]
    out["start_ts"] = [int(item["start_ts"])]
    out["raw_data_dir"] = [str(item["raw_data_dir"])]
    return out


def _sample_failure_batch(task_dataset, local_idx: int, device: torch.device) -> dict[str, torch.Tensor]:
    item = task_dataset[int(local_idx)]
    out: dict[str, torch.Tensor] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0).to(device)
    out["task_name"] = [str(item["task_name"])]
    out["episode_id"] = [int(item["episode_id"])]
    out["start_ts"] = [int(item["start_ts"])]
    out["raw_data_dir"] = [str(item["raw_data_dir"])]
    return out


def _compute_teacher_and_pred_tokens(
    model: ACTLatentStage1,
    teacher: EvacLatentTeacher,
    image_t: torch.Tensor,
    qpos_t: torch.Tensor,
    future_teacher_latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_act = model._extract_act_feature(image_t)
    z_wm_t = teacher.encode_image(image_t[:, 0])
    target_hw = (z_wm_t.shape[-2], z_wm_t.shape[-1])
    z_proj = model.projector(z_act, target_hw=target_hw)
    z_for_pred = z_proj.detach() if model.latent_loss_cfg.use_projector_detach_for_predictor else z_proj
    if model.latent_loss_cfg.use_raw_wm_targets:
        z_teacher_shared = future_teacher_latent.detach()
    else:
        z_teacher_shared = model._shared_wm_latent(future_teacher_latent.detach())
    teacher_token = model._latent_to_act_token(z_teacher_shared, scale=1.0)
    pred_token = model._predict_future_token(z_for_pred, qpos_t, scale=1.0)
    return teacher_token, pred_token


def _masked_mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def _eval_batch(
    *,
    model: ACTLatentStage1,
    teacher: EvacLatentTeacher,
    image_t: torch.Tensor,
    qpos_t: torch.Tensor,
    act_action_chunk: torch.Tensor,
    act_is_pad: torch.Tensor,
    future_teacher_latent: torch.Tensor,
) -> dict[str, float]:
    teacher_token, pred_token = _compute_teacher_and_pred_tokens(
        model=model,
        teacher=teacher,
        image_t=image_t,
        qpos_t=qpos_t,
        future_teacher_latent=future_teacher_latent,
    )
    with torch.no_grad():
        base_loss = model.base_act(
            qpos_t,
            image_t,
            actions=act_action_chunk,
            is_pad=act_is_pad,
            return_per_sample=True,
        )["loss_per_sample"].mean()
        pred_loss = model.base_act(
            qpos_t,
            image_t,
            actions=act_action_chunk,
            is_pad=act_is_pad,
            return_per_sample=True,
            external_latent_input=pred_token,
        )["loss_per_sample"].mean()
        teacher_loss = model.base_act(
            qpos_t,
            image_t,
            actions=act_action_chunk,
            is_pad=act_is_pad,
            return_per_sample=True,
            external_latent_input=teacher_token,
        )["loss_per_sample"].mean()

        base_action = model.predict_act_chunk(qpos_t, image_t)
        pred_action = model.base_act(qpos_t, image_t, external_latent_input=pred_token)
        teacher_action = model.base_act(qpos_t, image_t, external_latent_input=teacher_token)

    return {
        "base_loss": float(base_loss.item()),
        "pred_loss": float(pred_loss.item()),
        "teacher_loss": float(teacher_loss.item()),
        "token_mse": float(torch.mean((pred_token - teacher_token) ** 2).item()),
        "teacher_token_norm": float(teacher_token.norm(dim=1).mean().item()),
        "pred_token_norm": float(pred_token.norm(dim=1).mean().item()),
        "delta_pred_base": _masked_mean_abs_diff(pred_action, base_action),
        "delta_teacher_base": _masked_mean_abs_diff(teacher_action, base_action),
        "delta_teacher_pred": _masked_mean_abs_diff(teacher_action, pred_action),
    }


def _aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = list(records[0].keys())
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in records]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Offline branch diagnostics for ACT_LatentCorr stage1 checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--failure_table_paths", nargs="*", default=None)
    parser.add_argument("--num_normal_samples", type=int, default=16)
    parser.add_argument("--num_failure_samples", type=int, default=16)
    parser.add_argument("--failure_max_tries_per_sample", type=int, default=12)
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
    failure_table_paths = _resolve_failure_table_paths(ckpt_args, task_names, args.failure_table_paths)

    task_specs = resolve_multitask_specs(task_names, SIM_TASK_CONFIGS)
    shared_camera_names = list(task_specs[0].camera_names)
    normal_dataset, norm_stats = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=int(ckpt_args.get("act_chunk_size", ckpt_args["prefix_steps"])),
        prefix_steps=int(ckpt_args["prefix_steps"]),
        future_offset=int(ckpt_args.get("future_offset", ckpt_args["prefix_steps"])),
    )
    failure_dataset, _ = build_multitask_failure_table_dataset(
        task_specs=task_specs,
        act_chunk_size=int(ckpt_args.get("act_chunk_size", ckpt_args["prefix_steps"])),
        prefix_steps=int(ckpt_args["prefix_steps"]),
        future_offset=int(ckpt_args.get("future_offset", ckpt_args["prefix_steps"])),
        sample_phase_window_len=int(ckpt_args.get("act_aligned_sample_pregrasp_phase_window_len", 30)),
        sample_skip_head_ratio=float(ckpt_args.get("failure_sample_skip_head_ratio", 0.6)),
        start_margin=0,
        failure_mode="train",
        failure_table_paths=failure_table_paths,
        failure_phase_bins=int(ckpt_args.get("failure_phase_bins", 3)),
        failure_translation_dir_bins=int(ckpt_args.get("failure_translation_dir_bins", 6)),
        failure_translation_mag_bins=int(ckpt_args.get("failure_translation_mag_bins", 3)),
        failure_rotation_dir_bins=int(ckpt_args.get("failure_rotation_dir_bins", 6)),
        failure_rotation_mag_bins=int(ckpt_args.get("failure_rotation_mag_bins", 3)),
        failure_explore_k=int(ckpt_args.get("failure_explore_k", 4)),
    )
    failure_dataset.set_norm_stats(norm_stats)

    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=str(device))
    dataset_task_name = ",".join(task_names)
    model = _build_model_from_ckpt_args(ckpt_args, shared_camera_names, str(device), dataset_task_name)
    first = normal_dataset[0]
    model.initialize_latent_heads(first["image_t"].unsqueeze(0).to(device), teacher)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    correction_builder = None
    if int(args.num_failure_samples) > 0:
        correction_cfg = build_act_aligned_cfg_from_args(
            SimpleNamespace(**ckpt_args),
            max_action_len=int(ckpt_args.get("act_chunk_size", 50)),
        )
        correction_cfg.failure_mode = "train"
        correction_builder = ACTAlignedCorrectionBuilder(
            cfg=correction_cfg,
            urdf_path=str(ckpt_args["urdf_path"]),
            curobo_left_yml=str(ckpt_args.get("curobo_left_yml", "/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml")),
            curobo_right_yml=str(ckpt_args.get("curobo_right_yml", "/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml")),
            device=str(device),
            shared_evac_model=teacher.model,
            shared_evac_config=teacher.cfg,
        )

    print(
        f"[branch-metrics] loaded model missing={len(missing)} unexpected={len(unexpected)} "
        f"task={task_name} normal={args.num_normal_samples} failure={args.num_failure_samples}",
        flush=True,
    )

    normal_records: list[dict[str, float]] = []
    normal_indices = _normal_local_indices(normal_dataset, task_idx)
    if not normal_indices:
        raise RuntimeError(f"No normal indices found for task {task_name}")
    with torch.no_grad():
        for _ in range(max(0, int(args.num_normal_samples))):
            batch = _sample_normal_batch(normal_dataset, random.choice(normal_indices), device)
            normal_records.append(
                _eval_batch(
                    model=model,
                    teacher=teacher,
                    image_t=batch["image_t"],
                    qpos_t=batch["qpos_t"],
                    act_action_chunk=batch["act_action_chunk"],
                    act_is_pad=batch["act_is_pad"],
                    future_teacher_latent=teacher.encode_image(batch["image_t_future"][:, 0]),
                )
            )

    failure_records: list[dict[str, float]] = []
    raw_cache: dict[tuple[str, int], dict] = {}
    task_failure_ds = failure_dataset.datasets[task_idx]
    failure_local_indices = list(range(len(task_failure_ds)))
    if failure_local_indices and correction_builder is not None:
        for _ in range(max(0, int(args.num_failure_samples))):
            prepared = []
            for _try in range(max(1, int(args.failure_max_tries_per_sample))):
                failure_batch = _sample_failure_batch(task_failure_ds, random.choice(failure_local_indices), device)
                prepared, _, _, _ = _prepare_failure_samples(
                    batch=failure_batch,
                    raw_model=model,
                    correction_builder=correction_builder,
                    teacher=teacher,
                    norm_stats=norm_stats,
                    raw_cache=raw_cache,
                    args=SimpleNamespace(**ckpt_args),
                )
                if prepared:
                    break
            if not prepared:
                continue
            sample = prepared[0]
            failure_records.append(
                _eval_batch(
                    model=model,
                    teacher=teacher,
                    image_t=sample["image_t"].unsqueeze(0) if sample["image_t"].ndim == 4 else sample["image_t"],
                    qpos_t=sample["qpos_t"].unsqueeze(0) if sample["qpos_t"].ndim == 1 else sample["qpos_t"],
                    act_action_chunk=sample["act_action_chunk"].unsqueeze(0) if sample["act_action_chunk"].ndim == 2 else sample["act_action_chunk"],
                    act_is_pad=sample["act_is_pad"].unsqueeze(0) if sample["act_is_pad"].ndim == 1 else sample["act_is_pad"],
                    future_teacher_latent=sample["future_teacher_latent"].unsqueeze(0)
                    if sample["future_teacher_latent"].ndim == 3
                    else sample["future_teacher_latent"],
                )
            )

    results = {
        "ckpt_path": os.path.realpath(args.ckpt_path),
        "task_name": task_name,
        "num_normal_records": len(normal_records),
        "num_failure_records": len(failure_records),
        "normal": _aggregate(normal_records),
        "failure": _aggregate(failure_records),
    }
    print(json.dumps(results, indent=2), flush=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
