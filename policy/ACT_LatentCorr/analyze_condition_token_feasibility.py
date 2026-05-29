#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from policy.ACT.constants import SIM_TASK_CONFIGS

from .act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .deploy_policy import _build_model_from_ckpt_args
from .evac_interface import EvacLatentTeacher
from .stage2_failure_dataset import build_multitask_failure_table_dataset
from .utils_latent import load_raw_episode
from .utils_multitask_latent import build_multitask_stage1_dataset, resolve_multitask_specs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Analyze whether current input can predict future condition token "
        "under same-task normal/failure sampling."
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--failure_table_paths", nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_normal_samples", type=int, default=-1)
    parser.add_argument("--num_failure_samples", type=int, default=-1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--round_decimals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--failure_max_tries_per_sample", type=int, default=12)
    parser.add_argument("--output_json", type=str, default="")
    return parser


def _mean_pairwise_mse(y: torch.Tensor) -> float:
    if y.shape[0] < 2:
        return 0.0
    idx0 = torch.arange(0, y.shape[0] - 1, device=y.device)
    idx1 = torch.arange(1, y.shape[0], device=y.device)
    return float(((y[idx0] - y[idx1]) ** 2).mean().item())


def _finite_tensor(x: torch.Tensor | None) -> bool:
    return x is not None and bool(torch.isfinite(x).all().item())


def _resolve_target_counts(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_normal_samples >= 0 or args.num_failure_samples >= 0:
        n_normal = max(0, int(args.num_normal_samples))
        n_failure = max(0, int(args.num_failure_samples))
        if n_normal + n_failure <= 0:
            raise ValueError("At least one of num_normal_samples / num_failure_samples must be positive")
        return n_normal, n_failure
    n_total = max(1, int(args.num_samples))
    n_failure = n_total // 2
    n_normal = n_total - n_failure
    return n_normal, n_failure


def _resolve_task(task_names: list[str], task_name_arg: str) -> tuple[str, int]:
    if task_name_arg:
        if task_name_arg not in task_names:
            raise ValueError(f"task_name={task_name_arg!r} not found in checkpoint tasks: {task_names}")
        return task_name_arg, task_names.index(task_name_arg)
    if len(task_names) == 1:
        return task_names[0], 0
    return task_names[0], 0


def _resolve_failure_table_paths(
    ckpt_args: dict,
    task_names: list[str],
    cli_paths: list[str] | None,
) -> list[str]:
    paths = list(cli_paths or ckpt_args.get("failure_table_paths") or [])
    if not paths:
        raise ValueError("failure_table_paths not provided and not found in checkpoint args")
    if len(paths) == 1 and len(task_names) > 1:
        paths = paths * len(task_names)
    if len(paths) != len(task_names):
        raise ValueError(f"Expected 1 or {len(task_names)} failure_table_paths, got {len(paths)}")
    return [os.path.realpath(p) for p in paths]


def _feature_and_token(
    model,
    teacher,
    image_t: torch.Tensor,
    qpos_t: torch.Tensor,
    image_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image_t.ndim == 4:
        image_t = image_t.unsqueeze(0)
    if qpos_t.ndim == 1:
        qpos_t = qpos_t.unsqueeze(0)
    if image_target.ndim == 4:
        image_target = image_target.unsqueeze(0)

    z_act = model._extract_act_feature(image_t)
    z_wm_t = teacher.encode_image(image_t[:, 0])
    target_hw = (z_wm_t.shape[-2], z_wm_t.shape[-1])
    z_proj = model.projector(z_act, target_hw=target_hw)
    pooled = torch.nn.functional.adaptive_avg_pool2d(z_proj, output_size=1).flatten(1)
    feat = torch.cat([pooled, qpos_t], dim=1).squeeze(0).detach().cpu()

    z_wm_target = teacher.encode_image(image_target[:, 0])
    teacher_token = model._latent_to_act_token(z_wm_target, scale=1.0).squeeze(0).detach().cpu()
    return feat, teacher_token


def _collect_normal_samples(
    *,
    dataset,
    task_idx: int,
    num_samples: int,
    model,
    teacher,
    device: torch.device,
) -> list[dict]:
    if num_samples <= 0:
        return []
    local_indices = [i for i, (ds_task_idx, _) in enumerate(dataset.samples) if int(ds_task_idx) == int(task_idx)]
    if not local_indices:
        raise ValueError(f"No normal samples found for task_idx={task_idx}")
    out: list[dict] = []
    with torch.no_grad():
        for _ in range(num_samples):
            item = dataset[int(random.choice(local_indices))]
            feat, teacher_token = _feature_and_token(
                model,
                teacher,
                item["image_t"].to(device),
                item["qpos_t"].to(device),
                item["image_t_future"].to(device),
            )
            out.append(
                {
                    "source": "normal",
                    "task_name": str(item["task_name"]),
                    "episode_id": int(item["episode_id"]),
                    "start_ts": int(item["start_ts"]),
                    "feature": feat,
                    "target": teacher_token,
                }
            )
    return out


def _collect_failure_samples(
    *,
    dataset,
    task_idx: int,
    num_samples: int,
    max_tries_per_sample: int,
    model,
    teacher,
    correction_builder: ACTAlignedCorrectionBuilder,
    norm_stats: dict,
    task_name: str,
    raw_data_dir: str,
    device: torch.device,
) -> list[dict]:
    if num_samples <= 0:
        return []
    local_indices = [i for i in range(len(dataset.datasets[task_idx]))]
    if not local_indices:
        raise ValueError(f"No failure-table entries found for task_idx={task_idx} task={task_name}")
    raw_cache: dict[tuple[str, int], dict] = {}
    out: list[dict] = []
    with torch.no_grad():
        for _ in range(num_samples):
            built = None
            sampled_meta = None
            for _try in range(max(1, int(max_tries_per_sample))):
                item = dataset.datasets[task_idx][int(random.choice(local_indices))]
                ep_id = int(item["episode_id"])
                start_ts = int(item["start_ts"])
                cache_key = (task_name, ep_id)
                if cache_key not in raw_cache:
                    raw_cache[cache_key] = load_raw_episode(raw_data_dir, ep_id)
                corr = correction_builder.build(
                    model,
                    image_t=item["image_t"].to(device),
                    qpos_t=item["qpos_t"].to(device),
                    raw_data=raw_cache[cache_key],
                    norm_stats=norm_stats,
                    start_ts=start_ts,
                    failure_mode_override="train",
                    sampled_phase_id=int(item["sampled_phase_id"]),
                    pregrasp_seg_start=int(item["pregrasp_seg_start"]),
                    pregrasp_seg_end=int(item["pregrasp_seg_end"]),
                    sampled_phase_bin_id=int(item["sampled_phase_bin_id"]),
                    sampled_phase_instance_id=int(item["sampled_phase_instance_id"]),
                    forced_error_mode_id=int(item["forced_error_mode_id"]),
                    sampled_active_arm_pattern_id=int(item["sampled_active_arm_pattern_id"]),
                    forced_dir_bin_id=int(item["forced_dir_bin_id"]),
                    forced_mag_bin_id=int(item["forced_mag_bin_id"]),
                    sampled_mode_prob=None if not torch.isfinite(item["sampled_mode_prob"]) else float(item["sampled_mode_prob"]),
                    sampled_entry_prob_within_mode=None
                    if not torch.isfinite(item["sampled_entry_prob_within_mode"])
                    else float(item["sampled_entry_prob_within_mode"]),
                    sampled_unit_prob=None if not torch.isfinite(item["sampled_unit_prob"]) else float(item["sampled_unit_prob"]),
                )
                if corr is None:
                    continue
                corr_image = corr.get("corr_image")
                corr_qpos_norm = corr.get("corr_qpos_norm")
                if not (_finite_tensor(corr_image) and _finite_tensor(corr_qpos_norm)):
                    continue
                feat, teacher_token = _feature_and_token(
                    model,
                    teacher,
                    corr_image.to(device),
                    corr_qpos_norm.to(device),
                    corr_image.to(device),
                )
                sampled_meta = {
                    "episode_id": ep_id,
                    "start_ts": start_ts,
                    "error_mode": str(corr.get("corr_meta", {}).get("error_mode", "")),
                    "phase_key": str(corr.get("corr_meta", {}).get("phase_key", "")),
                }
                built = {
                    "source": "failure",
                    "task_name": task_name,
                    "episode_id": ep_id,
                    "start_ts": start_ts,
                    "feature": feat,
                    "target": teacher_token,
                    "error_mode": sampled_meta["error_mode"],
                    "phase_key": sampled_meta["phase_key"],
                }
                break
            if built is not None:
                out.append(built)
    return out


def _linear_probe_metrics(x: torch.Tensor, y: torch.Tensor, train_ratio: float) -> dict[str, float | None]:
    if x.shape[0] < 2:
        return {
            "linear_probe_mse": None,
            "mean_baseline_mse": None,
            "linear_probe_gain": None,
        }
    n_train = max(1, int(round(x.shape[0] * float(train_ratio))))
    n_train = min(n_train, x.shape[0] - 1)
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train_n = (x_train - x_mean) / x_std
    x_val_n = (x_val - x_mean) / x_std

    ones_train = torch.ones((x_train_n.shape[0], 1))
    ones_val = torch.ones((x_val_n.shape[0], 1))
    x_train_aug = torch.cat([x_train_n, ones_train], dim=1)
    x_val_aug = torch.cat([x_val_n, ones_val], dim=1)

    weight = torch.linalg.lstsq(x_train_aug, y_train).solution
    y_pred = x_val_aug @ weight
    linear_mse = float(torch.mean((y_pred - y_val) ** 2).item())

    mean_pred = y_train.mean(dim=0, keepdim=True).expand_as(y_val)
    mean_mse = float(torch.mean((mean_pred - y_val) ** 2).item())
    return {
        "linear_probe_mse": linear_mse,
        "mean_baseline_mse": mean_mse,
        "linear_probe_gain": (mean_mse / max(linear_mse, 1e-12)),
    }


def _cross_source_nn_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    sources: list[str],
) -> dict[str, float | int | None]:
    src_to_idx = defaultdict(list)
    for i, src in enumerate(sources):
        src_to_idx[str(src)].append(i)

    def _direction(src_a: str, src_b: str) -> tuple[float | None, float | None, int]:
        idx_a = src_to_idx.get(src_a, [])
        idx_b = src_to_idx.get(src_b, [])
        if not idx_a or not idx_b:
            return None, None, 0
        xa = x[idx_a]
        xb = x[idx_b]
        ya = y[idx_a]
        yb = y[idx_b]
        dist = torch.cdist(xa, xb)
        nn = torch.argmin(dist, dim=1)
        feat_dist = float(torch.mean(dist[torch.arange(dist.shape[0]), nn]).item())
        token_mse = float(torch.mean((ya - yb[nn]) ** 2).item())
        return feat_dist, token_mse, len(idx_a)

    nf_feat, nf_mse, nf_n = _direction("normal", "failure")
    fn_feat, fn_mse, fn_n = _direction("failure", "normal")
    return {
        "normal_to_failure_nn_feature_l2": nf_feat,
        "normal_to_failure_nn_token_mse": nf_mse,
        "normal_to_failure_count": nf_n,
        "failure_to_normal_nn_feature_l2": fn_feat,
        "failure_to_normal_nn_token_mse": fn_mse,
        "failure_to_normal_count": fn_n,
    }


def _duplicate_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    sources: list[str],
    round_decimals: int,
) -> dict[str, float | int]:
    keys = defaultdict(list)
    scaled = 10 ** int(round_decimals)
    for idx in range(x.shape[0]):
        rounded_key = tuple(torch.round(x[idx] * scaled).to(torch.int64).tolist())
        keys[rounded_key].append(idx)

    exact_groups = []
    mixed_groups = []
    for idxs in keys.values():
        if len(idxs) <= 1:
            continue
        exact_groups.append(idxs)
        srcs = {sources[i] for i in idxs}
        if len(srcs) > 1:
            mixed_groups.append(idxs)

    def _group_mse(groups: list[list[int]]) -> float:
        if not groups:
            return 0.0
        vals = []
        for idxs in groups:
            g = y[idxs]
            vals.append(float(((g - g.mean(dim=0, keepdim=True)) ** 2).mean().item()))
        return float(np.mean(vals))

    return {
        "exact_duplicate_groups": int(len(exact_groups)),
        "exact_duplicate_target_mse": _group_mse(exact_groups),
        "mixed_source_duplicate_groups": int(len(mixed_groups)),
        "mixed_source_duplicate_target_mse": _group_mse(mixed_groups),
    }


def main() -> None:
    args = _build_parser().parse_args()
    _set_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    ckpt_args = dict(ckpt["args"])
    task_names = list(ckpt_args.get("multi_task_names") or [ckpt_args["task_name"]])
    task_name, task_idx = _resolve_task(task_names, args.task_name)
    n_normal, n_failure = _resolve_target_counts(args)
    failure_table_paths = _resolve_failure_table_paths(ckpt_args, task_names, args.failure_table_paths)

    task_specs = resolve_multitask_specs(task_names, SIM_TASK_CONFIGS)
    shared_camera_names = list(task_specs[0].camera_names)
    dataset, norm_stats = build_multitask_stage1_dataset(
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

    first = dataset[0]
    model.initialize_latent_heads(first["image_t"].unsqueeze(0).to(device), teacher)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    correction_cfg = build_act_aligned_cfg_from_args(
        SimpleNamespace(**ckpt_args),
        max_action_len=int(ckpt_args.get("act_chunk_size", 50)),
    )
    correction_cfg.failure_mode = "train"
    correction_builder = ACTAlignedCorrectionBuilder(
        correction_cfg,
        urdf_path=str(ckpt_args["urdf_path"]),
        curobo_left_yml=str(ckpt_args.get("curobo_left_yml", "/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml")),
        curobo_right_yml=str(ckpt_args.get("curobo_right_yml", "/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml")),
        device=str(device),
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )

    print(
        f"[feasibility] loaded model missing={len(missing)} unexpected={len(unexpected)} "
        f"tasks={len(task_names)} task_name={task_name} normal={n_normal} failure={n_failure}",
        flush=True,
    )

    with torch.no_grad():
        normal_records = _collect_normal_samples(
            dataset=dataset,
            task_idx=task_idx,
            num_samples=n_normal,
            model=model,
            teacher=teacher,
            device=device,
        )
        failure_records = _collect_failure_samples(
            dataset=failure_dataset,
            task_idx=task_idx,
            num_samples=n_failure,
            max_tries_per_sample=args.failure_max_tries_per_sample,
            model=model,
            teacher=teacher,
            correction_builder=correction_builder,
            norm_stats=norm_stats,
            task_name=task_name,
            raw_data_dir=str(task_specs[task_idx].raw_data_dir),
            device=device,
        )

    records = normal_records + failure_records
    if len(records) < 2:
        raise RuntimeError(
            f"Too few valid records: total={len(records)} "
            f"(normal={len(normal_records)} failure={len(failure_records)})"
        )

    x = torch.stack([r["feature"] for r in records], dim=0).float()
    y = torch.stack([r["target"] for r in records], dim=0).float()
    sources = [str(r["source"]) for r in records]

    linear_metrics = _linear_probe_metrics(x, y, args.train_ratio)
    dup_metrics = _duplicate_metrics(x, y, sources, args.round_decimals)
    cross_metrics = _cross_source_nn_metrics(x, y, sources)

    source_counter = Counter(sources)
    failure_modes = Counter([str(r.get("error_mode", "")) for r in failure_records if r.get("error_mode", "")])
    phase_keys = Counter([str(r.get("phase_key", "")) for r in failure_records if r.get("phase_key", "")])

    results = {
        "ckpt_path": os.path.realpath(args.ckpt_path),
        "num_tasks_in_ckpt": len(task_names),
        "task_names_in_ckpt": task_names,
        "analyzed_task_name": task_name,
        "requested_normal_samples": int(n_normal),
        "requested_failure_samples": int(n_failure),
        "collected_normal_samples": int(len(normal_records)),
        "collected_failure_samples": int(len(failure_records)),
        "num_samples": int(x.shape[0]),
        "source_counts": dict(source_counter),
        "failure_error_mode_counts": dict(failure_modes),
        "failure_phase_key_counts": dict(phase_keys),
        "feature_dim": int(x.shape[1]),
        "token_dim": int(y.shape[1]),
        **linear_metrics,
        "knn_k": int(args.knn_k),
        "knn_future_mse": None,
        "global_adjacent_pair_mse": _mean_pairwise_mse(y),
        **dup_metrics,
        **cross_metrics,
    }

    if x.shape[0] > int(args.knn_k):
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_n = (x - x_mean) / x_std
        dists = torch.cdist(x_n, x_n)
        dists.fill_diagonal_(float("inf"))
        knn_idx = torch.topk(dists, k=int(args.knn_k), dim=1, largest=False).indices
        nn_targets = y[knn_idx]
        nn_mean = nn_targets.mean(dim=1)
        results["knn_future_mse"] = float(torch.mean((nn_mean - y) ** 2).item())

    print(json.dumps(results, indent=2), flush=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
