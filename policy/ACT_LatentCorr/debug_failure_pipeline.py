#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import random
import time
from argparse import Namespace
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBOTWIN_ROOT = os.path.realpath(os.path.join(_THIS_DIR, "..", ".."))
if _ROBOTWIN_ROOT not in sys.path:
    sys.path.insert(0, _ROBOTWIN_ROOT)

from policy.ACT.constants import SIM_TASK_CONFIGS

from policy.ACT_LatentCorr.act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from policy.ACT_LatentCorr.config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from policy.ACT_LatentCorr.evac_interface import EvacLatentTeacher
from policy.ACT_LatentCorr.latent_policy import ACTLatentStage1
from policy.ACT_LatentCorr.stage2_failure_dataset import build_multitask_failure_table_dataset
from policy.ACT_LatentCorr.train_stage1_unified_failure_multitask_latent import (
    _as_float_or_none,
    _build_act_args,
    _finite_tensor,
    _qpos_raw_from_norm,
    _action_raw_from_norm,
    build_argparser,
)
from policy.ACT_LatentCorr.utils_latent import load_raw_episode
from policy.ACT_LatentCorr.utils_multitask_latent import build_multitask_stage1_dataset, resolve_multitask_specs


def load_config_dict(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline().strip()
    if not first:
        raise ValueError(f'empty config file: {path}')
    data = ast.literal_eval(first)
    if not isinstance(data, dict):
        raise ValueError(f'expected dict in first line of {path}')
    return data


def build_args_from_config(config: dict[str, Any]) -> Namespace:
    parser = build_argparser()
    defaults = {}
    for action in parser._actions:
        if not getattr(action, 'dest', None) or action.dest == 'help':
            continue
        if action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
    defaults.update(config)
    return Namespace(**defaults)


def make_latent_cfgs(args: Namespace):
    latent_model_cfg = LatentModelConfig(
        projector_mid_channels=args.projector_mid_channels,
        wm_adapter_mid_channels=args.wm_adapter_mid_channels,
        readout_adapter_mid_channels=args.readout_adapter_mid_channels,
        predictor_num_blocks=args.predictor_num_blocks,
        predictor_mlp_hidden=args.predictor_mlp_hidden,
        action_decoder_hidden=args.action_decoder_hidden,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        prefix_steps=args.prefix_steps,
        token_adapter_hidden_dim=getattr(args, 'token_adapter_hidden_dim', 512),
        token_adapter_num_layers=getattr(args, 'token_adapter_num_layers', 2),
        token_adapter_dropout=getattr(args, 'token_adapter_dropout', 0.1),
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_action=args.lambda_action,
        normal_condition_keep_prob=getattr(args, 'normal_condition_keep_prob', 1.0),
        beta_dynamics_max=args.beta_dynamics_max,
        lambda_wm_action_current=args.lambda_wm_action_current,
        lambda_wm_action_future=args.lambda_wm_action_future,
        lambda_bridge_future=args.lambda_bridge_future,
        use_projector_detach_for_predictor=False,
        use_projector_detach_for_action_decoder=True,
        detach_act_feature_for_latent=args.detach_act_feature_for_latent,
        use_raw_wm_targets=args.use_raw_wm_targets,
        lambda_teacher_max=getattr(args, 'lambda_teacher_max', 0.5),
        lambda_pred_max=getattr(args, 'lambda_pred_max', 0.5),
        lambda_latent_max=getattr(args, 'lambda_latent_max', 0.3),
        lambda_token_init=getattr(args, 'lambda_token_init', 0.1),
        lambda_token_late=getattr(args, 'lambda_token_late', 0.02),
        latent_loss_type=getattr(args, 'latent_loss_type', 'normalized_mse'),
        token_loss_type=getattr(args, 'token_loss_type', 'mse'),
        teacher_decay_start_ratio=getattr(args, 'teacher_decay_start_ratio', 0.2),
        teacher_decay_end_ratio=getattr(args, 'teacher_decay_end_ratio', 0.9),
        pred_warmup_start_ratio=getattr(args, 'pred_warmup_start_ratio', 0.1),
        pred_warmup_end_ratio=getattr(args, 'pred_warmup_end_ratio', 0.7),
        latent_warmup_end_ratio=getattr(args, 'latent_warmup_end_ratio', 0.2),
        token_decay_start_ratio=getattr(args, 'token_decay_start_ratio', 0.2),
        token_decay_end_ratio=getattr(args, 'token_decay_end_ratio', 0.9),
        pred_only_finetune_start_ratio=getattr(args, 'pred_only_finetune_start_ratio', 0.9),
        stopgrad_wm_teacher=getattr(args, 'stopgrad_wm_teacher', True),
        stopgrad_teacher_token=getattr(args, 'stopgrad_teacher_token', True),
        stopgrad_adapter_input_for_token_loss=getattr(args, 'stopgrad_adapter_input_for_token_loss', False),
    )
    warmup_cfg = DynamicsWarmupConfig(
        zero_steps=args.dyn_zero_steps,
        ramp_steps=args.dyn_ramp_steps,
        max_weight=1.0,
        curve=args.dyn_warmup_curve,
    )
    return latent_model_cfg, latent_loss_cfg, warmup_cfg


def make_datasets(args: Namespace, task_specs):
    normal_dataset, norm_stats = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=args.act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
    )
    corr_start_margin = int(max(0, args.max_rollout_steps)) * int(max(1, args.act_aligned_rollout_exec_steps))
    failure_dataset, _ = build_multitask_failure_table_dataset(
        task_specs=task_specs,
        act_chunk_size=args.act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
        sample_phase_window_len=args.act_aligned_sample_pregrasp_phase_window_len,
        sample_skip_head_ratio=args.failure_sample_skip_head_ratio,
        start_margin=corr_start_margin,
        failure_mode='train',
        failure_table_paths=list(args.failure_table_paths),
        failure_phase_bins=args.failure_phase_bins,
        failure_translation_dir_bins=args.failure_translation_dir_bins,
        failure_translation_mag_bins=args.failure_translation_mag_bins,
        failure_rotation_dir_bins=args.failure_rotation_dir_bins,
        failure_rotation_mag_bins=args.failure_rotation_mag_bins,
        failure_explore_k=args.failure_explore_k,
        balance_tasks=args.balance_failure_tasks,
        rebalance_failure_groups=args.rebalance_failure_groups,
    )
    failure_dataset.set_norm_stats(norm_stats)
    return normal_dataset, failure_dataset, norm_stats


def load_model(args: Namespace, checkpoint_path: str, task_specs, normal_dataset, teacher: EvacLatentTeacher, device: str):
    latent_model_cfg, latent_loss_cfg, warmup_cfg = make_latent_cfgs(args)
    model = ACTLatentStage1(
        act_args=_build_act_args(list(task_specs[0].camera_names), args),
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
    ).to(device)
    init_loader = DataLoader(normal_dataset, batch_size=1, shuffle=False, num_workers=0)
    init_batch = next(iter(init_loader))
    model.initialize_latent_heads(init_batch['image_t'][:1].to(device), teacher)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, {'missing': len(missing), 'unexpected': len(unexpected)}


def diagnose(label: str, args: Namespace, checkpoint_path: str, samples: int, device: str):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS)
    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=device)
    normal_dataset, failure_dataset, norm_stats = make_datasets(args, task_specs)
    model, load_info = load_model(args, checkpoint_path, task_specs, normal_dataset, teacher, device)

    failure_loader = DataLoader(
        failure_dataset,
        batch_size=max(1, int(args.failure_batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    correction_cfg = build_act_aligned_cfg_from_args(args, max_action_len=args.act_chunk_size)
    correction_cfg.failure_mode = 'train'
    builder = ACTAlignedCorrectionBuilder(
        cfg=correction_cfg,
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )

    total = 0
    prepared = 0
    by_reason = Counter()
    by_task_total = Counter()
    by_task_prepared = Counter()
    by_task_reason: dict[str, Counter] = defaultdict(Counter)
    error_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}
    builder_time = 0.0
    rollout_time = 0.0

    with torch.no_grad():
        for batch in failure_loader:
            bs = int(batch['image_t'].shape[0])
            for i in range(bs):
                if total >= samples:
                    break
                total += 1
                task_name = str(batch['task_name'][i])
                by_task_total[task_name] += 1
                ep_id = int(batch['episode_id'][i])
                start_ts = int(batch['start_ts'][i])
                raw_dir = str(batch['raw_data_dir'][i])
                cache_key = (task_name, ep_id)
                if cache_key not in raw_cache:
                    raw_cache[cache_key] = load_raw_episode(raw_dir, ep_id)

                t0 = time.perf_counter()
                corr = builder.build(
                    model,
                    image_t=batch['image_t'][i].to(device, non_blocking=True),
                    qpos_t=batch['qpos_t'][i].to(device, non_blocking=True),
                    raw_data=raw_cache[cache_key],
                    norm_stats=norm_stats,
                    start_ts=start_ts,
                    failure_mode_override='train',
                    sampled_phase_id=int(batch['sampled_phase_id'][i]),
                    pregrasp_seg_start=int(batch['pregrasp_seg_start'][i]),
                    pregrasp_seg_end=int(batch['pregrasp_seg_end'][i]),
                    sampled_phase_bin_id=int(batch['sampled_phase_bin_id'][i]),
                    sampled_phase_instance_id=int(batch['sampled_phase_instance_id'][i]),
                    forced_error_mode_id=int(batch['forced_error_mode_id'][i]),
                    sampled_active_arm_pattern_id=int(batch['sampled_active_arm_pattern_id'][i]),
                    forced_dir_bin_id=int(batch['forced_dir_bin_id'][i]),
                    forced_mag_bin_id=int(batch['forced_mag_bin_id'][i]),
                    sampled_mode_prob=_as_float_or_none(batch['sampled_mode_prob'][i]),
                    sampled_entry_prob_within_mode=_as_float_or_none(batch['sampled_entry_prob_within_mode'][i]),
                    sampled_unit_prob=_as_float_or_none(batch['sampled_unit_prob'][i]),
                )
                builder_time += time.perf_counter() - t0

                if corr is None:
                    meta = builder.pop_last_skip_meta() or {}
                    reason = str(meta.get('skip_reason', 'builder_none')).strip().lower()
                    by_reason[reason] += 1
                    by_task_reason[task_name][reason] += 1
                    if len(error_examples[reason]) < 3:
                        error_examples[reason].append(
                            {
                                'task_name': task_name,
                                'episode_id': ep_id,
                                'start_ts': start_ts,
                                'skip_error': meta.get('skip_error'),
                                'correction_branch': meta.get('correction_branch'),
                            }
                        )
                    continue

                corr_image = corr.get('corr_image')
                corr_qpos_norm = corr.get('corr_qpos_norm')
                corr_action = corr.get('corr_action_chunk_norm')
                corr_is_pad = corr.get('corr_is_pad')
                if not (_finite_tensor(corr_image) and _finite_tensor(corr_qpos_norm) and _finite_tensor(corr_action)):
                    reason = 'non_finite_tensor'
                    by_reason[reason] += 1
                    by_task_reason[task_name][reason] += 1
                    continue
                if corr_is_pad is None or corr_action.ndim != 2 or corr_action.shape[0] < args.prefix_steps:
                    reason = 'invalid_pad_or_shape'
                    by_reason[reason] += 1
                    by_task_reason[task_name][reason] += 1
                    continue

                if str(args.failure_future_latent_mode).strip().lower() == 'rollout':
                    t1 = time.perf_counter()
                    corr_qpos_raw = _qpos_raw_from_norm(corr_qpos_norm, norm_stats)
                    corr_action_prefix_raw = _action_raw_from_norm(corr_action[: args.prefix_steps], norm_stats)
                    future_latent = teacher.rollout_latent_from_actions(
                        curr_image=corr_image[0],
                        curr_qpos_raw=corr_qpos_raw,
                        action_prefix_raw=corr_action_prefix_raw,
                        raw_data=raw_cache[cache_key],
                        fk=builder.fk,
                        ddim_steps=int(args.ddim_steps),
                    )[0]
                    rollout_time += time.perf_counter() - t1
                    if not _finite_tensor(future_latent):
                        reason = 'non_finite_rollout_latent'
                        by_reason[reason] += 1
                        by_task_reason[task_name][reason] += 1
                        continue

                prepared += 1
                by_reason['prepared'] += 1
                by_task_prepared[task_name] += 1
                by_task_reason[task_name]['prepared'] += 1
            if total >= samples:
                break

    summary = {
        'label': label,
        'checkpoint_path': checkpoint_path,
        'samples': total,
        'prepared': prepared,
        'prepared_ratio': 0.0 if total == 0 else prepared / float(total),
        'load_info': load_info,
        'by_reason': dict(by_reason),
        'by_task_total': dict(by_task_total),
        'by_task_prepared': dict(by_task_prepared),
        'by_task_reason': {k: dict(v) for k, v in by_task_reason.items()},
        'error_examples': dict(error_examples),
        'builder_time_sec_avg': 0.0 if total == 0 else builder_time / float(total),
        'rollout_time_sec_avg': 0.0 if prepared == 0 else rollout_time / float(prepared),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    cfg = load_config_dict(args.config)
    ns = build_args_from_config(cfg)
    diagnose(args.label, ns, args.checkpoint, args.samples, args.device)


if __name__ == '__main__':
    main()
