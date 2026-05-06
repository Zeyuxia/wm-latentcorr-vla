from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.utils.constants import ACTION
from omegaconf import OmegaConf

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
from policy.SmolVLA.latentcorr.correction_policy_adapter import SampleBoundSmolVLAAdapter
from policy.SmolVLA.latentcorr.evac_interface import EvacLatentTeacher
from policy.SmolVLA.latentcorr.failure_manifest_utils import load_failure_table_paths
from policy.SmolVLA.latentcorr.failure_utils import (
    active_arm_pattern_id_to_key,
    error_mode_id_to_key,
    phase_id_to_key,
)
from policy.SmolVLA.latentcorr.latent_config import DynamicsWarmupConfig, Stage1WarmupConfig
from policy.SmolVLA.latentcorr.latent_dataset_utils import load_raw_episode
from policy.SmolVLA.latentcorr.multitask_failure_dataset import MultiTaskFailureDatasetConfig, build_multitask_failure_dataset
from policy.SmolVLA.latentcorr.multitask_latent_utils import build_multitask_stage1_dataset, resolve_multitask_specs
from policy.SmolVLA.latentcorr.smolvla_data_utils import build_smolvla_batch, make_smolvla_processors, stack_smolvla_batches
from policy.SmolVLA.latentcorr.smolvla_latent_policy import SmolVLALatentBridgeConfig, SmolVLALatentPolicy


def parse_task_parts(task_name: str) -> tuple[str, str]:
    if not task_name.startswith("sim-"):
        raise ValueError(f"Expected sim task name, got {task_name}")
    parts = task_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected multitask name format: {task_name}")
    return parts[1], parts[2]


def parse_bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Expected 'true' or 'false', got {value}")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def scalar_to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _save_loss_batch_projection(
    save_dir: str,
    image_cam: torch.Tensor,
    action_norm: torch.Tensor,
    is_pad: torch.Tensor,
    raw_data: dict[str, Any],
    fk,
    norm_stats: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    import cv2

    act = np.asarray(action_norm.detach().cpu().numpy(), dtype=np.float32)
    pad = np.asarray(is_pad.detach().cpu().numpy(), dtype=bool).reshape(-1)
    if act.ndim != 2 or act.shape[1] < 14:
        return
    valid = np.where(~pad)[0]
    if valid.size == 0:
        return
    act = act[valid]
    act_raw = np.asarray(act * norm_stats["action_std"] + norm_stats["action_mean"], dtype=np.float32)
    if act_raw.shape[0] <= 0:
        return

    img_rgb = np.clip(
        image_cam.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    overlay = img_rgb[:, :, ::-1].copy()

    K = raw_data["intrinsic_cv"].astype(np.float32).copy()
    E = np.eye(4, dtype=np.float32)
    E[:3, :] = raw_data["extrinsic_cv"].astype(np.float32)
    h_native, w_native = raw_data.get("native_resolution", (overlay.shape[0], overlay.shape[1]))
    h_img, w_img = overlay.shape[:2]
    if w_native > 0 and h_native > 0 and (w_native != w_img or h_native != h_img):
        sx = float(w_img) / float(w_native)
        sy = float(h_img) / float(h_native)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    pose_list = []
    for i in range(act_raw.shape[0]):
        fr = fk.forward(act_raw[i, 0:6], act_raw[i, 7:13])
        lp, lq_wxyz = fr["left"]
        rp, rq_wxyz = fr["right"]
        lq_xyzw = np.array([lq_wxyz[1], lq_wxyz[2], lq_wxyz[3], lq_wxyz[0]], dtype=np.float32)
        rq_xyzw = np.array([rq_wxyz[1], rq_wxyz[2], rq_wxyz[3], rq_wxyz[0]], dtype=np.float32)
        if lq_xyzw[3] < 0:
            lq_xyzw = -lq_xyzw
        if rq_xyzw[3] < 0:
            rq_xyzw = -rq_xyzw
        lg = float(np.clip(act_raw[i, 6], 0.0, 1.0)) * 120.0
        rg = float(np.clip(act_raw[i, 13], 0.0, 1.0)) * 120.0
        pose_list.append(np.concatenate([lp, lq_xyzw, [lg], rp, rq_xyzw, [rg]], axis=0).astype(np.float32))
    pose_np = np.stack(pose_list, axis=0)

    try:
        import evac.lvdm.models.ddpm3d as ddpm3d_mod

        w2c_t = torch.from_numpy(E).float().unsqueeze(0).unsqueeze(0)
        intrinsic_t = torch.from_numpy(K).float().unsqueeze(0).unsqueeze(0)
        cvt_matrix = torch.tensor(ddpm3d_mod.Gripper2EEFCvt, dtype=torch.float32).view(1, 1, 4, 4)
        ee_key_pts = torch.tensor(ddpm3d_mod.EndEffectorPts, dtype=torch.float32).view(1, 1, 4, 4).permute(0, 1, 3, 2)

        def _project_base_uv_from_pose_np(pose_arr):
            pose_t = torch.from_numpy(pose_arr).float()
            pose_l_mat = ddpm3d_mod.get_transformation_matrix_from_quat(pose_t[:, 0:7]).unsqueeze(0)
            pose_r_mat = ddpm3d_mod.get_transformation_matrix_from_quat(pose_t[:, 8:15]).unsqueeze(0)
            ee2cam_l = torch.matmul(torch.matmul(w2c_t, pose_l_mat), cvt_matrix)
            ee2cam_r = torch.matmul(torch.matmul(w2c_t, pose_r_mat), cvt_matrix)
            pts_l = torch.matmul(ee2cam_l, ee_key_pts)
            pts_r = torch.matmul(ee2cam_r, ee_key_pts)
            uvs_l = torch.matmul(intrinsic_t, pts_l[:, :, :3, :])
            uvs_r = torch.matmul(intrinsic_t, pts_r[:, :, :3, :])
            uvs_l = (uvs_l / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)[0].cpu().numpy()
            uvs_r = (uvs_r / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)[0].cpu().numpy()
            return uvs_l, uvs_r, pts_l, pts_r

        def _extract_base_uv(uvs, pts):
            seq = []
            for i in range(uvs.shape[0]):
                z = float(pts[0, i, 2, 0].item())
                u = int(uvs[i, 0, 0])
                v = int(uvs[i, 0, 1])
                if z > 1e-6 and (0 <= u < w_img) and (0 <= v < h_img):
                    seq.append((u, v))
                else:
                    seq.append(None)
            return seq

        def _draw_polyline(img, seq, color):
            prev = None
            for pt in seq:
                if pt is None:
                    prev = None
                    continue
                if prev is not None:
                    cv2.line(img, (int(prev[0]), int(prev[1])), (int(pt[0]), int(pt[1])), color, 2, cv2.LINE_AA)
                prev = pt

        def _annotate_start_end(img, seq, prefix, color):
            if len(seq) == 0:
                return
            s = seq[0]
            e = seq[-1]
            if s is not None:
                cv2.circle(img, (int(s[0]), int(s[1])), 4, color, -1, cv2.LINE_AA)
                cv2.putText(img, f"{prefix}-S", (int(s[0]) + 6, int(s[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if e is not None:
                cv2.circle(img, (int(e[0]), int(e[1])), 4, color, -1, cv2.LINE_AA)
                cv2.putText(img, f"{prefix}-E", (int(e[0]) + 6, int(e[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        uvs_l, uvs_r, pts_l, pts_r = _project_base_uv_from_pose_np(pose_np)
        luv = _extract_base_uv(uvs_l, pts_l.reshape(1, pts_l.shape[1], 4, 4))
        ruv = _extract_base_uv(uvs_r, pts_r.reshape(1, pts_r.shape[1], 4, 4))
        _draw_polyline(overlay, luv, (0, 255, 0))
        _draw_polyline(overlay, ruv, (0, 0, 255))
        _annotate_start_end(overlay, luv, "L", (0, 255, 0))
        _annotate_start_end(overlay, ruv, "R", (0, 0, 255))
        cv2.imwrite(os.path.join(save_dir, "loss_projection_on_input.png"), overlay)
    except Exception:
        import traceback

        with open(os.path.join(save_dir, "loss_projection_error.txt"), "w") as f:
            f.write(traceback.format_exc())
        cv2.imwrite(os.path.join(save_dir, "loss_projection_on_input.png"), overlay)

    meta_out = {"n_valid_actions": int(act_raw.shape[0]), "image_hw": [int(h_img), int(w_img)]}
    if isinstance(meta, dict):
        meta_out.update(meta)
    with open(os.path.join(save_dir, "loss_projection_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)


def _tensor_to_numpy(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    return value.detach().cpu().numpy()


def _image_chw_to_uint8(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = value.detach().cpu().float().numpy()
    if arr.size > 0 and float(np.nanmax(arr)) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_stage1_correction_data(
    save_dir: Path,
    manifest_path: Path,
    *,
    rank: int,
    global_step: int,
    sample_index: int,
    task_name_full: str,
    episode_id: int,
    start_ts: int,
    raw_data_dir: str,
    correction_raw_batch: dict[str, Any],
    correction: dict[str, Any],
    corr_image: torch.Tensor,
    corr_qpos_raw: torch.Tensor,
    corr_action_raw: torch.Tensor,
) -> str:
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_name = f"step_{int(global_step):06d}_bi{int(sample_index):03d}.npz"
    sample_path = save_dir / sample_name
    corr_meta = correction.get("corr_meta") if isinstance(correction.get("corr_meta"), dict) else {}
    recover_eval_last = corr_meta.get("recover_eval_last") if isinstance(corr_meta.get("recover_eval_last"), dict) else {}
    arrays = {
        "corr_image_chw_uint8": _image_chw_to_uint8(corr_image[0]),
        "future_image_chw_uint8": _image_chw_to_uint8(correction_raw_batch["image_t"][sample_index, 0]),
        "source_image_chw_uint8": _image_chw_to_uint8(correction_raw_batch["image_t"][sample_index, 0]),
        "corr_qpos_norm": _tensor_to_numpy(correction["corr_qpos_norm"]),
        "corr_qpos_raw": _tensor_to_numpy(corr_qpos_raw),
        "corr_action_chunk_norm": _tensor_to_numpy(correction["corr_action_chunk_norm"]),
        "corr_action_chunk_raw": _tensor_to_numpy(corr_action_raw),
        "corr_is_pad": _tensor_to_numpy(correction.get("corr_is_pad")),
        "source_action_chunk_norm": _tensor_to_numpy(correction_raw_batch["act_action_chunk"][sample_index]),
        "source_is_pad": _tensor_to_numpy(correction_raw_batch["act_is_pad"][sample_index]),
        "error_action_prefix_norm": _tensor_to_numpy(correction.get("error_action_prefix_norm")),
        "error_action_prefix_raw": _tensor_to_numpy(correction.get("error_action_prefix_raw")),
        "error_is_pad_prefix": _tensor_to_numpy(correction.get("error_is_pad_prefix")),
        "perturb_action_prefix_raw": (
            None
            if corr_meta.get("perturb_action_prefix_raw") is None
            else np.asarray(corr_meta.get("perturb_action_prefix_raw"), dtype=np.float32)
        ),
        "perturb_start_qpos_raw": (
            None
            if corr_meta.get("perturb_start_qpos_raw") is None
            else np.asarray(corr_meta.get("perturb_start_qpos_raw"), dtype=np.float32)
        ),
        "perturb_final_qpos_raw": (
            None
            if corr_meta.get("perturb_final_qpos_raw") is None
            else np.asarray(corr_meta.get("perturb_final_qpos_raw"), dtype=np.float32)
        ),
    }
    np.savez_compressed(sample_path, **{k: v for k, v in arrays.items() if v is not None})
    manifest_record = {
        "rank": int(rank),
        "global_step": int(global_step),
        "batch_index": int(sample_index),
        "path": str(sample_path),
        "task_name": str(task_name_full),
        "episode_id": int(episode_id),
        "start_ts": int(start_ts),
        "raw_data_dir": str(raw_data_dir),
        "sampled_phase_key": corr_meta.get("sampled_phase_key"),
        "sampled_phase_bin_id": corr_meta.get("sampled_phase_bin_id"),
        "sampled_phase_instance_idx": corr_meta.get("sampled_phase_instance_idx"),
        "sampled_error_mode": corr_meta.get("sampled_error_mode"),
        "forced_error_mode": corr_meta.get("forced_error_mode"),
        "sampled_active_arm_pattern": corr_meta.get("sampled_active_arm_pattern"),
        "forced_dir_bin_id": corr_meta.get("forced_dir_bin_id"),
        "forced_mag_bin_id": corr_meta.get("forced_mag_bin_id"),
        "perturb_action_prefix_len": corr_meta.get("perturb_action_prefix_len"),
        "perturb_delta_linf": corr_meta.get("perturb_delta_linf"),
        "perturb_delta_l2": corr_meta.get("perturb_delta_l2"),
        "recover_eval_recoverable": recover_eval_last.get("recoverable"),
        "recover_eval_failed_thresholds": recover_eval_last.get("failed_thresholds"),
        "recover_eval_metrics": recover_eval_last.get("metrics"),
    }
    _append_jsonl(manifest_path, manifest_record)
    return str(sample_path)


def build_accelerator() -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(
        step_scheduler_with_optimizer=False,
        rng_types=[],
        kwargs_handlers=[ddp_kwargs],
    )


def load_evac_sample_size(evac_config_path: str) -> tuple[int, int] | None:
    path = str(evac_config_path).strip()
    if not path:
        return None
    cfg = OmegaConf.load(path)
    value = cfg.data.params.train.params.sample_size
    if value is None or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def compute_retain_weight(
    retain_weight: float,
    retain_weight_final: float,
    retain_decay_start_step: float,
    retain_decay_end_step: float,
    retain_decay_curve: str,
    step_progress: float,
) -> float:
    if retain_decay_end_step <= retain_decay_start_step:
        return retain_weight
    if step_progress <= retain_decay_start_step:
        return retain_weight
    if step_progress >= retain_decay_end_step:
        return retain_weight_final
    ratio = (step_progress - retain_decay_start_step) / max(1e-8, retain_decay_end_step - retain_decay_start_step)
    ratio = max(0.0, min(1.0, ratio))
    if retain_decay_curve == "cosine":
        ratio = 0.5 * (1.0 - math.cos(math.pi * ratio))
    return retain_weight + (retain_weight_final - retain_weight) * ratio


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smolvla_pretrained_path", type=str, required=True)
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--freeze_vision_encoder", type=str, required=True, choices=["true", "false"])
    parser.add_argument("--train_expert_only", type=str, required=True, choices=["true", "false"])
    parser.add_argument("--load_vlm_weights", type=str, required=True, choices=["true", "false"])
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--instruction_type", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--save_freq", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--scheduler_warmup_steps", type=int, required=True)
    parser.add_argument("--scheduler_decay_steps", type=int, required=True)
    parser.add_argument("--scheduler_decay_lr", type=float, required=True)
    parser.add_argument("--future_offset", type=int, required=True)
    parser.add_argument("--prefix_steps", type=int, required=True)
    parser.add_argument("--action_dim", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--adapter_hidden_dim", type=int, required=True)
    parser.add_argument("--predictor_hidden_dim", type=int, required=True)
    parser.add_argument("--dyn_zero_steps", type=int, required=True)
    parser.add_argument("--dyn_ramp_steps", type=int, required=True)
    parser.add_argument("--dyn_max_weight", type=float, required=True)
    parser.add_argument("--dyn_warmup_curve", type=str, required=True, choices=["linear", "cosine"])
    parser.add_argument("--cond_zero_steps", type=int, default=None)
    parser.add_argument("--cond_ramp_steps", type=int, default=None)
    parser.add_argument("--cond_max_weight", type=float, default=None)
    parser.add_argument("--cond_warmup_curve", type=str, default=None, choices=["linear", "cosine"])
    parser.add_argument("--act_chunk_size", type=int, required=True)


def add_evac_dual_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evac_use_dual_cache", type=str2bool, default=False)
    parser.add_argument("--evac_dc_v_bounds", nargs="*", type=int, default=[])
    parser.add_argument("--evac_dc_budget", type=float, default=-1.0)
    parser.add_argument("--evac_dc_enc_start", type=int, default=999)
    parser.add_argument("--evac_dc_replay_step_noise", type=str2bool, default=False)
    parser.add_argument("--evac_dc_hf_metric", type=str2bool, default=False)
    parser.add_argument("--evac_dc_v_blur_on_reuse", type=str2bool, default=False)
    parser.add_argument("--evac_dc_v_blur_kernel", type=int, default=3)
    parser.add_argument("--evac_dc_v_blur_strength", type=float, default=0.15)


def add_stage2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--failure_table_paths_json", type=str, required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--curobo_left_yml", type=str, required=True)
    parser.add_argument("--curobo_right_yml", type=str, required=True)
    parser.add_argument("--correction_batch_size", type=int, required=True)
    parser.add_argument("--retain_weight", type=float, required=True)
    parser.add_argument("--retain_weight_final", type=float, required=True)
    parser.add_argument("--retain_decay_start_step", type=float, required=True)
    parser.add_argument("--retain_decay_end_step", type=float, required=True)
    parser.add_argument("--retain_decay_curve", type=str, required=True, choices=["linear", "cosine"])
    parser.add_argument("--failure_phase_bins", type=int, required=True)
    parser.add_argument("--failure_translation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_translation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_explore_k", type=int, required=True)
    parser.add_argument("--sample_phase_window_len", type=int, required=True)
    parser.add_argument("--start_margin", type=int, required=True)
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, required=True)
    parser.add_argument("--act_aligned_full_chunk_recovery", type=str2bool, default=False)
    parser.add_argument("--planner_orient_weight", type=float, required=True)
    parser.add_argument("--planner_gripper_penalty", type=float, required=True)
    parser.add_argument("--planner_nearest_window_radius", type=int, required=True)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, required=True)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_save_video", type=str2bool, required=True)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, required=True)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, required=True)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, required=True)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, required=True)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, required=True)
    add_evac_dual_cache_args(parser)


def add_failure_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--failure_mode", type=str, default="off", choices=["off", "train"])
    parser.add_argument("--failure_table_paths_json", type=str, default="")
    parser.add_argument("--failure_task_names", nargs="*", default=None)
    parser.add_argument("--failure_corr_batch_ratio", type=float, default=0.0)
    parser.add_argument("--failure_phase_bins", type=int, default=3)
    parser.add_argument("--failure_translation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_translation_mag_bins", type=int, default=1)
    parser.add_argument("--failure_rotation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_rotation_mag_bins", type=int, default=1)
    parser.add_argument("--failure_explore_k", type=int, default=4)
    parser.add_argument("--sample_phase_window_len", type=int, default=20)
    parser.add_argument("--start_margin", type=int, default=16)
    parser.add_argument("--urdf_path", type=str, default="")
    parser.add_argument("--curobo_left_yml", type=str, default="")
    parser.add_argument("--curobo_right_yml", type=str, default="")
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, default=16)
    parser.add_argument("--act_aligned_full_chunk_recovery", type=str2bool, default=False)
    parser.add_argument("--planner_orient_weight", type=float, default=0.0573)
    parser.add_argument("--planner_gripper_penalty", type=float, default=1.0)
    parser.add_argument("--planner_nearest_window_radius", type=int, default=12)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, default=0.01)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, default=0.05)
    parser.add_argument("--recover_eval_save_video", type=str2bool, default=False)
    parser.add_argument("--save_perturb_rollout_video", type=str2bool, default=False)
    parser.add_argument("--save_correction_debug", type=str2bool, default=False)
    parser.add_argument("--save_correction_data", type=str2bool, default=False)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, default=0.2)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, default=0.04)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, default=8.0)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, default=16)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, default=16)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, default=True)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, default=0.05)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, default=15.0)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, default=0.10)
    parser.add_argument("--evac_blur_filter_enable", type=str2bool, default=False)
    parser.add_argument("--evac_blur_filter_metric", type=str, choices=["sharpness_ratio", "grad_cosine", "mode_aware"], default="sharpness_ratio")
    parser.add_argument("--evac_blur_filter_min_ratio", type=float, default=0.75)
    parser.add_argument("--evac_blur_filter_region", type=str, choices=["full_image", "active_gripper_patch"], default="active_gripper_patch")
    parser.add_argument("--evac_blur_filter_patch_pad_px", type=int, default=12)
    parser.add_argument("--evac_blur_filter_gripper_axis_m", type=float, default=0.04)
    parser.add_argument("--debug_wm_correction", type=str2bool, default=False)
    parser.add_argument("--debug_wm_all_ranks", type=str2bool, default=False)
    parser.add_argument("--debug_loss_batch_projection", type=str2bool, default=False)
    add_evac_dual_cache_args(parser)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("train smolvla latentcorr")
    subparsers = parser.add_subparsers(dest="train_mode", required=True)

    stage1_parser = subparsers.add_parser("stage1")
    add_common_args(stage1_parser)
    add_failure_train_args(stage1_parser)

    stage2_parser = subparsers.add_parser("stage2")
    add_common_args(stage2_parser)
    add_stage2_args(stage2_parser)

    return parser


def build_bridge_config(args: argparse.Namespace) -> SmolVLALatentBridgeConfig:
    return SmolVLALatentBridgeConfig(
        action_dim=int(args.action_dim),
        prefix_steps=int(args.prefix_steps),
        latent_dim=int(args.latent_dim),
        adapter_hidden_dim=int(args.adapter_hidden_dim),
        predictor_hidden_dim=int(args.predictor_hidden_dim),
    )


def build_warmup_config(args: argparse.Namespace) -> DynamicsWarmupConfig:
    return DynamicsWarmupConfig(
        zero_steps=int(args.dyn_zero_steps),
        ramp_steps=int(args.dyn_ramp_steps),
        max_weight=float(args.dyn_max_weight),
        curve=str(args.dyn_warmup_curve),
    )


def build_stage1_warmup_config(args: argparse.Namespace) -> Stage1WarmupConfig:
    dynamics = build_warmup_config(args)
    condition = DynamicsWarmupConfig(
        zero_steps=int(args.cond_zero_steps if args.cond_zero_steps is not None else args.dyn_zero_steps),
        ramp_steps=int(args.cond_ramp_steps if args.cond_ramp_steps is not None else args.dyn_ramp_steps),
        max_weight=float(args.cond_max_weight if args.cond_max_weight is not None else args.dyn_max_weight),
        curve=str(args.cond_warmup_curve if args.cond_warmup_curve is not None else args.dyn_warmup_curve),
    )
    return Stage1WarmupConfig(dynamics=dynamics, condition=condition)


def build_base_policy(args: argparse.Namespace) -> tuple[SmolVLAPolicy, Any, Any]:
    base_policy = SmolVLAPolicy.from_pretrained(args.smolvla_pretrained_path)
    base_policy.config.freeze_vision_encoder = parse_bool_flag(args.freeze_vision_encoder)
    base_policy.config.train_expert_only = parse_bool_flag(args.train_expert_only)
    base_policy.config.load_vlm_weights = parse_bool_flag(args.load_vlm_weights)
    base_policy.model.vlm_with_expert.freeze_vision_encoder = base_policy.config.freeze_vision_encoder
    base_policy.model.vlm_with_expert.train_expert_only = base_policy.config.train_expert_only
    base_policy.model.vlm_with_expert.set_requires_grad()
    base_policy.model.set_requires_grad()
    base_policy.to(torch.device(args.device))
    preprocess, postprocess = make_smolvla_processors(base_policy, args.smolvla_pretrained_path)
    return base_policy, preprocess, postprocess


def configure_optimizer(model: SmolVLALatentPolicy, args: argparse.Namespace, total_training_steps: int):
    model.base_policy.config.optimizer_lr = float(args.learning_rate)
    model.base_policy.config.optimizer_weight_decay = float(args.weight_decay)
    model.base_policy.config.scheduler_warmup_steps = int(args.scheduler_warmup_steps)
    model.base_policy.config.scheduler_decay_steps = int(args.scheduler_decay_steps)
    model.base_policy.config.scheduler_decay_lr = float(args.scheduler_decay_lr)
    optimizer_cfg = model.base_policy.config.get_optimizer_preset()
    optimizer = optimizer_cfg.build(model.parameters())
    scheduler_cfg = model.base_policy.config.get_scheduler_preset()
    lr_scheduler = scheduler_cfg.build(optimizer, int(total_training_steps))
    return optimizer_cfg, optimizer, lr_scheduler


def log_stage1_tensorboard(writer: SummaryWriter, step: int, output: Any, lr: float) -> None:
    writer.add_scalar("train/loss", float(output.loss.item()), step)
    writer.add_scalar("train/loss_action", float(output.loss_action.item()), step)
    writer.add_scalar("train/loss_action_conditioned", float(output.loss_action_conditioned.item()), step)
    writer.add_scalar("train/loss_dynamics", float(output.loss_dynamics.item()), step)
    writer.add_scalar("train/beta_condition", float(output.beta_condition), step)
    writer.add_scalar("train/beta_dynamics", float(output.beta_dynamics), step)
    writer.add_scalar("train/lr", float(lr), step)


def _mean_action_abs(raw_batch: dict[str, Any]) -> float:
    action = raw_batch.get("act_action_chunk")
    is_pad = raw_batch.get("act_is_pad")
    if not isinstance(action, torch.Tensor):
        return float("nan")
    action_cpu = action.detach().float().cpu()
    if isinstance(is_pad, torch.Tensor):
        valid = (~is_pad.detach().cpu().bool()).unsqueeze(-1).expand_as(action_cpu)
        values = action_cpu[valid]
    else:
        values = action_cpu.reshape(-1)
    if values.numel() == 0:
        return float("nan")
    return float(values.abs().mean().item())


def _slice_smolvla_batch(batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value[int(start) : int(end)]
        elif isinstance(value, list):
            sliced[key] = value[int(start) : int(end)]
        else:
            sliced[key] = value
    return sliced


@torch.no_grad()
def compute_action_loss_split(
    latent_model: SmolVLALatentPolicy,
    batch: dict[str, Any],
    *,
    base_batch_size: int,
    total_batch_size: int,
) -> tuple[float, float]:
    if ACTION not in batch:
        return float("nan"), float("nan")
    base_batch_size = int(base_batch_size)
    total_batch_size = int(total_batch_size)
    clean_loss = float("nan")
    corr_loss = float("nan")
    was_training = bool(latent_model.training)
    latent_model.eval()
    try:
        if base_batch_size > 0:
            clean_batch = _slice_smolvla_batch(batch, 0, base_batch_size)
            clean_loss = float(
                latent_model._action_loss(batch=clean_batch, actions=clean_batch[ACTION]).detach().float().item()
            )
        if total_batch_size > base_batch_size:
            corr_batch = _slice_smolvla_batch(batch, base_batch_size, total_batch_size)
            corr_loss = float(
                latent_model._action_loss(batch=corr_batch, actions=corr_batch[ACTION]).detach().float().item()
            )
    finally:
        if was_training:
            latent_model.train()
    return clean_loss, corr_loss


def _tensor_item(value: Any, index: int, default: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value[index].item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return default


def _safe_phase_id_to_key(phase_id: Any) -> str:
    try:
        return phase_id_to_key(int(phase_id))
    except Exception:
        return str(phase_id)


def _safe_error_mode_id_to_key(error_mode_id: Any) -> str:
    try:
        return error_mode_id_to_key(int(error_mode_id))
    except Exception:
        return str(error_mode_id)


def _safe_active_arm_id_to_key(active_arm_id: Any) -> str:
    try:
        return active_arm_pattern_id_to_key(int(active_arm_id))
    except Exception:
        return str(active_arm_id)


def summarize_correction_raw_batch(correction_raw_batch: dict[str, Any]) -> list[dict[str, Any]]:
    batch_size = int(correction_raw_batch["image_t"].shape[0])
    records = []
    for sample_index in range(batch_size):
        records.append(
            {
                "task_name": str(_tensor_item(correction_raw_batch.get("task_name"), sample_index, "")),
                "episode_id": int(_tensor_item(correction_raw_batch.get("episode_id"), sample_index, -1)),
                "start_ts": int(_tensor_item(correction_raw_batch.get("start_ts"), sample_index, -1)),
                "phase_key": _safe_phase_id_to_key(
                    _tensor_item(correction_raw_batch.get("sampled_phase_id"), sample_index, -1)
                ),
                "phase_instance_idx": int(
                    _tensor_item(correction_raw_batch.get("sampled_phase_instance_id"), sample_index, -1)
                ),
                "phase_bin_id": int(_tensor_item(correction_raw_batch.get("sampled_phase_bin_id"), sample_index, -1)),
                "error_mode": _safe_error_mode_id_to_key(
                    _tensor_item(correction_raw_batch.get("forced_error_mode_id"), sample_index, -1)
                ),
                "active_arm": _safe_active_arm_id_to_key(
                    _tensor_item(correction_raw_batch.get("sampled_active_arm_pattern_id"), sample_index, -1)
                ),
                "original_active_arm": _safe_active_arm_id_to_key(
                    _tensor_item(correction_raw_batch.get("original_active_arm_pattern_id"), sample_index, -1)
                ),
                "dir_bin_id": int(_tensor_item(correction_raw_batch.get("forced_dir_bin_id"), sample_index, -1)),
                "mag_bin_id": int(_tensor_item(correction_raw_batch.get("forced_mag_bin_id"), sample_index, -1)),
                "sampled_mode_prob": float(
                    _tensor_item(correction_raw_batch.get("sampled_mode_prob"), sample_index, float("nan"))
                ),
                "sampled_entry_prob_within_mode": float(
                    _tensor_item(
                        correction_raw_batch.get("sampled_entry_prob_within_mode"),
                        sample_index,
                        float("nan"),
                    )
                ),
                "sampled_unit_prob": float(
                    _tensor_item(correction_raw_batch.get("sampled_unit_prob"), sample_index, float("nan"))
                ),
            }
        )
    return records


def log_stage1_extra_tensorboard(
    writer: SummaryWriter,
    step: int,
    *,
    base_batch_size: int,
    corr_requested: int,
    corr_generated: int,
    corr_skipped: int,
    clean_action_abs_mean: float,
    corr_source_action_abs_mean: float,
    corr_generated_action_abs_mean: float,
    clean_action_loss_eval: float,
    corr_action_loss_eval: float,
) -> None:
    writer.add_scalar("train_extra/base_batch_size", float(base_batch_size), step)
    writer.add_scalar("train_extra/corr_requested", float(corr_requested), step)
    writer.add_scalar("train_extra/corr_generated", float(corr_generated), step)
    writer.add_scalar("train_extra/corr_skipped", float(corr_skipped), step)
    writer.add_scalar("train_extra/corr_generated_fraction", float(corr_generated) / max(1.0, float(corr_requested)), step)
    if np.isfinite(clean_action_abs_mean):
        writer.add_scalar("train_extra/clean_action_abs_mean_norm", float(clean_action_abs_mean), step)
    if np.isfinite(corr_source_action_abs_mean):
        writer.add_scalar("train_extra/corr_source_action_abs_mean_norm", float(corr_source_action_abs_mean), step)
    if np.isfinite(corr_generated_action_abs_mean):
        writer.add_scalar("train_extra/corr_generated_action_abs_mean_norm", float(corr_generated_action_abs_mean), step)
    if np.isfinite(clean_action_loss_eval):
        writer.add_scalar("train_extra/clean_action_loss_eval", float(clean_action_loss_eval), step)
    if np.isfinite(corr_action_loss_eval):
        writer.add_scalar("train_extra/corr_action_loss_eval", float(corr_action_loss_eval), step)
    if np.isfinite(clean_action_loss_eval) and np.isfinite(corr_action_loss_eval):
        writer.add_scalar(
            "train_extra/corr_to_clean_action_loss_eval",
            float(corr_action_loss_eval) / max(1e-8, float(clean_action_loss_eval)),
            step,
        )


def log_stage2_tensorboard(
    writer: SummaryWriter,
    step: int,
    output: Any,
    lr: float,
    retain_weight: float,
    skipped_batches: int,
) -> None:
    writer.add_scalar("train/loss", float(output.loss.item()), step)
    writer.add_scalar("train/loss_correct", float(output.loss_correct.item()), step)
    writer.add_scalar("train/loss_retain", float(output.loss_retain.item()), step)
    writer.add_scalar("train/loss_dynamics", float(output.loss_dynamics.item()), step)
    writer.add_scalar("train/beta_dynamics", float(output.beta_dynamics), step)
    writer.add_scalar("train/retain_weight", float(retain_weight), step)
    writer.add_scalar("train/lr", float(lr), step)
    writer.add_scalar("train/skipped_batches", float(skipped_batches), step)


def save_train_args(output_dir: Path, args: argparse.Namespace) -> None:
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)


def save_training_checkpoint(
    output_dir: Path,
    prefix: str,
    step: int,
    model: SmolVLALatentPolicy,
    optimizer: Any,
    lr_scheduler: Any,
    extra_state: dict[str, Any],
) -> None:
    checkpoint_path = output_dir / f"{prefix}_step_{step:06d}.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "global_step": step,
    }
    payload.update(extra_state)
    torch.save(payload, checkpoint_path)


def resume_training_checkpoint(
    checkpoint_path: str,
    model: SmolVLALatentPolicy,
    optimizer: Any,
    lr_scheduler: Any,
) -> tuple[int, int]:
    if not checkpoint_path:
        return 0, 0
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["global_step"]), int(checkpoint.get("epoch", 0))


def build_smolvla_batch_from_raw_sample(
    latent_policy: SmolVLALatentPolicy,
    preprocess,
    instruction_type: str,
    task_name_full: str,
    image_t: torch.Tensor,
    qpos_raw: torch.Tensor,
    action_chunk_raw: torch.Tensor,
    episode_id: int,
) -> dict[str, torch.Tensor]:
    task_only, task_config = parse_task_parts(task_name_full)
    return build_smolvla_batch(
        policy=latent_policy.base_policy,
        preprocess=preprocess,
        image_t=image_t,
        qpos_raw=qpos_raw,
        action_chunk_raw=action_chunk_raw,
        task_name=task_only,
        task_config=task_config,
        episode_id=episode_id,
        instruction_type=instruction_type,
    )


def initialize_latent_policy_from_sample(
    latent_policy: SmolVLALatentPolicy,
    preprocess,
    instruction_type: str,
    raw_sample: dict[str, Any],
    teacher: EvacLatentTeacher | None = None,
) -> None:
    init_batch = build_smolvla_batch_from_raw_sample(
        latent_policy=latent_policy,
        preprocess=preprocess,
        instruction_type=instruction_type,
        task_name_full=raw_sample["task_name"],
        image_t=raw_sample["image_t"],
        qpos_raw=raw_sample["qpos_raw"],
        action_chunk_raw=raw_sample["act_action_chunk_raw"],
        episode_id=scalar_to_int(raw_sample["episode_id"]),
    )
    teacher_latent = None
    if teacher is not None:
        teacher_input = raw_sample["image_t"][0].unsqueeze(0).to(device=latent_policy.device, dtype=torch.float32)
        teacher_latent = teacher.encode_image(teacher_input)
    latent_policy.initialize_from_batch(init_batch, teacher_latent=teacher_latent)


def build_stage1_samples_from_raw_batch(
    raw_batch: dict[str, Any],
    latent_model: SmolVLALatentPolicy,
    preprocess,
    instruction_type: str,
    prefix_steps: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor]]:
    samples = []
    future_images = []
    action_prefix = []
    for batch_index in range(raw_batch["image_t"].shape[0]):
        task_name = raw_batch["task_name"][batch_index]
        task_only, task_config = parse_task_parts(task_name)
        sample = build_smolvla_batch(
            policy=latent_model.base_policy,
            preprocess=preprocess,
            image_t=raw_batch["image_t"][batch_index],
            qpos_raw=raw_batch["qpos_raw"][batch_index],
            action_chunk_raw=raw_batch["act_action_chunk_raw"][batch_index],
            task_name=task_only,
            task_config=task_config,
            episode_id=scalar_to_int(raw_batch["episode_id"][batch_index]),
            instruction_type=instruction_type,
        )
        samples.append(sample)
        future_images.append(raw_batch["image_t_future"][batch_index, 0].detach().cpu())
        action_prefix.append(sample["action"][:, : int(prefix_steps), :])
    return samples, future_images, action_prefix


def build_stage1_correction_samples(
    correction_raw_batch: dict[str, Any],
    latent_model: SmolVLALatentPolicy,
    preprocess,
    postprocess,
    instruction_type: str,
    args: argparse.Namespace,
    correction_builder: ACTAlignedCorrectionBuilder,
    teacher: EvacLatentTeacher,
    norm_stats: dict[str, Any],
    raw_cache: dict[tuple[str, int], dict[str, Any]],
    step_debug_dir: str | None = None,
    correction_data_dir: str | None = None,
    correction_manifest_path: str | None = None,
    global_step: int | None = None,
    rank: int = 0,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor], Counter[str], list[dict[str, Any]]]:
    device = torch.device(args.device)
    qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=device)
    qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=device)
    action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=device)
    action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=device)

    samples = []
    future_images = []
    action_prefix = []
    skip_reasons: Counter[str] = Counter()
    projection_debug_records: list[dict[str, Any]] = []
    for sample_index in range(correction_raw_batch["image_t"].shape[0]):
        task_name_full = correction_raw_batch["task_name"][sample_index]
        task_only, task_config = parse_task_parts(task_name_full)
        episode_id = scalar_to_int(correction_raw_batch["episode_id"][sample_index])
        start_ts = scalar_to_int(correction_raw_batch["start_ts"][sample_index])
        adapter = SampleBoundSmolVLAAdapter(
            latent_policy=latent_model,
            preprocess=preprocess,
            postprocess=postprocess,
            task_name=task_only,
            task_config=task_config,
            episode_id=episode_id,
            instruction_type=instruction_type,
        )
        raw_cache_key = (str(correction_raw_batch["raw_data_dir"][sample_index]), episode_id)
        if raw_cache_key not in raw_cache:
            raw_cache[raw_cache_key] = load_raw_episode(
                correction_raw_batch["raw_data_dir"][sample_index],
                episode_id,
            )
        correction = correction_builder.build(
            latent_model=adapter,
            image_t=correction_raw_batch["image_t"][sample_index].to(device),
            qpos_t=correction_raw_batch["qpos_t"][sample_index].to(device),
            raw_data=raw_cache[raw_cache_key],
            norm_stats=norm_stats,
            start_ts=start_ts,
            failure_mode_override="train",
            sampled_phase_id=int(correction_raw_batch["sampled_phase_id"][sample_index].item()),
            sampled_phase_bin_id=int(correction_raw_batch["sampled_phase_bin_id"][sample_index].item()),
            sampled_phase_instance_id=int(correction_raw_batch["sampled_phase_instance_id"][sample_index].item()),
            forced_error_mode_id=int(correction_raw_batch["forced_error_mode_id"][sample_index].item()),
            sampled_active_arm_pattern_id=int(correction_raw_batch["sampled_active_arm_pattern_id"][sample_index].item()),
            original_active_arm_pattern_id=int(correction_raw_batch["original_active_arm_pattern_id"][sample_index].item()),
            forced_dir_bin_id=int(correction_raw_batch["forced_dir_bin_id"][sample_index].item()),
            forced_mag_bin_id=int(correction_raw_batch["forced_mag_bin_id"][sample_index].item()),
            sampled_mode_prob=float(correction_raw_batch["sampled_mode_prob"][sample_index].item()),
            sampled_entry_prob_within_mode=float(correction_raw_batch["sampled_entry_prob_within_mode"][sample_index].item()),
            sampled_unit_prob=float(correction_raw_batch["sampled_unit_prob"][sample_index].item()),
            debug_dir=(None if step_debug_dir is None else str(Path(step_debug_dir) / f"bi{sample_index}")),
            precomputed_action_chunk_norm=correction_raw_batch["act_action_chunk"][sample_index],
        )
        if correction is None:
            skip_meta = correction_builder.pop_last_skip_meta()
            skip_reason = "unknown_skip"
            if isinstance(skip_meta, dict) and skip_meta.get("skip_reason") is not None:
                skip_reason = str(skip_meta["skip_reason"]).strip() or "unknown_skip"
            skip_reasons[skip_reason] += 1
            continue
        if correction["corr_image"] is None or correction["corr_qpos_norm"] is None or correction["corr_action_chunk_norm"] is None:
            skip_reasons["missing_correction_targets"] += 1
            continue

        corr_qpos_raw = correction["corr_qpos_norm"] * qpos_std + qpos_mean
        corr_action_raw = correction["corr_action_chunk_norm"] * action_std.view(1, -1) + action_mean.view(1, -1)
        corr_image = correction["corr_image"]
        if corr_image.ndim != 4 or int(corr_image.shape[0]) != 1:
            raise ValueError(f"Expected single-camera correction image, got {tuple(corr_image.shape)}")
        correction_data_path = None
        if correction_data_dir is not None and correction_manifest_path is not None and global_step is not None:
            correction_data_path = save_stage1_correction_data(
                save_dir=Path(correction_data_dir),
                manifest_path=Path(correction_manifest_path),
                rank=int(rank),
                global_step=int(global_step),
                sample_index=int(sample_index),
                task_name_full=str(task_name_full),
                episode_id=int(episode_id),
                start_ts=int(start_ts),
                raw_data_dir=str(correction_raw_batch["raw_data_dir"][sample_index]),
                correction_raw_batch=correction_raw_batch,
                correction=correction,
                corr_image=corr_image,
                corr_qpos_raw=corr_qpos_raw,
                corr_action_raw=corr_action_raw,
            )
        smolvla_batch = build_smolvla_batch_from_raw_sample(
            latent_policy=latent_model,
            preprocess=preprocess,
            instruction_type=instruction_type,
            task_name_full=task_name_full,
            image_t=corr_image.detach().cpu(),
            qpos_raw=corr_qpos_raw.detach().cpu(),
            action_chunk_raw=corr_action_raw.detach().cpu(),
            episode_id=episode_id,
        )
        samples.append(smolvla_batch)
        # For correction samples, the first prefix_steps actions pull the
        # perturbed state back to the original start_ts state, so the stage1
        # dynamics/condition target should be the clean start_ts image.
        future_images.append(correction_raw_batch["image_t"][sample_index, 0].detach().cpu())
        if correction["error_action_prefix_norm"] is not None:
            action_prefix.append(correction["error_action_prefix_norm"][None, ...])
        else:
            action_prefix.append(smolvla_batch["action"][:, : int(args.prefix_steps), :])
        projection_debug_records.append(
            {
                "sample_type": "correction",
                "image_t": corr_image[0].detach().cpu(),
                "action_norm": correction["corr_action_chunk_norm"].detach().cpu(),
                "act_is_pad": torch.zeros(
                    int(correction["corr_action_chunk_norm"].shape[0]),
                    dtype=torch.bool,
                ),
                "raw_data_dir": str(correction_raw_batch["raw_data_dir"][sample_index]),
                "episode_id": int(episode_id),
                "start_ts": int(start_ts),
                "correction_data_path": correction_data_path,
            }
        )
    return samples, future_images, action_prefix, skip_reasons, projection_debug_records


def run_stage1(args: argparse.Namespace) -> None:
    accelerator = build_accelerator()
    is_main = bool(accelerator.is_main_process)
    args.device = str(accelerator.device)
    output_dir = Path(args.output_dir).resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    else:
        tb_writer = None

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    dataset, _ = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    failure_mode = str(args.failure_mode).strip().lower()
    use_failure_corr_loader = bool(failure_mode == "train")
    if use_failure_corr_loader:
        if not str(args.failure_table_paths_json).strip():
            raise ValueError("stage1 failure_mode=train requires --failure_table_paths_json")
        for path_arg, name in (
            (args.urdf_path, "urdf_path"),
            (args.curobo_left_yml, "curobo_left_yml"),
            (args.curobo_right_yml, "curobo_right_yml"),
        ):
            if not str(path_arg).strip():
                raise ValueError(f"stage1 failure_mode=train requires --{name}")
        failure_table_paths = load_failure_table_paths(args.failure_table_paths_json)
        failure_task_names = list(args.failure_task_names or [])
        failure_task_specs = (
            resolve_multitask_specs(failure_task_names, SIM_TASK_CONFIGS, raw_data_root_overrides=None)
            if failure_task_names else task_specs
        )
        corr_batch_size = int(max(1, round(float(args.batch_size) * float(args.failure_corr_batch_ratio))))
        evac_sample_size = load_evac_sample_size(args.evac_config)
        failure_cfg = MultiTaskFailureDatasetConfig(
            act_chunk_size=int(args.act_chunk_size),
            prefix_steps=int(args.prefix_steps),
            future_offset=int(args.future_offset),
            sample_phase_window_len=int(args.sample_phase_window_len),
            start_margin=int(args.start_margin),
            perturb_eef_fail_gain=float(args.act_aligned_perturb_eef_fail_gain),
            perturb_rot_max_deg=float(args.act_aligned_perturb_rot_max_deg),
            evac_sample_size=evac_sample_size,
            failure_table_paths={str(key): str(value) for key, value in failure_table_paths.items()},
            failure_phase_bins=int(args.failure_phase_bins),
            failure_translation_dir_bins=int(args.failure_translation_dir_bins),
            failure_translation_mag_bins=int(args.failure_translation_mag_bins),
            failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
            failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
            failure_explore_k=int(args.failure_explore_k),
        )
        correction_dataset, norm_stats = build_multitask_failure_dataset(
            task_specs=failure_task_specs,
            config=failure_cfg,
            mode="train",
        )
        correction_loader = DataLoader(
            correction_dataset,
            batch_size=corr_batch_size,
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    else:
        failure_table_paths = {}
        correction_dataset = None
        correction_loader = None
        norm_stats = dataset.stats if hasattr(dataset, "stats") else None

    base_policy, preprocess, postprocess = build_base_policy(args)
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_stage1_warmup_config(args),
    )
    model.to(torch.device(args.device))
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
        load_rollout_model=use_failure_corr_loader,
    )

    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=dataset[0],
        teacher=teacher,
    )
    correction_builder = None
    if use_failure_corr_loader:
        correction_builder = ACTAlignedCorrectionBuilder(
            cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
            urdf_path=args.urdf_path,
            curobo_left_yml=args.curobo_left_yml,
            curobo_right_yml=args.curobo_right_yml,
            device=args.device,
            shared_evac_model=teacher.model,
            shared_evac_config=teacher.cfg,
        )

    optimizer_cfg, optimizer, lr_scheduler = configure_optimizer(
        model=model,
        args=args,
        total_training_steps=int(args.max_steps),
    )
    if use_failure_corr_loader:
        model, optimizer, dataloader, correction_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, correction_loader, lr_scheduler
        )
    else:
        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    latent_model = accelerator.unwrap_model(model)
    if is_main:
        save_train_args(output_dir, args)

    correction_iter = iter(correction_loader) if use_failure_corr_loader else None
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}
    skip_reason_counter: Counter[str] = Counter()
    debug_wm = bool(getattr(args, "debug_wm_correction", False))
    debug_wm_all_ranks = bool(getattr(args, "debug_wm_all_ranks", False))
    debug_loss_batch_projection = bool(getattr(args, "debug_loss_batch_projection", False))
    debug_wm_root = (output_dir / "debug_wm") if debug_wm else None
    debug_wm_dir = None
    if debug_wm_root is not None:
        debug_wm_dir = (
            debug_wm_root / f"rank{int(accelerator.process_index):02d}"
            if debug_wm_all_ranks else debug_wm_root
        )
    debug_wm_should_save = bool(debug_wm_dir is not None and (debug_wm_all_ranks or accelerator.is_main_process))
    if debug_wm_should_save:
        debug_wm_dir.mkdir(parents=True, exist_ok=True)
    save_correction_data = bool(getattr(args, "save_correction_data", False)) and use_failure_corr_loader
    correction_data_dir = None
    correction_data_manifest_path = None
    if save_correction_data:
        correction_data_dir = output_dir / "correction_data" / f"rank{int(accelerator.process_index):02d}"
        correction_data_dir.mkdir(parents=True, exist_ok=True)
        correction_data_manifest_path = output_dir / f"correction_data_manifest_rank{int(accelerator.process_index):02d}.jsonl"
    correction_trace_path = output_dir / f"stage1_correction_trace_rank{int(accelerator.process_index):02d}.jsonl"
    global_step, epoch = resume_training_checkpoint(
        checkpoint_path=str(args.resume_ckpt),
        model=latent_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    max_steps = int(args.max_steps)
    progress = tqdm(total=max_steps, initial=global_step, desc="stage1", leave=True, disable=not is_main)
    try:
        while global_step < max_steps:
            for raw_batch in dataloader:
                samples, future_images, action_prefix = build_stage1_samples_from_raw_batch(
                    raw_batch=raw_batch,
                    latent_model=latent_model,
                    preprocess=preprocess,
                    instruction_type=args.instruction_type,
                    prefix_steps=int(args.prefix_steps),
                )
                base_batch_size = int(raw_batch["image_t"].shape[0])
                clean_action_abs_mean = _mean_action_abs(raw_batch)
                corr_requested = 0
                corr_source_action_abs_mean = float("nan")
                corr_generated_action_abs_mean = float("nan")
                corr_trace_records: list[dict[str, Any]] = []
                corr_skip_reasons: Counter[str] = Counter()
                if use_failure_corr_loader:
                    assert correction_iter is not None
                    assert correction_builder is not None
                    assert norm_stats is not None
                    try:
                        correction_raw_batch = next(correction_iter)
                    except StopIteration:
                        correction_iter = iter(correction_loader)
                        correction_raw_batch = next(correction_iter)
                    corr_requested = int(correction_raw_batch["image_t"].shape[0])
                    corr_source_action_abs_mean = _mean_action_abs(correction_raw_batch)
                    corr_trace_records = summarize_correction_raw_batch(correction_raw_batch)
                    step_debug_dir = None
                    if debug_wm_should_save:
                        step_debug_dir = str(debug_wm_dir / f"step_{global_step:06d}")
                        Path(step_debug_dir).mkdir(parents=True, exist_ok=True)
                    corr_samples, corr_future_images, corr_action_prefix, corr_skip_reasons, corr_projection_debug_records = build_stage1_correction_samples(
                        correction_raw_batch=correction_raw_batch,
                        latent_model=latent_model,
                        preprocess=preprocess,
                        postprocess=postprocess,
                        instruction_type=args.instruction_type,
                        args=args,
                        correction_builder=correction_builder,
                        teacher=teacher,
                        norm_stats=norm_stats,
                        raw_cache=raw_cache,
                        step_debug_dir=step_debug_dir,
                        correction_data_dir=(None if correction_data_dir is None else str(correction_data_dir)),
                        correction_manifest_path=(
                            None if correction_data_manifest_path is None else str(correction_data_manifest_path)
                        ),
                        global_step=int(global_step),
                        rank=int(accelerator.process_index),
                    )
                    skip_reason_counter.update(corr_skip_reasons)
                    corr_skip_reasons = Counter(corr_skip_reasons)
                    samples.extend(corr_samples)
                    future_images.extend(corr_future_images)
                    action_prefix.extend(corr_action_prefix)
                    if corr_projection_debug_records:
                        corr_generated_action_abs_mean = float(
                            np.mean(
                                [
                                    float(record["action_norm"].detach().float().abs().mean().item())
                                    for record in corr_projection_debug_records
                                ]
                            )
                        )
                    if not corr_samples:
                        trace_payload = {
                            "step": int(global_step),
                            "rank": int(accelerator.process_index),
                            "skipped_train_step": True,
                            "base_batch_size": int(base_batch_size),
                            "corr_requested": int(corr_requested),
                            "corr_generated": 0,
                            "corr_skipped": int(sum(corr_skip_reasons.values())),
                            "skip_reasons": dict(corr_skip_reasons),
                            "clean_action_abs_mean_norm": float(clean_action_abs_mean),
                            "corr_source_action_abs_mean_norm": float(corr_source_action_abs_mean),
                            "requested_units": corr_trace_records,
                        }
                        with open(correction_trace_path, "a", encoding="utf-8") as trace_f:
                            trace_f.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
                        continue
                else:
                    step_debug_dir = None
                    corr_projection_debug_records = []

                batch = stack_smolvla_batches(samples)
                future_image_tensor = torch.stack(future_images, dim=0).to(device=accelerator.device, dtype=torch.float32)
                future_teacher_latent = teacher.encode_image(future_image_tensor)
                action_prefix_tensor = torch.cat(action_prefix, dim=0).to(device=accelerator.device)

                if debug_loss_batch_projection and step_debug_dir is not None and correction_builder is not None:
                    loss_dbg_dir = Path(step_debug_dir) / "loss_batch_projection"
                    loss_dbg_dir.mkdir(parents=True, exist_ok=True)
                    fk = correction_builder.modules["fk"]

                    def _get_raw(raw_data_dir_i: str, episode_id_i: int) -> dict[str, Any]:
                        cache_key = (str(raw_data_dir_i), int(episode_id_i))
                        if cache_key not in raw_cache:
                            raw_cache[cache_key] = load_raw_episode(str(raw_data_dir_i), int(episode_id_i))
                        return raw_cache[cache_key]

                    for i in range(int(raw_batch["image_t"].shape[0])):
                        ep_i = scalar_to_int(raw_batch["episode_id"][i])
                        st_i = scalar_to_int(raw_batch["start_ts"][i])
                        raw_dir_i = str(raw_batch["raw_data_dir"][i])
                        _save_loss_batch_projection(
                            str(loss_dbg_dir / f"base_{i:03d}_ep{ep_i}_ts{st_i:04d}"),
                            raw_batch["image_t"][i, 0],
                            raw_batch["act_action_chunk"][i],
                            raw_batch["act_is_pad"][i],
                            _get_raw(raw_dir_i, ep_i),
                            fk,
                            norm_stats,
                            meta={"sample_type": "base", "episode_id": ep_i, "start_ts": st_i},
                        )

                    for j, record in enumerate(corr_projection_debug_records):
                        ep_j = int(record["episode_id"])
                        st_j = int(record["start_ts"])
                        _save_loss_batch_projection(
                            str(loss_dbg_dir / f"corr_{j:03d}_ep{ep_j}_ts{st_j:04d}"),
                            record["image_t"],
                            record["action_norm"],
                            record["act_is_pad"],
                            _get_raw(str(record["raw_data_dir"]), ep_j),
                            fk,
                            norm_stats,
                            meta={"sample_type": str(record["sample_type"]), "episode_id": ep_j, "start_ts": st_j},
                        )

                output = model(
                    train_stage="stage1",
                    batch=batch,
                    future_teacher_latent=future_teacher_latent,
                    action_prefix=action_prefix_tensor,
                    global_step=global_step,
                )
                clean_action_loss_eval, corr_action_loss_eval = compute_action_loss_split(
                    latent_model,
                    batch,
                    base_batch_size=base_batch_size,
                    total_batch_size=len(samples),
                )
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(output.loss)
                if optimizer_cfg.grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), optimizer_cfg.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()

                current_lr = float(optimizer.param_groups[0]["lr"])
                if tb_writer is not None:
                    log_stage1_tensorboard(
                        writer=tb_writer,
                        step=global_step,
                        output=output,
                        lr=current_lr,
                    )
                    log_stage1_extra_tensorboard(
                        writer=tb_writer,
                        step=global_step,
                        base_batch_size=base_batch_size,
                        corr_requested=corr_requested,
                        corr_generated=(len(samples) - base_batch_size),
                        corr_skipped=int(sum(corr_skip_reasons.values())),
                        clean_action_abs_mean=clean_action_abs_mean,
                        corr_source_action_abs_mean=corr_source_action_abs_mean,
                        corr_generated_action_abs_mean=corr_generated_action_abs_mean,
                        clean_action_loss_eval=clean_action_loss_eval,
                        corr_action_loss_eval=corr_action_loss_eval,
                    )
                if use_failure_corr_loader:
                    trace_payload = {
                        "step": int(global_step),
                        "rank": int(accelerator.process_index),
                        "base_batch_size": int(base_batch_size),
                        "corr_requested": int(corr_requested),
                        "corr_generated": int(len(samples) - base_batch_size),
                        "corr_skipped": int(sum(corr_skip_reasons.values())),
                        "skip_reasons": dict(corr_skip_reasons),
                        "clean_action_abs_mean_norm": float(clean_action_abs_mean),
                        "corr_source_action_abs_mean_norm": float(corr_source_action_abs_mean),
                        "corr_generated_action_abs_mean_norm": float(corr_generated_action_abs_mean),
                        "clean_action_loss_eval": float(clean_action_loss_eval),
                        "corr_action_loss_eval": float(corr_action_loss_eval),
                        "loss": float(output.loss.item()),
                        "loss_action": float(output.loss_action.item()),
                        "loss_action_conditioned": float(output.loss_action_conditioned.item()),
                        "loss_dynamics": float(output.loss_dynamics.item()),
                        "beta_condition": float(output.beta_condition),
                        "beta_dynamics": float(output.beta_dynamics),
                        "lr": float(current_lr),
                        "requested_units": corr_trace_records,
                    }
                    with open(correction_trace_path, "a", encoding="utf-8") as trace_f:
                        trace_f.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")

                progress.set_postfix(
                    loss=f"{output.loss.item():.4f}",
                    action=f"{output.loss_action.item():.4f}",
                    cond=f"{output.loss_action_conditioned.item():.4f}",
                    dyn=f"{output.loss_dynamics.item():.4f}",
                    beta_cond=f"{output.beta_condition:.4f}",
                    beta=f"{output.beta_dynamics:.4f}",
                    lr=f"{current_lr:.2e}",
                    corr=(len(samples) - int(raw_batch["image_t"].shape[0])),
                    skip=(skip_reason_counter.most_common(1)[0][0] if skip_reason_counter else "none"),
                    step=global_step,
                )
                global_step += 1
                progress.update(1)
                if global_step % int(args.save_freq) == 0 and is_main:
                    save_training_checkpoint(
                        output_dir=output_dir,
                        prefix="stage1",
                        step=global_step,
                        model=latent_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        extra_state={
                            "norm_stats": dataset.stats if hasattr(dataset, "stats") else None,
                            "target_hw": latent_model.target_hw,
                            "failure_table_paths": failure_table_paths,
                            "epoch": epoch,
                            "args": vars(args),
                        },
                    )
                if global_step >= max_steps:
                    break
            epoch += 1
        if global_step % int(args.save_freq) != 0 and is_main:
            save_training_checkpoint(
                output_dir=output_dir,
                prefix="stage1",
                step=global_step,
                model=latent_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                extra_state={
                    "norm_stats": dataset.stats if hasattr(dataset, "stats") else None,
                    "target_hw": latent_model.target_hw,
                    "failure_table_paths": failure_table_paths,
                    "epoch": epoch,
                    "args": vars(args),
                },
            )
    finally:
        progress.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


def run_stage2(args: argparse.Namespace) -> None:
    accelerator = build_accelerator()
    is_main = bool(accelerator.is_main_process)
    args.device = str(accelerator.device)
    output_dir = Path(args.output_dir).resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    else:
        tb_writer = None

    failure_table_paths = load_failure_table_paths(args.failure_table_paths_json)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    normal_dataset, norm_stats = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
    )
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    evac_sample_size = load_evac_sample_size(args.evac_config)
    failure_cfg = MultiTaskFailureDatasetConfig(
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
        sample_phase_window_len=int(args.sample_phase_window_len),
        start_margin=int(args.start_margin),
        perturb_eef_fail_gain=float(args.act_aligned_perturb_eef_fail_gain),
        perturb_rot_max_deg=float(args.act_aligned_perturb_rot_max_deg),
        evac_sample_size=evac_sample_size,
        failure_table_paths={str(key): str(value) for key, value in failure_table_paths.items()},
        failure_phase_bins=int(args.failure_phase_bins),
        failure_translation_dir_bins=int(args.failure_translation_dir_bins),
        failure_translation_mag_bins=int(args.failure_translation_mag_bins),
        failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
        failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
        failure_explore_k=int(args.failure_explore_k),
    )
    correction_dataset, _ = build_multitask_failure_dataset(task_specs=task_specs, config=failure_cfg, mode="train")
    correction_loader = DataLoader(
        correction_dataset,
        batch_size=int(args.correction_batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    base_policy, preprocess, postprocess = build_base_policy(args)
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_warmup_config(args),
    )
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
        load_rollout_model=True,
    )
    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=normal_dataset[0],
        teacher=teacher,
    )
    stage1_checkpoint = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(stage1_checkpoint["model"], strict=True)
    model.to(torch.device(args.device))
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=args.device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )
    optimizer_cfg, optimizer, lr_scheduler = configure_optimizer(
        model=model,
        args=args,
        total_training_steps=int(args.max_steps),
    )
    model, optimizer, normal_loader, correction_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, normal_loader, correction_loader, lr_scheduler
    )
    latent_model = accelerator.unwrap_model(model)
    if is_main:
        save_train_args(output_dir, args)

    qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=args.device)
    qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=args.device)
    action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=args.device)
    action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=args.device)
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}

    correction_iter = iter(correction_loader)
    global_step, epoch = resume_training_checkpoint(
        checkpoint_path=str(args.resume_ckpt),
        model=latent_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    max_steps = int(args.max_steps)
    skip_reason_counter: Counter[str] = Counter()
    progress = tqdm(total=max_steps, initial=global_step, desc="stage2", leave=True, disable=not is_main)
    try:
        while global_step < max_steps:
            for normal_raw_batch in normal_loader:
                try:
                    correction_raw_batch = next(correction_iter)
                except StopIteration:
                    correction_iter = iter(correction_loader)
                    correction_raw_batch = next(correction_iter)

                normal_samples = []
                for sample_index in range(normal_raw_batch["image_t"].shape[0]):
                    normal_samples.append(
                        build_smolvla_batch_from_raw_sample(
                            latent_policy=latent_model,
                            preprocess=preprocess,
                            instruction_type=args.instruction_type,
                            task_name_full=normal_raw_batch["task_name"][sample_index],
                            image_t=normal_raw_batch["image_t"][sample_index],
                            qpos_raw=normal_raw_batch["qpos_raw"][sample_index],
                            action_chunk_raw=normal_raw_batch["act_action_chunk_raw"][sample_index],
                            episode_id=scalar_to_int(normal_raw_batch["episode_id"][sample_index]),
                        )
                    )
                normal_batch = stack_smolvla_batches(normal_samples)

                correction_samples = []
                correction_teacher_latents = []
                rollout_teacher_latents = []
                correction_actions = []
                correction_action_prefixes = []

                for sample_index in range(correction_raw_batch["image_t"].shape[0]):
                    task_name_full = correction_raw_batch["task_name"][sample_index]
                    task_only, task_config = parse_task_parts(task_name_full)
                    episode_id = scalar_to_int(correction_raw_batch["episode_id"][sample_index])
                    adapter = SampleBoundSmolVLAAdapter(
                        latent_policy=latent_model,
                        preprocess=preprocess,
                        postprocess=postprocess,
                        task_name=task_only,
                        task_config=task_config,
                        episode_id=episode_id,
                        instruction_type=args.instruction_type,
                    )
                    raw_cache_key = (str(correction_raw_batch["raw_data_dir"][sample_index]), episode_id)
                    if raw_cache_key not in raw_cache:
                        raw_cache[raw_cache_key] = load_raw_episode(
                            correction_raw_batch["raw_data_dir"][sample_index],
                            episode_id,
                        )
                    correction = correction_builder.build(
                        latent_model=adapter,
                        image_t=correction_raw_batch["image_t"][sample_index].to(args.device),
                        qpos_t=correction_raw_batch["qpos_t"][sample_index].to(args.device),
                        raw_data=raw_cache[raw_cache_key],
                        norm_stats=norm_stats,
                        start_ts=scalar_to_int(correction_raw_batch["start_ts"][sample_index]),
                        failure_mode_override="train",
                        sampled_phase_id=int(correction_raw_batch["sampled_phase_id"][sample_index].item()),
                        sampled_phase_bin_id=int(correction_raw_batch["sampled_phase_bin_id"][sample_index].item()),
                        sampled_phase_instance_id=int(correction_raw_batch["sampled_phase_instance_id"][sample_index].item()),
                        forced_error_mode_id=int(correction_raw_batch["forced_error_mode_id"][sample_index].item()),
                        sampled_active_arm_pattern_id=int(correction_raw_batch["sampled_active_arm_pattern_id"][sample_index].item()),
                        original_active_arm_pattern_id=int(correction_raw_batch["original_active_arm_pattern_id"][sample_index].item()),
                        forced_dir_bin_id=int(correction_raw_batch["forced_dir_bin_id"][sample_index].item()),
                        forced_mag_bin_id=int(correction_raw_batch["forced_mag_bin_id"][sample_index].item()),
                        sampled_mode_prob=float(correction_raw_batch["sampled_mode_prob"][sample_index].item()),
                        sampled_entry_prob_within_mode=float(correction_raw_batch["sampled_entry_prob_within_mode"][sample_index].item()),
                        sampled_unit_prob=float(correction_raw_batch["sampled_unit_prob"][sample_index].item()),
                        precomputed_action_chunk_norm=correction_raw_batch["act_action_chunk"][sample_index],
                    )
                    if correction is None:
                        skip_meta = correction_builder.pop_last_skip_meta()
                        skip_reason = "unknown_skip"
                        if isinstance(skip_meta, dict) and skip_meta.get("skip_reason") is not None:
                            skip_reason = str(skip_meta["skip_reason"]).strip() or "unknown_skip"
                        skip_reason_counter[skip_reason] += 1
                        continue
                    if correction["corr_image"] is None or correction["corr_qpos_norm"] is None or correction["corr_action_chunk_norm"] is None:
                        skip_reason_counter["missing_correction_targets"] += 1
                        continue

                    corr_qpos_raw = correction["corr_qpos_norm"] * qpos_std + qpos_mean
                    corr_action_raw = correction["corr_action_chunk_norm"] * action_std.view(1, -1) + action_mean.view(1, -1)
                    corr_image = correction["corr_image"]
                    if corr_image.ndim != 4 or int(corr_image.shape[0]) != 1:
                        raise ValueError(f"Expected single-camera correction image, got {tuple(corr_image.shape)}")

                    smolvla_batch = build_smolvla_batch_from_raw_sample(
                        latent_policy=latent_model,
                        preprocess=preprocess,
                        instruction_type=args.instruction_type,
                        task_name_full=task_name_full,
                        image_t=corr_image.detach().cpu(),
                        qpos_raw=corr_qpos_raw.detach().cpu(),
                        action_chunk_raw=corr_action_raw.detach().cpu(),
                        episode_id=episode_id,
                    )
                    correction_samples.append(smolvla_batch)
                    correction_actions.append(smolvla_batch["action"])
                    if correction["error_action_prefix_norm"] is not None:
                        correction_action_prefixes.append(correction["error_action_prefix_norm"][None, ...])
                    else:
                        correction_action_prefixes.append(smolvla_batch["action"][:, : int(args.prefix_steps), :])
                    teacher_image = corr_image.to(device=torch.device(args.device), dtype=torch.float32)
                    teacher_latent = teacher.encode_image(teacher_image)
                    correction_teacher_latents.append(teacher_latent)
                    rollout_teacher_latents.append(teacher_latent)

                if not correction_samples:
                    continue

                correction_batch = stack_smolvla_batches(correction_samples)
                correction_actions_tensor = torch.cat(correction_actions, dim=0)
                correction_action_prefix_tensor = torch.cat(correction_action_prefixes, dim=0).to(torch.device(args.device))
                correction_teacher_latent_tensor = torch.cat(correction_teacher_latents, dim=0)
                rollout_teacher_latent_tensor = torch.cat(rollout_teacher_latents, dim=0)

                retain_weight_cur = compute_retain_weight(
                    retain_weight=float(args.retain_weight),
                    retain_weight_final=float(args.retain_weight_final),
                    retain_decay_start_step=float(args.retain_decay_start_step),
                    retain_decay_end_step=float(args.retain_decay_end_step),
                    retain_decay_curve=str(args.retain_decay_curve),
                    step_progress=float(global_step),
                )

                output = model(
                    train_stage="stage2",
                    normal_batch=normal_batch,
                    correction_batch=correction_batch,
                    correction_actions=correction_actions_tensor,
                    correction_action_prefix=correction_action_prefix_tensor,
                    correction_teacher_latent=correction_teacher_latent_tensor,
                    rollout_teacher_latent=rollout_teacher_latent_tensor,
                    global_step=global_step,
                    retain_weight=retain_weight_cur,
                )
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(output.loss)
                if optimizer_cfg.grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), optimizer_cfg.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()

                current_lr = float(optimizer.param_groups[0]["lr"])
                if tb_writer is not None:
                    log_stage2_tensorboard(
                        writer=tb_writer,
                        step=global_step,
                        output=output,
                        lr=current_lr,
                        retain_weight=retain_weight_cur,
                        skipped_batches=sum(skip_reason_counter.values()),
                    )

                progress.set_postfix(
                    loss=f"{output.loss.item():.4f}",
                    correct=f"{output.loss_correct.item():.4f}",
                    retain=f"{output.loss_retain.item():.4f}",
                    dyn=f"{output.loss_dynamics.item():.4f}",
                    beta=f"{output.beta_dynamics:.4f}",
                    rw=f"{retain_weight_cur:.4f}",
                    lr=f"{current_lr:.2e}",
                    skip=(skip_reason_counter.most_common(1)[0][0] if skip_reason_counter else "none"),
                    step=global_step,
                )
                global_step += 1
                progress.update(1)

                if global_step % int(args.save_freq) == 0 and is_main:
                    save_training_checkpoint(
                        output_dir=output_dir,
                        prefix="stage2",
                        step=global_step,
                        model=latent_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        extra_state={
                            "norm_stats": norm_stats,
                            "target_hw": latent_model.target_hw,
                            "failure_table_paths": failure_table_paths,
                            "epoch": epoch,
                            "args": vars(args),
                        },
                    )
                if global_step >= max_steps:
                    break
            epoch += 1
        if global_step % int(args.save_freq) != 0 and is_main:
            save_training_checkpoint(
                output_dir=output_dir,
                prefix="stage2",
                step=global_step,
                model=latent_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                extra_state={
                    "norm_stats": norm_stats,
                    "target_hw": latent_model.target_hw,
                    "failure_table_paths": failure_table_paths,
                    "epoch": epoch,
                    "args": vars(args),
                },
            )
    finally:
        progress.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


def main() -> None:
    args = build_argparser().parse_args()
    if args.train_mode == "stage1":
        run_stage1(args)
        return
    if args.train_mode == "stage2":
        run_stage2(args)
        return
    raise ValueError(f"Unsupported train mode: {args.train_mode}")


if __name__ == "__main__":
    main()
