from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
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
from policy.SmolVLA.latentcorr.smolvla_latent_policy import SmolVLAStage1LossWeights


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


def _shorten_text(value: Any, max_chars: int = 72) -> str:
    text = str(value)
    if len(text) <= int(max_chars):
        return text
    keep = max(8, int(max_chars) - 3)
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def _render_loss_projection_tile(
    image_cam: torch.Tensor,
    action_norm: torch.Tensor,
    is_pad: torch.Tensor,
    raw_data: dict[str, Any],
    fk,
    norm_stats: dict[str, Any],
    meta: dict[str, Any] | None = None,
    *,
    max_image_hw: tuple[int, int] = (240, 320),
    header_h: int = 96,
) -> tuple[np.ndarray, dict[str, Any]]:
    img_rgb = np.clip(
        image_cam.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    overlay = img_rgb[:, :, ::-1].copy()
    h_img, w_img = overlay.shape[:2]
    meta_out: dict[str, Any] = {"image_hw": [int(h_img), int(w_img)]}

    try:
        act = np.asarray(action_norm.detach().cpu().numpy(), dtype=np.float32)
        pad = np.asarray(is_pad.detach().cpu().numpy(), dtype=bool).reshape(-1)
        if act.ndim != 2 or act.shape[1] < 14:
            raise ValueError(f"Expected action shape [T, >=14], got {act.shape}")
        valid = np.where(~pad)[0]
        if valid.size == 0:
            raise ValueError("No valid actions to project")
        act = act[valid]
        act_raw = np.asarray(act * norm_stats["action_std"] + norm_stats["action_mean"], dtype=np.float32)
        meta_out["n_valid_actions"] = int(act_raw.shape[0])

        K = raw_data["intrinsic_cv"].astype(np.float32).copy()
        E = np.eye(4, dtype=np.float32)
        E[:3, :] = raw_data["extrinsic_cv"].astype(np.float32)
        h_native, w_native = raw_data.get("native_resolution", (h_img, w_img))
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
            if not seq:
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
    except Exception as exc:
        meta_out["projection_error"] = repr(exc)

    if isinstance(meta, dict):
        meta_out.update(meta)

    max_h, max_w = int(max_image_hw[0]), int(max_image_hw[1])
    scale = min(float(max_w) / float(max(1, w_img)), float(max_h) / float(max(1, h_img)), 1.0)
    out_w = max(1, int(round(w_img * scale)))
    out_h = max(1, int(round(h_img * scale)))
    overlay_small = cv2.resize(overlay, (out_w, out_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((int(header_h) + out_h, out_w, 3), 245, dtype=np.uint8)
    canvas[int(header_h) :, :, :] = overlay_small

    lines = [
        f"{_shorten_text(meta_out.get('sample_type', 'sample'), 20)} #{meta_out.get('batch_index', '?')}",
        f"task={_shorten_text(meta_out.get('task_name', '?'), 34)}",
        f"ep={meta_out.get('episode_id', '?')} ts={meta_out.get('start_ts', '?')} src_ep={meta_out.get('source_episode_id', '-')}",
        f"n={meta_out.get('n_valid_actions', 0)} path={_shorten_text(meta_out.get('correction_data_path', meta_out.get('raw_data_dir', '')), 38)}",
    ]
    if "projection_error" in meta_out:
        lines[-1] = f"ERROR={_shorten_text(meta_out['projection_error'], 48)}"
    y = 15
    for line in lines:
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
        y += 20
    cv2.rectangle(canvas, (0, int(header_h)), (out_w - 1, int(header_h) + out_h - 1), (80, 80, 80), 1)
    return canvas, meta_out


def _save_loss_batch_projection_grid(
    save_dir: str,
    items: list[dict[str, Any]],
    fk,
    norm_stats: dict[str, Any],
    *,
    max_tile_image_hw: tuple[int, int] = (240, 320),
    max_cols: int = 4,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if not items:
        return
    tiles = []
    metadata = []
    for idx, item in enumerate(items):
        meta = dict(item.get("meta", {}))
        meta["batch_index"] = int(idx)
        tile, meta_out = _render_loss_projection_tile(
            item["image_cam"],
            item["action_norm"],
            item["is_pad"],
            item["raw_data"],
            fk,
            norm_stats,
            meta=meta,
            max_image_hw=max_tile_image_hw,
        )
        tiles.append(tile)
        metadata.append(meta_out)

    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    cols = int(min(max(1, int(max_cols)), max(1, math.ceil(math.sqrt(len(tiles))))))
    rows = int(math.ceil(float(len(tiles)) / float(cols)))
    gap = 8
    grid = np.full((rows * tile_h + (rows + 1) * gap, cols * tile_w + (cols + 1) * gap, 3), 230, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y0 = gap + r * (tile_h + gap)
        x0 = gap + c * (tile_w + gap)
        grid[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    cv2.imwrite(os.path.join(save_dir, "loss_projection_batch.png"), grid)
    with open(os.path.join(save_dir, "loss_projection_batch_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"num_items": len(items), "metadata": metadata}, f, indent=2, ensure_ascii=False)


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
    parser.add_argument(
        "--stage1_latent_target",
        type=str,
        default="future_image",
        choices=["future_image", "wm_rollout"],
        help="Teacher latent source for stage1. wm_rollout is a normal-only ablation target.",
    )
    parser.add_argument("--stage1_rollout_ddim_steps", type=int, default=27)
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
    parser.add_argument("--token_loss_weight_init", type=float, default=0.1)
    parser.add_argument("--token_loss_weight_late", type=float, default=0.02)
    parser.add_argument("--token_loss_decay_start_ratio", type=float, default=0.0)
    parser.add_argument("--token_loss_decay_end_ratio", type=float, default=1.0)
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
    parser.add_argument(
        "--stage1_corr_source",
        type=str,
        default="online",
        choices=["online", "offline_export", "offline_export_mixed_dataset"],
    )
    parser.add_argument("--failure_table_paths_json", type=str, default="")
    parser.add_argument("--failure_task_names", nargs="*", default=None)
    parser.add_argument("--failure_corr_batch_ratio", type=float, default=0.0)
    parser.add_argument("--offline_corr_data_root", type=str, default="/data/zhenyangfan/RoboTwin/data")
    parser.add_argument("--offline_corr_task_config", type=str, default="demo_clean_corr_export_evac_exec16")
    parser.add_argument("--offline_corr_max_samples_per_task", type=int, default=0)
    parser.add_argument("--offline_corr_balance_tasks", type=str2bool, default=True)
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
    parser.add_argument("--debug_loss_batch_projection_freq", type=int, default=1)
    parser.add_argument("--debug_grad_cosine", type=str2bool, default=False)
    parser.add_argument("--debug_grad_cosine_freq", type=int, default=100)
    parser.add_argument("--stage1_corr_outlier_filter", type=str2bool, default=False)
    parser.add_argument("--stage1_corr_outlier_max_action_loss", type=float, default=0.3)
    parser.add_argument("--stage1_corr_prefix_loss_weight", type=float, default=0.0)
    parser.add_argument("--stage1_corr_prefix_loss_steps", type=int, default=0)
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


def build_stage1_loss_weights(args: argparse.Namespace) -> SmolVLAStage1LossWeights:
    return SmolVLAStage1LossWeights(
        token_init=float(args.token_loss_weight_init),
        token_late=float(args.token_loss_weight_late),
        token_decay_start_ratio=float(args.token_loss_decay_start_ratio),
        token_decay_end_ratio=float(args.token_loss_decay_end_ratio),
        total_steps=max(1, int(args.max_steps)),
    )


def _stage1_dataset_stats_for_processors(norm_stats: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    if norm_stats is None:
        return None
    if "qpos_mean" not in norm_stats or "action_mean" not in norm_stats:
        return norm_stats
    return {
        "observation.state": {
            "mean": np.asarray(norm_stats["qpos_mean"], dtype=np.float32),
            "std": np.asarray(norm_stats["qpos_std"], dtype=np.float32),
        },
        "action": {
            "mean": np.asarray(norm_stats["action_mean"], dtype=np.float32),
            "std": np.asarray(norm_stats["action_std"], dtype=np.float32),
        },
    }


def _stage1_adapted_smolvla_config(args: argparse.Namespace, sample: dict[str, Any]) -> PreTrainedConfig:
    config = PreTrainedConfig.from_pretrained(args.smolvla_pretrained_path)
    image_t = sample["image_t"]
    qpos_raw = sample["qpos_raw"]
    action_chunk_raw = sample["act_action_chunk_raw"]
    if not isinstance(image_t, torch.Tensor) or image_t.ndim != 4:
        raise ValueError(f"Expected sample image_t shaped (num_cam,C,H,W), got {type(image_t)!r}")
    if not isinstance(qpos_raw, torch.Tensor) or qpos_raw.ndim != 1:
        raise ValueError(f"Expected sample qpos_raw shaped (D,), got {type(qpos_raw)!r}")
    if not isinstance(action_chunk_raw, torch.Tensor) or action_chunk_raw.ndim != 2:
        raise ValueError(f"Expected sample act_action_chunk_raw shaped (T,A), got {type(action_chunk_raw)!r}")

    image_shape = tuple(int(v) for v in image_t.shape[1:])
    visual_features = {
        f"observation.images.camera{cam_index + 1}": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=image_shape,
        )
        for cam_index in range(int(image_t.shape[0]))
    }
    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(int(qpos_raw.shape[0]),),
        ),
        **visual_features,
    }
    config.output_features = {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION,
            shape=(int(action_chunk_raw.shape[1]),),
        )
    }
    config.chunk_size = int(args.act_chunk_size)
    config.n_action_steps = int(args.act_chunk_size)
    config.freeze_vision_encoder = parse_bool_flag(args.freeze_vision_encoder)
    config.train_expert_only = parse_bool_flag(args.train_expert_only)
    config.load_vlm_weights = parse_bool_flag(args.load_vlm_weights)
    config.device = str(args.device)
    return config


def build_base_policy(
    args: argparse.Namespace,
    dataset_sample: dict[str, Any] | None = None,
    dataset_stats: dict[str, Any] | None = None,
) -> tuple[SmolVLAPolicy, Any, Any]:
    config = _stage1_adapted_smolvla_config(args, dataset_sample) if dataset_sample is not None else None
    base_policy = SmolVLAPolicy.from_pretrained(args.smolvla_pretrained_path, config=config)
    base_policy.config.freeze_vision_encoder = parse_bool_flag(args.freeze_vision_encoder)
    base_policy.config.train_expert_only = parse_bool_flag(args.train_expert_only)
    base_policy.config.load_vlm_weights = parse_bool_flag(args.load_vlm_weights)
    base_policy.model.vlm_with_expert.freeze_vision_encoder = base_policy.config.freeze_vision_encoder
    base_policy.model.vlm_with_expert.train_expert_only = base_policy.config.train_expert_only
    base_policy.model.vlm_with_expert.set_requires_grad()
    base_policy.model.set_requires_grad()
    base_policy.to(torch.device(args.device))
    preprocess, postprocess = make_smolvla_processors(
        base_policy,
        args.smolvla_pretrained_path,
        dataset_stats=_stage1_dataset_stats_for_processors(dataset_stats),
    )
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
    writer.add_scalar("train/loss_condition_token", float(output.loss_condition_token.item()), step)
    writer.add_scalar("train/beta_condition", float(output.beta_condition), step)
    writer.add_scalar("train/beta_dynamics", float(output.beta_dynamics), step)
    writer.add_scalar("train/beta_token", float(output.beta_token), step)
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


def _mean_action_abs_for_indices(raw_batch: dict[str, Any], indices: list[int]) -> float:
    action = raw_batch.get("act_action_chunk")
    if not indices or not isinstance(action, torch.Tensor):
        return float("nan")
    index_tensor = torch.as_tensor(indices, device=action.device, dtype=torch.long)
    subset = action.index_select(0, index_tensor)
    is_pad = raw_batch.get("act_is_pad")
    subset_pad = is_pad.index_select(0, index_tensor) if isinstance(is_pad, torch.Tensor) else None
    return _mean_action_abs({"act_action_chunk": subset, "act_is_pad": subset_pad})


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


def _index_smolvla_batch(batch: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    if ACTION not in batch:
        raise KeyError(f"Missing required key: {ACTION}")
    index_tensor = torch.as_tensor(indices, device=batch[ACTION].device, dtype=torch.long)
    indexed: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == batch[ACTION].shape[:1]:
            indexed[key] = value.index_select(0, index_tensor)
        elif isinstance(value, list) and len(value) == batch[ACTION].shape[0]:
            indexed[key] = [value[int(i)] for i in indices]
        else:
            indexed[key] = value
    return indexed


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


@torch.no_grad()
def compute_action_loss_by_mask(
    latent_model: SmolVLALatentPolicy,
    batch: dict[str, Any],
    *,
    clean_indices: list[int],
    corr_indices: list[int],
) -> tuple[float, float]:
    if ACTION not in batch:
        return float("nan"), float("nan")

    def _index_batch(indices: list[int]) -> dict[str, Any]:
        return _index_smolvla_batch(batch, indices)

    clean_loss = float("nan")
    corr_loss = float("nan")
    was_training = bool(latent_model.training)
    latent_model.eval()
    try:
        if clean_indices:
            clean_batch = _index_batch(clean_indices)
            clean_loss = float(
                latent_model._action_loss(batch=clean_batch, actions=clean_batch[ACTION]).detach().float().item()
            )
        if corr_indices:
            corr_batch = _index_batch(corr_indices)
            corr_loss = float(
                latent_model._action_loss(batch=corr_batch, actions=corr_batch[ACTION]).detach().float().item()
            )
    finally:
        if was_training:
            latent_model.train()
    return clean_loss, corr_loss


@torch.no_grad()
def compute_action_loss_per_sample(
    latent_model: SmolVLALatentPolicy,
    samples: list[dict[str, Any]],
) -> list[float]:
    if not samples:
        return []
    was_training = bool(latent_model.training)
    latent_model.eval()
    try:
        batch = stack_smolvla_batches(samples)
        images, img_masks, lang_tokens, lang_masks, state = latent_model._policy_inputs_from_batch(batch)
        action_batch = {ACTION: batch[ACTION]}
        actions_padded = latent_model.base_policy.prepare_action(action_batch)
        losses = latent_model.base_policy.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions_padded,
        )
        per_sample = losses[:, :, : latent_model.base_policy.config.max_action_dim].mean(dim=(1, 2))
        return [float(value.detach().float().cpu().item()) for value in per_sample]
    finally:
        if was_training:
            latent_model.train()


def filter_stage1_correction_outliers(
    latent_model: SmolVLALatentPolicy,
    corr_samples: list[dict[str, Any]],
    corr_future_images: list[torch.Tensor],
    corr_action_prefix: list[torch.Tensor],
    corr_projection_debug_records: list[dict[str, Any]],
    *,
    max_action_loss: float,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    if not corr_samples:
        return corr_samples, corr_future_images, corr_action_prefix, corr_projection_debug_records, [], []
    losses = compute_action_loss_per_sample(latent_model, corr_samples)
    if len(losses) != len(corr_samples):
        raise RuntimeError(f"Expected {len(corr_samples)} correction losses, got {len(losses)}")

    kept_samples: list[dict[str, Any]] = []
    kept_future_images: list[torch.Tensor] = []
    kept_action_prefix: list[torch.Tensor] = []
    kept_projection_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    filtered_losses: list[float] = []
    threshold = float(max_action_loss)
    for sample_index, action_loss in enumerate(losses):
        keep = bool(np.isfinite(action_loss) and float(action_loss) <= threshold)
        projection_record = dict(corr_projection_debug_records[sample_index]) if sample_index < len(corr_projection_debug_records) else {}
        if keep:
            kept_samples.append(corr_samples[sample_index])
            kept_future_images.append(corr_future_images[sample_index])
            kept_action_prefix.append(corr_action_prefix[sample_index])
            projection_record["outlier_action_loss_eval"] = float(action_loss)
            kept_projection_records.append(projection_record)
            continue

        removed_record = {
            "sample_index": int(sample_index),
            "skip_reason": "outlier_action_loss",
            "action_loss_eval": float(action_loss),
            "max_action_loss": threshold,
        }
        for key, value in projection_record.items():
            if key in {"image_t", "action_norm", "act_is_pad"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                removed_record[key] = value
        removed_records.append(removed_record)
        filtered_losses.append(float(action_loss))
    return (
        kept_samples,
        kept_future_images,
        kept_action_prefix,
        kept_projection_records,
        removed_records,
        filtered_losses,
    )


def _zero_model_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _grad_probe_param_group_name(name: str) -> str:
    if name.startswith("base_policy."):
        if ".action_" in name or name.endswith("action_in_proj.weight") or name.endswith("action_out_proj.weight"):
            return "base_action_proj"
        if ".lm_expert." in name:
            return "base_lm_expert"
        if ".state_proj." in name:
            return "base_state_proj"
        if ".vision_model." in name:
            return "base_vision"
        if ".vlm." in name:
            return "base_vlm"
        return "base_other"
    if name.startswith("projector."):
        return "projector"
    if name.startswith("wm_adapter."):
        return "wm_adapter"
    if name.startswith("predictor."):
        return "predictor"
    if name.startswith("condition_proj."):
        return "condition_proj"
    return "other"


def _cosine_from_stats(dot: float, norm_a: float, norm_b: float) -> float:
    denom = float(norm_a) * float(norm_b)
    if denom <= 1e-20:
        return float("nan")
    return float(dot) / denom


def _stage1_probe_loss(
    latent_model: SmolVLALatentPolicy,
    *,
    loss_kind: str,
    batch: dict[str, Any],
    future_teacher_latent: torch.Tensor,
    action_prefix: torch.Tensor,
    global_step: int,
) -> torch.Tensor:
    if loss_kind == "action":
        return latent_model._action_loss(batch=batch, actions=batch[ACTION])
    if loss_kind == "total":
        return latent_model.compute_stage1_loss(
            batch=batch,
            future_teacher_latent=future_teacher_latent,
            action_prefix=action_prefix,
            global_step=int(global_step),
        ).loss
    raise ValueError(f"Unsupported stage1 grad probe loss kind: {loss_kind}")


def _stage1_grad_pair_metrics(
    latent_model: SmolVLALatentPolicy,
    *,
    clean_batch: dict[str, Any],
    corr_batch: dict[str, Any],
    clean_future_teacher_latent: torch.Tensor,
    corr_future_teacher_latent: torch.Tensor,
    clean_action_prefix: torch.Tensor,
    corr_action_prefix: torch.Tensor,
    global_step: int,
    loss_kind: str,
) -> dict[str, Any]:
    named_params = [(name, param) for name, param in latent_model.named_parameters() if param.requires_grad]
    if not named_params:
        return {
            "loss_kind": str(loss_kind),
            "clean_loss": float("nan"),
            "corr_loss": float("nan"),
            "cosine": float("nan"),
            "clean_norm": float("nan"),
            "corr_norm": float("nan"),
        }

    was_training = bool(latent_model.training)
    latent_model.eval()
    _zero_model_grads(latent_model)
    try:
        clean_loss = _stage1_probe_loss(
            latent_model,
            loss_kind=loss_kind,
            batch=clean_batch,
            future_teacher_latent=clean_future_teacher_latent.detach(),
            action_prefix=clean_action_prefix.detach(),
            global_step=int(global_step),
        )
        if not bool(clean_loss.requires_grad):
            return {
                "loss_kind": str(loss_kind),
                "clean_loss": float(clean_loss.detach().float().item()),
                "corr_loss": float("nan"),
                "cosine": float("nan"),
                "clean_norm": float("nan"),
                "corr_norm": float("nan"),
            }
        params = [param for _, param in named_params]
        clean_grads = torch.autograd.grad(
            clean_loss,
            params,
            allow_unused=True,
        )

        corr_loss = _stage1_probe_loss(
            latent_model,
            loss_kind=loss_kind,
            batch=corr_batch,
            future_teacher_latent=corr_future_teacher_latent.detach(),
            action_prefix=corr_action_prefix.detach(),
            global_step=int(global_step),
        )
        corr_grads = torch.autograd.grad(
            corr_loss,
            params,
            allow_unused=True,
        )

        device = next((param.device for _, param in named_params), torch.device("cpu"))
        dot = torch.zeros((), device=device, dtype=torch.float32)
        clean_sq = torch.zeros((), device=device, dtype=torch.float32)
        corr_sq = torch.zeros((), device=device, dtype=torch.float32)
        group_stats: dict[str, dict[str, torch.Tensor]] = {}
        for (name, _param), clean_grad, corr_grad in zip(named_params, clean_grads, corr_grads):
            if clean_grad is None and corr_grad is None:
                continue
            group = _grad_probe_param_group_name(name)
            if group not in group_stats:
                group_stats[group] = {
                    "dot": torch.zeros((), device=device, dtype=torch.float32),
                    "clean_sq": torch.zeros((), device=device, dtype=torch.float32),
                    "corr_sq": torch.zeros((), device=device, dtype=torch.float32),
                }
            if clean_grad is not None:
                clean_flat = clean_grad.detach().float().reshape(-1)
                clean_val = torch.sum(clean_flat * clean_flat)
                clean_sq = clean_sq + clean_val
                group_stats[group]["clean_sq"] = group_stats[group]["clean_sq"] + clean_val
            else:
                clean_flat = None
            if corr_grad is not None:
                corr_flat = corr_grad.detach().float().reshape(-1)
                corr_val = torch.sum(corr_flat * corr_flat)
                corr_sq = corr_sq + corr_val
                group_stats[group]["corr_sq"] = group_stats[group]["corr_sq"] + corr_val
            else:
                corr_flat = None
            if clean_flat is not None and corr_flat is not None:
                dot_val = torch.dot(clean_flat, corr_flat)
                dot = dot + dot_val
                group_stats[group]["dot"] = group_stats[group]["dot"] + dot_val

        dot_f = float(dot.detach().cpu().item())
        clean_norm = float(torch.sqrt(clean_sq).detach().cpu().item())
        corr_norm = float(torch.sqrt(corr_sq).detach().cpu().item())
        group_metrics: dict[str, dict[str, float]] = {}
        for group, stats in sorted(group_stats.items()):
            group_dot = float(stats["dot"].detach().cpu().item())
            group_clean_norm = float(torch.sqrt(stats["clean_sq"]).detach().cpu().item())
            group_corr_norm = float(torch.sqrt(stats["corr_sq"]).detach().cpu().item())
            group_metrics[group] = {
                "cosine": _cosine_from_stats(group_dot, group_clean_norm, group_corr_norm),
                "clean_norm": group_clean_norm,
                "corr_norm": group_corr_norm,
                "dot": group_dot,
            }
        return {
            "loss_kind": str(loss_kind),
            "clean_loss": float(clean_loss.detach().float().item()),
            "corr_loss": float(corr_loss.detach().float().item()),
            "cosine": _cosine_from_stats(dot_f, clean_norm, corr_norm),
            "clean_norm": clean_norm,
            "corr_norm": corr_norm,
            "dot": dot_f,
            "groups": group_metrics,
        }
    finally:
        _zero_model_grads(latent_model)
        if was_training:
            latent_model.train()


def compute_stage1_grad_cosine_probe(
    latent_model: SmolVLALatentPolicy,
    batch: dict[str, Any],
    future_teacher_latent: torch.Tensor,
    action_prefix: torch.Tensor,
    *,
    base_batch_size: int,
    total_batch_size: int,
    global_step: int,
) -> dict[str, Any]:
    base_batch_size = int(base_batch_size)
    total_batch_size = int(total_batch_size)
    if base_batch_size <= 0 or total_batch_size <= base_batch_size:
        return {}
    clean_batch = _slice_smolvla_batch(batch, 0, base_batch_size)
    corr_batch = _slice_smolvla_batch(batch, base_batch_size, total_batch_size)
    clean_future = future_teacher_latent[:base_batch_size]
    corr_future = future_teacher_latent[base_batch_size:total_batch_size]
    clean_prefix = action_prefix[:base_batch_size]
    corr_prefix = action_prefix[base_batch_size:total_batch_size]
    metrics: dict[str, Any] = {}
    for loss_kind in ("action", "total"):
        metrics[loss_kind] = _stage1_grad_pair_metrics(
            latent_model,
            clean_batch=clean_batch,
            corr_batch=corr_batch,
            clean_future_teacher_latent=clean_future,
            corr_future_teacher_latent=corr_future,
            clean_action_prefix=clean_prefix,
            corr_action_prefix=corr_prefix,
            global_step=int(global_step),
            loss_kind=loss_kind,
        )
    return metrics


def log_stage1_grad_cosine_tensorboard(writer: SummaryWriter, step: int, metrics: dict[str, Any]) -> None:
    for loss_kind, values in metrics.items():
        if not isinstance(values, dict):
            continue
        prefix = f"grad_cosine/{loss_kind}"
        cosine = values.get("cosine", float("nan"))
        clean_norm = values.get("clean_norm", float("nan"))
        corr_norm = values.get("corr_norm", float("nan"))
        dot = values.get("dot", float("nan"))
        clean_loss = values.get("clean_loss", float("nan"))
        corr_loss = values.get("corr_loss", float("nan"))
        if np.isfinite(cosine):
            writer.add_scalar(f"{prefix}_clean_corr", float(cosine), step)
        if np.isfinite(clean_norm):
            writer.add_scalar(f"grad_norm/{loss_kind}_clean", float(clean_norm), step)
        if np.isfinite(corr_norm):
            writer.add_scalar(f"grad_norm/{loss_kind}_corr", float(corr_norm), step)
        if np.isfinite(dot):
            writer.add_scalar(f"grad_dot/{loss_kind}_clean_corr", float(dot), step)
        if np.isfinite(clean_loss):
            writer.add_scalar(f"grad_probe_loss/{loss_kind}_clean", float(clean_loss), step)
        if np.isfinite(corr_loss):
            writer.add_scalar(f"grad_probe_loss/{loss_kind}_corr", float(corr_loss), step)
        groups = values.get("groups", {})
        if isinstance(groups, dict):
            for group, group_values in groups.items():
                if not isinstance(group_values, dict):
                    continue
                group_cos = group_values.get("cosine", float("nan"))
                if np.isfinite(group_cos):
                    writer.add_scalar(f"grad_cosine_group/{loss_kind}_{group}", float(group_cos), step)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _decode_robotwin_rgb_frame(frame: Any, *, width: int = 640, height: int = 480) -> np.ndarray:
    if isinstance(frame, np.ndarray) and frame.ndim >= 2 and frame.shape[-1] == 3:
        image = np.asarray(frame)
    else:
        image = cv2.imdecode(np.frombuffer(bytes(frame), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode RoboTwin RGB frame")
    if image.shape[0] != int(height) or image.shape[1] != int(width):
        image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def _load_seen_instruction(path: Path, instruction_type: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing correction instruction file: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    values = payload.get(str(instruction_type), None)
    if not values and str(instruction_type) != "seen":
        values = payload.get("seen", None)
    if not isinstance(values, list) or not values:
        raise ValueError(f"No usable {instruction_type!r} instruction in {path}")
    instruction = str(values[0]).strip()
    if not instruction:
        raise ValueError(f"Empty instruction in {path}")
    return instruction


def _pad_raw_action_window(actions: np.ndarray, start: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    action_dim = int(actions.shape[1])
    padded = np.zeros((int(horizon), action_dim), dtype=np.float32)
    is_pad = np.ones((int(horizon),), dtype=bool)
    if int(start) < int(actions.shape[0]):
        chunk = actions[int(start) : int(start) + int(horizon)].astype(np.float32)
        valid_len = int(chunk.shape[0])
        padded[:valid_len] = chunk
        is_pad[:valid_len] = False
    return padded, is_pad


class OfflineCorrectionExportDataset(Dataset):
    """One training sample per exported correction episode.

    The exported RoboTwin correction episode starts at the perturbed state.
    For stage1 we use frame 0 as the observation and the following joint
    states, vector[1:1+chunk_size], as the action chunk.
    """

    def __init__(
        self,
        *,
        task_names: list[str],
        norm_stats: dict[str, np.ndarray],
        data_root: str,
        task_config: str,
        act_chunk_size: int,
        prefix_steps: int,
        future_offset: int,
        instruction_type: str,
        max_samples_per_task: int = 0,
    ) -> None:
        self.task_names = list(task_names)
        self.stats = norm_stats
        self.data_root = Path(data_root).resolve()
        self.task_config = str(task_config)
        self.act_chunk_size = int(act_chunk_size)
        self.prefix_steps = int(prefix_steps)
        self.future_offset = int(future_offset)
        self.instruction_type = str(instruction_type)
        self.records: list[dict[str, Any]] = []
        max_samples_per_task = int(max(0, int(max_samples_per_task)))

        for task_name_full in self.task_names:
            task_only, _ = parse_task_parts(str(task_name_full))
            task_records = self._collect_task_records(task_only=task_only, task_name_full=str(task_name_full))
            if max_samples_per_task > 0:
                task_records = task_records[:max_samples_per_task]
            self.records.extend(task_records)

        if not self.records:
            raise ValueError(
                f"No offline correction records found under {self.data_root} for task_config={self.task_config!r}"
            )
        self.task_counts = Counter(str(record["task_only"]) for record in self.records)

    def build_balanced_sampler(self, seed: int = 0) -> WeightedRandomSampler:
        if not self.records:
            raise ValueError("Cannot build sampler for an empty offline correction dataset")
        weights = [
            1.0 / float(max(1, int(self.task_counts[str(record["task_only"])])))
            for record in self.records
        ]
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(len(self.records)),
            replacement=True,
            generator=generator,
        )

    def _collect_task_records(self, *, task_only: str, task_name_full: str) -> list[dict[str, Any]]:
        task_dir = self.data_root / str(task_only)
        shard_root = task_dir / f"{self.task_config}_shards"
        merged_root = task_dir / self.task_config
        roots: list[Path]
        if shard_root.is_dir():
            roots = sorted(path for path in shard_root.iterdir() if path.is_dir() and path.name.startswith("rank"))
        elif merged_root.is_dir():
            roots = [merged_root]
        else:
            raise FileNotFoundError(f"Missing offline correction dataset: {shard_root} or {merged_root}")

        out: list[dict[str, Any]] = []
        for root in roots:
            manifest_records = _read_jsonl_records(root / "correction_manifest.jsonl")
            if manifest_records:
                for item in manifest_records:
                    episode_id = int(item.get("merged_episode_id", item.get("episode_id", len(out))))
                    data_path = Path(item.get("merged_data_path", item.get("export_path", "")))
                    instruction_path = Path(item.get("merged_instruction_path", item.get("instruction_path", "")))
                    if not data_path.is_absolute():
                        data_path = root / "data" / f"episode{episode_id}.hdf5"
                    if not instruction_path.is_absolute():
                        instruction_path = root / "instructions" / f"episode{episode_id}.json"
                    out.append(
                        {
                            "task_name": task_name_full,
                            "task_only": task_only,
                            "rank_root": str(root),
                            "episode_id": int(episode_id),
                            "data_path": str(data_path),
                            "instruction_path": str(instruction_path),
                            "source_episode_id": int(item.get("source_episode_id", -1)),
                            "source_raw_data_dir": str(
                                item.get(
                                    "source_raw_data_dir",
                                    str(Path(item["source_hdf5"]).parent) if item.get("source_hdf5") else "",
                                )
                            ),
                            "source_start_ts": int(item.get("source_start_ts", item.get("start_ts", -1))),
                            "correction_prefix_len": int(item.get("correction_prefix_len", -1)),
                        }
                    )
                continue

            data_dir = root / "data"
            for data_path in sorted(data_dir.glob("episode*.hdf5")):
                stem = data_path.stem.replace("episode", "")
                try:
                    episode_id = int(stem)
                except ValueError:
                    episode_id = len(out)
                out.append(
                    {
                        "task_name": task_name_full,
                        "task_only": task_only,
                        "rank_root": str(root),
                        "episode_id": int(episode_id),
                        "data_path": str(data_path),
                        "instruction_path": str(root / "instructions" / f"episode{episode_id}.json"),
                        "source_episode_id": -1,
                        "source_raw_data_dir": "",
                        "source_start_ts": -1,
                        "correction_prefix_len": -1,
                    }
                )

        return sorted(out, key=lambda item: (str(item["rank_root"]), int(item["episode_id"])))

    @staticmethod
    def _load_joint_vector(root: h5py.File) -> np.ndarray:
        if "joint_action" in root and "vector" in root["joint_action"]:
            return root["joint_action/vector"][()].astype(np.float32)
        left_arm = root["joint_action/left_arm"][()].astype(np.float32)
        left_gripper = root["joint_action/left_gripper"][()].astype(np.float32).reshape(-1, 1)
        right_arm = root["joint_action/right_arm"][()].astype(np.float32)
        right_gripper = root["joint_action/right_gripper"][()].astype(np.float32).reshape(-1, 1)
        return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[int(index)])
        data_path = Path(record["data_path"])
        instruction_path = Path(record["instruction_path"])
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing correction hdf5: {data_path}")

        with h5py.File(data_path, "r") as root:
            vector = self._load_joint_vector(root)
            if vector.ndim != 2 or vector.shape[0] < 2 or vector.shape[1] != 14:
                raise ValueError(f"Expected joint_action/vector with shape (T,14), got {vector.shape} in {data_path}")
            episode_len = int(vector.shape[0])
            future_idx = int(np.clip(int(self.future_offset), 0, episode_len - 1))
            image0 = _decode_robotwin_rgb_frame(root["observation/head_camera/rgb"][0])
            image_future = _decode_robotwin_rgb_frame(root["observation/head_camera/rgb"][future_idx])

        qpos_raw = vector[0].astype(np.float32)
        action_chunk_raw, act_is_pad = _pad_raw_action_window(vector, start=1, horizon=self.act_chunk_size)
        action_prefix_raw = action_chunk_raw[: self.prefix_steps].copy()
        action_future_prefix_raw, is_pad_future = _pad_raw_action_window(
            vector,
            start=1 + self.future_offset,
            horizon=self.prefix_steps,
        )
        qpos_t = (qpos_raw - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        action_chunk = (action_chunk_raw - self.stats["action_mean"]) / self.stats["action_std"]
        action_prefix = (action_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]
        action_future_prefix = (action_future_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]

        image_t = torch.from_numpy(image0).permute(2, 0, 1).float().div(255.0).unsqueeze(0).contiguous()
        image_t_future = torch.from_numpy(image_future).permute(2, 0, 1).float().div(255.0).unsqueeze(0).contiguous()
        instruction = _load_seen_instruction(instruction_path, self.instruction_type)

        return {
            "image_t": image_t,
            "image_t_future": image_t_future,
            "qpos_t": torch.from_numpy(qpos_t.astype(np.float32)),
            "qpos_raw": torch.from_numpy(qpos_raw.astype(np.float32)),
            "act_action_chunk": torch.from_numpy(action_chunk.astype(np.float32)),
            "act_action_chunk_raw": torch.from_numpy(action_chunk_raw.astype(np.float32)),
            "act_is_pad": torch.from_numpy(act_is_pad).bool(),
            "action_prefix": torch.from_numpy(action_prefix.astype(np.float32)),
            "action_future_prefix": torch.from_numpy(action_future_prefix.astype(np.float32)),
            "action_prefix_raw": torch.from_numpy(action_prefix_raw.astype(np.float32)),
            "action_future_prefix_raw": torch.from_numpy(action_future_prefix_raw.astype(np.float32)),
            "is_pad_prefix": torch.from_numpy(act_is_pad[: self.prefix_steps]).bool(),
            "is_pad_future_prefix": torch.from_numpy(is_pad_future).bool(),
            "episode_id": torch.tensor(int(record["episode_id"]), dtype=torch.int64),
            "start_ts": torch.tensor(0, dtype=torch.int64),
            "ep_len": torch.tensor(int(episode_len), dtype=torch.int64),
            "task_name": str(record["task_name"]),
            "raw_data_dir": str(record.get("source_raw_data_dir") or data_path.parent),
            "instruction": instruction,
            "correction_data_path": str(data_path),
            "source_episode_id": torch.tensor(int(record.get("source_episode_id", -1)), dtype=torch.int64),
            "source_start_ts": torch.tensor(int(record.get("source_start_ts", -1)), dtype=torch.int64),
            "correction_prefix_len": torch.tensor(int(record.get("correction_prefix_len", -1)), dtype=torch.int64),
        }


class Stage1SampleTypeDataset(Dataset):
    """Attach stable metadata so clean and correction samples can be collated together."""

    def __init__(self, dataset: Dataset, *, sample_type: str, default_instruction: str = "") -> None:
        self.dataset = dataset
        self.sample_type = str(sample_type)
        self.default_instruction = str(default_instruction)
        if hasattr(dataset, "stats"):
            self.stats = getattr(dataset, "stats")

    def __len__(self) -> int:
        return int(len(self.dataset))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[int(index)])
        sample.setdefault("sample_type", self.sample_type)
        sample.setdefault("instruction", self.default_instruction)
        sample.setdefault("task_idx", torch.tensor(-1, dtype=torch.int64))
        sample.setdefault("correction_data_path", "")
        sample.setdefault("source_episode_id", torch.tensor(-1, dtype=torch.int64))
        sample.setdefault("source_start_ts", torch.tensor(-1, dtype=torch.int64))
        sample.setdefault("correction_prefix_len", torch.tensor(-1, dtype=torch.int64))
        for key in ("task_idx", "episode_id", "start_ts", "ep_len", "source_episode_id", "source_start_ts", "correction_prefix_len"):
            value = sample.get(key)
            if not isinstance(value, torch.Tensor):
                sample[key] = torch.tensor(int(value), dtype=torch.int64)
            elif value.ndim == 0:
                sample[key] = value.to(dtype=torch.int64)
        return sample


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
    corr_outlier_filtered: int = 0,
    corr_outlier_action_loss_mean: float = float("nan"),
    corr_outlier_action_loss_max: float = float("nan"),
    corr_prefix_extra_loss: float = float("nan"),
    corr_prefix_loss_weight: float = 0.0,
    train_loss_with_prefix_extra: float = float("nan"),
) -> None:
    writer.add_scalar("train_extra/base_batch_size", float(base_batch_size), step)
    writer.add_scalar("train_extra/corr_requested", float(corr_requested), step)
    writer.add_scalar("train_extra/corr_generated", float(corr_generated), step)
    writer.add_scalar("train_extra/corr_skipped", float(corr_skipped), step)
    writer.add_scalar("train_extra/corr_outlier_filtered", float(corr_outlier_filtered), step)
    writer.add_scalar("train_extra/corr_generated_fraction", float(corr_generated) / max(1.0, float(corr_requested)), step)
    if corr_requested > 0:
        writer.add_scalar(
            "train_extra/corr_outlier_filtered_fraction",
            float(corr_outlier_filtered) / max(1.0, float(corr_requested)),
            step,
        )
    if np.isfinite(clean_action_abs_mean):
        writer.add_scalar("train_extra/clean_action_abs_mean_norm", float(clean_action_abs_mean), step)
    if np.isfinite(corr_source_action_abs_mean):
        writer.add_scalar("train_extra/corr_source_action_abs_mean_norm", float(corr_source_action_abs_mean), step)
    if np.isfinite(corr_generated_action_abs_mean):
        writer.add_scalar("train_extra/corr_generated_action_abs_mean_norm", float(corr_generated_action_abs_mean), step)
    if np.isfinite(corr_outlier_action_loss_mean):
        writer.add_scalar("train_extra/corr_outlier_action_loss_mean", float(corr_outlier_action_loss_mean), step)
    if np.isfinite(corr_outlier_action_loss_max):
        writer.add_scalar("train_extra/corr_outlier_action_loss_max", float(corr_outlier_action_loss_max), step)
    if np.isfinite(clean_action_loss_eval):
        writer.add_scalar("train_extra/clean_action_loss_eval", float(clean_action_loss_eval), step)
    if np.isfinite(corr_action_loss_eval):
        writer.add_scalar("train_extra/corr_action_loss_eval", float(corr_action_loss_eval), step)
    if np.isfinite(corr_prefix_extra_loss):
        writer.add_scalar("train_extra/corr_prefix_extra_loss", float(corr_prefix_extra_loss), step)
        writer.add_scalar("train_extra/corr_prefix_loss_weight", float(corr_prefix_loss_weight), step)
    if np.isfinite(train_loss_with_prefix_extra):
        writer.add_scalar("train_extra/train_loss_with_prefix_extra", float(train_loss_with_prefix_extra), step)
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
    instruction_override: str | None = None,
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
        instruction_override=instruction_override,
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
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor], list[bool], list[dict[str, Any]]]:
    samples = []
    future_images = []
    action_prefix = []
    is_correction = []
    projection_debug_records: list[dict[str, Any]] = []
    for batch_index in range(raw_batch["image_t"].shape[0]):
        task_name = raw_batch["task_name"][batch_index]
        sample_type = str(_tensor_item(raw_batch.get("sample_type"), batch_index, "clean"))
        sample_is_correction = bool(sample_type == "offline_correction_export")
        instruction_override = None
        if sample_is_correction:
            instruction_override = str(_tensor_item(raw_batch.get("instruction"), batch_index, "")).strip()
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
            instruction_override=instruction_override,
        )
        samples.append(sample)
        future_images.append(raw_batch["image_t_future"][batch_index, 0].detach().cpu())
        action_prefix.append(sample["action"][:, : int(prefix_steps), :])
        is_correction.append(sample_is_correction)
        if sample_is_correction:
            projection_debug_records.append(
                {
                    "sample_type": "offline_correction_export",
                    "image_t": raw_batch["image_t"][batch_index, 0].detach().cpu(),
                    "action_norm": raw_batch["act_action_chunk"][batch_index].detach().cpu(),
                    "act_is_pad": raw_batch["act_is_pad"][batch_index].detach().cpu(),
                    "raw_data_dir": str(_tensor_item(raw_batch.get("raw_data_dir"), batch_index, "")),
                    "episode_id": scalar_to_int(raw_batch["episode_id"][batch_index]),
                    "start_ts": int(_tensor_item(raw_batch.get("start_ts"), batch_index, 0)),
                    "source_episode_id": int(_tensor_item(raw_batch.get("source_episode_id"), batch_index, -1)),
                    "source_start_ts": int(_tensor_item(raw_batch.get("source_start_ts"), batch_index, -1)),
                    "correction_data_path": str(
                        _tensor_item(raw_batch.get("correction_data_path"), batch_index, "")
                    ),
                }
            )
    return samples, future_images, action_prefix, is_correction, projection_debug_records


def summarize_offline_correction_batch(correction_raw_batch: dict[str, Any]) -> list[dict[str, Any]]:
    batch_size = int(correction_raw_batch["image_t"].shape[0])
    records = []
    for sample_index in range(batch_size):
        sample_type = str(_tensor_item(correction_raw_batch.get("sample_type"), sample_index, "offline_correction_export"))
        if sample_type != "offline_correction_export":
            continue
        records.append(
            {
                "sample_type": "offline_correction_export",
                "task_name": str(_tensor_item(correction_raw_batch.get("task_name"), sample_index, "")),
                "episode_id": int(_tensor_item(correction_raw_batch.get("episode_id"), sample_index, -1)),
                "start_ts": int(_tensor_item(correction_raw_batch.get("start_ts"), sample_index, 0)),
                "source_episode_id": int(
                    _tensor_item(correction_raw_batch.get("source_episode_id"), sample_index, -1)
                ),
                "source_start_ts": int(_tensor_item(correction_raw_batch.get("source_start_ts"), sample_index, -1)),
                "correction_prefix_len": int(
                    _tensor_item(correction_raw_batch.get("correction_prefix_len"), sample_index, -1)
                ),
                "correction_data_path": str(
                    _tensor_item(correction_raw_batch.get("correction_data_path"), sample_index, "")
                ),
            }
        )
    return records


def build_stage1_offline_correction_samples(
    correction_raw_batch: dict[str, Any],
    latent_model: SmolVLALatentPolicy,
    preprocess,
    instruction_type: str,
    prefix_steps: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]:
    samples = []
    future_images = []
    action_prefix = []
    projection_debug_records: list[dict[str, Any]] = []
    for sample_index in range(correction_raw_batch["image_t"].shape[0]):
        task_name_full = correction_raw_batch["task_name"][sample_index]
        instruction = str(_tensor_item(correction_raw_batch.get("instruction"), sample_index, "")).strip()
        episode_id = scalar_to_int(correction_raw_batch["episode_id"][sample_index])
        sample = build_smolvla_batch_from_raw_sample(
            latent_policy=latent_model,
            preprocess=preprocess,
            instruction_type=instruction_type,
            task_name_full=task_name_full,
            image_t=correction_raw_batch["image_t"][sample_index],
            qpos_raw=correction_raw_batch["qpos_raw"][sample_index],
            action_chunk_raw=correction_raw_batch["act_action_chunk_raw"][sample_index],
            episode_id=episode_id,
            instruction_override=instruction,
        )
        samples.append(sample)
        future_images.append(correction_raw_batch["image_t_future"][sample_index, 0].detach().cpu())
        action_prefix.append(sample["action"][:, : int(prefix_steps), :])
        projection_debug_records.append(
            {
                "sample_type": "offline_correction_export",
                "image_t": correction_raw_batch["image_t"][sample_index, 0].detach().cpu(),
                "action_norm": correction_raw_batch["act_action_chunk"][sample_index].detach().cpu(),
                "act_is_pad": correction_raw_batch["act_is_pad"][sample_index].detach().cpu(),
                "raw_data_dir": str(_tensor_item(correction_raw_batch.get("raw_data_dir"), sample_index, "")),
                "episode_id": int(episode_id),
                "start_ts": int(_tensor_item(correction_raw_batch.get("start_ts"), sample_index, 0)),
                "correction_data_path": str(
                    _tensor_item(correction_raw_batch.get("correction_data_path"), sample_index, "")
                ),
            }
        )
    return samples, future_images, action_prefix, projection_debug_records


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


def load_raw_items_for_stage1_rollout(
    raw_batch: dict[str, Any],
    raw_cache: dict[tuple[str, int], dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    raw_items: list[dict[str, Any]] = []
    for batch_index in range(int(count)):
        raw_dir = str(_tensor_item(raw_batch.get("raw_data_dir"), batch_index, ""))
        episode_id = scalar_to_int(raw_batch["episode_id"][batch_index])
        cache_key = (raw_dir, int(episode_id))
        if cache_key not in raw_cache:
            raw_cache[cache_key] = load_raw_episode(raw_dir, int(episode_id))
        raw_items.append(raw_cache[cache_key])
    return raw_items


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
    failure_mode = str(args.failure_mode).strip().lower()
    stage1_corr_source = str(getattr(args, "stage1_corr_source", "online")).strip().lower()
    stage1_latent_target = str(getattr(args, "stage1_latent_target", "future_image")).strip().lower()
    use_stage1_wm_rollout_target = bool(stage1_latent_target == "wm_rollout")
    use_corr_loader = bool(failure_mode == "train")
    use_online_corr_loader = bool(use_corr_loader and stage1_corr_source == "online")
    use_offline_corr_loader = bool(use_corr_loader and stage1_corr_source == "offline_export")
    use_offline_mixed_dataset = bool(use_corr_loader and stage1_corr_source == "offline_export_mixed_dataset")
    if use_stage1_wm_rollout_target:
        if use_corr_loader:
            raise ValueError("stage1_latent_target=wm_rollout is a normal-only ablation; set --failure_mode off.")
        if not str(getattr(args, "urdf_path", "")).strip():
            raise ValueError("stage1_latent_target=wm_rollout requires --urdf_path for EVAC rollout FK.")
    dataloader_dataset: Dataset = dataset
    if use_online_corr_loader:
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
    elif use_offline_corr_loader:
        failure_table_paths = {}
        failure_task_names = list(args.failure_task_names or [])
        failure_task_specs = (
            resolve_multitask_specs(failure_task_names, SIM_TASK_CONFIGS, raw_data_root_overrides=None)
            if failure_task_names else task_specs
        )
        corr_batch_size = int(max(1, round(float(args.batch_size) * float(args.failure_corr_batch_ratio))))
        norm_stats = dataset.stats if hasattr(dataset, "stats") else None
        if norm_stats is None:
            raise ValueError("Offline correction training requires normal dataset norm stats")
        correction_dataset = OfflineCorrectionExportDataset(
            task_names=[spec.task_name for spec in failure_task_specs],
            norm_stats=norm_stats,
            data_root=str(args.offline_corr_data_root),
            task_config=str(args.offline_corr_task_config),
            act_chunk_size=int(args.act_chunk_size),
            prefix_steps=int(args.prefix_steps),
            future_offset=int(args.future_offset),
            instruction_type=str(args.instruction_type),
            max_samples_per_task=int(args.offline_corr_max_samples_per_task),
        )
        correction_sampler = (
            correction_dataset.build_balanced_sampler(seed=int(args.seed) + 1009 * int(os.environ.get("RANK", "0")))
            if bool(args.offline_corr_balance_tasks) else None
        )
        correction_loader = DataLoader(
            correction_dataset,
            batch_size=corr_batch_size,
            shuffle=correction_sampler is None,
            sampler=correction_sampler,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    elif use_offline_mixed_dataset:
        failure_table_paths = {}
        failure_task_names = list(args.failure_task_names or [])
        failure_task_specs = (
            resolve_multitask_specs(failure_task_names, SIM_TASK_CONFIGS, raw_data_root_overrides=None)
            if failure_task_names else task_specs
        )
        norm_stats = dataset.stats if hasattr(dataset, "stats") else None
        if norm_stats is None:
            raise ValueError("Offline mixed correction training requires normal dataset norm stats")
        correction_dataset = OfflineCorrectionExportDataset(
            task_names=[spec.task_name for spec in failure_task_specs],
            norm_stats=norm_stats,
            data_root=str(args.offline_corr_data_root),
            task_config=str(args.offline_corr_task_config),
            act_chunk_size=int(args.act_chunk_size),
            prefix_steps=int(args.prefix_steps),
            future_offset=int(args.future_offset),
            instruction_type=str(args.instruction_type),
            max_samples_per_task=int(args.offline_corr_max_samples_per_task),
        )
        dataloader_dataset = ConcatDataset(
            [
                Stage1SampleTypeDataset(dataset, sample_type="clean"),
                Stage1SampleTypeDataset(correction_dataset, sample_type="offline_correction_export"),
            ]
        )
        correction_loader = None
    else:
        failure_table_paths = {}
        correction_dataset = None
        correction_loader = None
        norm_stats = dataset.stats if hasattr(dataset, "stats") else None

    dataloader = DataLoader(
        dataloader_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    init_raw_sample = dataset[0]
    base_policy, preprocess, postprocess = build_base_policy(
        args,
        dataset_sample=init_raw_sample,
        dataset_stats=(dataset.stats if hasattr(dataset, "stats") else None),
    )
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_stage1_warmup_config(args),
        loss_weights=build_stage1_loss_weights(args),
    )
    model.to(torch.device(args.device))
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
        load_rollout_model=(use_online_corr_loader or use_stage1_wm_rollout_target),
    )

    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=init_raw_sample,
        teacher=teacher,
    )
    correction_builder = None
    if use_online_corr_loader:
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
    if use_online_corr_loader or use_offline_corr_loader:
        assert correction_loader is not None
        model, optimizer, dataloader, correction_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, correction_loader, lr_scheduler
        )
    else:
        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    latent_model = accelerator.unwrap_model(model)
    if is_main:
        save_train_args(output_dir, args)

    correction_iter = iter(correction_loader) if (use_online_corr_loader or use_offline_corr_loader) else None
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}
    skip_reason_counter: Counter[str] = Counter()
    debug_wm = bool(getattr(args, "debug_wm_correction", False))
    debug_wm_all_ranks = bool(getattr(args, "debug_wm_all_ranks", False))
    debug_loss_batch_projection = bool(getattr(args, "debug_loss_batch_projection", False))
    debug_loss_batch_projection_freq = int(max(1, int(getattr(args, "debug_loss_batch_projection_freq", 1))))
    debug_grad_cosine = bool(getattr(args, "debug_grad_cosine", False))
    debug_grad_cosine_freq = int(max(1, int(getattr(args, "debug_grad_cosine_freq", 100))))
    stage1_corr_outlier_filter = bool(getattr(args, "stage1_corr_outlier_filter", False))
    stage1_corr_outlier_max_action_loss = float(getattr(args, "stage1_corr_outlier_max_action_loss", 0.3))
    stage1_corr_prefix_loss_weight = float(getattr(args, "stage1_corr_prefix_loss_weight", 0.0))
    stage1_corr_prefix_loss_steps = int(getattr(args, "stage1_corr_prefix_loss_steps", 0))
    if stage1_corr_prefix_loss_weight > 0.0 and stage1_corr_prefix_loss_steps <= 0:
        stage1_corr_prefix_loss_steps = int(args.prefix_steps)
    stage1_rollout_fk = None
    if use_stage1_wm_rollout_target:
        try:
            from policy.ACT.util.fk_sapien import SapienFK

            stage1_rollout_fk = SapienFK(str(args.urdf_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SapienFK for stage1 wm_rollout latent target: {exc}") from exc
        if is_main:
            print(
                "[stage1] latent target: wm_rollout "
                f"(normal-only, ddim_steps={int(args.stage1_rollout_ddim_steps)})",
                flush=True,
            )
    elif is_main:
        print("[stage1] latent target: future_image", flush=True)
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
    loss_projection_fk = None
    if debug_loss_batch_projection:
        try:
            from policy.ACT.util.fk_sapien import SapienFK

            loss_projection_fk = SapienFK(str(args.urdf_path))
        except Exception as exc:
            if is_main:
                print(f"[stage1] disabling loss_batch_projection; failed to initialize FK: {exc}", flush=True)
            debug_loss_batch_projection = False
    save_correction_data = bool(getattr(args, "save_correction_data", False)) and use_online_corr_loader
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
                (
                    samples,
                    future_images,
                    action_prefix,
                    raw_is_correction,
                    mixed_corr_projection_debug_records,
                ) = build_stage1_samples_from_raw_batch(
                    raw_batch=raw_batch,
                    latent_model=latent_model,
                    preprocess=preprocess,
                    instruction_type=args.instruction_type,
                    prefix_steps=int(args.prefix_steps),
                )
                base_batch_size = int(raw_batch["image_t"].shape[0])
                clean_action_abs_mean = _mean_action_abs(raw_batch)
                corr_requested = 0
                corr_generated_count = int(sum(1 for value in raw_is_correction if value))
                clean_generated_count = int(len(samples) - corr_generated_count)
                corr_source_action_abs_mean = float("nan")
                corr_generated_action_abs_mean = float("nan")
                corr_trace_records: list[dict[str, Any]] = []
                corr_skip_reasons: Counter[str] = Counter()
                corr_outlier_records: list[dict[str, Any]] = []
                corr_outlier_loss_values: list[float] = []
                normal_only_after_corr_filter = False
                if use_offline_mixed_dataset:
                    corr_requested = int(corr_generated_count)
                    corr_indices_for_raw = [idx for idx, value in enumerate(raw_is_correction) if value]
                    clean_indices_for_raw = [idx for idx, value in enumerate(raw_is_correction) if not value]
                    clean_action_abs_mean = _mean_action_abs_for_indices(raw_batch, clean_indices_for_raw)
                    corr_source_action_abs_mean = _mean_action_abs_for_indices(raw_batch, corr_indices_for_raw)
                    corr_projection_debug_records = mixed_corr_projection_debug_records
                    corr_trace_records = summarize_offline_correction_batch(raw_batch) if corr_requested > 0 else []
                    if corr_projection_debug_records:
                        corr_generated_action_abs_mean = float(
                            np.mean(
                                [
                                    float(record["action_norm"].detach().float().abs().mean().item())
                                    for record in corr_projection_debug_records
                                ]
                            )
                        )
                elif use_corr_loader:
                    assert correction_iter is not None
                    assert norm_stats is not None
                    try:
                        correction_raw_batch = next(correction_iter)
                    except StopIteration:
                        correction_iter = iter(correction_loader)
                        correction_raw_batch = next(correction_iter)
                    corr_requested = int(correction_raw_batch["image_t"].shape[0])
                    corr_source_action_abs_mean = _mean_action_abs(correction_raw_batch)
                    step_debug_dir = None
                    if debug_wm_should_save and use_online_corr_loader:
                        step_debug_dir = str(debug_wm_dir / f"step_{global_step:06d}")
                        Path(step_debug_dir).mkdir(parents=True, exist_ok=True)
                    if use_online_corr_loader:
                        assert correction_builder is not None
                        corr_trace_records = summarize_correction_raw_batch(correction_raw_batch)
                        (
                            corr_samples,
                            corr_future_images,
                            corr_action_prefix,
                            corr_skip_reasons,
                            corr_projection_debug_records,
                        ) = build_stage1_correction_samples(
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
                    else:
                        corr_trace_records = summarize_offline_correction_batch(correction_raw_batch)
                        corr_skip_reasons = Counter()
                        (
                            corr_samples,
                            corr_future_images,
                            corr_action_prefix,
                            corr_projection_debug_records,
                        ) = build_stage1_offline_correction_samples(
                            correction_raw_batch=correction_raw_batch,
                            latent_model=latent_model,
                            preprocess=preprocess,
                            instruction_type=args.instruction_type,
                            prefix_steps=int(args.prefix_steps),
                        )
                    corr_pre_outlier_count = int(len(corr_samples))
                    if stage1_corr_outlier_filter and corr_samples:
                        (
                            corr_samples,
                            corr_future_images,
                            corr_action_prefix,
                            corr_projection_debug_records,
                            corr_outlier_records,
                            corr_outlier_loss_values,
                        ) = filter_stage1_correction_outliers(
                            latent_model,
                            corr_samples,
                            corr_future_images,
                            corr_action_prefix,
                            corr_projection_debug_records,
                            max_action_loss=stage1_corr_outlier_max_action_loss,
                        )
                        if corr_outlier_records:
                            corr_skip_reasons["outlier_action_loss"] += int(len(corr_outlier_records))
                        normal_only_after_corr_filter = bool(
                            corr_pre_outlier_count > 0
                            and not corr_samples
                            and int(len(corr_outlier_records)) == int(corr_pre_outlier_count)
                        )
                    skip_reason_counter.update(corr_skip_reasons)
                    corr_skip_reasons = Counter(corr_skip_reasons)
                    samples.extend(corr_samples)
                    future_images.extend(corr_future_images)
                    action_prefix.extend(corr_action_prefix)
                    corr_generated_count = int(len(corr_samples))
                    clean_generated_count = int(base_batch_size)
                    if corr_projection_debug_records:
                        corr_generated_action_abs_mean = float(
                            np.mean(
                                [
                                    float(record["action_norm"].detach().float().abs().mean().item())
                                    for record in corr_projection_debug_records
                                ]
                            )
                        )
                    if not corr_samples and not normal_only_after_corr_filter:
                        trace_payload = {
                            "step": int(global_step),
                            "rank": int(accelerator.process_index),
                            "skipped_train_step": True,
                            "base_batch_size": int(base_batch_size),
                            "corr_requested": int(corr_requested),
                            "corr_generated": 0,
                            "corr_skipped": int(sum(corr_skip_reasons.values())),
                            "skip_reasons": dict(corr_skip_reasons),
                            "corr_outlier_filter": bool(stage1_corr_outlier_filter),
                            "corr_outlier_max_action_loss": float(stage1_corr_outlier_max_action_loss),
                            "corr_outlier_filtered": int(len(corr_outlier_records)),
                            "corr_outlier_action_loss_values": [float(value) for value in corr_outlier_loss_values],
                            "corr_outlier_records": corr_outlier_records,
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
                action_prefix_tensor = torch.cat(action_prefix, dim=0).to(device=accelerator.device)
                if use_stage1_wm_rollout_target:
                    assert stage1_rollout_fk is not None
                    if int(len(samples)) != int(base_batch_size):
                        raise RuntimeError(
                            "stage1_latent_target=wm_rollout expects normal-only batches; "
                            f"got len(samples)={len(samples)} base_batch_size={base_batch_size}"
                        )
                    rollout_raw_items = load_raw_items_for_stage1_rollout(
                        raw_batch=raw_batch,
                        raw_cache=raw_cache,
                        count=int(base_batch_size),
                    )
                    future_teacher_latent = teacher.rollout_latent_from_actions_batch(
                        curr_image=raw_batch["image_t"][:base_batch_size, 0].detach().cpu(),
                        curr_qpos_raw=raw_batch["qpos_raw"][:base_batch_size].detach().cpu(),
                        action_prefix_raw=raw_batch["act_action_chunk_raw"][
                            :base_batch_size, : int(args.prefix_steps)
                        ].detach().cpu(),
                        raw_data=rollout_raw_items,
                        fk=stage1_rollout_fk,
                        ddim_steps=int(args.stage1_rollout_ddim_steps),
                    ).to(device=accelerator.device, dtype=torch.float32)
                else:
                    future_image_tensor = torch.stack(future_images, dim=0).to(
                        device=accelerator.device, dtype=torch.float32
                    )
                    future_teacher_latent = teacher.encode_image(future_image_tensor)

                if (
                    debug_loss_batch_projection
                    and loss_projection_fk is not None
                    and norm_stats is not None
                    and int(global_step) % int(debug_loss_batch_projection_freq) == 0
                ):
                    loss_dbg_dir = (
                        output_dir
                        / "loss_batch_projection"
                        / f"rank{int(accelerator.process_index):02d}"
                        / f"step_{global_step:06d}"
                    )
                    loss_dbg_dir.mkdir(parents=True, exist_ok=True)
                    fk = loss_projection_fk

                    def _get_raw(raw_data_dir_i: str, episode_id_i: int) -> dict[str, Any]:
                        cache_key = (str(raw_data_dir_i), int(episode_id_i))
                        if cache_key not in raw_cache:
                            raw_cache[cache_key] = load_raw_episode(str(raw_data_dir_i), int(episode_id_i))
                        return raw_cache[cache_key]

                    projection_items: list[dict[str, Any]] = []
                    for i in range(int(raw_batch["image_t"].shape[0])):
                        ep_i = scalar_to_int(raw_batch["episode_id"][i])
                        st_i = scalar_to_int(raw_batch["start_ts"][i])
                        raw_dir_i = str(raw_batch["raw_data_dir"][i])
                        sample_type_i = str(_tensor_item(raw_batch.get("sample_type"), i, "clean"))
                        correction_path_i = str(_tensor_item(raw_batch.get("correction_data_path"), i, ""))
                        source_ep_i = int(_tensor_item(raw_batch.get("source_episode_id"), i, -1))
                        source_start_i = int(_tensor_item(raw_batch.get("source_start_ts"), i, -1))
                        raw_episode_id_i = ep_i
                        raw_start_ts_i = st_i
                        if sample_type_i == "offline_correction_export":
                            raw_episode_id_i = source_ep_i if source_ep_i >= 0 else ep_i
                            raw_start_ts_i = source_start_i if source_start_i >= 0 else st_i
                        projection_items.append(
                            {
                                "image_cam": raw_batch["image_t"][i, 0],
                                "action_norm": raw_batch["act_action_chunk"][i],
                                "is_pad": raw_batch["act_is_pad"][i],
                                "raw_data": _get_raw(raw_dir_i, raw_episode_id_i),
                                "meta": {
                                    "sample_type": sample_type_i,
                                    "task_name": str(_tensor_item(raw_batch.get("task_name"), i, "")),
                                    "episode_id": ep_i,
                                    "start_ts": st_i,
                                    "source_episode_id": source_ep_i,
                                    "source_start_ts": source_start_i,
                                    "raw_projection_episode_id": raw_episode_id_i,
                                    "raw_projection_start_ts": raw_start_ts_i,
                                    "raw_data_dir": raw_dir_i,
                                    "correction_data_path": correction_path_i,
                                },
                            }
                        )

                    extra_projection_records = [] if use_offline_mixed_dataset else corr_projection_debug_records
                    for j, record in enumerate(extra_projection_records):
                        ep_j = int(record["episode_id"])
                        st_j = int(record["start_ts"])
                        projection_items.append(
                            {
                                "image_cam": record["image_t"],
                                "action_norm": record["action_norm"],
                                "is_pad": record["act_is_pad"],
                                "raw_data": _get_raw(str(record["raw_data_dir"]), ep_j),
                                "meta": {
                                    **{
                                        key: value
                                        for key, value in record.items()
                                        if key
                                        not in {
                                            "image_t",
                                            "action_norm",
                                            "act_is_pad",
                                        }
                                    },
                                    "episode_id": ep_j,
                                    "start_ts": st_j,
                                },
                            }
                        )
                    _save_loss_batch_projection_grid(
                        str(loss_dbg_dir),
                        projection_items,
                        fk,
                        norm_stats,
                    )

                output = model(
                    train_stage="stage1",
                    batch=batch,
                    future_teacher_latent=future_teacher_latent,
                    action_prefix=action_prefix_tensor,
                    global_step=global_step,
                )
                corr_prefix_extra_loss = None
                if stage1_corr_prefix_loss_weight > 0.0 and corr_generated_count > 0:
                    if use_offline_mixed_dataset:
                        corr_indices_for_loss = [idx for idx, value in enumerate(raw_is_correction) if value]
                        corr_prefix_batch = _index_smolvla_batch(batch, corr_indices_for_loss)
                    else:
                        corr_prefix_batch = _slice_smolvla_batch(batch, base_batch_size, len(samples))
                    corr_prefix_extra_loss = model(
                        train_stage="action_prefix",
                        batch=corr_prefix_batch,
                        actions=corr_prefix_batch[ACTION],
                        prefix_steps=stage1_corr_prefix_loss_steps,
                    )
                train_loss = output.loss
                if corr_prefix_extra_loss is not None:
                    train_loss = train_loss + (stage1_corr_prefix_loss_weight * corr_prefix_extra_loss)
                if use_offline_mixed_dataset:
                    clean_indices = [idx for idx, value in enumerate(raw_is_correction) if not value]
                    corr_indices = [idx for idx, value in enumerate(raw_is_correction) if value]
                    clean_action_loss_eval, corr_action_loss_eval = compute_action_loss_by_mask(
                        latent_model,
                        batch,
                        clean_indices=clean_indices,
                        corr_indices=corr_indices,
                    )
                else:
                    clean_action_loss_eval, corr_action_loss_eval = compute_action_loss_split(
                        latent_model,
                        batch,
                        base_batch_size=base_batch_size,
                        total_batch_size=len(samples),
                    )
                grad_cosine_metrics: dict[str, Any] = {}
                if (
                    debug_grad_cosine
                    and use_corr_loader
                    and (len(samples) > base_batch_size)
                    and int(global_step) % int(debug_grad_cosine_freq) == 0
                ):
                    grad_cosine_metrics = compute_stage1_grad_cosine_probe(
                        latent_model,
                        batch,
                        future_teacher_latent,
                        action_prefix_tensor,
                        base_batch_size=base_batch_size,
                        total_batch_size=len(samples),
                        global_step=int(global_step),
                    )
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(train_loss)
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
                        base_batch_size=clean_generated_count,
                        corr_requested=corr_requested,
                        corr_generated=corr_generated_count,
                        corr_skipped=int(sum(corr_skip_reasons.values())),
                        clean_action_abs_mean=clean_action_abs_mean,
                        corr_source_action_abs_mean=corr_source_action_abs_mean,
                        corr_generated_action_abs_mean=corr_generated_action_abs_mean,
                        clean_action_loss_eval=clean_action_loss_eval,
                        corr_action_loss_eval=corr_action_loss_eval,
                        corr_outlier_filtered=len(corr_outlier_records),
                        corr_outlier_action_loss_mean=(
                            float(np.mean(corr_outlier_loss_values)) if corr_outlier_loss_values else float("nan")
                        ),
                        corr_outlier_action_loss_max=(
                            float(np.max(corr_outlier_loss_values)) if corr_outlier_loss_values else float("nan")
                        ),
                        corr_prefix_extra_loss=(
                            float(corr_prefix_extra_loss.detach().float().item())
                            if corr_prefix_extra_loss is not None else float("nan")
                        ),
                        corr_prefix_loss_weight=float(stage1_corr_prefix_loss_weight),
                        train_loss_with_prefix_extra=float(train_loss.detach().float().item()),
                    )
                    if grad_cosine_metrics:
                        log_stage1_grad_cosine_tensorboard(
                            writer=tb_writer,
                            step=global_step,
                            metrics=grad_cosine_metrics,
                        )
                if use_corr_loader:
                    trace_payload = {
                        "step": int(global_step),
                        "rank": int(accelerator.process_index),
                        "stage1_corr_source": str(stage1_corr_source),
                        "base_batch_size": int(clean_generated_count),
                        "corr_requested": int(corr_requested),
                        "corr_generated": int(corr_generated_count),
                        "total_batch_size": int(len(samples)),
                        "corr_skipped": int(sum(corr_skip_reasons.values())),
                        "skip_reasons": dict(corr_skip_reasons),
                        "corr_outlier_filter": bool(stage1_corr_outlier_filter),
                        "corr_outlier_max_action_loss": float(stage1_corr_outlier_max_action_loss),
                        "corr_outlier_filtered": int(len(corr_outlier_records)),
                        "corr_outlier_action_loss_values": [float(value) for value in corr_outlier_loss_values],
                        "corr_outlier_records": corr_outlier_records,
                        "normal_only_after_corr_filter": bool(normal_only_after_corr_filter),
                        "clean_action_abs_mean_norm": float(clean_action_abs_mean),
                        "corr_source_action_abs_mean_norm": float(corr_source_action_abs_mean),
                        "corr_generated_action_abs_mean_norm": float(corr_generated_action_abs_mean),
                        "clean_action_loss_eval": float(clean_action_loss_eval),
                        "corr_action_loss_eval": float(corr_action_loss_eval),
                        "loss": float(output.loss.item()),
                        "train_loss": float(train_loss.detach().float().item()),
                        "corr_prefix_extra_loss": (
                            float(corr_prefix_extra_loss.detach().float().item())
                            if corr_prefix_extra_loss is not None else float("nan")
                        ),
                        "corr_prefix_loss_weight": float(stage1_corr_prefix_loss_weight),
                        "corr_prefix_loss_steps": int(stage1_corr_prefix_loss_steps),
                        "loss_action": float(output.loss_action.item()),
                        "loss_action_conditioned": float(output.loss_action_conditioned.item()),
                        "loss_dynamics": float(output.loss_dynamics.item()),
                        "loss_condition_token": float(output.loss_condition_token.item()),
                        "beta_condition": float(output.beta_condition),
                        "beta_dynamics": float(output.beta_dynamics),
                        "beta_token": float(output.beta_token),
                        "lr": float(current_lr),
                        "requested_units": corr_trace_records,
                    }
                    if grad_cosine_metrics:
                        trace_payload["grad_cosine"] = grad_cosine_metrics
                    with open(correction_trace_path, "a", encoding="utf-8") as trace_f:
                        trace_f.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")

                progress.set_postfix(
                    loss=f"{output.loss.item():.4f}",
                    action=f"{output.loss_action.item():.4f}",
                    cond=f"{output.loss_action_conditioned.item():.4f}",
                    dyn=f"{output.loss_dynamics.item():.4f}",
                    ctoken=f"{output.loss_condition_token.item():.4f}",
                    beta_cond=f"{output.beta_condition:.4f}",
                    beta=f"{output.beta_dynamics:.4f}",
                    beta_tok=f"{output.beta_token:.4f}",
                    lr=f"{current_lr:.2e}",
                    target=stage1_latent_target,
                    corr=corr_generated_count,
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
                    correction_teacher_latent = teacher.encode_image(teacher_image)
                    correction_teacher_latents.append(correction_teacher_latent)
                    corr_action_prefix_raw = (
                        correction["error_action_prefix_raw"]
                        if correction["error_action_prefix_raw"] is not None
                        else correction_raw_batch["act_action_chunk"][sample_index][: int(args.prefix_steps)].detach().cpu().numpy()
                    )
                    rollout_teacher_latent = teacher.rollout_latent_from_actions(
                        curr_image=corr_image[0].detach().cpu(),
                        curr_qpos_raw=corr_qpos_raw.detach().cpu(),
                        action_prefix_raw=torch.as_tensor(corr_action_prefix_raw, dtype=torch.float32),
                        raw_data=raw_cache[raw_cache_key],
                        fk=correction_builder.fk,
                        ddim_steps=int(args.ddim_steps),
                    )
                    rollout_teacher_latents.append(rollout_teacher_latent)

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
