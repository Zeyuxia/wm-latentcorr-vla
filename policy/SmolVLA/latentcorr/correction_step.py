from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

from policy.SmolVLA.latentcorr.failure_utils import (
    active_arm_pattern_id_to_key,
    error_mode_id_to_key,
    phase_id_to_key,
)
from policy.SmolVLA.latentcorr.correction_episode_io import (
    export_correction_sample_as_episode,
    init_export_episode_id,
    load_raw_data,
)
from policy.SmolVLA.latentcorr.correction_evac_rollout import (
    evac_inference,
    find_nearest_traj_point,
    quat_geodesic_deg_wxyz,
)
from policy.SmolVLA.latentcorr.correction_cosmos_rollout import cosmos_inference
from policy.SmolVLA.latentcorr.correction_utils import resample_trajectory
from policy.SmolVLA.latentcorr.correction_projection_debug import (
    _tensor_chw_to_bgr_u8,
    _write_bgr_image,
    compute_phase_projection,
    draw_phase_bin_starts,
    draw_phase_legend,
    draw_phase_polyline,
    save_gt_projection_on_original,
    save_loss_batch_projection,
    save_perturb_compare_image,
    save_recover_eval_compare_image,
)
from policy.SmolVLA.latentcorr.correction_perturbation import (
    _infer_active_arms_from_gt_window,
    perturb_action_chunk_online,
)


def _debug_relpath(path):
    if path is None:
        return None
    try:
        return f"./{os.path.relpath(path, start=os.getcwd())}"
    except Exception:
        return path


def _world_model_backend(cfg: dict) -> str:
    backend = str(cfg.get("world_model_backend", "evac")).strip().lower()
    if backend not in {"evac", "cosmos"}:
        raise ValueError(f"Unsupported world_model_backend={backend!r}; expected 'evac' or 'cosmos'.")
    return backend


def _world_model_runtime_meta_filename(backend: str) -> str:
    return "cosmos_runtime_meta.json" if str(backend).strip().lower() == "cosmos" else "evac_runtime_meta.json"


def _world_model_inference(
    modules: dict,
    cfg: dict,
    curr_image: torch.Tensor,
    fk_poses: list,
    grippers: list,
    raw_data: dict,
    device,
    save_dir: str | None = None,
    backend_override: str | None = None,
) -> torch.Tensor:
    backend = _world_model_backend(cfg) if backend_override is None else str(backend_override).strip().lower()
    if backend not in {"evac", "cosmos"}:
        raise ValueError(f"Unsupported world-model backend override={backend!r}.")
    if backend == "cosmos":
        client = modules.get("cosmos_client")
        if client is None:
            raise RuntimeError("world_model_backend='cosmos' but modules['cosmos_client'] is missing.")
        return cosmos_inference(
            client,
            curr_image,
            fk_poses,
            grippers,
            raw_data,
            device,
            save_dir=save_dir,
        )
    if modules.get("evac_model") is None or modules.get("evac_config") is None:
        raise RuntimeError("world_model_backend='evac' but EVAC model/config is missing.")
    return evac_inference(
        modules["evac_model"],
        modules["evac_config"],
        curr_image,
        fk_poses,
        grippers,
        raw_data,
        device,
        save_dir=save_dir,
        infer_kwargs=cfg.get("evac_infer_kwargs"),
    )


def _world_model_compare_backends(cfg: dict) -> list[str]:
    if not bool(cfg.get("world_model_compare_mode", False)):
        return []
    raw = cfg.get("world_model_compare_backends", ("evac", "cosmos"))
    if raw is None or raw == "":
        raw_items = ("evac", "cosmos")
    elif isinstance(raw, str):
        raw_items = raw.replace(";", ",").split(",")
    else:
        raw_items = list(raw)
    backends: list[str] = []
    for item in raw_items:
        backend = str(item).strip().lower()
        if not backend:
            continue
        if backend not in {"evac", "cosmos"}:
            raise ValueError(f"Unsupported world-model compare backend={backend!r}.")
        if backend not in backends:
            backends.append(backend)
    return backends or ["evac", "cosmos"]


def _copy_world_model_artifacts(src_dir: str | None, dst_dir: str, backend: str) -> None:
    if not src_dir:
        return
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    if src_dir == dst_dir:
        return
    os.makedirs(dst_dir, exist_ok=True)
    for filename in ("outputs.mp4", _world_model_runtime_meta_filename(backend)):
        src = os.path.join(src_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, filename))


def _serialize_world_model_request(
    save_dir: str,
    curr_image: torch.Tensor,
    fk_poses: list,
    grippers: list,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    pose_arr = np.zeros((len(fk_poses), 14), dtype=np.float32)
    grip_arr = np.zeros((len(grippers), 2), dtype=np.float32)
    for idx, ((lp, lq, rp, rq), (lg, rg)) in enumerate(zip(fk_poses, grippers)):
        pose_arr[idx, 0:3] = np.asarray(lp, dtype=np.float32).reshape(3,)
        pose_arr[idx, 3:7] = np.asarray(lq, dtype=np.float32).reshape(4,)
        pose_arr[idx, 7:10] = np.asarray(rp, dtype=np.float32).reshape(3,)
        pose_arr[idx, 10:14] = np.asarray(rq, dtype=np.float32).reshape(4,)
        grip_arr[idx, :] = [float(lg), float(rg)]
    img = torch.clamp(curr_image.detach().float().cpu(), 0.0, 1.0).numpy().astype(np.float32)
    request_path = os.path.join(save_dir, "world_model_request.npz")
    np.savez_compressed(
        request_path,
        curr_image=img,
        fk_poses=pose_arr,
        grippers=grip_arr,
    )
    return request_path


def _write_cosmos_offline_command(cfg: dict, save_dir: str, request_path: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    repo_root = str(cfg.get("compare_repo_root", "") or "/data/zhenyangfan/RoboTwin")
    command_path = os.path.join(save_dir, "run_cosmos_offline.sh")
    cmd = [
        sys.executable,
        "policy/SmolVLA/latentcorr/run_cosmos_compare_request.py",
        "--request_npz",
        request_path,
        "--save_dir",
        save_dir,
        "--cosmos_root",
        str(cfg.get("cosmos_root", "/data/zhenyangfan/cosmos-predict2.5")),
        "--checkpoint_path",
        str(cfg.get("cosmos_checkpoint_path", "")),
        "--experiment",
        str(cfg.get("cosmos_experiment", "robotwin_dualarm_actioncond_2b_256_320")),
        "--config_file",
        str(cfg.get("cosmos_config_file", "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py")),
        "--chunk_size",
        str(int(cfg.get("cosmos_chunk_size", 12))),
        "--guidance",
        str(int(cfg.get("cosmos_guidance", 7))),
        "--resolution",
        str(cfg.get("cosmos_resolution", "256,320")),
        "--fps_downsample_ratio",
        str(int(cfg.get("cosmos_fps_downsample_ratio", 1))),
        "--gripper_scale",
        str(float(cfg.get("cosmos_gripper_scale", 1.0))),
        "--num_steps",
        str(int(cfg.get("cosmos_num_steps", 35))),
        "--save_fps",
        str(int(cfg.get("cosmos_save_fps", 30))),
        "--num_latent_conditional_frames",
        str(int(cfg.get("cosmos_num_latent_conditional_frames", 1))),
        "--action_scaler",
        str(float(cfg.get("cosmos_action_scaler", 20.0))),
        "--action_stats_path",
        str(cfg.get("cosmos_action_stats_path", "")),
        "--action_normalization_clip",
        "" if cfg.get("cosmos_action_normalization_clip", None) in {None, "", "none", "None"} else str(cfg.get("cosmos_action_normalization_clip")),
        "--quat_input_order",
        str(cfg.get("cosmos_quat_input_order", "xyzw")),
        "--prompt",
        str(cfg.get("cosmos_prompt", "")),
        "--negative_prompt",
        str(cfg.get("cosmos_negative_prompt", "")),
        "--seed",
        str(int(cfg.get("cosmos_seed", 0))),
    ]
    cmd.append("--invert_gripper" if bool(cfg.get("cosmos_invert_gripper", True)) else "--no-invert_gripper")
    cmd.append("--use_quat" if bool(cfg.get("cosmos_use_quat", False)) else "--no-use_quat")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euo pipefail\n")
        f.write(f"cd {repo_root}\n")
        f.write(" ".join(cmd) + "\n")
    try:
        os.chmod(command_path, 0o755)
    except OSError:
        pass
    return command_path


def _world_model_video_record(
    *,
    backend: str,
    step: int,
    save_dir: str,
    source: str,
    error: str | None = None,
) -> dict:
    meta_name = _world_model_runtime_meta_filename(backend)
    video_path = os.path.join(save_dir, "outputs.mp4")
    meta_path = os.path.join(save_dir, meta_name)
    record = {
        "step": int(step),
        "backend": str(backend),
        "source": str(source),
        "path": _debug_relpath(video_path),
        "exists": bool(os.path.exists(video_path)),
        "runtime_meta_path": _debug_relpath(meta_path),
        "runtime_meta_exists": bool(os.path.exists(meta_path)),
    }
    if error:
        record["error"] = str(error)
    return record


def _run_world_model_compare(
    modules: dict,
    cfg: dict,
    curr_image: torch.Tensor,
    fk_poses: list,
    grip_list: list,
    raw_data: dict,
    device,
    *,
    debug_dir: str | None,
    step: int,
    main_backend: str,
    main_save_dir: str | None,
) -> list[dict]:
    if debug_dir is None:
        return []
    records: list[dict] = []
    for backend in _world_model_compare_backends(cfg):
        compare_dir = os.path.join(debug_dir, "compare", backend, f"rollout_step_{int(step):03d}")
        try:
            if backend == main_backend:
                _copy_world_model_artifacts(main_save_dir, compare_dir, backend)
                source = "main_backend_copy"
            elif backend == "cosmos" and not bool(cfg.get("world_model_compare_cosmos_autorun", False)):
                request_path = _serialize_world_model_request(compare_dir, curr_image, fk_poses, grip_list)
                command_path = _write_cosmos_offline_command(cfg, compare_dir, request_path)
                with open(os.path.join(compare_dir, "cosmos_offline_request.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "backend": "cosmos",
                            "step": int(step),
                            "request_npz": _debug_relpath(request_path),
                            "command_path": _debug_relpath(command_path),
                            "note": "Run command_path on an idle GPU to generate outputs.mp4.",
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                source = "compare_backend_offline_request"
            else:
                _world_model_inference(
                    modules,
                    cfg,
                    curr_image,
                    fk_poses,
                    grip_list,
                    raw_data,
                    device,
                    save_dir=compare_dir,
                    backend_override=backend,
                )
                source = "compare_backend_inference"
            records.append(
                _world_model_video_record(
                    backend=backend,
                    step=int(step),
                    save_dir=compare_dir,
                    source=source,
                )
            )
        except Exception as exc:
            os.makedirs(compare_dir, exist_ok=True)
            error_payload = {
                "backend": backend,
                "step": int(step),
                "error": repr(exc),
            }
            try:
                with open(os.path.join(compare_dir, "compare_error.json"), "w", encoding="utf-8") as f:
                    json.dump(error_payload, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
            records.append(
                _world_model_video_record(
                    backend=backend,
                    step=int(step),
                    save_dir=compare_dir,
                    source="compare_backend_error",
                    error=repr(exc),
                )
            )
            if bool(cfg.get("fail_fast_on_error", False)):
                raise
    return records


def _infer_compare_task_name_from_episode_path(episode_path: str | None) -> str:
    if not episode_path:
        return ""
    parts = os.path.normpath(str(episode_path)).split(os.sep)
    try:
        data_idx = len(parts) - 1 - parts[::-1].index("data")
    except ValueError:
        return ""
    if data_idx >= 2:
        task = parts[data_idx - 2]
        task_config = parts[data_idx - 1]
        if task and task_config:
            return f"sim-{task}-{task_config}-50"
    return ""


def _write_compare_sim_artifact(
    cfg: dict,
    debug_dir: str | None,
    raw_data: dict,
    start_ts: int,
    *,
    sampled_unit: dict,
    perturb_action_prefix_raw: np.ndarray | None,
    perturb_start_qpos_raw: np.ndarray | None,
    perturb_final_qpos_raw: np.ndarray | None,
    corr_action_chunk_raw: np.ndarray | None = None,
    corr_qpos_raw: np.ndarray | None = None,
    error_action_prefix_raw: np.ndarray | None = None,
) -> dict | None:
    if (
        debug_dir is None
        or not bool(cfg.get("world_model_compare_mode", False))
        or not bool(cfg.get("world_model_compare_sim", True))
    ):
        return None
    sim_dir = os.path.join(debug_dir, "compare", "sim")
    os.makedirs(sim_dir, exist_ok=True)
    episode_path = str(raw_data.get("episode_path", "") or "")
    raw_data_dir = str(cfg.get("compare_raw_data_dir", "") or "")
    if not raw_data_dir and episode_path:
        raw_data_dir = os.path.dirname(episode_path)
    task_name_full = str(cfg.get("compare_task_name_full", "") or "")
    if not task_name_full:
        task_name_full = _infer_compare_task_name_from_episode_path(episode_path)
    episode_id_value = cfg.get("compare_episode_id", None)
    if episode_id_value is None:
        match = re.search(r"episode(\d+)\.hdf5$", os.path.basename(episode_path))
        episode_id_value = int(match.group(1)) if match else -1
    rank = int(cfg.get("compare_rank", -1))
    global_step = int(cfg.get("compare_global_step", -1))
    episode_id = int(episode_id_value)
    sample_name = f"compare_sim_rank{rank:02d}_step{global_step:06d}_ep{episode_id:02d}_ts{int(start_ts):04d}.npz"
    sample_path = os.path.join(sim_dir, sample_name)
    manifest_path = os.path.join(sim_dir, f"correction_data_manifest_rank{max(0, rank):02d}.jsonl")
    corr_actions = (
        np.zeros((0, 14), dtype=np.float32)
        if corr_action_chunk_raw is None
        else np.asarray(corr_action_chunk_raw, dtype=np.float32)
    )
    arrays = {
        "corr_action_chunk_raw": corr_actions,
    }
    if perturb_action_prefix_raw is not None:
        # This is the exact action sequence used by EVAC/Cosmos world-model rollout.
        # Simulator compare should replay this sequence, not infer it from correction fields.
        arrays["sim_compare_action_chunk_raw"] = np.asarray(perturb_action_prefix_raw, dtype=np.float32)
    if corr_qpos_raw is not None:
        arrays["corr_qpos_raw"] = np.asarray(corr_qpos_raw, dtype=np.float32)
    if perturb_action_prefix_raw is not None:
        arrays["perturb_action_prefix_raw"] = np.asarray(perturb_action_prefix_raw, dtype=np.float32)
    if perturb_start_qpos_raw is not None:
        arrays["perturb_start_qpos_raw"] = np.asarray(perturb_start_qpos_raw, dtype=np.float32)
    if perturb_final_qpos_raw is not None:
        arrays["perturb_final_qpos_raw"] = np.asarray(perturb_final_qpos_raw, dtype=np.float32)
    if error_action_prefix_raw is not None:
        arrays["error_action_prefix_raw"] = np.asarray(error_action_prefix_raw, dtype=np.float32)
    np.savez_compressed(sample_path, **arrays)

    manifest_record = {
        "rank": int(rank),
        "global_step": int(global_step),
        "batch_index": 0,
        "path": str(sample_path),
        "task_name": str(task_name_full),
        "episode_id": int(episode_id),
        "start_ts": int(start_ts),
        "raw_data_dir": str(raw_data_dir),
        "sampled_phase_key": sampled_unit.get("phase_key"),
        "sampled_phase_bin_id": sampled_unit.get("phase_bin_id"),
        "sampled_phase_instance_idx": sampled_unit.get("phase_instance_idx"),
        "sampled_error_mode": sampled_unit.get("error_mode"),
        "sampled_active_arm_pattern": sampled_unit.get("active_arm_pattern"),
        "forced_dir_bin_id": sampled_unit.get("dir_bin_id"),
        "forced_mag_bin_id": sampled_unit.get("mag_bin_id"),
        "perturb_action_prefix_len": (
            0 if perturb_action_prefix_raw is None else int(np.asarray(perturb_action_prefix_raw).shape[0])
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

    repo_root = str(cfg.get("compare_repo_root", "") or "/data/zhenyangfan/RoboTwin")
    command = [
        sys.executable,
        "policy/SmolVLA/replay_correction_samples.py",
        "--run-dir",
        sim_dir,
        "--output-dir",
        sim_dir,
        "--select",
        "first",
        "--num-samples",
        "1",
        "--fps",
        str(int(cfg.get("world_model_compare_sim_fps", 10))),
        "--hold-frames",
        str(int(cfg.get("world_model_compare_sim_hold_frames", 4))),
    ]
    command_path = os.path.join(sim_dir, "run_sim_replay.sh")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euo pipefail\n")
        f.write(f"cd {repo_root}\n")
        f.write(" ".join(command) + "\n")
    try:
        os.chmod(command_path, 0o755)
    except OSError:
        pass

    result = {
        "sample_path": _debug_relpath(sample_path),
        "manifest_path": _debug_relpath(manifest_path),
        "command_path": _debug_relpath(command_path),
        "autorun": bool(cfg.get("world_model_compare_sim_autorun", False)),
        "note": "Run command_path to produce simulator replay video under compare/sim.",
    }
    if bool(cfg.get("world_model_compare_sim_autorun", False)):
        stdout_path = os.path.join(sim_dir, "sim_replay_stdout.txt")
        stderr_path = os.path.join(sim_dir, "sim_replay_stderr.txt")
        run_env = os.environ.copy()
        visible = [item.strip() for item in str(run_env.get("CUDA_VISIBLE_DEVICES", "")).split(",") if item.strip()]
        if visible and 0 <= rank < len(visible):
            # replay_correction_samples imports CUDA-backed planners that default to cuda:0.
            # In multi-rank compare runs, restrict each replay subprocess to its rank-local GPU.
            run_env["CUDA_VISIBLE_DEVICES"] = visible[rank]
        # This subprocess is just a local simulator replay.  It must not inherit
        # accelerate's distributed environment, otherwise CUDA/SAPIEN/Warp may
        # think it belongs to the outer 4-rank job.
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "ACCELERATE_PROCESS_INDEX",
            "ACCELERATE_NUM_PROCESSES",
            "ACCELERATE_LOCAL_PROCESS_INDEX",
        ):
            run_env.pop(key, None)
        run_env.setdefault("MPLCONFIGDIR", os.path.join(sim_dir, "mplconfig"))
        try:
            with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    stdout=out_f,
                    stderr=err_f,
                    check=False,
                    timeout=float(cfg.get("world_model_compare_sim_timeout_s", 600.0)),
                    env=run_env,
                )
            result.update(
                {
                    "returncode": int(completed.returncode),
                    "stdout_path": _debug_relpath(stdout_path),
                    "stderr_path": _debug_relpath(stderr_path),
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
    return result


def _image_sharpness_score(image_t: torch.Tensor) -> float:
    """Mean absolute gray-image gradient; lower means blurrier/flatter."""
    img = torch.clamp(image_t.detach().float(), 0.0, 1.0)
    if img.ndim != 3:
        return 0.0
    if img.shape[0] >= 3:
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        gray = img.mean(dim=0)
    dx = torch.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else torch.tensor(0.0, device=gray.device)
    dy = torch.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else torch.tensor(0.0, device=gray.device)
    return float((dx + dy).detach().cpu().item())


def _image_gray_tensor(image_t: torch.Tensor) -> torch.Tensor:
    img = torch.clamp(image_t.detach().float(), 0.0, 1.0)
    if img.ndim != 3:
        return torch.zeros(1, 1, device=img.device if isinstance(img, torch.Tensor) else "cpu")
    if img.shape[0] >= 3:
        return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return img.mean(dim=0)


def _gradient_magnitude(image_t: torch.Tensor) -> torch.Tensor:
    gray = _image_gray_tensor(image_t)
    if gray.ndim != 2 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return torch.zeros_like(gray).reshape(-1)
    dx = torch.zeros_like(gray)
    dy = torch.zeros_like(gray)
    dx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    dy[1:, :] = gray[1:, :] - gray[:-1, :]
    return torch.sqrt(dx * dx + dy * dy + 1e-12).reshape(-1)


def _centered_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.device != b.device:
        b = b.to(device=a.device)
    n = int(min(a.numel(), b.numel()))
    if n <= 1:
        return 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.detach().cpu().item()) <= 1e-12:
        return 0.0
    return float(torch.clamp(torch.dot(a, b) / denom, -1.0, 1.0).detach().cpu().item())


def _crop_image_patch(image_t: torch.Tensor, bbox_xyxy: tuple[int, int, int, int] | None) -> torch.Tensor:
    if bbox_xyxy is None or image_t.ndim != 3:
        return image_t
    _, height, width = image_t.shape
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x0 = int(np.clip(x0, 0, max(0, width - 1)))
    x1 = int(np.clip(x1, x0 + 1, width))
    y0 = int(np.clip(y0, 0, max(0, height - 1)))
    y1 = int(np.clip(y1, y0 + 1, height))
    return image_t[:, y0:y1, x0:x1]


def _evac_blur_filter_result(
    pred: torch.Tensor,
    ref: torch.Tensor,
    cfg: dict,
    bbox_xyxy: tuple[int, int, int, int] | None = None,
    error_mode: str | None = None,
) -> dict:
    pred_patch = _crop_image_patch(pred, bbox_xyxy)
    ref_patch = _crop_image_patch(ref, bbox_xyxy)
    pred_score = _image_sharpness_score(pred_patch)
    ref_score = _image_sharpness_score(ref_patch)
    sharpness_ratio = pred_score / max(ref_score, 1e-8)
    grad_cosine = _centered_cosine(_gradient_magnitude(pred_patch), _gradient_magnitude(ref_patch))
    requested_metric = str(cfg.get("evac_blur_filter_metric", "sharpness_ratio")).strip().lower()
    metric = requested_metric
    if metric == "mode_aware":
        mode_key = str(error_mode or "").strip().lower()
        metric = "grad_cosine" if mode_key == "gripper_close" else "sharpness_ratio"
    if metric not in {"sharpness_ratio", "grad_cosine"}:
        metric = "grad_cosine"
    score = grad_cosine if metric == "grad_cosine" else sharpness_ratio
    min_ratio = float(cfg.get("evac_blur_filter_min_ratio", 0.75))
    passed = bool(score >= min_ratio)
    return {
        "passed": passed,
        "requested_metric": requested_metric,
        "metric": metric,
        "score": float(score),
        "pred_sharpness": float(pred_score),
        "ref_sharpness": float(ref_score),
        "sharpness_ratio": float(sharpness_ratio),
        "grad_cosine": float(grad_cosine),
        "min_ratio": float(min_ratio),
        "region": "active_gripper_patch" if bbox_xyxy is not None else "full_image",
        "bbox_xyxy": None if bbox_xyxy is None else [int(v) for v in bbox_xyxy],
    }


def _correction_prefix_len(cfg: dict, rollout_exec_steps: int, chunk_size: int) -> int:
    if bool(cfg.get("full_chunk_recovery", False)):
        return int(max(1, chunk_size))
    return int(np.clip(int(rollout_exec_steps), 1, max(1, chunk_size - 1)))


def _pose_wxyz_to_matrix_np(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    mat = np.eye(4, dtype=np.float32)
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4,)
    mat[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().astype(np.float32)
    mat[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3,)
    return mat


def _project_active_gripper_bbox(
    raw_data: dict,
    active_arm: str,
    left_pos: np.ndarray,
    left_quat_wxyz: np.ndarray,
    right_pos: np.ndarray,
    right_quat_wxyz: np.ndarray,
    image_hw: tuple[int, int],
    pad_px: int = 12,
    axis_len_m: float = 0.04,
) -> tuple[int, int, int, int] | None:
    intrinsic = np.asarray(raw_data.get("intrinsic_cv"), dtype=np.float32).copy()
    ext_cv = np.asarray(raw_data.get("extrinsic_cv"), dtype=np.float32)
    native_h, native_w = raw_data.get("native_resolution", image_hw)
    img_h, img_w = int(image_hw[0]), int(image_hw[1])
    if intrinsic.shape != (3, 3) or ext_cv.shape != (3, 4):
        return None
    if int(native_w) > 0 and int(native_h) > 0:
        intrinsic[0, 0] *= float(img_w) / float(native_w)
        intrinsic[0, 2] *= float(img_w) / float(native_w)
        intrinsic[1, 1] *= float(img_h) / float(native_h)
        intrinsic[1, 2] *= float(img_h) / float(native_h)
    pose_mat = (
        _pose_wxyz_to_matrix_np(left_pos, left_quat_wxyz)
        if str(active_arm) == "left_arm"
        else _pose_wxyz_to_matrix_np(right_pos, right_quat_wxyz)
    )
    axis_len = float(np.clip(float(axis_len_m), 0.005, 0.20))
    end_effector_pts = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [axis_len, 0.0, 0.0, 1.0], [0.0, axis_len, 0.0, 1.0], [0.0, 0.0, axis_len, 1.0]],
        dtype=np.float32,
    ).T
    gripper_to_eef = np.asarray(
        [[1.0, 0.0, 0.0, 0.085], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pts_world = (pose_mat @ gripper_to_eef @ end_effector_pts)[:3, :].T
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = ext_cv
    uvs = []
    for pt in pts_world:
        cam = (w2c @ np.append(pt, 1.0).astype(np.float32))[:3]
        if (not np.all(np.isfinite(cam))) or float(cam[2]) <= 1e-6:
            continue
        uvw = intrinsic @ cam
        if (not np.all(np.isfinite(uvw))) or abs(float(uvw[2])) <= 1e-6:
            continue
        uvs.append((float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])))
    if not uvs:
        return None
    xs = np.asarray([uv[0] for uv in uvs], dtype=np.float32)
    ys = np.asarray([uv[1] for uv in uvs], dtype=np.float32)
    x0 = int(np.floor(float(np.min(xs)) - float(pad_px)))
    y0 = int(np.floor(float(np.min(ys)) - float(pad_px)))
    x1 = int(np.ceil(float(np.max(xs)) + float(pad_px)))
    y1 = int(np.ceil(float(np.max(ys)) + float(pad_px)))
    if x1 <= 0 or y1 <= 0 or x0 >= img_w or y0 >= img_h:
        return None
    return (
        int(np.clip(x0, 0, img_w - 1)),
        int(np.clip(y0, 0, img_h - 1)),
        int(np.clip(x1, 1, img_w)),
        int(np.clip(y1, 1, img_h)),
    )


def correction_step(policy_unwrapped, image_data_s, qpos_data_s, raw_data,
                    norm_stats, modules, cfg, device, debug_dir=None, start_ts=0,
                    sampled_phase_id=None,
                    sampled_phase_bin_id=None, sampled_phase_instance_id=None, forced_error_mode_id=None,
                    sampled_active_arm_pattern_id=None, original_active_arm_pattern_id=None, forced_dir_bin_id=None,
                    forced_mag_bin_id=None, sampled_mode_prob=None,
                    sampled_entry_prob_within_mode=None, sampled_unit_prob=None,
                    precomputed_action_chunk_raw=None):
    fk = modules['fk']
    world_model_backend = _world_model_backend(cfg)
    world_model_meta_filename = _world_model_runtime_meta_filename(world_model_backend)
    planner_l = modules['planner_left']
    planner_r = modules['planner_right']

    start_ts = int(max(0, start_ts))
    left_ep = raw_data['left_endpose'][start_ts:]
    right_ep = raw_data['right_endpose'][start_ts:]
    max_steps = cfg['max_rollout_steps']
    correction_force_generate = bool(cfg['correction_force_generate'])
    chunk_size = cfg['chunk_size']
    rollout_exec_steps = int(cfg['rollout_exec_steps'])
    max_action_len = cfg['max_action_len']
    orient_weight = float(cfg['orient_weight'])
    gripper_penalty = float(cfg['gripper_penalty'])
    recover_eval_enable = bool(cfg.get('recover_eval_enable', False))
    save_perturb_rollout_video = bool(cfg.get('save_perturb_rollout_video', False))
    recover_eval_save_video = bool(cfg.get('recover_eval_save_video', False))
    recover_eval_debug = bool(cfg.get('recover_eval_save_video', False))
    # Historical arg name: now used as GT-aligned gripper absolute-error threshold.
    recover_eval_gripper_open_thresh = float(cfg.get('recover_eval_gripper_open_thresh', 0.2))
    recover_eval_pos_thresh_m = float(cfg.get('recover_eval_pos_thresh_m', 0.03))
    recover_eval_rot_thresh_deg = float(cfg.get('recover_eval_rot_thresh_deg', 10.0))

    left_grip_traj = raw_data['left_gripper'][start_ts:]
    right_grip_traj = raw_data['right_gripper'][start_ts:]
    if sampled_phase_id is None:
        raise RuntimeError(
            f"Missing sampled_phase_id at correction_step (start_ts={int(start_ts)}). "
            "Dataloader must provide sampled_phase_id for each sample."
        )
    phase_key_fixed = phase_id_to_key(int(sampled_phase_id))
    phase_bin_fixed = (None if sampled_phase_bin_id is None else int(sampled_phase_bin_id))
    phase_instance_fixed = (None if sampled_phase_instance_id is None else int(sampled_phase_instance_id))
    forced_error_mode_key = None
    if forced_error_mode_id is not None and int(forced_error_mode_id) >= 0:
        forced_error_mode_key = error_mode_id_to_key(int(forced_error_mode_id))
    sampled_active_arm_pattern_key = None
    if sampled_active_arm_pattern_id is not None and int(sampled_active_arm_pattern_id) >= 0:
        sampled_active_arm_pattern_key = active_arm_pattern_id_to_key(int(sampled_active_arm_pattern_id))
    original_active_arm_pattern_key = None
    if original_active_arm_pattern_id is not None and int(original_active_arm_pattern_id) >= 0:
        original_active_arm_pattern_key = active_arm_pattern_id_to_key(int(original_active_arm_pattern_id))
    forced_dir_bin_fixed = (None if forced_dir_bin_id is None else int(forced_dir_bin_id))
    forced_mag_bin_fixed = (None if forced_mag_bin_id is None else int(forced_mag_bin_id))
    sampled_mode_prob_fixed = (
        None if sampled_mode_prob is None or (not np.isfinite(float(sampled_mode_prob)))
        else float(sampled_mode_prob)
    )
    sampled_entry_prob_within_mode_fixed = (
        None
        if sampled_entry_prob_within_mode is None or (not np.isfinite(float(sampled_entry_prob_within_mode)))
        else float(sampled_entry_prob_within_mode)
    )
    sampled_unit_prob_fixed = (
        None if sampled_unit_prob is None or (not np.isfinite(float(sampled_unit_prob)))
        else float(sampled_unit_prob)
    )
    phase_window_len = int(cfg['sample_phase_window_len'])
    phase_bins = int(cfg.get('failure_phase_bins', 5))
    sampled_unit = {
        "phase_key": str(phase_key_fixed),
        "phase_bin_id": (None if phase_bin_fixed is None else int(phase_bin_fixed)),
        "phase_instance_idx": (None if phase_instance_fixed is None else int(phase_instance_fixed)),
        "error_mode": (None if forced_error_mode_key is None else str(forced_error_mode_key)),
        "active_arm_pattern": (
            None if sampled_active_arm_pattern_key is None else str(sampled_active_arm_pattern_key)
        ),
        "original_active_arm_pattern": (
            None if original_active_arm_pattern_key is None else str(original_active_arm_pattern_key)
        ),
        "dir_bin_id": (None if forced_dir_bin_fixed is None else int(forced_dir_bin_fixed)),
        "mag_bin_id": (None if forced_mag_bin_fixed is None else int(forced_mag_bin_fixed)),
        "sampled_mode_prob": sampled_mode_prob_fixed,
        "sampled_entry_prob_within_mode": sampled_entry_prob_within_mode_fixed,
        "sampled_unit_prob": sampled_unit_prob_fixed,
    }

    qpos_raw = qpos_data_s.cpu().numpy() * norm_stats['qpos_std'] + norm_stats['qpos_mean']
    curr_image = image_data_s.clone()
    curr_qpos_raw = qpos_raw.copy()

    left_q = right_q = None
    left_grip = right_grip = 0.0
    t_star = 0
    min_dist = 0.0
    _dbg_rollout = []  # collect per-step debug info
    dyn_state = None
    act_raw = None
    perturb_action_prefix_chunks = []
    rollout_steps_total = 0
    evac_rollout_videos = []
    world_model_rollout_videos = []
    world_model_compare_records = []
    compare_sim_artifact = None
    recover_eval_any_unrecoverable = False
    recover_eval_first_unrecoverable_step = None
    recover_eval_last = {
        'recoverable': False,
        'mode': None,
        'metric_name': None,
        'metric': None,
        'threshold': None,
        'metrics': None,
        'thresholds': None,
        'passes': None,
        'failed_thresholds': None,
        'horizon': None,
        'gt_ref_idx': None,
    }
    failure_mode_key = str(cfg.get('failure_mode', 'off')).strip().lower()
    if failure_mode_key in {'explore', 'train'}:
        if sampled_active_arm_pattern_key is None:
            raise RuntimeError(
                "Missing sampled_active_arm_pattern in failure_mode. "
                "Dataloader must provide active_arm_pattern for each sample."
            )
        _pat = str(sampled_active_arm_pattern_key).strip().lower()
        if _pat in {"left_only", "left_arm"}:
            left_side_active, right_side_active = True, False
        elif _pat in {"right_only", "right_arm"}:
            left_side_active, right_side_active = False, True
        elif _pat == "both":
            left_side_active, right_side_active = True, True
        else:
            raise RuntimeError(
                f"Invalid sampled_active_arm_pattern={sampled_active_arm_pattern_key!r} "
                "in failure_mode. Expected left_arm/right_arm."
            )
        active_info_fixed = {
            "left_arm": bool(left_side_active),
            "right_arm": bool(right_side_active),
            "left_gripper": bool(left_side_active),
            "right_gripper": bool(right_side_active),
        }
        original_both_active = bool(str(original_active_arm_pattern_key).strip().lower() == "both")
    else:
        active_info_fixed = _infer_active_arms_from_gt_window(
            raw_data.get('gt_left_arm'),
            raw_data.get('gt_right_arm'),
            raw_data.get('left_gripper'),
            raw_data.get('right_gripper'),
            t_idx=start_ts,
            window_len=rollout_exec_steps,
            joint_delta_thresh=float(cfg['perturb_active_joint_delta_thresh']),
            gripper_delta_thresh=float(cfg['perturb_active_gripper_delta_thresh']),
        )
        left_side_active = bool(active_info_fixed['left_arm'] or active_info_fixed['left_gripper'])
        right_side_active = bool(active_info_fixed['right_arm'] or active_info_fixed['right_gripper'])
        active_info_fixed['left_arm'] = left_side_active
        active_info_fixed['right_arm'] = right_side_active
        active_info_fixed['left_gripper'] = left_side_active
        active_info_fixed['right_gripper'] = right_side_active
        original_both_active = bool(left_side_active and right_side_active)

    max_steps_eff = max_steps

    for step in range(max_steps_eff):
        world_model_debug_dir = (
            os.path.join(debug_dir, world_model_backend, f'rollout_step_{step:03d}')
            if (debug_dir and save_perturb_rollout_video)
            else None
        )
        left_q, right_q = curr_qpos_raw[0:6], curr_qpos_raw[7:13]
        left_grip, right_grip = curr_qpos_raw[6], curr_qpos_raw[13]
        fk_r = fk.forward(left_q, right_q)
        lp, lq = fk_r['left']
        rp, rq = fk_r['right']
        near_w = int(max(1, cfg.get('nearest_window_radius', 32)))
        expected_idx = int(np.clip(rollout_steps_total, 0, len(left_ep) - 1))
        t_star, min_dist = find_nearest_traj_point(
            lp, lq, rp, rq, left_ep, right_ep, orient_weight,
            curr_left_grip=left_grip, curr_right_grip=right_grip,
            left_gripper_traj=left_grip_traj, right_gripper_traj=right_grip_traj,
            gripper_penalty=gripper_penalty,
            window_start=max(0, expected_idx - near_w),
            window_end=min(len(left_ep), expected_idx + near_w + 1),
        )

        # Failure perturbation is defined from the sampled start state only.
        # Do not inherit future GT/policy chunk values; non-perturbed channels
        # should stay fixed at the start qpos/gripper.
        exec_len = int(max(1, int(rollout_exec_steps)))
        act_raw = np.repeat(curr_qpos_raw[None, :], exec_len, axis=0).astype(np.float32)

        active_info = active_info_fixed
        progress = float(t_star) / max(1, len(left_ep) - 1)
        # Choose error phase from the whole GT action window (prefix) instead of a single anchor.
        # Keep phase consistent with sampled start_ts stage.
        phase_key = phase_key_fixed
        act_raw, perturbed, pert_mode, dyn_state = perturb_action_chunk_online(
            act_raw, cfg, dyn_state, fk=fk, planner_l=planner_l, planner_r=planner_r,
            phase_key=phase_key, init_left_grip=float(left_grip), init_right_grip=float(right_grip),
            curr_left_q=left_q, curr_right_q=right_q,
            active_left_arm=active_info['left_arm'], active_right_arm=active_info['right_arm'],
            active_left_gripper=active_info['left_gripper'], active_right_gripper=active_info['right_gripper'],
            left_ep=left_ep, right_ep=right_ep,
            left_grip_traj=left_grip_traj, right_grip_traj=right_grip_traj,
            t_star=int(t_star), rollout_exec_steps=int(rollout_exec_steps),
            forced_error_mode=forced_error_mode_key,
            forced_dir_bin_id=forced_dir_bin_fixed,
            forced_mag_bin_id=forced_mag_bin_fixed,
        )
        perturb_action_prefix_chunks.append(np.asarray(act_raw, dtype=np.float32).copy())

        fk_poses = [(lp.copy(), lq.copy(), rp.copy(), rq.copy())]  # current state
        grip_list = [(left_grip, right_grip)]
        # act_raw is already generated at rollout_exec_steps horizon.
        for ai in range(len(act_raw)):
            ar = act_raw[ai]
            fr = fk.forward(ar[0:6], ar[7:13])
            fk_poses.append((fr['left'][0].copy(), fr['left'][1].copy(),
                             fr['right'][0].copy(), fr['right'][1].copy()))
            grip_list.append((ar[6], ar[13]))

        pred = _world_model_inference(
            modules,
            cfg,
            curr_image[0],
            fk_poses,
            grip_list,
            raw_data,
            device,
            save_dir=world_model_debug_dir,
        )
        if world_model_debug_dir is not None:
            _main_video_record = _world_model_video_record(
                backend=world_model_backend,
                step=int(step),
                save_dir=world_model_debug_dir,
                source="main_backend",
            )
            evac_rollout_videos.append(_main_video_record)
            world_model_rollout_videos.append(_main_video_record)
            _compare_records = _run_world_model_compare(
                modules,
                cfg,
                curr_image[0],
                fk_poses,
                grip_list,
                raw_data,
                device,
                debug_dir=debug_dir,
                step=int(step),
                main_backend=world_model_backend,
                main_save_dir=world_model_debug_dir,
            )
            world_model_compare_records.extend(_compare_records)
        new_img = curr_image.clone()
        if tuple(pred.shape) != tuple(new_img[0].shape):
            pred_rs = torch.nn.functional.interpolate(
                pred.unsqueeze(0),
                size=(new_img.shape[-2], new_img.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )[0]
            pred = torch.clamp(pred_rs, 0.0, 1.0)
        evac_blur_filter = None
        if bool(cfg.get("evac_blur_filter_enable", False)):
            blur_region = str(cfg.get("evac_blur_filter_region", "active_gripper_patch")).strip().lower()
            blur_bbox = None
            if blur_region == "active_gripper_patch":
                active_arm_for_blur = "left_arm" if bool(active_info.get("left_arm", False)) else "right_arm"
                blur_lp, blur_lq, blur_rp, blur_rq = fk_poses[-1]
                blur_bbox = _project_active_gripper_bbox(
                    raw_data,
                    active_arm_for_blur,
                    blur_lp,
                    blur_lq,
                    blur_rp,
                    blur_rq,
                    image_hw=(int(pred.shape[-2]), int(pred.shape[-1])),
                    pad_px=int(cfg.get("evac_blur_filter_patch_pad_px", 12)),
                    axis_len_m=float(cfg.get("evac_blur_filter_gripper_axis_m", 0.04)),
                )
            sampled_error_mode = dyn_state.get('error_mode') if isinstance(dyn_state, dict) else forced_error_mode_key
            evac_blur_filter = _evac_blur_filter_result(
                pred,
                curr_image[0],
                cfg,
                bbox_xyxy=blur_bbox,
                error_mode=sampled_error_mode,
            )
            if not bool(evac_blur_filter.get("passed", False)):
                blur_rollout_record = {
                    "step": int(step),
                    "t_star": int(t_star),
                    "min_dist": float(min_dist),
                    "progress": float(progress),
                    "phase_key": phase_key,
                    "action_perturbed": bool(perturbed),
                    "perturb_mode": pert_mode,
                    "sampled_error_mode": sampled_error_mode,
                    "evac_blur_filter": evac_blur_filter,
                }
                perturb_action_prefix_raw = (
                    np.concatenate(perturb_action_prefix_chunks, axis=0).astype(np.float32)
                    if len(perturb_action_prefix_chunks) > 0
                    else None
                )
                perturb_start_qpos_raw = np.asarray(qpos_raw, dtype=np.float32).copy()
                perturb_final_qpos_raw = (
                    np.asarray(act_raw[-1], dtype=np.float32).copy()
                    if act_raw is not None and len(act_raw) > 0
                    else np.asarray(curr_qpos_raw, dtype=np.float32).copy()
                )
                compare_sim_artifact = _write_compare_sim_artifact(
                    cfg,
                    debug_dir,
                    raw_data,
                    int(start_ts),
                    sampled_unit=sampled_unit,
                    perturb_action_prefix_raw=perturb_action_prefix_raw,
                    perturb_start_qpos_raw=perturb_start_qpos_raw,
                    perturb_final_qpos_raw=perturb_final_qpos_raw,
                )
                corr_meta = {
                    "correction_generated": False,
                    "closed_loop_fallback_used": False,
                    "correction_branch": "evac_invalid",
                    "skip_reason": "evac_blurry_after_perturb",
                    "invalid_trial": True,
                    "invalid_reason": "evac_blurry_after_perturb",
                    "sampled_error_mode": sampled_error_mode,
                    "forced_error_mode": forced_error_mode_key,
                    "sampled_phase_key": phase_key_fixed,
                    "sampled_phase_bin_id": phase_bin_fixed,
                    "sampled_phase_instance_idx": phase_instance_fixed,
                    "sampled_active_arm_pattern": sampled_active_arm_pattern_key,
                    "forced_dir_bin_id": forced_dir_bin_fixed,
                    "forced_mag_bin_id": forced_mag_bin_fixed,
                    "nearest_mode": str(sampled_error_mode or ""),
                    "rollout_exec_steps": int(rollout_exec_steps),
                    "t_star": int(t_star),
                    "min_dist": float(min_dist),
                    "recover_eval_enable": bool(recover_eval_enable),
                    "recover_eval_last": {"recoverable": None, "mode": sampled_error_mode},
                    "world_model_backend": world_model_backend,
                    "evac_blur_filter": evac_blur_filter,
                    "evac_rollout_videos": evac_rollout_videos,
                    "world_model_rollout_videos": world_model_rollout_videos,
                    "world_model_compare": world_model_compare_records,
                    "world_model_compare_sim": compare_sim_artifact,
                    "debug_dir": debug_dir,
                    "error_action_prefix_raw": np.asarray(act_raw, dtype=np.float32).copy(),
                }
                if debug_dir is not None:
                    _dbg_corr = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr, exist_ok=True)
                    try:
                        save_perturb_compare_image(
                            _dbg_corr,
                            image_data_s[0],
                            pred,
                            sampled_unit,
                            rollout_last_record=blur_rollout_record,
                        )
                    except Exception:
                        pass
                    with open(os.path.join(_dbg_corr, 'skipped.json'), 'w', encoding='utf-8') as _f:
                        json.dump({
                            'reason': 'evac_blurry_after_perturb',
                            'sampled_unit': sampled_unit,
                            'evac_blur_filter': evac_blur_filter,
                            't_star': int(t_star),
                            'min_dist': float(min_dist),
                            'world_model_compare_sim': compare_sim_artifact,
                            'rollout': _dbg_rollout + [blur_rollout_record],
                        }, _f, indent=2, ensure_ascii=False)
                return (None, None, None, None, corr_meta)
        new_img[0] = pred
        curr_image = new_img
        # Advance state to the end of executed actions.
        a_last = act_raw[-1]
        curr_qpos_raw = a_last
        rollout_steps_total += int(len(act_raw))
        left_q_e, right_q_e = curr_qpos_raw[0:6], curr_qpos_raw[7:13]
        left_grip, right_grip = curr_qpos_raw[6], curr_qpos_raw[13]
        fk_r_e = fk.forward(left_q_e, right_q_e)
        lp_e, lq_e = fk_r_e['left']
        rp_e, rq_e = fk_r_e['right']
        sampled_error_mode = None
        perturb_dir_bin_id = None
        perturb_mag_bin_id = None
        perturb_axis_bin_id = None
        perturb_translation_gain_m = None
        perturb_rotation_deg = None
        perturb_dir_left = None
        perturb_dir_right = None
        perturb_axis_left = None
        perturb_axis_right = None
        if isinstance(dyn_state, dict):
            sampled_error_mode = dyn_state.get('error_mode')
            perturb_dir_bin_id = dyn_state.get('dir_bin_id')
            perturb_mag_bin_id = dyn_state.get('mag_bin_id')
            perturb_axis_bin_id = dyn_state.get('axis_bin_id')
            perturb_translation_gain_m = dyn_state.get('perturb_translation_gain_m')
            perturb_rotation_deg = dyn_state.get('perturb_rotation_deg')
            perturb_dir_left = dyn_state.get('perturb_dir_left')
            perturb_dir_right = dyn_state.get('perturb_dir_right')
            perturb_axis_left = dyn_state.get('perturb_axis_left')
            perturb_axis_right = dyn_state.get('perturb_axis_right')

        recover_eval_mode = sampled_error_mode
        recover_eval_recoverable = None
        recover_eval_metric_name = None
        recover_eval_metric = None
        recover_eval_threshold = None
        recover_eval_metrics = None
        recover_eval_thresholds = None
        recover_eval_passes = None
        recover_eval_failed_thresholds = None
        recover_eval_horizon = None
        recover_eval_gt_ref_idx = None
        recover_eval_nearest_dist = None
        recover_eval_window_start = None
        recover_eval_window_end = None
        recover_eval_error = None
        recover_eval_video = None
        recover_eval_compare = None

        if recover_eval_enable:
            try:
                recover_eval_use_rollout_state = bool(cfg.get('enable_perturb', True))
                if recover_eval_use_rollout_state:
                    qt_eval = torch.from_numpy(curr_qpos_raw).float().unsqueeze(0).to(device)
                    it_eval = curr_image.unsqueeze(0).to(device)
                else:
                    qpos_eval_raw = qpos_data_s.detach().cpu().numpy() * norm_stats['qpos_std'] + norm_stats['qpos_mean']
                    qt_eval = torch.from_numpy(qpos_eval_raw).float().unsqueeze(0).to(device)
                    it_eval = image_data_s.unsqueeze(0).to(device)
                recover_eval_batch = None
                recover_eval_action_post = None

                _was_training_eval = policy_unwrapped.training
                policy_unwrapped.eval()
                try:
                    with torch.no_grad():
                        if not hasattr(policy_unwrapped, "build_batch"):
                            raise AttributeError("recover-eval policy adapter must implement build_batch")
                        if not hasattr(policy_unwrapped, "predict_base_action_chunk"):
                            raise AttributeError("recover-eval policy adapter must implement predict_base_action_chunk")
                        if not hasattr(policy_unwrapped, "postprocess_action_chunk"):
                            raise AttributeError("recover-eval policy adapter must implement postprocess_action_chunk")
                        if hasattr(policy_unwrapped, "build_eval_batch"):
                            recover_eval_batch = policy_unwrapped.build_eval_batch(
                                image_t=it_eval[0],
                                qpos_raw=qt_eval[0],
                            )
                        else:
                            action_dim = int(qt_eval.shape[-1])
                            chunk_size_eval = int(cfg.get("chunk_size", rollout_exec_steps))
                            dummy_action_chunk = torch.zeros(
                                (chunk_size_eval, action_dim),
                                dtype=torch.float32,
                                device=qt_eval.device,
                            )
                            recover_eval_batch = policy_unwrapped.build_batch(
                                image_t=it_eval[0],
                                qpos_raw=qt_eval[0],
                                action_chunk_raw=dummy_action_chunk,
                            )
                        act_chunk_eval = policy_unwrapped.predict_base_action_chunk(recover_eval_batch)
                        recover_eval_action_post = policy_unwrapped.postprocess_action_chunk(act_chunk_eval)
                finally:
                    if _was_training_eval:
                        policy_unwrapped.train()
                act_eval_raw = recover_eval_action_post.squeeze(0).detach().cpu().numpy().astype(np.float32)

                h_eval = int(np.clip(rollout_exec_steps, 1, max(1, act_eval_raw.shape[0])))
                act_eval_raw = act_eval_raw[:h_eval].copy()
                k_eval = int(max(0, h_eval - 1))
                fr_eval_last = fk.forward(act_eval_raw[k_eval, 0:6], act_eval_raw[k_eval, 7:13])
                pred_left_grip_eval = float(np.clip(act_eval_raw[k_eval, 6], 0.0, 1.0))
                pred_right_grip_eval = float(np.clip(act_eval_raw[k_eval, 13], 0.0, 1.0))
                recover_eval_horizon = int(h_eval)
                gt_ref_idx = int(np.clip(start_ts + h_eval, 0, raw_data['left_endpose'].shape[0] - 1))
                recover_eval_rollout_last_img = None

                if recover_eval_save_video and (debug_dir is not None):
                    try:
                        # Video-only smoothing: bridge from current perturbed state
                        # to ACT first recover action using planner for 16 steps.
                        bridge_steps = int(max(1, cfg.get('recover_eval_video_bridge_steps', 16)))
                        act_eval_vis_raw = np.asarray(act_eval_raw, dtype=np.float32).copy()
                        _bridge_meta = {
                            'bridge_steps': int(bridge_steps),
                            'left_status': 'Inactive',
                            'right_status': 'Inactive',
                        }
                        if act_eval_vis_raw.shape[0] > 0:
                            _target0 = np.asarray(act_eval_vis_raw[0], dtype=np.float32)
                            _prefix = np.repeat(_target0[None, :], bridge_steps, axis=0).astype(np.float32)
                            _prefix[:, 0:6] = np.repeat(np.asarray(left_q_e, dtype=np.float32)[None, :], bridge_steps, axis=0)
                            _prefix[:, 7:13] = np.repeat(np.asarray(right_q_e, dtype=np.float32)[None, :], bridge_steps, axis=0)

                            if bool(active_info.get('left_gripper', True)):
                                _prefix[:, 6] = np.linspace(float(left_grip), float(_target0[6]), bridge_steps, dtype=np.float32)
                            else:
                                _prefix[:, 6] = float(left_grip)
                            if bool(active_info.get('right_gripper', True)):
                                _prefix[:, 13] = np.linspace(float(right_grip), float(_target0[13]), bridge_steps, dtype=np.float32)
                            else:
                                _prefix[:, 13] = float(right_grip)

                            _qpos_full = np.zeros(len(fk.jnames), dtype=np.float32)
                            _qpos_full[fk.fl_idx] = np.asarray(left_q_e, dtype=np.float32)
                            _qpos_full[fk.fr_idx] = np.asarray(right_q_e, dtype=np.float32)
                            import sapien
                            _fr_tgt = fk.forward(_target0[0:6], _target0[7:13])

                            if bool(active_info.get('left_arm', True)):
                                _pose_l = sapien.Pose(
                                    np.asarray(_fr_tgt['left'][0], dtype=np.float32),
                                    np.asarray(_fr_tgt['left'][1], dtype=np.float32),
                                )
                                _res_l = planner_l.plan_path(_qpos_full, _pose_l, arms_tag='left')
                                _bridge_meta['left_status'] = str(_res_l.get('status'))
                                if str(_res_l.get('status')) == 'Success':
                                    _path_l = np.asarray(_res_l.get('position', []), dtype=np.float32)
                                    if _path_l.ndim == 2 and _path_l.shape[0] > 0:
                                        _future_l = _path_l[1:] if _path_l.shape[0] > 1 else _path_l
                                        _prefix[:, 0:6] = resample_trajectory(_future_l, bridge_steps).astype(np.float32)

                            if bool(active_info.get('right_arm', True)):
                                _pose_r = sapien.Pose(
                                    np.asarray(_fr_tgt['right'][0], dtype=np.float32),
                                    np.asarray(_fr_tgt['right'][1], dtype=np.float32),
                                )
                                _res_r = planner_r.plan_path(_qpos_full, _pose_r, arms_tag='right')
                                _bridge_meta['right_status'] = str(_res_r.get('status'))
                                if str(_res_r.get('status')) == 'Success':
                                    _path_r = np.asarray(_res_r.get('position', []), dtype=np.float32)
                                    if _path_r.ndim == 2 and _path_r.shape[0] > 0:
                                        _future_r = _path_r[1:] if _path_r.shape[0] > 1 else _path_r
                                        _prefix[:, 7:13] = resample_trajectory(_future_r, bridge_steps).astype(np.float32)

                            act_eval_vis_raw = np.concatenate([_prefix, act_eval_vis_raw], axis=0).astype(np.float32)

                        fk_eval_poses = [(lp_e.copy(), lq_e.copy(), rp_e.copy(), rq_e.copy())]
                        grip_eval_list = [(float(left_grip), float(right_grip))]
                        for _ai in range(act_eval_vis_raw.shape[0]):
                            _ar = act_eval_vis_raw[_ai]
                            _fr = fk.forward(_ar[0:6], _ar[7:13])
                            fk_eval_poses.append((
                                _fr['left'][0].copy(), _fr['left'][1].copy(),
                                _fr['right'][0].copy(), _fr['right'][1].copy(),
                            ))
                            grip_eval_list.append((float(_ar[6]), float(_ar[13])))
                        world_model_recover_eval_dir = os.path.join(
                            debug_dir, f'{world_model_backend}_recover_eval', f'rollout_step_{step:03d}'
                        )
                        recover_eval_rollout_last_img = _world_model_inference(
                            modules,
                            cfg,
                            curr_image[0],
                            fk_eval_poses,
                            grip_eval_list,
                            raw_data,
                            device,
                            save_dir=world_model_recover_eval_dir,
                        )
                        _vpath_recover = os.path.join(world_model_recover_eval_dir, 'outputs.mp4')
                        _mpath_recover = os.path.join(world_model_recover_eval_dir, world_model_meta_filename)
                        recover_eval_video = {
                            'path': _debug_relpath(_vpath_recover),
                            'exists': bool(os.path.exists(_vpath_recover)),
                            'runtime_meta_path': _debug_relpath(_mpath_recover),
                            'runtime_meta_exists': bool(os.path.exists(_mpath_recover)),
                            'backend': world_model_backend,
                            'bridge_steps': int(_bridge_meta.get('bridge_steps', 0)),
                            'bridge_left_status': _bridge_meta.get('left_status'),
                            'bridge_right_status': _bridge_meta.get('right_status'),
                        }
                    except Exception as _exc_recover_video:
                        recover_eval_video = {
                            'path': None,
                            'exists': False,
                            'error': str(_exc_recover_video),
                        }
                        if bool(cfg.get('fail_fast_on_error', False)):
                            raise

                mode_eval = str(recover_eval_mode).strip().lower() if recover_eval_mode is not None else ""
                if mode_eval == "gripper_close":
                    fr_eval = fr_eval_last
                    near_w_eval = int(max(1, cfg.get('recover_eval_nearest_window_radius', int(max(1, rollout_exec_steps)))))
                    ws = int(np.clip(gt_ref_idx, 0, raw_data['left_endpose'].shape[0] - 1))
                    we = int(np.clip(gt_ref_idx + near_w_eval + 1, ws + 1, raw_data['left_endpose'].shape[0]))
                    nidx_abs, ndist = find_nearest_traj_point(
                        np.asarray(fr_eval['left'][0], dtype=np.float32),
                        np.asarray(fr_eval['left'][1], dtype=np.float32),
                        np.asarray(fr_eval['right'][0], dtype=np.float32),
                        np.asarray(fr_eval['right'][1], dtype=np.float32),
                        raw_data['left_endpose'],
                        raw_data['right_endpose'],
                        orient_weight=orient_weight,
                        curr_left_grip=pred_left_grip_eval,
                        curr_right_grip=pred_right_grip_eval,
                        left_gripper_traj=raw_data.get('left_gripper'),
                        right_gripper_traj=raw_data.get('right_gripper'),
                        gripper_penalty=gripper_penalty,
                        window_start=ws,
                        window_end=we,
                    )
                    recover_eval_nearest_dist = float(ndist)
                    recover_eval_window_start = int(ws)
                    recover_eval_window_end = int(we)
                    recover_eval_gt_ref_idx = int(nidx_abs)
                    gripper_errs = []
                    if bool(active_info.get('left_gripper', True)):
                        gripper_errs.append(float(abs(
                            pred_left_grip_eval -
                            float(np.clip(raw_data['left_gripper'][recover_eval_gt_ref_idx], 0.0, 1.0))
                        )))
                    if bool(active_info.get('right_gripper', True)):
                        gripper_errs.append(float(abs(
                            pred_right_grip_eval -
                            float(np.clip(raw_data['right_gripper'][recover_eval_gt_ref_idx], 0.0, 1.0))
                        )))
                    recover_eval_metric_name = "gripper_err"
                    recover_eval_metric = (None if len(gripper_errs) == 0 else float(np.mean(gripper_errs)))
                    recover_eval_threshold = float(recover_eval_gripper_open_thresh)
                    recover_eval_recoverable = (
                        recover_eval_metric is not None and
                        float(recover_eval_metric) <= float(recover_eval_threshold)
                    )
                elif mode_eval == "translation":
                    fr_eval = fr_eval_last
                    # Forward-only nearest matching around center=start_ts+H,
                    # using the same full metric as rollout nearest search.
                    near_w_eval = int(max(1, cfg.get('recover_eval_nearest_window_radius', int(max(1, rollout_exec_steps)))))
                    ws = int(np.clip(gt_ref_idx, 0, raw_data['left_endpose'].shape[0] - 1))
                    we = int(np.clip(gt_ref_idx + near_w_eval + 1, ws + 1, raw_data['left_endpose'].shape[0]))
                    nidx_abs, ndist = find_nearest_traj_point(
                        np.asarray(fr_eval['left'][0], dtype=np.float32),
                        np.asarray(fr_eval['left'][1], dtype=np.float32),
                        np.asarray(fr_eval['right'][0], dtype=np.float32),
                        np.asarray(fr_eval['right'][1], dtype=np.float32),
                        raw_data['left_endpose'],
                        raw_data['right_endpose'],
                        orient_weight=orient_weight,
                        curr_left_grip=pred_left_grip_eval,
                        curr_right_grip=pred_right_grip_eval,
                        left_gripper_traj=raw_data.get('left_gripper'),
                        right_gripper_traj=raw_data.get('right_gripper'),
                        gripper_penalty=gripper_penalty,
                        window_start=ws,
                        window_end=we,
                    )
                    recover_eval_nearest_dist = float(ndist)
                    recover_eval_window_start = int(ws)
                    recover_eval_window_end = int(we)
                    recover_eval_gt_ref_idx = int(nidx_abs)
                    pos_errs = []
                    if bool(active_info.get('left_arm', True)):
                        pos_errs.append(float(np.linalg.norm(
                            np.asarray(fr_eval['left'][0], dtype=np.float32) -
                            np.asarray(raw_data['left_endpose'][recover_eval_gt_ref_idx, :3], dtype=np.float32)
                        )))
                    if bool(active_info.get('right_arm', True)):
                        pos_errs.append(float(np.linalg.norm(
                            np.asarray(fr_eval['right'][0], dtype=np.float32) -
                            np.asarray(raw_data['right_endpose'][recover_eval_gt_ref_idx, :3], dtype=np.float32)
                        )))
                    recover_eval_metric_name = "pos_err_m"
                    recover_eval_metric = (None if len(pos_errs) == 0 else float(np.mean(pos_errs)))
                    recover_eval_threshold = float(recover_eval_pos_thresh_m)
                    recover_eval_recoverable = (
                        recover_eval_metric is not None and
                        float(recover_eval_metric) <= float(recover_eval_threshold)
                    )
                elif mode_eval == "rotation":
                    fr_eval = fr_eval_last
                    # Forward-only nearest matching around center=start_ts+H,
                    # using the same full metric as rollout nearest search.
                    near_w_eval = int(max(1, cfg.get('recover_eval_nearest_window_radius', int(max(1, rollout_exec_steps)))))
                    ws = int(np.clip(gt_ref_idx, 0, raw_data['left_endpose'].shape[0] - 1))
                    we = int(np.clip(gt_ref_idx + near_w_eval + 1, ws + 1, raw_data['left_endpose'].shape[0]))
                    nidx_abs, ndist = find_nearest_traj_point(
                        np.asarray(fr_eval['left'][0], dtype=np.float32),
                        np.asarray(fr_eval['left'][1], dtype=np.float32),
                        np.asarray(fr_eval['right'][0], dtype=np.float32),
                        np.asarray(fr_eval['right'][1], dtype=np.float32),
                        raw_data['left_endpose'],
                        raw_data['right_endpose'],
                        orient_weight=orient_weight,
                        curr_left_grip=pred_left_grip_eval,
                        curr_right_grip=pred_right_grip_eval,
                        left_gripper_traj=raw_data.get('left_gripper'),
                        right_gripper_traj=raw_data.get('right_gripper'),
                        gripper_penalty=gripper_penalty,
                        window_start=ws,
                        window_end=we,
                    )
                    recover_eval_nearest_dist = float(ndist)
                    recover_eval_window_start = int(ws)
                    recover_eval_window_end = int(we)
                    recover_eval_gt_ref_idx = int(nidx_abs)
                    rot_errs = []
                    if bool(active_info.get('left_arm', True)):
                        rot_errs.append(quat_geodesic_deg_wxyz(
                            fr_eval['left'][1],
                            raw_data['left_endpose'][recover_eval_gt_ref_idx, 3:7],
                        ))
                    if bool(active_info.get('right_arm', True)):
                        rot_errs.append(quat_geodesic_deg_wxyz(
                            fr_eval['right'][1],
                            raw_data['right_endpose'][recover_eval_gt_ref_idx, 3:7],
                        ))
                    recover_eval_metric_name = "rot_err_deg"
                    recover_eval_metric = (None if len(rot_errs) == 0 else float(np.mean(rot_errs)))
                    recover_eval_threshold = float(recover_eval_rot_thresh_deg)
                    recover_eval_recoverable = (
                        recover_eval_metric is not None and
                        float(recover_eval_metric) <= float(recover_eval_threshold)
                    )
                else:
                    recover_eval_recoverable = None

                if mode_eval in {"translation", "rotation", "gripper_close"} and recover_eval_gt_ref_idx is not None:
                    ref_idx = int(recover_eval_gt_ref_idx)
                    pos_errs = []
                    if bool(active_info.get('left_arm', True)):
                        pos_errs.append(float(np.linalg.norm(
                            np.asarray(fr_eval_last['left'][0], dtype=np.float32) -
                            np.asarray(raw_data['left_endpose'][ref_idx, :3], dtype=np.float32)
                        )))
                    if bool(active_info.get('right_arm', True)):
                        pos_errs.append(float(np.linalg.norm(
                            np.asarray(fr_eval_last['right'][0], dtype=np.float32) -
                            np.asarray(raw_data['right_endpose'][ref_idx, :3], dtype=np.float32)
                        )))

                    rot_errs = []
                    if bool(active_info.get('left_arm', True)):
                        rot_errs.append(quat_geodesic_deg_wxyz(
                            fr_eval_last['left'][1],
                            raw_data['left_endpose'][ref_idx, 3:7],
                        ))
                    if bool(active_info.get('right_arm', True)):
                        rot_errs.append(quat_geodesic_deg_wxyz(
                            fr_eval_last['right'][1],
                            raw_data['right_endpose'][ref_idx, 3:7],
                        ))

                    gripper_errs = []
                    if bool(active_info.get('left_gripper', True)):
                        gripper_errs.append(float(abs(
                            pred_left_grip_eval -
                            float(np.clip(raw_data['left_gripper'][ref_idx], 0.0, 1.0))
                        )))
                    if bool(active_info.get('right_gripper', True)):
                        gripper_errs.append(float(abs(
                            pred_right_grip_eval -
                            float(np.clip(raw_data['right_gripper'][ref_idx], 0.0, 1.0))
                        )))

                    recover_eval_metrics = {
                        "pos_err_m": None if len(pos_errs) == 0 else float(np.mean(pos_errs)),
                        "rot_err_deg": None if len(rot_errs) == 0 else float(np.mean(rot_errs)),
                        "gripper_err": None if len(gripper_errs) == 0 else float(np.mean(gripper_errs)),
                    }
                    recover_eval_thresholds = {
                        "pos_err_m": float(recover_eval_pos_thresh_m),
                        "rot_err_deg": float(recover_eval_rot_thresh_deg),
                        "gripper_err": float(recover_eval_gripper_open_thresh),
                    }
                    recover_eval_passes = {}
                    for _name, _value in recover_eval_metrics.items():
                        _threshold = recover_eval_thresholds.get(_name)
                        if _value is None or _threshold is None or float(_threshold) <= 0.0:
                            recover_eval_passes[_name] = False
                        else:
                            recover_eval_passes[_name] = bool(float(_value) <= float(_threshold))
                    recover_eval_failed_thresholds = [
                        str(name) for name, passed in recover_eval_passes.items()
                        if not bool(passed)
                    ]
                    recover_eval_metric_name = "all_thresholds_passed"
                    recover_eval_metric = None
                    recover_eval_threshold = None
                    recover_eval_recoverable = bool(
                        recover_eval_passes
                        and all(bool(v) for v in recover_eval_passes.values())
                    )

                if (
                    (recover_eval_gt_ref_idx is not None)
                    and recover_eval_debug
                    and (debug_dir is not None)
                    and (recover_eval_rollout_last_img is not None)
                ):
                    _dbg_corr_cmp = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr_cmp, exist_ok=True)
                    recover_eval_compare = save_recover_eval_compare_image(
                        _dbg_corr_cmp,
                        raw_data,
                        int(recover_eval_gt_ref_idx),
                        recover_eval_rollout_last_img,
                        int(step),
                        mode=mode_eval,
                        recoverable=recover_eval_recoverable,
                        metric_name=recover_eval_metric_name,
                        metric=recover_eval_metric,
                        threshold=recover_eval_threshold,
                        metrics=recover_eval_metrics,
                        thresholds=recover_eval_thresholds,
                        passes=recover_eval_passes,
                        failed_thresholds=recover_eval_failed_thresholds,
                        nearest_dist=recover_eval_nearest_dist,
                    )
            except Exception as exc:
                recover_eval_recoverable = None
                recover_eval_error = str(exc)
                if debug_dir is not None:
                    _dbg_corr = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr, exist_ok=True)
                    try:
                        import traceback

                        with open(
                            os.path.join(_dbg_corr, f'recover_eval_error_step_{step:03d}.txt'),
                            'w',
                            encoding='utf-8',
                        ) as _f:
                            _f.write(traceback.format_exc())
                        with open(
                            os.path.join(_dbg_corr, f'recover_eval_error_step_{step:03d}.json'),
                            'w',
                            encoding='utf-8',
                        ) as _f:
                            json.dump(
                                {
                                    'step': int(step),
                                    'error': str(exc),
                                    'input_state': (
                                        'rollout_state' if bool(cfg.get('enable_perturb', True)) else 'original_state'
                                    ),
                                    'recover_eval_video': recover_eval_video,
                                },
                                _f,
                                indent=2,
                                ensure_ascii=False,
                            )
                    except Exception:
                        pass
                if bool(cfg.get('fail_fast_on_error', False)):
                    raise

            if recover_eval_recoverable is False:
                recover_eval_any_unrecoverable = True
                if recover_eval_first_unrecoverable_step is None:
                    recover_eval_first_unrecoverable_step = int(step)
            recover_eval_last = {
                'recoverable': (None if recover_eval_recoverable is None else bool(recover_eval_recoverable)),
                'mode': recover_eval_mode,
                'input_state': ('rollout_state' if bool(cfg.get('enable_perturb', True)) else 'original_state'),
                'metric_name': recover_eval_metric_name,
                'metric': (None if recover_eval_metric is None else float(recover_eval_metric)),
                'threshold': (None if recover_eval_threshold is None else float(recover_eval_threshold)),
                'metrics': recover_eval_metrics,
                'thresholds': recover_eval_thresholds,
                'passes': recover_eval_passes,
                'failed_thresholds': recover_eval_failed_thresholds,
                'horizon': (None if recover_eval_horizon is None else int(recover_eval_horizon)),
                'gt_ref_idx': (None if recover_eval_gt_ref_idx is None else int(recover_eval_gt_ref_idx)),
                'error': recover_eval_error,
            }

        # collect debug info for this rollout step
        if debug_dir is not None:
            _dbg_rollout.append({
                'step': step, 't_star': int(t_star), 'min_dist': float(min_dist),
                'progress': float(progress), 'phase_key': phase_key,
                'active_left_arm': bool(active_info.get('left_arm', True)),
                'active_right_arm': bool(active_info.get('right_arm', True)),
                'active_left_gripper': bool(active_info.get('left_gripper', True)),
                'active_right_gripper': bool(active_info.get('right_gripper', True)),
                'fk_left_pos': lp.tolist(), 'fk_right_pos': rp.tolist(),
                'left_grip': float(left_grip), 'right_grip': float(right_grip),
                'action_perturbed': bool(perturbed),
                'perturb_mode': pert_mode,
                'sampled_error_mode': sampled_error_mode,
                'perturb_dir_bin_id': (
                    None if perturb_dir_bin_id is None else int(perturb_dir_bin_id)
                ),
                'perturb_mag_bin_id': (
                    None if perturb_mag_bin_id is None else int(perturb_mag_bin_id)
                ),
                'perturb_axis_bin_id': (
                    None if perturb_axis_bin_id is None else int(perturb_axis_bin_id)
                ),
                'perturb_translation_gain_m': (
                    None if perturb_translation_gain_m is None else float(perturb_translation_gain_m)
                ),
                'perturb_rotation_deg': (
                    None if perturb_rotation_deg is None else float(perturb_rotation_deg)
                ),
                'perturb_dir_left': perturb_dir_left,
                'perturb_dir_right': perturb_dir_right,
                'perturb_axis_left': perturb_axis_left,
                'perturb_axis_right': perturb_axis_right,
                'rollout_exec_len': int(len(act_raw)),
                'recover_eval_enable': bool(recover_eval_enable),
                'recover_eval_mode': recover_eval_mode,
                'recover_eval_recoverable': (
                    None if recover_eval_recoverable is None else bool(recover_eval_recoverable)
                ),
                'recover_eval_metric_name': recover_eval_metric_name,
                'recover_eval_metric': (
                    None if recover_eval_metric is None else float(recover_eval_metric)
                ),
                'recover_eval_threshold': (
                    None if recover_eval_threshold is None else float(recover_eval_threshold)
                ),
                'recover_eval_metrics': recover_eval_metrics,
                'recover_eval_thresholds': recover_eval_thresholds,
                'recover_eval_passes': recover_eval_passes,
                'recover_eval_failed_thresholds': recover_eval_failed_thresholds,
                'recover_eval_horizon': (
                    None if recover_eval_horizon is None else int(recover_eval_horizon)
                ),
                'recover_eval_gt_ref_idx': (
                    None if recover_eval_gt_ref_idx is None else int(recover_eval_gt_ref_idx)
                ),
                'recover_eval_nearest_dist': (
                    None if recover_eval_nearest_dist is None else float(recover_eval_nearest_dist)
                ),
                'recover_eval_window_start': (
                    None if recover_eval_window_start is None else int(recover_eval_window_start)
                ),
                'recover_eval_window_end': (
                    None if recover_eval_window_end is None else int(recover_eval_window_end)
                ),
                'recover_eval_error': recover_eval_error,
                'recover_eval_video': recover_eval_video,
                'recover_eval_compare': recover_eval_compare,
                'evac_blur_filter': evac_blur_filter,
            })

    force_generate_correction = False
    if correction_force_generate:
        force_generate_correction = True
    if (
        not force_generate_correction
        and bool(cfg.get("correction_generate_on_unrecoverable", False))
        and recover_eval_last.get("recoverable") is False
    ):
        force_generate_correction = True
    if debug_dir is not None:
        _dbg_rollout.append({
            'step': int(max_steps_eff),
            'reason': (
                'correction_force_generate'
                if correction_force_generate
                else 'correction_generate_on_unrecoverable'
            ),
            'max_rollout_steps': int(max_steps),
        })
    perturb_action_prefix_raw = (
        np.concatenate(perturb_action_prefix_chunks, axis=0).astype(np.float32)
        if len(perturb_action_prefix_chunks) > 0
        else None
    )
    perturb_start_qpos_raw = np.asarray(qpos_raw, dtype=np.float32).copy()
    perturb_final_qpos_raw = np.asarray(curr_qpos_raw, dtype=np.float32).copy()
    perturb_delta_linf = (
        None
        if perturb_action_prefix_raw is None
        else float(np.max(np.abs(perturb_final_qpos_raw - perturb_start_qpos_raw)))
    )
    perturb_delta_l2 = (
        None
        if perturb_action_prefix_raw is None
        else float(np.linalg.norm(perturb_final_qpos_raw - perturb_start_qpos_raw))
    )
    compare_sim_artifact = _write_compare_sim_artifact(
        cfg,
        debug_dir,
        raw_data,
        int(start_ts),
        sampled_unit=sampled_unit,
        perturb_action_prefix_raw=perturb_action_prefix_raw,
        perturb_start_qpos_raw=perturb_start_qpos_raw,
        perturb_final_qpos_raw=perturb_final_qpos_raw,
    )
    perturb_compare_meta = None
    if debug_dir is not None:
        _dbg_corr_cmp = os.path.join(debug_dir, 'correction')
        os.makedirs(_dbg_corr_cmp, exist_ok=True)
        _rr_last = None
        for _rr in reversed(_dbg_rollout):
            if bool(_rr.get('action_perturbed', False)):
                _rr_last = _rr
                break
        if _rr_last is None and len(_dbg_rollout) > 0:
            _rr_last = _dbg_rollout[-1]
        perturb_compare_meta = save_perturb_compare_image(
            _dbg_corr_cmp,
            image_data_s[0],
            curr_image[0],
            sampled_unit,
            rollout_last_record=_rr_last,
        )
    # IMPORTANT: correction planning must start from post-rollout state.
    # `left_q/right_q` inside rollout loop are sampled before executing act_raw,
    # so refresh them from `curr_qpos_raw` here.
    left_q = np.asarray(curr_qpos_raw[0:6], dtype=np.float32)
    right_q = np.asarray(curr_qpos_raw[7:13], dtype=np.float32)
    left_grip = float(curr_qpos_raw[6])
    right_grip = float(curr_qpos_raw[13])

    nearest_mode = ""
    if isinstance(dyn_state, dict):
        nearest_mode = str(dyn_state.get("error_mode", "")).strip().lower()

    corr_left_active = bool(active_info_fixed['left_arm'])
    corr_right_active = bool(active_info_fixed['right_arm'])

    if not force_generate_correction:
        # debug: save rollout info even on skip
        if debug_dir is not None:
            _dbg_corr = os.path.join(debug_dir, 'correction')
            os.makedirs(_dbg_corr, exist_ok=True)
            with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                _skip_reason = (
                    'correction_not_forced'
                )
                def _fmt_vec3(v):
                    if v is None:
                        return "None"
                    arr = np.asarray(v, dtype=np.float32).reshape(-1)
                    if arr.size < 3:
                        return "None"
                    return f"({arr[0]:+.3f}, {arr[1]:+.3f}, {arr[2]:+.3f})"

                def _mk_perturb_nl(_r):
                    mode = _r.get('sampled_error_mode')
                    if mode is None:
                        return "未施加扰动（sampled_error_mode=None）"
                    mode = str(mode)
                    if mode == 'translation':
                        _gain = _r.get('perturb_translation_gain_m')
                        _gain_s = "None" if _gain is None else f"{float(_gain):.3f}"
                        return (
                            f"平移扰动: gain={_gain_s} m, "
                            f"left_dir={_fmt_vec3(_r.get('perturb_dir_left'))}, "
                            f"right_dir={_fmt_vec3(_r.get('perturb_dir_right'))}"
                        )
                    if mode == 'rotation':
                        _angle = _r.get('perturb_rotation_deg')
                        _angle_s = "None" if _angle is None else f"{float(_angle):.3f}"
                        return (
                            f"旋转扰动: angle={_angle_s} deg, "
                            f"left_axis={_fmt_vec3(_r.get('perturb_axis_left'))}, "
                            f"right_axis={_fmt_vec3(_r.get('perturb_axis_right'))}"
                        )
                    if mode == 'gripper_close':
                        return "夹爪闭合扰动（arm 保持当前关节，夹爪向 0 收拢）"
                    return f"扰动模式={mode}"

                _rollout_skip = []
                for _r in _dbg_rollout:
                    _rr = dict(_r)
                    _rr.pop('perturb_dir_bin_id', None)
                    _rr.pop('perturb_mag_bin_id', None)
                    _rr.pop('perturb_axis_bin_id', None)
                    _rr['perturb_nl'] = _mk_perturb_nl(_rr)
                    _rollout_skip.append(_rr)
                gt_projection_meta = save_gt_projection_on_original(
                    _dbg_corr,
                    image_data_s,
                    raw_data,
                    start_ts,
                    phase_window_len,
                    phase_bins,
                )
                json.dump({
                    'reason': _skip_reason,
                    'sampled_unit': sampled_unit,
                    'perturb_compare': perturb_compare_meta,
                    'min_dist': float(min_dist),
                    'recover_eval_enable': bool(recover_eval_enable),
                    'recover_eval_any_unrecoverable': bool(recover_eval_any_unrecoverable),
                    'recover_eval_first_unrecoverable_step': (
                        None if recover_eval_first_unrecoverable_step is None else int(recover_eval_first_unrecoverable_step)
                    ),
                    'recover_eval_last': recover_eval_last,
                    'gt_projection_on_original': gt_projection_meta,
                    'world_model_compare': world_model_compare_records,
                    'world_model_compare_sim': compare_sim_artifact,
                    'rollout': _rollout_skip
                }, _f, indent=2)
        if nearest_mode == "gripper_close":
            correction_branch = "gripper_close"
        elif nearest_mode == "translation":
            correction_branch = "translation"
        elif nearest_mode == "rotation":
            correction_branch = "rotation"
        else:
            correction_branch = "unsupported"

        sampled_error_mode = None
        if isinstance(dyn_state, dict):
            sampled_error_mode = dyn_state.get("error_mode")
        error_action_prefix_raw = None if act_raw is None else np.asarray(act_raw, dtype=np.float32).copy()
        corr_meta = {
            "correction_generated": False,
            "closed_loop_fallback_used": False,
            "correction_branch": correction_branch,
            "sampled_error_mode": sampled_error_mode,
            "forced_error_mode": forced_error_mode_key,
            "sampled_phase_key": phase_key_fixed,
            "sampled_phase_bin_id": phase_bin_fixed,
            "sampled_phase_instance_idx": phase_instance_fixed,
            "sampled_active_arm_pattern": sampled_active_arm_pattern_key,
            "forced_dir_bin_id": forced_dir_bin_fixed,
            "forced_mag_bin_id": forced_mag_bin_fixed,
            "nearest_mode": nearest_mode,
            "rollout_exec_steps": int(rollout_exec_steps),
            "t_star": int(t_star),
            "min_dist": float(min_dist),
            "recover_eval_enable": bool(recover_eval_enable),
            "recover_eval_any_unrecoverable": bool(recover_eval_any_unrecoverable),
            "recover_eval_first_unrecoverable_step": (
                None if recover_eval_first_unrecoverable_step is None else int(recover_eval_first_unrecoverable_step)
            ),
            "recover_eval_last": recover_eval_last,
            "world_model_backend": world_model_backend,
            "evac_rollout_videos": evac_rollout_videos,
            "world_model_rollout_videos": world_model_rollout_videos,
            "world_model_compare": world_model_compare_records,
            "world_model_compare_sim": compare_sim_artifact,
            "debug_dir": debug_dir,
            "error_action_prefix_raw": error_action_prefix_raw,
            "perturb_action_prefix_raw": perturb_action_prefix_raw,
            "perturb_action_prefix_len": (
                0 if perturb_action_prefix_raw is None else int(perturb_action_prefix_raw.shape[0])
            ),
            "perturb_start_qpos_raw": perturb_start_qpos_raw,
            "perturb_final_qpos_raw": perturb_final_qpos_raw,
            "perturb_delta_linf": perturb_delta_linf,
            "perturb_delta_l2": perturb_delta_l2,
        }
        return (None, None, None, None, corr_meta)

    gt_left_arm = raw_data.get('gt_left_arm', None)
    gt_right_arm = raw_data.get('gt_right_arm', None)

    res_l = {'status': 'SkippedInactive', 'position': np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], 1, axis=0)}
    res_r = {'status': 'SkippedInactive', 'position': np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], 1, axis=0)}
    gripper_close_suffix_len_dbg = None
    gripper_close_uniformized_points_dbg = None
    gripper_close_uniformized_spans_dbg = None
    def _take_tail(gt_seq, start_idx, n, fallback):
        if n <= 0:
            return np.zeros((0,) + np.asarray(fallback).shape, dtype=np.float32)
        if gt_seq is None:
            return np.repeat(np.asarray(fallback, dtype=np.float32)[None, ...], n, axis=0)
        g = np.asarray(gt_seq, dtype=np.float32)
        if g.ndim == 1:
            g = g[:, None]
        s = int(np.clip(start_idx, 0, max(0, g.shape[0])))
        tail = g[s:s + n]
        if tail.shape[0] >= n:
            return tail.astype(np.float32)
        last = np.asarray(fallback, dtype=np.float32)
        if tail.shape[0] > 0:
            last = tail[-1]
        pad = np.repeat(last[None, ...], n - tail.shape[0], axis=0).astype(np.float32)
        if tail.shape[0] == 0:
            return pad
        return np.concatenate([tail.astype(np.float32), pad], axis=0).astype(np.float32)

    def _fit_prefix(path_future, curr_q, n):
        p = np.asarray(path_future, dtype=np.float32)
        if p.ndim != 2 or p.shape[0] == 0:
            return np.repeat(np.asarray(curr_q, dtype=np.float32)[None, :], n, axis=0)
        if p.shape[0] == n:
            return p.astype(np.float32)
        if p.shape[0] > n:
            return resample_trajectory(p, n).astype(np.float32)
        pad = np.repeat(p[-1:].astype(np.float32), n - p.shape[0], axis=0)
        return np.concatenate([p.astype(np.float32), pad], axis=0).astype(np.float32)

    # Build correction only on active sides; inactive sides stay at current state.
    lt = np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], chunk_size, axis=0)
    rt = np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], chunk_size, axis=0)
    lg = np.full((chunk_size,), float(left_grip), dtype=np.float32)
    rg = np.full((chunk_size,), float(right_grip), dtype=np.float32)
    tl_grip = float(left_grip)
    tr_grip = float(right_grip)

    if nearest_mode == "gripper_close":
        # Dedicated correction for early-close error:
        # 1) quickly reopen gripper, 2) follow GT forward.
        prefix_len = _correction_prefix_len(cfg, rollout_exec_steps, chunk_size)
        # For gripper_close correction, compose tail directly from sampled start_ts.
        # Do not shift by rollout_steps_total.
        follow_idx = 0
        follow_abs = int(start_ts)
        suffix_len = int(max(0, chunk_size - prefix_len))

        l_tgt_q = np.asarray(left_q, dtype=np.float32) if gt_left_arm is None else np.asarray(gt_left_arm[follow_abs], dtype=np.float32)
        r_tgt_q = np.asarray(right_q, dtype=np.float32) if gt_right_arm is None else np.asarray(gt_right_arm[follow_abs], dtype=np.float32)

        tl_grip = float(np.clip(left_grip_traj[follow_idx], 0.0, 1.0))
        tr_grip = float(np.clip(right_grip_traj[follow_idx], 0.0, 1.0))
        # For gripper_close recovery correction, force full open.
        open_tgt = 1.0
        open_prefix_len = int(prefix_len)

        def _build_grip_prefix(curr_g, gt_g):
            curr_g = float(np.clip(curr_g, 0.0, 1.0))
            gt_g = float(np.clip(gt_g, 0.0, 1.0))
            g_open = float(max(curr_g, open_tgt))
            if open_prefix_len <= 1:
                return np.array([g_open], dtype=np.float32)
            return np.linspace(curr_g, g_open, open_prefix_len, dtype=np.float32)

        use_left_gt_suffix = bool(corr_left_active or original_both_active)
        use_right_gt_suffix = bool(corr_right_active or original_both_active)

        if corr_left_active:
            l_prefix = np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], prefix_len, axis=0)
            l_grip_prefix = _build_grip_prefix(left_grip, tl_grip)
            lt[:prefix_len] = l_prefix.astype(np.float32)
            lg[:prefix_len] = l_grip_prefix.astype(np.float32)
        if use_left_gt_suffix and suffix_len > 0:
            l_tail = _take_tail(gt_left_arm, follow_abs, suffix_len, l_tgt_q)
            l_grip_tail = _take_tail(left_grip_traj, follow_idx + 1, suffix_len, np.array([tl_grip], dtype=np.float32))[:, 0]
            lt[prefix_len:prefix_len + suffix_len] = l_tail.astype(np.float32)
            lg[prefix_len:prefix_len + suffix_len] = l_grip_tail.astype(np.float32)

        if corr_right_active:
            r_prefix = np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], prefix_len, axis=0)
            r_grip_prefix = _build_grip_prefix(right_grip, tr_grip)
            rt[:prefix_len] = r_prefix.astype(np.float32)
            rg[:prefix_len] = r_grip_prefix.astype(np.float32)
        if use_right_gt_suffix and suffix_len > 0:
            r_tail = _take_tail(gt_right_arm, follow_abs, suffix_len, r_tgt_q)
            r_grip_tail = _take_tail(right_grip_traj, follow_idx + 1, suffix_len, np.array([tr_grip], dtype=np.float32))[:, 0]
            rt[prefix_len:prefix_len + suffix_len] = r_tail.astype(np.float32)
            rg[prefix_len:prefix_len + suffix_len] = r_grip_tail.astype(np.float32)
        gripper_close_suffix_len_dbg = int(suffix_len)
        # Temporarily disable tail uniformization: keep GT tail as-is.
        gripper_close_uniformized_points_dbg = 0
        gripper_close_uniformized_spans_dbg = 0

        if corr_left_active:
            res_l = {'status': 'BypassGripperCloseRecover', 'position': np.stack([np.asarray(left_q, dtype=np.float32), l_tgt_q], axis=0)}
        if corr_right_active:
            res_r = {'status': 'BypassGripperCloseRecover', 'position': np.stack([np.asarray(right_q, dtype=np.float32), r_tgt_q], axis=0)}
    elif nearest_mode in {"translation", "rotation"}:
        # Dedicated correction for translation/rotation errors:
        # 1) use rollout_exec_steps actions to pull current perturbed state
        #    back to the original GT pose at sampled start_ts,
        # 2) then follow GT from start_ts forward to fill the rest of chunk_size.
        import sapien
        _recover_mode = str(nearest_mode)
        prefix_len = _correction_prefix_len(cfg, rollout_exec_steps, chunk_size)
        follow_idx = 0
        follow_abs = int(start_ts)
        suffix_len = int(max(0, chunk_size - prefix_len))

        qpos_full_tr = np.zeros(len(fk.jnames), dtype=np.float32)
        qpos_full_tr[fk.fl_idx] = left_q
        qpos_full_tr[fk.fr_idx] = right_q

        target_lp_tr = sapien.Pose(left_ep[follow_idx, :3], left_ep[follow_idx, 3:7])
        target_rp_tr = sapien.Pose(right_ep[follow_idx, :3], right_ep[follow_idx, 3:7])

        l_prefix = np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], prefix_len, axis=0)
        r_prefix = np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], prefix_len, axis=0)

        res_l = {'status': 'SkippedInactive', 'position': np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], 1, axis=0)}
        res_r = {'status': 'SkippedInactive', 'position': np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], 1, axis=0)}

        if corr_left_active:
            try:
                res_l = planner_l.plan_path(qpos_full_tr, target_lp_tr, arms_tag='left')
            except Exception as exc:
                if debug_dir is not None:
                    _dbg_corr = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr, exist_ok=True)
                    with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                        json.dump({
                            'reason': f'{_recover_mode}_recover_planner_exception_left',
                            'sampled_unit': sampled_unit,
                            'error': str(exc),
                            't_star': int(t_star),
                            'min_dist': float(min_dist),
                            'rollout': _dbg_rollout,
                        }, _f, indent=2)
                return None
            if res_l.get('status') != 'Success':
                if debug_dir is not None:
                    _dbg_corr = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr, exist_ok=True)
                    with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                        json.dump({
                            'reason': f'{_recover_mode}_recover_planner_status_fail_left',
                            'sampled_unit': sampled_unit,
                            'planner_left_status': res_l.get('status'),
                            't_star': int(t_star),
                            'min_dist': float(min_dist),
                            'rollout': _dbg_rollout,
                        }, _f, indent=2)
                return None
            l_path = np.asarray(res_l['position'], dtype=np.float32)
            l_future = l_path[1:] if l_path.shape[0] > 1 else l_path
            l_prefix = _fit_prefix(l_future, left_q, prefix_len)

        if corr_right_active:
            try:
                res_r = planner_r.plan_path(qpos_full_tr, target_rp_tr, arms_tag='right')
            except Exception as exc:
                if debug_dir is not None:
                    _dbg_corr = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr, exist_ok=True)
                    with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                        json.dump({
                            'reason': f'{_recover_mode}_recover_planner_exception_right',
                            'sampled_unit': sampled_unit,
                            'error': str(exc),
                            't_star': int(t_star),
                            'min_dist': float(min_dist),
                            'rollout': _dbg_rollout,
                        }, _f, indent=2)
                return None
            if res_r.get('status') != 'Success':
                if debug_dir is not None:
                    _dbg_corr = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr, exist_ok=True)
                    with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                        json.dump({
                            'reason': f'{_recover_mode}_recover_planner_status_fail_right',
                            'sampled_unit': sampled_unit,
                            'planner_right_status': res_r.get('status'),
                            't_star': int(t_star),
                            'min_dist': float(min_dist),
                            'rollout': _dbg_rollout,
                        }, _f, indent=2)
                return None
            r_path = np.asarray(res_r['position'], dtype=np.float32)
            r_future = r_path[1:] if r_path.shape[0] > 1 else r_path
            r_prefix = _fit_prefix(r_future, right_q, prefix_len)

        l_tgt_q = np.asarray(left_q, dtype=np.float32) if gt_left_arm is None else np.asarray(gt_left_arm[follow_abs], dtype=np.float32)
        r_tgt_q = np.asarray(right_q, dtype=np.float32) if gt_right_arm is None else np.asarray(gt_right_arm[follow_abs], dtype=np.float32)

        tl_grip = float(np.clip(left_grip_traj[follow_idx], 0.0, 1.0))
        tr_grip = float(np.clip(right_grip_traj[follow_idx], 0.0, 1.0))
        use_left_gt_suffix = bool(corr_left_active or original_both_active)
        use_right_gt_suffix = bool(corr_right_active or original_both_active)
        if corr_left_active:
            if prefix_len <= 1:
                l_grip_prefix = np.array([tl_grip], dtype=np.float32)
            else:
                l_grip_prefix = np.linspace(float(left_grip), tl_grip, prefix_len, dtype=np.float32)
            lt[:prefix_len] = l_prefix.astype(np.float32)
            lg[:prefix_len] = l_grip_prefix.astype(np.float32)
        if use_left_gt_suffix and suffix_len > 0:
            l_tail = _take_tail(gt_left_arm, follow_abs, suffix_len, l_tgt_q)
            l_grip_tail = _take_tail(left_grip_traj, follow_idx, suffix_len, np.array([tl_grip], dtype=np.float32))[:, 0]
            lt[prefix_len:prefix_len + suffix_len] = l_tail.astype(np.float32)
            lg[prefix_len:prefix_len + suffix_len] = l_grip_tail.astype(np.float32)

        if corr_right_active:
            if prefix_len <= 1:
                r_grip_prefix = np.array([tr_grip], dtype=np.float32)
            else:
                r_grip_prefix = np.linspace(float(right_grip), tr_grip, prefix_len, dtype=np.float32)
            rt[:prefix_len] = r_prefix.astype(np.float32)
            rg[:prefix_len] = r_grip_prefix.astype(np.float32)
        if use_right_gt_suffix and suffix_len > 0:
            r_tail = _take_tail(gt_right_arm, follow_abs, suffix_len, r_tgt_q)
            r_grip_tail = _take_tail(right_grip_traj, follow_idx, suffix_len, np.array([tr_grip], dtype=np.float32))[:, 0]
            rt[prefix_len:prefix_len + suffix_len] = r_tail.astype(np.float32)
            rg[prefix_len:prefix_len + suffix_len] = r_grip_tail.astype(np.float32)
    else:
        if debug_dir is not None:
            _dbg_corr = os.path.join(debug_dir, 'correction')
            os.makedirs(_dbg_corr, exist_ok=True)
            with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                json.dump({
                    'reason': 'unsupported_nearest_mode',
                    'sampled_unit': sampled_unit,
                    'nearest_mode': str(nearest_mode),
                    't_star': int(t_star),
                    'min_dist': float(min_dist),
                    'rollout': _dbg_rollout,
                }, _f, indent=2)
        return None
    corr = np.zeros((chunk_size, 14), dtype=np.float32)
    corr[:, 0:6] = lt
    corr[:, 7:13] = rt
    corr[:, 6] = np.clip(lg, 0.0, 1.0)
    corr[:, 13] = np.clip(rg, 0.0, 1.0)
    corr_norm = (corr - norm_stats['action_mean']) / norm_stats['action_std']

    padded = np.zeros((max_action_len, 14), dtype=np.float32)
    padded[:chunk_size] = corr_norm
    is_pad = np.ones(max_action_len, dtype=bool)
    is_pad[:chunk_size] = False

    qn = (curr_qpos_raw - norm_stats['qpos_mean']) / norm_stats['qpos_std']
    compare_sim_error_prefix_raw = corr[
        : int(np.clip(int(rollout_exec_steps), 1, int(corr.shape[0])))
    ].astype(np.float32).copy()
    compare_sim_artifact = _write_compare_sim_artifact(
        cfg,
        debug_dir,
        raw_data,
        int(start_ts),
        sampled_unit=sampled_unit,
        perturb_action_prefix_raw=perturb_action_prefix_raw,
        perturb_start_qpos_raw=perturb_start_qpos_raw,
        perturb_final_qpos_raw=perturb_final_qpos_raw,
        corr_action_chunk_raw=corr,
        corr_qpos_raw=curr_qpos_raw,
        error_action_prefix_raw=compare_sim_error_prefix_raw,
    )

    # --- debug: save correction summary jsons ---
    evac_corr_video = None
    if debug_dir is not None:
        import cv2
        _dbg_corr = os.path.join(debug_dir, 'correction')
        os.makedirs(_dbg_corr, exist_ok=True)
        if bool(cfg.get('save_correction_debug', False)):
            try:
                fk_corr_poses = []
                grip_corr_list = []
                fr_start = fk.forward(curr_qpos_raw[0:6], curr_qpos_raw[7:13])
                fk_corr_poses.append((
                    fr_start['left'][0].copy(), fr_start['left'][1].copy(),
                    fr_start['right'][0].copy(), fr_start['right'][1].copy(),
                ))
                grip_corr_list.append((float(curr_qpos_raw[6]), float(curr_qpos_raw[13])))
                for _ai in range(corr.shape[0]):
                    _ar = corr[_ai]
                    _fr = fk.forward(_ar[0:6], _ar[7:13])
                    fk_corr_poses.append((
                        _fr['left'][0].copy(), _fr['left'][1].copy(),
                        _fr['right'][0].copy(), _fr['right'][1].copy(),
                    ))
                    grip_corr_list.append((float(_ar[6]), float(_ar[13])))
                world_model_corr_dir = os.path.join(
                    debug_dir, f'{world_model_backend}_correction', 'rollout_step_000'
                )
                _world_model_inference(
                    modules,
                    cfg,
                    curr_image[0],
                    fk_corr_poses,
                    grip_corr_list,
                    raw_data,
                    device,
                    save_dir=world_model_corr_dir,
                )
                _vpath_corr = os.path.join(world_model_corr_dir, 'outputs.mp4')
                _mpath_corr = os.path.join(world_model_corr_dir, world_model_meta_filename)
                evac_corr_video = {
                    'path': _debug_relpath(_vpath_corr),
                    'exists': bool(os.path.exists(_vpath_corr)),
                    'runtime_meta_path': _debug_relpath(_mpath_corr),
                    'runtime_meta_exists': bool(os.path.exists(_mpath_corr)),
                    'backend': world_model_backend,
                    'num_actions': int(corr.shape[0]),
                }
            except Exception as _exc_corr_video:
                evac_corr_video = {
                    'path': None,
                    'exists': False,
                    'error': str(_exc_corr_video),
                    'num_actions': int(corr.shape[0]),
                }
                if bool(cfg.get('fail_fast_on_error', False)):
                    raise
        # Save correction projection overlays on original/corrected images.
        try:
            _cimg = _tensor_chw_to_bgr_u8(curr_image[0])
            _oimg = _tensor_chw_to_bgr_u8(image_data_s[0])
            gt_ref_idx = int(np.clip(start_ts, 0, raw_data['left_endpose'].shape[0] - 1))

            if _oimg.shape[:2] != _cimg.shape[:2]:
                _oimg = cv2.resize(_oimg, (_cimg.shape[1], _cimg.shape[0]), interpolation=cv2.INTER_LINEAR)

            _overlay_o = _oimg.copy()
            _overlay_c = _cimg.copy()
            _overlay_gt = _oimg.copy()

            from evac.lvdm.models.ddpm3d import ACWMLatentDiffusion
            import evac.lvdm.models.ddpm3d as ddpm3d_mod

            K = raw_data['intrinsic_cv'].astype(np.float32).copy()
            E = np.eye(4, dtype=np.float32)
            E[:3, :] = raw_data['extrinsic_cv'].astype(np.float32)

            h_native, w_native = raw_data.get('native_resolution', (_oimg.shape[0], _oimg.shape[1]))
            h_img, w_img = _oimg.shape[:2]
            if w_native > 0 and h_native > 0 and (w_native != w_img or h_native != h_img):
                sx = float(w_img) / float(w_native)
                sy = float(h_img) / float(h_native)
                K[0, 0] *= sx
                K[0, 2] *= sx
                K[1, 1] *= sy
                K[1, 2] *= sy
            c2w = np.linalg.inv(E).astype(np.float32)

            fk_seq = [fk.forward(corr[i, 0:6], corr[i, 7:13]) for i in range(corr.shape[0])]
            pose_list = []
            for ai, fr in enumerate(fk_seq):
                lp_i, lq_wxyz = fr['left']
                rp_i, rq_wxyz = fr['right']
                lq_xyzw = np.array([lq_wxyz[1], lq_wxyz[2], lq_wxyz[3], lq_wxyz[0]], dtype=np.float32)
                rq_xyzw = np.array([rq_wxyz[1], rq_wxyz[2], rq_wxyz[3], rq_wxyz[0]], dtype=np.float32)
                if lq_xyzw[3] < 0:
                    lq_xyzw = -lq_xyzw
                if rq_xyzw[3] < 0:
                    rq_xyzw = -rq_xyzw
                lg_i = float(np.clip(corr[ai, 6], 0.0, 1.0)) * 120.0
                rg_i = float(np.clip(corr[ai, 13], 0.0, 1.0)) * 120.0
                pose_list.append(
                    np.concatenate([lp_i, lq_xyzw, [lg_i], rp_i, rq_xyzw, [rg_i]], axis=0).astype(np.float32)
                )
            pose_np = np.stack(pose_list, axis=0)

            def _render_endpoint_mask(pose_arr):
                _orig_eef_pts = ddpm3d_mod.EndEffectorPts
                ddpm3d_mod.EndEffectorPts = [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.04, 0.0, 0.0, 1.0],
                    [0.0, 0.04, 0.0, 1.0],
                    [0.0, 0.0, 0.04, 1.0],
                ]
                try:
                    traj_tensor = ACWMLatentDiffusion.get_traj(
                        None,
                        (h_img, w_img),
                        pose_arr,
                        E[None, ...],
                        c2w[None, ...],
                        torch.from_numpy(K).float().unsqueeze(0),
                        radius=20,
                    )
                finally:
                    ddpm3d_mod.EndEffectorPts = _orig_eef_pts
                traj_np = traj_tensor.detach().cpu().numpy()[:, 0]
                traj_np = np.transpose(traj_np, (1, 2, 3, 0))
                t_len = traj_np.shape[0]
                idx_end = max(0, t_len - 1)
                traj_endpoints = np.maximum(traj_np[0], traj_np[idx_end])
                traj_u8 = np.clip(traj_endpoints * 255.0, 0.0, 255.0).astype(np.uint8)
                mask = np.any(np.abs(traj_u8.astype(np.int16) - 50) > 2, axis=2)
                return traj_u8, mask

            def _project_base_uv_from_pose_np(pose_arr):
                w2c_t = torch.from_numpy(E).float().unsqueeze(0).unsqueeze(0)
                intrinsic_t = torch.from_numpy(K).float().unsqueeze(0).unsqueeze(0)
                cvt_matrix = torch.tensor(ddpm3d_mod.Gripper2EEFCvt, dtype=torch.float32).view(1, 1, 4, 4)
                ee_key_pts = torch.tensor(ddpm3d_mod.EndEffectorPts, dtype=torch.float32).view(1, 1, 4, 4).permute(0, 1, 3, 2)

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

            def _extract_key_uvs(uvs, pts):
                pts_uv = []
                if getattr(uvs, "ndim", 0) != 3:
                    return pts_uv
                n_key = int(uvs.shape[1])
                for j in range(n_key):
                    try:
                        z = float(pts[0, 0, 2, j].item())
                        u = int(uvs[0, j, 0])
                        v = int(uvs[0, j, 1])
                    except Exception:
                        pts_uv.append(None)
                        continue
                    if z > 1e-6 and (0 <= u < w_img) and (0 <= v < h_img):
                        pts_uv.append((u, v))
                    else:
                        pts_uv.append(None)
                return pts_uv

            def _pack_pose_row(lp_i, lq_wxyz, lg_01, rp_i, rq_wxyz, rg_01):
                lq_xyzw = np.array([lq_wxyz[1], lq_wxyz[2], lq_wxyz[3], lq_wxyz[0]], dtype=np.float32)
                rq_xyzw = np.array([rq_wxyz[1], rq_wxyz[2], rq_wxyz[3], rq_wxyz[0]], dtype=np.float32)
                if lq_xyzw[3] < 0:
                    lq_xyzw = -lq_xyzw
                if rq_xyzw[3] < 0:
                    rq_xyzw = -rq_xyzw
                lg = float(np.clip(lg_01, 0.0, 1.0)) * 120.0
                rg = float(np.clip(rg_01, 0.0, 1.0)) * 120.0
                return np.concatenate([lp_i, lq_xyzw, [lg], rp_i, rq_xyzw, [rg]], axis=0).astype(np.float32)

            def _draw_pose_axes(img, uv_pts, label, label_color, thickness=2):
                if uv_pts is None or len(uv_pts) < 1:
                    return
                base = uv_pts[0]
                if base is None:
                    return
                axis_cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x,y,z (BGR)
                bx, by = int(base[0]), int(base[1])
                cv2.circle(img, (bx, by), 4, label_color, -1, cv2.LINE_AA)
                cv2.putText(
                    img, label, (bx + 6, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA
                )
                for ai in range(1, min(4, len(uv_pts))):
                    pt = uv_pts[ai]
                    if pt is None:
                        continue
                    px, py = int(pt[0]), int(pt[1])
                    cv2.line(img, (bx, by), (px, py), axis_cols[ai - 1], thickness, cv2.LINE_AA)
                    cv2.circle(img, (px, py), 2, axis_cols[ai - 1], -1, cv2.LINE_AA)

            traj_u8, mask = _render_endpoint_mask(pose_np)
            uvs_l, uvs_r, pts_l, pts_r = _project_base_uv_from_pose_np(pose_np)
            luv = _extract_base_uv(uvs_l, pts_l.reshape(1, pts_l.shape[1], 4, 4))
            ruv = _extract_base_uv(uvs_r, pts_r.reshape(1, pts_r.shape[1], 4, 4))

            def _draw_polyline(img, seq, color):
                prev = None
                for pt in seq:
                    if pt is None:
                        prev = None
                        continue
                    if prev is not None:
                        cv2.line(img, (int(prev[0]), int(prev[1])), (int(pt[0]), int(pt[1])), color, 2, cv2.LINE_AA)
                    prev = pt
                if len(seq) > 0 and seq[0] is not None:
                    cv2.circle(img, (int(seq[0][0]), int(seq[0][1])), 4, color, -1, cv2.LINE_AA)
                if len(seq) > 0 and seq[-1] is not None:
                    cv2.circle(img, (int(seq[-1][0]), int(seq[-1][1])), 4, color, -1, cv2.LINE_AA)

            def _annotate_start_end(img, seq, prefix, color):
                if len(seq) == 0:
                    return
                s = seq[0]
                e = seq[-1]
                if s is not None:
                    cv2.putText(
                        img, f"{prefix}-S", (int(s[0]) + 6, int(s[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
                    )
                if e is not None:
                    cv2.putText(
                        img, f"{prefix}-E", (int(e[0]) + 6, int(e[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
                    )

            def _draw_legend(img):
                x0, y0 = 10, 10
                rows = [("L: green", (0, 255, 0)), ("R: red", (0, 0, 255))]
                row_h = 18
                w = 120
                h = 8 + row_h * len(rows) + 8
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1, cv2.LINE_AA)
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (220, 220, 220), 1, cv2.LINE_AA)
                for i, (name, color) in enumerate(rows):
                    y = y0 + 18 + i * row_h
                    cv2.circle(img, (x0 + 10, y - 4), 4, color, -1, cv2.LINE_AA)
                    cv2.putText(img, name, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

            def _phase_color(phase_key):
                k = str(phase_key).strip().lower()
                cmap = {
                    "approach": (255, 120, 0),   # blue-ish (BGR)
                    "pregrasp": (0, 165, 255),   # orange
                    "transport": (0, 255, 255),  # yellow
                    "place": (255, 0, 255),      # magenta
                }
                return cmap.get(k, (180, 180, 180))

            def _draw_phase_polyline(img, seq, phase_seq, width=1):
                prev = None
                for i, pt in enumerate(seq):
                    if pt is None:
                        prev = None
                        continue
                    if prev is not None:
                        c = _phase_color(phase_seq[i] if i < len(phase_seq) else "unknown")
                        cv2.line(img, (int(prev[0]), int(prev[1])), (int(pt[0]), int(pt[1])), c, width, cv2.LINE_AA)
                    prev = pt

            def _draw_phase_legend(img):
                rows = [
                    ("approach", _phase_color("approach")),
                    ("pregrasp", _phase_color("pregrasp")),
                    ("transport", _phase_color("transport")),
                    ("place", _phase_color("place")),
                ]
                x0, y0 = 10, 62
                row_h = 18
                w = 150
                h = 8 + row_h * len(rows) + 8
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1, cv2.LINE_AA)
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (220, 220, 220), 1, cv2.LINE_AA)
                for i, (name, color) in enumerate(rows):
                    y = y0 + 18 + i * row_h
                    cv2.circle(img, (x0 + 10, y - 4), 4, color, -1, cv2.LINE_AA)
                    cv2.putText(img, name, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

            def _build_pose_np_from_raw_indices(raw_idx_list):
                pose_list = []
                for gi in raw_idx_list:
                    lp_gt = raw_data['left_endpose'][gi, :3].astype(np.float32)
                    lq_gt_wxyz = raw_data['left_endpose'][gi, 3:7].astype(np.float32)
                    rp_gt = raw_data['right_endpose'][gi, :3].astype(np.float32)
                    rq_gt_wxyz = raw_data['right_endpose'][gi, 3:7].astype(np.float32)
                    lq_gt_xyzw = np.array([lq_gt_wxyz[1], lq_gt_wxyz[2], lq_gt_wxyz[3], lq_gt_wxyz[0]], dtype=np.float32)
                    rq_gt_xyzw = np.array([rq_gt_wxyz[1], rq_gt_wxyz[2], rq_gt_wxyz[3], rq_gt_wxyz[0]], dtype=np.float32)
                    if lq_gt_xyzw[3] < 0:
                        lq_gt_xyzw = -lq_gt_xyzw
                    if rq_gt_xyzw[3] < 0:
                        rq_gt_xyzw = -rq_gt_xyzw
                    lg_gt = float(np.clip(raw_data['left_gripper'][gi], 0.0, 1.0)) * 120.0
                    rg_gt = float(np.clip(raw_data['right_gripper'][gi], 0.0, 1.0)) * 120.0
                    pose_list.append(
                        np.concatenate([lp_gt, lq_gt_xyzw, [lg_gt], rp_gt, rq_gt_xyzw, [rg_gt]], axis=0).astype(np.float32)
                    )
                if len(pose_list) == 0:
                    return None
                return np.stack(pose_list, axis=0)

            # Phase-colored GT trajectory projection (shared with skip/debug path).
            p_l_seq, p_r_seq, phase_seq = compute_phase_projection(raw_data, phase_window_len, K, E)
            if p_l_seq is not None and p_r_seq is not None and phase_seq is not None:
                draw_phase_polyline(_overlay_gt, p_l_seq, phase_seq, width=2)
                draw_phase_polyline(_overlay_gt, p_r_seq, phase_seq, width=2)
                draw_phase_polyline(_overlay_c, p_l_seq, phase_seq, width=1)
                draw_phase_polyline(_overlay_c, p_r_seq, phase_seq, width=1)
                draw_phase_bin_starts(_overlay_gt, p_l_seq, p_r_seq, phase_seq, phase_bins)
                draw_phase_bin_starts(_overlay_c, p_l_seq, p_r_seq, phase_seq, phase_bins)
                draw_phase_legend(_overlay_gt)

            # Corrected overlay: keep only correction trajectories + phase-colored GT path.
            _overlay_c[mask] = (0.6 * _overlay_c[mask] + 0.4 * traj_u8[mask]).astype(np.uint8)
            _draw_polyline(_overlay_c, luv, (0, 255, 0))
            _draw_polyline(_overlay_c, ruv, (0, 0, 255))
            _annotate_start_end(_overlay_c, luv, "L", (0, 255, 0))
            _annotate_start_end(_overlay_c, ruv, "R", (0, 0, 255))

            _write_bgr_image(os.path.join(_dbg_corr, 'corr_projection_on_corrected.png'), _overlay_c)
            _write_bgr_image(os.path.join(_dbg_corr, 'gt_projection_on_original.png'), _overlay_gt)
        except Exception:
            try:
                import traceback
                with open(os.path.join(_dbg_corr, 'corr_projection_error.txt'), 'w') as _f:
                    _f.write(traceback.format_exc())
            except Exception:
                pass
            # Best-effort fallback: still dump base original/corrected images.
            try:
                if '_overlay_c' in locals():
                    _write_bgr_image(os.path.join(_dbg_corr, 'corr_projection_on_corrected.png'), _overlay_c)
                elif '_cimg' in locals():
                    _write_bgr_image(os.path.join(_dbg_corr, 'corr_projection_on_corrected.png'), _cimg)
                if '_overlay_gt' in locals():
                    _write_bgr_image(os.path.join(_dbg_corr, 'gt_projection_on_original.png'), _overlay_gt)
                elif '_oimg' in locals():
                    _write_bgr_image(os.path.join(_dbg_corr, 'gt_projection_on_original.png'), _oimg)
            except Exception:
                pass

        # save planner results summary (rollout-centric and concise)
        _video_map = {}
        for _v in evac_rollout_videos:
            try:
                _video_map[int(_v.get('step'))] = {
                    'path': _v.get('path'),
                    'exists': bool(_v.get('exists', False)),
                }
            except Exception:
                continue
        _rollout_records = []
        for _r in _dbg_rollout:
            _step = _r.get('step')
            _video = _video_map.get(int(_step)) if _step is not None and str(_step).isdigit() else None
            _rollout_records.append({
                'step': (None if _step is None else int(_step)),
                't_star': (None if _r.get('t_star') is None else int(_r.get('t_star'))),
                'min_dist': (None if _r.get('min_dist') is None else float(_r.get('min_dist'))),
                'phase_key': _r.get('phase_key'),
                'perturb_mode': _r.get('perturb_mode'),
                'sampled_error_mode': _r.get('sampled_error_mode'),
                'perturb_dir_bin_id': (
                    None if _r.get('perturb_dir_bin_id') is None else int(_r.get('perturb_dir_bin_id'))
                ),
                'perturb_mag_bin_id': (
                    None if _r.get('perturb_mag_bin_id') is None else int(_r.get('perturb_mag_bin_id'))
                ),
                'perturb_axis_bin_id': (
                    None if _r.get('perturb_axis_bin_id') is None else int(_r.get('perturb_axis_bin_id'))
                ),
                'perturb_translation_gain_m': (
                    None if _r.get('perturb_translation_gain_m') is None else float(_r.get('perturb_translation_gain_m'))
                ),
                'perturb_rotation_deg': (
                    None if _r.get('perturb_rotation_deg') is None else float(_r.get('perturb_rotation_deg'))
                ),
                'perturb_dir_left': _r.get('perturb_dir_left'),
                'perturb_dir_right': _r.get('perturb_dir_right'),
                'perturb_axis_left': _r.get('perturb_axis_left'),
                'perturb_axis_right': _r.get('perturb_axis_right'),
                'active_left_arm': bool(_r.get('active_left_arm', True)),
                'active_right_arm': bool(_r.get('active_right_arm', True)),
                'active_left_gripper': bool(_r.get('active_left_gripper', True)),
                'active_right_gripper': bool(_r.get('active_right_gripper', True)),
                'action_perturbed': bool(_r.get('action_perturbed', False)),
                'left_grip': (None if _r.get('left_grip') is None else float(_r.get('left_grip'))),
                'right_grip': (None if _r.get('right_grip') is None else float(_r.get('right_grip'))),
                'reason': _r.get('reason'),
                'recover_eval_enable': bool(_r.get('recover_eval_enable', False)),
                'recover_eval_mode': _r.get('recover_eval_mode'),
                'recover_eval_recoverable': (
                    None if _r.get('recover_eval_recoverable') is None else bool(_r.get('recover_eval_recoverable'))
                ),
                'recover_eval_metric_name': _r.get('recover_eval_metric_name'),
                'recover_eval_metric': (
                    None if _r.get('recover_eval_metric') is None else float(_r.get('recover_eval_metric'))
                ),
                'recover_eval_threshold': (
                    None if _r.get('recover_eval_threshold') is None else float(_r.get('recover_eval_threshold'))
                ),
                'recover_eval_metrics': _r.get('recover_eval_metrics'),
                'recover_eval_thresholds': _r.get('recover_eval_thresholds'),
                'recover_eval_passes': _r.get('recover_eval_passes'),
                'recover_eval_failed_thresholds': _r.get('recover_eval_failed_thresholds'),
                'recover_eval_horizon': (
                    None if _r.get('recover_eval_horizon') is None else int(_r.get('recover_eval_horizon'))
                ),
                'recover_eval_gt_ref_idx': (
                    None if _r.get('recover_eval_gt_ref_idx') is None else int(_r.get('recover_eval_gt_ref_idx'))
                ),
                'recover_eval_nearest_dist': (
                    None if _r.get('recover_eval_nearest_dist') is None else float(_r.get('recover_eval_nearest_dist'))
                ),
                'recover_eval_window_start': (
                    None if _r.get('recover_eval_window_start') is None else int(_r.get('recover_eval_window_start'))
                ),
                'recover_eval_window_end': (
                    None if _r.get('recover_eval_window_end') is None else int(_r.get('recover_eval_window_end'))
                ),
                'recover_eval_video': _r.get('recover_eval_video'),
                'evac_video': _video,
                'evac_video_dir': (None if _video is None else os.path.dirname(_video.get('path'))),
            })

        # Save a compact closed-loop-only view for quick debugging.
        def _drop_none(d):
            return {k: v for k, v in d.items() if v is not None}

        _closed_loop_rollouts = []
        for _r in _rollout_records:
            _rec = {
                'step': _r.get('step'),
                't_star': _r.get('t_star'),
                'phase_key': _r.get('phase_key'),
                'action_perturbed': _r.get('action_perturbed'),
                'perturb_mode': _r.get('perturb_mode'),
                'sampled_error_mode': _r.get('sampled_error_mode'),
                'perturb_dir_bin_id': _r.get('perturb_dir_bin_id'),
                'perturb_mag_bin_id': _r.get('perturb_mag_bin_id'),
                'perturb_axis_bin_id': _r.get('perturb_axis_bin_id'),
                'perturb_translation_gain_m': _r.get('perturb_translation_gain_m'),
                'perturb_rotation_deg': _r.get('perturb_rotation_deg'),
                'perturb_dir_left': _r.get('perturb_dir_left'),
                'perturb_dir_right': _r.get('perturb_dir_right'),
                'perturb_axis_left': _r.get('perturb_axis_left'),
                'perturb_axis_right': _r.get('perturb_axis_right'),
                'recover_eval_mode': _r.get('recover_eval_mode'),
                'recover_eval_recoverable': _r.get('recover_eval_recoverable'),
                'recover_eval_metric_name': _r.get('recover_eval_metric_name'),
                'recover_eval_metric': _r.get('recover_eval_metric'),
                'recover_eval_threshold': _r.get('recover_eval_threshold'),
                'recover_eval_metrics': _r.get('recover_eval_metrics'),
                'recover_eval_thresholds': _r.get('recover_eval_thresholds'),
                'recover_eval_passes': _r.get('recover_eval_passes'),
                'recover_eval_failed_thresholds': _r.get('recover_eval_failed_thresholds'),
                'recover_eval_gt_ref_idx': _r.get('recover_eval_gt_ref_idx'),
                'recover_eval_nearest_dist': _r.get('recover_eval_nearest_dist'),
                'recover_eval_window_start': _r.get('recover_eval_window_start'),
                'recover_eval_window_end': _r.get('recover_eval_window_end'),
                'recover_eval_video': _r.get('recover_eval_video'),
            }
            _closed_loop_rollouts.append(_drop_none(_rec))

        _closed_loop_info = {
            'reason': 'closed_loop_generated',
            'sampled_unit': sampled_unit,
            'perturb_compare': perturb_compare_meta,
            'min_dist': float(min_dist),
            'recover_eval_enable': bool(recover_eval_enable),
            'recover_eval_any_unrecoverable': bool(recover_eval_any_unrecoverable),
            'recover_eval_first_unrecoverable_step': (
                None if recover_eval_first_unrecoverable_step is None else int(recover_eval_first_unrecoverable_step)
            ),
            'recover_eval_last': recover_eval_last,
            'rollout': _closed_loop_rollouts,
            'world_model_backend': world_model_backend,
            'world_model_compare': world_model_compare_records,
            'world_model_compare_sim': compare_sim_artifact,
            'evac_correction_video': evac_corr_video,
        }
        with open(os.path.join(_dbg_corr, 'closed_loop_info.json'), 'w') as _f:
            json.dump(_closed_loop_info, _f, indent=2)

    prefix_len_meta = int(np.clip(int(rollout_exec_steps), 1, int(corr.shape[0])))
    error_action_prefix_raw = np.asarray(corr[:prefix_len_meta], dtype=np.float32).copy()
    if nearest_mode == "gripper_close":
        correction_branch = "gripper_close"
    elif nearest_mode == "translation":
        correction_branch = "translation"
    elif nearest_mode == "rotation":
        correction_branch = "rotation"
    else:
        correction_branch = "unsupported"

    sampled_error_mode = None
    if isinstance(dyn_state, dict):
        sampled_error_mode = dyn_state.get("error_mode")

    corr_meta = {
        "correction_generated": bool(force_generate_correction),
        "closed_loop_fallback_used": bool(force_generate_correction),
        "correction_branch": correction_branch,
        "sampled_phase_key": phase_key_fixed,
        "sampled_error_mode": sampled_error_mode,
        "forced_error_mode": forced_error_mode_key,
        "sampled_phase_bin_id": phase_bin_fixed,
        "sampled_phase_instance_idx": phase_instance_fixed,
        "sampled_active_arm_pattern": sampled_active_arm_pattern_key,
        "forced_dir_bin_id": forced_dir_bin_fixed,
        "forced_mag_bin_id": forced_mag_bin_fixed,
        "nearest_mode": nearest_mode,
        "rollout_exec_steps": int(rollout_exec_steps),
        "full_chunk_recovery": bool(cfg.get("full_chunk_recovery", False)),
        "correction_prefix_len": int(prefix_len),
        "error_action_prefix_len": int(prefix_len_meta),
        "t_star": int(t_star),
        "min_dist": float(min_dist),
        "recover_eval_enable": bool(recover_eval_enable),
        "recover_eval_any_unrecoverable": bool(recover_eval_any_unrecoverable),
        "recover_eval_first_unrecoverable_step": (
            None if recover_eval_first_unrecoverable_step is None else int(recover_eval_first_unrecoverable_step)
        ),
        "recover_eval_last": recover_eval_last,
        "world_model_backend": world_model_backend,
        "evac_rollout_videos": evac_rollout_videos,
        "world_model_rollout_videos": world_model_rollout_videos,
        "world_model_compare": world_model_compare_records,
        "world_model_compare_sim": compare_sim_artifact,
        "evac_correction_video": evac_corr_video,
        "debug_dir": debug_dir,
        "source_start_ts": int(start_ts),
        "start_idx": int(start_ts),
        "deviation_idx": int(start_ts),
        "error_action_prefix_raw": error_action_prefix_raw,
        "corr_action_chunk_raw": np.asarray(corr, dtype=np.float32).copy(),
        "corr_qpos_raw": np.asarray(curr_qpos_raw, dtype=np.float32).copy(),
        "corr_image_chw_float": curr_image[0].detach().cpu().numpy().astype(np.float32),
        # Exported correction episodes splice the clean original tail at the
        # sampled GT time.  recover_eval gt_ref_idx is only a nearest-match
        # diagnostic for judging base-policy recovery, and using it here can
        # jump far ahead in the episode.
        "attach_idx": int(start_ts),
        "perturb_action_prefix_raw": perturb_action_prefix_raw,
        "perturb_action_prefix_len": (
            0 if perturb_action_prefix_raw is None else int(perturb_action_prefix_raw.shape[0])
        ),
        "perturb_start_qpos_raw": perturb_start_qpos_raw,
        "perturb_final_qpos_raw": perturb_final_qpos_raw,
        "perturb_delta_linf": perturb_delta_linf,
        "perturb_delta_l2": perturb_delta_l2,
    }

    return (
        curr_image.to(device),
        torch.from_numpy(qn.astype(np.float32)).to(device),
        torch.from_numpy(padded).float().to(device),
        torch.from_numpy(is_pad).bool().to(device),
        corr_meta,
    )
