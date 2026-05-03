from __future__ import annotations

import math
import json
import os
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch


def resolve_evac_dataset_name(evac_cfg, dataset_name=None):
    if dataset_name:
        return str(dataset_name)
    try:
        domains = evac_cfg.data.params.train.params.domains
        if domains and len(domains) > 0:
            return str(domains[0])
    except Exception:
        pass
    return "agibotworld"


def find_nearest_traj_point(
    fk_left_pos,
    fk_left_quat,
    fk_right_pos,
    fk_right_quat,
    left_endpose,
    right_endpose,
    orient_weight=0.0,
    curr_left_grip=None,
    curr_right_grip=None,
    left_gripper_traj=None,
    right_gripper_traj=None,
    gripper_penalty=0.0,
    window_start=None,
    window_end=None,
):
    left_d = np.linalg.norm(left_endpose[:, :3] - fk_left_pos, axis=1)
    right_d = np.linalg.norm(right_endpose[:, :3] - fk_right_pos, axis=1)
    dists = (left_d + right_d) / 2
    if orient_weight > 0:
        left_dot = np.clip(np.abs(np.sum(left_endpose[:, 3:7] * fk_left_quat, axis=1)), 0.0, 1.0)
        right_dot = np.clip(np.abs(np.sum(right_endpose[:, 3:7] * fk_right_quat, axis=1)), 0.0, 1.0)
        left_d_ori = 2.0 * np.arccos(left_dot)
        right_d_ori = 2.0 * np.arccos(right_dot)
        dists += orient_weight * (left_d_ori + right_d_ori) / 2
    if gripper_penalty > 0 and left_gripper_traj is not None:
        curr_lg = 0.0 if curr_left_grip <= 0.5 else 1.0
        curr_rg = 0.0 if curr_right_grip <= 0.5 else 1.0
        traj_lg = (left_gripper_traj > 0.5).astype(np.float32)
        traj_rg = (right_gripper_traj > 0.5).astype(np.float32)
        lg_mismatch = np.abs(traj_lg - curr_lg)
        rg_mismatch = np.abs(traj_rg - curr_rg)
        dists += gripper_penalty * (lg_mismatch + rg_mismatch) / 2
    if window_start is not None or window_end is not None:
        ws = 0 if window_start is None else int(np.clip(window_start, 0, len(dists) - 1))
        we = len(dists) if window_end is None else int(np.clip(window_end, ws + 1, len(dists)))
        d_win = np.full_like(dists, np.inf, dtype=np.float32)
        d_win[ws:we] = dists[ws:we]
        if np.any(np.isfinite(d_win)):
            dists = d_win
    t_star = np.argmin(dists)
    return t_star, dists[t_star]


def quat_geodesic_deg_wxyz(q1_wxyz, q2_wxyz):
    q1 = np.asarray(q1_wxyz, dtype=np.float32).reshape(4,)
    q2 = np.asarray(q2_wxyz, dtype=np.float32).reshape(4,)
    n1 = float(np.linalg.norm(q1))
    n2 = float(np.linalg.norm(q2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    q1 = q1 / n1
    q2 = q2 / n2
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    rad = 2.0 * float(np.arccos(dot))
    return float(np.rad2deg(rad))


def _evac_denoise_runtime_meta(
    *,
    target_dir,
    dataset_name,
    ddim_steps,
    n_valid,
    num_chunk,
    chunk,
    n_previous,
    saving_video,
    infer_kwargs,
):
    kwargs = {} if infer_kwargs is None else dict(infer_kwargs)
    meta = {
        "target_dir": target_dir,
        "dataset_name": str(dataset_name),
        "ddim_steps": int(ddim_steps),
        "denoise_step_indices": list(range(int(ddim_steps))),
        "n_valid": int(n_valid),
        "num_chunk": int(num_chunk),
        "chunk": int(chunk),
        "n_previous": int(n_previous),
        "saving_video": bool(saving_video),
        "infer_kwargs": kwargs,
    }
    if bool(kwargs.get("use_dual_cache", False)):
        dc_budget = kwargs.get("dc_budget", None)
        dc_v_bounds = kwargs.get("dc_v_bounds", []) or []
        dc_v_bounds = [int(x) for x in dc_v_bounds]
        dual_cache = {
            "enabled": True,
            "dc_v_bounds": dc_v_bounds,
            "dc_budget": None if dc_budget is None else float(dc_budget),
            "dc_enc_start": int(kwargs.get("dc_enc_start", 999)),
            "dc_replay_step_noise": bool(kwargs.get("dc_replay_step_noise", False)),
            "dc_hf_metric": bool(kwargs.get("dc_hf_metric", False)),
            "dc_v_blur_on_reuse": bool(kwargs.get("dc_v_blur_on_reuse", False)),
        }
        if dc_budget is None and dc_v_bounds:
            anchors = [0] + dc_v_bounds[:-1]
            reuse_steps = []
            for anchor, end in zip(anchors, dc_v_bounds):
                reuse_steps.extend(range(int(anchor) + 1, min(int(end), int(ddim_steps))))
            reuse_steps = sorted(set(step for step in reuse_steps if 0 <= int(step) < int(ddim_steps)))
            full_steps = [step for step in range(int(ddim_steps)) if step not in set(reuse_steps)]
            dual_cache.update({
                "schedule_mode": "explicit_bounds",
                "anchor_steps": [int(x) for x in anchors if 0 <= int(x) < int(ddim_steps)],
                "reuse_steps": [int(x) for x in reuse_steps],
                "full_unet_steps": [int(x) for x in full_steps],
                "reuse_step_count": int(len(reuse_steps)),
                "full_unet_step_count": int(len(full_steps)),
            })
        elif dc_budget is not None:
            dual_cache.update({
                "schedule_mode": "auto_budget",
                "note": "Actual reuse schedule is derived inside DDIM sampler after calibration.",
            })
        else:
            dual_cache.update({
                "schedule_mode": "enabled_no_bounds",
                "reuse_steps": [],
                "full_unet_steps": list(range(int(ddim_steps))),
                "reuse_step_count": 0,
                "full_unet_step_count": int(ddim_steps),
            })
        meta["dual_cache"] = dual_cache
    else:
        meta["dual_cache"] = {
            "enabled": False,
            "reuse_steps": [],
            "full_unet_steps": list(range(int(ddim_steps))),
            "reuse_step_count": 0,
            "full_unet_step_count": int(ddim_steps),
        }
    return meta


def _print_evac_runtime_meta(meta):
    env_value = os.environ.get("SMOLVLA_EVAC_PRINT_RUNTIME", None)
    if env_value is None:
        if not bool(meta.get("saving_video", False)):
            return
    elif str(env_value).strip().lower() in {"0", "false", "no", "off"}:
        return
    dc = meta.get("dual_cache", {}) if isinstance(meta, dict) else {}
    if bool(dc.get("enabled", False)):
        print(
            "[EVAC runtime] "
            f"dir={meta.get('target_dir')} "
            f"ddim_steps={meta.get('ddim_steps')} chunks={meta.get('num_chunk')} n_valid={meta.get('n_valid')} "
            f"dual_cache=True bounds={dc.get('dc_v_bounds')} mode={dc.get('schedule_mode')} "
            f"full_unet={dc.get('full_unet_step_count')}/{meta.get('ddim_steps')} "
            f"reuse={dc.get('reuse_step_count')}/{meta.get('ddim_steps')}",
            flush=True,
        )
    else:
        print(
            "[EVAC runtime] "
            f"dir={meta.get('target_dir')} "
            f"ddim_steps={meta.get('ddim_steps')} chunks={meta.get('num_chunk')} n_valid={meta.get('n_valid')} "
            "dual_cache=False",
            flush=True,
        )


def evac_inference(
    evac_model,
    evac_cfg,
    curr_image,
    fk_poses,
    grippers,
    raw_data,
    device,
    save_dir=None,
    ddim_steps=27,
    infer_kwargs=None,
    dataset_name=None,
):
    import cv2
    import torchvision.transforms as tvt
    from evac.lvdm.data.get_actions import get_actions
    from evac.lvdm.data.statistics import StatisticInfo

    dataset_name = resolve_evac_dataset_name(evac_cfg, dataset_name)
    chunk = evac_cfg.chunk
    n_prev = evac_cfg.n_previous
    n_states = len(fk_poses)

    h_native, w_native = raw_data["native_resolution"]
    # Stage1 tensors are aligned to SmolVLA's RGB convention before reaching EVAC.
    img_rgb = tvt.Resize((h_native, w_native))(curr_image)
    memories = img_rgb.unsqueeze(1).repeat(1, n_prev, 1, 1)

    all_ends_p = np.zeros((n_states, 2, 3), dtype=np.float32)
    all_ends_o = np.zeros((n_states, 2, 4), dtype=np.float32)
    gripper_arr = np.zeros((n_states, 2), dtype=np.float32)
    for idx, ((lp, lq, rp, rq), (lg, rg)) in enumerate(zip(fk_poses, grippers)):
        all_ends_p[idx, 0], all_ends_p[idx, 1] = lp, rp
        lq_xyzw = np.array([lq[1], lq[2], lq[3], lq[0]])
        rq_xyzw = np.array([rq[1], rq[2], rq[3], rq[0]])
        if lq_xyzw[3] < 0:
            lq_xyzw = -lq_xyzw
        if rq_xyzw[3] < 0:
            rq_xyzw = -rq_xyzw
        all_ends_o[idx, 0] = lq_xyzw
        all_ends_o[idx, 1] = rq_xyzw
        gripper_arr[idx] = [lg * 120.0, rg * 120.0]

    slices = [0] * (n_prev - 1) + list(range(n_states))
    action, delta_action = get_actions(
        gripper=gripper_arr,
        all_ends_p=all_ends_p,
        all_ends_o=all_ends_o,
        slices=slices,
        delta_act_sidx=n_prev,
    )
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    sep = 2.0
    mean_v = torch.tensor(StatisticInfo[dataset_name]["mean"]).unsqueeze(0)
    std_v = torch.tensor(StatisticInfo[dataset_name]["std"]).unsqueeze(0)
    delta_action[:, :6] = (delta_action[:, :6] - sep * mean_v[:, :6]) / (sep * std_v[:, :6])
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep * mean_v[:, 6:]) / (sep * std_v[:, 6:])

    ext_cv = raw_data["extrinsic_cv"]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = ext_cv
    c2w = np.linalg.inv(w2c)
    n_act = action.shape[0]
    c2w_t = torch.from_numpy(c2w).float().unsqueeze(0).repeat(n_act, 1, 1)
    w2c_t = torch.from_numpy(w2c).float().unsqueeze(0).repeat(n_act, 1, 1)
    intrinsic = torch.from_numpy(raw_data["intrinsic_cv"]).float().clone()

    n_valid = n_states - 1
    num_chunk = int(math.ceil(float(n_valid) / chunk))
    tmp_dir = None
    if save_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="evac_")
        target_dir = tmp_dir
    else:
        target_dir = save_dir
    os.makedirs(target_dir, exist_ok=True)

    if infer_kwargs is None:
        infer_kwargs = {}

    runtime_meta = _evac_denoise_runtime_meta(
        target_dir=target_dir,
        dataset_name=dataset_name,
        ddim_steps=ddim_steps,
        n_valid=n_valid,
        num_chunk=num_chunk,
        chunk=chunk,
        n_previous=n_prev,
        saving_video=(save_dir is not None),
        infer_kwargs=infer_kwargs,
    )
    if save_dir is not None:
        try:
            with open(os.path.join(target_dir, "evac_runtime_meta.json"), "w", encoding="utf-8") as f:
                json.dump(runtime_meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    _print_evac_runtime_meta(runtime_meta)

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                frames, _ = evac_model.inference(
                    evac_cfg,
                    memories,
                    action,
                    delta_action,
                    c2w_t,
                    w2c_t,
                    intrinsic,
                    target_dir,
                    num_chunk,
                    chunk=chunk,
                    n_previous=n_prev,
                    n_valid=n_valid,
                    unconditional_guidance_scale=1.0,
                    guidance_rescale=0.7,
                    ddim_steps=ddim_steps,
                    dataset_name=dataset_name,
                    saving_video=(save_dir is not None),
                    saving_fps=30,
                    video_dir=target_dir,
                    **infer_kwargs,
                )
                torch.cuda.empty_cache()

    if save_dir is not None:
        try:
            inp_rgb = np.clip((img_rgb.permute(1, 2, 0).cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
            inp_bgr = inp_rgb[:, :, ::-1].copy()
            cv2.imwrite(os.path.join(target_dir, "input_frame.png"), inp_bgr)
        except Exception:
            pass

    last_rgb = cv2.resize(frames[-1], (640, 480))

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return torch.from_numpy(last_rgb).float().permute(2, 0, 1) / 255.0
