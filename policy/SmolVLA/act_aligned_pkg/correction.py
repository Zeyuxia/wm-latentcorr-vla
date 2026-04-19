from __future__ import annotations

import os
import re
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch

from policy.SmolVLA.failure_utils import (
    active_arm_pattern_id_to_key,
    error_mode_id_to_key,
    phase_id_to_key,
)
from policy.SmolVLA.act_aligned_pkg.utils_common import resample_trajectory
from policy.SmolVLA.act_aligned_pkg.perturbation import (
    _infer_active_arms_from_gt_window,
    _infer_phase_key_from_gt_window,
    perturb_action_chunk_online,
)

def load_raw_data(raw_data_dir, episode_id):
    import h5py, cv2 as _cv2
    path = os.path.join(raw_data_dir, f'episode{episode_id}.hdf5')
    with h5py.File(path, 'r') as f:
        # decode one frame to get native resolution (intrinsic is calibrated for this size)
        _frame0 = bytes(f['observation/head_camera/rgb'][0])
        _img0 = _cv2.imdecode(np.frombuffer(_frame0, np.uint8), _cv2.IMREAD_COLOR)
        h_native, w_native = _img0.shape[:2]
        return {
            'episode_path': path,
            'left_endpose': f['endpose/left_endpose'][()].astype(np.float32),
            'right_endpose': f['endpose/right_endpose'][()].astype(np.float32),
            'left_gripper': f['endpose/left_gripper'][()].astype(np.float32),
            'right_gripper': f['endpose/right_gripper'][()].astype(np.float32),
            'gt_left_arm': f['joint_action/left_arm'][()].astype(np.float32) if ('joint_action' in f and 'left_arm' in f['joint_action']) else None,
            'gt_right_arm': f['joint_action/right_arm'][()].astype(np.float32) if ('joint_action' in f and 'right_arm' in f['joint_action']) else None,
            'intrinsic_cv': f['observation/head_camera/intrinsic_cv'][0].astype(np.float32),
            'extrinsic_cv': f['observation/head_camera/extrinsic_cv'][0].astype(np.float32),
            'native_resolution': (h_native, w_native),  # intrinsic is calibrated for this
        }

def _init_export_episode_id(export_dir):
    if not os.path.isdir(export_dir):
        return 0
    max_id = -1
    for fn in os.listdir(export_dir):
        m = re.match(r"episode_(\d+)\.hdf5$", fn)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1

def _export_correction_sample_as_episode(
    export_dir,
    episode_id,
    cam_names,
    corr_image,
    corr_qpos_norm,
    corr_action_norm,
    corr_is_pad,
    norm_stats,
):
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"episode_{episode_id}.hdf5")

    img = corr_image.detach().cpu().numpy()            # (K,C,H,W), [0,1]
    qn = corr_qpos_norm.detach().cpu().numpy()         # (14,)
    an = corr_action_norm.detach().cpu().numpy()       # (Tmax,14)
    pad = corr_is_pad.detach().cpu().numpy().astype(bool)

    valid_len = int(np.sum(~pad))
    valid_len = max(1, valid_len)

    qpos_raw = (qn * norm_stats["qpos_std"] + norm_stats["qpos_mean"]).astype(np.float32)
    action_raw = (an * norm_stats["action_std"] + norm_stats["action_mean"]).astype(np.float32)
    action_raw = action_raw[:valid_len]
    qpos_seq = np.repeat(qpos_raw[None, :], valid_len, axis=0).astype(np.float32)

    img_u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    img_hwc = np.transpose(img_u8, (0, 2, 3, 1))  # (K,H,W,C)

    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("/action", data=action_raw, compression="gzip", compression_opts=1)
        f.create_dataset("/observations/qpos", data=qpos_seq, compression="gzip", compression_opts=1)
        g_img = f.create_group("/observations/images")
        for k, cam_name in enumerate(cam_names):
            frames = np.repeat(img_hwc[k][None, ...], valid_len, axis=0)
            g_img.create_dataset(cam_name, data=frames, compression="gzip", compression_opts=1)

def find_nearest_traj_point(fk_left_pos, fk_left_quat, fk_right_pos, fk_right_quat,
                            left_endpose, right_endpose, orient_weight=0.0,
                            curr_left_grip=None, curr_right_grip=None,
                            left_gripper_traj=None, right_gripper_traj=None,
                            gripper_penalty=0.0,
                            window_start=None, window_end=None):
    """Find nearest trajectory point using position + orientation + gripper distance.
    Quaternions are in wxyz format. orient_weight scales the geodesic
    orientation distance (radians) relative to position distance (meters).
    gripper_penalty penalizes matching to points with different gripper state
    (binarized at 0.5 threshold)."""
    left_d = np.linalg.norm(left_endpose[:, :3] - fk_left_pos, axis=1)
    right_d = np.linalg.norm(right_endpose[:, :3] - fk_right_pos, axis=1)
    dists = (left_d + right_d) / 2
    if orient_weight > 0:
        # quaternion geodesic distance: 2 * arccos(|q1 · q2|)
        left_dot = np.clip(np.abs(np.sum(left_endpose[:, 3:7] * fk_left_quat, axis=1)), 0.0, 1.0)
        right_dot = np.clip(np.abs(np.sum(right_endpose[:, 3:7] * fk_right_quat, axis=1)), 0.0, 1.0)
        left_d_ori = 2.0 * np.arccos(left_dot)
        right_d_ori = 2.0 * np.arccos(right_dot)
        dists += orient_weight * (left_d_ori + right_d_ori) / 2
    if gripper_penalty > 0 and left_gripper_traj is not None:
        # binarize gripper: <=0.5 -> closed(0), >0.5 -> open(1)
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


def _quat_geodesic_deg_wxyz(q1_wxyz, q2_wxyz):
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

def evac_inference(evac_model, evac_cfg, curr_image, fk_poses, grippers, raw_data, device,
                   save_dir=None, ddim_steps=27, infer_kwargs=None):
    """EVAC prediction via model.inference() — matches infer_all.py exactly.

    Args:
        curr_image: (3, H, W) BGR [0,1] tensor (current observation)
        fk_poses: list of (lp, lq_wxyz, rp, rq_wxyz).
                  First entry = current state, rest = action outcomes. Total N entries.
        grippers: list of (lg, rg), same length as fk_poses.
        raw_data: dict with extrinsic_cv, intrinsic_cv, native_resolution
        save_dir: if set, save frames + traj video there
        infer_kwargs: optional kwargs forwarded to evac_model.inference()
    Returns: last predicted frame as (3, H, W) BGR [0,1] tensor
    """
    import cv2, math, tempfile, shutil
    from evac.lvdm.data.get_actions import get_actions
    from evac.lvdm.data.statistics import StatisticInfo
    import torchvision.transforms as tvt

    chunk = evac_cfg.chunk
    n_prev = evac_cfg.n_previous
    N = len(fk_poses)  # 1 (init state) + N_actions

    # --- Image: BGR→RGB [0,1], resize to native resolution, repeat n_prev ---
    # inference() scales intrinsic by sample_size / image_size, so image must
    # be at the same resolution as the intrinsic calibration (native_resolution).
    h_native, w_native = raw_data['native_resolution']
    img_rgb = curr_image[[2, 1, 0]]  # (3, 480, 640) → RGB
    img_rgb = tvt.Resize((h_native, w_native))(img_rgb)  # → native res
    memories = img_rgb.unsqueeze(1).repeat(1, n_prev, 1, 1)  # (3, n_prev, h, w)

    # --- Actions: same format as infer_all.py h5 variant ---
    all_ends_p = np.zeros((N, 2, 3), dtype=np.float32)
    all_ends_o = np.zeros((N, 2, 4), dtype=np.float32)
    gripper_arr = np.zeros((N, 2), dtype=np.float32)
    for i, ((lp, lq, rp, rq), (lg, rg)) in enumerate(zip(fk_poses, grippers)):
        all_ends_p[i, 0], all_ends_p[i, 1] = lp, rp
        # wxyz→xyzw, canonicalize so w>=0 (match HDF5 sign convention)
        lq_xyzw = np.array([lq[1], lq[2], lq[3], lq[0]])
        rq_xyzw = np.array([rq[1], rq[2], rq[3], rq[0]])
        if lq_xyzw[3] < 0: lq_xyzw = -lq_xyzw
        if rq_xyzw[3] < 0: rq_xyzw = -rq_xyzw
        all_ends_o[i, 0] = lq_xyzw
        all_ends_o[i, 1] = rq_xyzw
        gripper_arr[i] = [lg * 120.0, rg * 120.0]

    slices = [0] * (n_prev - 1) + list(range(N))
    action, delta_action = get_actions(
        gripper=gripper_arr, all_ends_p=all_ends_p, all_ends_o=all_ends_o,
        slices=slices, delta_act_sidx=n_prev)
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    mv = torch.tensor(StatisticInfo['agibotworld']['mean']).unsqueeze(0)
    sv = torch.tensor(StatisticInfo['agibotworld']['std']).unsqueeze(0)
    delta_action[:, :6] = (delta_action[:, :6] - mv[:, :6]) / sv[:, :6]
    delta_action[:, 7:13] = (delta_action[:, 7:13] - mv[:, 6:]) / sv[:, 6:]

    # --- Camera: same format as infer_all.py ---
    ext_cv = raw_data['extrinsic_cv']
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = ext_cv
    c2w = np.linalg.inv(w2c)
    n_act = action.shape[0]
    c2w_t = torch.from_numpy(c2w).float().unsqueeze(0).repeat(n_act, 1, 1)
    w2c_t = torch.from_numpy(w2c).float().unsqueeze(0).repeat(n_act, 1, 1)
    # clone() to prevent inference() in-place intrinsic scaling from corrupting raw_data
    intrinsic = torch.from_numpy(raw_data['intrinsic_cv']).float().clone()

    # --- Run inference ---
    n_valid = N - 1  # predicted frames = number of action steps
    num_chunk = int(math.ceil(float(n_valid) / chunk))
    tmp_dir = None
    if save_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix='evac_')
        target_dir = tmp_dir
    else:
        target_dir = save_dir
    os.makedirs(target_dir, exist_ok=True)

    if infer_kwargs is None:
        infer_kwargs = {}

    with open(os.devnull, "w") as _devnull:
        with redirect_stdout(_devnull), redirect_stderr(_devnull):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                frames, _ = evac_model.inference(
                    evac_cfg, memories, action, delta_action,
                    c2w_t, w2c_t, intrinsic,
                    target_dir, num_chunk,
                    chunk=chunk, n_previous=n_prev, n_valid=n_valid,
                    unconditional_guidance_scale=1.0,
                    guidance_rescale=0.7,
                    ddim_steps=ddim_steps,
                    dataset_name="agibotworld",
                    saving_video=(save_dir is not None),
                    saving_fps=30,
                    video_dir=target_dir,
                    **infer_kwargs,
                )
                torch.cuda.empty_cache()

    # Debug continuity helper: keep explicit input frame.
    if save_dir is not None:
        try:
            inp_rgb = np.clip((img_rgb.permute(1, 2, 0).cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
            inp_bgr = inp_rgb[:, :, ::-1].copy()
            cv2.imwrite(os.path.join(target_dir, 'input_frame.png'), inp_bgr)
        except Exception:
            pass

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # frames: (n_valid, H, W, 3) RGB uint8 at sample_size resolution
    last_rgb = cv2.resize(frames[-1], (640, 480))
    last_bgr = last_rgb[:, :, ::-1].copy()
    return torch.from_numpy(last_bgr).float().permute(2, 0, 1) / 255.0


def _shared_phase_color(phase_key):
    k = str(phase_key).strip().lower()
    cmap = {
        "approach": (255, 120, 0),
        "pregrasp": (0, 165, 255),
        "transport": (0, 255, 255),
        "place": (255, 0, 255),
    }
    return cmap.get(k, (180, 180, 180))


def _shared_draw_phase_polyline(img, seq, phase_seq, width=1):
    import cv2

    prev = None
    for i, pt in enumerate(seq):
        if pt is None:
            prev = None
            continue
        if prev is not None:
            c = _shared_phase_color(phase_seq[i] if i < len(phase_seq) else "unknown")
            cv2.line(img, (int(prev[0]), int(prev[1])), (int(pt[0]), int(pt[1])), c, width, cv2.LINE_AA)
        prev = pt


def _shared_draw_phase_legend(img):
    import cv2

    rows = [
        ("approach", _shared_phase_color("approach")),
        ("pregrasp", _shared_phase_color("pregrasp")),
        ("transport", _shared_phase_color("transport")),
        ("place", _shared_phase_color("place")),
    ]
    x0, y0 = 8, 8
    row_h = 14
    w = 126
    h = 6 + row_h * len(rows) + 6
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (220, 220, 220), 1, cv2.LINE_AA)
    for i, (name, color) in enumerate(rows):
        y = y0 + 14 + i * row_h
        cv2.circle(img, (x0 + 9, y - 3), 3, color, -1, cv2.LINE_AA)
        cv2.putText(img, name, (x0 + 17, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)


def _shared_build_pose_np_from_raw_indices(raw_data, raw_idx_list):
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


def _shared_project_base_uv_from_pose_np(pose_arr, K, E):
    import evac.lvdm.models.ddpm3d as ddpm3d_mod

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


def _shared_extract_base_uv(uvs, pts):
    out = []
    n = int(uvs.shape[0])
    for i in range(n):
        try:
            z = float(pts[0, i, 2, 0].item())
            u = int(uvs[i, 0, 0])
            v = int(uvs[i, 0, 1])
        except Exception:
            out.append(None)
            continue
        if z > 1e-6 and np.isfinite(uvs[i, 0, :]).all():
            out.append((u, v))
        else:
            out.append(None)
    return out


def _shared_compute_phase_projection(raw_data, phase_window_len, K, E):
    n_ep_total = int(raw_data['left_endpose'].shape[0])
    full_left_grip = np.asarray(raw_data['left_gripper'], dtype=np.float32).reshape(-1)
    full_right_grip = np.asarray(raw_data['right_gripper'], dtype=np.float32).reshape(-1)
    phase_raw_idx = []
    phase_seq = []
    if n_ep_total <= 1:
        return None, None, None
    stride = int(max(1, n_ep_total // 240))
    for raw_idx in range(0, n_ep_total, stride):
        ph = _infer_phase_key_from_gt_window(
            full_left_grip[raw_idx:],
            full_right_grip[raw_idx:],
            phase_window_len,
        )
        phase_raw_idx.append(raw_idx)
        phase_seq.append(ph)
    if len(phase_raw_idx) < 2:
        return None, None, None
    phase_pose_np = _shared_build_pose_np_from_raw_indices(raw_data, phase_raw_idx)
    if phase_pose_np is None:
        return None, None, None
    p_uvs_l, p_uvs_r, p_pts_l, p_pts_r = _shared_project_base_uv_from_pose_np(phase_pose_np, K, E)
    p_l_seq = _shared_extract_base_uv(p_uvs_l, p_pts_l.reshape(1, p_pts_l.shape[1], 4, 4))
    p_r_seq = _shared_extract_base_uv(p_uvs_r, p_pts_r.reshape(1, p_pts_r.shape[1], 4, 4))
    return p_l_seq, p_r_seq, phase_seq


def _shared_draw_phase_bin_starts(img, l_seq, r_seq, phase_seq, phase_bins):
    import cv2

    phase_bins = int(max(1, phase_bins))
    n = len(phase_seq)
    if n <= 0:
        return

    i = 0
    while i < n:
        key = str(phase_seq[i])
        j = i + 1
        while j < n and str(phase_seq[j]) == key:
            j += 1
        seg_len = j - i
        if seg_len > 0:
            for b in range(phase_bins):
                rel = int(np.floor(float(b) * float(seg_len) / float(phase_bins)))
                idx = i + min(seg_len - 1, max(0, rel))

                # Find a drawable uv near idx.
                uv_l = None
                uv_r = None
                for k in range(idx, j):
                    if uv_l is None and k < len(l_seq) and l_seq[k] is not None:
                        uv_l = l_seq[k]
                    if uv_r is None and k < len(r_seq) and r_seq[k] is not None:
                        uv_r = r_seq[k]
                    if uv_l is not None and uv_r is not None:
                        break
                if uv_l is None and uv_r is None:
                    for k in range(idx - 1, i - 1, -1):
                        if uv_l is None and k < len(l_seq) and l_seq[k] is not None:
                            uv_l = l_seq[k]
                        if uv_r is None and k < len(r_seq) and r_seq[k] is not None:
                            uv_r = r_seq[k]
                        if uv_l is not None and uv_r is not None:
                            break

                label_anchor = None
                if uv_l is not None:
                    ul, vl = int(uv_l[0]), int(uv_l[1])
                    cv2.circle(img, (ul, vl), 2, (0, 255, 0), -1, cv2.LINE_AA)
                    label_anchor = (ul, vl) if label_anchor is None else label_anchor
                if uv_r is not None:
                    ur, vr = int(uv_r[0]), int(uv_r[1])
                    cv2.circle(img, (ur, vr), 2, (0, 0, 255), -1, cv2.LINE_AA)
                    label_anchor = (ur, vr) if label_anchor is None else label_anchor

                if label_anchor is not None:
                    _tag_prefix = {
                        "approach": "A",
                        "pregrasp": "G",
                        "transport": "T",
                        "place": "L",
                    }.get(key, key[:1].upper() if len(key) > 0 else "?")
                    tag = f"{_tag_prefix}{b}"
                    tx, ty = (int(label_anchor[0]) + 4, int(label_anchor[1]) - 4)
                    cv2.putText(
                        img,
                        tag,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        img,
                        tag,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
        i = j


def _save_gt_projection_on_original(
    debug_corr_dir,
    image_data_s,
    curr_image,
    raw_data,
    start_ts,
    rollout_steps_total,
    phase_window_len,
    phase_bins,
):
    import cv2

    try:
        _oimg = (image_data_s[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_ref_idx = int(np.clip(start_ts, 0, raw_data['left_endpose'].shape[0] - 1))
        try:
            ep_path = raw_data.get('episode_path', None)
            if ep_path is not None and os.path.isfile(ep_path):
                import h5py
                with h5py.File(ep_path, 'r') as _f_gt:
                    _enc = bytes(_f_gt['observation/head_camera/rgb'][gt_ref_idx])
                _gt = cv2.imdecode(np.frombuffer(_enc, np.uint8), cv2.IMREAD_COLOR)
                if _gt is not None and _gt.size > 0:
                    _oimg = _gt
        except Exception:
            pass

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
        _ = np.linalg.inv(E).astype(np.float32)

        _overlay_gt = _oimg.copy()
        p_l_seq, p_r_seq, phase_seq = _shared_compute_phase_projection(raw_data, phase_window_len, K, E)
        if p_l_seq is not None and p_r_seq is not None and phase_seq is not None:
            _shared_draw_phase_polyline(_overlay_gt, p_l_seq, phase_seq, width=2)
            _shared_draw_phase_polyline(_overlay_gt, p_r_seq, phase_seq, width=2)
            _shared_draw_phase_bin_starts(_overlay_gt, p_l_seq, p_r_seq, phase_seq, phase_bins)
            _shared_draw_phase_legend(_overlay_gt)

        out_path = os.path.join(debug_corr_dir, 'gt_projection_on_original.png')
        cv2.imwrite(out_path, _overlay_gt)
        return {
            'path': out_path,
            'exists': bool(os.path.exists(out_path)),
            'gt_ref_idx': int(gt_ref_idx),
        }
    except Exception:
        try:
            import traceback
            with open(os.path.join(debug_corr_dir, 'gt_projection_on_original_error.txt'), 'w') as _f:
                _f.write(traceback.format_exc())
        except Exception:
            pass
        return None


def _save_recover_eval_compare_image(
    debug_corr_dir,
    raw_data,
    gt_ref_idx,
    recover_pred_img,
    step_idx,
    mode=None,
    recoverable=None,
    metric_name=None,
    metric=None,
    threshold=None,
    nearest_dist=None,
):
    import cv2

    def _put_text_hc(img, text, org, scale=0.5):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 1, cv2.LINE_AA)

    try:
        gi = int(np.clip(int(gt_ref_idx), 0, raw_data['left_endpose'].shape[0] - 1))
        _gt = None
        ep_path = raw_data.get('episode_path', None)
        if ep_path is not None and os.path.isfile(ep_path):
            import h5py
            with h5py.File(ep_path, 'r') as _f_gt:
                _enc = bytes(_f_gt['observation/head_camera/rgb'][gi])
            _gt = cv2.imdecode(np.frombuffer(_enc, np.uint8), cv2.IMREAD_COLOR)
        if _gt is None or _gt.size == 0:
            return None

        _pred = (recover_pred_img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        if _pred.shape[:2] != _gt.shape[:2]:
            _pred = cv2.resize(_pred, (_gt.shape[1], _gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        _img_cmp = np.concatenate([_gt, _pred], axis=1)
        info_lines = []
        info_lines.append(f"LEFT=GT_REF(idx={gi})   RIGHT=RECOVER_ROLLOUT_LAST")
        info_lines.append(f"mode={mode}")
        if recoverable is None:
            info_lines.append("recoverable=unknown")
        else:
            info_lines.append(f"recoverable={bool(recoverable)}")
        if metric_name is not None and metric is not None:
            info_lines.append(f"{metric_name}={float(metric):.6f}")
        if threshold is not None:
            info_lines.append(f"threshold={float(threshold):.6f}")
        if nearest_dist is not None:
            info_lines.append(f"nearest_dist={float(nearest_dist):.6f}")

        panel_h = 28 + 18 * len(info_lines)
        _cmp = np.zeros((_img_cmp.shape[0] + panel_h, _img_cmp.shape[1], 3), dtype=np.uint8)
        _cmp[:_img_cmp.shape[0], :, :] = _img_cmp
        _cmp[_img_cmp.shape[0]:, :, :] = 245
        cv2.line(_cmp, (0, _img_cmp.shape[0]), (_img_cmp.shape[1] - 1, _img_cmp.shape[0]), (120, 120, 120), 1, cv2.LINE_AA)
        _x0 = 10
        _y0 = _img_cmp.shape[0] + 20
        for _li, _txt in enumerate(info_lines):
            _y = _y0 + _li * 16
            _put_text_hc(_cmp, _txt, (_x0, _y), scale=0.46)
        out_path = os.path.join(debug_corr_dir, f"recover_eval_gtref_vs_rollout_last_step_{int(step_idx):03d}.png")
        cv2.imwrite(out_path, _cmp)
        return {
            'path': out_path,
            'exists': bool(os.path.exists(out_path)),
            'gt_ref_idx': int(gi),
            'step': int(step_idx),
            'mode': mode,
            'recoverable': (None if recoverable is None else bool(recoverable)),
            'metric_name': metric_name,
            'metric': (None if metric is None else float(metric)),
            'threshold': (None if threshold is None else float(threshold)),
            'nearest_dist': (None if nearest_dist is None else float(nearest_dist)),
        }
    except Exception:
        try:
            import traceback
            with open(os.path.join(debug_corr_dir, 'recover_eval_compare_error.txt'), 'w') as _f:
                _f.write(traceback.format_exc())
        except Exception:
            pass
        return None


def _save_perturb_compare_image(
    debug_corr_dir,
    first_img,
    last_img,
    sampled_unit,
    rollout_last_record=None,
):
    import cv2

    def _put_text_hc(img, text, org, scale=0.5):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 1, cv2.LINE_AA)

    try:
        _first = (first_img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        _last = (last_img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        if _last.shape[:2] != _first.shape[:2]:
            _last = cv2.resize(_last, (_first.shape[1], _first.shape[0]), interpolation=cv2.INTER_LINEAR)
        _img_cmp = np.concatenate([_first, _last], axis=1)

        su = sampled_unit if isinstance(sampled_unit, dict) else {}
        lines = [
            "LEFT=INPUT_FIRST(start)   RIGHT=PERTURBED_LAST(after_rollout)",
            f"phase={su.get('phase_key')} bin={su.get('phase_bin_id')} inst={su.get('phase_instance_idx')}",
            f"error_mode={su.get('error_mode')} arm={su.get('active_arm_pattern')}",
            f"dir_bin={su.get('dir_bin_id')} mag_bin={su.get('mag_bin_id')}",
        ]
        rr = rollout_last_record if isinstance(rollout_last_record, dict) else {}
        _tr_gain = rr.get('perturb_translation_gain_m')
        _rot_deg = rr.get('perturb_rotation_deg')
        _tr_gain_s = ("None" if _tr_gain is None else f"{float(_tr_gain):.3f}")
        _rot_deg_s = ("None" if _rot_deg is None else f"{float(_rot_deg):.3f}")
        lines.extend([
            f"sampled_error_mode={rr.get('sampled_error_mode')}",
            f"translation_gain_m={_tr_gain_s} rotation_deg={_rot_deg_s}",
        ])

        panel_h = 28 + 18 * len(lines)
        _cmp = np.zeros((_img_cmp.shape[0] + panel_h, _img_cmp.shape[1], 3), dtype=np.uint8)
        _cmp[:_img_cmp.shape[0], :, :] = _img_cmp
        _cmp[_img_cmp.shape[0]:, :, :] = 245
        cv2.line(_cmp, (0, _img_cmp.shape[0]), (_img_cmp.shape[1] - 1, _img_cmp.shape[0]), (120, 120, 120), 1, cv2.LINE_AA)
        y0 = _img_cmp.shape[0] + 20
        for i, t in enumerate(lines):
            _put_text_hc(_cmp, str(t), (10, y0 + 16 * i), scale=0.44)

        out_path = os.path.join(debug_corr_dir, 'perturb_input_vs_last.png')
        cv2.imwrite(out_path, _cmp)
        return {
            'path': out_path,
            'exists': bool(os.path.exists(out_path)),
            'sampled_unit': su,
        }
    except Exception:
        try:
            import traceback
            with open(os.path.join(debug_corr_dir, 'perturb_compare_error.txt'), 'w') as _f:
                _f.write(traceback.format_exc())
        except Exception:
            pass
        return None


def correction_step(policy_unwrapped, image_data_s, qpos_data_s, raw_data,
                    norm_stats, modules, cfg, device, debug_dir=None, start_ts=0,
                    sampled_phase_id=None, pregrasp_seg_start=None, pregrasp_seg_end=None,
                    sampled_phase_bin_id=None, sampled_phase_instance_id=None, forced_error_mode_id=None,
                    sampled_active_arm_pattern_id=None, forced_dir_bin_id=None,
                    forced_mag_bin_id=None, sampled_mode_prob=None,
                    sampled_entry_prob_within_mode=None, sampled_unit_prob=None,
                    precomputed_action_chunk_raw=None):
    fk = modules['fk']
    evac_model = modules['evac_model']
    evac_cfg = modules['evac_config']
    planner_l = modules['planner_left']
    planner_r = modules['planner_right']

    start_ts = int(max(0, start_ts))
    left_ep = raw_data['left_endpose'][start_ts:]
    right_ep = raw_data['right_endpose'][start_ts:]
    max_steps = cfg['max_rollout_steps']
    correction_force_generate = bool(cfg['correction_force_generate'])
    debug_correction_evac_rollout = bool(cfg.get('debug_correction_evac_rollout'))
    chunk_size = cfg['chunk_size']
    rollout_exec_steps = int(cfg['rollout_exec_steps'])
    max_action_len = cfg['max_action_len']
    orient_weight = float(cfg['orient_weight'])
    gripper_penalty = float(cfg['gripper_penalty'])
    recover_eval_enable = bool(cfg.get('recover_eval_enable', False))
    recover_eval_save_video = bool(cfg.get('recover_eval_save_video', False))
    recover_eval_debug = bool(cfg.get('recover_eval_save_video', False))
    recover_eval_gripper_open_thresh = float(cfg.get('recover_eval_gripper_open_thresh', 0.8))
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
    rollout_steps_total = 0
    evac_rollout_videos = []
    recover_eval_any_unrecoverable = False
    recover_eval_first_unrecoverable_step = None
    recover_eval_last = {
        'recoverable': False,
        'mode': None,
        'metric_name': None,
        'metric': None,
        'threshold': None,
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
        if _pat == "left_only":
            left_side_active, right_side_active = True, False
        elif _pat == "right_only":
            left_side_active, right_side_active = False, True
        elif _pat == "both":
            left_side_active, right_side_active = True, True
        else:
            raise RuntimeError(
                f"Invalid sampled_active_arm_pattern={sampled_active_arm_pattern_key!r} "
                "in failure_mode. Expected left_only/right_only/both."
            )
        active_info_fixed = {
            "left_arm": bool(left_side_active),
            "right_arm": bool(right_side_active),
            "left_gripper": bool(left_side_active),
            "right_gripper": bool(right_side_active),
        }
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

    max_steps_eff = max_steps

    for step in range(max_steps_eff):
        evac_debug_dir = os.path.join(debug_dir, 'evac', f'rollout_step_{step:03d}') if debug_dir else None
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

        if precomputed_action_chunk_raw is not None:
            act_raw = np.asarray(precomputed_action_chunk_raw, dtype=np.float32).copy()
        else:
            qn = (curr_qpos_raw - norm_stats['qpos_mean']) / norm_stats['qpos_std']
            qt = torch.from_numpy(qn).float().unsqueeze(0).to(device)
            it = curr_image.unsqueeze(0).to(device)
            # Use eval mode for rollout action generation to match deployment behavior
            # and avoid training-time dropout noise in correction targets.
            _was_training = policy_unwrapped.training
            policy_unwrapped.eval()
            with torch.no_grad():
                act_chunk = policy_unwrapped(qt, it)
            if _was_training:
                policy_unwrapped.train()
            act_np = act_chunk.squeeze(0).cpu().numpy()
            act_raw = act_np * norm_stats['action_std'] + norm_stats['action_mean']
        # Rollout executes a short online horizon; keep this aligned with rollout_exec_steps.
        exec_len = int(np.clip(rollout_exec_steps, 1, max(1, act_raw.shape[0])))
        act_raw = act_raw[:exec_len].copy()

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

        fk_poses = [(lp.copy(), lq.copy(), rp.copy(), rq.copy())]  # current state
        grip_list = [(left_grip, right_grip)]
        # act_raw is already generated at rollout_exec_steps horizon.
        for ai in range(len(act_raw)):
            ar = act_raw[ai]
            fr = fk.forward(ar[0:6], ar[7:13])
            fk_poses.append((fr['left'][0].copy(), fr['left'][1].copy(),
                             fr['right'][0].copy(), fr['right'][1].copy()))
            grip_list.append((ar[6], ar[13]))

        pred = evac_inference(
            evac_model,
            evac_cfg,
            curr_image[0],
            fk_poses,
            grip_list,
            raw_data,
            device,
            save_dir=evac_debug_dir,
            infer_kwargs=cfg['evac_infer_kwargs'],
        )
        if evac_debug_dir is not None:
            _vpath = os.path.join(evac_debug_dir, 'outputs.mp4')
            evac_rollout_videos.append({
                'step': int(step),
                'path': _vpath,
                'exists': bool(os.path.exists(_vpath)),
            })
        new_img = curr_image.clone()
        if tuple(pred.shape) != tuple(new_img[0].shape):
            pred_rs = torch.nn.functional.interpolate(
                pred.unsqueeze(0),
                size=(new_img.shape[-2], new_img.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )[0]
            pred = torch.clamp(pred_rs, 0.0, 1.0)
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
                qn_eval = (curr_qpos_raw - norm_stats['qpos_mean']) / norm_stats['qpos_std']
                qt_eval = torch.from_numpy(qn_eval).float().unsqueeze(0).to(device)
                it_eval = curr_image.unsqueeze(0).to(device)
                _was_training_eval = policy_unwrapped.training
                policy_unwrapped.eval()
                try:
                    with torch.no_grad():
                        act_chunk_eval = policy_unwrapped(qt_eval, it_eval)
                finally:
                    if _was_training_eval:
                        policy_unwrapped.train()
                act_eval_np = act_chunk_eval.squeeze(0).cpu().numpy()
                act_eval_raw = act_eval_np * norm_stats['action_std'] + norm_stats['action_mean']

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
                        evac_recover_eval_dir = os.path.join(
                            debug_dir, 'evac_recover_eval', f'rollout_step_{step:03d}'
                        )
                        recover_eval_rollout_last_img = evac_inference(
                            evac_model,
                            evac_cfg,
                            curr_image[0],
                            fk_eval_poses,
                            grip_eval_list,
                            raw_data,
                            device,
                            save_dir=evac_recover_eval_dir,
                            infer_kwargs=cfg.get('evac_infer_kwargs'),
                        )
                        _vpath_recover = os.path.join(evac_recover_eval_dir, 'outputs.mp4')
                        recover_eval_video = {
                            'path': _vpath_recover,
                            'exists': bool(os.path.exists(_vpath_recover)),
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

                mode_eval = str(recover_eval_mode).strip().lower() if recover_eval_mode is not None else ""
                if mode_eval == "gripper_close":
                    recover_eval_gt_ref_idx = int(gt_ref_idx)
                    open_vals = []
                    if bool(active_info.get('left_gripper', True)):
                        open_vals.append(float(np.max(act_eval_raw[:, 6])))
                    if bool(active_info.get('right_gripper', True)):
                        open_vals.append(float(np.max(act_eval_raw[:, 13])))
                    recover_eval_metric_name = "gripper_open_max"
                    recover_eval_metric = (None if len(open_vals) == 0 else float(np.max(open_vals)))
                    recover_eval_threshold = float(recover_eval_gripper_open_thresh)
                    recover_eval_recoverable = (
                        recover_eval_metric is not None and
                        float(recover_eval_metric) >= float(recover_eval_threshold)
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
                        rot_errs.append(_quat_geodesic_deg_wxyz(
                            fr_eval['left'][1],
                            raw_data['left_endpose'][recover_eval_gt_ref_idx, 3:7],
                        ))
                    if bool(active_info.get('right_arm', True)):
                        rot_errs.append(_quat_geodesic_deg_wxyz(
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

                if (
                    (recover_eval_gt_ref_idx is not None)
                    and recover_eval_debug
                    and (debug_dir is not None)
                    and (recover_eval_rollout_last_img is not None)
                ):
                    _dbg_corr_cmp = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr_cmp, exist_ok=True)
                    recover_eval_compare = _save_recover_eval_compare_image(
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
                        nearest_dist=recover_eval_nearest_dist,
                    )
            except Exception as exc:
                recover_eval_recoverable = None
                recover_eval_error = str(exc)

            if recover_eval_recoverable is False:
                recover_eval_any_unrecoverable = True
                if recover_eval_first_unrecoverable_step is None:
                    recover_eval_first_unrecoverable_step = int(step)
            recover_eval_last = {
                'recoverable': (None if recover_eval_recoverable is None else bool(recover_eval_recoverable)),
                'mode': recover_eval_mode,
                'metric_name': recover_eval_metric_name,
                'metric': (None if recover_eval_metric is None else float(recover_eval_metric)),
                'threshold': (None if recover_eval_threshold is None else float(recover_eval_threshold)),
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
            })

    force_generate_correction = False
    if correction_force_generate:
        force_generate_correction = True
        if debug_dir is not None:
            _dbg_rollout.append({
                'step': int(max_steps_eff),
                'reason': 'correction_force_generate',
                'max_rollout_steps': int(max_steps),
            })
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
        perturb_compare_meta = _save_perturb_compare_image(
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
            import json
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
                gt_projection_meta = _save_gt_projection_on_original(
                    _dbg_corr,
                    image_data_s,
                    curr_image,
                    raw_data,
                    start_ts,
                    rollout_steps_total,
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
            "error_action_prefix_raw": error_action_prefix_raw,
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
        prefix_len = int(np.clip(int(rollout_exec_steps), 1, max(1, chunk_size - 1)))
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

        if corr_left_active:
            l_prefix = np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], prefix_len, axis=0)
            l_tail = _take_tail(gt_left_arm, follow_abs, suffix_len, l_tgt_q)
            l_grip_prefix = _build_grip_prefix(left_grip, tl_grip)
            l_grip_tail = _take_tail(left_grip_traj, follow_idx + 1, suffix_len, np.array([tl_grip], dtype=np.float32))[:, 0]
            lt = np.concatenate([l_prefix, l_tail], axis=0).astype(np.float32)
            lg = np.concatenate([l_grip_prefix, l_grip_tail], axis=0).astype(np.float32)

        if corr_right_active:
            r_prefix = np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], prefix_len, axis=0)
            r_tail = _take_tail(gt_right_arm, follow_abs, suffix_len, r_tgt_q)
            r_grip_prefix = _build_grip_prefix(right_grip, tr_grip)
            r_grip_tail = _take_tail(right_grip_traj, follow_idx + 1, suffix_len, np.array([tr_grip], dtype=np.float32))[:, 0]
            rt = np.concatenate([r_prefix, r_tail], axis=0).astype(np.float32)
            rg = np.concatenate([r_grip_prefix, r_grip_tail], axis=0).astype(np.float32)
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
        #    back to the original first GT action pose at sampled start_ts,
        # 2) then follow GT forward to fill chunk_size.
        import sapien
        _recover_mode = str(nearest_mode)
        prefix_len = int(np.clip(int(rollout_exec_steps), 1, max(1, chunk_size - 1)))
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
                    import json
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
                    import json
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
                    import json
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
                    import json
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
        if corr_left_active:
            l_tail = _take_tail(gt_left_arm, follow_abs + 1, suffix_len, l_tgt_q)
            lt = np.concatenate([l_prefix, l_tail], axis=0).astype(np.float32)
            if prefix_len <= 1:
                l_grip_prefix = np.array([tl_grip], dtype=np.float32)
            else:
                l_grip_prefix = np.linspace(float(left_grip), tl_grip, prefix_len, dtype=np.float32)
            l_grip_tail = _take_tail(left_grip_traj, follow_idx + 1, suffix_len, np.array([tl_grip], dtype=np.float32))[:, 0]
            lg = np.concatenate([l_grip_prefix, l_grip_tail], axis=0).astype(np.float32)

        if corr_right_active:
            r_tail = _take_tail(gt_right_arm, follow_abs + 1, suffix_len, r_tgt_q)
            rt = np.concatenate([r_prefix, r_tail], axis=0).astype(np.float32)
            if prefix_len <= 1:
                r_grip_prefix = np.array([tr_grip], dtype=np.float32)
            else:
                r_grip_prefix = np.linspace(float(right_grip), tr_grip, prefix_len, dtype=np.float32)
            r_grip_tail = _take_tail(right_grip_traj, follow_idx + 1, suffix_len, np.array([tr_grip], dtype=np.float32))[:, 0]
            rg = np.concatenate([r_grip_prefix, r_grip_tail], axis=0).astype(np.float32)
    else:
        if debug_dir is not None:
            import json
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

    # --- debug: save correction summary jsons ---
    if debug_dir is not None:
        import json
        import cv2
        _dbg_corr = os.path.join(debug_dir, 'correction')
        os.makedirs(_dbg_corr, exist_ok=True)
        evac_corr_video = None
        if debug_correction_evac_rollout:
            try:
                _lq0, _rq0 = curr_qpos_raw[0:6], curr_qpos_raw[7:13]
                _lg0, _rg0 = curr_qpos_raw[6], curr_qpos_raw[13]
                _fk0 = fk.forward(_lq0, _rq0)
                _lp0, _lq0w = _fk0['left']
                _rp0, _rq0w = _fk0['right']
                fk_poses_corr = [(_lp0.copy(), _lq0w.copy(), _rp0.copy(), _rq0w.copy())]
                grip_list_corr = [(float(_lg0), float(_rg0))]
                for _ai in range(corr.shape[0]):
                    _ar = corr[_ai]
                    _fr = fk.forward(_ar[0:6], _ar[7:13])
                    fk_poses_corr.append((
                        _fr['left'][0].copy(), _fr['left'][1].copy(),
                        _fr['right'][0].copy(), _fr['right'][1].copy()
                    ))
                    grip_list_corr.append((float(_ar[6]), float(_ar[13])))
                evac_corr_debug_dir = os.path.join(_dbg_corr, 'evac_correction_generated')
                _ = evac_inference(
                    evac_model,
                    evac_cfg,
                    curr_image[0],
                    fk_poses_corr,
                    grip_list_corr,
                    raw_data,
                    device,
                    save_dir=evac_corr_debug_dir,
                    infer_kwargs=cfg.get('evac_infer_kwargs'),
                )
                _vpath_corr = os.path.join(evac_corr_debug_dir, 'outputs.mp4')
                evac_corr_video = {
                    'path': _vpath_corr,
                    'exists': bool(os.path.exists(_vpath_corr)),
                }
            except Exception:
                try:
                    import traceback
                    with open(os.path.join(_dbg_corr, 'evac_correction_generated_error.txt'), 'w') as _f:
                        _f.write(traceback.format_exc())
                except Exception:
                    pass
        # Save correction projection overlays on original/corrected images.
        try:
            _cimg = (curr_image[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            _oimg = (image_data_s[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            gt_ref_idx = int(np.clip(start_ts, 0, raw_data['left_endpose'].shape[0] - 1))
            try:
                ep_path = raw_data.get('episode_path', None)
                if ep_path is not None and os.path.isfile(ep_path):
                    import h5py
                    with h5py.File(ep_path, 'r') as _f_gt:
                        _enc = bytes(_f_gt['observation/head_camera/rgb'][gt_ref_idx])
                    _gt = cv2.imdecode(np.frombuffer(_enc, np.uint8), cv2.IMREAD_COLOR)
                    if _gt is not None and _gt.size > 0:
                        _oimg = _gt
            except Exception:
                pass

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
            p_l_seq, p_r_seq, phase_seq = _shared_compute_phase_projection(raw_data, phase_window_len, K, E)
            if p_l_seq is not None and p_r_seq is not None and phase_seq is not None:
                _shared_draw_phase_polyline(_overlay_gt, p_l_seq, phase_seq, width=2)
                _shared_draw_phase_polyline(_overlay_gt, p_r_seq, phase_seq, width=2)
                _shared_draw_phase_polyline(_overlay_c, p_l_seq, phase_seq, width=1)
                _shared_draw_phase_polyline(_overlay_c, p_r_seq, phase_seq, width=1)
                _shared_draw_phase_bin_starts(_overlay_gt, p_l_seq, p_r_seq, phase_seq, phase_bins)
                _shared_draw_phase_bin_starts(_overlay_c, p_l_seq, p_r_seq, phase_seq, phase_bins)
                _shared_draw_phase_legend(_overlay_gt)

            # Corrected overlay: keep only correction trajectories + phase-colored GT path.
            _overlay_c[mask] = (0.6 * _overlay_c[mask] + 0.4 * traj_u8[mask]).astype(np.uint8)
            _draw_polyline(_overlay_c, luv, (0, 255, 0))
            _draw_polyline(_overlay_c, ruv, (0, 0, 255))
            _annotate_start_end(_overlay_c, luv, "L", (0, 255, 0))
            _annotate_start_end(_overlay_c, ruv, "R", (0, 0, 255))

            cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_corrected.png'), _overlay_c)
            cv2.imwrite(os.path.join(_dbg_corr, 'gt_projection_on_original.png'), _overlay_gt)
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
                    cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_corrected.png'), _overlay_c)
                elif '_cimg' in locals():
                    cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_corrected.png'), _cimg)
                if '_overlay_gt' in locals():
                    cv2.imwrite(os.path.join(_dbg_corr, 'gt_projection_on_original.png'), _overlay_gt)
                elif '_oimg' in locals():
                    cv2.imwrite(os.path.join(_dbg_corr, 'gt_projection_on_original.png'), _oimg)
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
        }
        with open(os.path.join(_dbg_corr, 'closed_loop_info.json'), 'w') as _f:
            json.dump(_closed_loop_info, _f, indent=2)

    error_action_prefix_raw = np.asarray(act_raw, dtype=np.float32).copy()
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
        "sampled_error_mode": sampled_error_mode,
        "forced_error_mode": forced_error_mode_key,
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
        "error_action_prefix_raw": error_action_prefix_raw,
    }

    return (
        curr_image.to(device),
        torch.from_numpy(qn.astype(np.float32)).to(device),
        torch.from_numpy(padded).float().to(device),
        torch.from_numpy(is_pad).bool().to(device),
        corr_meta,
    )

def _save_loss_batch_projection(
    save_dir,
    image_cam,
    action_norm,
    is_pad,
    raw_data,
    fk,
    norm_stats,
    meta=None,
):
    os.makedirs(save_dir, exist_ok=True)
    import json
    import cv2

    act = np.asarray(action_norm.detach().cpu().numpy(), dtype=np.float32)
    pad = np.asarray(is_pad.detach().cpu().numpy(), dtype=bool).reshape(-1)
    if act.ndim != 2 or act.shape[1] < 14:
        return
    valid = np.where(~pad)[0]
    if valid.size == 0:
        return
    act = act[valid]
    act_raw = np.asarray(act * norm_stats['action_std'] + norm_stats['action_mean'], dtype=np.float32)
    if act_raw.shape[0] <= 0:
        return

    img_u8 = np.clip(
        image_cam.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    overlay = img_u8.copy()

    K = raw_data['intrinsic_cv'].astype(np.float32).copy()
    E = np.eye(4, dtype=np.float32)
    E[:3, :] = raw_data['extrinsic_cv'].astype(np.float32)
    h_native, w_native = raw_data.get('native_resolution', (overlay.shape[0], overlay.shape[1]))
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
        lp, lq_wxyz = fr['left']
        rp, rq_wxyz = fr['right']
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
        cv2.imwrite(os.path.join(save_dir, 'loss_projection_on_input.png'), overlay)
    except Exception:
        import traceback
        with open(os.path.join(save_dir, 'loss_projection_error.txt'), 'w') as _f:
            _f.write(traceback.format_exc())
        cv2.imwrite(os.path.join(save_dir, 'loss_projection_on_input.png'), overlay)

    meta_out = {'n_valid_actions': int(act_raw.shape[0]), 'image_hw': [int(h_img), int(w_img)]}
    if isinstance(meta, dict):
        meta_out.update(meta)
    with open(os.path.join(save_dir, 'loss_projection_meta.json'), 'w') as _f:
        json.dump(meta_out, _f, indent=2)
