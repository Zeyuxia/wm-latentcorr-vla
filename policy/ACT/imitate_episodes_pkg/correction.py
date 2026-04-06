from __future__ import annotations

import os
import re

import numpy as np
import torch

from imitate_episodes_pkg.utils import resample_trajectory
from imitate_episodes_pkg.perturbation import (
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

def correction_step(policy_unwrapped, image_data_s, qpos_data_s, raw_data,
                    norm_stats, modules, cfg, device, debug_dir=None, start_ts=0,
                    pregrasp_seg_start=None, pregrasp_seg_end=None):
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

    left_grip_traj = raw_data['left_gripper'][start_ts:]
    right_grip_traj = raw_data['right_gripper'][start_ts:]
    phase_window_len = int(cfg['sample_phase_window_len'])
    phase_key_fixed = _infer_phase_key_from_gt_window(
        left_grip_traj=left_grip_traj,
        right_grip_traj=right_grip_traj,
        window_len=phase_window_len,
    )

    qpos_raw = qpos_data_s.cpu().numpy() * norm_stats['qpos_std'] + norm_stats['qpos_mean']
    curr_image = image_data_s.clone()
    curr_qpos_raw = qpos_raw.copy()

    left_q = right_q = None
    left_grip = right_grip = 0.0
    t_star = 0
    min_dist = 0.0
    _dbg_rollout = []  # collect per-step debug info
    dyn_state = None
    rollout_steps_total = 0
    evac_rollout_videos = []

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
        min_dist_start = float(min_dist)

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
        # collect debug info for this rollout step
        if debug_dir is not None:
            sampled_error_mode = None
            if isinstance(dyn_state, dict):
                sampled_error_mode = dyn_state.get('error_mode')
            _dbg_rollout.append({
                'step': step, 't_star': int(t_star), 'min_dist': float(min_dist),
                'min_dist_start': float(min_dist_start),
                'progress': float(progress), 'phase_key': phase_key,
                'active_left_arm': bool(active_info.get('left_arm', True)),
                'active_right_arm': bool(active_info.get('right_arm', True)),
                'active_left_gripper': bool(active_info.get('left_gripper', True)),
                'active_right_gripper': bool(active_info.get('right_gripper', True)),
                'active_left_arm_score': float(active_info.get('left_arm_score', 0.0)),
                'active_right_arm_score': float(active_info.get('right_arm_score', 0.0)),
                'fk_left_pos': lp.tolist(), 'fk_right_pos': rp.tolist(),
                'left_grip': float(left_grip), 'right_grip': float(right_grip),
                'action_perturbed': bool(perturbed),
                'perturb_mode': pert_mode,
                'sampled_error_mode': sampled_error_mode,
                'act_chunk_raw_first': act_raw[0].tolist(),
                'act_chunk_raw_last': act_raw[-1].tolist(),
                'rollout_exec_len': int(len(act_raw)),
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
                _skip_reason = 'correction_not_forced'
                json.dump({'reason': _skip_reason, 'min_dist': float(min_dist),
                           'threshold': None,
                           'rollout': _dbg_rollout}, _f, indent=2)
        return None

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
            gt_ref_idx = int(np.clip(start_ts + int(rollout_steps_total), 0, raw_data['left_endpose'].shape[0] - 1))
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

            # Phase-colored GT trajectory projection (reuse current phase judging logic).
            # Draw downsampled GT path on the full episode timeline.
            n_ep_total = int(raw_data['left_endpose'].shape[0])
            full_left_grip = np.asarray(raw_data['left_gripper'], dtype=np.float32).reshape(-1)
            full_right_grip = np.asarray(raw_data['right_gripper'], dtype=np.float32).reshape(-1)
            phase_raw_idx = []
            phase_seq = []
            if n_ep_total > 1:
                stride = int(max(1, n_ep_total // 240))
                for raw_idx in range(0, n_ep_total, stride):
                    ph = _infer_phase_key_from_gt_window(
                        full_left_grip[raw_idx:],
                        full_right_grip[raw_idx:],
                        phase_window_len,
                    )
                    phase_raw_idx.append(raw_idx)
                    phase_seq.append(ph)
                if len(phase_raw_idx) >= 2:
                    phase_pose_np = _build_pose_np_from_raw_indices(phase_raw_idx)
                    if phase_pose_np is not None:
                        p_uvs_l, p_uvs_r, p_pts_l, p_pts_r = _project_base_uv_from_pose_np(phase_pose_np)
                        p_l_seq = _extract_base_uv(p_uvs_l, p_pts_l.reshape(1, p_pts_l.shape[1], 4, 4))
                        p_r_seq = _extract_base_uv(p_uvs_r, p_pts_r.reshape(1, p_pts_r.shape[1], 4, 4))
                        _draw_phase_polyline(_overlay_gt, p_l_seq, phase_seq, width=2)
                        _draw_phase_polyline(_overlay_gt, p_r_seq, phase_seq, width=2)
                        _draw_phase_polyline(_overlay_c, p_l_seq, phase_seq, width=1)
                        _draw_phase_polyline(_overlay_c, p_r_seq, phase_seq, width=1)
                        _draw_phase_legend(_overlay_gt)

            # Corrected overlay: keep only correction trajectories + phase-colored GT path.
            _overlay_c[mask] = (0.6 * _overlay_c[mask] + 0.4 * traj_u8[mask]).astype(np.uint8)
            _draw_polyline(_overlay_c, luv, (0, 255, 0))
            _draw_polyline(_overlay_c, ruv, (0, 0, 255))
            _annotate_start_end(_overlay_c, luv, "L", (0, 255, 0))
            _annotate_start_end(_overlay_c, ruv, "R", (0, 0, 255))

            cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_original.png'), _overlay_o)
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
                if '_overlay_o' in locals():
                    cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_original.png'), _overlay_o)
                elif '_oimg' in locals():
                    cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_original.png'), _oimg)
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
                'min_dist_start': (None if _r.get('min_dist_start') is None else float(_r.get('min_dist_start'))),
                'phase_key': _r.get('phase_key'),
                'perturb_mode': _r.get('perturb_mode'),
                'sampled_error_mode': _r.get('sampled_error_mode'),
                'active_left_arm': bool(_r.get('active_left_arm', True)),
                'active_right_arm': bool(_r.get('active_right_arm', True)),
                'active_left_gripper': bool(_r.get('active_left_gripper', True)),
                'active_right_gripper': bool(_r.get('active_right_gripper', True)),
                'active_left_arm_score': (None if _r.get('active_left_arm_score') is None else float(_r.get('active_left_arm_score'))),
                'active_right_arm_score': (None if _r.get('active_right_arm_score') is None else float(_r.get('active_right_arm_score'))),
                'action_perturbed': bool(_r.get('action_perturbed', False)),
                'left_grip': (None if _r.get('left_grip') is None else float(_r.get('left_grip'))),
                'right_grip': (None if _r.get('right_grip') is None else float(_r.get('right_grip'))),
                'reason': _r.get('reason'),
                'evac_video': _video,
                'evac_video_dir': (None if _video is None else os.path.dirname(_video.get('path'))),
            })

        _plan_info = {
            'version': 2,
            'force_generate_correction': bool(force_generate_correction),
            'rollout_steps_total': int(rollout_steps_total),
            'gt_ref_index_for_original': int(np.clip(start_ts + int(rollout_steps_total), 0, raw_data['left_endpose'].shape[0] - 1)),
            'num_rollouts': int(len(_rollout_records)),
            'final_match': {
                'min_dist': float(min_dist),
                'nearest_mode': nearest_mode,
            },
            'correction_plan': {
                'left_active': bool(corr_left_active),
                'right_active': bool(corr_right_active),
                'planner_left_status': res_l.get('status'),
                'planner_right_status': res_r.get('status'),
                'planner_left_len': len(res_l.get('position', [])),
                'planner_right_len': len(res_r.get('position', [])),
                'gripper_left': [float(left_grip), float(tl_grip)],
                'gripper_right': [float(right_grip), float(tr_grip)],
                'correction_prefix_len': int(prefix_len),
                'debug_correction_evac_rollout': bool(debug_correction_evac_rollout),
                'evac_correction_generated_video': evac_corr_video,
            },
        }
        with open(os.path.join(_dbg_corr, 'correction_info.json'), 'w') as _f:
            json.dump(_plan_info, _f, indent=2)

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
            }
            _closed_loop_rollouts.append(_drop_none(_rec))

        _closed_loop_info = {
            'version': 1,
            'start_ts': int(start_ts),
            'sample_pregrasp_segment_start': (None if pregrasp_seg_start is None else int(pregrasp_seg_start)),
            'sample_pregrasp_segment_end': (None if pregrasp_seg_end is None else int(pregrasp_seg_end)),
            'rollout_exec_steps': int(rollout_exec_steps),
            'final_match': {
                't_star': int(t_star),
            },
            'rollouts': _closed_loop_rollouts,
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
        "nearest_mode": nearest_mode,
        "rollout_exec_steps": int(rollout_exec_steps),
        "t_star": int(t_star),
        "min_dist": float(min_dist),
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