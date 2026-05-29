from __future__ import annotations

import os
import re
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch

try:
    from ..failure_utils import (
        active_arm_pattern_id_to_key,
        error_mode_id_to_key,
        phase_id_to_key,
    )
    from .utils_common import resample_trajectory
    from .perturbation import (
        _find_gripper_toggle_anchor_idx,
        _find_next_gripper_toggle_idx_from,
        _infer_active_arms_from_gt_window,
        _infer_phase_key_from_gt_window,
        _traj_dir_after_anchor_with_idx,
        perturb_action_chunk_online,
    )
except ImportError:
    from policy.ACT_LatentCorr.failure_utils import (
        active_arm_pattern_id_to_key,
        error_mode_id_to_key,
        phase_id_to_key,
    )
    from act_aligned_pkg.utils_common import resample_trajectory
    from act_aligned_pkg.perturbation import (
        _find_gripper_toggle_anchor_idx,
        _find_next_gripper_toggle_idx_from,
        _infer_active_arms_from_gt_window,
        _infer_phase_key_from_gt_window,
        _traj_dir_after_anchor_with_idx,
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
    failure_mode_key = str(cfg.get('failure_mode', 'off')).strip().lower()
    min_dist_fallback_force_correction = bool(cfg['min_dist_fallback_force_correction'])
    min_dist_recover_ratio = float(cfg['min_dist_recover_ratio'])
    real_error_trigger_enable = bool(cfg.get('real_error_trigger_enable', True))
    real_error_min_dist_thresh = float(cfg.get('real_error_min_dist_thresh', 0.01))
    real_error_min_dist_delta_thresh = float(cfg.get('real_error_min_dist_delta_thresh', 0.005))
    real_error_trigger_active = bool(real_error_trigger_enable and (not bool(cfg.get('enable_perturb', True))))
    if failure_mode_key in {'train', 'explore'}:
        real_error_trigger_active = False
    debug_recover_eval_rollout = bool(cfg['debug_recover_eval_rollout'])
    debug_correction_evac_rollout = bool(cfg.get('debug_correction_evac_rollout', False))
    recover_eval_enable = bool(cfg.get('recover_eval_enable', False))
    recover_eval_save_video = bool(cfg.get('recover_eval_save_video', False))
    recover_eval_gripper_open_thresh = float(cfg.get('recover_eval_gripper_open_thresh', 0.8))
    recover_eval_pos_thresh_m = float(cfg.get('recover_eval_pos_thresh_m', 0.03))
    recover_eval_rot_thresh_deg = float(cfg.get('recover_eval_rot_thresh_deg', 10.0))
    evac_ddim_steps = int(cfg.get('evac_ddim_steps', 27))
    target_mode = str(cfg['target_mode']).strip().lower()
    target_lookahead_steps = int(cfg['target_lookahead_steps'])
    correction_interp_nearest_enable = bool(cfg['correction_interp_nearest_enable'])
    correction_interp_prefix_ratio = float(np.clip(cfg['correction_interp_prefix_ratio'], 0.0, 1.0))
    correction_planner_prefix_ratio = float(np.clip(cfg.get('correction_planner_prefix_ratio', 0.8), 0.0, 1.0))
    correction_gripper_close_prefix_ratio = float(
        np.clip(cfg.get('correction_gripper_close_prefix_ratio', correction_planner_prefix_ratio), 0.0, 1.0)
    )
    correction_compose_gt_tail_enable = bool(cfg.get('correction_compose_gt_tail_enable', True))
    chunk_size = cfg['chunk_size']
    rollout_exec_steps = int(cfg['rollout_exec_steps'])
    max_action_len = cfg['max_action_len']
    orient_weight = float(cfg['orient_weight'])
    gripper_penalty = float(cfg['gripper_penalty'])
    recover_gripper_penalty = float(cfg['recover_gripper_penalty'])

    left_grip_traj = raw_data['left_gripper'][start_ts:]
    right_grip_traj = raw_data['right_gripper'][start_ts:]
    phase_window_len = int(cfg['sample_pregrasp_phase_window_len'])
    if failure_mode_key in {'train', 'explore'} and sampled_phase_id is not None and int(sampled_phase_id) >= 0:
        phase_key_fixed = phase_id_to_key(int(sampled_phase_id))
    else:
        phase_key_fixed = _infer_phase_key_from_gt_window(
            left_grip_traj=left_grip_traj,
            right_grip_traj=right_grip_traj,
            window_len=phase_window_len,
        )
        if str(cfg['perturb_error_mode']).strip().lower() == "open_laptop_pregrasp":
            phase_key_fixed = "pregrasp"

    forced_error_mode_key = None
    if forced_error_mode_id is not None and int(forced_error_mode_id) >= 0:
        forced_error_mode_key = error_mode_id_to_key(int(forced_error_mode_id))
    sampled_active_arm_pattern_key = None
    if sampled_active_arm_pattern_id is not None and int(sampled_active_arm_pattern_id) >= 0:
        sampled_active_arm_pattern_key = active_arm_pattern_id_to_key(int(sampled_active_arm_pattern_id))
    forced_dir_bin_fixed = (None if forced_dir_bin_id is None else int(forced_dir_bin_id))
    forced_mag_bin_fixed = (None if forced_mag_bin_id is None else int(forced_mag_bin_id))
    sampled_phase_bin_fixed = (None if sampled_phase_bin_id is None else int(sampled_phase_bin_id))
    sampled_phase_instance_fixed = (None if sampled_phase_instance_id is None else int(sampled_phase_instance_id))

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
    evac_recover_eval_videos = []
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
    min_dist_triggered = False
    min_dist_trigger_step = None
    min_dist_trigger_dist = None
    min_dist_trigger_source = None
    min_dist_trigger_delta = None
    pending_recover_valid = False
    pending_recover_added = 0.0
    pending_recover_from_step = None
    pending_recover_mode = None
    perturb_anchor_idx_abs_fixed = None
    # Use pregrasp segment END as the only fixed perturb anchor source.
    # Do not fall back to segment start.
    if pregrasp_seg_end is not None:
        try:
            _pa = int(pregrasp_seg_end)
            if _pa >= 0:
                perturb_anchor_idx_abs_fixed = _pa
        except Exception:
            perturb_anchor_idx_abs_fixed = None

    def _attach_recover_to_rollout(
        target_step, eval_pre, eval_end, gain, thr, triggered=False, reason=None, eval_step=None,
        eval_t_star=None, eval_anchor_idx=None, eval_terminal=False, gripper_recover=None
    ):
        if target_step is None:
            return False
        ts = int(target_step)
        for i in range(len(_dbg_rollout) - 1, -1, -1):
            r = _dbg_rollout[i]
            if isinstance(r, dict) and int(r.get('step', -1)) == ts:
                r['recover_pre_first_dist'] = float(eval_pre)
                r['recover_post_last_dist'] = float(eval_end)
                r['min_dist_recover_gain'] = float(gain)
                r['min_dist_recover_threshold'] = float(thr)
                r['min_dist_triggered'] = bool(triggered)
                if eval_step is not None:
                    r['recover_eval_step'] = int(eval_step)
                if eval_t_star is not None:
                    r['recover_eval_t_star'] = int(eval_t_star)
                if eval_anchor_idx is not None:
                    r['recover_eval_anchor_idx'] = int(eval_anchor_idx)
                r['recover_eval_terminal_pass'] = bool(eval_terminal)
                if isinstance(gripper_recover, dict):
                    r['recover_gripper_mode'] = gripper_recover.get('mode')
                    r['recover_gripper_close_min'] = gripper_recover.get('close_min')
                    r['recover_gripper_open_max'] = gripper_recover.get('open_max')
                    r['recover_gripper_hit_left'] = gripper_recover.get('hit_left')
                    r['recover_gripper_hit_right'] = gripper_recover.get('hit_right')
                    r['recover_gripper_passed'] = gripper_recover.get('passed')
                if reason is not None:
                    r['reason'] = reason
                return True
        return False

    def _nearest_with_window(lp, lq, rp, rq, expected_idx, gp):
        near_w = int(max(1, cfg['nearest_window_radius']))
        expected_idx = int(np.clip(expected_idx, 0, len(left_ep) - 1))
        return find_nearest_traj_point(
            lp, lq, rp, rq, left_ep, right_ep, orient_weight,
            curr_left_grip=left_grip, curr_right_grip=right_grip,
            left_gripper_traj=left_grip_traj, right_gripper_traj=right_grip_traj,
            gripper_penalty=gp,
            window_start=max(0, expected_idx - near_w),
            window_end=min(len(left_ep), expected_idx + near_w + 1),
        )

    anchor_idx = int(np.clip(_find_gripper_toggle_anchor_idx(left_grip_traj, right_grip_traj), 0, len(left_ep) - 1))

    def _dist_to_anchor(lp, lq, rp, rq, curr_lg, curr_rg, gp, ref_idx=None):
        idx_ref = anchor_idx if ref_idx is None else int(np.clip(int(ref_idx), 0, len(left_ep) - 1))
        li = np.asarray(left_ep[idx_ref], dtype=np.float32)
        ri = np.asarray(right_ep[idx_ref], dtype=np.float32)
        left_on = bool(active_info_fixed.get('left_arm', True))
        right_on = bool(active_info_fixed.get('right_arm', True))
        if (not left_on) and (not right_on):
            left_on, right_on = True, True
        d = 0.0
        n_pos = 0
        if left_on:
            d += float(np.linalg.norm(li[:3] - np.asarray(lp, dtype=np.float32)))
            n_pos += 1
        if right_on:
            d += float(np.linalg.norm(ri[:3] - np.asarray(rp, dtype=np.float32)))
            n_pos += 1
        d = d / float(max(1, n_pos))
        if orient_weight > 0:
            d_ori = 0.0
            n_ori = 0
            if left_on:
                ldot = float(np.clip(np.abs(np.dot(li[3:7], np.asarray(lq, dtype=np.float32))), 0.0, 1.0))
                d_ori += 2.0 * np.arccos(ldot)
                n_ori += 1
            if right_on:
                rdot = float(np.clip(np.abs(np.dot(ri[3:7], np.asarray(rq, dtype=np.float32))), 0.0, 1.0))
                d_ori += 2.0 * np.arccos(rdot)
                n_ori += 1
            d += float(orient_weight) * (d_ori / float(max(1, n_ori)))
        if gp > 0 and left_grip_traj is not None and right_grip_traj is not None:
            curr_l_bin = 0.0 if float(curr_lg) <= 0.5 else 1.0
            curr_r_bin = 0.0 if float(curr_rg) <= 0.5 else 1.0
            anc_l_bin = 0.0 if float(left_grip_traj[idx_ref]) <= 0.5 else 1.0
            anc_r_bin = 0.0 if float(right_grip_traj[idx_ref]) <= 0.5 else 1.0
            d_grip = 0.0
            n_grip = 0
            if left_on:
                d_grip += abs(anc_l_bin - curr_l_bin)
                n_grip += 1
            if right_on:
                d_grip += abs(anc_r_bin - curr_r_bin)
                n_grip += 1
            d += float(gp) * (d_grip / float(max(1, n_grip)))
        return float(d)

    if failure_mode_key in {'train', 'explore'} and sampled_active_arm_pattern_key is not None:
        pattern = str(sampled_active_arm_pattern_key).strip().lower()
        if pattern == 'left_only':
            left_side_active, right_side_active = True, False
        elif pattern == 'right_only':
            left_side_active, right_side_active = False, True
        else:
            left_side_active, right_side_active = True, True
        active_info_fixed = {
            'left_arm': bool(left_side_active),
            'right_arm': bool(right_side_active),
            'left_gripper': bool(left_side_active),
            'right_gripper': bool(right_side_active),
            'left_arm_score': 1.0 if left_side_active else 0.0,
            'right_arm_score': 1.0 if right_side_active else 0.0,
            'left_gripper_score': 1.0 if left_side_active else 0.0,
            'right_gripper_score': 1.0 if right_side_active else 0.0,
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

    def _eval_recover_metrics(act_eval_chunk, recover_mode, anchor_dist_ref, anchor_idx_ref):
        recover_mode = str(recover_mode).strip().lower()
        act_eval_chunk = np.asarray(act_eval_chunk, dtype=np.float32)
        if act_eval_chunk.ndim == 1:
            act_eval_chunk = act_eval_chunk[None, :]
        if act_eval_chunk.shape[0] <= 0:
            return 1.0, -1.0, 0.0, True, None
        gripper_recover_info = None
        if recover_mode == "gripper_close":
            # gripper_close perturbation should recover back to open state.
            open_max = float(np.clip(cfg['perturb_gripper_open_max'], 0.0, 1.0))
            gl_seq = np.clip(act_eval_chunk[:, 6], 0.0, 1.0)
            gr_seq = np.clip(act_eval_chunk[:, 13], 0.0, 1.0)
            l_on = bool(active_info_fixed.get('left_gripper', True))
            r_on = bool(active_info_fixed.get('right_gripper', True))
            hit_l = bool(np.any(gl_seq >= open_max)) if l_on else True
            hit_r = bool(np.any(gr_seq >= open_max)) if r_on else True
            recover_failed = bool(not (hit_l and hit_r))
            eval_end = float(1.0 if recover_failed else 0.0)
            gain = float(-eval_end)
            thresh = 0.0
            gripper_recover_info = {
                'mode': recover_mode,
                'close_min': None,
                'open_max': float(open_max),
                'hit_left': (None if not l_on else bool(hit_l)),
                'hit_right': (None if not r_on else bool(hit_r)),
                'passed': bool(not recover_failed),
            }
            return eval_end, gain, thresh, recover_failed, gripper_recover_info
        if recover_mode == "gripper_open":
            # gripper_open perturbation should recover back to close state.
            close_min = float(np.clip(cfg['perturb_gripper_close_min'], 0.0, 1.0))
            gl_seq = np.clip(act_eval_chunk[:, 6], 0.0, 1.0)
            gr_seq = np.clip(act_eval_chunk[:, 13], 0.0, 1.0)
            l_on = bool(active_info_fixed.get('left_gripper', True))
            r_on = bool(active_info_fixed.get('right_gripper', True))
            hit_l = bool(np.any(gl_seq <= close_min)) if l_on else True
            hit_r = bool(np.any(gr_seq <= close_min)) if r_on else True
            recover_failed = bool(not (hit_l and hit_r))
            eval_end = float(1.0 if recover_failed else 0.0)
            gain = float(-eval_end)
            thresh = 0.0
            gripper_recover_info = {
                'mode': recover_mode,
                'close_min': float(close_min),
                'open_max': None,
                'hit_left': (None if not l_on else bool(hit_l)),
                'hit_right': (None if not r_on else bool(hit_r)),
                'passed': bool(not recover_failed),
            }
            return eval_end, gain, thresh, recover_failed, gripper_recover_info

        q_eval_end = np.asarray(act_eval_chunk[-1], dtype=np.float32)
        fk_eval = fk.forward(q_eval_end[0:6], q_eval_end[7:13])
        lp_ev, lq_ev = fk_eval['left']
        rp_ev, rq_ev = fk_eval['right']
        eval_end = _dist_to_anchor(
            lp_ev, lq_ev, rp_ev, rq_ev, q_eval_end[6], q_eval_end[13], recover_gripper_penalty, ref_idx=anchor_idx_ref
        )
        gain = float(anchor_dist_ref - float(eval_end))
        thresh = float(max(0.0, pending_recover_added) * max(0.0, min_dist_recover_ratio))
        recover_failed = bool(gain < float(thresh))
        return eval_end, gain, thresh, recover_failed, None

    max_steps_eff = max_steps

    for step in range(max_steps_eff):
        step_role = 'recover_eval+perturb' if pending_recover_valid else 'perturb'
        recovery_pair_step = pending_recover_from_step
        evac_debug_dir = os.path.join(debug_dir, 'evac', f'rollout_step_{step:03d}') if debug_dir else None
        left_q, right_q = curr_qpos_raw[0:6], curr_qpos_raw[7:13]
        left_grip, right_grip = curr_qpos_raw[6], curr_qpos_raw[13]
        fk_r = fk.forward(left_q, right_q)
        lp, lq = fk_r['left']
        rp, rq = fk_r['right']
        expected_idx = int(np.clip(rollout_steps_total, 0, len(left_ep) - 1))
        t_star, min_dist = _nearest_with_window(lp, lq, rp, rq, expected_idx, gripper_penalty)
        min_dist_start = float(min_dist)
        t_star_start = int(t_star)
        # Perturbation uses fixed approach->pregrasp anchor; recovery eval keeps dynamic anchor.
        if perturb_anchor_idx_abs_fixed is not None:
            perturb_anchor_idx_step = int(np.clip(perturb_anchor_idx_abs_fixed - int(start_ts), 0, len(left_ep) - 1))
        else:
            perturb_anchor_idx_step = int(np.clip(anchor_idx, 0, len(left_ep) - 1))
        anchor_idx_step = int(np.clip(
            _find_next_gripper_toggle_idx_from(left_grip_traj, right_grip_traj, t_star_start),
            0, len(left_ep) - 1
        ))
        anchor_dist_start = _dist_to_anchor(
            lp, lq, rp, rq, left_grip, right_grip, recover_gripper_penalty, ref_idx=anchor_idx_step
        )
        min_dist_end = None
        anchor_dist_end = None
        min_dist_delta = None
        min_dist_recover_gain = None
        min_dist_recover_threshold = None
        min_dist_recover_eval_end = None
        if t_star >= len(left_ep) - 2:
            if debug_dir is not None:
                import json
                _dbg_corr = os.path.join(debug_dir, 'correction')
                os.makedirs(_dbg_corr, exist_ok=True)
                with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                    json.dump({'reason': 'near_end_of_trajectory',
                               't_star': int(t_star), 'traj_len': len(left_ep),
                               'min_dist': float(min_dist), 'step': step,
                               'rollout': _dbg_rollout}, _f, indent=2)
            return None

        if step == 0 and precomputed_action_chunk_raw is not None:
            act_raw = np.asarray(precomputed_action_chunk_raw, dtype=np.float32)
            if act_raw.ndim == 3:
                if act_raw.shape[0] != 1:
                    raise ValueError(
                        f"precomputed_action_chunk_raw batch dim must be 1, got {act_raw.shape}"
                    )
                act_raw = act_raw[0]
            if act_raw.ndim != 2 or act_raw.shape[1] != 14:
                raise ValueError(
                    f"precomputed_action_chunk_raw must be [T,14], got {act_raw.shape}"
                )
            act_raw = act_raw.copy()
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
        act_pred_raw = act_raw.copy()
        q_raw_end = np.asarray(act_pred_raw[-1], dtype=np.float32)
        fk_raw_end = fk.forward(q_raw_end[0:6], q_raw_end[7:13])
        perturb_pre_last_dist = _dist_to_anchor(
            fk_raw_end['left'][0], fk_raw_end['left'][1],
            fk_raw_end['right'][0], fk_raw_end['right'][1],
            q_raw_end[6], q_raw_end[13], recover_gripper_penalty, ref_idx=anchor_idx_step
        )

        # Recovery is evaluated by THIS step's raw (non-perturbed) VLA action, for the
        # previous perturb step. After evaluation passes, this step still gets perturbed
        # and executed to generate the next perturb state.
        if pending_recover_valid:
            recover_mode = str(pending_recover_mode).strip().lower()
            min_dist_recover_eval_end, min_dist_recover_gain, min_dist_recover_threshold, recover_failed, gripper_recover_info = (
                _eval_recover_metrics(act_pred_raw, recover_mode, anchor_dist_start, anchor_idx_step)
            )
            if debug_recover_eval_rollout and evac_debug_dir is not None:
                fk_poses_eval = [(lp.copy(), lq.copy(), rp.copy(), rq.copy())]
                grip_list_eval = [(left_grip, right_grip)]
                for ai in range(len(act_pred_raw)):
                    ar = act_pred_raw[ai]
                    fr = fk.forward(ar[0:6], ar[7:13])
                    fk_poses_eval.append((fr['left'][0].copy(), fr['left'][1].copy(),
                                          fr['right'][0].copy(), fr['right'][1].copy()))
                    grip_list_eval.append((ar[6], ar[13]))
                evac_eval_debug_dir = os.path.join(evac_debug_dir, 'recover_eval_raw')
                _ = evac_inference(
                    evac_model,
                    evac_cfg,
                    curr_image[0],
                    fk_poses_eval,
                    grip_list_eval,
                    raw_data,
                    device,
                    save_dir=evac_eval_debug_dir,
                    ddim_steps=evac_ddim_steps,
                    infer_kwargs=cfg['evac_infer_kwargs'],
                )
                _vpath_eval = os.path.join(evac_eval_debug_dir, 'outputs.mp4')
                evac_recover_eval_videos.append({
                    'step': (int(recovery_pair_step) if recovery_pair_step is not None else int(step)),
                    'path': _vpath_eval,
                    'exists': bool(os.path.exists(_vpath_eval)),
                })
            _attach_recover_to_rollout(
                recovery_pair_step,
                anchor_dist_start,
                min_dist_recover_eval_end,
                min_dist_recover_gain,
                min_dist_recover_threshold,
                triggered=False,
                eval_step=step,
                eval_t_star=t_star_start,
                eval_anchor_idx=anchor_idx_step,
                eval_terminal=False,
                gripper_recover=gripper_recover_info,
            )
            if recover_failed:
                min_dist_triggered = True
                min_dist_trigger_step = (int(recovery_pair_step) if recovery_pair_step is not None else int(step))
                if recover_mode in {"gripper_close", "gripper_open"}:
                    min_dist_trigger_dist = float(min_dist_recover_eval_end)
                else:
                    min_dist_trigger_dist = float(min_dist_recover_gain)
                min_dist_trigger_delta = None
                min_dist_trigger_source = 'min_dist_recovery'
                if debug_dir is not None:
                    _attach_recover_to_rollout(
                        recovery_pair_step,
                        anchor_dist_start,
                        min_dist_recover_eval_end,
                        min_dist_recover_gain,
                        min_dist_recover_threshold,
                        triggered=True,
                        reason='min_dist_recovery_insufficient',
                        eval_step=step,
                        eval_t_star=t_star_start,
                        eval_anchor_idx=anchor_idx_step,
                        eval_terminal=False,
                        gripper_recover=gripper_recover_info,
                    )
                break
            # Recovery belongs to previous perturb step; do not duplicate on current step record.
            min_dist_recover_gain = None
            min_dist_recover_threshold = None
            min_dist_recover_eval_end = None
            pending_recover_valid = False
            pending_recover_added = 0.0
            pending_recover_from_step = None
            pending_recover_mode = None

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
            forced_dir_bin_id=(-1 if forced_dir_bin_fixed is None else int(forced_dir_bin_fixed)),
            forced_mag_bin_id=(-1 if forced_mag_bin_fixed is None else int(forced_mag_bin_fixed)),
            perturb_anchor_idx=int(perturb_anchor_idx_step),
            left_ep_full=raw_data.get('left_endpose'),
            right_ep_full=raw_data.get('right_endpose'),
            perturb_anchor_idx_abs=perturb_anchor_idx_abs_fixed,
            start_ts_offset=int(start_ts),
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
            ddim_steps=evac_ddim_steps,
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
        t_star_e, min_dist_end = _nearest_with_window(
            lp_e, lq_e, rp_e, rq_e,
            int(np.clip(rollout_steps_total, 0, len(left_ep) - 1)),
            gripper_penalty,
        )
        anchor_dist_end = _dist_to_anchor(
            lp_e, lq_e, rp_e, rq_e, left_grip, right_grip, recover_gripper_penalty, ref_idx=anchor_idx_step
        )
        min_dist_delta = float(anchor_dist_end - anchor_dist_start)
        # Use end-of-rollout nearest as latest anchor for downstream phase/targeting.
        t_star = int(t_star_e)
        min_dist = float(min_dist_end)
        # Record added error from this perturb execution; next step will evaluate recovery.
        pending_recover_valid = True
        pending_recover_added = float(max(0.0, min_dist_delta))
        pending_recover_from_step = int(step)
        _pm = str(pert_mode).lower()
        if "gripper_close" in _pm:
            pending_recover_mode = "gripper_close"
        elif "gripper_open" in _pm:
            pending_recover_mode = "gripper_open"
        else:
            pending_recover_mode = None

        sampled_error_mode = dyn_state.get('error_mode') if isinstance(dyn_state, dict) else None
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

                if recover_eval_save_video and (debug_dir is not None):
                    try:
                        bridge_steps = int(max(1, cfg.get('recover_eval_video_bridge_steps', 16)))
                        act_eval_vis_raw = np.asarray(act_eval_raw, dtype=np.float32).copy()
                        bridge_meta = {
                            'bridge_steps': int(bridge_steps),
                            'left_status': 'Inactive',
                            'right_status': 'Inactive',
                        }
                        if act_eval_vis_raw.shape[0] > 0:
                            target0 = np.asarray(act_eval_vis_raw[0], dtype=np.float32)
                            prefix = np.repeat(target0[None, :], bridge_steps, axis=0).astype(np.float32)
                            prefix[:, 0:6] = np.repeat(np.asarray(left_q_e, dtype=np.float32)[None, :], bridge_steps, axis=0)
                            prefix[:, 7:13] = np.repeat(np.asarray(right_q_e, dtype=np.float32)[None, :], bridge_steps, axis=0)

                            if bool(active_info.get('left_gripper', True)):
                                prefix[:, 6] = np.linspace(float(left_grip), float(target0[6]), bridge_steps, dtype=np.float32)
                            else:
                                prefix[:, 6] = float(left_grip)
                            if bool(active_info.get('right_gripper', True)):
                                prefix[:, 13] = np.linspace(float(right_grip), float(target0[13]), bridge_steps, dtype=np.float32)
                            else:
                                prefix[:, 13] = float(right_grip)

                            qpos_full = np.zeros(len(fk.jnames), dtype=np.float32)
                            qpos_full[fk.fl_idx] = np.asarray(left_q_e, dtype=np.float32)
                            qpos_full[fk.fr_idx] = np.asarray(right_q_e, dtype=np.float32)
                            import sapien

                            fr_tgt = fk.forward(target0[0:6], target0[7:13])
                            if bool(active_info.get('left_arm', True)):
                                pose_l = sapien.Pose(
                                    np.asarray(fr_tgt['left'][0], dtype=np.float32),
                                    np.asarray(fr_tgt['left'][1], dtype=np.float32),
                                )
                                res_l = planner_l.plan_path(qpos_full, pose_l, arms_tag='left')
                                bridge_meta['left_status'] = str(res_l.get('status'))
                                if str(res_l.get('status')) == 'Success':
                                    path_l = np.asarray(res_l.get('position', []), dtype=np.float32)
                                    if path_l.ndim == 2 and path_l.shape[0] > 0:
                                        future_l = path_l[1:] if path_l.shape[0] > 1 else path_l
                                        prefix[:, 0:6] = resample_trajectory(future_l, bridge_steps).astype(np.float32)
                            if bool(active_info.get('right_arm', True)):
                                pose_r = sapien.Pose(
                                    np.asarray(fr_tgt['right'][0], dtype=np.float32),
                                    np.asarray(fr_tgt['right'][1], dtype=np.float32),
                                )
                                res_r = planner_r.plan_path(qpos_full, pose_r, arms_tag='right')
                                bridge_meta['right_status'] = str(res_r.get('status'))
                                if str(res_r.get('status')) == 'Success':
                                    path_r = np.asarray(res_r.get('position', []), dtype=np.float32)
                                    if path_r.ndim == 2 and path_r.shape[0] > 0:
                                        future_r = path_r[1:] if path_r.shape[0] > 1 else path_r
                                        prefix[:, 7:13] = resample_trajectory(future_r, bridge_steps).astype(np.float32)
                            act_eval_vis_raw = np.concatenate([prefix, act_eval_vis_raw], axis=0).astype(np.float32)

                        fk_eval_poses = [(lp_e.copy(), lq_e.copy(), rp_e.copy(), rq_e.copy())]
                        grip_eval_list = [(float(left_grip), float(right_grip))]
                        for ai in range(act_eval_vis_raw.shape[0]):
                            ar = act_eval_vis_raw[ai]
                            fr = fk.forward(ar[0:6], ar[7:13])
                            fk_eval_poses.append((
                                fr['left'][0].copy(), fr['left'][1].copy(),
                                fr['right'][0].copy(), fr['right'][1].copy(),
                            ))
                            grip_eval_list.append((float(ar[6]), float(ar[13])))
                        evac_recover_eval_dir = os.path.join(debug_dir, 'evac_recover_eval', f'rollout_step_{step:03d}')
                        _ = evac_inference(
                            evac_model,
                            evac_cfg,
                            curr_image[0],
                            fk_eval_poses,
                            grip_eval_list,
                            raw_data,
                            device,
                            save_dir=evac_recover_eval_dir,
                            ddim_steps=evac_ddim_steps,
                            infer_kwargs=cfg.get('evac_infer_kwargs'),
                        )
                        vpath_recover = os.path.join(evac_recover_eval_dir, 'outputs.mp4')
                        recover_eval_video = {
                            'path': vpath_recover,
                            'exists': bool(os.path.exists(vpath_recover)),
                            'bridge_steps': int(bridge_meta.get('bridge_steps', 0)),
                            'bridge_left_status': bridge_meta.get('left_status'),
                            'bridge_right_status': bridge_meta.get('right_status'),
                        }
                    except Exception as exc_recover_video:
                        recover_eval_video = {
                            'path': None,
                            'exists': False,
                            'error': str(exc_recover_video),
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
                        recover_eval_metric is not None and float(recover_eval_metric) >= float(recover_eval_threshold)
                    )
                elif mode_eval == "translation":
                    near_w_eval = int(max(1, cfg.get('recover_eval_nearest_window_radius', int(max(1, rollout_exec_steps)))))
                    ws = int(np.clip(gt_ref_idx, 0, raw_data['left_endpose'].shape[0] - 1))
                    we = int(np.clip(gt_ref_idx + near_w_eval + 1, ws + 1, raw_data['left_endpose'].shape[0]))
                    nidx_abs, ndist = find_nearest_traj_point(
                        np.asarray(fr_eval_last['left'][0], dtype=np.float32),
                        np.asarray(fr_eval_last['left'][1], dtype=np.float32),
                        np.asarray(fr_eval_last['right'][0], dtype=np.float32),
                        np.asarray(fr_eval_last['right'][1], dtype=np.float32),
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
                            np.asarray(fr_eval_last['left'][0], dtype=np.float32)
                            - np.asarray(raw_data['left_endpose'][recover_eval_gt_ref_idx, :3], dtype=np.float32)
                        )))
                    if bool(active_info.get('right_arm', True)):
                        pos_errs.append(float(np.linalg.norm(
                            np.asarray(fr_eval_last['right'][0], dtype=np.float32)
                            - np.asarray(raw_data['right_endpose'][recover_eval_gt_ref_idx, :3], dtype=np.float32)
                        )))
                    recover_eval_metric_name = "pos_err_m"
                    recover_eval_metric = (None if len(pos_errs) == 0 else float(np.mean(pos_errs)))
                    recover_eval_threshold = float(recover_eval_pos_thresh_m)
                    recover_eval_recoverable = (
                        recover_eval_metric is not None and float(recover_eval_metric) <= float(recover_eval_threshold)
                    )
                elif mode_eval == "rotation":
                    near_w_eval = int(max(1, cfg.get('recover_eval_nearest_window_radius', int(max(1, rollout_exec_steps)))))
                    ws = int(np.clip(gt_ref_idx, 0, raw_data['left_endpose'].shape[0] - 1))
                    we = int(np.clip(gt_ref_idx + near_w_eval + 1, ws + 1, raw_data['left_endpose'].shape[0]))
                    nidx_abs, ndist = find_nearest_traj_point(
                        np.asarray(fr_eval_last['left'][0], dtype=np.float32),
                        np.asarray(fr_eval_last['left'][1], dtype=np.float32),
                        np.asarray(fr_eval_last['right'][0], dtype=np.float32),
                        np.asarray(fr_eval_last['right'][1], dtype=np.float32),
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
                            fr_eval_last['left'][1],
                            raw_data['left_endpose'][recover_eval_gt_ref_idx, 3:7],
                        ))
                    if bool(active_info.get('right_arm', True)):
                        rot_errs.append(_quat_geodesic_deg_wxyz(
                            fr_eval_last['right'][1],
                            raw_data['right_endpose'][recover_eval_gt_ref_idx, 3:7],
                        ))
                    recover_eval_metric_name = "rot_err_deg"
                    recover_eval_metric = (None if len(rot_errs) == 0 else float(np.mean(rot_errs)))
                    recover_eval_threshold = float(recover_eval_rot_thresh_deg)
                    recover_eval_recoverable = (
                        recover_eval_metric is not None and float(recover_eval_metric) <= float(recover_eval_threshold)
                    )
                else:
                    recover_eval_recoverable = None
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
            }

        # collect debug info for this rollout step
        if debug_dir is not None:
            eef_solved_cnt = None
            eef_total_cnt = None
            eef_solved_ratio = None
            if isinstance(dyn_state, dict):
                eef_solved_cnt = dyn_state.get('eef_solved_cnt')
                eef_total_cnt = dyn_state.get('eef_total_cnt')
                eef_solved_ratio = dyn_state.get('eef_solved_ratio')
            _dbg_rollout.append({
                'step': step, 't_star': int(t_star), 'min_dist': float(min_dist),
                'min_dist_start': float(min_dist_start),
                'min_dist_end': (None if min_dist_end is None else float(min_dist_end)),
                'min_dist_delta': (None if min_dist_delta is None else float(min_dist_delta)),
                'step_role': step_role,
                'paired_perturb_step': (None if recovery_pair_step is None else int(recovery_pair_step)),
                'perturb_pre_last_dist': (None if perturb_pre_last_dist is None else float(perturb_pre_last_dist)),
                'perturb_post_last_dist': (None if anchor_dist_end is None else float(anchor_dist_end)),
                'recover_pre_first_dist': None,
                'recover_post_last_dist': None,
                'anchor_idx': int(anchor_idx_step),
                'progress': float(progress), 'phase_key': phase_key,
                'active_left_arm': bool(active_info.get('left_arm', True)),
                'active_right_arm': bool(active_info.get('right_arm', True)),
                'active_left_gripper': bool(active_info.get('left_gripper', True)),
                'active_right_gripper': bool(active_info.get('right_gripper', True)),
                'active_left_arm_score': float(active_info.get('left_arm_score', 0.0)),
                'active_right_arm_score': float(active_info.get('right_arm_score', 0.0)),
                'active_left_gripper_score': float(active_info.get('left_gripper_score', 0.0)),
                'active_right_gripper_score': float(active_info.get('right_gripper_score', 0.0)),
                'fk_left_pos': lp.tolist(), 'fk_right_pos': rp.tolist(),
                'left_grip': float(left_grip), 'right_grip': float(right_grip),
                'action_perturbed': bool(perturbed),
                'perturb_mode': pert_mode,
                'sampled_error_mode': sampled_error_mode,
                'anti_gt_cos_thresh': (None if not isinstance(dyn_state, dict) else dyn_state.get('anti_gt_cos_thresh')),
                'anti_gt_left_cos': (None if not isinstance(dyn_state, dict) else dyn_state.get('anti_gt_left_cos')),
                'anti_gt_right_cos': (None if not isinstance(dyn_state, dict) else dyn_state.get('anti_gt_right_cos')),
                'anti_gt_left_flipped': (False if not isinstance(dyn_state, dict) else bool(dyn_state.get('anti_gt_left_flipped', False))),
                'anti_gt_right_flipped': (False if not isinstance(dyn_state, dict) else bool(dyn_state.get('anti_gt_right_flipped', False))),
                'reject_sampling_enabled': (False if not isinstance(dyn_state, dict) else bool(dyn_state.get('reject_sampling_enabled', False))),
                'reject_sampling_trial': (None if not isinstance(dyn_state, dict) else dyn_state.get('reject_sampling_trial')),
                'reject_sampling_trials_max': (None if not isinstance(dyn_state, dict) else dyn_state.get('reject_sampling_trials_max')),
                'reject_sampling_base_err': (None if not isinstance(dyn_state, dict) else dyn_state.get('reject_sampling_base_err')),
                'reject_sampling_cand_err': (None if not isinstance(dyn_state, dict) else dyn_state.get('reject_sampling_cand_err')),
                'reject_sampling_min_delta': (None if not isinstance(dyn_state, dict) else dyn_state.get('reject_sampling_min_delta')),
                'reject_sampling_accepted': (None if not isinstance(dyn_state, dict) else dyn_state.get('reject_sampling_accepted')),
                'reject_sampling_fallback_best': (False if not isinstance(dyn_state, dict) else bool(dyn_state.get('reject_sampling_fallback_best', False))),
                'eef_ik_solved_cnt': eef_solved_cnt,
                'eef_ik_total_cnt': eef_total_cnt,
                'eef_ik_solved_ratio': eef_solved_ratio,
                'act_chunk_raw_first': act_raw[0].tolist(),
                'act_chunk_raw_last': act_raw[-1].tolist(),
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
            })

        if (not min_dist_triggered) and real_error_trigger_active:
            real_error_min_dist_delta = float(max(0.0, float(min_dist_end) - float(min_dist_start)))
            real_error_abs_ok = bool(float(min_dist_end) >= real_error_min_dist_thresh)
            real_error_delta_ok = bool(real_error_min_dist_delta >= real_error_min_dist_delta_thresh)
            if debug_dir is not None and len(_dbg_rollout) > 0:
                _dbg_rollout[-1]['real_error_min_dist_thresh'] = float(real_error_min_dist_thresh)
                _dbg_rollout[-1]['real_error_min_dist_delta_thresh'] = float(real_error_min_dist_delta_thresh)
                _dbg_rollout[-1]['real_error_min_dist_delta'] = float(real_error_min_dist_delta)
                _dbg_rollout[-1]['real_error_abs_ok'] = bool(real_error_abs_ok)
                _dbg_rollout[-1]['real_error_delta_ok'] = bool(real_error_delta_ok)
            if real_error_abs_ok and real_error_delta_ok:
                min_dist_triggered = True
                min_dist_trigger_step = int(step)
                min_dist_trigger_dist = float(min_dist_end)
                min_dist_trigger_delta = float(real_error_min_dist_delta)
                min_dist_trigger_source = 'real_error_min_dist'
                if debug_dir is not None and len(_dbg_rollout) > 0:
                    _dbg_rollout[-1]['min_dist_triggered'] = True
                    _dbg_rollout[-1]['reason'] = 'real_error_min_dist_threshold'
                break

    # Terminal recover-eval pass:
    # if the last perturb step has no following rollout step, run one extra
    # non-perturbed VLA eval and attach recover metrics back to that step.
    if pending_recover_valid and (not min_dist_triggered):
        left_q_t, right_q_t = curr_qpos_raw[0:6], curr_qpos_raw[7:13]
        left_grip_t, right_grip_t = curr_qpos_raw[6], curr_qpos_raw[13]
        fk_t = fk.forward(left_q_t, right_q_t)
        lp_t, lq_t = fk_t['left']
        rp_t, rq_t = fk_t['right']
        t_star_t, _ = _nearest_with_window(
            lp_t, lq_t, rp_t, rq_t,
            int(np.clip(rollout_steps_total, 0, len(left_ep) - 1)),
            gripper_penalty,
        )
        anchor_idx_t = int(np.clip(
            _find_next_gripper_toggle_idx_from(left_grip_traj, right_grip_traj, int(t_star_t)),
            0, len(left_ep) - 1
        ))
        eval_pre_t = _dist_to_anchor(
            lp_t, lq_t, rp_t, rq_t, left_grip_t, right_grip_t, recover_gripper_penalty, ref_idx=anchor_idx_t
        )

        qn_t = (curr_qpos_raw - norm_stats['qpos_mean']) / norm_stats['qpos_std']
        qt_t = torch.from_numpy(qn_t).float().unsqueeze(0).to(device)
        it_t = curr_image.unsqueeze(0).to(device)
        _was_training_t = policy_unwrapped.training
        policy_unwrapped.eval()
        with torch.no_grad():
            act_chunk_t = policy_unwrapped(qt_t, it_t)
        if _was_training_t:
            policy_unwrapped.train()
        act_np_t = act_chunk_t.squeeze(0).cpu().numpy()
        act_raw_t = act_np_t * norm_stats['action_std'] + norm_stats['action_mean']
        exec_len_t = int(np.clip(rollout_exec_steps, 1, max(1, act_raw_t.shape[0])))
        act_raw_t = act_raw_t[:exec_len_t].copy()
        recover_mode_t = str(pending_recover_mode).strip().lower()
        eval_end_t, gain_t, thresh_t, _, gripper_recover_info_t = _eval_recover_metrics(
            act_raw_t, recover_mode_t, eval_pre_t, anchor_idx_t
        )

        if debug_recover_eval_rollout and debug_dir is not None:
            evac_eval_debug_dir_t = os.path.join(debug_dir, 'evac', 'rollout_step_terminal', 'recover_eval_raw')
            fk_poses_eval_t = [(lp_t.copy(), lq_t.copy(), rp_t.copy(), rq_t.copy())]
            grip_list_eval_t = [(left_grip_t, right_grip_t)]
            for ai in range(len(act_raw_t)):
                ar = act_raw_t[ai]
                fr = fk.forward(ar[0:6], ar[7:13])
                fk_poses_eval_t.append((fr['left'][0].copy(), fr['left'][1].copy(),
                                        fr['right'][0].copy(), fr['right'][1].copy()))
                grip_list_eval_t.append((ar[6], ar[13]))
            _ = evac_inference(
                evac_model,
                evac_cfg,
                curr_image[0],
                fk_poses_eval_t,
                grip_list_eval_t,
                raw_data,
                device,
                save_dir=evac_eval_debug_dir_t,
                ddim_steps=evac_ddim_steps,
                infer_kwargs=cfg['evac_infer_kwargs'],
            )
            _vpath_eval_t = os.path.join(evac_eval_debug_dir_t, 'outputs.mp4')
            evac_recover_eval_videos.append({
                'step': (int(pending_recover_from_step) if pending_recover_from_step is not None else None),
                'path': _vpath_eval_t,
                'exists': bool(os.path.exists(_vpath_eval_t)),
            })

        _attach_recover_to_rollout(
            pending_recover_from_step,
            eval_pre_t,
            eval_end_t,
            gain_t,
            thresh_t,
            eval_t_star=t_star_t,
            eval_anchor_idx=anchor_idx_t,
            eval_terminal=True,
            gripper_recover=gripper_recover_info_t,
        )
        pending_recover_valid = False
        pending_recover_added = 0.0
        pending_recover_from_step = None
        pending_recover_mode = None

    force_generate_correction = False
    if failure_mode_key == 'train':
        force_generate_correction = True
        if debug_dir is not None:
            _dbg_rollout.append({
                'step': int(max_steps_eff),
                'min_dist_triggered': bool(min_dist_triggered),
                'reason': 'failure_mode_train_force_generate',
                'max_rollout_steps': int(max_steps),
            })
    elif min_dist_fallback_force_correction and (not min_dist_triggered):
        force_generate_correction = True
        if debug_dir is not None:
            _dbg_rollout.append({
                'step': int(max_steps_eff),
                'min_dist_triggered': bool(min_dist_triggered),
                'reason': 'min_dist_fallback_force_correction',
                'max_rollout_steps': int(max_steps),
            })
    elif min_dist_triggered:
        # In min_dist mode, use in-loop trigger as the single source of truth.
        force_generate_correction = True

    # Recompute nearest-point metrics at the final post-rollout state.
    # For pure gripper error modes, disable gripper penalty during nearest-point
    # matching to avoid being pulled to distant segments with similar gripper state.
    nearest_gripper_penalty = float(gripper_penalty)
    nearest_mode = ""
    if isinstance(dyn_state, dict):
        nearest_mode = str(dyn_state.get("error_mode", "")).strip().lower()
        if nearest_mode == "gripper_close":
            nearest_gripper_penalty = 0.0

    left_q, right_q = curr_qpos_raw[0:6], curr_qpos_raw[7:13]
    left_grip, right_grip = curr_qpos_raw[6], curr_qpos_raw[13]
    fk_r = fk.forward(left_q, right_q)
    lp, lq = fk_r['left']
    rp, rq = fk_r['right']
    t_star, min_dist = find_nearest_traj_point(
        lp, lq, rp, rq, left_ep, right_ep, orient_weight,
        curr_left_grip=left_grip, curr_right_grip=right_grip,
        left_gripper_traj=left_grip_traj, right_gripper_traj=right_grip_traj,
        gripper_penalty=nearest_gripper_penalty,
        window_start=max(0, int(rollout_steps_total) - int(max(1, cfg['nearest_window_radius']))),
        window_end=min(len(left_ep), int(rollout_steps_total) + int(max(1, cfg['nearest_window_radius'])) + 1))

    corr_left_active = bool(active_info_fixed['left_arm'])
    corr_right_active = bool(active_info_fixed['right_arm'])

    if not force_generate_correction:
        # debug: save rollout info even on skip
        if debug_dir is not None:
            import json
            _dbg_corr = os.path.join(debug_dir, 'correction')
            os.makedirs(_dbg_corr, exist_ok=True)
            with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                if real_error_trigger_active:
                    _skip_reason = 'real_error_not_triggered'
                    _threshold = {
                        'real_error_min_dist_thresh': float(real_error_min_dist_thresh),
                        'real_error_min_dist_delta_thresh': float(real_error_min_dist_delta_thresh),
                    }
                else:
                    _skip_reason = 'min_dist_delta_not_triggered'
                    _threshold = {
                        'min_dist_recover_ratio': float(min_dist_recover_ratio),
                    }
                json.dump({'reason': _skip_reason, 'min_dist': float(min_dist),
                           'threshold': _threshold,
                           'min_dist_triggered': bool(min_dist_triggered),
                           'min_dist_trigger_step': (None if min_dist_trigger_step is None else int(min_dist_trigger_step)),
                           'min_dist_trigger_dist': (None if min_dist_trigger_dist is None else float(min_dist_trigger_dist)),
                           'min_dist_trigger_source': min_dist_trigger_source,
                           'min_dist_trigger_delta': (None if min_dist_trigger_delta is None else float(min_dist_trigger_delta)),
                           'rollout': _dbg_rollout}, _f, indent=2)
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
            "min_dist_triggered": bool(min_dist_triggered),
            "closed_loop_fallback_used": False,
            "failure_mode": failure_mode_key,
            "sampled_phase_key": str(phase_key_fixed),
            "sampled_phase_bin_id": sampled_phase_bin_fixed,
            "sampled_phase_instance_id": sampled_phase_instance_fixed,
            "forced_error_mode": forced_error_mode_key,
            "sampled_active_arm_pattern": sampled_active_arm_pattern_key,
            "forced_dir_bin_id": forced_dir_bin_fixed,
            "forced_mag_bin_id": forced_mag_bin_fixed,
            "sampled_mode_prob": (
                None if sampled_mode_prob is None or (not np.isfinite(float(sampled_mode_prob)))
                else float(sampled_mode_prob)
            ),
            "sampled_entry_prob_within_mode": (
                None
                if sampled_entry_prob_within_mode is None or (not np.isfinite(float(sampled_entry_prob_within_mode)))
                else float(sampled_entry_prob_within_mode)
            ),
            "sampled_unit_prob": (
                None if sampled_unit_prob is None or (not np.isfinite(float(sampled_unit_prob)))
                else float(sampled_unit_prob)
            ),
            "correction_branch": correction_branch,
            "sampled_error_mode": sampled_error_mode,
            "nearest_mode": nearest_mode,
            "target_mode": target_mode,
            "rollout_exec_steps": int(rollout_exec_steps),
            "t_star": int(t_star),
            "t_target": None,
            "min_dist": float(min_dist),
            "recover_eval_enable": bool(recover_eval_enable),
            "recover_eval_any_unrecoverable": bool(recover_eval_any_unrecoverable),
            "recover_eval_first_unrecoverable_step": (
                None if recover_eval_first_unrecoverable_step is None else int(recover_eval_first_unrecoverable_step)
            ),
            "recover_eval_last": recover_eval_last,
            "trigger_mode": ("real_error" if real_error_trigger_active else "min_dist_recovery"),
            "trigger_source": min_dist_trigger_source,
            "min_dist_trigger_step": (None if min_dist_trigger_step is None else int(min_dist_trigger_step)),
            "min_dist_trigger_dist": (None if min_dist_trigger_dist is None else float(min_dist_trigger_dist)),
            "min_dist_trigger_delta": (None if min_dist_trigger_delta is None else float(min_dist_trigger_delta)),
            "error_action_prefix_raw": error_action_prefix_raw,
        }
        return (None, None, None, None, corr_meta)

    if nearest_mode == "gripper_close":
        # For early-close timing error, follow rollout time directly
        # instead of nearest-point matching.
        t_target = int(np.clip(rollout_steps_total, 0, len(left_ep) - 1))
    elif correction_interp_nearest_enable:
        t_target = int(np.clip(t_star, 0, len(left_ep) - 1))
    else:
        if target_mode == 'backward':
            t_target = int(np.clip(t_star - target_lookahead_steps, 0, len(left_ep) - 1))
        else:
            t_target = int(np.clip(t_star + target_lookahead_steps, 0, len(left_ep) - 1))

    gt_left_arm = raw_data.get('gt_left_arm', None)
    gt_right_arm = raw_data.get('gt_right_arm', None)

    res_l = {'status': 'SkippedInactive', 'position': np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], 1, axis=0)}
    res_r = {'status': 'SkippedInactive', 'position': np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], 1, axis=0)}
    gripper_close_suffix_len_dbg = None
    gripper_close_uniformized_points_dbg = None
    gripper_close_uniformized_spans_dbg = None
    if correction_interp_nearest_enable:
        prefix_len = int(np.clip(np.floor(chunk_size * correction_interp_prefix_ratio), 1, max(1, chunk_size - 1)))

        def _interp_prefix(curr, target, n):
            curr = np.asarray(curr, dtype=np.float32)
            target = np.asarray(target, dtype=np.float32)
            if n <= 1:
                return target[None, :].astype(np.float32)
            alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            return ((1.0 - alpha) * curr[None, :] + alpha * target[None, :]).astype(np.float32)

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

        suffix_len = int(max(0, chunk_size - prefix_len))
        tgt_abs = int(start_ts + t_target)
        l_tgt_q = np.asarray(left_q, dtype=np.float32) if gt_left_arm is None else np.asarray(gt_left_arm[tgt_abs], dtype=np.float32)
        r_tgt_q = np.asarray(right_q, dtype=np.float32) if gt_right_arm is None else np.asarray(gt_right_arm[tgt_abs], dtype=np.float32)
        l_prefix = _interp_prefix(left_q, l_tgt_q, prefix_len)
        r_prefix = _interp_prefix(right_q, r_tgt_q, prefix_len)
        if correction_compose_gt_tail_enable:
            l_tail = _take_tail(gt_left_arm, tgt_abs + 1, suffix_len, l_tgt_q)
            r_tail = _take_tail(gt_right_arm, tgt_abs + 1, suffix_len, r_tgt_q)
            lt = np.concatenate([l_prefix, l_tail], axis=0).astype(np.float32)
            rt = np.concatenate([r_prefix, r_tail], axis=0).astype(np.float32)
        else:
            lt = resample_trajectory(l_prefix, chunk_size).astype(np.float32)
            rt = resample_trajectory(r_prefix, chunk_size).astype(np.float32)

        tl_grip = float(np.clip(left_grip_traj[t_target], 0.0, 1.0))
        tr_grip = float(np.clip(right_grip_traj[t_target], 0.0, 1.0))
        if prefix_len <= 1:
            l_grip_prefix = np.array([tl_grip], dtype=np.float32)
            r_grip_prefix = np.array([tr_grip], dtype=np.float32)
        else:
            l_grip_prefix = np.linspace(float(left_grip), tl_grip, prefix_len, dtype=np.float32)
            r_grip_prefix = np.linspace(float(right_grip), tr_grip, prefix_len, dtype=np.float32)
        if correction_compose_gt_tail_enable:
            l_grip_tail = _take_tail(left_grip_traj, t_target + 1, suffix_len, np.array([tl_grip], dtype=np.float32))[:, 0]
            r_grip_tail = _take_tail(right_grip_traj, t_target + 1, suffix_len, np.array([tr_grip], dtype=np.float32))[:, 0]
            lg = np.concatenate([l_grip_prefix, l_grip_tail], axis=0).astype(np.float32)
            rg = np.concatenate([r_grip_prefix, r_grip_tail], axis=0).astype(np.float32)
        else:
            lg = resample_trajectory(l_grip_prefix[:, None], chunk_size).astype(np.float32)[:, 0]
            rg = resample_trajectory(r_grip_prefix[:, None], chunk_size).astype(np.float32)[:, 0]

        res_l = {'status': 'BypassInterpNearest', 'position': np.stack([np.asarray(left_q, dtype=np.float32), l_tgt_q], axis=0)}
        res_r = {'status': 'BypassInterpNearest', 'position': np.stack([np.asarray(right_q, dtype=np.float32), r_tgt_q], axis=0)}
    else:
        import sapien
        target_lp = sapien.Pose(left_ep[t_target, :3], left_ep[t_target, 3:7])
        target_rp = sapien.Pose(right_ep[t_target, :3], right_ep[t_target, 3:7])
        qpos_full = np.zeros(len(fk.jnames), dtype=np.float32)
        qpos_full[fk.fl_idx] = left_q
        qpos_full[fk.fr_idx] = right_q

        if corr_left_active:
            try:
                res_l = planner_l.plan_path(qpos_full, target_lp, arms_tag='left')
            except Exception as exc:
                if debug_dir is not None:
                    import json
                    _dbg_corr = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr, exist_ok=True)
                    with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                        json.dump({
                            'reason': 'planner_exception_left',
                            'error': str(exc),
                            't_star': int(t_star),
                            'min_dist': float(min_dist),
                            'rollout': _dbg_rollout,
                        }, _f, indent=2)
                return None
        if corr_right_active:
            try:
                res_r = planner_r.plan_path(qpos_full, target_rp, arms_tag='right')
            except Exception as exc:
                if debug_dir is not None:
                    import json
                    _dbg_corr = os.path.join(debug_dir, 'correction')
                    os.makedirs(_dbg_corr, exist_ok=True)
                    with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                        json.dump({
                            'reason': 'planner_exception_right',
                            'error': str(exc),
                            't_star': int(t_star),
                            'min_dist': float(min_dist),
                            'rollout': _dbg_rollout,
                        }, _f, indent=2)
                return None
        if (corr_left_active and res_l.get('status') != 'Success') or (corr_right_active and res_r.get('status') != 'Success'):
            if debug_dir is not None:
                import json
                _dbg_corr = os.path.join(debug_dir, 'correction')
                os.makedirs(_dbg_corr, exist_ok=True)
                with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                    json.dump({
                        'reason': 'planner_status_fail',
                        'planner_left_status': res_l.get('status'),
                        'planner_right_status': res_r.get('status'),
                        't_star': int(t_star),
                        'min_dist': float(min_dist),
                        'rollout': _dbg_rollout,
                    }, _f, indent=2)
            return None

        l_path = np.asarray(res_l['position'], dtype=np.float32)
        r_path = np.asarray(res_r['position'], dtype=np.float32)
        l_future = l_path[1:] if l_path.shape[0] > 1 else l_path
        r_future = r_path[1:] if r_path.shape[0] > 1 else r_path
        prefix_len = int(max(1, l_future.shape[0], r_future.shape[0]))

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

    def _compose_with_gt_tail(prefix, gt_tail, out_len, force_resample=False):
        pfx = np.asarray(prefix, dtype=np.float32)
        if force_resample:
            # Keep planner/GT composition ratio stable: consume at most the
            # remaining budget from GT tail before optional global resample.
            if pfx.shape[0] >= out_len:
                return resample_trajectory(pfx.astype(np.float32), out_len).astype(np.float32)
            tail_budget = int(max(0, out_len - pfx.shape[0]))
            seq = [pfx]
            if gt_tail is not None and len(gt_tail) > 0 and tail_budget > 0:
                g = np.asarray(gt_tail, dtype=np.float32)
                seq.append(g[:tail_budget])
            cat = np.concatenate(seq, axis=0)
            if cat.shape[0] < out_len:
                pad = np.repeat(cat[-1:].astype(np.float32), out_len - cat.shape[0], axis=0)
                cat = np.concatenate([cat.astype(np.float32), pad], axis=0)
            return resample_trajectory(cat.astype(np.float32), out_len).astype(np.float32)
        seq = [pfx]
        if gt_tail is not None and len(gt_tail) > 0:
            seq.append(np.asarray(gt_tail, dtype=np.float32))
        cat = np.concatenate(seq, axis=0)
        # If correction-prefix alone is longer than chunk, compress it so the
        # last step still reaches the correction target (instead of truncating early).
        if pfx.shape[0] >= out_len:
            return resample_trajectory(pfx, out_len).astype(np.float32)
        if cat.shape[0] >= out_len:
            return cat[:out_len].astype(np.float32)
        pad = np.repeat(cat[-1:].astype(np.float32), out_len - cat.shape[0], axis=0)
        return np.concatenate([cat, pad], axis=0).astype(np.float32)

    if not correction_interp_nearest_enable:
        if nearest_mode == "gripper_close":
            # Dedicated correction for early-close error:
            # 1) quickly reopen gripper, 2) follow GT forward.
            prefix_len = int(np.clip(np.floor(chunk_size * correction_gripper_close_prefix_ratio), 1, max(1, chunk_size - 1)))
            # For gripper_close correction, compose tail directly from sampled start_ts.
            # Do not shift by rollout_steps_total.
            follow_idx = 0
            follow_abs = int(start_ts)
            suffix_len = int(max(0, chunk_size - prefix_len))

            def _interp_prefix(curr, target, n):
                curr = np.asarray(curr, dtype=np.float32)
                target = np.asarray(target, dtype=np.float32)
                if n <= 1:
                    return target[None, :].astype(np.float32)
                alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                return ((1.0 - alpha) * curr[None, :] + alpha * target[None, :]).astype(np.float32)

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

            def _resample_joint_pair_by_eef_distance(l_seg, r_seg):
                """
                Uniformize by EEF travel distance (active arm only), then interpolate
                both joint sequences on that distance parameter.
                """
                lq = np.asarray(l_seg, dtype=np.float32)
                rq = np.asarray(r_seg, dtype=np.float32)
                if lq.ndim != 2 or rq.ndim != 2:
                    return lq, rq
                m = int(min(lq.shape[0], rq.shape[0]))
                if m <= 2:
                    return lq, rq
                left_on = bool(active_info_fixed.get('left_arm', True))
                right_on = bool(active_info_fixed.get('right_arm', True))
                if (not left_on) and (not right_on):
                    left_on, right_on = True, True
                try:
                    lp_seq, rp_seq = [], []
                    for i in range(m):
                        fr = fk.forward(lq[i], rq[i])
                        lp_seq.append(np.asarray(fr['left'][0], dtype=np.float32))
                        rp_seq.append(np.asarray(fr['right'][0], dtype=np.float32))
                    lp = np.stack(lp_seq, axis=0)
                    rp = np.stack(rp_seq, axis=0)
                except Exception:
                    # If FK fails for this segment, keep original sequence.
                    return lq, rq
                step = np.zeros((m - 1,), dtype=np.float32)
                n_used = 0
                if left_on:
                    step += np.linalg.norm(lp[1:] - lp[:-1], axis=1)
                    n_used += 1
                if right_on:
                    step += np.linalg.norm(rp[1:] - rp[:-1], axis=1)
                    n_used += 1
                step /= float(max(1, n_used))
                cum = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(step, dtype=np.float32)])
                total = float(cum[-1])
                if (not np.isfinite(total)) or total <= 1e-8:
                    return lq, rq
                tgt = np.linspace(0.0, total, m, dtype=np.float32)
                lo = np.zeros_like(lq, dtype=np.float32)
                ro = np.zeros_like(rq, dtype=np.float32)
                for j in range(lq.shape[1]):
                    lo[:, j] = np.interp(tgt, cum, lq[:, j]).astype(np.float32)
                for j in range(rq.shape[1]):
                    ro[:, j] = np.interp(tgt, cum, rq[:, j]).astype(np.float32)
                lo[0], lo[-1] = lq[0], lq[-1]
                ro[0], ro[-1] = rq[0], rq[-1]
                return lo.astype(np.float32), ro.astype(np.float32)

            def _uniformize_arm_tail_on_static_gripper(l_arm, r_arm, l_gr, r_gr):
                """
                Only uniformize arm motion on sub-spans where gripper stays unchanged.
                Spans that include gripper transitions are kept untouched.
                """
                la = np.asarray(l_arm, dtype=np.float32).copy()
                ra = np.asarray(r_arm, dtype=np.float32).copy()
                lg = np.asarray(l_gr, dtype=np.float32).reshape(-1)
                rg = np.asarray(r_gr, dtype=np.float32).reshape(-1)
                n = int(min(la.shape[0], ra.shape[0], lg.shape[0], rg.shape[0]))
                if n <= 2:
                    return la, ra, 0, 0
                eps = float(cfg.get('correction_gripper_static_eps', 1e-4))
                edge_change = np.zeros((n - 1,), dtype=bool)
                for k in range(n - 1):
                    edge_change[k] = (abs(float(lg[k + 1] - lg[k])) > eps) or (abs(float(rg[k + 1] - rg[k])) > eps)
                s = 0
                uniform_pts = 0
                uniform_spans = 0
                while s < n:
                    e = s
                    while e < n - 1 and (not edge_change[e]):
                        e += 1
                    seg_len = int(e - s + 1)
                    if seg_len >= 3:
                        _l_new, _r_new = _resample_joint_pair_by_eef_distance(la[s:e + 1], ra[s:e + 1])
                        la[s:e + 1] = _l_new
                        ra[s:e + 1] = _r_new
                        uniform_pts += int(seg_len)
                        uniform_spans += 1
                    s = e + 1
                return la, ra, int(uniform_pts), int(uniform_spans)

            l_tgt_q = np.asarray(left_q, dtype=np.float32) if gt_left_arm is None else np.asarray(gt_left_arm[follow_abs], dtype=np.float32)
            r_tgt_q = np.asarray(right_q, dtype=np.float32) if gt_right_arm is None else np.asarray(gt_right_arm[follow_abs], dtype=np.float32)
            # Keep arm still while reopening gripper.
            l_prefix = np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], prefix_len, axis=0)
            r_prefix = np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], prefix_len, axis=0)
            # Then follow GT forward from current rollout time.
            l_tail = _take_tail(gt_left_arm, follow_abs, suffix_len, l_tgt_q)
            r_tail = _take_tail(gt_right_arm, follow_abs, suffix_len, r_tgt_q)

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

            l_grip_prefix = _build_grip_prefix(left_grip, tl_grip)
            r_grip_prefix = _build_grip_prefix(right_grip, tr_grip)
            l_grip_tail = _take_tail(left_grip_traj, follow_idx + 1, suffix_len, np.array([tl_grip], dtype=np.float32))[:, 0]
            r_grip_tail = _take_tail(right_grip_traj, follow_idx + 1, suffix_len, np.array([tr_grip], dtype=np.float32))[:, 0]
            gripper_close_suffix_len_dbg = int(suffix_len)
            # Temporarily disable tail uniformization: keep GT tail as-is.
            gripper_close_uniformized_points_dbg = 0
            gripper_close_uniformized_spans_dbg = 0
            lt = np.concatenate([l_prefix, l_tail], axis=0).astype(np.float32)
            rt = np.concatenate([r_prefix, r_tail], axis=0).astype(np.float32)
            lg = np.concatenate([l_grip_prefix, l_grip_tail], axis=0).astype(np.float32)
            rg = np.concatenate([r_grip_prefix, r_grip_tail], axis=0).astype(np.float32)

            res_l = {'status': 'BypassGripperCloseRecover', 'position': np.stack([np.asarray(left_q, dtype=np.float32), l_tgt_q], axis=0)}
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
            l_tail = _take_tail(gt_left_arm, follow_abs + 1, suffix_len, l_tgt_q)
            r_tail = _take_tail(gt_right_arm, follow_abs + 1, suffix_len, r_tgt_q)
            lt = np.concatenate([l_prefix, l_tail], axis=0).astype(np.float32)
            rt = np.concatenate([r_prefix, r_tail], axis=0).astype(np.float32)

            tl_grip = float(np.clip(left_grip_traj[follow_idx], 0.0, 1.0))
            tr_grip = float(np.clip(right_grip_traj[follow_idx], 0.0, 1.0))
            if prefix_len <= 1:
                l_grip_prefix = np.array([tl_grip], dtype=np.float32)
                r_grip_prefix = np.array([tr_grip], dtype=np.float32)
            else:
                l_grip_prefix = np.linspace(float(left_grip), tl_grip, prefix_len, dtype=np.float32)
                r_grip_prefix = np.linspace(float(right_grip), tr_grip, prefix_len, dtype=np.float32)
            l_grip_tail = _take_tail(left_grip_traj, follow_idx + 1, suffix_len, np.array([tl_grip], dtype=np.float32))[:, 0]
            r_grip_tail = _take_tail(right_grip_traj, follow_idx + 1, suffix_len, np.array([tr_grip], dtype=np.float32))[:, 0]
            lg = np.concatenate([l_grip_prefix, l_grip_tail], axis=0).astype(np.float32)
            rg = np.concatenate([r_grip_prefix, r_grip_tail], axis=0).astype(np.float32)

        if correction_compose_gt_tail_enable:
            prefix_len = int(np.clip(np.floor(chunk_size * correction_planner_prefix_ratio), 1, max(1, chunk_size - 1)))
        if nearest_mode not in {"gripper_close", "translation", "rotation"}:
            l_prefix = _fit_prefix(l_future, left_q, prefix_len)
            r_prefix = _fit_prefix(r_future, right_q, prefix_len)

        l_tail = None
        r_tail = None
        if nearest_mode not in {"gripper_close", "translation", "rotation"} and correction_compose_gt_tail_enable and gt_left_arm is not None and corr_left_active:
            l_tail = np.asarray(gt_left_arm[start_ts + t_target + 1:], dtype=np.float32)
        if nearest_mode not in {"gripper_close", "translation", "rotation"} and correction_compose_gt_tail_enable and gt_right_arm is not None and corr_right_active:
            r_tail = np.asarray(gt_right_arm[start_ts + t_target + 1:], dtype=np.float32)

        if nearest_mode not in {"gripper_close", "translation", "rotation"}:
            lt = _compose_with_gt_tail(l_prefix, l_tail, chunk_size, force_resample=True)
            rt = _compose_with_gt_tail(r_prefix, r_tail, chunk_size, force_resample=True)

        if nearest_mode not in {"gripper_close", "translation", "rotation"}:
            tl_grip = float(np.clip(left_grip_traj[t_target], 0.0, 1.0))
            tr_grip = float(np.clip(right_grip_traj[t_target], 0.0, 1.0))

        # Prefix gripper profile: hold current value first, then change near the end.
        switch_ratio = float(np.clip(cfg['correction_gripper_switch_ratio'], 0.0, 1.0))
        switch_idx = int(np.clip(np.floor(prefix_len * switch_ratio), 0, max(0, prefix_len - 1)))
        l_grip_prefix = np.full((prefix_len,), float(left_grip), dtype=np.float32) if nearest_mode not in {"gripper_close", "translation", "rotation"} else None
        r_grip_prefix = np.full((prefix_len,), float(right_grip), dtype=np.float32) if nearest_mode not in {"gripper_close", "translation", "rotation"} else None
        tail_len = prefix_len - switch_idx
        if nearest_mode not in {"gripper_close", "translation", "rotation"} and tail_len > 1:
            l_grip_prefix[switch_idx:] = np.linspace(float(left_grip), tl_grip, tail_len, dtype=np.float32)
            r_grip_prefix[switch_idx:] = np.linspace(float(right_grip), tr_grip, tail_len, dtype=np.float32)
        elif nearest_mode not in {"gripper_close", "translation", "rotation"} and tail_len == 1:
            l_grip_prefix[-1] = tl_grip
            r_grip_prefix[-1] = tr_grip
        l_grip_tail = (
            np.asarray(left_grip_traj[t_target + 1:], dtype=np.float32)
            if (correction_compose_gt_tail_enable and corr_left_active)
            else np.asarray([], dtype=np.float32)
        ) if nearest_mode not in {"gripper_close", "translation", "rotation"} else None
        r_grip_tail = (
            np.asarray(right_grip_traj[t_target + 1:], dtype=np.float32)
            if (correction_compose_gt_tail_enable and corr_right_active)
            else np.asarray([], dtype=np.float32)
        ) if nearest_mode not in {"gripper_close", "translation", "rotation"} else None
        if nearest_mode not in {"gripper_close", "translation", "rotation"}:
            lg = _compose_with_gt_tail(
            l_grip_prefix[:, None], l_grip_tail[:, None], chunk_size, force_resample=True
        )[:, 0]
            rg = _compose_with_gt_tail(
            r_grip_prefix[:, None], r_grip_tail[:, None], chunk_size, force_resample=True
        )[:, 0]
    if not corr_left_active:
        lt = np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], chunk_size, axis=0)
        lg = np.full((chunk_size,), float(left_grip), dtype=np.float32)
    if not corr_right_active:
        rt = np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], chunk_size, axis=0)
        rg = np.full((chunk_size,), float(right_grip), dtype=np.float32)

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
                    ddim_steps=evac_ddim_steps,
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

            def _draw_pregrasp_quarter_points(img, l_seq, r_seq, labels):
                c_l = (0, 255, 255)
                c_r = (255, 255, 0)
                for i, lb in enumerate(labels):
                    if i < len(l_seq) and l_seq[i] is not None:
                        lx, ly = int(l_seq[i][0]), int(l_seq[i][1])
                        cv2.circle(img, (lx, ly), 5, c_l, -1, cv2.LINE_AA)
                        cv2.putText(img, f"L-{lb}", (lx + 6, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_l, 1, cv2.LINE_AA)
                    if i < len(r_seq) and r_seq[i] is not None:
                        rx, ry = int(r_seq[i][0]), int(r_seq[i][1])
                        cv2.circle(img, (rx, ry), 5, c_r, -1, cv2.LINE_AA)
                        cv2.putText(img, f"R-{lb}", (rx + 6, ry - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_r, 1, cv2.LINE_AA)

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

            def _extract_first_uv_pair(pose_np):
                if pose_np is None or pose_np.shape[0] < 1:
                    return None, None
                uvs_l, uvs_r, pts_l, pts_r = _project_base_uv_from_pose_np(pose_np)
                l_seq = _extract_base_uv(uvs_l, pts_l.reshape(1, pts_l.shape[1], 4, 4))
                r_seq = _extract_base_uv(uvs_r, pts_r.reshape(1, pts_r.shape[1], 4, 4))
                l_pt = l_seq[0] if len(l_seq) > 0 else None
                r_pt = r_seq[0] if len(r_seq) > 0 else None
                return l_pt, r_pt

            def _draw_point_labeled(img, pt, text, color):
                if pt is None:
                    return
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(img, (x, y), 6, color, -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

            def _draw_cross_labeled(img, pt, text, color):
                if pt is None:
                    return
                x, y = int(pt[0]), int(pt[1])
                s = 7
                cv2.line(img, (x - s, y), (x + s, y), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.line(img, (x, y - s), (x, y + s), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.line(img, (x - s, y), (x + s, y), color, 2, cv2.LINE_AA)
                cv2.line(img, (x, y - s), (x, y + s), color, 2, cv2.LINE_AA)
                cv2.putText(img, text, (x + 8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, text, (x + 8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

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

            # Mark pregrasp segment quartile points (Q0/Q1/Q2/Q3/Q4) from
            # dataloader-provided absolute segment range.
            try:
                if pregrasp_seg_start is not None and pregrasp_seg_end is not None and n_ep_total > 0:
                    ps = int(np.clip(int(pregrasp_seg_start), 0, n_ep_total - 1))
                    pe = int(np.clip(int(pregrasp_seg_end), 0, n_ep_total - 1))
                    if pe < ps:
                        ps, pe = pe, ps
                    if pe >= ps:
                        q_rel = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
                        q_idx = [int(np.clip(round(ps + float(r) * (pe - ps)), 0, n_ep_total - 1)) for r in q_rel]
                        q_pose_np = _build_pose_np_from_raw_indices(q_idx)
                        if q_pose_np is not None and q_pose_np.shape[0] > 0:
                            q_uvs_l, q_uvs_r, q_pts_l, q_pts_r = _project_base_uv_from_pose_np(q_pose_np)
                            q_l_seq = _extract_base_uv(q_uvs_l, q_pts_l.reshape(1, q_pts_l.shape[1], 4, 4))
                            q_r_seq = _extract_base_uv(q_uvs_r, q_pts_r.reshape(1, q_pts_r.shape[1], 4, 4))
                            q_labels = [f"Q{i}" for i in range(len(q_idx))]
                            _draw_pregrasp_quarter_points(_overlay_gt, q_l_seq, q_r_seq, q_labels)
                            _draw_pregrasp_quarter_points(_overlay_c, q_l_seq, q_r_seq, q_labels)
            except Exception:
                pass

            # Corrected overlay: keep only correction trajectories + phase-colored GT path.
            _overlay_c[mask] = (0.6 * _overlay_c[mask] + 0.4 * traj_u8[mask]).astype(np.uint8)
            _draw_polyline(_overlay_c, luv, (0, 255, 0))
            _draw_polyline(_overlay_c, ruv, (0, 0, 255))
            _annotate_start_end(_overlay_c, luv, "L", (0, 255, 0))
            _annotate_start_end(_overlay_c, ruv, "R", (0, 0, 255))
            # Mark the actual perturb-anchor point used for translation direction.
            # This is the fixed absolute anchor chosen from dataloader pregrasp segment end.
            if perturb_anchor_idx_abs_fixed is not None and n_ep_total > 0:
                pidx = int(np.clip(int(perturb_anchor_idx_abs_fixed), 0, n_ep_total - 1))
                p_row = _pack_pose_row(
                    raw_data['left_endpose'][pidx, :3].astype(np.float32),
                    raw_data['left_endpose'][pidx, 3:7].astype(np.float32),
                    raw_data['left_gripper'][pidx],
                    raw_data['right_endpose'][pidx, :3].astype(np.float32),
                    raw_data['right_endpose'][pidx, 3:7].astype(np.float32),
                    raw_data['right_gripper'][pidx],
                )
                p_l_pt, p_r_pt = _extract_first_uv_pair(np.stack([p_row], axis=0))
                for _img in (_overlay_o, _overlay_c):
                    _draw_point_labeled(_img, p_l_pt, "L-PANCH", (0, 240, 255))
                    _draw_point_labeled(_img, p_r_pt, "R-PANCH", (255, 230, 0))

                # Also mark the actual "forward-searched" point used by translation direction.
                _, pnext_l_idx = _traj_dir_after_anchor_with_idx(raw_data['left_endpose'], pidx, lookahead=128)
                _, pnext_r_idx = _traj_dir_after_anchor_with_idx(raw_data['right_endpose'], pidx, lookahead=128)
                if pnext_l_idx is None:
                    pnext_l_idx = int(np.clip(pidx + 128, 0, n_ep_total - 1))
                if pnext_r_idx is None:
                    pnext_r_idx = int(np.clip(pidx + 128, 0, n_ep_total - 1))
                pnext_row = _pack_pose_row(
                    raw_data['left_endpose'][pnext_l_idx, :3].astype(np.float32),
                    raw_data['left_endpose'][pnext_l_idx, 3:7].astype(np.float32),
                    raw_data['left_gripper'][pnext_l_idx],
                    raw_data['right_endpose'][pnext_r_idx, :3].astype(np.float32),
                    raw_data['right_endpose'][pnext_r_idx, 3:7].astype(np.float32),
                    raw_data['right_gripper'][pnext_r_idx],
                )
                pn_l_pt, pn_r_pt = _extract_first_uv_pair(np.stack([pnext_row], axis=0))
                # If projected points overlap, nudge PNEXT marker for visibility.
                def _nudge_if_overlap(a, b):
                    if a is None or b is None:
                        return b, False
                    if int(a[0]) == int(b[0]) and int(a[1]) == int(b[1]):
                        return (int(b[0]) + 10, int(b[1]) + 10), True
                    return b, False
                pn_l_pt, ov_l = _nudge_if_overlap(p_l_pt, pn_l_pt)
                pn_r_pt, ov_r = _nudge_if_overlap(p_r_pt, pn_r_pt)
                for _img in (_overlay_o, _overlay_c):
                    _draw_cross_labeled(_img, pn_l_pt, ("L-PNEXT*" if ov_l else "L-PNEXT"), (40, 255, 120))
                    _draw_cross_labeled(_img, pn_r_pt, ("R-PNEXT*" if ov_r else "R-PNEXT"), (120, 255, 40))
                # Draw anchor->next line for visibility when points are close in projection.
                def _draw_link(img, a, b, color):
                    if a is None or b is None:
                        return
                    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 2, cv2.LINE_AA)
                for _img in (_overlay_o, _overlay_c):
                    _draw_link(_img, p_l_pt, pn_l_pt, (80, 255, 200))
                    _draw_link(_img, p_r_pt, pn_r_pt, (200, 255, 80))
                    if ov_l or ov_r:
                        cv2.putText(_img, "OVERLAP: PNEXT nudged +10px", (12, max(40, h_img - 34)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2, cv2.LINE_AA)
                        cv2.putText(_img, "OVERLAP: PNEXT nudged +10px", (12, max(40, h_img - 34)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(
                        _img,
                        f"PANCH={pidx} PNEXTL={pnext_l_idx} PNEXTR={pnext_r_idx}",
                        (12, max(20, h_img - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        _img,
                        f"PANCH={pidx} PNEXTL={pnext_l_idx} PNEXTR={pnext_r_idx}",
                        (12, max(20, h_img - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

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
        _eval_video_map = {}
        for _v in evac_recover_eval_videos:
            try:
                _eval_video_map[int(_v.get('step'))] = {
                    'path': _v.get('path'),
                    'exists': bool(_v.get('exists', False)),
                }
            except Exception:
                continue

        _rollout_records = []
        for _r in _dbg_rollout:
            _step = _r.get('step')
            _video = _video_map.get(int(_step)) if _step is not None and str(_step).isdigit() else None
            _video_eval = _eval_video_map.get(int(_step)) if _step is not None and str(_step).isdigit() else None
            _rollout_records.append({
                'step': (None if _step is None else int(_step)),
                'step_role': _r.get('step_role'),
                'paired_perturb_step': (None if _r.get('paired_perturb_step') is None else int(_r.get('paired_perturb_step'))),
                't_star': (None if _r.get('t_star') is None else int(_r.get('t_star'))),
                'min_dist': (None if _r.get('min_dist') is None else float(_r.get('min_dist'))),
                'min_dist_start': (None if _r.get('min_dist_start') is None else float(_r.get('min_dist_start'))),
                'min_dist_end': (None if _r.get('min_dist_end') is None else float(_r.get('min_dist_end'))),
                'min_dist_delta': (None if _r.get('min_dist_delta') is None else float(_r.get('min_dist_delta'))),
                'perturb_pre_last_dist': (None if _r.get('perturb_pre_last_dist') is None else float(_r.get('perturb_pre_last_dist'))),
                'perturb_post_last_dist': (None if _r.get('perturb_post_last_dist') is None else float(_r.get('perturb_post_last_dist'))),
                'recover_pre_first_dist': (None if _r.get('recover_pre_first_dist') is None else float(_r.get('recover_pre_first_dist'))),
                'recover_post_last_dist': (None if _r.get('recover_post_last_dist') is None else float(_r.get('recover_post_last_dist'))),
                'recover_eval_step': (None if _r.get('recover_eval_step') is None else int(_r.get('recover_eval_step'))),
                'recover_eval_t_star': (None if _r.get('recover_eval_t_star') is None else int(_r.get('recover_eval_t_star'))),
                'recover_eval_anchor_idx': (None if _r.get('recover_eval_anchor_idx') is None else int(_r.get('recover_eval_anchor_idx'))),
                'recover_eval_terminal_pass': (None if _r.get('recover_eval_terminal_pass') is None else bool(_r.get('recover_eval_terminal_pass'))),
                'recover_gripper_mode': _r.get('recover_gripper_mode'),
                'recover_gripper_close_min': (None if _r.get('recover_gripper_close_min') is None else float(_r.get('recover_gripper_close_min'))),
                'recover_gripper_open_max': (None if _r.get('recover_gripper_open_max') is None else float(_r.get('recover_gripper_open_max'))),
                'recover_gripper_hit_left': (None if _r.get('recover_gripper_hit_left') is None else bool(_r.get('recover_gripper_hit_left'))),
                'recover_gripper_hit_right': (None if _r.get('recover_gripper_hit_right') is None else bool(_r.get('recover_gripper_hit_right'))),
                'recover_gripper_passed': (None if _r.get('recover_gripper_passed') is None else bool(_r.get('recover_gripper_passed'))),
                'anchor_idx': (None if _r.get('anchor_idx') is None else int(_r.get('anchor_idx'))),
                'phase_key': _r.get('phase_key'),
                'perturb_mode': _r.get('perturb_mode'),
                'sampled_error_mode': _r.get('sampled_error_mode'),
                'active_left_arm': bool(_r.get('active_left_arm', True)),
                'active_right_arm': bool(_r.get('active_right_arm', True)),
                'active_left_gripper': bool(_r.get('active_left_gripper', True)),
                'active_right_gripper': bool(_r.get('active_right_gripper', True)),
                'active_left_arm_score': (None if _r.get('active_left_arm_score') is None else float(_r.get('active_left_arm_score'))),
                'active_right_arm_score': (None if _r.get('active_right_arm_score') is None else float(_r.get('active_right_arm_score'))),
                'active_left_gripper_score': (None if _r.get('active_left_gripper_score') is None else float(_r.get('active_left_gripper_score'))),
                'active_right_gripper_score': (None if _r.get('active_right_gripper_score') is None else float(_r.get('active_right_gripper_score'))),
                'action_perturbed': bool(_r.get('action_perturbed', False)),
                'left_grip': (None if _r.get('left_grip') is None else float(_r.get('left_grip'))),
                'right_grip': (None if _r.get('right_grip') is None else float(_r.get('right_grip'))),
                'reject_sampling_enabled': bool(_r.get('reject_sampling_enabled', False)),
                'reject_sampling_trial': (None if _r.get('reject_sampling_trial') is None else int(_r.get('reject_sampling_trial'))),
                'reject_sampling_trials_max': (None if _r.get('reject_sampling_trials_max') is None else int(_r.get('reject_sampling_trials_max'))),
                'reject_sampling_dir_jitter_eps': (None if _r.get('reject_sampling_dir_jitter_eps') is None else float(_r.get('reject_sampling_dir_jitter_eps'))),
                'reject_sampling_accepted': (None if _r.get('reject_sampling_accepted') is None else bool(_r.get('reject_sampling_accepted'))),
                'reject_sampling_fallback_best': (None if _r.get('reject_sampling_fallback_best') is None else bool(_r.get('reject_sampling_fallback_best'))),
                'min_dist_triggered': bool(_r.get('min_dist_triggered', False)),
                'reason': _r.get('reason'),
                'evac_video': _video,
                'evac_recover_eval_video': _video_eval,
                'evac_video_dir': (None if _video is None else os.path.dirname(_video.get('path'))),
                'evac_recover_eval_video_dir': (None if _video_eval is None else os.path.dirname(_video_eval.get('path'))),
            })

        _trigger_source = None
        _trigger_step = None
        _trigger_value = None
        if bool(min_dist_triggered):
            _trigger_source = ('min_dist' if min_dist_trigger_source is None else str(min_dist_trigger_source))
            _trigger_step = (None if min_dist_trigger_step is None else int(min_dist_trigger_step))
            _trigger_value = (None if min_dist_trigger_dist is None else float(min_dist_trigger_dist))
        elif bool(force_generate_correction):
            _trigger_source = 'fallback'

        _plan_info = {
            'version': 2,
            'trigger_mode': ('real_error' if real_error_trigger_active else 'min_dist_recovery'),
            'trigger_threshold': {
                'min_dist': (float(real_error_min_dist_thresh) if real_error_trigger_active else None),
                'min_dist_delta': (float(real_error_min_dist_delta_thresh) if real_error_trigger_active else None),
                'min_dist_recover_ratio': float(min_dist_recover_ratio),
                'min_dist_recover_use_added': bool(not real_error_trigger_active),
            },
            'fallback': {
                'min_dist': bool(min_dist_fallback_force_correction),
                'used': bool(force_generate_correction and (not min_dist_triggered)),
            },
            'trigger_result': {
                'triggered': bool(force_generate_correction),
                'source': _trigger_source,
                'step': _trigger_step,
                'value': _trigger_value,
            },
            'rollout_summary': {
                'start_ts': int(start_ts),
                'rollout_exec_steps': int(rollout_exec_steps),
                'rollout_steps_total': int(rollout_steps_total),
                'gt_ref_index_for_original': int(np.clip(start_ts + int(rollout_steps_total), 0, raw_data['left_endpose'].shape[0] - 1)),
                'num_rollouts': int(len(_rollout_records)),
            },
            'final_match': {
                't_star': int(t_star),
                'min_dist': float(min_dist),
                'anchor_idx': int(np.clip(
                    _find_next_gripper_toggle_idx_from(left_grip_traj, right_grip_traj, int(t_star)),
                    0, len(left_ep) - 1
                )),
                'nearest_mode': nearest_mode,
                'nearest_gripper_penalty_used': float(nearest_gripper_penalty),
            },
            'correction_target': {
                'target_mode': target_mode,
                't_target': int(t_target),
                'target_lookahead_steps': int(target_lookahead_steps),
                'left_pose': left_ep[t_target].tolist(),
                'right_pose': right_ep[t_target].tolist(),
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
                'interp_nearest_enable': bool(correction_interp_nearest_enable),
                'interp_prefix_ratio': float(correction_interp_prefix_ratio),
                'planner_prefix_ratio': float(correction_planner_prefix_ratio),
                'gripper_close_prefix_ratio': float(correction_gripper_close_prefix_ratio),
                'compose_gt_tail_enable': bool(correction_compose_gt_tail_enable),
                'correction_prefix_len': int(prefix_len),
                'gripper_close_suffix_len': (None if gripper_close_suffix_len_dbg is None else int(gripper_close_suffix_len_dbg)),
                'gripper_close_uniformized_points': (None if gripper_close_uniformized_points_dbg is None else int(gripper_close_uniformized_points_dbg)),
                'gripper_close_uniformized_spans': (None if gripper_close_uniformized_spans_dbg is None else int(gripper_close_uniformized_spans_dbg)),
                'debug_correction_evac_rollout': bool(debug_correction_evac_rollout),
                'evac_correction_generated_video': evac_corr_video,
            },
        }
        # closed_loop_info.json already records detailed closed-loop rollout traces.
        # Keep correction_info focused on correction planning/targeting summary.
        _plan_info['rollout_summary'].pop('start_ts', None)
        _plan_info['rollout_summary'].pop('rollout_exec_steps', None)
        _plan_info['final_match'].pop('t_star', None)
        _plan_info['final_match'].pop('anchor_idx', None)
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
                'anchor_idx': _r.get('anchor_idx'),
                'action_perturbed': _r.get('action_perturbed'),
                'perturb_mode': _r.get('perturb_mode'),
                'sampled_error_mode': _r.get('sampled_error_mode'),
                'reject_sampling_enabled': _r.get('reject_sampling_enabled'),
                'reject_sampling_trial': _r.get('reject_sampling_trial'),
                'reject_sampling_trials_max': _r.get('reject_sampling_trials_max'),
                'reject_sampling_dir_jitter_eps': _r.get('reject_sampling_dir_jitter_eps'),
                'reject_sampling_accepted': _r.get('reject_sampling_accepted'),
                'reject_sampling_fallback_best': _r.get('reject_sampling_fallback_best'),
                'perturb_pre_last_dist': _r.get('perturb_pre_last_dist'),
                'perturb_post_last_dist': _r.get('perturb_post_last_dist'),
                'recover_pre_first_dist': _r.get('recover_pre_first_dist'),
                'recover_post_last_dist': _r.get('recover_post_last_dist'),
                'recover_eval_step': _r.get('recover_eval_step'),
                'recover_eval_t_star': _r.get('recover_eval_t_star'),
                'recover_eval_anchor_idx': _r.get('recover_eval_anchor_idx'),
                'recover_eval_terminal_pass': _r.get('recover_eval_terminal_pass'),
                'recover_gripper_mode': _r.get('recover_gripper_mode'),
                'recover_gripper_close_min': _r.get('recover_gripper_close_min'),
                'recover_gripper_open_max': _r.get('recover_gripper_open_max'),
                'recover_gripper_hit_left': _r.get('recover_gripper_hit_left'),
                'recover_gripper_hit_right': _r.get('recover_gripper_hit_right'),
                'recover_gripper_passed': _r.get('recover_gripper_passed'),
            }
            _closed_loop_rollouts.append(_drop_none(_rec))

        _closed_loop_info = {
            'version': 1,
            'start_ts': int(start_ts),
            'sample_pregrasp_segment_start': (None if pregrasp_seg_start is None else int(pregrasp_seg_start)),
            'sample_pregrasp_segment_end': (None if pregrasp_seg_end is None else int(pregrasp_seg_end)),
            'rollout_exec_steps': int(rollout_exec_steps),
            'final_match': {
                't_star': _plan_info['final_match'].get('t_star'),
                'anchor_idx': _plan_info['final_match'].get('anchor_idx'),
            },
            'rollouts': _closed_loop_rollouts,
        }
        with open(os.path.join(_dbg_corr, 'closed_loop_info.json'), 'w') as _f:
            json.dump(_closed_loop_info, _f, indent=2)

    error_action_prefix_raw = np.asarray(act_raw, dtype=np.float32).copy()
    if correction_interp_nearest_enable:
        correction_branch = "interp_nearest"
    elif nearest_mode == "gripper_close":
        correction_branch = "gripper_close"
    elif nearest_mode == "translation":
        correction_branch = "translation"
    elif nearest_mode == "rotation":
        correction_branch = "rotation"
    else:
        correction_branch = "planner"

    sampled_error_mode = None
    if isinstance(dyn_state, dict):
        sampled_error_mode = dyn_state.get("error_mode")

    corr_meta = {
        "correction_generated": bool(force_generate_correction),
        "min_dist_triggered": bool(min_dist_triggered),
        "closed_loop_fallback_used": bool(force_generate_correction and (not min_dist_triggered)),
        "failure_mode": failure_mode_key,
        "sampled_phase_key": str(phase_key_fixed),
        "sampled_phase_bin_id": sampled_phase_bin_fixed,
        "sampled_phase_instance_id": sampled_phase_instance_fixed,
        "forced_error_mode": forced_error_mode_key,
        "sampled_active_arm_pattern": sampled_active_arm_pattern_key,
        "forced_dir_bin_id": forced_dir_bin_fixed,
        "forced_mag_bin_id": forced_mag_bin_fixed,
        "sampled_mode_prob": (
            None if sampled_mode_prob is None or (not np.isfinite(float(sampled_mode_prob)))
            else float(sampled_mode_prob)
        ),
        "sampled_entry_prob_within_mode": (
            None
            if sampled_entry_prob_within_mode is None or (not np.isfinite(float(sampled_entry_prob_within_mode)))
            else float(sampled_entry_prob_within_mode)
        ),
        "sampled_unit_prob": (
            None if sampled_unit_prob is None or (not np.isfinite(float(sampled_unit_prob)))
            else float(sampled_unit_prob)
        ),
        "correction_branch": correction_branch,
        "sampled_error_mode": sampled_error_mode,
        "nearest_mode": nearest_mode,
        "target_mode": target_mode,
        "rollout_exec_steps": int(rollout_exec_steps),
        "t_star": int(t_star),
        "t_target": int(t_target),
        "min_dist": float(min_dist),
        "recover_eval_enable": bool(recover_eval_enable),
        "recover_eval_any_unrecoverable": bool(recover_eval_any_unrecoverable),
        "recover_eval_first_unrecoverable_step": (
            None if recover_eval_first_unrecoverable_step is None else int(recover_eval_first_unrecoverable_step)
        ),
        "recover_eval_last": recover_eval_last,
        "trigger_mode": ("real_error" if real_error_trigger_active else "min_dist_recovery"),
        "trigger_source": min_dist_trigger_source,
        "min_dist_trigger_step": (None if min_dist_trigger_step is None else int(min_dist_trigger_step)),
        "min_dist_trigger_dist": (None if min_dist_trigger_dist is None else float(min_dist_trigger_dist)),
        "min_dist_trigger_delta": (None if min_dist_trigger_delta is None else float(min_dist_trigger_delta)),
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
