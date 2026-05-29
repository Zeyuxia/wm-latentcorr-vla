from __future__ import annotations

import numpy as np

try:
    from .phase_utils import infer_phase_key_from_gt_window as shared_infer_phase_key_from_gt_window
    from .utils_common import resample_trajectory
    from ..failure_utils import get_failure_param_bins
except ImportError:
    from act_aligned_pkg.phase_utils import infer_phase_key_from_gt_window as shared_infer_phase_key_from_gt_window
    from act_aligned_pkg.utils_common import resample_trajectory
    from policy.ACT_LatentCorr.failure_utils import get_failure_param_bins

def _infer_phase_key_from_gt_window(left_grip_traj, right_grip_traj, window_len):
    return shared_infer_phase_key_from_gt_window(left_grip_traj, right_grip_traj, window_len)

def _find_gripper_toggle_anchor_idx(left_grip_traj, right_grip_traj):
    """Use the earliest gripper open/close switch as a fixed anchor index."""
    left = np.asarray(left_grip_traj, dtype=np.float32).reshape(-1)
    right = np.asarray(right_grip_traj, dtype=np.float32).reshape(-1)
    n = int(min(len(left), len(right)))
    if n <= 1:
        return 0
    l_bin = (left[:n] > 0.5).astype(np.int32)
    r_bin = (right[:n] > 0.5).astype(np.int32)
    toggles = np.where((l_bin[1:] != l_bin[:-1]) | (r_bin[1:] != r_bin[:-1]))[0]
    if toggles.size > 0:
        return int(toggles[0] + 1)
    # No switch in this episode slice: fallback to final point.
    return int(n - 1)

def _find_next_gripper_toggle_idx_from(left_grip_traj, right_grip_traj, start_idx):
    """Find first gripper open/close switch at/after start_idx; fallback to final point."""
    left = np.asarray(left_grip_traj, dtype=np.float32).reshape(-1)
    right = np.asarray(right_grip_traj, dtype=np.float32).reshape(-1)
    n = int(min(len(left), len(right)))
    if n <= 1:
        return 0
    s = int(np.clip(int(start_idx), 0, n - 1))
    l_bin = (left[:n] > 0.5).astype(np.int32)
    r_bin = (right[:n] > 0.5).astype(np.int32)
    toggles = np.where((l_bin[1:] != l_bin[:-1]) | (r_bin[1:] != r_bin[:-1]))[0] + 1
    toggles = toggles[toggles >= s]
    if toggles.size > 0:
        return int(toggles[0])
    return int(n - 1)

def _infer_active_arms_from_gt_window(
    gt_left_arm, gt_right_arm, gt_left_grip, gt_right_grip,
    t_idx, window_len, joint_delta_thresh=0.02, gripper_delta_thresh=0.05
):
    """Infer active arm/gripper sides from GT trajectories near current index."""
    def _bounds(n):
        if n <= 1:
            return 0, 1
        s = int(np.clip(t_idx, 0, n - 1))
        e = int(np.clip(s + max(2, int(window_len)), s + 1, n))
        return s, e

    def _arm_active(arr):
        if arr is None:
            return True, 0.0
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] <= 1:
            return True, 0.0
        s, e = _bounds(a.shape[0])
        seg = a[s:e]
        if seg.shape[0] <= 1:
            return True, 0.0
        dq = np.diff(seg, axis=0)
        score = float(np.max(np.linalg.norm(dq, axis=1))) if dq.shape[0] > 0 else 0.0
        return bool(score >= float(joint_delta_thresh)), score

    def _grip_active(arr):
        if arr is None:
            return True, 0.0
        g = np.asarray(arr, dtype=np.float32).reshape(-1)
        if g.size <= 1:
            return True, 0.0
        s, e = _bounds(g.size)
        seg = g[s:e]
        if seg.size <= 1:
            return True, 0.0
        dg = np.diff(seg)
        score = float(np.max(np.abs(dg))) if dg.size > 0 else 0.0
        return bool(score >= float(gripper_delta_thresh)), score

    la, la_s = _arm_active(gt_left_arm)
    ra, ra_s = _arm_active(gt_right_arm)
    lg, lg_s = _grip_active(gt_left_grip)
    rg, rg_s = _grip_active(gt_right_grip)

    if (not la) and (not ra):
        la, ra = True, True

    return {
        "left_arm": bool(la), "right_arm": bool(ra),
        "left_gripper": bool(lg), "right_gripper": bool(rg),
        "left_arm_score": float(la_s), "right_arm_score": float(ra_s),
        "left_gripper_score": float(lg_s), "right_gripper_score": float(rg_s),
    }

def _sample_unit_vec3():
    v = np.random.normal(0.0, 1.0, size=(3,)).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return v / n

def _unit_vec3(v):
    arr = np.asarray(v, dtype=np.float32).reshape(3,)
    n = float(np.linalg.norm(arr))
    if n < 1e-8:
        return None
    return arr / n

def _eef_forward_dir_from_wxyz(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4,)
    if not np.all(np.isfinite(q)):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    q = q / n
    # scipy uses xyzw
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
    try:
        from scipy.spatial.transform import Rotation as R
        fwd = R.from_quat(q_xyzw).apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        uv = _unit_vec3(fwd)
        if uv is not None:
            return uv.astype(np.float32)
    except Exception:
        pass
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)

def _forward_away_dir(forward_dir, curr_pos, anchor_pos):
    fwd = _unit_vec3(forward_dir)
    if fwd is None:
        fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    away = _unit_vec3(np.asarray(curr_pos, dtype=np.float32) - np.asarray(anchor_pos, dtype=np.float32))
    if away is None:
        return fwd.astype(np.float32)
    mix = _unit_vec3(np.asarray(fwd, dtype=np.float32) + np.asarray(away, dtype=np.float32))
    if mix is None:
        return fwd.astype(np.float32)
    # Keep translation inside forward hemisphere.
    if float(np.dot(mix, fwd)) < 0.0:
        return fwd.astype(np.float32)
    return mix.astype(np.float32)

def _sample_in_forward_hemisphere(forward_dir):
    """Sample a random unit direction on the forward hemisphere around forward_dir."""
    fwd = _unit_vec3(forward_dir)
    if fwd is None:
        fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Build an orthonormal basis {u, v, fwd}.
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(fwd, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = _unit_vec3(np.cross(ref, fwd))
    if u is None:
        return fwd.astype(np.float32)
    v = _unit_vec3(np.cross(fwd, u))
    if v is None:
        return fwd.astype(np.float32)

    # Spherical on hemisphere: theta in [0, pi/2], phi in [0, 2pi].
    phi = float(np.random.uniform(0.0, 2.0 * np.pi))
    z = float(np.random.uniform(0.0, 1.0))  # cos(theta), hemisphere => non-negative
    r = float(np.sqrt(max(0.0, 1.0 - z * z)))
    d = r * np.cos(phi) * u + r * np.sin(phi) * v + z * fwd
    d = _unit_vec3(d)
    if d is None:
        return fwd.astype(np.float32)
    return d.astype(np.float32)

def _build_basis_from_forward(forward_dir):
    fwd = _unit_vec3(forward_dir)
    if fwd is None:
        fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(fwd, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = _unit_vec3(np.cross(ref, fwd))
    if u is None:
        u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = _unit_vec3(np.cross(fwd, u))
    if v is None:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (
        np.asarray(fwd, dtype=np.float32),
        np.asarray(u, dtype=np.float32),
        np.asarray(v, dtype=np.float32),
    )

def _mag_value_from_bin(max_value, mag_bin_id, n_mag):
    n = int(max(1, int(n_mag)))
    b = int(np.clip(int(mag_bin_id), 0, n - 1))
    scale = float(b + 1) / float(n)
    return float(max_value) * scale, b

def _front_hemisphere_dir_from_bin(quat_wxyz, dir_bin_id, n_dir, theta_deg=60.0):
    fwd = _eef_forward_dir_from_wxyz(quat_wxyz)
    fwd, u, v = _build_basis_from_forward(fwd)
    n = int(max(1, int(n_dir)))
    idx = int(np.clip(int(dir_bin_id), 0, n - 1))
    if n == 1:
        return np.asarray(fwd, dtype=np.float32), idx
    phi = -2.0 * np.pi * (float(idx) / float(n))
    theta = np.deg2rad(float(theta_deg))
    d = np.cos(theta) * fwd + np.sin(theta) * (np.cos(phi) * u + np.sin(phi) * v)
    du = _unit_vec3(d)
    if du is None:
        du = fwd
    return np.asarray(du, dtype=np.float32), idx

def _translation_dir_from_bin(quat_wxyz, dir_bin_id):
    n = int(max(1, get_failure_param_bins()["translation_dir_bins"]))
    return _front_hemisphere_dir_from_bin(quat_wxyz, dir_bin_id, n_dir=n, theta_deg=60.0)

def _rotation_axis_from_bin(quat_wxyz, dir_bin_id):
    n = int(max(1, get_failure_param_bins()["rotation_dir_bins"]))
    return _front_hemisphere_dir_from_bin(quat_wxyz, dir_bin_id, n_dir=n, theta_deg=60.0)

def _traj_dir_after_anchor(ep, anchor_idx, lookahead=128):
    d, _ = _traj_dir_after_anchor_with_idx(ep, anchor_idx, lookahead=lookahead)
    return d

def _traj_dir_after_anchor_with_idx(ep, anchor_idx, lookahead=128):
    if ep is None:
        return None, None
    ep = np.asarray(ep, dtype=np.float32)
    if ep.ndim != 2 or ep.shape[0] < 2:
        return None, None
    n = int(ep.shape[0])
    s = int(np.clip(int(anchor_idx), 0, n - 2))
    j = int(min(n - 1, s + int(max(1, lookahead))))
    d = _unit_vec3(ep[j, :3] - ep[s, :3])
    return (None if d is None else d.astype(np.float32)), int(j)

def _sample_error_mode(phase_key, cfg):
    req = str(cfg.get("perturb_error_mode", "legacy")).strip().lower()
    if req == "open_laptop_pregrasp":
        # Task-specific mode: only perturb in pregrasp stage.
        if str(phase_key).strip().lower() != "pregrasp":
            return None
        p_close = float(max(0.0, cfg.get("perturb_open_laptop_pregrasp_close_prob", 0.8)))
        p_trans = float(max(0.0, cfg.get("perturb_open_laptop_pregrasp_translation_prob", 0.1)))
        p_rot = float(max(0.0, cfg.get("perturb_open_laptop_pregrasp_rotation_prob", 0.1)))
        names = ["gripper_close", "translation", "rotation"]
        p = np.array([p_close, p_trans, p_rot], dtype=np.float64)
        if float(np.sum(p)) <= 1e-12:
            p = np.array([0.8, 0.1, 0.1], dtype=np.float64)
        p = p / np.sum(p)
        return np.random.choice(names, p=p).item()
    if req in {"translation", "rotation", "gripper_close"}:
        return req
    if req != "auto":
        return None
    probs = {
        "approach": [("translation", 0.55), ("rotation", 0.25), ("gripper_close", 0.20)],
        "pregrasp": [("gripper_close", 0.65), ("translation", 0.25), ("rotation", 0.10)],
        "transport": [("translation", 0.75), ("rotation", 0.25)],
        "place": [("translation", 0.75), ("rotation", 0.25)],
    }
    items = probs.get(phase_key, probs["transport"])
    names = [x[0] for x in items]
    p = np.array([x[1] for x in items], dtype=np.float64)
    p = p / np.sum(p)
    return np.random.choice(names, p=p).item()

def _gate_gripper_error_mode(
    selected_mode, init_left_grip, init_right_grip, cfg,
    active_left_gripper=True, active_right_gripper=True
):
    mode = str(selected_mode).strip().lower() if selected_mode is not None else None
    if mode != "gripper_close":
        return selected_mode
    close_min = float(np.clip(cfg.get("perturb_gripper_close_min", 0.35), 0.0, 1.0))
    gl = float(np.clip(init_left_grip, 0.0, 1.0))
    gr = float(np.clip(init_right_grip, 0.0, 1.0))
    l_on = bool(active_left_gripper)
    r_on = bool(active_right_gripper)
    if (not l_on) and (not r_on):
        return "translation"
    need_close = bool((l_on and gl > close_min) or (r_on and gr > close_min))
    return "gripper_close" if need_close else "translation"

def _perturb_action_chunk_target_pose_forced(
    act_raw, cfg, fk, planner_l, planner_r,
    curr_left_q, curr_right_q, init_left_grip, init_right_grip,
    left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
    forced_dir_bin_id=-1, forced_mag_bin_id=-1,
    error_mode="translation",
    active_left_arm=True, active_right_arm=True,
    active_left_gripper=True, active_right_gripper=True,
    forced_dir_left=None, forced_dir_right=None,
    forced_axis_left=None, forced_axis_right=None,
):
    """Failure-table forced perturbation path aligned with upstream ACT train/explore."""
    del left_grip_traj, right_grip_traj, t_star, rollout_exec_steps
    if fk is None or planner_l is None or planner_r is None:
        return act_raw, False, "target_pose_missing_modules", None

    import sapien
    from scipy.spatial.transform import Rotation as R

    T = int(max(1, int(act_raw.shape[0])))
    out = np.asarray(act_raw, dtype=np.float32).copy()
    if T <= 0 or left_ep is None or right_ep is None or len(left_ep) == 0:
        return out, False, "target_pose_missing_gt", None

    active_left_arm = bool(active_left_arm)
    active_right_arm = bool(active_right_arm)

    idx = 0
    l_pose_gt = left_ep[idx].astype(np.float32).copy()
    r_pose_gt = right_ep[idx].astype(np.float32).copy()
    l_pose = l_pose_gt.copy()
    r_pose = r_pose_gt.copy()

    mode = str(error_mode).lower()
    bin_cfg = get_failure_param_bins()
    if mode == "translation":
        n_dir = int(max(1, bin_cfg["translation_dir_bins"]))
        n_mag = int(max(1, bin_cfg["translation_mag_bins"]))
        dir_bin_id = int(np.clip(int(forced_dir_bin_id), 0, n_dir - 1))
        mag_bin_id = int(np.clip(int(forced_mag_bin_id), 0, n_mag - 1))
        axis_bin_id = None
    elif mode == "rotation":
        n_dir = int(max(1, bin_cfg["rotation_dir_bins"]))
        n_mag = int(max(1, bin_cfg["rotation_mag_bins"]))
        dir_bin_id = int(np.clip(int(forced_dir_bin_id), 0, n_dir - 1))
        mag_bin_id = int(np.clip(int(forced_mag_bin_id), 0, n_mag - 1))
        axis_bin_id = int(dir_bin_id)
    else:
        n_dir = 0
        n_mag = 0
        dir_bin_id = -1
        mag_bin_id = -1
        axis_bin_id = None

    curr_left_q = np.asarray(curr_left_q, dtype=np.float32)
    curr_right_q = np.asarray(curr_right_q, dtype=np.float32)
    sampled_axis_l = None
    sampled_axis_r = None
    sampled_dir_l = None
    sampled_dir_r = None
    sampled_rotation_deg = None
    sampled_translation_gain_m = None

    if mode == "rotation":
        angle_max_deg, _ = _mag_value_from_bin(
            float(cfg.get("perturb_rot_max_deg", 15.0)),
            mag_bin_id,
            n_mag,
        )
        angle = np.deg2rad(angle_max_deg)
        sampled_rotation_deg = float(angle_max_deg)
        axis_l = None
        axis_r = None
        if forced_axis_left is not None:
            axis_l = np.asarray(forced_axis_left, dtype=np.float32)
        if forced_axis_right is not None:
            axis_r = np.asarray(forced_axis_right, dtype=np.float32)
        if forced_axis_left is None:
            axis_l, _ = _rotation_axis_from_bin(l_pose[3:7], int(dir_bin_id if dir_bin_id >= 0 else 0))
        if forced_axis_right is None:
            axis_r, _ = _rotation_axis_from_bin(r_pose[3:7], int(dir_bin_id if dir_bin_id >= 0 else 0))
        axis_l = _unit_vec3(axis_l) if axis_l is not None else _sample_unit_vec3()
        axis_r = _unit_vec3(axis_r) if axis_r is not None else _sample_unit_vec3()
        sampled_axis_l = np.asarray(axis_l, dtype=np.float32).tolist()
        sampled_axis_r = np.asarray(axis_r, dtype=np.float32).tolist()
        if active_left_arm:
            ql_xyzw = np.array([l_pose[4], l_pose[5], l_pose[6], l_pose[3]], dtype=np.float32)
            ql_new = (R.from_rotvec(np.asarray(axis_l, dtype=np.float32) * angle) * R.from_quat(ql_xyzw)).as_quat()
            l_pose[3:7] = np.array([ql_new[3], ql_new[0], ql_new[1], ql_new[2]], dtype=np.float32)
        if active_right_arm:
            qr_xyzw = np.array([r_pose[4], r_pose[5], r_pose[6], r_pose[3]], dtype=np.float32)
            qr_new = (R.from_rotvec(np.asarray(axis_r, dtype=np.float32) * angle) * R.from_quat(qr_xyzw)).as_quat()
            r_pose[3:7] = np.array([qr_new[3], qr_new[0], qr_new[1], qr_new[2]], dtype=np.float32)
        mode_tag = "target_pose_rotation"
    elif mode == "translation":
        fail_gain, _ = _mag_value_from_bin(
            float(cfg.get("perturb_eef_fail_gain", 0.03)),
            mag_bin_id,
            n_mag,
        )
        sampled_translation_gain_m = float(fail_gain)
        dir_l = None
        dir_r = None
        if forced_dir_left is not None:
            dir_l = np.asarray(forced_dir_left, dtype=np.float32)
        if forced_dir_right is not None:
            dir_r = np.asarray(forced_dir_right, dtype=np.float32)
        if dir_l is None:
            dir_l, _ = _translation_dir_from_bin(l_pose[3:7], int(dir_bin_id if dir_bin_id >= 0 else 0))
        if dir_r is None:
            dir_r, _ = _translation_dir_from_bin(r_pose[3:7], int(dir_bin_id if dir_bin_id >= 0 else 0))
        sampled_dir_l = np.asarray(dir_l, dtype=np.float32).tolist()
        sampled_dir_r = np.asarray(dir_r, dtype=np.float32).tolist()
        if active_left_arm:
            l_pose[:3] = l_pose[:3] + fail_gain * np.asarray(dir_l, dtype=np.float32)
        if active_right_arm:
            r_pose[:3] = r_pose[:3] + fail_gain * np.asarray(dir_r, dtype=np.float32)
        mode_tag = "target_pose_translation"
    elif mode == "gripper_close":
        mode_tag = "target_pose_gripper_close"
    else:
        return act_raw, False, "target_pose_unknown_mode", {"error_mode": mode}

    qpos_full = np.zeros(len(fk.jnames), dtype=np.float32)
    qpos_full[fk.fl_idx] = curr_left_q
    qpos_full[fk.fr_idx] = curr_right_q

    def _fit_plan_to_len(plan_future, curr_q, out_len):
        p = np.asarray(plan_future, dtype=np.float32)
        if p.ndim != 2 or p.shape[0] == 0:
            return np.repeat(np.asarray(curr_q, dtype=np.float32)[None, :], out_len, axis=0)
        if p.shape[0] == out_len:
            return p.astype(np.float32)
        return resample_trajectory(p, out_len).astype(np.float32)

    lt = np.repeat(curr_left_q[None, :], T, axis=0)
    rt = np.repeat(curr_right_q[None, :], T, axis=0)

    if active_left_arm and mode != "gripper_close":
        res_l = planner_l.plan_path(qpos_full, sapien.Pose(l_pose[:3], l_pose[3:7]), arms_tag="left")
        if res_l.get("status") != "Success":
            return act_raw, False, "target_pose_left_plan_fail", {"target_idx": int(idx)}
        l_path = np.asarray(res_l["position"], dtype=np.float32)
        if l_path.ndim != 2 or l_path.shape[0] == 0:
            return act_raw, False, "target_pose_left_empty_path", {"target_idx": int(idx)}
        l_future = l_path[1:] if l_path.shape[0] > 1 else l_path
        lt = _fit_plan_to_len(l_future, curr_left_q, T)
    if active_right_arm and mode != "gripper_close":
        res_r = planner_r.plan_path(qpos_full, sapien.Pose(r_pose[:3], r_pose[3:7]), arms_tag="right")
        if res_r.get("status") != "Success":
            return act_raw, False, "target_pose_right_plan_fail", {"target_idx": int(idx)}
        r_path = np.asarray(res_r["position"], dtype=np.float32)
        if r_path.ndim != 2 or r_path.shape[0] == 0:
            return act_raw, False, "target_pose_right_empty_path", {"target_idx": int(idx)}
        r_future = r_path[1:] if r_path.shape[0] > 1 else r_path
        rt = _fit_plan_to_len(r_future, curr_right_q, T)

    if mode in {"translation", "rotation"}:
        out[:, 0:6] = lt
        out[:, 7:13] = rt
        out[:, 6] = np.asarray(act_raw[:, 6], dtype=np.float32)
        out[:, 13] = np.asarray(act_raw[:, 13], dtype=np.float32)
    else:
        out[:, 0:6] = np.repeat(curr_left_q[None, :], T, axis=0).astype(np.float32)
        out[:, 7:13] = np.repeat(curr_right_q[None, :], T, axis=0).astype(np.float32)
        lg_tgt = 0.0
        rg_tgt = 0.0
        l_prog = (np.arange(T, dtype=np.float32) + 1.0) / float(T)
        r_prog = (np.arange(T, dtype=np.float32) + 1.0) / float(T)
        for t in range(T):
            l_val = (1.0 - l_prog[t]) * float(init_left_grip) + l_prog[t] * lg_tgt
            r_val = (1.0 - r_prog[t]) * float(init_right_grip) + r_prog[t] * rg_tgt
            out[t, 6] = l_val if active_left_gripper else float(act_raw[t, 6])
            out[t, 13] = r_val if active_right_gripper else float(act_raw[t, 13])
        out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
        out[:, 13] = np.clip(out[:, 13], 0.0, 1.0)

    info = {
        "target_idx": int(idx),
        "dir_bin_id": (None if dir_bin_id < 0 else int(dir_bin_id)),
        "mag_bin_id": (None if mag_bin_id < 0 else int(mag_bin_id)),
        "axis_bin_id": (None if axis_bin_id is None else int(axis_bin_id)),
        "perturb_translation_gain_m": sampled_translation_gain_m,
        "perturb_rotation_deg": sampled_rotation_deg,
        "perturb_dir_left": sampled_dir_l,
        "perturb_dir_right": sampled_dir_r,
        "perturb_axis_left": sampled_axis_l,
        "perturb_axis_right": sampled_axis_r,
        "error_mode": mode,
    }
    return out, True, mode_tag, info

def _perturb_action_chunk_target_pose(
    act_raw, cfg, fk, planner_l, planner_r,
    curr_left_q, curr_right_q, init_left_grip, init_right_grip,
    left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
    error_mode="translation",
    active_left_arm=True, active_right_arm=True,
    active_left_gripper=True, active_right_gripper=True,
    forced_dir_left=None, forced_dir_right=None,
    forced_axis_left=None, forced_axis_right=None,
    perturb_anchor_idx=None,
    left_ep_full=None, right_ep_full=None,
    perturb_anchor_idx_abs=None,
    start_ts_offset=0,
):
    """Two-point perturbation: p0=current, p1=GT@exec_step, perturb p1, then plan p0->p1'."""
    if fk is None or planner_l is None or planner_r is None:
        return act_raw, False, "target_pose_missing_modules", None

    import sapien
    from scipy.spatial.transform import Rotation as R

    T = int(max(1, int(act_raw.shape[0])))
    out = np.asarray(act_raw, dtype=np.float32).copy()
    if T <= 0 or left_ep is None or right_ep is None or len(left_ep) == 0:
        return out, False, "target_pose_missing_gt", None

    active_left_arm = bool(active_left_arm)
    active_right_arm = bool(active_right_arm)

    target_horizon = int(rollout_exec_steps) if rollout_exec_steps is not None else T
    target_horizon = int(max(1, target_horizon))
    idx = int(np.clip(int(t_star) + target_horizon - 1, 0, len(left_ep) - 1))
    local_target_idx = int(np.clip(idx - int(max(0, int(t_star))), 0, T - 1))
    l_pose_gt = left_ep[idx].astype(np.float32).copy()   # [x,y,z,qw,qx,qy,qz]
    r_pose_gt = right_ep[idx].astype(np.float32).copy()
    l_pose = l_pose_gt.copy()
    r_pose = r_pose_gt.copy()

    mag_rand = bool(cfg.get("perturb_mag_random", False))
    mag_min = float(cfg.get("perturb_mag_rand_min", 0.8))
    mag_max = float(cfg.get("perturb_mag_rand_max", 1.2))
    if mag_max < mag_min:
        mag_min, mag_max = mag_max, mag_min
    mag_scale = float(np.random.uniform(mag_min, mag_max)) if mag_rand else 1.0

    mode = str(error_mode).lower()
    if mode == "rotation":
        # For rotation-only pregrasp setting, perturb around sampled start_ts pose.
        idx = 0
        local_target_idx = 0
    curr_left_q = np.asarray(curr_left_q, dtype=np.float32)
    curr_right_q = np.asarray(curr_right_q, dtype=np.float32)
    # Use the first unperturbed action pose as perturbation base.
    q_base = np.asarray(act_raw[0], dtype=np.float32)
    fr_base = fk.forward(np.asarray(q_base[0:6], dtype=np.float32), np.asarray(q_base[7:13], dtype=np.float32))
    l_base_p = np.asarray(fr_base["left"][0], dtype=np.float32)
    l_base_q = np.asarray(fr_base["left"][1], dtype=np.float32)
    r_base_p = np.asarray(fr_base["right"][0], dtype=np.float32)
    r_base_q = np.asarray(fr_base["right"][1], dtype=np.float32)
    l_pose_base = np.array([l_base_p[0], l_base_p[1], l_base_p[2], l_base_q[0], l_base_q[1], l_base_q[2], l_base_q[3]], dtype=np.float32)
    r_pose_base = np.array([r_base_p[0], r_base_p[1], r_base_p[2], r_base_q[0], r_base_q[1], r_base_q[2], r_base_q[3]], dtype=np.float32)
    anti_dbg = {}
    if (not active_left_arm) and (not active_right_arm) and mode != "gripper_close":
        return out, False, "target_pose_inactive_arms", None

    if mode == "rotation":
        # Use sampled start_ts GT pose as perturbation base.
        l_pose = np.asarray(left_ep[0], dtype=np.float32).copy()
        r_pose = np.asarray(right_ep[0], dtype=np.float32).copy()
        angle_max_deg = float(cfg.get("perturb_rot_max_deg", 15.0)) * mag_scale
        angle = np.deg2rad(angle_max_deg)
        fr_curr = fk.forward(curr_left_q, curr_right_q)
        axis_l = None
        axis_r = None
        if forced_axis_left is not None:
            axis_l = np.asarray(forced_axis_left, dtype=np.float32)
        if forced_axis_right is not None:
            axis_r = np.asarray(forced_axis_right, dtype=np.float32)
        if forced_axis_left is None and forced_axis_right is None:
            axis_l = _sample_unit_vec3()
            axis_r = _sample_unit_vec3()
        axis_l = _unit_vec3(axis_l) if axis_l is not None else _sample_unit_vec3()
        axis_r = _unit_vec3(axis_r) if axis_r is not None else _sample_unit_vec3()
        if active_left_arm:
            ql_xyzw = np.array([l_pose[4], l_pose[5], l_pose[6], l_pose[3]], dtype=np.float32)
            ql_new = (R.from_rotvec(np.asarray(axis_l, dtype=np.float32) * angle) * R.from_quat(ql_xyzw)).as_quat()
            l_pose[3:7] = np.array([ql_new[3], ql_new[0], ql_new[1], ql_new[2]], dtype=np.float32)
        if active_right_arm:
            qr_xyzw = np.array([r_pose[4], r_pose[5], r_pose[6], r_pose[3]], dtype=np.float32)
            qr_new = (R.from_rotvec(np.asarray(axis_r, dtype=np.float32) * angle) * R.from_quat(qr_xyzw)).as_quat()
            r_pose[3:7] = np.array([qr_new[3], qr_new[0], qr_new[1], qr_new[2]], dtype=np.float32)
        mode_tag = "target_pose_rotation"
    elif mode == "translation":
        # Translation perturbation base: sampled start_ts pose (left_ep[0]/right_ep[0]),
        # instead of the first predicted action pose.
        l_pose = np.asarray(left_ep[0], dtype=np.float32).copy()
        r_pose = np.asarray(right_ep[0], dtype=np.float32).copy()
        fail_gain = float(cfg.get("perturb_eef_fail_gain", 0.03)) * mag_scale
        dir_l = None
        dir_r = None
        if forced_dir_left is not None:
            dir_l = np.asarray(forced_dir_left, dtype=np.float32)
        if forced_dir_right is not None:
            dir_r = np.asarray(forced_dir_right, dtype=np.float32)
        if dir_l is None:
            dir_l = _sample_in_forward_hemisphere(_eef_forward_dir_from_wxyz(l_pose[3:7]))
        if dir_r is None:
            dir_r = _sample_in_forward_hemisphere(_eef_forward_dir_from_wxyz(r_pose[3:7]))
        if active_left_arm:
            l_pose[:3] = l_pose[:3] + fail_gain * np.asarray(dir_l, dtype=np.float32)
        if active_right_arm:
            r_pose[:3] = r_pose[:3] + fail_gain * np.asarray(dir_r, dtype=np.float32)
        mode_tag = "target_pose_translation"
    elif mode == "gripper_close":
        # Arm follows GT target, failure is mainly on gripper channel.
        mode_tag = f"target_pose_{mode}"
    else:
        return act_raw, False, "target_pose_unknown_mode", {"error_mode": mode}

    qpos_full = np.zeros(len(fk.jnames), dtype=np.float32)
    qpos_full[fk.fl_idx] = curr_left_q
    qpos_full[fk.fr_idx] = curr_right_q

    def _fit_plan_to_len(plan_future, curr_q, out_len):
        """Use planner prefix only: resample/pad to fixed output length."""
        p = np.asarray(plan_future, dtype=np.float32)
        if p.ndim != 2 or p.shape[0] == 0:
            return np.repeat(np.asarray(curr_q, dtype=np.float32)[None, :], out_len, axis=0)
        if p.shape[0] == out_len:
            return p.astype(np.float32)
        return resample_trajectory(p, out_len).astype(np.float32)

    def _time_warp_plan(plan_seq, gamma):
        """
        Time-warp a joint trajectory while keeping endpoints unchanged.
        gamma < 1.0 => faster progress at the beginning, slower near the end.
        """
        p = np.asarray(plan_seq, dtype=np.float32)
        if p.ndim != 2 or p.shape[0] <= 2:
            return p
        g = float(gamma)
        if not np.isfinite(g) or g <= 0.0 or abs(g - 1.0) < 1e-6:
            return p
        n = int(p.shape[0])
        src_t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        dst_t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        dst_t_warp = np.power(dst_t, g).astype(np.float32)
        out = np.zeros_like(p, dtype=np.float32)
        for j in range(p.shape[1]):
            out[:, j] = np.interp(dst_t_warp, src_t, p[:, j]).astype(np.float32)
        out[0] = p[0]
        out[-1] = p[-1]
        return out

    # For pure gripper-error modes, keep arm pose fixed at current state.
    if mode == "gripper_close":
        lt = np.repeat(curr_left_q[None, :], T, axis=0)
        rt = np.repeat(curr_right_q[None, :], T, axis=0)
    else:
        lt = np.repeat(curr_left_q[None, :], T, axis=0)
        rt = np.repeat(curr_right_q[None, :], T, axis=0)

    if active_left_arm and mode != "gripper_close":
        res_l = planner_l.plan_path(qpos_full, sapien.Pose(l_pose[:3], l_pose[3:7]), arms_tag="left")
        if res_l.get("status") != "Success":
            return act_raw, False, "target_pose_left_plan_fail", {"target_idx": int(idx)}
        l_path = np.asarray(res_l["position"], dtype=np.float32)
        if l_path.ndim != 2 or l_path.shape[0] == 0:
            return act_raw, False, "target_pose_left_empty_path", {"target_idx": int(idx)}
        # Planner path usually includes current state at index 0; action chunk should be future commands.
        l_future = l_path[1:] if l_path.shape[0] > 1 else l_path
        if mode == "translation":
            lift_gamma = float(cfg.get("perturb_translation_lift_gamma", 1.0))
            l_future = _time_warp_plan(l_future, lift_gamma)
        lt = _fit_plan_to_len(l_future, curr_left_q, T)
    if active_right_arm and mode != "gripper_close":
        res_r = planner_r.plan_path(qpos_full, sapien.Pose(r_pose[:3], r_pose[3:7]), arms_tag="right")
        if res_r.get("status") != "Success":
            return act_raw, False, "target_pose_right_plan_fail", {"target_idx": int(idx)}
        r_path = np.asarray(res_r["position"], dtype=np.float32)
        if r_path.ndim != 2 or r_path.shape[0] == 0:
            return act_raw, False, "target_pose_right_empty_path", {"target_idx": int(idx)}
        r_future = r_path[1:] if r_path.shape[0] > 1 else r_path
        if mode == "translation":
            lift_gamma = float(cfg.get("perturb_translation_lift_gamma", 1.0))
            r_future = _time_warp_plan(r_future, lift_gamma)
        rt = _fit_plan_to_len(r_future, curr_right_q, T)

    out[:, 0:6] = lt
    out[:, 7:13] = rt

    lg_gt = float(np.clip(left_grip_traj[idx], 0.0, 1.0)) if left_grip_traj is not None and len(left_grip_traj) > idx else float(init_left_grip)
    rg_gt = float(np.clip(right_grip_traj[idx], 0.0, 1.0)) if right_grip_traj is not None and len(right_grip_traj) > idx else float(init_right_grip)
    if mode == "gripper_close":
        lg_tgt = 0.0
        rg_tgt = 0.0
    else:
        lg_tgt, rg_tgt = lg_gt, rg_gt

    # Gripper interpolation:
    # - gripper_close: start changing from the 1st action and finish at step T.
    # - other modes: keep previous behavior (tail-fast fallback) for mild smoothing.
    if mode == "gripper_close":
        l_prog = (np.arange(T, dtype=np.float32) + 1.0) / float(T)
        r_prog = (np.arange(T, dtype=np.float32) + 1.0) / float(T)
    else:
        fast_ratio = float(np.clip(cfg.get("perturb_gripper_fast_ratio", 0.2), 0.02, 1.0))
        sw = int(np.clip(np.ceil(T * fast_ratio), 1, max(1, T)))
        hold_until = int(max(0, T - sw))
        l_prog = np.zeros((T,), dtype=np.float32)
        r_prog = np.zeros((T,), dtype=np.float32)
        for i in range(T):
            if i >= hold_until:
                p = float(i - hold_until) / float(max(1, sw - 1))
                l_prog[i] = np.clip(p, 0.0, 1.0)
                r_prog[i] = np.clip(p, 0.0, 1.0)

    for t in range(T):
        l_val = (1.0 - l_prog[t]) * float(init_left_grip) + l_prog[t] * lg_tgt
        r_val = (1.0 - r_prog[t]) * float(init_right_grip) + r_prog[t] * rg_tgt
        out[t, 6] = l_val if active_left_gripper else float(act_raw[t, 6])
        out[t, 13] = r_val if active_right_gripper else float(act_raw[t, 13])
    out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
    out[:, 13] = np.clip(out[:, 13], 0.0, 1.0)

    info = {
        "target_idx": int(idx),
        "perturb_anchor_idx": (None if perturb_anchor_idx is None else int(perturb_anchor_idx)),
        "perturb_anchor_idx_abs": (None if perturb_anchor_idx_abs is None else int(perturb_anchor_idx_abs)),
        "local_target_idx": int(local_target_idx),
        "perturb_mag_scale": float(mag_scale),
        "active_left_arm": bool(active_left_arm),
        "active_right_arm": bool(active_right_arm),
        "error_mode": mode,
    }
    info.update(anti_dbg)
    return out, True, mode_tag, info

def perturb_action_chunk_online(
    act_raw, cfg, dyn_state, fk=None, planner_l=None, planner_r=None,
    phase_key=None, init_left_grip=0.0, init_right_grip=0.0,
    curr_left_q=None, curr_right_q=None,
    active_left_arm=True, active_right_arm=True,
    active_left_gripper=True, active_right_gripper=True,
    left_ep=None, right_ep=None, left_grip_traj=None, right_grip_traj=None,
    t_star=None, rollout_exec_steps=None, forced_error_mode=None,
    forced_dir_bin_id=-1, forced_mag_bin_id=-1, perturb_anchor_idx=None,
    left_ep_full=None, right_ep_full=None, perturb_anchor_idx_abs=None, start_ts_offset=0
):
    """
    Dynamics-aware perturbation in action space:
    low-pass execution + AR(1) colored noise + optional bias +
    velocity/acceleration limits.
    """
    if not cfg.get("enable_perturb", False):
        return act_raw, False, "disabled", dyn_state
    forced_mode = None if forced_error_mode is None else str(forced_error_mode).strip().lower()
    if forced_mode:
        if forced_mode not in {"translation", "rotation", "gripper_close"}:
            return act_raw, False, "invalid_forced_error_mode", {"error_mode": forced_mode}
        if forced_mode == "gripper_close":
            close_min = float(np.clip(cfg.get("perturb_gripper_close_min", 0.10), 0.0, 1.0))
            l_open = bool(active_left_gripper) and bool(float(np.clip(init_left_grip, 0.0, 1.0)) > close_min)
            r_open = bool(active_right_gripper) and bool(float(np.clip(init_right_grip, 0.0, 1.0)) > close_min)
            if not (l_open or r_open):
                return act_raw, False, "invalid_forced_gripper_close_not_open", {"error_mode": forced_mode}
        if (left_ep is not None) and (right_ep is not None):
            out, ok, tag, info = _perturb_action_chunk_target_pose_forced(
                act_raw, cfg, fk, planner_l, planner_r,
                curr_left_q, curr_right_q, init_left_grip, init_right_grip,
                left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
                forced_dir_bin_id=forced_dir_bin_id,
                forced_mag_bin_id=forced_mag_bin_id,
                error_mode=forced_mode,
                active_left_arm=active_left_arm, active_right_arm=active_right_arm,
                active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper,
            )
        else:
            out, ok, tag, info = act_raw, False, "target_pose_missing_context", {"error_mode": forced_mode}
        if isinstance(info, dict):
            info["error_mode"] = forced_mode
        return out, ok, f"mode_{tag}", info
    prob = float(cfg.get("perturb_prob", 0.0))
    if prob <= 0.0 or np.random.rand() >= prob:
        return act_raw, False, "skip_prob", dyn_state

    selected_mode = _sample_error_mode(phase_key, cfg)
    selected_mode = _gate_gripper_error_mode(
        selected_mode, init_left_grip, init_right_grip, cfg,
        active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper
    )
    req_mode = str(cfg.get("perturb_error_mode", "legacy")).strip().lower()
    if req_mode == "open_laptop_pregrasp" and selected_mode is None:
        return act_raw, False, "skip_non_pregrasp", dyn_state
    if (
        req_mode == "open_laptop_pregrasp"
        and str(selected_mode).strip().lower() == "translation"
        and perturb_anchor_idx_abs is None
    ):
        return act_raw, False, "skip_translation_missing_pregrasp_seg_end", {
            "error_mode": selected_mode
        }
    if selected_mode is not None:
        if (t_star is not None) and (left_ep is not None) and (right_ep is not None):
            if selected_mode in {"translation", "rotation", "gripper_close"}:
                reject_enable = bool(cfg.get("perturb_reject_sampling_enable", True))
                reject_trials = int(max(1, cfg.get("perturb_reject_max_trials", 4)))
                reject_dir_jitter_eps = float(max(0.0, cfg.get("perturb_reject_dir_jitter_eps", 0.2)))
                do_reject = reject_enable and selected_mode in {"translation", "rotation"} and (fk is not None)
                if do_reject:
                    target_h = int(rollout_exec_steps) if rollout_exec_steps is not None else int(act_raw.shape[0])
                    target_h = int(max(1, target_h))
                    if perturb_anchor_idx is not None:
                        gt_idx = int(np.clip(int(perturb_anchor_idx), 0, len(left_ep) - 1))
                    else:
                        gt_idx = int(np.clip(int(t_star) + target_h - 1, 0, len(left_ep) - 1))
                    fr_curr = fk.forward(curr_left_q, curr_right_q)
                    lp_curr = np.asarray(fr_curr["left"][0], dtype=np.float32)
                    rp_curr = np.asarray(fr_curr["right"][0], dtype=np.float32)
                    lq_curr = np.asarray(fr_curr["left"][1], dtype=np.float32)
                    rq_curr = np.asarray(fr_curr["right"][1], dtype=np.float32)
                    gt_lp = np.asarray(left_ep[gt_idx, :3], dtype=np.float32)
                    gt_rp = np.asarray(right_ep[gt_idx, :3], dtype=np.float32)
                    main_l = main_r = None
                    main_a_l = main_a_r = None
                    if selected_mode == "translation":
                        # Main direction: random forward-hemisphere direction from sampled start_ts pose.
                        main_l = (
                            _sample_in_forward_hemisphere(_eef_forward_dir_from_wxyz(np.asarray(left_ep[0, 3:7], dtype=np.float32)))
                            if active_left_arm else None
                        )
                        main_r = (
                            _sample_in_forward_hemisphere(_eef_forward_dir_from_wxyz(np.asarray(right_ep[0, 3:7], dtype=np.float32)))
                            if active_right_arm else None
                        )

                        def _sample_near(main_v):
                            if main_v is None:
                                return _sample_unit_vec3()
                            if reject_dir_jitter_eps <= 0.0:
                                return main_v.copy()
                            return _unit_vec3(main_v + reject_dir_jitter_eps * _sample_unit_vec3())
                    else:
                        from scipy.spatial.transform import Rotation as R
                        ql_gt = np.asarray(left_ep[gt_idx, 3:7], dtype=np.float32)
                        qr_gt = np.asarray(right_ep[gt_idx, 3:7], dtype=np.float32)
                        # Main axis is "away from anchor orientation":
                        # use -(curr->anchor rotvec direction), i.e. continue rotating away.
                        main_a_l = None
                        main_a_r = None
                        if active_left_arm:
                            q_curr_xyzw = np.array([lq_curr[1], lq_curr[2], lq_curr[3], lq_curr[0]], dtype=np.float32)
                            q_gt_xyzw = np.array([ql_gt[1], ql_gt[2], ql_gt[3], ql_gt[0]], dtype=np.float32)
                            gt_rv_l = _unit_vec3((R.from_quat(q_gt_xyzw) * R.from_quat(q_curr_xyzw).inv()).as_rotvec())
                            main_a_l = _unit_vec3(-gt_rv_l)
                        if active_right_arm:
                            q_curr_xyzw = np.array([rq_curr[1], rq_curr[2], rq_curr[3], rq_curr[0]], dtype=np.float32)
                            q_gt_xyzw = np.array([qr_gt[1], qr_gt[2], qr_gt[3], qr_gt[0]], dtype=np.float32)
                            gt_rv_r = _unit_vec3((R.from_quat(q_gt_xyzw) * R.from_quat(q_curr_xyzw).inv()).as_rotvec())
                            main_a_r = _unit_vec3(-gt_rv_r)

                        def _sample_axis_near(main_a):
                            if main_a is None:
                                return _sample_unit_vec3()
                            if reject_dir_jitter_eps <= 0.0:
                                return main_a.copy()
                            return _unit_vec3(main_a + reject_dir_jitter_eps * _sample_unit_vec3())
                    out, ok, tag, info = act_raw, False, "target_pose_reject_no_candidate", {
                        "error_mode": selected_mode,
                        "reject_sampling_enabled": True,
                        "reject_sampling_trials_max": int(reject_trials),
                        "reject_sampling_dir_jitter_eps": float(reject_dir_jitter_eps),
                    }
                    for ridx in range(reject_trials):
                        if selected_mode == "translation":
                            if ridx == 0:
                                dl = _sample_unit_vec3() if main_l is None else main_l.copy()
                                dr = _sample_unit_vec3() if main_r is None else main_r.copy()
                            else:
                                dl = _sample_near(main_l)
                                dr = _sample_near(main_r)
                            trial_out, trial_ok, trial_tag, trial_info = _perturb_action_chunk_target_pose(
                                act_raw, cfg, fk, planner_l, planner_r,
                                curr_left_q, curr_right_q, init_left_grip, init_right_grip,
                                left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
                                error_mode=selected_mode,
                                active_left_arm=active_left_arm, active_right_arm=active_right_arm,
                                active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper,
                                forced_dir_left=dl, forced_dir_right=dr,
                                perturb_anchor_idx=perturb_anchor_idx,
                                left_ep_full=left_ep_full, right_ep_full=right_ep_full,
                                perturb_anchor_idx_abs=perturb_anchor_idx_abs,
                                start_ts_offset=start_ts_offset,
                            )
                        else:
                            if ridx == 0:
                                al = _sample_unit_vec3() if main_a_l is None else main_a_l.copy()
                                ar = _sample_unit_vec3() if main_a_r is None else main_a_r.copy()
                            else:
                                al = _sample_axis_near(main_a_l)
                                ar = _sample_axis_near(main_a_r)
                            trial_out, trial_ok, trial_tag, trial_info = _perturb_action_chunk_target_pose(
                                act_raw, cfg, fk, planner_l, planner_r,
                                curr_left_q, curr_right_q, init_left_grip, init_right_grip,
                                left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
                                error_mode=selected_mode,
                                active_left_arm=active_left_arm, active_right_arm=active_right_arm,
                                active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper,
                                forced_axis_left=al, forced_axis_right=ar,
                                perturb_anchor_idx=perturb_anchor_idx,
                                left_ep_full=left_ep_full, right_ep_full=right_ep_full,
                                perturb_anchor_idx_abs=perturb_anchor_idx_abs,
                                start_ts_offset=start_ts_offset,
                            )
                        trial_info = {} if not isinstance(trial_info, dict) else trial_info
                        trial_info.update({
                            "reject_sampling_enabled": True,
                            "reject_sampling_trial": int(ridx + 1),
                            "reject_sampling_trials_max": int(reject_trials),
                            "reject_sampling_dir_jitter_eps": float(reject_dir_jitter_eps),
                            "reject_sampling_accepted": bool(trial_ok),
                            "reject_sampling_fallback_best": False,
                        })
                        if trial_ok:
                            out, ok, tag, info = trial_out, trial_ok, trial_tag, trial_info
                            break
                        out, ok, tag, info = trial_out, trial_ok, trial_tag, trial_info
                else:
                    out, ok, tag, info = _perturb_action_chunk_target_pose(
                        act_raw, cfg, fk, planner_l, planner_r,
                        curr_left_q, curr_right_q, init_left_grip, init_right_grip,
                        left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
                        error_mode=selected_mode,
                        active_left_arm=active_left_arm, active_right_arm=active_right_arm,
                        active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper,
                        perturb_anchor_idx=perturb_anchor_idx,
                        left_ep_full=left_ep_full, right_ep_full=right_ep_full,
                        perturb_anchor_idx_abs=perturb_anchor_idx_abs,
                        start_ts_offset=start_ts_offset,
                    )
            else:
                out, ok, tag, info = act_raw, False, "unsupported_mode", {"error_mode": selected_mode}
        else:
            out, ok, tag, info = act_raw, False, "target_pose_missing_context", {"error_mode": selected_mode}
        if isinstance(info, dict):
            info["error_mode"] = selected_mode
        return out, ok, f"mode_{tag}", info
    return act_raw, False, "skip_mode_none", dyn_state
