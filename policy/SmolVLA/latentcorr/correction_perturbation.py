from __future__ import annotations

import numpy as np

from policy.SmolVLA.latentcorr.failure_utils import get_failure_param_bins
from policy.SmolVLA.latentcorr.correction_utils import resample_trajectory
from policy.SmolVLA.latentcorr.phase_utils import infer_phase_key_from_gt_window as _infer_phase_key_from_gt_window


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


def _eef_local_axis_from_wxyz(quat_wxyz, local_axis, fallback_axis):
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4,)
    if not np.all(np.isfinite(q)):
        return np.asarray(fallback_axis, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.asarray(fallback_axis, dtype=np.float32)
    q = q / n
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
    try:
        from scipy.spatial.transform import Rotation as R
        axis = R.from_quat(q_xyzw).apply(np.asarray(local_axis, dtype=np.float32))
        uv = _unit_vec3(axis)
        if uv is not None:
            return uv.astype(np.float32)
    except Exception:
        pass
    return np.asarray(fallback_axis, dtype=np.float32)

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
    # Use non-zero bins: (1/N, 2/N, ..., N/N) * max_value.
    scale = float(b + 1) / float(n)
    return float(max_value) * scale, b


def _front_hemisphere_dir_from_bin(quat_wxyz, dir_bin_id, n_dir, theta_deg=60.0):
    fwd = _eef_forward_dir_from_wxyz(quat_wxyz)
    fwd, u, v = _build_basis_from_forward(fwd)
    n = int(max(1, int(n_dir)))
    idx = int(np.clip(int(dir_bin_id), 0, n - 1))
    if n == 1:
        return np.asarray(fwd, dtype=np.float32), idx
    # Clockwise around forward axis.
    phi = -2.0 * np.pi * (float(idx) / float(n))
    theta = np.deg2rad(float(theta_deg))
    d = (
        np.cos(theta) * fwd
        + np.sin(theta) * (np.cos(phi) * u + np.sin(phi) * v)
    )
    du = _unit_vec3(d)
    if du is None:
        du = fwd
    return np.asarray(du, dtype=np.float32), idx


def _translation_dir_from_bin(quat_wxyz, dir_bin_id):
    n = int(max(1, get_failure_param_bins()["translation_dir_bins"]))
    local_axes = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    idx = int(np.clip(int(dir_bin_id), 0, min(n, len(local_axes)) - 1))
    direction = _eef_local_axis_from_wxyz(
        quat_wxyz,
        local_axis=local_axes[idx],
        fallback_axis=local_axes[idx],
    )
    return np.asarray(direction, dtype=np.float32), idx


def _rotation_axis_from_bin(quat_wxyz, dir_bin_id):
    # Six orientation directions: +/- roll, +/- yaw, +/- pitch in gripper-local axes.
    # Local axes are mapped as roll=X, yaw=Z, pitch=Y.
    axes = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    n = int(max(1, get_failure_param_bins()["rotation_dir_bins"]))
    idx = int(np.clip(int(dir_bin_id), 0, min(n, len(axes)) - 1))
    axis = _eef_local_axis_from_wxyz(
        quat_wxyz,
        local_axis=axes[idx],
        fallback_axis=axes[idx],
    )
    return np.asarray(axis, dtype=np.float32), idx


def _rotation_sign_from_bin(dir_bin_id, n_dir):
    n = int(max(1, int(n_dir)))
    idx = int(np.clip(int(dir_bin_id), 0, n - 1))
    return (1.0 if idx % 2 == 0 else -1.0), idx

def _perturb_action_chunk_target_pose(
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
    """Perturb sampled-start pose, then plan current state to perturbed target pose."""
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

    # Current perturbation design uses sampled-start reference directly.
    # Keep target indices fixed to the first step in sliced GT.
    idx = 0
    l_pose_gt = left_ep[idx].astype(np.float32).copy()   # [x,y,z,qw,qx,qy,qz]
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
        _, axis_bin_id = _rotation_sign_from_bin(dir_bin_id, n_dir)
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
        # Rotate around the unified sampled-start target pose.
        angle_max_deg, _ = _mag_value_from_bin(
            float(cfg.get("perturb_rot_max_deg", 15.0)),
            mag_bin_id,
            n_mag,
        )
        rot_sign, _ = _rotation_sign_from_bin(dir_bin_id, n_dir)
        angle = np.deg2rad(angle_max_deg) * float(rot_sign)
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
        # Translate from the same sampled-start target pose.
        fail_gain, _ = _mag_value_from_bin(
            float(cfg.get("perturb_eef_fail_gain", 0.05)),
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
    else:
        # For gripper_close, keep arms fixed at sampled start (current qpos).
        out[:, 0:6] = np.repeat(curr_left_q[None, :], T, axis=0).astype(np.float32)
        out[:, 7:13] = np.repeat(curr_right_q[None, :], T, axis=0).astype(np.float32)

    if mode == "gripper_close":
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
    else:
        # For translation/rotation, only arm target is perturbed.
        # Keep grippers fixed at the sampled start state.
        out[:, 6] = float(init_left_grip)
        out[:, 13] = float(init_right_grip)

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

def perturb_action_chunk_online(
    act_raw, cfg, dyn_state, fk=None, planner_l=None, planner_r=None,
    phase_key=None, init_left_grip=0.0, init_right_grip=0.0,
    curr_left_q=None, curr_right_q=None,
    active_left_arm=True, active_right_arm=True,
    active_left_gripper=True, active_right_gripper=True,
    left_ep=None, right_ep=None, left_grip_traj=None, right_grip_traj=None,
    t_star=None, rollout_exec_steps=None, forced_error_mode=None,
    forced_dir_bin_id=-1, forced_mag_bin_id=-1
):
    """
    Apply forced perturbation mode/dir-bin/mag-bin on action chunk.
    In failure train, mode and bins are provided by dataloader.
    """
    if not cfg.get("enable_perturb", False):
        return act_raw, False, "disabled", dyn_state

    del phase_key
    selected_mode = None if forced_error_mode is None else str(forced_error_mode).strip().lower()
    if selected_mode is None or selected_mode == "":
        return act_raw, False, "missing_forced_error_mode", dyn_state
    if selected_mode not in {"translation", "rotation", "gripper_close"}:
        return act_raw, False, "invalid_forced_error_mode", {"error_mode": selected_mode}
    if selected_mode == "gripper_close":
        close_min = float(np.clip(cfg.get("perturb_gripper_close_min", 0.10), 0.0, 1.0))
        l_open = bool(active_left_gripper) and bool(float(np.clip(init_left_grip, 0.0, 1.0)) > close_min)
        r_open = bool(active_right_gripper) and bool(float(np.clip(init_right_grip, 0.0, 1.0)) > close_min)
        if not (l_open or r_open):
            return act_raw, False, "invalid_forced_gripper_close_not_open", {"error_mode": selected_mode}

    if selected_mode is not None:
        if (left_ep is not None) and (right_ep is not None):
            if selected_mode in {"translation", "rotation", "gripper_close"}:
                out, ok, tag, info = _perturb_action_chunk_target_pose(
                    act_raw, cfg, fk, planner_l, planner_r,
                    curr_left_q, curr_right_q, init_left_grip, init_right_grip,
                    left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
                    forced_dir_bin_id=forced_dir_bin_id,
                    forced_mag_bin_id=forced_mag_bin_id,
                    error_mode=selected_mode,
                    active_left_arm=active_left_arm, active_right_arm=active_right_arm,
                    active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper,
                )
            else:
                out, ok, tag, info = act_raw, False, "unsupported_mode", {"error_mode": selected_mode}
        else:
            out, ok, tag, info = act_raw, False, "target_pose_missing_context", {"error_mode": selected_mode}
        if isinstance(info, dict):
            info["error_mode"] = selected_mode
        return out, ok, f"mode_{tag}", info
    return act_raw, False, "skip_mode_none", dyn_state
