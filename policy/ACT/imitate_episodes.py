import os
import sys
import time
import re

# Set rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"
# Required for deterministic CuBLAS operations
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evac'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import pickle
import argparse

from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from phase_utils import infer_phase_key_from_gt_window as shared_infer_phase_key_from_gt_window
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, detach_dict  # helper functions
from act_policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython

e = IPython.embed


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def build_evac_infer_kwargs(cfg):
    del cfg
    return {}


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


def _perturb_action_chunk_target_pose(
    act_raw, cfg, fk, planner_l, planner_r,
    curr_left_q, curr_right_q, init_left_grip, init_right_grip,
    left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
    error_mode="translation",
    active_left_arm=True, active_right_arm=True,
    active_left_gripper=True, active_right_gripper=True,
    forced_dir_left=None, forced_dir_right=None,
    forced_axis_left=None, forced_axis_right=None,
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
    curr_left_q = np.asarray(curr_left_q, dtype=np.float32)
    curr_right_q = np.asarray(curr_right_q, dtype=np.float32)
    # Use the unperturbed action chunk terminal pose as perturbation base
    # (instead of GT target pose), so perturb_pre/post compare in the same frame.
    q_end = np.asarray(act_raw[-1], dtype=np.float32)
    fr_end = fk.forward(np.asarray(q_end[0:6], dtype=np.float32), np.asarray(q_end[7:13], dtype=np.float32))
    l_end_p = np.asarray(fr_end["left"][0], dtype=np.float32)
    l_end_q = np.asarray(fr_end["left"][1], dtype=np.float32)
    r_end_p = np.asarray(fr_end["right"][0], dtype=np.float32)
    r_end_q = np.asarray(fr_end["right"][1], dtype=np.float32)
    l_pose_base = np.array([l_end_p[0], l_end_p[1], l_end_p[2], l_end_q[0], l_end_q[1], l_end_q[2], l_end_q[3]], dtype=np.float32)
    r_pose_base = np.array([r_end_p[0], r_end_p[1], r_end_p[2], r_end_q[0], r_end_q[1], r_end_q[2], r_end_q[3]], dtype=np.float32)
    anti_dbg = {}
    if (not active_left_arm) and (not active_right_arm) and mode != "gripper_close":
        return out, False, "target_pose_inactive_arms", None

    if mode == "rotation":
        l_pose = l_pose_base.copy()
        r_pose = r_pose_base.copy()
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
        l_pose = l_pose_base.copy()
        r_pose = r_pose_base.copy()
        fail_gain = float(cfg.get("perturb_eef_fail_gain", 0.03)) * mag_scale
        fr_curr = fk.forward(curr_left_q, curr_right_q)
        lp_curr = np.asarray(fr_curr["left"][0], dtype=np.float32)
        rp_curr = np.asarray(fr_curr["right"][0], dtype=np.float32)
        dir_l = None
        dir_r = None
        if forced_dir_left is not None:
            dir_l = np.asarray(forced_dir_left, dtype=np.float32)
        if forced_dir_right is not None:
            dir_r = np.asarray(forced_dir_right, dtype=np.float32)
        if forced_dir_left is None and forced_dir_right is None:
            dir_l = _sample_unit_vec3()
            dir_r = _sample_unit_vec3()
        else:
            if dir_l is None:
                dir_l = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if dir_r is None:
                dir_r = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
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

    def _compose_plan_with_gt_tail(plan_future, gt_chunk, target_local_idx, out_len):
        """Keep short planner path; append GT tail after target; resample only when planner path is too long."""
        p = np.asarray(plan_future, dtype=np.float32)
        g = np.asarray(gt_chunk, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != g.shape[1] or p.shape[0] == 0:
            return g[:out_len].copy()
        if p.shape[0] >= out_len:
            return resample_trajectory(p, out_len).astype(np.float32)

        seq = [p]
        gt_tail_start = int(np.clip(target_local_idx + 1, 0, g.shape[0]))
        gt_tail = g[gt_tail_start:]
        if gt_tail.shape[0] > 0:
            seq.append(gt_tail)
        cat = np.concatenate(seq, axis=0)
        if cat.shape[0] >= out_len:
            return cat[:out_len].astype(np.float32)
        last = cat[-1:] if cat.shape[0] > 0 else g[-1:]
        pad = np.repeat(last.astype(np.float32), out_len - cat.shape[0], axis=0)
        return np.concatenate([cat, pad], axis=0).astype(np.float32)

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
        lt = _compose_plan_with_gt_tail(l_future, out[:, 0:6], local_target_idx, T)
    if active_right_arm and mode != "gripper_close":
        res_r = planner_r.plan_path(qpos_full, sapien.Pose(r_pose[:3], r_pose[3:7]), arms_tag="right")
        if res_r.get("status") != "Success":
            return act_raw, False, "target_pose_right_plan_fail", {"target_idx": int(idx)}
        r_path = np.asarray(res_r["position"], dtype=np.float32)
        if r_path.ndim != 2 or r_path.shape[0] == 0:
            return act_raw, False, "target_pose_right_empty_path", {"target_idx": int(idx)}
        r_future = r_path[1:] if r_path.shape[0] > 1 else r_path
        rt = _compose_plan_with_gt_tail(r_future, out[:, 7:13], local_target_idx, T)

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
    t_star=None, rollout_exec_steps=None, perturb_anchor_idx=None
):
    """
    Dynamics-aware perturbation in action space:
    low-pass execution + AR(1) colored noise + optional bias +
    velocity/acceleration limits.
    """
    if not cfg.get("enable_perturb", False):
        return act_raw, False, "disabled", dyn_state
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
                        # Main direction is "away from anchor": curr - anchor
                        main_l = _unit_vec3(lp_curr - gt_lp) if active_left_arm else None
                        main_r = _unit_vec3(rp_curr - gt_rp) if active_right_arm else None

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
                    )
            else:
                out, ok, tag, info = act_raw, False, "unsupported_mode", {"error_mode": selected_mode}
        else:
            out, ok, tag, info = act_raw, False, "target_pose_missing_context", {"error_mode": selected_mode}
        if isinstance(info, dict):
            info["error_mode"] = selected_mode
        return out, ok, f"mode_{tag}", info
    return act_raw, False, "skip_mode_none", dyn_state


def main(args):
    set_seed(int(args["seed"]))
    _rank = int(os.environ.get("RANK", -1))
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"[main][rank={_rank} local_rank={_local_rank}] start")
    # command line parameters
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args.get("onscreen_render", False)
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    num_epochs = args["num_epochs"]

    # get task parameters
    is_sim = task_name[:4] == "sim-"
    if is_sim:
        from constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14  # yiheng
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "save_freq": args['save_freq'],
        "sp_reg_enable": bool(args["sp_reg_enable"]),
        "sp_reg_lambda": float(args["sp_reg_lambda"]),
        "lr_sched_enable": bool(args["lr_sched_enable"]),
        "lr_warmup_steps": int(args["lr_warmup_steps"]),
        "lr_min_ratio": float(args["lr_min_ratio"]),
    }

    enable_wm = args['enable_wm_correction']
    start_margin = 0
    sample_skip_head = int(max(0, args['sample_skip_head']))
    sample_pregrasp_bias_enable = bool(args['sample_pregrasp_bias_enable'])
    sample_pregrasp_prob = float(args['sample_pregrasp_prob'])
    sample_pregrasp_phase_window_len = int(args['sample_pregrasp_phase_window_len'])
    sample_pregrasp_avoid_switch_tail = int(args['sample_pregrasp_avoid_switch_tail'])
    if enable_wm:
        wm_required = ['evac_ckpt', 'evac_config', 'urdf_path', 'curobo_left_yml',
                        'curobo_right_yml', 'raw_data_dir', 'act_init_ckpt',
                        'max_rollout_steps',
                        'correction_weight', 'orient_weight', 'gripper_penalty']
        missing = [k for k in wm_required if args.get(k) is None]
        if missing:
            raise ValueError(f"--enable_wm_correction requires these args: {missing}")
        # Reserve tail horizon based on actual rollout execution steps.
        if args['rollout_exec_steps'] is None:
            raise ValueError("rollout_exec_steps must be explicitly provided when enable_wm_correction=true")
        exec_steps = int(args['rollout_exec_steps'])
        start_margin = int(args['max_rollout_steps']) * int(exec_steps)
    raw_data_dir = args['raw_data_dir'] if enable_wm else None
    print(f"[main][rank={_rank}] before load_data | dataset_dir={dataset_dir} | num_episodes={num_episodes} | start_margin={start_margin}")
    _t_load = time.time()
    train_dataloader, _, stats, _, max_action_len = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_train,
        raw_data_dir=raw_data_dir,
        start_margin=start_margin,
        sample_skip_head=sample_skip_head,
        sample_pregrasp_bias_enable=sample_pregrasp_bias_enable,
        sample_pregrasp_prob=sample_pregrasp_prob,
        sample_pregrasp_phase_window_len=sample_pregrasp_phase_window_len,
        sample_pregrasp_avoid_switch_tail=sample_pregrasp_avoid_switch_tail,
    )
    print(f"[main][rank={_rank}] after load_data | elapsed={time.time() - _t_load:.2f}s | max_action_len={max_action_len}")

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    config['enable_wm_correction'] = enable_wm
    if enable_wm:
        config['wm_args'] = args
        config['raw_data_dir'] = raw_data_dir
        config['norm_stats'] = stats
        config['dataset_dir'] = dataset_dir
        rollout_exec_steps = args.get('rollout_exec_steps', None)
        if rollout_exec_steps is None or int(rollout_exec_steps) <= 0:
            rollout_exec_steps = int(args['chunk_size'])
        config['correction_cfg'] = {
            'max_rollout_steps': args['max_rollout_steps'],
            'target_mode': args['target_mode'],
            'target_lookahead_steps': args['target_lookahead_steps'],
            'min_dist_fallback_force_correction': args['min_dist_fallback_force_correction'],
            'min_dist_recover_ratio': args['min_dist_recover_ratio'],
            'debug_recover_eval_rollout': args['debug_recover_eval_rollout'],
            'correction_interp_nearest_enable': args['correction_interp_nearest_enable'],
            'correction_interp_prefix_ratio': args['correction_interp_prefix_ratio'],
            'rollout_exec_steps': int(rollout_exec_steps),
            'sample_pregrasp_phase_window_len': int(args['sample_pregrasp_phase_window_len']),
            'chunk_size': args['chunk_size'],
            'max_action_len': max_action_len,
            'correction_weight': args['correction_weight'],
            'orient_weight': args['orient_weight'],
            'gripper_penalty': args['gripper_penalty'],
            'recover_gripper_penalty': args['recover_gripper_penalty'],
            'evac_infer_kwargs': build_evac_infer_kwargs(args),
            'enable_perturb': args['enable_perturb'],
            'perturb_prob': args['perturb_prob'],
            'perturb_error_mode': args['perturb_error_mode'],
            'perturb_open_laptop_pregrasp_close_prob': args['perturb_open_laptop_pregrasp_close_prob'],
            'perturb_open_laptop_pregrasp_translation_prob': args['perturb_open_laptop_pregrasp_translation_prob'],
            'perturb_open_laptop_pregrasp_rotation_prob': args['perturb_open_laptop_pregrasp_rotation_prob'],
            'perturb_eef_fail_gain': args['perturb_eef_fail_gain'],
            'perturb_rot_max_deg': args['perturb_rot_max_deg'],
            'perturb_mag_random': args['perturb_mag_random'],
            'perturb_mag_rand_min': args['perturb_mag_rand_min'],
            'perturb_mag_rand_max': args['perturb_mag_rand_max'],
            'perturb_reject_sampling_enable': args['perturb_reject_sampling_enable'],
            'perturb_reject_max_trials': args['perturb_reject_max_trials'],
            'perturb_reject_dir_jitter_eps': args['perturb_reject_dir_jitter_eps'],
            'nearest_window_radius': args['nearest_window_radius'],
            'perturb_gripper_close_min': args['perturb_gripper_close_min'],
            'perturb_gripper_fast_ratio': args['perturb_gripper_fast_ratio'],
            'perturb_active_joint_delta_thresh': args['perturb_active_joint_delta_thresh'],
            'perturb_active_gripper_delta_thresh': args['perturb_active_gripper_delta_thresh'],
            'export_correction_dataset': args['export_correction_dataset'],
            'export_correction_dir': args['export_correction_dir'],
        }
    config['act_init_ckpt'] = args.get('act_init_ckpt')
    config['debug_wm_correction'] = args.get('debug_wm_correction', False)
    print(f"[main][rank={_rank}] before train_bc | enable_wm={enable_wm}")
    train_bc(train_dataloader, config)


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def init_correction(args, device):
    import sapien
    from omegaconf import OmegaConf
    # EVAC internal modules (e.g. ddpm3d) do `from utils.general_utils import ...`
    # which needs evac/evac/ on sys.path AND `utils` in sys.modules to point to
    # evac's utils package (not ACT's utils.py which is already cached).
    _evac_evac = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evac', 'evac')
    if _evac_evac not in sys.path:
        sys.path.insert(0, _evac_evac)
    # Temporarily swap out ACT's utils module so EVAC can load its own utils package
    _act_utils = sys.modules.pop('utils', None)
    from evac.utils.general_utils import load_checkpoints, instantiate_from_config
    from util.fk_sapien import SapienFK
    _robotwin_root = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    sys.path.insert(0, _robotwin_root)
    sys.path.insert(0, os.path.join(_robotwin_root, 'envs', 'robot'))
    _prev_cwd = os.getcwd()
    os.chdir(_robotwin_root)
    from planner import CuroboPlanner
    os.chdir(_prev_cwd)

    evac_cfg = OmegaConf.load(args['evac_config'])
    evac_cfg.model.pretrained_checkpoint = args['evac_ckpt']
    evac_model = instantiate_from_config(evac_cfg.model)
    evac_model = load_checkpoints(evac_model, evac_cfg.model, ignore_mismatched_sizes=False)
    evac_model = evac_model.to(device)
    evac_model.eval()
    for p in evac_model.parameters():
        p.requires_grad = False

    fk = SapienFK(args['urdf_path'])

    root_pose = sapien.Pose([0, -0.65, 0], [0.707, 0, 0, 0.707])
    left_joints = [f'fl_joint{i}' for i in range(1, 7)]
    right_joints = [f'fr_joint{i}' for i in range(1, 7)]
    planner_l = CuroboPlanner(root_pose, left_joints, fk.jnames, yml_path=args['curobo_left_yml'])
    planner_r = CuroboPlanner(root_pose, right_joints, fk.jnames, yml_path=args['curobo_right_yml'])

    return {
        'evac_model': evac_model, 'evac_config': evac_cfg,
        'fk': fk, 'planner_left': planner_l, 'planner_right': planner_r,
    }


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


def resample_trajectory(traj, target_len):
    n = len(traj)
    if n == target_len:
        return traj
    if n == 0:
        return np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    indices = np.linspace(0, n - 1, target_len)
    result = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        result[i] = traj[lo] * (1 - frac) + traj[hi] * frac
    return result

@torch.no_grad()
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
                    norm_stats, modules, cfg, device, debug_dir=None, start_ts=0):
    fk = modules['fk']
    evac_model = modules['evac_model']
    evac_cfg = modules['evac_config']
    planner_l = modules['planner_left']
    planner_r = modules['planner_right']

    start_ts = int(max(0, start_ts))
    left_ep = raw_data['left_endpose'][start_ts:]
    right_ep = raw_data['right_endpose'][start_ts:]
    max_steps = cfg['max_rollout_steps']
    min_dist_fallback_force_correction = bool(cfg['min_dist_fallback_force_correction'])
    min_dist_recover_ratio = float(cfg['min_dist_recover_ratio'])
    debug_recover_eval_rollout = bool(cfg['debug_recover_eval_rollout'])
    target_mode = str(cfg['target_mode']).strip().lower()
    target_lookahead_steps = int(cfg['target_lookahead_steps'])
    correction_interp_nearest_enable = bool(cfg['correction_interp_nearest_enable'])
    correction_interp_prefix_ratio = float(np.clip(cfg['correction_interp_prefix_ratio'], 0.0, 1.0))
    chunk_size = cfg['chunk_size']
    rollout_exec_steps = int(cfg['rollout_exec_steps'])
    max_action_len = cfg['max_action_len']
    orient_weight = float(cfg['orient_weight'])
    gripper_penalty = float(cfg['gripper_penalty'])
    recover_gripper_penalty = float(cfg['recover_gripper_penalty'])

    left_grip_traj = raw_data['left_gripper'][start_ts:]
    right_grip_traj = raw_data['right_gripper'][start_ts:]

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
    min_dist_triggered = False
    min_dist_trigger_step = None
    min_dist_trigger_dist = None
    pending_recover_valid = False
    pending_recover_added = 0.0
    pending_recover_from_step = None
    pending_recover_mode = None

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
                    r['recover_gripper_violation_left'] = gripper_recover.get('violation_left')
                    r['recover_gripper_violation_right'] = gripper_recover.get('violation_right')
                    r['recover_gripper_violation_max'] = gripper_recover.get('violation_max')
                    r['recover_gripper_passed'] = gripper_recover.get('passed')
                if reason is not None:
                    r['reason'] = reason
                return True
        return False

    def _nearest_with_window(lp, lq, rp, rq, expected_idx, gp):
        near_w = int(max(1, cfg.get('nearest_window_radius', 32)))
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
        d_pos_l = float(np.linalg.norm(li[:3] - np.asarray(lp, dtype=np.float32)))
        d_pos_r = float(np.linalg.norm(ri[:3] - np.asarray(rp, dtype=np.float32)))
        d = 0.5 * (d_pos_l + d_pos_r)
        if orient_weight > 0:
            ldot = float(np.clip(np.abs(np.dot(li[3:7], np.asarray(lq, dtype=np.float32))), 0.0, 1.0))
            rdot = float(np.clip(np.abs(np.dot(ri[3:7], np.asarray(rq, dtype=np.float32))), 0.0, 1.0))
            d_ori_l = 2.0 * np.arccos(ldot)
            d_ori_r = 2.0 * np.arccos(rdot)
            d += float(orient_weight) * 0.5 * (d_ori_l + d_ori_r)
        if gp > 0 and left_grip_traj is not None and right_grip_traj is not None:
            curr_l_bin = 0.0 if float(curr_lg) <= 0.5 else 1.0
            curr_r_bin = 0.0 if float(curr_rg) <= 0.5 else 1.0
            anc_l_bin = 0.0 if float(left_grip_traj[idx_ref]) <= 0.5 else 1.0
            anc_r_bin = 0.0 if float(right_grip_traj[idx_ref]) <= 0.5 else 1.0
            d += float(gp) * 0.5 * (abs(anc_l_bin - curr_l_bin) + abs(anc_r_bin - curr_r_bin))
        return float(d)

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
        t_star_start = int(t_star)
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
            q_eval_end = np.asarray(act_pred_raw[-1], dtype=np.float32)
            fk_eval = fk.forward(q_eval_end[0:6], q_eval_end[7:13])
            lp_ev, lq_ev = fk_eval['left']
            rp_ev, rq_ev = fk_eval['right']
            gripper_recover_info = None
            if str(pending_recover_mode).strip().lower() == "gripper_close":
                close_min = float(np.clip(cfg.get("perturb_gripper_close_min", 0.35), 0.0, 1.0))
                gl = float(np.clip(q_eval_end[6], 0.0, 1.0))
                gr = float(np.clip(q_eval_end[13], 0.0, 1.0))
                l_on = bool(active_info.get('left_gripper', True))
                r_on = bool(active_info.get('right_gripper', True))
                # Close perturb should stay closed; reopening implies recovery happened.
                viol_l = max(0.0, gl - close_min) if l_on else 0.0
                viol_r = max(0.0, gr - close_min) if r_on else 0.0
                violation = float(max(viol_l, viol_r))
                min_dist_recover_eval_end = violation
                # Keep existing trigger interface: gain < threshold means "insufficient perturb persistence".
                min_dist_recover_gain = float(-violation)
                min_dist_recover_threshold = 0.0
                gripper_recover_info = {
                    'mode': str(pending_recover_mode).strip().lower(),
                    'close_min': float(close_min),
                    'violation_left': float(viol_l),
                    'violation_right': float(viol_r),
                    'violation_max': float(violation),
                    'passed': bool(violation <= 0.0),
                }
            else:
                min_dist_recover_eval_end = _dist_to_anchor(
                    lp_ev, lq_ev, rp_ev, rq_ev, q_eval_end[6], q_eval_end[13], recover_gripper_penalty, ref_idx=anchor_idx_step
                )
                min_dist_recover_gain = float(anchor_dist_start - float(min_dist_recover_eval_end))
                min_dist_recover_threshold = float(max(0.0, pending_recover_added) * max(0.0, min_dist_recover_ratio))
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
                    infer_kwargs=cfg.get('evac_infer_kwargs'),
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
            if min_dist_recover_gain < float(min_dist_recover_threshold):
                min_dist_triggered = True
                min_dist_trigger_step = (int(recovery_pair_step) if recovery_pair_step is not None else int(step))
                min_dist_trigger_dist = float(min_dist_recover_gain)
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

        active_info = _infer_active_arms_from_gt_window(
            raw_data.get('gt_left_arm'),
            raw_data.get('gt_right_arm'),
            raw_data.get('left_gripper'),
            raw_data.get('right_gripper'),
            t_idx=start_ts + int(t_star),
            window_len=rollout_exec_steps,
            joint_delta_thresh=float(cfg['perturb_active_joint_delta_thresh']),
            gripper_delta_thresh=float(cfg['perturb_active_gripper_delta_thresh']),
        )
        # Side-level activity: arm motion OR gripper motion.
        left_side_active = bool(active_info['left_arm'] or active_info['left_gripper'])
        right_side_active = bool(active_info['right_arm'] or active_info['right_gripper'])
        active_info['left_arm'] = left_side_active
        active_info['right_arm'] = right_side_active
        active_info['left_gripper'] = left_side_active
        active_info['right_gripper'] = right_side_active
        progress = float(t_star) / max(1, len(left_ep) - 1)
        # Choose error phase from the whole GT action window (prefix) instead of a single anchor.
        phase_window_len = int(cfg.get('sample_pregrasp_phase_window_len', rollout_exec_steps))
        phase_key = _infer_phase_key_from_gt_window(
            left_grip_traj=left_grip_traj[int(t_star):],
            right_grip_traj=right_grip_traj[int(t_star):],
            window_len=phase_window_len,
        )
        act_raw, perturbed, pert_mode, dyn_state = perturb_action_chunk_online(
            act_raw, cfg, dyn_state, fk=fk, planner_l=planner_l, planner_r=planner_r,
            phase_key=phase_key, init_left_grip=float(left_grip), init_right_grip=float(right_grip),
            curr_left_q=left_q, curr_right_q=right_q,
            active_left_arm=active_info['left_arm'], active_right_arm=active_info['right_arm'],
            active_left_gripper=active_info['left_gripper'], active_right_gripper=active_info['right_gripper'],
            left_ep=left_ep, right_ep=right_ep,
            left_grip_traj=left_grip_traj, right_grip_traj=right_grip_traj,
            t_star=int(t_star), rollout_exec_steps=int(rollout_exec_steps),
            perturb_anchor_idx=int(anchor_idx_step),
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
            infer_kwargs=cfg.get('evac_infer_kwargs'),
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
        else:
            pending_recover_mode = None

        # collect debug info for this rollout step
        if debug_dir is not None:
            eef_solved_cnt = None
            eef_total_cnt = None
            eef_solved_ratio = None
            sampled_error_mode = None
            if isinstance(dyn_state, dict):
                eef_solved_cnt = dyn_state.get('eef_solved_cnt')
                eef_total_cnt = dyn_state.get('eef_total_cnt')
                eef_solved_ratio = dyn_state.get('eef_solved_ratio')
                sampled_error_mode = dyn_state.get('error_mode')
            _dbg_rollout.append({
                'step': step, 't_star': int(t_star), 'min_dist': float(min_dist),
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
            })

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
        q_eval_end_t = np.asarray(act_raw_t[-1], dtype=np.float32)
        fk_eval_t = fk.forward(q_eval_end_t[0:6], q_eval_end_t[7:13])
        lp_ev_t, lq_ev_t = fk_eval_t['left']
        rp_ev_t, rq_ev_t = fk_eval_t['right']

        gripper_recover_info_t = None
        if str(pending_recover_mode).strip().lower() == "gripper_close":
            active_info_t = _infer_active_arms_from_gt_window(
                raw_data.get('gt_left_arm'),
                raw_data.get('gt_right_arm'),
                raw_data.get('left_gripper'),
                raw_data.get('right_gripper'),
                t_idx=start_ts + int(t_star_t),
                window_len=rollout_exec_steps,
                joint_delta_thresh=float(cfg['perturb_active_joint_delta_thresh']),
                gripper_delta_thresh=float(cfg['perturb_active_gripper_delta_thresh']),
            )
            close_min = float(np.clip(cfg.get("perturb_gripper_close_min", 0.35), 0.0, 1.0))
            gl_t = float(np.clip(q_eval_end_t[6], 0.0, 1.0))
            gr_t = float(np.clip(q_eval_end_t[13], 0.0, 1.0))
            l_on_t = bool(active_info_t.get('left_gripper', True))
            r_on_t = bool(active_info_t.get('right_gripper', True))
            viol_l_t = max(0.0, gl_t - close_min) if l_on_t else 0.0
            viol_r_t = max(0.0, gr_t - close_min) if r_on_t else 0.0
            eval_end_t = float(max(viol_l_t, viol_r_t))
            gain_t = float(-eval_end_t)
            thresh_t = 0.0
            gripper_recover_info_t = {
                'mode': str(pending_recover_mode).strip().lower(),
                'close_min': float(close_min),
                'violation_left': float(viol_l_t),
                'violation_right': float(viol_r_t),
                'violation_max': float(eval_end_t),
                'passed': bool(eval_end_t <= 0.0),
            }
        else:
            eval_end_t = _dist_to_anchor(
                lp_ev_t, lq_ev_t, rp_ev_t, rq_ev_t, q_eval_end_t[6], q_eval_end_t[13],
                recover_gripper_penalty, ref_idx=anchor_idx_t
            )
            gain_t = float(eval_pre_t - float(eval_end_t))
            thresh_t = float(max(0.0, pending_recover_added) * max(0.0, min_dist_recover_ratio))

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
                infer_kwargs=cfg.get('evac_infer_kwargs'),
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
    if min_dist_fallback_force_correction and (not min_dist_triggered):
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
        window_start=max(0, int(rollout_steps_total) - int(max(1, cfg.get('nearest_window_radius', 32)))),
        window_end=min(len(left_ep), int(rollout_steps_total) + int(max(1, cfg.get('nearest_window_radius', 32))) + 1))

    corr_active_info = _infer_active_arms_from_gt_window(
        raw_data.get('gt_left_arm'),
        raw_data.get('gt_right_arm'),
        raw_data.get('left_gripper'),
        raw_data.get('right_gripper'),
        t_idx=start_ts + int(t_star),
        window_len=rollout_exec_steps,
        joint_delta_thresh=float(cfg['perturb_active_joint_delta_thresh']),
        gripper_delta_thresh=float(cfg['perturb_active_gripper_delta_thresh']),
    )
    corr_left_active = bool(corr_active_info['left_arm'] or corr_active_info['left_gripper'])
    corr_right_active = bool(corr_active_info['right_arm'] or corr_active_info['right_gripper'])

    if not force_generate_correction:
        # debug: save rollout info even on skip
        if debug_dir is not None:
            import json
            _dbg_corr = os.path.join(debug_dir, 'correction')
            os.makedirs(_dbg_corr, exist_ok=True)
            with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                _skip_reason = 'min_dist_delta_not_triggered'
                json.dump({'reason': _skip_reason, 'min_dist': float(min_dist),
                           'threshold': None,
                           'min_dist_triggered': bool(min_dist_triggered),
                           'min_dist_trigger_step': (None if min_dist_trigger_step is None else int(min_dist_trigger_step)),
                           'min_dist_trigger_dist': (None if min_dist_trigger_dist is None else float(min_dist_trigger_dist)),
                           'rollout': _dbg_rollout}, _f, indent=2)
        return None

    if correction_interp_nearest_enable:
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
        l_tail = _take_tail(gt_left_arm, tgt_abs + 1, suffix_len, l_tgt_q)
        r_tail = _take_tail(gt_right_arm, tgt_abs + 1, suffix_len, r_tgt_q)
        lt = np.concatenate([l_prefix, l_tail], axis=0).astype(np.float32)
        rt = np.concatenate([r_prefix, r_tail], axis=0).astype(np.float32)

        tl_grip = float(np.clip(left_grip_traj[t_target], 0.0, 1.0))
        tr_grip = float(np.clip(right_grip_traj[t_target], 0.0, 1.0))
        if prefix_len <= 1:
            l_grip_prefix = np.array([tl_grip], dtype=np.float32)
            r_grip_prefix = np.array([tr_grip], dtype=np.float32)
        else:
            l_grip_prefix = np.linspace(float(left_grip), tl_grip, prefix_len, dtype=np.float32)
            r_grip_prefix = np.linspace(float(right_grip), tr_grip, prefix_len, dtype=np.float32)
        l_grip_tail = _take_tail(left_grip_traj, t_target + 1, suffix_len, np.array([tl_grip], dtype=np.float32))[:, 0]
        r_grip_tail = _take_tail(right_grip_traj, t_target + 1, suffix_len, np.array([tr_grip], dtype=np.float32))[:, 0]
        lg = np.concatenate([l_grip_prefix, l_grip_tail], axis=0).astype(np.float32)
        rg = np.concatenate([r_grip_prefix, r_grip_tail], axis=0).astype(np.float32)

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

    def _compose_with_gt_tail(prefix, gt_tail, out_len):
        pfx = np.asarray(prefix, dtype=np.float32)
        # If correction-prefix alone is longer than chunk, compress it so the
        # last step still reaches the correction target (instead of truncating early).
        if pfx.shape[0] >= out_len:
            return resample_trajectory(pfx, out_len).astype(np.float32)
        seq = [pfx]
        if gt_tail is not None and len(gt_tail) > 0:
            seq.append(np.asarray(gt_tail, dtype=np.float32))
        cat = np.concatenate(seq, axis=0)
        if cat.shape[0] >= out_len:
            return cat[:out_len].astype(np.float32)
        pad = np.repeat(cat[-1:].astype(np.float32), out_len - cat.shape[0], axis=0)
        return np.concatenate([cat, pad], axis=0).astype(np.float32)

    if not correction_interp_nearest_enable:
        l_prefix = _fit_prefix(l_future, left_q, prefix_len)
        r_prefix = _fit_prefix(r_future, right_q, prefix_len)

        l_tail = None
        r_tail = None
        if gt_left_arm is not None and corr_left_active:
            l_tail = np.asarray(gt_left_arm[start_ts + t_target + 1:], dtype=np.float32)
        if gt_right_arm is not None and corr_right_active:
            r_tail = np.asarray(gt_right_arm[start_ts + t_target + 1:], dtype=np.float32)

        lt = _compose_with_gt_tail(l_prefix, l_tail, chunk_size)
        rt = _compose_with_gt_tail(r_prefix, r_tail, chunk_size)

        tl_grip = float(np.clip(left_grip_traj[t_target], 0.0, 1.0))
        tr_grip = float(np.clip(right_grip_traj[t_target], 0.0, 1.0))

        # Prefix gripper profile: hold current value first, then change near the end.
        switch_ratio = float(np.clip(cfg.get('correction_gripper_switch_ratio', 0.8), 0.0, 1.0))
        switch_idx = int(np.clip(np.floor(prefix_len * switch_ratio), 0, max(0, prefix_len - 1)))
        l_grip_prefix = np.full((prefix_len,), float(left_grip), dtype=np.float32)
        r_grip_prefix = np.full((prefix_len,), float(right_grip), dtype=np.float32)
        tail_len = prefix_len - switch_idx
        if tail_len > 1:
            l_grip_prefix[switch_idx:] = np.linspace(float(left_grip), tl_grip, tail_len, dtype=np.float32)
            r_grip_prefix[switch_idx:] = np.linspace(float(right_grip), tr_grip, tail_len, dtype=np.float32)
        elif tail_len == 1:
            l_grip_prefix[-1] = tl_grip
            r_grip_prefix[-1] = tr_grip
        l_grip_tail = np.asarray(left_grip_traj[t_target + 1:], dtype=np.float32) if corr_left_active else np.asarray([], dtype=np.float32)
        r_grip_tail = np.asarray(right_grip_traj[t_target + 1:], dtype=np.float32) if corr_right_active else np.asarray([], dtype=np.float32)
        lg = _compose_with_gt_tail(l_grip_prefix[:, None], l_grip_tail[:, None], chunk_size)[:, 0]
        rg = _compose_with_gt_tail(r_grip_prefix[:, None], r_grip_tail[:, None], chunk_size)[:, 0]
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
        _dbg_corr = os.path.join(debug_dir, 'correction')
        os.makedirs(_dbg_corr, exist_ok=True)
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

            for _img in (_overlay_o, _overlay_c):
                _img[mask] = (0.6 * _img[mask] + 0.4 * traj_u8[mask]).astype(np.uint8)
                _draw_polyline(_img, luv, (0, 255, 0))
                _draw_polyline(_img, ruv, (0, 0, 255))

            cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_original.png'), _overlay_o)
            cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_corrected.png'), _overlay_c)
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
                'recover_gripper_violation_left': (None if _r.get('recover_gripper_violation_left') is None else float(_r.get('recover_gripper_violation_left'))),
                'recover_gripper_violation_right': (None if _r.get('recover_gripper_violation_right') is None else float(_r.get('recover_gripper_violation_right'))),
                'recover_gripper_violation_max': (None if _r.get('recover_gripper_violation_max') is None else float(_r.get('recover_gripper_violation_max'))),
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
            _trigger_source = 'min_dist'
            _trigger_step = (None if min_dist_trigger_step is None else int(min_dist_trigger_step))
            _trigger_value = (None if min_dist_trigger_dist is None else float(min_dist_trigger_dist))
        elif bool(force_generate_correction):
            _trigger_source = 'fallback'

        _plan_info = {
            'version': 2,
            'trigger_mode': 'min_dist_recovery',
            'trigger_threshold': {
                'min_dist': None,
                'min_dist_recover_ratio': float(min_dist_recover_ratio),
                'min_dist_recover_use_added': True,
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
                'correction_prefix_len': int(prefix_len),
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
                'recover_gripper_violation_left': _r.get('recover_gripper_violation_left'),
                'recover_gripper_violation_right': _r.get('recover_gripper_violation_right'),
                'recover_gripper_violation_max': _r.get('recover_gripper_violation_max'),
                'recover_gripper_passed': _r.get('recover_gripper_passed'),
            }
            _closed_loop_rollouts.append(_drop_none(_rec))

        _closed_loop_info = {
            'version': 1,
            'start_ts': int(start_ts),
            'rollout_exec_steps': int(rollout_exec_steps),
            'final_match': {
                't_star': _plan_info['final_match'].get('t_star'),
                'anchor_idx': _plan_info['final_match'].get('anchor_idx'),
            },
            'rollouts': _closed_loop_rollouts,
        }
        with open(os.path.join(_dbg_corr, 'closed_loop_info.json'), 'w') as _f:
            json.dump(_closed_loop_info, _f, indent=2)

    corr_meta = {
        "closed_loop_fallback_used": bool(force_generate_correction and (not min_dist_triggered)),
    }

    return (
        curr_image.to(device),
        torch.from_numpy(qn.astype(np.float32)).to(device),
        torch.from_numpy(padded).float().to(device),
        torch.from_numpy(is_pad).bool().to(device),
        corr_meta,
    )

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data[0], data[1], data[2], data[3]
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, config):
    _r = int(os.environ.get("RANK", -1))
    _lr = int(os.environ.get("LOCAL_RANK", -1))
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    enable_wm = bool(config["enable_wm_correction"])
    if enable_wm:
        sp_reg_enable = bool(config["sp_reg_enable"])
        sp_reg_lambda = float(config["sp_reg_lambda"])
        lr_sched_enable = bool(config["lr_sched_enable"])
        lr_warmup_steps = int(max(0, config["lr_warmup_steps"]))
        lr_min_ratio = float(np.clip(config["lr_min_ratio"], 0.0, 1.0))
    else:
        sp_reg_enable = False
        sp_reg_lambda = 0.0
        lr_sched_enable = False
        lr_warmup_steps = 0
        lr_min_ratio = 0.1

    # Accelerate: prepare model, optimizer, dataloader
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Set seed after Accelerator init so each process gets proper seed offset
    set_seed(seed)

    print(f"[train_bc][rank={_r} local_rank={_lr}] before make_policy")
    _t_make_policy = time.time()
    policy = make_policy(policy_class, policy_config)
    print(f"[train_bc][rank={_r} local_rank={_lr}] after make_policy | elapsed={time.time() - _t_make_policy:.2f}s")

    if config.get('act_init_ckpt'):
        print(f"[train_bc][rank={_r} local_rank={_lr}] before load_act_init_ckpt")
        _t_load_act = time.time()
        ckpt = torch.load(config['act_init_ckpt'], map_location='cpu')
        policy.load_state_dict(ckpt)
        print(f"[train_bc][rank={_r} local_rank={_lr}] after load_act_init_ckpt | elapsed={time.time() - _t_load_act:.2f}s")
        print(f"Loaded ACT init weights from {config['act_init_ckpt']}")

    print(f"[train_bc][rank={_r} local_rank={_lr}] before make_optimizer")
    _t_make_opt = time.time()
    optimizer = make_optimizer(policy_class, policy)
    print(f"[train_bc][rank={_r} local_rank={_lr}] after make_optimizer | elapsed={time.time() - _t_make_opt:.2f}s")
    print(f"[train_bc][rank={_r} local_rank={_lr}] before accelerator.prepare")
    _t_prepare = time.time()
    policy, optimizer, train_dataloader = accelerator.prepare(
        policy, optimizer, train_dataloader
    )
    print(f"[train_bc][rank={_r} local_rank={_lr}] after accelerator.prepare | elapsed={time.time() - _t_prepare:.2f}s")

    lr_scheduler = None
    if lr_sched_enable:
        total_steps = int(max(1, num_epochs * max(1, len(train_dataloader))))
        warmup_steps = int(min(lr_warmup_steps, max(0, total_steps - 1)))

        def _lr_lambda(step):
            s = int(max(0, step))
            if warmup_steps > 0 and s < warmup_steps:
                return float(s + 1) / float(max(1, warmup_steps))
            if total_steps <= warmup_steps + 1:
                return 1.0
            progress = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
            progress = float(np.clip(progress, 0.0, 1.0))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return float(lr_min_ratio + (1.0 - lr_min_ratio) * cosine)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        if accelerator.is_main_process:
            print(f"[train_bc] LR scheduler enabled | total_steps={total_steps} warmup_steps={warmup_steps} min_ratio={lr_min_ratio}")

    sp_ref_params = None
    if sp_reg_enable and sp_reg_lambda > 0.0:
        sp_ref_params = {
            n: p.detach().clone()
            for n, p in policy.named_parameters()
            if p.requires_grad
        }
        if accelerator.is_main_process:
            print(f"[train_bc] L2-SP enabled | lambda={sp_reg_lambda} | n_params={len(sp_ref_params)}")

    correction_modules = None
    debug_wm = config.get('debug_wm_correction', False)
    debug_wm_dir = os.path.join(ckpt_dir, 'debug_wm') if debug_wm else None
    if debug_wm_dir and accelerator.is_main_process:
        os.makedirs(debug_wm_dir, exist_ok=True)
    # correction statistics tracker
    _corr_stats = {'n_triggered': 0, 'n_success': 0, 'n_skipped': 0,
                   'n_plan_fail': 0, 'n_error': 0, 'dists': [], 'steps_used': [],
                   'n_fallback': 0}
    if enable_wm:
        print(f"[train_bc][rank={_r} local_rank={_lr}] before init_correction")
        _t_init_corr = time.time()
        correction_modules = init_correction(config['wm_args'], accelerator.device)
        print(f"[train_bc][rank={_r} local_rank={_lr}] after init_correction | elapsed={time.time() - _t_init_corr:.2f}s")
        norm_stats = config['norm_stats']
        correction_cfg = config['correction_cfg']
        raw_data_dir = config['raw_data_dir']
        export_corr = bool(correction_cfg.get('export_correction_dataset', False))
        export_corr_dir = str(correction_cfg.get('export_correction_dir', '')).strip()
        if export_corr and export_corr_dir == '':
            export_corr_dir = os.path.join(ckpt_dir, 'correction_dataset')
        export_next_id = 0
        if export_corr and accelerator.is_main_process:
            os.makedirs(export_corr_dir, exist_ok=True)
            export_next_id = _init_export_episode_id(export_corr_dir)
            print(f"[train_bc] correction export enabled: {export_corr_dir}, next_episode_id={export_next_id}")

    train_history = []

    # TensorBoard (only on main process)
    writer = None
    if accelerator.is_main_process:
        tb_log_dir = os.path.join(ckpt_dir, 'tb_logs')
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f'TensorBoard log dir: {tb_log_dir}')
    global_step = 0

    for epoch in tqdm(range(num_epochs), disable=not accelerator.is_main_process):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}")

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            apply_wm = enable_wm and correction_modules is not None
            if not apply_wm:
                forward_dict = forward_pass(data, policy)
                loss = forward_dict["loss"]
            else:
                try:
                    bs = data[0].shape[0]
                    corr_images, corr_qpos, corr_actions, corr_pads = [], [], [], []
                    corr_mask = []
                    # per-step debug dir (only on main process, only every 100 steps)
                    _step_dbg = None
                    if debug_wm_dir and accelerator.is_main_process:
                        _step_dbg = os.path.join(debug_wm_dir, f'step_{global_step:06d}')
                        os.makedirs(_step_dbg, exist_ok=True)
                    for bi in range(bs):
                        ep_id = data[4][bi].item()
                        raw = load_raw_data(raw_data_dir, ep_id)
                        _bi_dbg = os.path.join(_step_dbg, f'bi{bi}') if _step_dbg else None
                        if _bi_dbg:
                            os.makedirs(_bi_dbg, exist_ok=True)
                        corr = correction_step(
                            accelerator.unwrap_model(policy),
                            data[0][bi], data[1][bi], raw,
                            norm_stats, correction_modules, correction_cfg,
                            accelerator.device, debug_dir=_bi_dbg,
                            start_ts=data[5][bi].item())
                        _corr_stats['n_triggered'] += 1
                        if corr is not None:
                            if isinstance(corr, (tuple, list)) and len(corr) >= 5:
                                ci, cq, ca, cp, cmeta = corr
                            else:
                                ci, cq, ca, cp = corr
                                cmeta = {}
                            corr_images.append(ci)
                            corr_qpos.append(cq)
                            corr_actions.append(ca)
                            corr_pads.append(cp)
                            corr_mask.append(1.0)
                            _corr_stats['n_success'] += 1
                            if isinstance(cmeta, dict):
                                if bool(cmeta.get('closed_loop_fallback_used', False)):
                                    _corr_stats['n_fallback'] += 1
                            if enable_wm and export_corr and accelerator.is_main_process:
                                _export_correction_sample_as_episode(
                                    export_corr_dir,
                                    export_next_id,
                                    config['camera_names'],
                                    ci,
                                    cq,
                                    ca,
                                    cp,
                                    norm_stats,
                                )
                                export_next_id += 1
                        else:
                            _corr_stats['n_skipped'] += 1
                            # dummy data to keep batch size consistent across ranks
                            corr_images.append(data[0][bi].to(accelerator.device))
                            corr_qpos.append(data[1][bi].to(accelerator.device))
                            corr_actions.append(data[2][bi].to(accelerator.device))
                            corr_pads.append(data[3][bi].to(accelerator.device))
                            corr_mask.append(0.0)

                    # Build a fixed-size mixed batch: [base bs] + [correction bs].
                    ci_b = torch.stack(corr_images, dim=0)
                    cq_b = torch.stack(corr_qpos, dim=0)
                    ca_b = torch.stack(corr_actions, dim=0)
                    cp_b = torch.stack(corr_pads, dim=0)

                    base_images = data[0].to(accelerator.device)
                    base_qpos = data[1].to(accelerator.device)
                    base_actions = data[2].to(accelerator.device)
                    base_pads = data[3].to(accelerator.device)

                    mixed_images = torch.cat([base_images, ci_b], dim=0)
                    mixed_qpos = torch.cat([base_qpos, cq_b], dim=0)
                    mixed_actions = torch.cat([base_actions, ca_b], dim=0)
                    mixed_pads = torch.cat([base_pads, cp_b], dim=0)

                    mixed_dict = policy(
                        mixed_qpos,
                        mixed_images,
                        mixed_actions,
                        mixed_pads,
                        return_per_sample=True,
                    )

                    per_sample_loss = mixed_dict['loss_per_sample']
                    mask_t = torch.tensor(corr_mask, device=accelerator.device, dtype=per_sample_loss.dtype)
                    n_corr = int(mask_t.sum().item())

                    base_weight = torch.ones(bs, device=accelerator.device, dtype=per_sample_loss.dtype)
                    corr_weight = correction_cfg['correction_weight'] * mask_t
                    sample_weight = torch.cat([base_weight, corr_weight], dim=0)
                    loss = (per_sample_loss * sample_weight).sum() / (bs + n_corr)
                    forward_dict = {"loss": loss}

                    if accelerator.is_main_process and writer is not None:
                        if n_corr > 0:
                            corr_loss = (per_sample_loss[bs:] * mask_t).sum() / mask_t.sum()
                        else:
                            corr_loss = torch.tensor(0.0, device=accelerator.device, dtype=per_sample_loss.dtype)
                        writer.add_scalar('train/correction_loss', corr_loss.item(), global_step)
                        writer.add_scalar('train/n_correction_samples', n_corr, global_step)
                except Exception as exc:
                    _corr_stats['n_error'] += 1
                    if accelerator.is_main_process:
                        import traceback
                        print(f'[WM correction] step {global_step} error: {exc}')
                        traceback.print_exc()
                    forward_dict = forward_pass(data, policy)
                    loss = forward_dict["loss"]

            # Optional L2-SP regularization to reduce catastrophic forgetting.
            if sp_ref_params is not None:
                sp_loss = None
                for n, p in policy.named_parameters():
                    if (not p.requires_grad) or (n not in sp_ref_params):
                        continue
                    d = p - sp_ref_params[n]
                    term = torch.sum(d * d)
                    sp_loss = term if sp_loss is None else (sp_loss + term)
                if sp_loss is None:
                    sp_loss = torch.tensor(0.0, device=accelerator.device)
                base_loss = loss
                loss = base_loss + sp_reg_lambda * sp_loss
                forward_dict = dict(forward_dict)
                forward_dict["loss_base"] = base_loss
                forward_dict["sp_loss"] = sp_loss
                forward_dict["loss"] = loss

            accelerator.backward(loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            if accelerator.is_main_process and writer is not None:
                writer.add_scalar('train/loss_step', forward_dict['loss'].item(), global_step)
                if sp_ref_params is not None:
                    writer.add_scalar('train/loss_base_step', forward_dict['loss_base'].item(), global_step)
                    writer.add_scalar('train/sp_loss_step', forward_dict['sp_loss'].item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        # TensorBoard: log epoch-level metrics
        if accelerator.is_main_process and writer is not None:
            for k, v in epoch_summary.items():
                writer.add_scalar(f'train/{k}_epoch', v.item(), epoch)
            # log correction statistics
            if enable_wm:
                writer.add_scalar('correction/n_triggered', _corr_stats['n_triggered'], epoch)
                writer.add_scalar('correction/n_success', _corr_stats['n_success'], epoch)
                writer.add_scalar('correction/n_skipped', _corr_stats['n_skipped'], epoch)
                writer.add_scalar('correction/n_error', _corr_stats['n_error'], epoch)
                writer.add_scalar('correction/fallback_count', _corr_stats['n_fallback'], epoch)
                _total = _corr_stats['n_triggered'] or 1
                writer.add_scalar('correction/success_rate', _corr_stats['n_success'] / _total, epoch)
                _succ = _corr_stats['n_success'] or 1
                writer.add_scalar('correction/fallback_rate', _corr_stats['n_fallback'] / _succ, epoch)
                print(f'  [Correction] triggered={_corr_stats["n_triggered"]} '
                      f'success={_corr_stats["n_success"]} '
                      f'skipped={_corr_stats["n_skipped"]} '
                      f'error={_corr_stats["n_error"]} '
                      f'fallback={_corr_stats["n_fallback"]} '
                      f'rate={_corr_stats["n_success"]/_total:.2%}')
                # reset per-epoch
                _corr_stats = {'n_triggered': 0, 'n_success': 0, 'n_skipped': 0,
                               'n_plan_fail': 0, 'n_error': 0, 'dists': [], 'steps_used': [],
                               'n_fallback': 0}

        if (epoch + 1) % config['save_freq'] == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            unwrapped_policy = accelerator.unwrap_model(policy)
            torch.save(unwrapped_policy.state_dict(), ckpt_path)

    if accelerator.is_main_process:
        if writer is not None:
            writer.close()

        ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
        unwrapped_policy = accelerator.unwrap_model(policy)
        torch.save(unwrapped_policy.state_dict(), ckpt_path)

    print(f"Training finished: Seed {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Core training (required)
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # ACT model/training
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--state_dim", action="store", type=int, help="state dim", required=True)
    parser.add_argument("--save_freq", action="store", type=int, help="save ckpt frequency", required=False, default=6000)
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")

    # Dataloader sampling bias
    parser.add_argument("--sample_skip_head", action="store", type=int, default=0,
                        help="Skip first N timesteps when sampling start_ts in dataloader")
    parser.add_argument("--sample_pregrasp_bias_enable", type=str2bool, default=False,
                        help="Bias dataloader start_ts sampling towards pregrasp candidates")
    parser.add_argument("--sample_pregrasp_prob", type=float, default=0.0,
                        help="When pregrasp bias is enabled, probability of sampling from pregrasp candidates")
    parser.add_argument("--sample_pregrasp_phase_window_len", type=int, default=16,
                        help="Window length used by shared phase inference when mining pregrasp start_ts")
    parser.add_argument("--sample_pregrasp_avoid_switch_tail", type=int, default=0,
                        help="Exclude last N steps before first pregrasp switch from pregrasp-biased sampling")

    # Online perturbation
    parser.add_argument("--enable_perturb", type=str2bool, default=False,
                        help="Enable online rollout action perturbation in WM correction")
    parser.add_argument("--perturb_prob", type=float, default=1.0,
                        help="Probability to apply online perturbation per rollout chunk")
    parser.add_argument("--perturb_error_mode", type=str, default="legacy",
                        help="Error mode: legacy|auto|open_laptop_pregrasp|translation|rotation|gripper_close")
    parser.add_argument("--perturb_open_laptop_pregrasp_close_prob", type=float, default=0.8,
                        help="When perturb_error_mode=open_laptop_pregrasp, probability of gripper_close in pregrasp")
    parser.add_argument("--perturb_open_laptop_pregrasp_translation_prob", type=float, default=0.1,
                        help="When perturb_error_mode=open_laptop_pregrasp, probability of translation in pregrasp")
    parser.add_argument("--perturb_open_laptop_pregrasp_rotation_prob", type=float, default=0.1,
                        help="When perturb_error_mode=open_laptop_pregrasp, probability of rotation in pregrasp")
    parser.add_argument("--perturb_eef_fail_gain", type=float, default=0.03,
                        help="Directional EEF perturbation magnitude in meters")
    parser.add_argument("--perturb_rot_max_deg", type=float, default=15.0,
                        help="Max EEF rotation perturbation angle in degrees")
    parser.add_argument("--perturb_mag_random", type=str2bool, default=False,
                        help="Randomize perturbation magnitude per rollout chunk")
    parser.add_argument("--perturb_mag_rand_min", type=float, default=0.8,
                        help="Minimum random magnitude scale")
    parser.add_argument("--perturb_mag_rand_max", type=float, default=1.2,
                        help="Maximum random magnitude scale")
    parser.add_argument("--perturb_reject_sampling_enable", type=str2bool, default=True,
                        help="Enable rejection sampling for translation/rotation perturbation")
    parser.add_argument("--perturb_reject_max_trials", type=int, default=4,
                        help="Maximum IK trials per rollout chunk")
    parser.add_argument("--perturb_reject_dir_jitter_eps", type=float, default=0.2,
                        help="Direction jitter around main away-from-anchor direction for trial sampling")
    parser.add_argument("--nearest_window_radius", type=int, default=32,
                        help="Nearest-point matching window radius around expected progress index")
    parser.add_argument("--perturb_gripper_close_min", type=float, default=0.35,
                        help="Minimum gripper value in close-failure mode (cannot fully close)")
    parser.add_argument("--perturb_gripper_fast_ratio", type=float, default=0.2,
                        help="Fraction of chunk used to complete gripper transition in target-pose perturbation")
    parser.add_argument("--perturb_active_joint_delta_thresh", type=float, default=0.02,
                        help="GT joint delta threshold (rad) to mark an arm as active")
    parser.add_argument("--perturb_active_gripper_delta_thresh", type=float, default=0.05,
                        help="GT gripper delta threshold to mark gripper side active")

    # WM correction (closed-loop)
    parser.add_argument("--enable_wm_correction", action="store_true",
                        help="Enable world-model based correction training")
    parser.add_argument("--evac_ckpt", type=str,
                        help="Path to EVAC model checkpoint")
    parser.add_argument("--evac_config", type=str,
                        help="Path to EVAC train_config.yaml")
    parser.add_argument("--urdf_path", type=str,
                        help="Path to robot URDF file")
    parser.add_argument("--curobo_left_yml", type=str,
                        help="Path to curobo left arm config yml")
    parser.add_argument("--curobo_right_yml", type=str,
                        help="Path to curobo right arm config yml")
    parser.add_argument("--raw_data_dir", type=str,
                        help="Path to raw episode data directory")
    parser.add_argument("--act_init_ckpt", type=str,
                        help="Path to ACT initial checkpoint")
    parser.add_argument("--max_rollout_steps", type=int,
                        help="Max rollout steps for correction")
    parser.add_argument("--target_mode", type=str, default="forward",
                        help="Correction target mode: forward or backward")
    parser.add_argument("--target_lookahead_steps", type=int, default=0,
                        help="Correction target lookahead steps from nearest t_star on expert trajectory")
    parser.add_argument("--min_dist_fallback_force_correction", type=str2bool, default=False,
                        help="If true, force correction generation when min_dist mode does not reach threshold")
    parser.add_argument("--min_dist_recover_ratio", type=float, default=0.5,
                        help="Dynamic recovery threshold ratio: ratio * max(0, perturb_added_dist)")
    parser.add_argument("--debug_recover_eval_rollout", type=str2bool, default=False,
                        help="If true, also run EVAC rollout video for unperturbed recovery-eval action in each paired step")
    parser.add_argument("--correction_interp_nearest_enable", type=str2bool, default=False,
                        help="If true, set correction target to nearest point t_star and bypass planner with interpolation+GT tail")
    parser.add_argument("--correction_interp_prefix_ratio", type=float, default=0.4,
                        help="Prefix ratio for interpolation when correction_interp_nearest_enable=true (e.g., 0.4 means 2/5 chunk)")
    parser.add_argument("--rollout_exec_steps", type=int, default=None,
                        help="Number of actions executed per rollout step (prefix of chunk)")
    parser.add_argument("--correction_weight", type=float,
                        help="Weight for correction loss (auto-divided by batch_size)")
    parser.add_argument("--orient_weight", type=float,
                        help="Weight for orientation distance in nearest-point matching (0=position only)")
    parser.add_argument("--gripper_penalty", type=float,
                        help="Penalty for gripper state mismatch in nearest-point matching (0=ignore gripper)")
    parser.add_argument("--recover_gripper_penalty", type=float, default=None,
                        help="Penalty for gripper mismatch used only in closed-loop anchor-distance/recovery metrics "
                             "(default: follow gripper_penalty)")
    parser.add_argument("--debug_wm_correction", action="store_true",
                        help="Enable debug visualization for WM correction (saves images/stats to ckpt_dir/debug_wm)")
    parser.add_argument("--export_correction_dataset", type=str2bool, default=False,
                        help="Export successful correction samples as ACT-compatible hdf5 episodes")
    parser.add_argument("--export_correction_dir", type=str, default="",
                        help="Output directory for exported correction episodes (default: ckpt_dir/correction_dataset)")

    # Optimizer/regularization
    parser.add_argument("--lr_sched_enable", type=str2bool, default=False,
                        help="Enable per-step warmup+cosine LR scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of warmup steps for LR scheduler")
    parser.add_argument("--lr_min_ratio", type=float, default=0.1,
                        help="Minimum LR ratio (min_lr = lr * ratio) for cosine decay")
    parser.add_argument("--sp_reg_enable", type=str2bool, default=False,
                        help="Enable L2-SP regularization to keep parameters close to initialization")
    parser.add_argument("--sp_reg_lambda", type=float, default=0.0,
                        help="Weight of L2-SP regularization term")

    main(vars(parser.parse_args()))
