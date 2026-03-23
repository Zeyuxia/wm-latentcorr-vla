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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
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


def parse_optional_float(v):
    if v is None:
        return None
    if isinstance(v, float):
        return v
    if isinstance(v, str) and v.lower() == "none":
        return None
    try:
        return float(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float value: {v}") from exc


def build_evac_infer_kwargs(cfg):
    use_budget_accel = cfg.get('evac_budget_accel', False)
    use_rank_transfer = cfg.get('evac_rank_transfer', False)
    if use_budget_accel and use_rank_transfer:
        raise ValueError('evac_budget_accel and evac_rank_transfer cannot be enabled together')

    ddim_eta = cfg.get('evac_ddim_eta', None)
    if ddim_eta is None:
        ddim_eta = 0.0 if use_rank_transfer else 1.0

    if use_budget_accel:
        return {
            'use_dual_cache': True,
            'dc_budget': cfg.get('evac_dc_budget', 0.5),
            'ddim_eta': ddim_eta,
        }
    if use_rank_transfer:
        return {
            'dc_rank_transfer': True,
            'rt_full_chunks': cfg.get('evac_rt_full_chunks', 3),
            'rt_per_channel': cfg.get('evac_rt_per_channel', True),
            'ddim_eta': ddim_eta,
        }
    return {'ddim_eta': ddim_eta}


def _first_threshold_cross(arr, threshold=0.5, from_high_to_low=True):
    if arr is None or len(arr) < 2:
        return None
    arr = np.asarray(arr).reshape(-1)
    prev = arr[:-1]
    curr = arr[1:]
    if from_high_to_low:
        idx = np.where((prev > threshold) & (curr <= threshold))[0]
    else:
        idx = np.where((prev <= threshold) & (curr > threshold))[0]
    if idx.size == 0:
        return None
    return int(idx[0] + 1)


def _infer_phase_key_event_driven(t_star, left_grip_traj, right_grip_traj, traj_len):
    """
    Local gripper-state-driven phase split around current t_star.
    Robust for short rollouts and episodes that start already grasping:
    stable_open -> approach
    closing/transition -> pregrasp
    stable_close -> transport
    opening -> place
    """
    t_star = int(max(0, t_star))
    traj_len = int(max(1, traj_len))
    left = np.asarray(left_grip_traj).reshape(-1)
    right = np.asarray(right_grip_traj).reshape(-1)
    if left.size == 0 or right.size == 0:
        return "transport"

    idx = int(np.clip(t_star, 0, min(left.size, right.size) - 1))
    win = max(2, traj_len // 60)
    lo = max(0, idx - win)
    hi = min(min(left.size, right.size), idx + win + 1)

    l_seg = left[lo:hi]
    r_seg = right[lo:hi]
    g_seg = 0.5 * (l_seg + r_seg)
    g_now = float(g_seg[idx - lo])

    # Hysteresis thresholds for robust stable-state classification.
    # Raise open threshold so "approach" is stricter; ambiguous area leans to pregrasp.
    th_open = 0.7
    th_close = 0.4

    if g_seg.size >= 2:
        slope = float(np.mean(np.diff(g_seg)))
        opening_cross = bool(np.any((g_seg[:-1] <= 0.5) & (g_seg[1:] > 0.5)))
        closing_cross = bool(np.any((g_seg[:-1] > 0.5) & (g_seg[1:] <= 0.5)))
    else:
        slope = 0.0
        opening_cross = False
        closing_cross = False

    eps_open_slope = 0.01
    eps_close_slope = 0.004
    if opening_cross or slope > eps_open_slope:
        return "place"
    if closing_cross or slope < -eps_close_slope:
        return "pregrasp"
    # Even at high opening, a slight closing trend should prefer pregrasp.
    if g_now >= th_open and slope >= -eps_close_slope:
        return "approach"
    if g_now <= th_close:
        return "transport"
    return "pregrasp"


def _infer_phase_key_from_gt_window(left_grip_traj, right_grip_traj, window_len):
    """
    Infer phase from the whole GT action window (prefix), not a single point.
    """
    left = np.asarray(left_grip_traj).reshape(-1)
    right = np.asarray(right_grip_traj).reshape(-1)
    n = int(max(1, min(len(left), len(right), int(window_len))))
    l = left[:n]
    r = right[:n]

    def _smooth3(arr):
        if arr.size < 3:
            return arr
        k = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
        return np.convolve(arr, k, mode="same")

    l_s = _smooth3(l)
    r_s = _smooth3(r)

    def _first_cross(arr, open_event=True):
        if arr.size < 2:
            return None
        if open_event:
            idx = np.where((arr[:-1] <= 0.5) & (arr[1:] > 0.5))[0]
        else:
            idx = np.where((arr[:-1] > 0.5) & (arr[1:] <= 0.5))[0]
        return int(idx[0]) if idx.size > 0 else None

    # OR-trigger across arms: any arm with salient event drives phase.
    open_l, open_r = _first_cross(l_s, True), _first_cross(r_s, True)
    close_l, close_r = _first_cross(l_s, False), _first_cross(r_s, False)
    open_events = [x for x in (open_l, open_r) if x is not None]
    close_events = [x for x in (close_l, close_r) if x is not None]
    first_open = min(open_events) if open_events else None
    first_close = min(close_events) if close_events else None
    if first_open is not None and first_close is not None:
        return "pregrasp" if first_close < first_open else "place"
    if first_close is not None:
        return "pregrasp"
    if first_open is not None:
        return "place"

    eps = 0.02
    amp_min = 0.06
    slope_l = float(np.mean(np.diff(l_s))) if l_s.size >= 2 else 0.0
    slope_r = float(np.mean(np.diff(r_s))) if r_s.size >= 2 else 0.0
    amp_l = float(np.max(l_s) - np.min(l_s)) if l_s.size > 0 else 0.0
    amp_r = float(np.max(r_s) - np.min(r_s)) if r_s.size > 0 else 0.0
    if ((slope_l > eps) and (amp_l >= amp_min)) or ((slope_r > eps) and (amp_r >= amp_min)):
        return "place"
    if ((slope_l < -eps) and (amp_l >= amp_min)) or ((slope_r < -eps) and (amp_r >= amp_min)):
        return "pregrasp"

    # No salient behavior: infer from stable states.
    # Make approach stricter so boundary region falls into pregrasp.
    mean_l = float(np.mean(l_s))
    mean_r = float(np.mean(r_s))
    if mean_l >= 0.7 and mean_r >= 0.7:
        return "approach"
    if mean_l <= 0.4 and mean_r <= 0.4:
        return "transport"
    # Arms disagree (one open, one close) -> likely in manipulation/transport regime.
    if (mean_l <= 0.4) or (mean_r <= 0.4):
        return "transport"
    return "pregrasp"


def _extract_arm(a):
    return a[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]


def _write_arm(a, arm):
    a[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] = arm
    return a


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


def _parse_vec12(s):
    if s is None:
        return None
    txt = str(s).strip()
    if txt == "":
        return None
    vals = [v.strip() for v in txt.split(",") if v.strip() != ""]
    if len(vals) != 12:
        raise ValueError("perturb_fail_direction must contain 12 comma-separated values")
    arr = np.array([float(v) for v in vals], dtype=np.float32)
    n = np.linalg.norm(arr)
    if n < 1e-8:
        raise ValueError("perturb_fail_direction norm is zero")
    return arr / n


def _parse_vec3(s):
    if s is None:
        return None
    txt = str(s).strip()
    if txt == "":
        return None
    vals = [v.strip() for v in txt.split(",") if v.strip() != ""]
    if len(vals) != 3:
        raise ValueError("3D direction must contain 3 comma-separated values")
    arr = np.array([float(v) for v in vals], dtype=np.float32)
    n = np.linalg.norm(arr)
    if n < 1e-8:
        raise ValueError("3D direction norm is zero")
    return arr / n


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


def _anti_align_with_gt(v_noise, v_gt, cos_thresh):
    """
    If sampled perturb direction is strongly aligned with GT trend, flip it.
    cos_thresh in [0,1], larger => looser (flip less often).
    """
    u_n = _unit_vec3(v_noise)
    u_g = _unit_vec3(v_gt)
    if u_n is None or u_g is None:
        return np.asarray(v_noise, dtype=np.float32), None, False
    cos = float(np.dot(u_n, u_g))
    if cos > float(cos_thresh):
        return (-u_n).astype(np.float32), cos, True
    return u_n.astype(np.float32), cos, False


def _link6_to_tcp_pose(link_pos_w, link_quat_wxyz, offset_x=0.085):
    # Approximate link6 -> gripper/TCP conversion for ALOHA-Agilex.
    from scipy.spatial.transform import Rotation as R
    r = R.from_quat([link_quat_wxyz[1], link_quat_wxyz[2], link_quat_wxyz[3], link_quat_wxyz[0]])
    tcp_pos = np.asarray(link_pos_w, dtype=np.float32) + r.as_matrix() @ np.array([offset_x, 0.0, 0.0], dtype=np.float32)
    return tcp_pos.astype(np.float32), np.asarray(link_quat_wxyz, dtype=np.float32)


def _perturb_action_chunk_eef(act_raw, cfg, fk, planner_l, planner_r, active_left_arm=True, active_right_arm=True):
    # EEF-space perturbation: FK joint action -> perturb EEF pose -> planner/IK back to joint action.
    if fk is None or planner_l is None or planner_r is None:
        return act_raw, False, "eef_missing_modules", None

    import sapien
    out = act_raw.copy()
    active_left_arm = bool(active_left_arm)
    active_right_arm = bool(active_right_arm)
    if (not active_left_arm) and (not active_right_arm):
        return out, False, "eef_inactive_arms", {"active_left_arm": False, "active_right_arm": False}
    eef_mode = str(cfg.get("perturb_eef_mode", "gaussian")).strip().lower()
    pos_std = float(cfg.get("perturb_eef_pos_std", 0.01))
    fail_gain = float(cfg.get("perturb_eef_fail_gain", 0.03))
    mag_rand = bool(cfg.get("perturb_mag_random", False))
    mag_min = float(cfg.get("perturb_mag_rand_min", 0.8))
    mag_max = float(cfg.get("perturb_mag_rand_max", 1.2))
    if mag_max < mag_min:
        mag_min, mag_max = mag_max, mag_min
    mag_scale = float(np.random.uniform(mag_min, mag_max)) if mag_rand else 1.0
    tcp_offset_x = float(cfg.get("perturb_eef_tcp_offset_x", 0.085))
    ramp_enable = bool(cfg.get("perturb_eef_ramp", True))
    ramp_min = float(cfg.get("perturb_eef_ramp_min", 0.0))
    ramp_power = float(cfg.get("perturb_eef_ramp_power", 1.0))
    ramp_apply_eps = float(cfg.get("perturb_eef_ramp_apply_eps", 1e-4))
    joint_delta_cap = float(cfg.get("perturb_eef_joint_delta_cap", 0.12))
    rollout_exec_steps = int(cfg.get("rollout_exec_steps", 0))
    dir_l = cfg.get("perturb_eef_fail_dir_left_vec", None)
    dir_r = cfg.get("perturb_eef_fail_dir_right_vec", None)
    rand_dir = bool(cfg.get("perturb_translation_random_dir", True))

    if eef_mode == "directional_fail":
        if rand_dir:
            dir_l = _sample_unit_vec3()
            dir_r = _sample_unit_vec3()
        else:
            if dir_l is None:
                dir_l = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if dir_r is None:
                dir_r = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        bias_l = (fail_gain * mag_scale) * np.asarray(dir_l, dtype=np.float32)
        bias_r = (fail_gain * mag_scale) * np.asarray(dir_r, dtype=np.float32)
        mode_tag = "eef_directional_fail"
    else:
        bias_l = np.random.normal(0.0, pos_std * mag_scale, size=(3,)).astype(np.float32)
        bias_r = np.random.normal(0.0, pos_std * mag_scale, size=(3,)).astype(np.float32)
        mode_tag = "eef_gaussian"

    solved_any = False
    solved_cnt = 0
    total_cnt = int(out.shape[0])
    exec_horizon = total_cnt if rollout_exec_steps <= 0 else int(np.clip(rollout_exec_steps, 1, total_cnt))
    for t in range(out.shape[0]):
        if ramp_enable:
            frac = float(min(t, exec_horizon - 1)) / float(max(1, exec_horizon - 1))
            ramp = ramp_min + (1.0 - ramp_min) * (frac ** ramp_power)
        else:
            ramp = 1.0
        if abs(ramp) <= ramp_apply_eps:
            # Important: do not call IK when perturbation magnitude is effectively zero,
            # otherwise IK branch switching can introduce artificial early deviation.
            continue

        ql = out[t, 0:6].astype(np.float32)
        qr = out[t, 7:13].astype(np.float32)
        fr = fk.forward(ql, qr)

        l_link_p, l_link_q = fr["left"][0].astype(np.float32), fr["left"][1].astype(np.float32)
        r_link_p, r_link_q = fr["right"][0].astype(np.float32), fr["right"][1].astype(np.float32)
        lp, lq = _link6_to_tcp_pose(l_link_p, l_link_q, offset_x=tcp_offset_x)
        rp, rq = _link6_to_tcp_pose(r_link_p, r_link_q, offset_x=tcp_offset_x)

        target_lp = sapien.Pose((lp + ramp * bias_l).astype(np.float32), lq.astype(np.float32))
        target_rp = sapien.Pose((rp + ramp * bias_r).astype(np.float32), rq.astype(np.float32))

        qpos_full = np.zeros(len(fk.jnames), dtype=np.float32)
        qpos_full[fk.fl_idx] = ql
        qpos_full[fk.fr_idx] = qr

        left_ok, right_ok = (not active_left_arm), (not active_right_arm)
        q_sol_l, q_sol_r = ql.copy(), qr.copy()
        try:
            if active_left_arm:
                res_l = planner_l.plan_path(qpos_full, target_lp, arms_tag="left")
                if res_l.get("status") == "Success":
                    q_sol_l = np.asarray(res_l["position"][-1], dtype=np.float32)
                    left_ok = True
            if active_right_arm:
                res_r = planner_r.plan_path(qpos_full, target_rp, arms_tag="right")
                if res_r.get("status") == "Success":
                    q_sol_r = np.asarray(res_r["position"][-1], dtype=np.float32)
                    right_ok = True
        except Exception:
            continue

        cap = max(0.0, joint_delta_cap * float(ramp))
        if cap <= 1e-8:
            continue
        updated = False
        if active_left_arm and left_ok:
            dq_l = np.clip(q_sol_l - ql, -cap, cap)
            out[t, 0:6] = ql + dq_l
            updated = True
        if active_right_arm and right_ok:
            dq_r = np.clip(q_sol_r - qr, -cap, cap)
            out[t, 7:13] = qr + dq_r
            updated = True
        if updated:
            solved_any = True
            solved_cnt += 1

    out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
    out[:, 13] = np.clip(out[:, 13], 0.0, 1.0)
    info = {"eef_solved_cnt": int(solved_cnt), "eef_total_cnt": int(total_cnt)}
    if total_cnt > 0:
        info["eef_solved_ratio"] = float(solved_cnt) / float(total_cnt)
    info["eef_ramp_enable"] = bool(ramp_enable)
    info["eef_ramp_min"] = float(ramp_min)
    info["eef_ramp_power"] = float(ramp_power)
    info["eef_joint_delta_cap"] = float(joint_delta_cap)
    info["eef_exec_horizon"] = int(exec_horizon)
    info["perturb_mag_scale"] = float(mag_scale)
    info["active_left_arm"] = bool(active_left_arm)
    info["active_right_arm"] = bool(active_right_arm)
    return out, solved_any, mode_tag, info


def _sample_error_mode(phase_key, cfg):
    req = str(cfg.get("perturb_error_mode", "legacy")).strip().lower()
    if req in {"translation", "rotation", "no_ops", "gripper_close", "gripper_open"}:
        return req
    if req != "auto":
        return None
    probs = {
        "approach": [("translation", 0.55), ("rotation", 0.25), ("gripper_close", 0.20)],
        "pregrasp": [("gripper_close", 0.65), ("translation", 0.25), ("rotation", 0.10)],
        "transport": [("translation", 0.65), ("rotation", 0.25), ("gripper_open", 0.10)],
        "place": [("gripper_open", 0.65), ("translation", 0.25), ("rotation", 0.10)],
    }
    items = probs.get(phase_key, probs["transport"])
    names = [x[0] for x in items]
    p = np.array([x[1] for x in items], dtype=np.float64)
    p = p / np.sum(p)
    return np.random.choice(names, p=p).item()


def _perturb_action_chunk_no_ops(
    act_raw, cfg, curr_left_q=None, curr_right_q=None, curr_left_g=None, curr_right_g=None,
    active_left_arm=True, active_right_arm=True, active_left_gripper=True, active_right_gripper=True
):
    out = act_raw.copy()
    beta = float(cfg.get("perturb_noop_beta", 0.1))
    beta = float(np.clip(beta, 0.0, 1.0))
    beta_ramp = bool(cfg.get("perturb_noop_beta_ramp", True))
    beta_end = float(cfg.get("perturb_noop_beta_end", min(1.0, beta * 3.0)))
    beta_end = float(np.clip(beta_end, 0.0, 1.0))

    if curr_left_q is not None and curr_right_q is not None:
        prev_l = np.asarray(curr_left_q, dtype=np.float32).copy()
        prev_r = np.asarray(curr_right_q, dtype=np.float32).copy()
    else:
        prev_l = out[0, 0:6].astype(np.float32).copy()
        prev_r = out[0, 7:13].astype(np.float32).copy()

    if curr_left_g is not None and curr_right_g is not None:
        prev_lg = float(curr_left_g)
        prev_rg = float(curr_right_g)
    else:
        prev_lg = float(out[0, 6])
        prev_rg = float(out[0, 13])

    for t in range(out.shape[0]):
        if beta_ramp:
            frac = float(t) / float(max(1, out.shape[0] - 1))
            b_t = beta + (beta_end - beta) * frac
        else:
            b_t = beta
        b_t = float(np.clip(b_t, 0.0, 1.0))

        cmd_l = act_raw[t, 0:6].astype(np.float32)
        cmd_r = act_raw[t, 7:13].astype(np.float32)
        prev_l = prev_l + b_t * (cmd_l - prev_l)
        prev_r = prev_r + b_t * (cmd_r - prev_r)
        out[t, 0:6] = prev_l if active_left_arm else cmd_l
        out[t, 7:13] = prev_r if active_right_arm else cmd_r

        cmd_lg = float(act_raw[t, 6])
        cmd_rg = float(act_raw[t, 13])
        prev_lg = prev_lg + b_t * (cmd_lg - prev_lg)
        prev_rg = prev_rg + b_t * (cmd_rg - prev_rg)
        out[t, 6] = prev_lg if active_left_gripper else cmd_lg
        out[t, 13] = prev_rg if active_right_gripper else cmd_rg
    out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
    out[:, 13] = np.clip(out[:, 13], 0.0, 1.0)
    return out, True, "no_ops", None


def _perturb_action_chunk_gripper(
    out, mode, init_left_grip, init_right_grip, cfg, active_left_gripper=True, active_right_gripper=True
):
    out = out.copy()
    delay = int(max(0, cfg.get("perturb_gripper_delay_steps", 3)))
    close_min = float(cfg.get("perturb_gripper_close_min", 0.35))
    open_max = float(cfg.get("perturb_gripper_open_max", 0.75))
    trans_steps = int(max(1, cfg.get("perturb_gripper_transition_steps", 4)))
    for gi, init_g, active in [(6, init_left_grip, active_left_gripper), (13, init_right_grip, active_right_gripper)]:
        if not bool(active):
            continue
        seq = out[:, gi].copy()
        if mode == "gripper_close":
            closing = np.where(seq < init_g - 1e-4)[0]
            if closing.size > 0:
                s = int(closing[0])
                hold_end = min(out.shape[0], s + delay)
                out[s:hold_end, gi] = init_g
                target = np.maximum(seq[min(hold_end, out.shape[0] - 1)], close_min)
                trans_end = min(out.shape[0], hold_end + trans_steps)
                if hold_end < trans_end:
                    out[hold_end:trans_end, gi] = np.linspace(init_g, target, trans_end - hold_end, endpoint=False)
                if trans_end < out.shape[0]:
                    out[trans_end:, gi] = np.maximum(out[trans_end:, gi], target)
        else:
            opening = np.where(seq > init_g + 1e-4)[0]
            if opening.size > 0:
                s = int(opening[0])
                hold_end = min(out.shape[0], s + delay)
                out[s:hold_end, gi] = init_g
                target = np.minimum(seq[min(hold_end, out.shape[0] - 1)], open_max)
                trans_end = min(out.shape[0], hold_end + trans_steps)
                if hold_end < trans_end:
                    out[hold_end:trans_end, gi] = np.linspace(init_g, target, trans_end - hold_end, endpoint=False)
                if trans_end < out.shape[0]:
                    out[trans_end:, gi] = np.minimum(out[trans_end:, gi], target)
    out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
    out[:, 13] = np.clip(out[:, 13], 0.0, 1.0)
    tag = "gripper_close" if mode == "gripper_close" else "gripper_open"
    return out, True, tag, None


def _perturb_action_chunk_rotation_eef(act_raw, cfg, fk, planner_l, planner_r, active_left_arm=True, active_right_arm=True):
    if fk is None or planner_l is None or planner_r is None:
        return act_raw, False, "rot_missing_modules", None
    import sapien
    from scipy.spatial.transform import Rotation as R
    out = act_raw.copy()
    active_left_arm = bool(active_left_arm)
    active_right_arm = bool(active_right_arm)
    if (not active_left_arm) and (not active_right_arm):
        return out, False, "rot_inactive_arms", {"active_left_arm": False, "active_right_arm": False}
    tcp_offset_x = float(cfg.get("perturb_eef_tcp_offset_x", 0.085))
    angle_max_deg = float(cfg.get("perturb_rot_max_deg", 15.0))
    mag_rand = bool(cfg.get("perturb_mag_random", False))
    mag_min = float(cfg.get("perturb_mag_rand_min", 0.8))
    mag_max = float(cfg.get("perturb_mag_rand_max", 1.2))
    if mag_max < mag_min:
        mag_min, mag_max = mag_max, mag_min
    mag_scale = float(np.random.uniform(mag_min, mag_max)) if mag_rand else 1.0
    angle_max_deg = angle_max_deg * mag_scale
    angle_max = np.deg2rad(angle_max_deg)
    ramp_enable = bool(cfg.get("perturb_eef_ramp", True))
    ramp_min = float(cfg.get("perturb_eef_ramp_min", 0.0))
    ramp_power = float(cfg.get("perturb_eef_ramp_power", 1.0))
    ramp_apply_eps = float(cfg.get("perturb_eef_ramp_apply_eps", 1e-4))
    joint_delta_cap = float(cfg.get("perturb_eef_joint_delta_cap", 0.12))
    rollout_exec_steps = int(cfg.get("rollout_exec_steps", 0))
    axis_l = cfg.get("perturb_rot_axis_left_vec", np.array([0.0, 0.0, 1.0], dtype=np.float32))
    axis_r = cfg.get("perturb_rot_axis_right_vec", np.array([0.0, 0.0, 1.0], dtype=np.float32))
    rand_axis = bool(cfg.get("perturb_rotation_random_axis", True))
    if rand_axis:
        axis_l = _sample_unit_vec3()
        axis_r = _sample_unit_vec3()

    solved = 0
    total = int(out.shape[0])
    exec_horizon = total if rollout_exec_steps <= 0 else int(np.clip(rollout_exec_steps, 1, total))
    for t in range(total):
        frac = float(min(t, exec_horizon - 1)) / float(max(1, exec_horizon - 1))
        ramp = (ramp_min + (1.0 - ramp_min) * (frac ** ramp_power)) if ramp_enable else 1.0
        if abs(ramp) <= ramp_apply_eps:
            continue
        ql = out[t, 0:6].astype(np.float32)
        qr = out[t, 7:13].astype(np.float32)
        fr = fk.forward(ql, qr)
        l_link_p, l_link_q = fr["left"][0].astype(np.float32), fr["left"][1].astype(np.float32)
        r_link_p, r_link_q = fr["right"][0].astype(np.float32), fr["right"][1].astype(np.float32)
        lp, lq = _link6_to_tcp_pose(l_link_p, l_link_q, offset_x=tcp_offset_x)
        rp, rq = _link6_to_tcp_pose(r_link_p, r_link_q, offset_x=tcp_offset_x)
        rot_l = R.from_rotvec(np.asarray(axis_l, dtype=np.float32) * (angle_max * ramp))
        rot_r = R.from_rotvec(np.asarray(axis_r, dtype=np.float32) * (angle_max * ramp))
        ql_xyzw = np.array([lq[1], lq[2], lq[3], lq[0]], dtype=np.float32)
        qr_xyzw = np.array([rq[1], rq[2], rq[3], rq[0]], dtype=np.float32)
        ql_new = (rot_l * R.from_quat(ql_xyzw)).as_quat()
        qr_new = (rot_r * R.from_quat(qr_xyzw)).as_quat()
        lq_new = np.array([ql_new[3], ql_new[0], ql_new[1], ql_new[2]], dtype=np.float32)
        rq_new = np.array([qr_new[3], qr_new[0], qr_new[1], qr_new[2]], dtype=np.float32)
        target_lp = sapien.Pose(lp.astype(np.float32), lq_new)
        target_rp = sapien.Pose(rp.astype(np.float32), rq_new)
        qpos_full = np.zeros(len(fk.jnames), dtype=np.float32)
        qpos_full[fk.fl_idx] = ql
        qpos_full[fk.fr_idx] = qr
        left_ok, right_ok = (not active_left_arm), (not active_right_arm)
        q_sol_l, q_sol_r = ql.copy(), qr.copy()
        try:
            if active_left_arm:
                res_l = planner_l.plan_path(qpos_full, target_lp, arms_tag="left")
                if res_l.get("status") == "Success":
                    q_sol_l = np.asarray(res_l["position"][-1], dtype=np.float32)
                    left_ok = True
            if active_right_arm:
                res_r = planner_r.plan_path(qpos_full, target_rp, arms_tag="right")
                if res_r.get("status") == "Success":
                    q_sol_r = np.asarray(res_r["position"][-1], dtype=np.float32)
                    right_ok = True
        except Exception:
            continue
        cap = max(0.0, joint_delta_cap * float(ramp))
        if cap <= 1e-8:
            continue
        updated = False
        if active_left_arm and left_ok:
            out[t, 0:6] = ql + np.clip(q_sol_l - ql, -cap, cap)
            updated = True
        if active_right_arm and right_ok:
            out[t, 7:13] = qr + np.clip(q_sol_r - qr, -cap, cap)
            updated = True
        if updated:
            solved += 1
    out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
    out[:, 13] = np.clip(out[:, 13], 0.0, 1.0)
    info = {"eef_solved_cnt": int(solved), "eef_total_cnt": int(total), "error_mode": "rotation", "perturb_mag_scale": float(mag_scale)}
    if total > 0:
        info["eef_solved_ratio"] = float(solved) / float(total)
    return out, solved > 0, "rotation", info


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
    l_pose = left_ep[idx].astype(np.float32).copy()   # [x,y,z,qw,qx,qy,qz]
    r_pose = right_ep[idx].astype(np.float32).copy()

    mag_rand = bool(cfg.get("perturb_mag_random", False))
    mag_min = float(cfg.get("perturb_mag_rand_min", 0.8))
    mag_max = float(cfg.get("perturb_mag_rand_max", 1.2))
    if mag_max < mag_min:
        mag_min, mag_max = mag_max, mag_min
    mag_scale = float(np.random.uniform(mag_min, mag_max)) if mag_rand else 1.0

    mode = str(error_mode).lower()
    curr_left_q = np.asarray(curr_left_q, dtype=np.float32)
    curr_right_q = np.asarray(curr_right_q, dtype=np.float32)
    anti_gt_cos_thresh = float(np.clip(cfg.get("perturb_anti_gt_cos_thresh", 0.6), 0.0, 1.0))
    anti_dbg = {
        "anti_gt_cos_thresh": float(anti_gt_cos_thresh),
        "anti_gt_left_cos": None,
        "anti_gt_right_cos": None,
        "anti_gt_left_flipped": False,
        "anti_gt_right_flipped": False,
    }
    if (not active_left_arm) and (not active_right_arm) and mode not in {"gripper_close", "gripper_open"}:
        return out, False, "target_pose_inactive_arms", None

    if mode == "rotation":
        angle_max_deg = float(cfg.get("perturb_rot_max_deg", 15.0)) * mag_scale
        angle = np.deg2rad(angle_max_deg)
        fr_curr = fk.forward(curr_left_q, curr_right_q)
        lq_curr_wxyz = np.asarray(fr_curr["left"][1], dtype=np.float32)
        rq_curr_wxyz = np.asarray(fr_curr["right"][1], dtype=np.float32)
        axis_l = cfg.get("perturb_rot_axis_left_vec", np.array([0.0, 0.0, 1.0], dtype=np.float32))
        axis_r = cfg.get("perturb_rot_axis_right_vec", np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if forced_axis_left is not None:
            axis_l = np.asarray(forced_axis_left, dtype=np.float32)
        if forced_axis_right is not None:
            axis_r = np.asarray(forced_axis_right, dtype=np.float32)
        if forced_axis_left is None and forced_axis_right is None and bool(cfg.get("perturb_rotation_random_axis", True)):
            axis_l = _sample_unit_vec3()
            axis_r = _sample_unit_vec3()
        axis_l = _unit_vec3(axis_l) if axis_l is not None else _sample_unit_vec3()
        axis_r = _unit_vec3(axis_r) if axis_r is not None else _sample_unit_vec3()
        if active_left_arm:
            q_curr_xyzw = np.array([lq_curr_wxyz[1], lq_curr_wxyz[2], lq_curr_wxyz[3], lq_curr_wxyz[0]], dtype=np.float32)
            q_gt_xyzw = np.array([l_pose[4], l_pose[5], l_pose[6], l_pose[3]], dtype=np.float32)
            gt_rotvec_l = (R.from_quat(q_gt_xyzw) * R.from_quat(q_curr_xyzw).inv()).as_rotvec().astype(np.float32)
            axis_l, cos_l, flip_l = _anti_align_with_gt(axis_l, gt_rotvec_l, anti_gt_cos_thresh)
            anti_dbg["anti_gt_left_cos"] = None if cos_l is None else float(cos_l)
            anti_dbg["anti_gt_left_flipped"] = bool(flip_l)
        if active_right_arm:
            q_curr_xyzw = np.array([rq_curr_wxyz[1], rq_curr_wxyz[2], rq_curr_wxyz[3], rq_curr_wxyz[0]], dtype=np.float32)
            q_gt_xyzw = np.array([r_pose[4], r_pose[5], r_pose[6], r_pose[3]], dtype=np.float32)
            gt_rotvec_r = (R.from_quat(q_gt_xyzw) * R.from_quat(q_curr_xyzw).inv()).as_rotvec().astype(np.float32)
            axis_r, cos_r, flip_r = _anti_align_with_gt(axis_r, gt_rotvec_r, anti_gt_cos_thresh)
            anti_dbg["anti_gt_right_cos"] = None if cos_r is None else float(cos_r)
            anti_dbg["anti_gt_right_flipped"] = bool(flip_r)
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
        fail_gain = float(cfg.get("perturb_eef_fail_gain", 0.03)) * mag_scale
        fr_curr = fk.forward(curr_left_q, curr_right_q)
        lp_curr = np.asarray(fr_curr["left"][0], dtype=np.float32)
        rp_curr = np.asarray(fr_curr["right"][0], dtype=np.float32)
        dir_l = cfg.get("perturb_eef_fail_dir_left_vec", None)
        dir_r = cfg.get("perturb_eef_fail_dir_right_vec", None)
        if forced_dir_left is not None:
            dir_l = np.asarray(forced_dir_left, dtype=np.float32)
        if forced_dir_right is not None:
            dir_r = np.asarray(forced_dir_right, dtype=np.float32)
        if forced_dir_left is None and forced_dir_right is None and bool(cfg.get("perturb_translation_random_dir", True)):
            dir_l = _sample_unit_vec3()
            dir_r = _sample_unit_vec3()
        else:
            if dir_l is None:
                dir_l = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if dir_r is None:
                dir_r = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        if active_left_arm:
            gt_vec_l = l_pose[:3] - lp_curr
            dir_l, cos_l, flip_l = _anti_align_with_gt(dir_l, gt_vec_l, anti_gt_cos_thresh)
            anti_dbg["anti_gt_left_cos"] = None if cos_l is None else float(cos_l)
            anti_dbg["anti_gt_left_flipped"] = bool(flip_l)
        if active_right_arm:
            gt_vec_r = r_pose[:3] - rp_curr
            dir_r, cos_r, flip_r = _anti_align_with_gt(dir_r, gt_vec_r, anti_gt_cos_thresh)
            anti_dbg["anti_gt_right_cos"] = None if cos_r is None else float(cos_r)
            anti_dbg["anti_gt_right_flipped"] = bool(flip_r)
        if active_left_arm:
            l_pose[:3] = l_pose[:3] + fail_gain * np.asarray(dir_l, dtype=np.float32)
        if active_right_arm:
            r_pose[:3] = r_pose[:3] + fail_gain * np.asarray(dir_r, dtype=np.float32)
        mode_tag = "target_pose_translation"
    elif mode == "no_ops":
        # Keep near current state: use an earlier/closer GT waypoint as target.
        # This avoids direct pose interpolation in Cartesian space.
        beta = float(np.clip(cfg.get("perturb_noop_beta_end", 0.3), 0.0, 1.0))
        idx_start = int(np.clip(int(t_star), 0, len(left_ep) - 1))
        idx_near = int(np.clip(idx_start + int(np.floor(beta * max(1, int(rollout_exec_steps)))),
                               idx_start, len(left_ep) - 1))
        l_pose = left_ep[idx_near].astype(np.float32).copy()
        r_pose = right_ep[idx_near].astype(np.float32).copy()
        idx = idx_near
        mode_tag = "target_pose_no_ops"
    elif mode in ("gripper_close", "gripper_open"):
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
    if mode in {"gripper_close", "gripper_open"}:
        lt = np.repeat(curr_left_q[None, :], T, axis=0)
        rt = np.repeat(curr_right_q[None, :], T, axis=0)
    else:
        lt = np.repeat(curr_left_q[None, :], T, axis=0)
        rt = np.repeat(curr_right_q[None, :], T, axis=0)

    if active_left_arm and mode not in {"gripper_close", "gripper_open"}:
        res_l = planner_l.plan_path(qpos_full, sapien.Pose(l_pose[:3], l_pose[3:7]), arms_tag="left")
        if res_l.get("status") != "Success":
            return act_raw, False, "target_pose_left_plan_fail", {"target_idx": int(idx)}
        l_path = np.asarray(res_l["position"], dtype=np.float32)
        if l_path.ndim != 2 or l_path.shape[0] == 0:
            return act_raw, False, "target_pose_left_empty_path", {"target_idx": int(idx)}
        # Planner path usually includes current state at index 0; action chunk should be future commands.
        l_future = l_path[1:] if l_path.shape[0] > 1 else l_path
        lt = _compose_plan_with_gt_tail(l_future, out[:, 0:6], local_target_idx, T)
    if active_right_arm and mode not in {"gripper_close", "gripper_open"}:
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
    elif mode == "gripper_open":
        lg_tgt = 1.0
        rg_tgt = 1.0
    else:
        lg_tgt, rg_tgt = lg_gt, rg_gt

    # Gripper interpolation:
    # - gripper_open/close: start changing from the 1st action and finish at step T.
    # - other modes: keep previous behavior (tail-fast fallback) for mild smoothing.
    if mode in {"gripper_close", "gripper_open"}:
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
    t_star=None, rollout_exec_steps=None
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
    if selected_mode is not None:
        use_target_pose = bool(cfg.get("perturb_use_target_pose_planner", True))
        if use_target_pose and (t_star is not None) and (left_ep is not None) and (right_ep is not None):
            if selected_mode in {"translation", "rotation", "no_ops", "gripper_close", "gripper_open"}:
                reject_enable = bool(cfg.get("perturb_reject_sampling_enable", True))
                reject_trials = int(max(1, cfg.get("perturb_reject_max_trials", 4)))
                reject_min_score = float(cfg.get("perturb_reject_min_score", 0.15))
                do_reject = reject_enable and selected_mode in {"translation", "rotation"} and (fk is not None)
                if do_reject:
                    target_h = int(rollout_exec_steps) if rollout_exec_steps is not None else int(act_raw.shape[0])
                    target_h = int(max(1, target_h))
                    gt_idx = int(np.clip(int(t_star) + target_h - 1, 0, len(left_ep) - 1))
                    prefilter_pool = int(max(reject_trials, cfg.get("perturb_reject_prefilter_pool", reject_trials * 2)))
                    pre_candidates = []  # (score, spec, aux_debug)
                    fr_curr = fk.forward(curr_left_q, curr_right_q)
                    lp_curr = np.asarray(fr_curr["left"][0], dtype=np.float32)
                    rp_curr = np.asarray(fr_curr["right"][0], dtype=np.float32)
                    lq_curr = np.asarray(fr_curr["left"][1], dtype=np.float32)
                    rq_curr = np.asarray(fr_curr["right"][1], dtype=np.float32)
                    gt_lp = np.asarray(left_ep[gt_idx, :3], dtype=np.float32)
                    gt_rp = np.asarray(right_ep[gt_idx, :3], dtype=np.float32)
                    if selected_mode == "translation":
                        gt_v_l = _unit_vec3(gt_lp - lp_curr) if active_left_arm else None
                        gt_v_r = _unit_vec3(gt_rp - rp_curr) if active_right_arm else None
                        for _ in range(prefilter_pool):
                            dl = _sample_unit_vec3()
                            dr = _sample_unit_vec3()
                            s, c = 0.0, 0
                            cos_l, cos_r = None, None
                            if gt_v_l is not None:
                                cos_l = float(np.dot(dl, gt_v_l))
                                s += -cos_l; c += 1
                            if gt_v_r is not None:
                                cos_r = float(np.dot(dr, gt_v_r))
                                s += -cos_r; c += 1
                            pre_candidates.append((s / max(1, c), {"forced_dir_left": dl, "forced_dir_right": dr}, {"cos_l": cos_l, "cos_r": cos_r}))
                    else:
                        from scipy.spatial.transform import Rotation as R
                        ql_gt = np.asarray(left_ep[gt_idx, 3:7], dtype=np.float32)
                        qr_gt = np.asarray(right_ep[gt_idx, 3:7], dtype=np.float32)
                        gt_rv_l = None
                        gt_rv_r = None
                        if active_left_arm:
                            q_curr_xyzw = np.array([lq_curr[1], lq_curr[2], lq_curr[3], lq_curr[0]], dtype=np.float32)
                            q_gt_xyzw = np.array([ql_gt[1], ql_gt[2], ql_gt[3], ql_gt[0]], dtype=np.float32)
                            gt_rv_l = _unit_vec3((R.from_quat(q_gt_xyzw) * R.from_quat(q_curr_xyzw).inv()).as_rotvec())
                        if active_right_arm:
                            q_curr_xyzw = np.array([rq_curr[1], rq_curr[2], rq_curr[3], rq_curr[0]], dtype=np.float32)
                            q_gt_xyzw = np.array([qr_gt[1], qr_gt[2], qr_gt[3], qr_gt[0]], dtype=np.float32)
                            gt_rv_r = _unit_vec3((R.from_quat(q_gt_xyzw) * R.from_quat(q_curr_xyzw).inv()).as_rotvec())
                        for _ in range(prefilter_pool):
                            al = _sample_unit_vec3()
                            ar = _sample_unit_vec3()
                            s, c = 0.0, 0
                            cos_l, cos_r = None, None
                            if gt_rv_l is not None:
                                cos_l = float(np.dot(al, gt_rv_l))
                                s += -cos_l; c += 1
                            if gt_rv_r is not None:
                                cos_r = float(np.dot(ar, gt_rv_r))
                                s += -cos_r; c += 1
                            pre_candidates.append((s / max(1, c), {"forced_axis_left": al, "forced_axis_right": ar}, {"cos_l": cos_l, "cos_r": cos_r}))
                    pre_candidates.sort(key=lambda x: x[0], reverse=True)
                    rank_accept = None
                    for ridx, (s, _, _) in enumerate(pre_candidates[:reject_trials]):
                        if float(s) >= float(reject_min_score):
                            rank_accept = ridx
                            break
                    if rank_accept is None:
                        rank_accept = 0 if len(pre_candidates) > 0 else None
                    if rank_accept is None:
                        out, ok, tag, info = act_raw, False, "target_pose_reject_no_candidate", {"error_mode": selected_mode}
                    else:
                        s_sel, t_spec, aux = pre_candidates[rank_accept]
                        out, ok, tag, info = _perturb_action_chunk_target_pose(
                            act_raw, cfg, fk, planner_l, planner_r,
                            curr_left_q, curr_right_q, init_left_grip, init_right_grip,
                            left_ep, right_ep, left_grip_traj, right_grip_traj, t_star, rollout_exec_steps,
                            error_mode=selected_mode,
                            active_left_arm=active_left_arm, active_right_arm=active_right_arm,
                            active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper,
                            forced_dir_left=t_spec.get("forced_dir_left"),
                            forced_dir_right=t_spec.get("forced_dir_right"),
                            forced_axis_left=t_spec.get("forced_axis_left"),
                            forced_axis_right=t_spec.get("forced_axis_right"),
                        )
                        info = {} if not isinstance(info, dict) else info
                        info.update({
                            "reject_sampling_enabled": True,
                            "reject_sampling_trial": int(rank_accept + 1),
                            "reject_sampling_trials_max": int(reject_trials),
                            "reject_sampling_prefilter_pool": int(prefilter_pool),
                            "reject_sampling_base_err": None,
                            "reject_sampling_cand_err": None,
                            "reject_sampling_min_delta": None,
                            "reject_sampling_min_score": float(reject_min_score),
                            "reject_sampling_pose_score": float(s_sel),
                            "reject_sampling_accepted": bool(float(s_sel) >= float(reject_min_score)),
                            "reject_sampling_fallback_best": bool(float(s_sel) < float(reject_min_score)),
                            "reject_sampling_cos_left": aux.get("cos_l", None),
                            "reject_sampling_cos_right": aux.get("cos_r", None),
                        })
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
        elif selected_mode == "translation":
            out, ok, tag, info = _perturb_action_chunk_eef(
                act_raw, cfg, fk, planner_l, planner_r,
                active_left_arm=active_left_arm, active_right_arm=active_right_arm
            )
        elif selected_mode == "rotation":
            out, ok, tag, info = _perturb_action_chunk_rotation_eef(
                act_raw, cfg, fk, planner_l, planner_r,
                active_left_arm=active_left_arm, active_right_arm=active_right_arm
            )
        elif selected_mode == "no_ops":
            out, ok, tag, info = _perturb_action_chunk_no_ops(
                act_raw, cfg,
                curr_left_q=curr_left_q, curr_right_q=curr_right_q,
                curr_left_g=init_left_grip, curr_right_g=init_right_grip,
                active_left_arm=active_left_arm, active_right_arm=active_right_arm,
                active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper,
            )
        elif selected_mode == "gripper_close":
            out, ok, tag, info = _perturb_action_chunk_gripper(
                act_raw, "gripper_close", init_left_grip, init_right_grip, cfg,
                active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper
            )
        else:
            out, ok, tag, info = _perturb_action_chunk_gripper(
                act_raw, "gripper_open", init_left_grip, init_right_grip, cfg,
                active_left_gripper=active_left_gripper, active_right_gripper=active_right_gripper
            )
        if isinstance(info, dict):
            info["error_mode"] = selected_mode
        return out, ok, f"mode_{tag}", info

    perturb_space = str(cfg.get("perturb_space", "joint")).strip().lower()
    if perturb_space == "eef":
        return _perturb_action_chunk_eef(
            act_raw, cfg, fk, planner_l, planner_r,
            active_left_arm=active_left_arm, active_right_arm=active_right_arm
        )

    out = act_raw.copy()
    alpha = float(cfg.get("perturb_lp_alpha", 0.35))
    rho = float(cfg.get("perturb_noise_rho", 0.85))
    noise_std = float(cfg.get("perturb_noise_std", 0.01))
    vel_lim = float(cfg.get("perturb_vel_limit", 0.08))
    acc_lim = float(cfg.get("perturb_acc_limit", 0.04))
    bias_prob = float(cfg.get("perturb_bias_prob", 0.2))
    bias_std = float(cfg.get("perturb_bias_std", 0.03))
    q_abs_lim = float(cfg.get("perturb_joint_limit_abs", 3.14))
    perturb_mode = str(cfg.get("perturb_mode", "dyn")).strip().lower()
    fail_gain = float(cfg.get("perturb_fail_gain", 0.12))
    fail_dir = cfg.get("perturb_fail_direction_vec", None)

    if dyn_state is None:
        dyn_state = {
            "prev_exec_arm": _extract_arm(out[0]).astype(np.float32),
            "prev_vel_arm": np.zeros((12,), dtype=np.float32),
            "prev_noise_arm": np.zeros((12,), dtype=np.float32),
            "bias_arm": np.zeros((12,), dtype=np.float32),
        }

    if perturb_mode == "directional_fail":
        if fail_dir is None:
            fail_dir = np.ones((12,), dtype=np.float32)
            fail_dir /= np.linalg.norm(fail_dir)
        dyn_state["bias_arm"] = (fail_gain * np.asarray(fail_dir, dtype=np.float32)).astype(np.float32)
        mode_tag = "dyn_directional_fail"
    else:
        if np.random.rand() < bias_prob:
            dyn_state["bias_arm"] = np.random.normal(0.0, bias_std, size=(12,)).astype(np.float32)
        else:
            dyn_state["bias_arm"] *= 0.95  # decay old bias smoothly
        mode_tag = "dyn_lp_ar"

    prev_exec = dyn_state["prev_exec_arm"].astype(np.float32)
    prev_vel = dyn_state["prev_vel_arm"].astype(np.float32)
    prev_noise = dyn_state["prev_noise_arm"].astype(np.float32)
    bias = dyn_state["bias_arm"].astype(np.float32)

    arm_mask = np.zeros((12,), dtype=np.float32)
    if bool(active_left_arm):
        arm_mask[0:6] = 1.0
    if bool(active_right_arm):
        arm_mask[6:12] = 1.0
    if np.all(arm_mask <= 0.0):
        return act_raw, False, "inactive_arms", dyn_state

    for t in range(out.shape[0]):
        cmd_arm = _extract_arm(out[t]).astype(np.float32)
        noise = rho * prev_noise + np.random.normal(0.0, noise_std, size=(12,)).astype(np.float32)
        target = cmd_arm + bias + noise

        # low-pass actuator response
        exec_arm = prev_exec + alpha * (target - prev_exec)

        # velocity limit
        vel = np.clip(exec_arm - prev_exec, -vel_lim, vel_lim)
        exec_arm = prev_exec + vel

        # acceleration limit
        acc = np.clip(vel - prev_vel, -acc_lim, acc_lim)
        vel = prev_vel + acc
        exec_arm = prev_exec + vel

        # coarse joint bound
        exec_arm = np.clip(exec_arm, -q_abs_lim, q_abs_lim)
        exec_arm = arm_mask * exec_arm + (1.0 - arm_mask) * cmd_arm

        out[t] = _write_arm(out[t], exec_arm)
        out[t, 6] = np.clip(out[t, 6], 0.0, 1.0)
        out[t, 13] = np.clip(out[t, 13], 0.0, 1.0)

        prev_exec = arm_mask * exec_arm + (1.0 - arm_mask) * cmd_arm
        prev_vel = arm_mask * vel
        prev_noise = arm_mask * noise

    dyn_state["prev_exec_arm"] = prev_exec
    dyn_state["prev_vel_arm"] = prev_vel
    dyn_state["prev_noise_arm"] = prev_noise
    return out, True, mode_tag, dyn_state


def main(args):
    set_seed(1)
    _rank = int(os.environ.get("RANK", -1))
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"[main][rank={_rank} local_rank={_local_rank}] start")
    # command line parameters
    is_eval = args.get("eval", False)
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args.get("onscreen_render", False)
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
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
        "save_freq": args['save_freq']
    }

    # if is_eval:
    #     ckpt_names = [f"policy_best.ckpt"]
    #     results = []
    #     for ckpt_name in ckpt_names:
    #         success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
    #         results.append([ckpt_name, success_rate, avg_return])

    #     for ckpt_name, success_rate, avg_return in results:
    #         print(f"{ckpt_name}: {success_rate=} {avg_return=}")
    #     print()
    #     exit()

    enable_wm = args['enable_wm_correction']
    fail_dir_vec = _parse_vec12(args.get("perturb_fail_direction", ""))
    eef_fail_dir_left_vec = _parse_vec3(args.get("perturb_eef_fail_dir_left", ""))
    eef_fail_dir_right_vec = _parse_vec3(args.get("perturb_eef_fail_dir_right", ""))
    rot_axis_left_vec = _parse_vec3(args.get("perturb_rot_axis_left", ""))
    rot_axis_right_vec = _parse_vec3(args.get("perturb_rot_axis_right", ""))
    start_margin = 0
    sample_skip_head = int(max(0, args.get('sample_skip_head', 0)))
    if enable_wm:
        wm_required = ['evac_ckpt', 'evac_config', 'urdf_path', 'curobo_left_yml',
                        'curobo_right_yml', 'raw_data_dir', 'act_init_ckpt',
                        'correction_threshold', 'max_rollout_steps', 'correction_freq',
                        'correction_weight', 'orient_weight', 'gripper_penalty']
        missing = [k for k in wm_required if args.get(k) is None]
        if missing:
            raise ValueError(f"--enable_wm_correction requires these args: {missing}")
        start_margin = int(args['max_rollout_steps']) * int(args['chunk_size'])
    raw_data_dir = args['raw_data_dir'] if enable_wm else None
    print(f"[main][rank={_rank}] before load_data | dataset_dir={dataset_dir} | num_episodes={num_episodes} | start_margin={start_margin}")
    _t_load = time.time()
    train_dataloader, val_dataloader, stats, _, max_action_len = load_data(dataset_dir, num_episodes, camera_names,
                                                                          batch_size_train, batch_size_val,
                                                                          raw_data_dir=raw_data_dir,
                                                                          start_margin=start_margin,
                                                                          sample_skip_head=sample_skip_head)
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
            'threshold': args['correction_threshold'],
            'max_rollout_steps': args['max_rollout_steps'],
            'single_rollout_correction': args.get('single_rollout_correction', False),
            'target_mode': args.get('target_mode', 'forward'),
            'target_lookahead_steps': args.get('target_lookahead_steps', 0),
            'closed_loop_action_trigger_enable': args.get('closed_loop_action_trigger_enable', False),
            'closed_loop_action_threshold': args.get('closed_loop_action_threshold', 0.2),
            'closed_loop_action_gripper_weight': args.get('closed_loop_action_gripper_weight', 1.0),
            'closed_loop_action_dist_mode': args.get('closed_loop_action_dist_mode', 'mean_chunk'),
            'closed_loop_action_last_k': args.get('closed_loop_action_last_k', 4),
            'closed_loop_max_rollouts': args.get('closed_loop_max_rollouts', 3),
            'closed_loop_fallback_force_correction': args.get('closed_loop_fallback_force_correction', True),
            'min_dist_fallback_force_correction': args.get('min_dist_fallback_force_correction', False),
            'min_dist_trigger_use_delta': args.get('min_dist_trigger_use_delta', False),
            'min_dist_recover_use_added': args.get('min_dist_recover_use_added', True),
            'min_dist_recover_ratio': args.get('min_dist_recover_ratio', 0.5),
            'debug_recover_eval_rollout': args.get('debug_recover_eval_rollout', False),
            'correction_interp_nearest_enable': args.get('correction_interp_nearest_enable', False),
            'correction_interp_prefix_ratio': args.get('correction_interp_prefix_ratio', 0.4),
            'correction_interp_smooth_enable': args.get('correction_interp_smooth_enable', True),
            'correction_interp_smooth_steps': args.get('correction_interp_smooth_steps', 3),
            'correction_interp_smooth_passes': args.get('correction_interp_smooth_passes', 2),
            'rollout_exec_steps': int(rollout_exec_steps),
            'chunk_size': args['chunk_size'],
            'max_action_len': max_action_len,
            'correction_freq': args['correction_freq'],
            'correction_weight': args['correction_weight'],
            'orient_weight': args['orient_weight'],
            'gripper_penalty': args['gripper_penalty'],
            'always_correction': args.get('always_correction', False),
            'evac_infer_kwargs': build_evac_infer_kwargs(args),
            'enable_perturb': args.get('enable_perturb', False),
            'perturb_prob': args.get('perturb_prob', 1.0),
            'perturb_error_mode': args.get('perturb_error_mode', 'legacy'),
            'perturb_space': args.get('perturb_space', 'joint'),
            'perturb_lp_alpha': args.get('perturb_lp_alpha', 0.35),
            'perturb_noise_std': args.get('perturb_noise_std', 0.01),
            'perturb_noise_rho': args.get('perturb_noise_rho', 0.85),
            'perturb_vel_limit': args.get('perturb_vel_limit', 0.08),
            'perturb_acc_limit': args.get('perturb_acc_limit', 0.04),
            'perturb_bias_prob': args.get('perturb_bias_prob', 0.2),
            'perturb_bias_std': args.get('perturb_bias_std', 0.03),
            'perturb_joint_limit_abs': args.get('perturb_joint_limit_abs', 3.14),
            'perturb_mode': args.get('perturb_mode', 'dyn'),
            'perturb_fail_gain': args.get('perturb_fail_gain', 0.12),
            'perturb_fail_direction_vec': fail_dir_vec,
            'perturb_eef_mode': args.get('perturb_eef_mode', 'gaussian'),
            'perturb_use_target_pose_planner': args.get('perturb_use_target_pose_planner', True),
            'perturb_eef_pos_std': args.get('perturb_eef_pos_std', 0.01),
            'perturb_eef_fail_gain': args.get('perturb_eef_fail_gain', 0.03),
            'perturb_eef_tcp_offset_x': args.get('perturb_eef_tcp_offset_x', 0.085),
            'perturb_eef_ramp': args.get('perturb_eef_ramp', True),
            'perturb_eef_ramp_min': args.get('perturb_eef_ramp_min', 0.0),
            'perturb_eef_ramp_power': args.get('perturb_eef_ramp_power', 1.0),
            'perturb_eef_ramp_apply_eps': args.get('perturb_eef_ramp_apply_eps', 1e-4),
            'perturb_eef_joint_delta_cap': args.get('perturb_eef_joint_delta_cap', 0.12),
            'perturb_translation_random_dir': args.get('perturb_translation_random_dir', True),
            'perturb_eef_fail_dir_left_vec': eef_fail_dir_left_vec,
            'perturb_eef_fail_dir_right_vec': eef_fail_dir_right_vec,
            'perturb_rot_max_deg': args.get('perturb_rot_max_deg', 15.0),
            'perturb_mag_random': args.get('perturb_mag_random', False),
            'perturb_mag_rand_min': args.get('perturb_mag_rand_min', 0.8),
            'perturb_mag_rand_max': args.get('perturb_mag_rand_max', 1.2),
            'perturb_rotation_random_axis': args.get('perturb_rotation_random_axis', True),
            'perturb_anti_gt_cos_thresh': args.get('perturb_anti_gt_cos_thresh', 0.6),
            'perturb_reject_sampling_enable': args.get('perturb_reject_sampling_enable', True),
            'perturb_reject_max_trials': args.get('perturb_reject_max_trials', 4),
            'perturb_reject_min_delta': args.get('perturb_reject_min_delta', 5e-4),
            'perturb_reject_min_score': args.get('perturb_reject_min_score', 0.15),
            'perturb_reject_prefilter_pool': args.get('perturb_reject_prefilter_pool', 8),
            'perturb_reject_orient_weight': args.get('perturb_reject_orient_weight', args.get('orient_weight', 0.0)),
            'perturb_reject_gripper_penalty': args.get('perturb_reject_gripper_penalty', args.get('gripper_penalty', 0.0)),
            'nearest_window_radius': args.get('nearest_window_radius', 32),
            'perturb_rot_axis_left_vec': rot_axis_left_vec,
            'perturb_rot_axis_right_vec': rot_axis_right_vec,
            'perturb_noop_lag_steps': args.get('perturb_noop_lag_steps', 3),
            'perturb_noop_alpha': args.get('perturb_noop_alpha', 0.85),
            'perturb_noop_transition_steps': args.get('perturb_noop_transition_steps', 4),
            'perturb_noop_beta': args.get('perturb_noop_beta', 0.1),
            'perturb_noop_beta_ramp': args.get('perturb_noop_beta_ramp', True),
            'perturb_noop_beta_end': args.get('perturb_noop_beta_end', 0.3),
            'perturb_gripper_delay_steps': args.get('perturb_gripper_delay_steps', 3),
            'perturb_gripper_transition_steps': args.get('perturb_gripper_transition_steps', 4),
            'perturb_gripper_close_min': args.get('perturb_gripper_close_min', 0.35),
            'perturb_gripper_open_max': args.get('perturb_gripper_open_max', 0.75),
            'perturb_gripper_fast_ratio': args.get('perturb_gripper_fast_ratio', 0.2),
            'perturb_active_joint_delta_thresh': args.get('perturb_active_joint_delta_thresh', 0.02),
            'perturb_active_gripper_delta_thresh': args.get('perturb_active_gripper_delta_thresh', 0.05),
            'vla_input_noise_enable': args.get('vla_input_noise_enable', False),
            'vla_img_noise_std': args.get('vla_img_noise_std', 0.0),
            'vla_qpos_noise_std': args.get('vla_qpos_noise_std', 0.0),
            'export_correction_dataset': args.get('export_correction_dataset', False),
            'export_correction_dir': args.get('export_correction_dir', ''),
        }
    config['act_init_ckpt'] = args.get('act_init_ckpt')
    config['debug_wm_correction'] = args.get('debug_wm_correction', False)
    print(f"[main][rank={_rank}] before train_bc | enable_wm={enable_wm}")
    train_bc(train_dataloader, val_dataloader, config)
    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # # save best checkpoint
    # ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    # torch.save(best_state_dict, ckpt_path)
    # print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


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


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


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
        frames, traj_frames = evac_model.inference(
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

    # Debug continuity helpers: keep explicit input frame and stitched video
    # whose first frame is input_frame.png, followed by model outputs.mp4.
    if save_dir is not None:
        try:
            inp_rgb = np.clip((img_rgb.permute(1, 2, 0).cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
            inp_bgr = inp_rgb[:, :, ::-1].copy()
            cv2.imwrite(os.path.join(target_dir, 'input_frame.png'), inp_bgr)
            out_mp4 = os.path.join(target_dir, 'outputs.mp4')
            if os.path.isfile(out_mp4):
                cap = cv2.VideoCapture(out_mp4)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 1e-3:
                    fps = 30.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    inp0 = inp_bgr if (inp_bgr.shape[1] == w and inp_bgr.shape[0] == h) else cv2.resize(
                        inp_bgr, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                    out_with_in = os.path.join(target_dir, 'outputs_with_input.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vw = cv2.VideoWriter(out_with_in, fourcc, float(fps), (w, h))
                    vw.write(inp0)
                    while True:
                        ok, fr = cap.read()
                        if not ok:
                            break
                        vw.write(fr)
                    vw.release()
                cap.release()
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

    # Align correction target trajectory with sampled qpos/action timestamp origin.
    start_ts = int(max(0, start_ts))
    left_ep = raw_data['left_endpose'][start_ts:]
    right_ep = raw_data['right_endpose'][start_ts:]
    threshold = cfg['threshold']
    always_correction = bool(cfg.get('always_correction', False))
    max_steps = cfg['max_rollout_steps']
    closed_loop_action_trigger_enable = bool(cfg.get('closed_loop_action_trigger_enable', False))
    closed_loop_action_threshold = float(cfg.get('closed_loop_action_threshold', 0.2))
    closed_loop_action_gripper_weight = float(cfg.get('closed_loop_action_gripper_weight', 1.0))
    closed_loop_action_dist_mode = str(cfg.get('closed_loop_action_dist_mode', 'mean_chunk')).strip().lower()
    closed_loop_action_last_k = int(max(1, cfg.get('closed_loop_action_last_k', 4)))
    closed_loop_max_rollouts = int(max(1, cfg.get('closed_loop_max_rollouts', 3)))
    closed_loop_fallback_force_correction = bool(cfg.get('closed_loop_fallback_force_correction', True))
    min_dist_fallback_force_correction = bool(cfg.get('min_dist_fallback_force_correction', False))
    min_dist_trigger_use_delta = bool(cfg.get('min_dist_trigger_use_delta', False))
    min_dist_recover_use_added = bool(cfg.get('min_dist_recover_use_added', True))
    min_dist_recover_ratio = float(cfg.get('min_dist_recover_ratio', 0.5))
    debug_recover_eval_rollout = bool(cfg.get('debug_recover_eval_rollout', False))
    single_rollout_correction = bool(cfg.get('single_rollout_correction', False))
    target_mode = str(cfg.get('target_mode', 'forward')).strip().lower()
    target_lookahead_steps = int(cfg.get('target_lookahead_steps', 0))
    correction_interp_nearest_enable = bool(cfg.get('correction_interp_nearest_enable', False))
    correction_interp_prefix_ratio = float(np.clip(cfg.get('correction_interp_prefix_ratio', 0.4), 0.0, 1.0))
    correction_interp_smooth_enable = bool(cfg.get('correction_interp_smooth_enable', True))
    correction_interp_smooth_steps = int(max(0, cfg.get('correction_interp_smooth_steps', 3)))
    correction_interp_smooth_passes = int(max(1, cfg.get('correction_interp_smooth_passes', 2)))
    chunk_size = cfg['chunk_size']
    rollout_exec_steps = int(cfg.get('rollout_exec_steps', chunk_size))
    max_action_len = cfg['max_action_len']
    n_previous = evac_cfg.n_previous
    orient_weight = cfg.get('orient_weight', 0.0)
    gripper_penalty = cfg.get('gripper_penalty', 0.0)

    left_grip_traj = raw_data['left_gripper'][start_ts:]
    right_grip_traj = raw_data['right_gripper'][start_ts:]
    vla_input_noise_enable = bool(cfg.get('vla_input_noise_enable', False))
    vla_img_noise_std = float(cfg.get('vla_img_noise_std', 0.0))
    vla_qpos_noise_std = float(cfg.get('vla_qpos_noise_std', 0.0))

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
    action_triggered = False
    action_trigger_step = None
    action_trigger_dist = None
    min_dist_triggered = False
    min_dist_trigger_step = None
    min_dist_trigger_dist = None
    pending_recover_valid = False
    pending_recover_added = 0.0
    pending_recover_from_step = None
    force_generate_correction = False

    gt_left_arm_full = raw_data.get('gt_left_arm', None)
    gt_right_arm_full = raw_data.get('gt_right_arm', None)
    gt_left_grip_full = raw_data.get('left_gripper', None)
    gt_right_grip_full = raw_data.get('right_gripper', None)

    def _build_gt_action_chunk(abs_start_idx, n_step, fallback_action):
        n_step = int(max(1, n_step))
        gt = np.zeros((n_step, 14), dtype=np.float32)
        fb = np.asarray(fallback_action, dtype=np.float32)
        for t in range(n_step):
            gi = int(np.clip(abs_start_idx + t, 0, max(0, raw_data['left_endpose'].shape[0] - 1)))
            if gt_left_arm_full is not None:
                gt[t, 0:6] = np.asarray(gt_left_arm_full[gi], dtype=np.float32)
            else:
                gt[t, 0:6] = fb[0:6]
            if gt_right_arm_full is not None:
                gt[t, 7:13] = np.asarray(gt_right_arm_full[gi], dtype=np.float32)
            else:
                gt[t, 7:13] = fb[7:13]
            if gt_left_grip_full is not None:
                gt[t, 6] = float(np.clip(gt_left_grip_full[gi], 0.0, 1.0))
            else:
                gt[t, 6] = float(np.clip(fb[6], 0.0, 1.0))
            if gt_right_grip_full is not None:
                gt[t, 13] = float(np.clip(gt_right_grip_full[gi], 0.0, 1.0))
            else:
                gt[t, 13] = float(np.clip(fb[13], 0.0, 1.0))
        return gt

    def _action_chunk_dist(pred_chunk, gt_chunk, gripper_w=1.0, mode='mean_chunk', last_k=4):
        p = np.asarray(pred_chunk, dtype=np.float32)
        g = np.asarray(gt_chunk, dtype=np.float32)
        if p.shape != g.shape:
            m = min(len(p), len(g))
            p = p[:m]
            g = g[:m]
        if p.shape[0] == 0:
            return 0.0
        d_l = np.linalg.norm(p[:, 0:6] - g[:, 0:6], axis=1)
        d_r = np.linalg.norm(p[:, 7:13] - g[:, 7:13], axis=1)
        d_g = np.abs(p[:, 6] - g[:, 6]) + np.abs(p[:, 13] - g[:, 13])
        d = d_l + d_r + float(gripper_w) * d_g
        if mode == 'last':
            return float(d[-1])
        if mode == 'last_k':
            k = int(max(1, min(int(last_k), d.shape[0])))
            return float(np.mean(d[-k:]))
        return float(np.mean(d))

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

    max_steps_eff = closed_loop_max_rollouts if closed_loop_action_trigger_enable else max_steps

    for step in range(max_steps_eff):
        step_role = 'recover_eval+perturb' if (min_dist_trigger_use_delta and pending_recover_valid) else 'perturb'
        recovery_pair_step = pending_recover_from_step
        evac_debug_dir = os.path.join(debug_dir, 'evac', f'rollout_step_{step:03d}') if debug_dir else None
        left_q, right_q = curr_qpos_raw[0:6], curr_qpos_raw[7:13]
        left_grip, right_grip = curr_qpos_raw[6], curr_qpos_raw[13]
        fk_r = fk.forward(left_q, right_q)
        lp, lq = fk_r['left']
        rp, rq = fk_r['right']
        if step == 0 and (not min_dist_trigger_use_delta):
            # Strictly align the first rollout step with start_ts slice origin.
            t_star, min_dist = 0, 0.0
        else:
            expected_idx = int(np.clip(rollout_steps_total, 0, len(left_ep) - 1))
            t_star, min_dist = _nearest_with_window(lp, lq, rp, rq, expected_idx, gripper_penalty)
        min_dist_start = float(min_dist)
        min_dist_end = None
        min_dist_delta = None
        min_dist_recover_gain = None
        min_dist_recover_threshold = None
        min_dist_recover_eval_end = None
        if (not closed_loop_action_trigger_enable) and (not min_dist_trigger_use_delta) and (min_dist >= threshold):
            min_dist_triggered = True
            min_dist_trigger_step = int(step)
            min_dist_trigger_dist = float(min_dist)
            if debug_dir is not None:
                _dbg_rollout.append({
                    'step': int(step),
                    't_star': int(t_star),
                    'min_dist': float(min_dist),
                    'min_dist_threshold': float(threshold),
                    'min_dist_triggered': True,
                    'reason': 'min_dist_threshold',
                })
            break
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
        input_noised = False
        if vla_input_noise_enable:
            if vla_qpos_noise_std > 0:
                qt = qt + torch.randn_like(qt) * vla_qpos_noise_std
                input_noised = True
            if vla_img_noise_std > 0:
                it = torch.clamp(it + torch.randn_like(it) * vla_img_noise_std, 0.0, 1.0)
                input_noised = True
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

        # Recovery is evaluated by THIS step's raw (non-perturbed) VLA action, for the
        # previous perturb step. After evaluation passes, this step still gets perturbed
        # and executed to generate the next perturb state.
        if (not closed_loop_action_trigger_enable) and min_dist_trigger_use_delta and pending_recover_valid:
            q_eval_end = np.asarray(act_pred_raw[-1], dtype=np.float32)
            fk_eval = fk.forward(q_eval_end[0:6], q_eval_end[7:13])
            lp_ev, lq_ev = fk_eval['left']
            rp_ev, rq_ev = fk_eval['right']
            _, min_dist_recover_eval_end = _nearest_with_window(
                lp_ev, lq_ev, rp_ev, rq_ev,
                int(np.clip(rollout_steps_total + int(exec_len), 0, len(left_ep) - 1)),
                gripper_penalty,
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
                    infer_kwargs=cfg.get('evac_infer_kwargs'),
                )
                _vpath_eval = os.path.join(evac_eval_debug_dir, 'outputs.mp4')
                evac_recover_eval_videos.append({
                    'step': int(step),
                    'path': _vpath_eval,
                    'exists': bool(os.path.exists(_vpath_eval)),
                })
            min_dist_recover_gain = float(min_dist_start - float(min_dist_recover_eval_end))
            if min_dist_recover_use_added:
                min_dist_recover_threshold = float(max(0.0, pending_recover_added) * max(0.0, min_dist_recover_ratio))
            else:
                min_dist_recover_threshold = float(threshold)
            if min_dist_recover_gain < float(min_dist_recover_threshold):
                min_dist_triggered = True
                min_dist_trigger_step = int(step)
                min_dist_trigger_dist = float(min_dist_recover_gain)
                if debug_dir is not None:
                    _dbg_rollout.append({
                        'step': int(step),
                        'step_role': step_role,
                        'paired_perturb_step': (None if recovery_pair_step is None else int(recovery_pair_step)),
                        't_star': int(t_star),
                        'min_dist_start': float(min_dist_start),
                        'min_dist_recover_eval_end': float(min_dist_recover_eval_end),
                        'min_dist_recover_gain': float(min_dist_recover_gain),
                        'min_dist_recover_threshold': float(min_dist_recover_threshold),
                        'min_dist_triggered': True,
                        'reason': 'min_dist_recovery_insufficient',
                    })
                break
            pending_recover_valid = False
            pending_recover_added = 0.0
            pending_recover_from_step = None

        action_dist = None
        if closed_loop_action_trigger_enable:
            gt_chunk = _build_gt_action_chunk(
                abs_start_idx=int(start_ts + int(t_star)),
                n_step=int(exec_len),
                fallback_action=curr_qpos_raw,
            )
            action_dist = _action_chunk_dist(
                act_pred_raw,
                gt_chunk,
                gripper_w=closed_loop_action_gripper_weight,
                mode=closed_loop_action_dist_mode,
                last_k=closed_loop_action_last_k,
            )
            if action_dist >= closed_loop_action_threshold:
                action_triggered = True
                action_trigger_step = int(step)
                action_trigger_dist = float(action_dist)
                if debug_dir is not None:
                    _dbg_rollout.append({
                        'step': int(step),
                        't_star': int(t_star),
                        'min_dist': float(min_dist),
                        'action_dist': float(action_dist),
                        'action_threshold': float(closed_loop_action_threshold),
                        'action_dist_mode': closed_loop_action_dist_mode,
                        'action_dist_last_k': int(closed_loop_action_last_k),
                        'action_triggered': True,
                        'reason': 'closed_loop_action_threshold',
                    })
                break
        active_info = _infer_active_arms_from_gt_window(
            raw_data.get('gt_left_arm'),
            raw_data.get('gt_right_arm'),
            raw_data.get('left_gripper'),
            raw_data.get('right_gripper'),
            t_idx=start_ts + int(t_star),
            window_len=rollout_exec_steps,
            joint_delta_thresh=float(cfg.get('perturb_active_joint_delta_thresh', 0.02)),
            gripper_delta_thresh=float(cfg.get('perturb_active_gripper_delta_thresh', 0.05)),
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
        phase_key = _infer_phase_key_from_gt_window(
            left_grip_traj=left_grip_traj,
            right_grip_traj=right_grip_traj,
            window_len=rollout_exec_steps,
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
        if (not closed_loop_action_trigger_enable) and min_dist_trigger_use_delta:
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
            min_dist_delta = float(min_dist_end - min_dist_start)
            # Use end-of-rollout nearest as latest anchor for downstream phase/targeting.
            t_star = int(t_star_e)
            min_dist = float(min_dist_end)
            # Record added error from this perturb execution; next step will evaluate recovery.
            pending_recover_valid = True
            pending_recover_added = float(max(0.0, min_dist_delta))
            pending_recover_from_step = int(step)

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
                'min_dist_start': float(min_dist_start),
                'min_dist_end': (None if min_dist_end is None else float(min_dist_end)),
                'min_dist_delta': (None if min_dist_delta is None else float(min_dist_delta)),
                'min_dist_recover_gain': (None if min_dist_recover_gain is None else float(min_dist_recover_gain)),
                'min_dist_recover_threshold': (None if min_dist_recover_threshold is None else float(min_dist_recover_threshold)),
                'min_dist_recover_eval_end': (None if min_dist_recover_eval_end is None else float(min_dist_recover_eval_end)),
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
                'input_noised': bool(input_noised),
                'vla_img_noise_std': float(vla_img_noise_std),
                'vla_qpos_noise_std': float(vla_qpos_noise_std),
                'act_chunk_raw_first': act_raw[0].tolist(),
                'act_chunk_raw_last': act_raw[-1].tolist(),
                'rollout_exec_len': int(len(act_raw)),
                'action_dist': (None if action_dist is None else float(action_dist)),
                'action_threshold': (None if not closed_loop_action_trigger_enable else float(closed_loop_action_threshold)),
                'action_dist_mode': (None if not closed_loop_action_trigger_enable else closed_loop_action_dist_mode),
                'action_dist_last_k': (None if not closed_loop_action_trigger_enable else int(closed_loop_action_last_k)),
                'action_triggered': bool(False),
            })
        if single_rollout_correction:
            break

    if closed_loop_action_trigger_enable:
        if action_triggered:
            force_generate_correction = True
        elif closed_loop_fallback_force_correction:
            force_generate_correction = True
            if debug_dir is not None:
                _dbg_rollout.append({
                    'step': int(max_steps_eff),
                    'action_triggered': False,
                    'reason': 'closed_loop_fallback_force_correction',
                    'closed_loop_max_rollouts': int(closed_loop_max_rollouts),
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
        if nearest_mode in {"gripper_close", "gripper_open"}:
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
        joint_delta_thresh=float(cfg.get('perturb_active_joint_delta_thresh', 0.02)),
        gripper_delta_thresh=float(cfg.get('perturb_active_gripper_delta_thresh', 0.05)),
    )
    corr_left_active = bool(corr_active_info['left_arm'] or corr_active_info['left_gripper'])
    corr_right_active = bool(corr_active_info['right_arm'] or corr_active_info['right_gripper'])

    if (not always_correction) and (not force_generate_correction):
        # debug: save rollout info even on skip
        if debug_dir is not None:
            import json
            _dbg_corr = os.path.join(debug_dir, 'correction')
            os.makedirs(_dbg_corr, exist_ok=True)
            with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                if closed_loop_action_trigger_enable:
                    _skip_reason = 'closed_loop_not_triggered'
                elif min_dist_trigger_use_delta:
                    _skip_reason = 'min_dist_delta_not_triggered'
                else:
                    _skip_reason = 'min_dist_not_triggered'
                json.dump({'reason': _skip_reason, 'min_dist': float(min_dist),
                           'threshold': threshold,
                           'min_dist_triggered': bool(min_dist_triggered),
                           'min_dist_trigger_step': (None if min_dist_trigger_step is None else int(min_dist_trigger_step)),
                           'min_dist_trigger_dist': (None if min_dist_trigger_dist is None else float(min_dist_trigger_dist)),
                           'action_triggered': bool(action_triggered),
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

        if correction_interp_smooth_enable and correction_interp_smooth_steps > 0 and prefix_len < chunk_size:
            def _smooth_splice(seq, split_idx, k):
                x = np.asarray(seq, dtype=np.float32).copy()
                n = x.shape[0]
                if split_idx <= 0 or split_idx >= n or k <= 0:
                    return x
                if x.ndim == 1:
                    x = x[:, None]
                    squeeze_back = True
                else:
                    squeeze_back = False
                # Stronger local smoothing around splice: low-pass in a local window.
                i_start = int(max(0, split_idx - k))
                i_end_ex = int(min(n, split_idx + k))
                if i_end_ex - i_start < 3:
                    return x[:, 0] if squeeze_back else x
                for _ in range(correction_interp_smooth_passes):
                    seg = x[i_start:i_end_ex]
                    seg_pad = np.concatenate([seg[:1], seg, seg[-1:]], axis=0)
                    seg_s = (seg_pad[:-2] + 2.0 * seg_pad[1:-1] + seg_pad[2:]) * 0.25
                    x[i_start:i_end_ex] = seg_s

                if squeeze_back:
                    return x[:, 0]
                return x

            lt = _smooth_splice(lt, prefix_len, correction_interp_smooth_steps)
            rt = _smooth_splice(rt, prefix_len, correction_interp_smooth_steps)
            lg = _smooth_splice(lg, prefix_len, correction_interp_smooth_steps)
            rg = _smooth_splice(rg, prefix_len, correction_interp_smooth_steps)

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

    # --- debug: save full correction info ---
    if debug_dir is not None:
        import json, cv2
        _dbg_corr = os.path.join(debug_dir, 'correction')
        os.makedirs(_dbg_corr, exist_ok=True)
        # save correction trajectory
        np.savetxt(os.path.join(_dbg_corr, 'corr_action_raw.csv'), corr, fmt='%.6f', delimiter=',')
        # save corrected image (curr_image is BGR, cv2 expects BGR)
        _cimg = (curr_image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(_dbg_corr, 'corrected_image.png'), _cimg)
        # save original image as GT frame at matched time (start_ts + t_star)
        # fallback to current sampled image if GT decode fails.
        _oimg = (image_data_s[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
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
        # Keep original/corrected overlays in the same resolution.
        if _oimg.shape[:2] != _cimg.shape[:2]:
            _oimg = cv2.resize(_oimg, (_cimg.shape[1], _cimg.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(_dbg_corr, 'original_image.png'), _oimg)
        # Project correction trajectory via EVAC original get_traj for consistency.
        _overlay_o = _oimg.copy()
        _overlay_c = _cimg.copy()
        traj_u8 = np.zeros_like(_oimg, dtype=np.uint8) + 50
        try:
            from evac.lvdm.models.ddpm3d import ACWMLatentDiffusion
            import evac.lvdm.models.ddpm3d as ddpm3d_mod
            K = raw_data['intrinsic_cv'].astype(np.float32).copy()
            E = np.eye(4, dtype=np.float32)
            E[:3, :] = raw_data['extrinsic_cv'].astype(np.float32)
            # Intrinsics are calibrated at native resolution. Overlay images are
            # 640x480, so scale K to the current image size before projection.
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
                lp, lq_wxyz = fr['left']
                rp, rq_wxyz = fr['right']
                # Follow evac_inference exactly: wxyz->xyzw and canonicalize w>=0.
                lq_xyzw = np.array([lq_wxyz[1], lq_wxyz[2], lq_wxyz[3], lq_wxyz[0]], dtype=np.float32)
                rq_xyzw = np.array([rq_wxyz[1], rq_wxyz[2], rq_wxyz[3], rq_wxyz[0]], dtype=np.float32)
                if lq_xyzw[3] < 0:
                    lq_xyzw = -lq_xyzw
                if rq_xyzw[3] < 0:
                    rq_xyzw = -rq_xyzw
                # Follow EVAC get_traj convention: gripper in [0,120].
                lg = float(np.clip(corr[ai, 6], 0.0, 1.0)) * 120.0
                rg = float(np.clip(corr[ai, 13], 0.0, 1.0)) * 120.0
                pose_list.append(np.concatenate([lp, lq_xyzw, [lg], rp, rq_xyzw, [rg]], axis=0).astype(np.float32))

            pose_np = np.stack(pose_list, axis=0)

            def _render_endpoint_mask(pose_arr):
                # Shorten orientation rays for endpoint markers only.
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
                    )  # (3,1,T,H,W), in [0,1]
                finally:
                    ddpm3d_mod.EndEffectorPts = _orig_eef_pts
                traj_np = traj_tensor.detach().cpu().numpy()[:, 0]   # (3,T,H,W)
                traj_np = np.transpose(traj_np, (1, 2, 3, 0))        # (T,H,W,3)
                t_len = traj_np.shape[0]
                idx_start, idx_end = 0, max(0, t_len - 1)
                traj_endpoints = np.maximum(traj_np[idx_start], traj_np[idx_end])  # (H,W,3)
                traj_u8 = np.clip(traj_endpoints * 255.0, 0.0, 255.0).astype(np.uint8)
                mask = np.any(np.abs(traj_u8.astype(np.int16) - 50) > 2, axis=2)
                return traj_u8, mask, idx_end

            traj_u8, mask, idx_end = _render_endpoint_mask(pose_np)

            # Keep UV projection utility for endpoint labels and original-vs-corrected comparison.
            w2c_t = torch.from_numpy(E).float().unsqueeze(0).unsqueeze(0)         # (1,1,4,4)
            intrinsic_t = torch.from_numpy(K).float().unsqueeze(0).unsqueeze(0)   # (1,1,3,3)
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
                uvs_l = (uvs_l / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)[0].cpu().numpy()  # (T,4,2)
                uvs_r = (uvs_r / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)[0].cpu().numpy()  # (T,4,2)
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

            uvs_l, uvs_r, pts_l, pts_r = _project_base_uv_from_pose_np(pose_np)
            # reshape pts to [1,T,4,4] for consistent indexing above
            pts_l_bt = pts_l.reshape(1, pts_l.shape[1], 4, 4)
            pts_r_bt = pts_r.reshape(1, pts_r.shape[1], 4, 4)
            luv = _extract_base_uv(uvs_l, pts_l_bt)
            ruv = _extract_base_uv(uvs_r, pts_r_bt)

            C_LEFT = (0, 255, 0)
            C_RIGHT = (0, 0, 255)
            C_LEFT_GT = (0, 255, 255)
            C_RIGHT_GT = (255, 255, 0)

            # Middle trajectory: draw only base-point polyline (no orientation rays, no alpha blending).
            for seq, color in ((luv, C_LEFT), (ruv, C_RIGHT)):
                prev = None
                for k, pt in enumerate(seq):
                    if k == 0 or k == idx_end:
                        prev = pt
                        continue
                    if pt is None:
                        prev = None
                        continue
                    if prev is not None:
                        cv2.line(_overlay_c, (int(prev[0]), int(prev[1])), (int(pt[0]), int(pt[1])), color, 2, cv2.LINE_AA)
                    prev = pt

            def _mark_start_end(img, seq, color, prefix):
                if len(seq) == 0:
                    return
                s = seq[0]
                e = seq[-1]
                if s is not None:
                    cv2.circle(img, (int(s[0]), int(s[1])), 4, color, -1, cv2.LINE_AA)
                    cv2.putText(img, f"{prefix}-S", (int(s[0]) + 6, int(s[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                if e is not None:
                    cv2.circle(img, (int(e[0]), int(e[1])), 4, color, -1, cv2.LINE_AA)
                    cv2.putText(img, f"{prefix}-E", (int(e[0]) + 6, int(e[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            _overlay_c[mask] = (0.6 * _overlay_c[mask] + 0.4 * traj_u8[mask]).astype(np.uint8)
            _mark_start_end(_overlay_c, luv, C_LEFT, "L")
            _mark_start_end(_overlay_c, ruv, C_RIGHT, "R")

            # On original image, show corrected-start vs GT-reference using the same get_traj circle rendering.
            n_total = int(raw_data['left_endpose'].shape[0])
            gt_end = int(min(n_total, gt_ref_idx + pose_np.shape[0]))
            if gt_end > gt_ref_idx:
                gt_pose_list = []
                for gi in range(gt_ref_idx, gt_end):
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
                    gt_pose_list.append(np.concatenate([lp_gt, lq_gt_xyzw, [lg_gt], rp_gt, rq_gt_xyzw, [rg_gt]], axis=0).astype(np.float32))
                gt_pose_np = np.stack(gt_pose_list, axis=0)
                gt_uvs_l, gt_uvs_r, gt_pts_l, gt_pts_r = _project_base_uv_from_pose_np(gt_pose_np)
                gt_l_seq = _extract_base_uv(gt_uvs_l, gt_pts_l.reshape(1, gt_pts_l.shape[1], 4, 4))
                gt_r_seq = _extract_base_uv(gt_uvs_r, gt_pts_r.reshape(1, gt_pts_r.shape[1], 4, 4))
                corr_start_map, corr_start_mask, _ = _render_endpoint_mask(pose_np[:1])
                gt_start_map, gt_start_mask, _ = _render_endpoint_mask(gt_pose_np[:1])
                _overlay_o[corr_start_mask] = (0.55 * _overlay_o[corr_start_mask] + 0.45 * corr_start_map[corr_start_mask]).astype(np.uint8)
                _overlay_o[gt_start_mask] = (0.55 * _overlay_o[gt_start_mask] + 0.45 * gt_start_map[gt_start_mask]).astype(np.uint8)

                def _label_point(img, pt, text, color):
                    if pt is None:
                        return
                    x, y = int(pt[0]), int(pt[1])
                    cv2.putText(img, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                _label_point(_overlay_o, luv[0] if len(luv) > 0 else None, "L-C", C_LEFT)
                _label_point(_overlay_o, ruv[0] if len(ruv) > 0 else None, "R-C", C_RIGHT)
                _label_point(_overlay_o, gt_l_seq[0] if len(gt_l_seq) > 0 else None, "L-GT", C_LEFT_GT)
                _label_point(_overlay_o, gt_r_seq[0] if len(gt_r_seq) > 0 else None, "R-GT", C_RIGHT_GT)

            def _draw_legend(img, rows):
                if len(rows) == 0:
                    return
                x0, y0 = 10, 10
                row_h = 20
                w = 170
                h = 10 + row_h * len(rows) + 8
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1, cv2.LINE_AA)
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (220, 220, 220), 1, cv2.LINE_AA)
                for i, (name, color) in enumerate(rows):
                    y = y0 + 18 + i * row_h
                    cv2.circle(img, (x0 + 12, y - 4), 4, color, -1, cv2.LINE_AA)
                    cv2.putText(img, name, (x0 + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

            _draw_legend(_overlay_c, [
                ("L start/end", C_LEFT),
                ("R start/end", C_RIGHT),
            ])
            _draw_legend(_overlay_o, [
                ("L-C corrected", C_LEFT),
                ("R-C corrected", C_RIGHT),
                ("L-GT reference", C_LEFT_GT),
                ("R-GT reference", C_RIGHT_GT),
            ])
            with open(os.path.join(_dbg_corr, 'corr_projection_debug.json'), 'w') as _f:
                json.dump({
                    'projection_chain': 'evac_inference_compatible_link6',
                    'mask_pixels': int(np.count_nonzero(mask)),
                    'gt_ref_index_for_original': int(gt_ref_idx),
                    'image_hw': [int(h_img), int(w_img)],
                    'native_hw': [int(h_native), int(w_native)],
                }, _f, indent=2)
        except Exception:
            import traceback
            with open(os.path.join(_dbg_corr, 'corr_projection_error.txt'), 'w') as _f:
                _f.write(traceback.format_exc())
        cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_traj_raw.png'), traj_u8)
        cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_original.png'), _overlay_o)
        cv2.imwrite(os.path.join(_dbg_corr, 'corr_projection_on_corrected.png'), _overlay_c)
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
                'min_dist_recover_gain': (None if _r.get('min_dist_recover_gain') is None else float(_r.get('min_dist_recover_gain'))),
                'min_dist_recover_threshold': (None if _r.get('min_dist_recover_threshold') is None else float(_r.get('min_dist_recover_threshold'))),
                'min_dist_recover_eval_end': (None if _r.get('min_dist_recover_eval_end') is None else float(_r.get('min_dist_recover_eval_end'))),
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
                'action_dist': (None if _r.get('action_dist') is None else float(_r.get('action_dist'))),
                'action_triggered': bool(_r.get('action_triggered', False)),
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
        if bool(action_triggered):
            _trigger_source = 'action_dist'
            _trigger_step = (None if action_trigger_step is None else int(action_trigger_step))
            _trigger_value = (None if action_trigger_dist is None else float(action_trigger_dist))
        elif bool(min_dist_triggered):
            _trigger_source = 'min_dist'
            _trigger_step = (None if min_dist_trigger_step is None else int(min_dist_trigger_step))
            _trigger_value = (None if min_dist_trigger_dist is None else float(min_dist_trigger_dist))
        elif bool(force_generate_correction):
            _trigger_source = 'fallback'

        _plan_info = {
            'version': 2,
            'trigger_mode': (
                'action_dist'
                if closed_loop_action_trigger_enable else
                ('min_dist_recovery' if min_dist_trigger_use_delta else 'min_dist')
            ),
            'trigger_threshold': {
                'min_dist': float(threshold),
                'action_dist': (float(closed_loop_action_threshold) if closed_loop_action_trigger_enable else None),
                'min_dist_recover_ratio': (float(min_dist_recover_ratio) if min_dist_trigger_use_delta else None),
                'min_dist_recover_use_added': (bool(min_dist_recover_use_added) if min_dist_trigger_use_delta else None),
            },
            'fallback': {
                'closed_loop_action': bool(closed_loop_fallback_force_correction),
                'min_dist': bool(min_dist_fallback_force_correction),
                'used': bool(force_generate_correction and (not action_triggered) and (not min_dist_triggered)),
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
            'rollouts': _rollout_records,
            'final_match': {
                't_star': int(t_star),
                'min_dist': float(min_dist),
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
                'interp_smooth_enable': bool(correction_interp_smooth_enable),
                'interp_smooth_steps': int(correction_interp_smooth_steps),
                'interp_smooth_passes': int(correction_interp_smooth_passes),
                'correction_prefix_len': int(prefix_len),
            },
        }
        with open(os.path.join(_dbg_corr, 'correction_info.json'), 'w') as _f:
            json.dump(_plan_info, _f, indent=2)
        # save correction trajectory plot
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 6))
            labels_l = ['lj1','lj2','lj3','lj4','lj5','lj6','lg']
            labels_r = ['rj1','rj2','rj3','rj4','rj5','rj6','rg']
            for j in range(7):
                axes[0].plot(corr[:, j], label=labels_l[j])
                axes[1].plot(corr[:, 7+j], label=labels_r[j])
            axes[0].set_title('Left arm correction trajectory')
            axes[0].legend(fontsize=7, ncol=4)
            axes[1].set_title('Right arm correction trajectory')
            axes[1].legend(fontsize=7, ncol=4)
            plt.tight_layout()
            plt.savefig(os.path.join(_dbg_corr, 'corr_trajectory.png'), dpi=100)
            plt.close(fig)
        except Exception:
            pass

    action_dist_list = []
    for _r in _dbg_rollout:
        _ad = _r.get('action_dist', None) if isinstance(_r, dict) else None
        if _ad is not None:
            try:
                action_dist_list.append(float(_ad))
            except Exception:
                pass
    action_dist_step0 = None
    for _r in _dbg_rollout:
        if isinstance(_r, dict) and int(_r.get('step', -1)) == 0 and (_r.get('action_dist', None) is not None):
            action_dist_step0 = float(_r.get('action_dist'))
            break

    corr_meta = {
        "action_triggered": bool(action_triggered),
        "action_trigger_step": (None if action_trigger_step is None else int(action_trigger_step)),
        "action_trigger_dist": (None if action_trigger_dist is None else float(action_trigger_dist)),
        "closed_loop_fallback_used": bool(force_generate_correction and (not action_triggered)),
        "action_dist_step0": action_dist_step0,
        "action_dist_last": (None if len(action_dist_list) == 0 else float(action_dist_list[-1])),
        "action_dist_mean": (None if len(action_dist_list) == 0 else float(np.mean(action_dist_list))),
        "action_dist_p90": (None if len(action_dist_list) == 0 else float(np.percentile(action_dist_list, 90))),
    }

    return (
        curr_image.to(device),
        torch.from_numpy(qn.astype(np.float32)).to(device),
        torch.from_numpy(padded).float().to(device),
        torch.from_numpy(is_pad).bool().to(device),
        corr_meta,
    )


# def eval_bc(config, ckpt_name, save_episode=True):
#     set_seed(1000)
#     ckpt_dir = config["ckpt_dir"]
#     state_dim = config["state_dim"]
#     real_robot = config["real_robot"]
#     policy_class = config["policy_class"]
#     onscreen_render = config["onscreen_render"]
#     policy_config = config["policy_config"]
#     camera_names = config["camera_names"]
#     max_timesteps = config["episode_len"]
#     task_name = config["task_name"]
#     temporal_agg = config["temporal_agg"]
#     onscreen_cam = "angle"
#
#     # load policy and stats
#     ckpt_path = os.path.join(ckpt_dir, ckpt_name)
#     policy = make_policy(policy_class, policy_config)
#     loading_status = policy.load_state_dict(torch.load(ckpt_path))
#     print(loading_status)
#     policy.cuda()
#     policy.eval()
#     print(f"Loaded: {ckpt_path}")
#     stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
#     with open(stats_path, "rb") as f:
#         stats = pickle.load(f)
#
#     pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
#     post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
#
#     # load environment
#     if real_robot:
#         from aloha_scripts.robot_utils import move_grippers  # requires aloha
#         from aloha_scripts.real_env import make_real_env  # requires aloha
#
#         env = make_real_env(init_node=True)
#         env_max_reward = 0
#     else:
#         from sim_env import make_sim_env
#
#         env = make_sim_env(task_name)
#         env_max_reward = env.task.max_reward
#
#     query_frequency = policy_config["num_queries"]
#     if temporal_agg:
#         query_frequency = 1
#         num_queries = policy_config["num_queries"]
#
#     max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
#
#     num_rollouts = 50
#     episode_returns = []
#     highest_rewards = []
#     for rollout_id in range(num_rollouts):
#         rollout_id += 0
#         ### set task
#         if "sim_transfer_cube" in task_name:
#             BOX_POSE[0] = sample_box_pose()  # used in sim reset
#         elif "sim_insertion" in task_name:
#             BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
#
#         ts = env.reset()
#
#         ### onscreen render
#         if onscreen_render:
#             ax = plt.subplot()
#             plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
#             plt.ion()
#
#         ### evaluation loop
#         if temporal_agg:
#             all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()
#
#         qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
#         image_list = []  # for visualization
#         qpos_list = []
#         target_qpos_list = []
#         rewards = []
#         with torch.inference_mode():
#             for t in range(max_timesteps):
#                 ### update onscreen render and wait for DT
#                 if onscreen_render:
#                     image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
#                     plt_img.set_data(image)
#                     plt.pause(DT)
#
#                 ### process previous timestep to get qpos and image_list
#                 obs = ts.observation
#                 if "images" in obs:
#                     image_list.append(obs["images"])
#                 else:
#                     image_list.append({"main": obs["image"]})
#                 qpos_numpy = np.array(obs["qpos"])
#                 qpos = pre_process(qpos_numpy)
#                 qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
#                 qpos_history[:, t] = qpos
#                 curr_image = get_image(ts, camera_names)
#
#                 ### query policy
#                 if config["policy_class"] == "ACT":
#                     if t % query_frequency == 0:
#                         all_actions = policy(qpos, curr_image)
#                     if temporal_agg:
#                         all_time_actions[[t], t:t + num_queries] = all_actions
#                         actions_for_curr_step = all_time_actions[:, t]
#                         actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
#                         actions_for_curr_step = actions_for_curr_step[actions_populated]
#                         k = 0.01
#                         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
#                         exp_weights = exp_weights / exp_weights.sum()
#                         exp_weights = (torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1))
#                         raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
#                     else:
#                         raw_action = all_actions[:, t % query_frequency]
#                 elif config["policy_class"] == "CNNMLP":
#                     raw_action = policy(qpos, curr_image)
#                 else:
#                     raise NotImplementedError
#
#                 ### post-process actions
#                 raw_action = raw_action.squeeze(0).cpu().numpy()
#                 action = post_process(raw_action)
#                 target_qpos = action
#
#                 ### step the environment
#                 ts = env.step(target_qpos)
#
#                 ### for visualization
#                 qpos_list.append(qpos_numpy)
#                 target_qpos_list.append(target_qpos)
#                 rewards.append(ts.reward)
#
#             plt.close()
#         if real_robot:
#             move_grippers(
#                 [env.puppet_bot_left, env.puppet_bot_right],
#                 [PUPPET_GRIPPER_JOINT_OPEN] * 2,
#                 move_time=0.5,
#             )  # open
#             pass
#
#         rewards = np.array(rewards)
#         episode_return = np.sum(rewards[rewards != None])
#         episode_returns.append(episode_return)
#         episode_highest_reward = np.max(rewards)
#         highest_rewards.append(episode_highest_reward)
#         print(
#             f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
#         )
#
#         if save_episode:
#             save_videos(
#                 image_list,
#                 DT,
#                 video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
#             )
#
#     success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
#     avg_return = np.mean(episode_returns)
#     summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
#     for r in range(env_max_reward + 1):
#         more_or_equal_r = (np.array(highest_rewards) >= r).sum()
#         more_or_equal_r_rate = more_or_equal_r / num_rollouts
#         summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"
#
#     # save success rate to txt
#     result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
#     with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
#         f.write(summary_str)
#         f.write(repr(episode_returns))
#         f.write("\n\n")
#         f.write(repr(highest_rewards))
#
#     return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data[0], data[1], data[2], data[3]
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    _r = int(os.environ.get("RANK", -1))
    _lr = int(os.environ.get("LOCAL_RANK", -1))
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    # Accelerate: prepare model, optimizer, dataloader
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Set seed after Accelerator init so each process gets proper seed offset
    set_seed(seed)

    print(f"[train_bc][rank={_r} local_rank={_lr}] before make_policy")
    _t_make_policy = time.time()
    policy = make_policy(policy_class, policy_config)
    print(f"[train_bc][rank={_r} local_rank={_lr}] after make_policy | elapsed={time.time() - _t_make_policy:.2f}s")

    enable_wm = config['enable_wm_correction']
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

    correction_modules = None
    debug_wm = config.get('debug_wm_correction', False)
    debug_wm_dir = os.path.join(ckpt_dir, 'debug_wm') if debug_wm else None
    if debug_wm_dir and accelerator.is_main_process:
        os.makedirs(debug_wm_dir, exist_ok=True)
    # correction statistics tracker
    _corr_stats = {'n_triggered': 0, 'n_success': 0, 'n_skipped': 0,
                   'n_plan_fail': 0, 'n_error': 0, 'dists': [], 'steps_used': [],
                   'n_action_triggered': 0, 'n_action_fallback': 0, 'action_trigger_steps': [],
                   'action_dist_step0': [], 'action_dist_last': [], 'action_dist_mean': [], 'action_dist_p90': []}
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
    validation_history = []
    # min_val_loss = np.inf
    # best_ckpt_info = None

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
        # # validation
        # with torch.inference_mode():
        #     policy.eval()
        #     epoch_dicts = []
        #     for batch_idx, data in enumerate(val_dataloader):
        #         forward_dict = forward_pass(data, policy)
        #         epoch_dicts.append(forward_dict)
        #     epoch_summary = compute_dict_mean(epoch_dicts)
        #     validation_history.append(epoch_summary)
        #
        #     epoch_val_loss = epoch_summary["loss"]
        #     if epoch_val_loss < min_val_loss:
        #         min_val_loss = epoch_val_loss
        #         best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        # print(f"Val loss:   {epoch_val_loss:.5f}")
        # summary_string = ""
        # for k, v in epoch_summary.items():
        #     summary_string += f"{k}: {v.item():.3f} "

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            apply_wm = enable_wm and correction_modules is not None and global_step % correction_cfg['correction_freq'] == 0
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
                                if bool(cmeta.get('action_triggered', False)):
                                    _corr_stats['n_action_triggered'] += 1
                                    _st = cmeta.get('action_trigger_step', None)
                                    if _st is not None:
                                        _corr_stats['action_trigger_steps'].append(int(_st))
                                if bool(cmeta.get('closed_loop_fallback_used', False)):
                                    _corr_stats['n_action_fallback'] += 1
                                for _k in ('action_dist_step0', 'action_dist_last', 'action_dist_mean', 'action_dist_p90'):
                                    _v = cmeta.get(_k, None)
                                    if _v is not None:
                                        _corr_stats[_k].append(float(_v))
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

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            if accelerator.is_main_process and writer is not None:
                writer.add_scalar('train/loss_step', forward_dict['loss'].item(), global_step)
            global_step += 1
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary["loss"]
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
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
                writer.add_scalar('correction/action_trigger_count', _corr_stats['n_action_triggered'], epoch)
                writer.add_scalar('correction/action_fallback_count', _corr_stats['n_action_fallback'], epoch)
                _total = _corr_stats['n_triggered'] or 1
                writer.add_scalar('correction/success_rate', _corr_stats['n_success'] / _total, epoch)
                _succ = _corr_stats['n_success'] or 1
                writer.add_scalar('correction/action_trigger_rate', _corr_stats['n_action_triggered'] / _succ, epoch)
                writer.add_scalar('correction/action_fallback_rate', _corr_stats['n_action_fallback'] / _succ, epoch)
                if len(_corr_stats['action_trigger_steps']) > 0:
                    writer.add_scalar('correction/action_trigger_step_mean',
                                      float(np.mean(_corr_stats['action_trigger_steps'])), epoch)
                for _k in ('action_dist_step0', 'action_dist_last', 'action_dist_mean', 'action_dist_p90'):
                    if len(_corr_stats[_k]) > 0:
                        writer.add_scalar(f'correction/{_k}_mean', float(np.mean(_corr_stats[_k])), epoch)
                        writer.add_scalar(f'correction/{_k}_p50', float(np.percentile(_corr_stats[_k], 50)), epoch)
                        writer.add_scalar(f'correction/{_k}_p90', float(np.percentile(_corr_stats[_k], 90)), epoch)
                print(f'  [Correction] triggered={_corr_stats["n_triggered"]} '
                      f'success={_corr_stats["n_success"]} '
                      f'skipped={_corr_stats["n_skipped"]} '
                      f'error={_corr_stats["n_error"]} '
                      f'action_triggered={_corr_stats["n_action_triggered"]} '
                      f'action_fallback={_corr_stats["n_action_fallback"]} '
                      f'rate={_corr_stats["n_success"]/_total:.2%}')
                if len(_corr_stats['action_dist_step0']) > 0:
                    print(f'  [ActionDist] step0 p50={np.percentile(_corr_stats["action_dist_step0"], 50):.4f} '
                          f'p90={np.percentile(_corr_stats["action_dist_step0"], 90):.4f} '
                          f'mean={np.mean(_corr_stats["action_dist_step0"]):.4f}')
                # reset per-epoch
                _corr_stats = {'n_triggered': 0, 'n_success': 0, 'n_skipped': 0,
                               'n_plan_fail': 0, 'n_error': 0, 'dists': [], 'steps_used': [],
                               'n_action_triggered': 0, 'n_action_fallback': 0, 'action_trigger_steps': [],
                               'action_dist_step0': [], 'action_dist_last': [], 'action_dist_mean': [], 'action_dist_p90': []}

        if (epoch + 1) % config['save_freq'] == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            unwrapped_policy = accelerator.unwrap_model(policy)
            torch.save(unwrapped_policy.state_dict(), ckpt_path)
            # plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    if accelerator.is_main_process:
        if writer is not None:
            writer.close()

        ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
        unwrapped_policy = accelerator.unwrap_model(policy)
        torch.save(unwrapped_policy.state_dict(), ckpt_path)

    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    # torch.save(best_state_dict, ckpt_path)
    # print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")
    print(f"Training finished: Seed {seed}")

    # # save training curves
    # plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)


# def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
#     # save training curves
#     for key in train_history[0]:
#         plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
#         plt.figure()
#         train_values = [summary[key].item() for summary in train_history]
#         val_values = [summary[key].item() for summary in validation_history]
#         plt.plot(
#             np.linspace(0, num_epochs - 1, len(train_history)),
#             train_values,
#             label="train",
#         )
#         plt.plot(
#             np.linspace(0, num_epochs - 1, len(validation_history)),
#             val_values,
#             label="validation",
#         )
#         # plt.ylim([-0.1, 1])
#         plt.tight_layout()
#         plt.legend()
#         plt.title(key)
#         plt.savefig(plot_path)
#     print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--eval", action="store_true")
    # parser.add_argument("--onscreen_render", action="store_true")
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

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--sample_skip_head", action="store", type=int, default=0,
                        help="Skip first N timesteps when sampling start_ts in dataloader")
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
    parser.add_argument("--enable_perturb", type=str2bool, default=False,
                        help="Enable online rollout action perturbation in WM correction")
    parser.add_argument("--perturb_prob", type=float, default=1.0,
                        help="Probability to apply online perturbation per rollout chunk")
    parser.add_argument("--perturb_error_mode", type=str, default="legacy",
                        help="Error mode: legacy|auto|translation|rotation|no_ops|gripper_close|gripper_open")
    parser.add_argument("--perturb_space", type=str, default="joint",
                        help="Perturbation space: joint or eef")
    parser.add_argument("--perturb_lp_alpha", type=float, default=0.35,
                        help="Low-pass blending factor for actuator response")
    parser.add_argument("--perturb_noise_std", type=float, default=0.01,
                        help="Std of colored noise injected in action space (rad)")
    parser.add_argument("--perturb_noise_rho", type=float, default=0.85,
                        help="AR(1) coefficient for colored noise")
    parser.add_argument("--perturb_vel_limit", type=float, default=0.08,
                        help="Per-step velocity limit on perturbed arm action (rad)")
    parser.add_argument("--perturb_acc_limit", type=float, default=0.04,
                        help="Per-step acceleration limit on perturbed arm action (rad)")
    parser.add_argument("--perturb_bias_prob", type=float, default=0.2,
                        help="Probability to refresh a chunk-level bias term")
    parser.add_argument("--perturb_bias_std", type=float, default=0.03,
                        help="Std of chunk-level bias term (rad)")
    parser.add_argument("--perturb_joint_limit_abs", type=float, default=3.14,
                        help="Absolute joint bound used by perturbation model (rad)")
    parser.add_argument("--perturb_mode", type=str, default="dyn",
                        help="Perturbation mode: dyn or directional_fail")
    parser.add_argument("--perturb_fail_gain", type=float, default=0.12,
                        help="Gain for directional_fail mode (rad)")
    parser.add_argument("--perturb_fail_direction", type=str, default="",
                        help="12-dim comma-separated unit direction in arm-joint space for directional_fail")
    parser.add_argument("--perturb_eef_mode", type=str, default="gaussian",
                        help="EEF perturbation mode: gaussian or directional_fail")
    parser.add_argument("--perturb_use_target_pose_planner", type=str2bool, default=True,
                        help="Use two-point target-pose planner for translation/rotation perturbation")
    parser.add_argument("--perturb_eef_pos_std", type=float, default=0.01,
                        help="Std of EEF position perturbation in meters")
    parser.add_argument("--perturb_eef_fail_gain", type=float, default=0.03,
                        help="Directional EEF perturbation magnitude in meters")
    parser.add_argument("--perturb_eef_tcp_offset_x", type=float, default=0.085,
                        help="Approximate link6->TCP offset along local X in meters")
    parser.add_argument("--perturb_eef_ramp", type=str2bool, default=True,
                        help="Enable time-ramped EEF perturbation over action chunk")
    parser.add_argument("--perturb_eef_ramp_min", type=float, default=0.0,
                        help="Minimum ramp scale at first action step")
    parser.add_argument("--perturb_eef_ramp_power", type=float, default=1.0,
                        help="Ramp curve power; >1 slower start, <1 faster start")
    parser.add_argument("--perturb_eef_ramp_apply_eps", type=float, default=1e-4,
                        help="Skip EEF IK perturbation when ramp scale is below this threshold")
    parser.add_argument("--perturb_eef_joint_delta_cap", type=float, default=0.12,
                        help="Per-joint max delta (rad) at ramp=1.0 for EEF IK perturbation")
    parser.add_argument("--perturb_translation_random_dir", type=str2bool, default=True,
                        help="Use random direction for translation failure per rollout")
    parser.add_argument("--perturb_rot_max_deg", type=float, default=15.0,
                        help="Max EEF rotation perturbation angle in degrees")
    parser.add_argument("--perturb_mag_random", type=str2bool, default=False,
                        help="Randomize perturbation magnitude per rollout chunk")
    parser.add_argument("--perturb_mag_rand_min", type=float, default=0.8,
                        help="Minimum random magnitude scale")
    parser.add_argument("--perturb_mag_rand_max", type=float, default=1.2,
                        help="Maximum random magnitude scale")
    parser.add_argument("--perturb_rotation_random_axis", type=str2bool, default=True,
                        help="Use random rotation axis for rotation failure per rollout")
    parser.add_argument("--perturb_anti_gt_cos_thresh", type=float, default=0.6,
                        help="Flip perturb dir only when cosine with GT trend exceeds this threshold (larger is looser)")
    parser.add_argument("--perturb_reject_sampling_enable", type=str2bool, default=True,
                        help="Enable rejection sampling for translation/rotation perturbation")
    parser.add_argument("--perturb_reject_max_trials", type=int, default=4,
                        help="Maximum rejection-sampling trials per rollout chunk")
    parser.add_argument("--perturb_reject_min_delta", type=float, default=5e-4,
                        help="Required minimum increase of terminal GT error to accept perturbation")
    parser.add_argument("--perturb_reject_min_score", type=float, default=0.15,
                        help="Minimum pose-level anti-GT score to accept a sampled direction before planner")
    parser.add_argument("--perturb_reject_prefilter_pool", type=int, default=8,
                        help="Number of cheap direction candidates before planner calls in reject sampling")
    parser.add_argument("--perturb_reject_orient_weight", type=float, default=0.01,
                        help="Orientation weight used by reject-sampling acceptance metric")
    parser.add_argument("--perturb_reject_gripper_penalty", type=float, default=1.0,
                        help="Gripper mismatch penalty used by reject-sampling acceptance metric")
    parser.add_argument("--nearest_window_radius", type=int, default=32,
                        help="Nearest-point matching window radius around expected progress index")
    parser.add_argument("--perturb_rot_axis_left", type=str, default="0,0,1",
                        help="3-dim axis for left-arm rotation failure")
    parser.add_argument("--perturb_rot_axis_right", type=str, default="0,0,1",
                        help="3-dim axis for right-arm rotation failure")
    parser.add_argument("--perturb_noop_lag_steps", type=int, default=3,
                        help="Lag steps used by no_ops failure")
    parser.add_argument("--perturb_noop_alpha", type=float, default=0.85,
                        help="Smoothing alpha for no_ops failure")
    parser.add_argument("--perturb_noop_transition_steps", type=int, default=4,
                        help="Transition steps after true no-op hold")
    parser.add_argument("--perturb_noop_beta", type=float, default=0.1,
                        help="Low-speed follow factor for no_ops failure")
    parser.add_argument("--perturb_noop_beta_ramp", type=str2bool, default=True,
                        help="Linearly ramp no_ops beta over chunk")
    parser.add_argument("--perturb_noop_beta_end", type=float, default=0.3,
                        help="End beta when perturb_noop_beta_ramp is enabled")
    parser.add_argument("--perturb_gripper_delay_steps", type=int, default=3,
                        help="Delay steps for gripper close/open failure")
    parser.add_argument("--perturb_gripper_transition_steps", type=int, default=4,
                        help="Transition steps for gradual gripper close/open failure")
    parser.add_argument("--perturb_gripper_close_min", type=float, default=0.35,
                        help="Minimum gripper value in close-failure mode (cannot fully close)")
    parser.add_argument("--perturb_gripper_open_max", type=float, default=0.75,
                        help="Maximum gripper value in open-failure mode (cannot fully open)")
    parser.add_argument("--perturb_gripper_fast_ratio", type=float, default=0.2,
                        help="Fraction of chunk used to complete gripper transition in target-pose perturbation")
    parser.add_argument("--perturb_active_joint_delta_thresh", type=float, default=0.02,
                        help="GT joint delta threshold (rad) to mark an arm as active")
    parser.add_argument("--perturb_active_gripper_delta_thresh", type=float, default=0.05,
                        help="GT gripper delta threshold to mark gripper side active")
    parser.add_argument("--perturb_eef_fail_dir_left", type=str, default="",
                        help="3-dim comma-separated direction for left-arm EEF directional_fail")
    parser.add_argument("--perturb_eef_fail_dir_right", type=str, default="",
                        help="3-dim comma-separated direction for right-arm EEF directional_fail")
    parser.add_argument("--vla_input_noise_enable", type=str2bool, default=False,
                        help="Enable Gaussian noise on VLA rollout inputs (qpos/image)")
    parser.add_argument("--vla_img_noise_std", type=float, default=0.0,
                        help="Image noise std for VLA rollout input in [0,1] space")
    parser.add_argument("--vla_qpos_noise_std", type=float, default=0.0,
                        help="Qpos noise std for VLA rollout input in normalized space")

    # World-model correction arguments
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
    parser.add_argument("--correction_threshold", type=float,
                        help="Distance threshold to trigger correction")
    parser.add_argument("--max_rollout_steps", type=int,
                        help="Max rollout steps for correction")
    parser.add_argument("--single_rollout_correction", type=str2bool, default=False,
                        help="If true, execute at most one rollout step before correction planning")
    parser.add_argument("--target_mode", type=str, default="forward",
                        help="Correction target mode: forward or backward")
    parser.add_argument("--target_lookahead_steps", type=int, default=0,
                        help="Correction target lookahead steps from nearest t_star on expert trajectory")
    parser.add_argument("--closed_loop_action_trigger_enable", type=str2bool, default=False,
                        help="Enable closed-loop trigger by ACT action-vs-GT deviation on EVAC-updated images")
    parser.add_argument("--closed_loop_action_threshold", type=float, default=0.2,
                        help="Trigger correction when action chunk distance to GT exceeds this threshold")
    parser.add_argument("--closed_loop_action_gripper_weight", type=float, default=1.0,
                        help="Gripper term weight in closed-loop action distance")
    parser.add_argument("--closed_loop_action_dist_mode", type=str, default="mean_chunk",
                        help="Action distance mode: mean_chunk | last | last_k")
    parser.add_argument("--closed_loop_action_last_k", type=int, default=4,
                        help="Use last k steps when closed_loop_action_dist_mode=last_k")
    parser.add_argument("--closed_loop_max_rollouts", type=int, default=3,
                        help="Maximum closed-loop rollout attempts before fallback")
    parser.add_argument("--closed_loop_fallback_force_correction", type=str2bool, default=True,
                        help="If true, force correction generation when closed-loop attempts do not exceed threshold")
    parser.add_argument("--min_dist_fallback_force_correction", type=str2bool, default=False,
                        help="If true, force correction generation when min_dist mode does not reach threshold")
    parser.add_argument("--min_dist_trigger_use_delta", type=str2bool, default=False,
                        help="If true in min_dist mode, use perturb/recovery paired-step trigger based on recovery gain")
    parser.add_argument("--min_dist_recover_use_added", type=str2bool, default=True,
                        help="If true, in delta mode use dynamic recovery threshold = ratio * max(0, end-start)")
    parser.add_argument("--min_dist_recover_ratio", type=float, default=0.5,
                        help="Dynamic recovery ratio used when min_dist_recover_use_added=true")
    parser.add_argument("--debug_recover_eval_rollout", type=str2bool, default=False,
                        help="If true, also run EVAC rollout video for unperturbed recovery-eval action in each paired step")
    parser.add_argument("--correction_interp_nearest_enable", type=str2bool, default=False,
                        help="If true, set correction target to nearest point t_star and bypass planner with interpolation+GT tail")
    parser.add_argument("--correction_interp_prefix_ratio", type=float, default=0.4,
                        help="Prefix ratio for interpolation when correction_interp_nearest_enable=true (e.g., 0.4 means 2/5 chunk)")
    parser.add_argument("--correction_interp_smooth_enable", type=str2bool, default=True,
                        help="Enable local smoothing near interpolation/GT splice when correction_interp_nearest_enable=true")
    parser.add_argument("--correction_interp_smooth_steps", type=int, default=3,
                        help="Number of steps after splice to smooth when correction_interp_nearest_enable=true")
    parser.add_argument("--correction_interp_smooth_passes", type=int, default=2,
                        help="Number of local low-pass passes near splice when correction_interp_nearest_enable=true")
    parser.add_argument("--rollout_exec_steps", type=int, default=None,
                        help="Number of actions executed per rollout step (prefix of chunk)")
    parser.add_argument("--correction_freq", type=int,
                        help="Apply correction every N steps")
    parser.add_argument("--correction_weight", type=float,
                        help="Weight for correction loss (auto-divided by batch_size)")
    parser.add_argument("--orient_weight", type=float,
                        help="Weight for orientation distance in nearest-point matching (0=position only)")
    parser.add_argument("--gripper_penalty", type=float,
                        help="Penalty for gripper state mismatch in nearest-point matching (0=ignore gripper)")
    parser.add_argument("--always_correction", type=str2bool, default=False,
                        help="Always generate correction target even when below threshold")
    parser.add_argument("--evac_budget_accel", type=str2bool, default=False,
                        help="Enable dual-cache budget acceleration for EVAC inference")
    parser.add_argument("--evac_rank_transfer", type=str2bool, default=False,
                        help="Enable cross-chunk rank-transfer acceleration for EVAC inference")
    parser.add_argument("--evac_ddim_eta", type=parse_optional_float, default=None,
                        help="DDIM eta for EVAC inference. Default: auto (rank-transfer: 0.0, otherwise: 1.0)")
    parser.add_argument("--evac_dc_budget", type=float, default=0.5,
                        help="Dual-cache budget when evac_budget_accel=true")
    parser.add_argument("--evac_rt_full_chunks", type=int, default=3,
                        help="Full chunks for rank-transfer when evac_rank_transfer=true")
    parser.add_argument("--evac_rt_per_channel", type=str2bool, default=True,
                        help="Per-channel rank-transfer when evac_rank_transfer=true")
    parser.add_argument("--debug_wm_correction", action="store_true",
                        help="Enable debug visualization for WM correction (saves images/stats to ckpt_dir/debug_wm)")
    parser.add_argument("--export_correction_dataset", type=str2bool, default=False,
                        help="Export successful correction samples as ACT-compatible hdf5 episodes")
    parser.add_argument("--export_correction_dir", type=str, default="",
                        help="Output directory for exported correction episodes (default: ckpt_dir/correction_dataset)")

    main(vars(parser.parse_args()))
