from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from policy.ACT.util.fk_sapien import SapienFK


def find_nearest_traj_point(
    fk_left_pos: np.ndarray,
    fk_left_quat: np.ndarray,
    fk_right_pos: np.ndarray,
    fk_right_quat: np.ndarray,
    left_endpose: np.ndarray,
    right_endpose: np.ndarray,
    orient_weight: float = 0.0,
    curr_left_grip: float | None = None,
    curr_right_grip: float | None = None,
    left_gripper_traj: np.ndarray | None = None,
    right_gripper_traj: np.ndarray | None = None,
    gripper_penalty: float = 0.0,
    window_start: int | None = None,
    window_end: int | None = None,
) -> tuple[int, float]:
    left_d = np.linalg.norm(left_endpose[:, :3] - fk_left_pos, axis=1)
    right_d = np.linalg.norm(right_endpose[:, :3] - fk_right_pos, axis=1)
    dists = (left_d + right_d) / 2.0
    if orient_weight > 0:
        left_dot = np.clip(np.abs(np.sum(left_endpose[:, 3:7] * fk_left_quat, axis=1)), 0.0, 1.0)
        right_dot = np.clip(np.abs(np.sum(right_endpose[:, 3:7] * fk_right_quat, axis=1)), 0.0, 1.0)
        left_d_ori = 2.0 * np.arccos(left_dot)
        right_d_ori = 2.0 * np.arccos(right_dot)
        dists += float(orient_weight) * (left_d_ori + right_d_ori) / 2.0
    if (
        gripper_penalty > 0.0
        and left_gripper_traj is not None
        and right_gripper_traj is not None
        and curr_left_grip is not None
        and curr_right_grip is not None
    ):
        curr_lg = 0.0 if float(curr_left_grip) <= 0.5 else 1.0
        curr_rg = 0.0 if float(curr_right_grip) <= 0.5 else 1.0
        traj_lg = (np.asarray(left_gripper_traj, dtype=np.float32) > 0.5).astype(np.float32)
        traj_rg = (np.asarray(right_gripper_traj, dtype=np.float32) > 0.5).astype(np.float32)
        dists += float(gripper_penalty) * (np.abs(traj_lg - curr_lg) + np.abs(traj_rg - curr_rg)) / 2.0
    if window_start is not None or window_end is not None:
        ws = 0 if window_start is None else int(np.clip(window_start, 0, len(dists) - 1))
        we = len(dists) if window_end is None else int(np.clip(window_end, ws + 1, len(dists)))
        d_win = np.full_like(dists, np.inf, dtype=np.float32)
        d_win[ws:we] = dists[ws:we]
        if np.any(np.isfinite(d_win)):
            dists = d_win
    t_star = int(np.argmin(dists))
    return t_star, float(dists[t_star])


def resample_trajectory(traj: np.ndarray, target_len: int) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    n = int(len(traj))
    if n == target_len:
        return traj.astype(np.float32)
    if n == 0:
        return np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    indices = np.linspace(0, n - 1, target_len)
    result = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = float(idx - lo)
        result[i] = traj[lo] * (1.0 - frac) + traj[hi] * frac
    return result


def _find_next_gripper_toggle_idx_from(
    left_grip_traj: np.ndarray,
    right_grip_traj: np.ndarray,
    start_idx: int,
) -> int:
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
    gt_left_arm: np.ndarray | None,
    gt_right_arm: np.ndarray | None,
    gt_left_grip: np.ndarray | None,
    gt_right_grip: np.ndarray | None,
    t_idx: int,
    window_len: int,
    joint_delta_thresh: float = 0.02,
    gripper_delta_thresh: float = 0.05,
) -> dict[str, Any]:
    def _bounds(n: int) -> tuple[int, int]:
        if n <= 1:
            return 0, 1
        s = int(np.clip(t_idx, 0, n - 1))
        e = int(np.clip(s + max(2, int(window_len)), s + 1, n))
        return s, e

    def _arm_active(arr: np.ndarray | None) -> tuple[bool, float]:
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

    def _grip_active(arr: np.ndarray | None) -> tuple[bool, float]:
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
        "left_arm": bool(la),
        "right_arm": bool(ra),
        "left_gripper": bool(lg),
        "right_gripper": bool(rg),
        "left_arm_score": float(la_s),
        "right_arm_score": float(ra_s),
        "left_gripper_score": float(lg_s),
        "right_gripper_score": float(rg_s),
    }


@dataclass
class PlannerCorrectionConfig:
    correction_horizon: int = 16
    target_mode: str = "backward"
    target_lookahead_steps: int = 6
    correction_interp_nearest_enable: bool = False
    correction_interp_prefix_ratio: float = 0.4
    correction_gripper_switch_ratio: float = 0.8
    use_interp_fallback_on_planner_fail: bool = True
    orient_weight: float = 0.0573
    gripper_penalty: float = 1.0
    nearest_window_radius: int = 12
    active_joint_delta_thresh: float = 0.01
    active_gripper_delta_thresh: float = 0.05


class PlannerCorrectionBuilder:
    def __init__(
        self,
        urdf_path: str,
        curobo_left_yml: str,
        curobo_right_yml: str,
        cfg: PlannerCorrectionConfig,
        device: str | torch.device | None = None,
        planner_warmup: bool = True,
    ):
        self.cfg = cfg
        self.fk = SapienFK(urdf_path)
        self.planner_left = None
        self.planner_right = None

        # When using interpolation-only correction targets, planner instantiation is
        # unnecessary and can destabilize multi-process training.
        if bool(self.cfg.correction_interp_nearest_enable):
            return

        import sapien

        robotwin_root = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        env_robot_dir = os.path.join(robotwin_root, "envs", "robot")
        curobo_src_dir = os.path.join(robotwin_root, "envs", "curobo", "src")
        for path in (robotwin_root, env_robot_dir, curobo_src_dir):
            if path not in sys.path:
                sys.path.insert(0, path)
        prev_cwd = os.getcwd()
        os.chdir(robotwin_root)
        try:
            from planner import CuroboPlanner  # noqa: WPS433
        finally:
            os.chdir(prev_cwd)

        root_pose = sapien.Pose([0, -0.65, 0], [0.707, 0, 0, 0.707])
        left_joints = [f"fl_joint{i}" for i in range(1, 7)]
        right_joints = [f"fr_joint{i}" for i in range(1, 7)]
        self.planner_left = CuroboPlanner(
            root_pose,
            left_joints,
            self.fk.jnames,
            yml_path=curobo_left_yml,
            device=device,
            do_warmup=planner_warmup,
        )
        self.planner_right = CuroboPlanner(
            root_pose,
            right_joints,
            self.fk.jnames,
            yml_path=curobo_right_yml,
            device=device,
            do_warmup=planner_warmup,
        )

    def _fit_prefix(self, path_future: np.ndarray, curr_q: np.ndarray, n: int) -> np.ndarray:
        p = np.asarray(path_future, dtype=np.float32)
        if p.ndim != 2 or p.shape[0] == 0:
            return np.repeat(np.asarray(curr_q, dtype=np.float32)[None, :], n, axis=0)
        if p.shape[0] == n:
            return p.astype(np.float32)
        if p.shape[0] > n:
            return resample_trajectory(p, n).astype(np.float32)
        pad = np.repeat(p[-1:].astype(np.float32), n - p.shape[0], axis=0)
        return np.concatenate([p.astype(np.float32), pad], axis=0).astype(np.float32)

    def _interp_prefix(self, curr: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
        curr = np.asarray(curr, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if n <= 1:
            return target[None, :].astype(np.float32)
        alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        return ((1.0 - alpha) * curr[None, :] + alpha * target[None, :]).astype(np.float32)

    def _build_gripper_prefix(
        self,
        current_value: float,
        target_value: float,
        prefix_len: int,
    ) -> np.ndarray:
        switch_ratio = float(np.clip(self.cfg.correction_gripper_switch_ratio, 0.0, 1.0))
        switch_idx = int(np.clip(np.floor(prefix_len * switch_ratio), 0, max(0, prefix_len - 1)))
        prefix = np.full((prefix_len,), float(current_value), dtype=np.float32)
        tail_len = prefix_len - switch_idx
        if tail_len > 1:
            prefix[switch_idx:] = np.linspace(float(current_value), float(target_value), tail_len, dtype=np.float32)
        elif tail_len == 1:
            prefix[-1] = float(target_value)
        return prefix

    def _build_interp_target(
        self,
        left_q: np.ndarray,
        right_q: np.ndarray,
        left_grip: float,
        right_grip: float,
        raw_data: dict[str, Any],
        start_ts: int,
        t_target: int,
        corr_left_active: bool,
        corr_right_active: bool,
    ) -> dict[str, Any] | None:
        gt_left_arm = raw_data.get("gt_left_arm")
        gt_right_arm = raw_data.get("gt_right_arm")
        abs_idx = int(start_ts + t_target)
        if gt_left_arm is None or gt_right_arm is None:
            return None

        l_tgt_q = np.asarray(gt_left_arm[abs_idx], dtype=np.float32)
        r_tgt_q = np.asarray(gt_right_arm[abs_idx], dtype=np.float32)
        lt = self._interp_prefix(left_q, l_tgt_q, self.cfg.correction_horizon)
        rt = self._interp_prefix(right_q, r_tgt_q, self.cfg.correction_horizon)

        tl_grip = float(np.clip(raw_data["left_gripper"][start_ts + t_target], 0.0, 1.0))
        tr_grip = float(np.clip(raw_data["right_gripper"][start_ts + t_target], 0.0, 1.0))
        lg = self._build_gripper_prefix(left_grip, tl_grip, self.cfg.correction_horizon)
        rg = self._build_gripper_prefix(right_grip, tr_grip, self.cfg.correction_horizon)

        if not corr_left_active:
            lt = np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], self.cfg.correction_horizon, axis=0)
            lg = np.full((self.cfg.correction_horizon,), float(left_grip), dtype=np.float32)
        if not corr_right_active:
            rt = np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], self.cfg.correction_horizon, axis=0)
            rg = np.full((self.cfg.correction_horizon,), float(right_grip), dtype=np.float32)

        return {
            "left_traj": lt,
            "right_traj": rt,
            "left_grip": lg,
            "right_grip": rg,
            "source": "joint_interp",
            "planner_left_status": "InterpNearest",
            "planner_right_status": "InterpNearest",
        }

    def build(
        self,
        action_dev_raw: torch.Tensor | np.ndarray,
        raw_data: dict[str, Any],
        norm_stats: dict[str, Any],
        start_ts: int,
    ) -> dict[str, Any] | None:
        import sapien

        act = action_dev_raw.detach().cpu().float().numpy() if isinstance(action_dev_raw, torch.Tensor) else np.asarray(action_dev_raw, dtype=np.float32)
        if act.ndim != 2 or act.shape[1] != 14:
            raise ValueError(f"action_dev_raw must be (K,14), got {tuple(act.shape)}")

        left_ep = np.asarray(raw_data["left_endpose"][start_ts:], dtype=np.float32)
        right_ep = np.asarray(raw_data["right_endpose"][start_ts:], dtype=np.float32)
        left_grip_traj = np.asarray(raw_data["left_gripper"][start_ts:], dtype=np.float32)
        right_grip_traj = np.asarray(raw_data["right_gripper"][start_ts:], dtype=np.float32)
        if len(left_ep) <= 1 or len(right_ep) <= 1:
            return None

        curr_q = np.asarray(act[-1], dtype=np.float32)
        left_q, right_q = curr_q[0:6], curr_q[7:13]
        left_grip, right_grip = float(curr_q[6]), float(curr_q[13])
        fk_r = self.fk.forward(left_q, right_q)
        lp, lq = fk_r["left"]
        rp, rq = fk_r["right"]

        expected_idx = int(np.clip(act.shape[0], 0, len(left_ep) - 1))
        near_w = int(max(1, self.cfg.nearest_window_radius))
        t_star, min_dist = find_nearest_traj_point(
            lp,
            lq,
            rp,
            rq,
            left_ep,
            right_ep,
            orient_weight=float(self.cfg.orient_weight),
            curr_left_grip=left_grip,
            curr_right_grip=right_grip,
            left_gripper_traj=left_grip_traj,
            right_gripper_traj=right_grip_traj,
            gripper_penalty=float(self.cfg.gripper_penalty),
            window_start=max(0, expected_idx - near_w),
            window_end=min(len(left_ep), expected_idx + near_w + 1),
        )

        corr_active_info = _infer_active_arms_from_gt_window(
            raw_data.get("gt_left_arm"),
            raw_data.get("gt_right_arm"),
            raw_data.get("left_gripper"),
            raw_data.get("right_gripper"),
            t_idx=int(start_ts + t_star),
            window_len=int(self.cfg.correction_horizon),
            joint_delta_thresh=float(self.cfg.active_joint_delta_thresh),
            gripper_delta_thresh=float(self.cfg.active_gripper_delta_thresh),
        )
        corr_left_active = bool(corr_active_info["left_arm"] or corr_active_info["left_gripper"])
        corr_right_active = bool(corr_active_info["right_arm"] or corr_active_info["right_gripper"])

        if bool(self.cfg.correction_interp_nearest_enable):
            t_target = int(np.clip(t_star, 0, len(left_ep) - 1))
            plan_out = self._build_interp_target(
                left_q=left_q,
                right_q=right_q,
                left_grip=left_grip,
                right_grip=right_grip,
                raw_data=raw_data,
                start_ts=int(start_ts),
                t_target=t_target,
                corr_left_active=corr_left_active,
                corr_right_active=corr_right_active,
            )
            if plan_out is None:
                return None
        else:
            if str(self.cfg.target_mode).strip().lower() == "backward":
                t_target = int(np.clip(t_star - int(self.cfg.target_lookahead_steps), 0, len(left_ep) - 1))
            else:
                t_target = int(np.clip(t_star + int(self.cfg.target_lookahead_steps), 0, len(left_ep) - 1))

            qpos_full = np.zeros(len(self.fk.jnames), dtype=np.float32)
            qpos_full[self.fk.fl_idx] = left_q
            qpos_full[self.fk.fr_idx] = right_q
            target_lp = sapien.Pose(left_ep[t_target, :3], left_ep[t_target, 3:7])
            target_rp = sapien.Pose(right_ep[t_target, :3], right_ep[t_target, 3:7])

            res_l = {"status": "SkippedInactive", "position": np.repeat(left_q[None, :], 1, axis=0)}
            res_r = {"status": "SkippedInactive", "position": np.repeat(right_q[None, :], 1, axis=0)}
            planner_failed = False
            if corr_left_active:
                try:
                    res_l = self.planner_left.plan_path(qpos_full, target_lp, arms_tag="left")
                except Exception:
                    planner_failed = True
                    res_l = {"status": "Exception"}
            if corr_right_active:
                try:
                    res_r = self.planner_right.plan_path(qpos_full, target_rp, arms_tag="right")
                except Exception:
                    planner_failed = True
                    res_r = {"status": "Exception"}
            if (corr_left_active and res_l.get("status") != "Success") or (corr_right_active and res_r.get("status") != "Success"):
                planner_failed = True

            if planner_failed and bool(self.cfg.use_interp_fallback_on_planner_fail):
                plan_out = self._build_interp_target(
                    left_q=left_q,
                    right_q=right_q,
                    left_grip=left_grip,
                    right_grip=right_grip,
                    raw_data=raw_data,
                    start_ts=int(start_ts),
                    t_target=t_target,
                    corr_left_active=corr_left_active,
                    corr_right_active=corr_right_active,
                )
                if plan_out is None:
                    return None
                plan_out["source"] = "interp_fallback"
                plan_out["planner_left_status"] = str(res_l.get("status"))
                plan_out["planner_right_status"] = str(res_r.get("status"))
            elif planner_failed:
                return None
            else:
                l_path = np.asarray(res_l["position"], dtype=np.float32)
                r_path = np.asarray(res_r["position"], dtype=np.float32)
                l_future = l_path[1:] if l_path.shape[0] > 1 else l_path
                r_future = r_path[1:] if r_path.shape[0] > 1 else r_path
                lt = self._fit_prefix(l_future, left_q, self.cfg.correction_horizon)
                rt = self._fit_prefix(r_future, right_q, self.cfg.correction_horizon)

                tl_grip = float(np.clip(left_grip_traj[t_target], 0.0, 1.0))
                tr_grip = float(np.clip(right_grip_traj[t_target], 0.0, 1.0))
                lg = self._build_gripper_prefix(left_grip, tl_grip, self.cfg.correction_horizon)
                rg = self._build_gripper_prefix(right_grip, tr_grip, self.cfg.correction_horizon)

                if not corr_left_active:
                    lt = np.repeat(np.asarray(left_q, dtype=np.float32)[None, :], self.cfg.correction_horizon, axis=0)
                    lg = np.full((self.cfg.correction_horizon,), float(left_grip), dtype=np.float32)
                if not corr_right_active:
                    rt = np.repeat(np.asarray(right_q, dtype=np.float32)[None, :], self.cfg.correction_horizon, axis=0)
                    rg = np.full((self.cfg.correction_horizon,), float(right_grip), dtype=np.float32)

                plan_out = {
                    "left_traj": lt,
                    "right_traj": rt,
                    "left_grip": lg,
                    "right_grip": rg,
                    "source": "planner",
                    "planner_left_status": str(res_l.get("status")),
                    "planner_right_status": str(res_r.get("status")),
                }

        corr = np.zeros((self.cfg.correction_horizon, 14), dtype=np.float32)
        corr[:, 0:6] = plan_out["left_traj"]
        corr[:, 7:13] = plan_out["right_traj"]
        corr[:, 6] = np.clip(plan_out["left_grip"], 0.0, 1.0)
        corr[:, 13] = np.clip(plan_out["right_grip"], 0.0, 1.0)

        action_mean = np.asarray(norm_stats["action_mean"], dtype=np.float32)
        action_std = np.asarray(norm_stats["action_std"], dtype=np.float32)
        corr_norm = (corr - action_mean) / action_std
        is_pad = np.zeros((self.cfg.correction_horizon,), dtype=bool)

        return {
            "correction_target_raw": torch.from_numpy(corr.astype(np.float32)),
            "correction_target_norm": torch.from_numpy(corr_norm.astype(np.float32)),
            "is_pad": torch.from_numpy(is_pad),
            "meta": {
                "source": plan_out["source"],
                "t_star": int(t_star),
                "t_target": int(t_target),
                "min_dist": float(min_dist),
                "planner_left_status": plan_out["planner_left_status"],
                "planner_right_status": plan_out["planner_right_status"],
                "corr_left_active": bool(corr_left_active),
                "corr_right_active": bool(corr_right_active),
                "anchor_idx": int(np.clip(_find_next_gripper_toggle_idx_from(left_grip_traj, right_grip_traj, int(t_star)), 0, len(left_ep) - 1)),
            },
        }
