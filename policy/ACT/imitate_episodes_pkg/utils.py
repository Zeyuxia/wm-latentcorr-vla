from __future__ import annotations

import argparse
import numpy as np


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


def resample_trajectory(traj, target_len):
    """Linearly resample a 2D trajectory array to target length."""
    arr = np.asarray(traj, dtype=np.float32)
    n = int(len(arr))
    if n == int(target_len):
        return arr.astype(np.float32)
    if n == 0:
        return np.zeros((int(target_len), arr.shape[1]), dtype=np.float32)
    indices = np.linspace(0, n - 1, int(target_len), dtype=np.float32)
    out = np.zeros((int(target_len), arr.shape[1]), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = float(idx - lo)
        out[i] = arr[lo] * (1.0 - frac) + arr[hi] * frac
    return out.astype(np.float32)


def infer_phase_key_from_gt_window(left_grip_traj, right_grip_traj, window_len):
    """
    Infer phase from the whole GT action window (prefix), not a single point.
    Shared by rollout logic and dataloader sampling.
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

    # For open_laptop-style tasks, slope-based trend fallback is intentionally
    # removed to avoid noisy short-window misclassification.
    # Fallback now relies only on gripper openness level.
    mean_l = float(np.mean(l_s))
    mean_r = float(np.mean(r_s))
    if mean_l > 0.5 and mean_r > 0.5:
        return "approach"
    if (mean_l <= 0.5) or (mean_r <= 0.5):
        return "transport"
    return "approach"


def build_parser() -> argparse.ArgumentParser:
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
        parser.add_argument("--multi_task_names", action="store", type=str, default="",
                            help="Comma-separated task names for multi-task training, e.g. sim-open_laptop-demo_clean-50,sim-blocks_ranking_rgb-demo_clean-50")
        parser.add_argument("--multi_task_weights", action="store", type=str, default="",
                            help="Comma-separated sampling weights for multi_task_names, e.g. 1.0,1.0")
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
        parser.add_argument("--sample_skip_head_ratio", type=float, default=0.25,
                            help="Skip first ratio of first-stage(approach) length when sampling start_ts")
        parser.add_argument("--sample_pregrasp_bias_enable", type=str2bool, default=False,
                            help="Bias dataloader start_ts sampling towards pregrasp candidates")
        parser.add_argument("--sample_pregrasp_prob", type=float, default=0.0,
                            help="When pregrasp bias is enabled, probability of sampling from pregrasp candidates")
        parser.add_argument("--sample_phase_window_len", type=int, default=16,
                            help="Window length used by shared phase inference during start_ts sampling")
        parser.add_argument("--sample_pregrasp_keep_start_ratio", type=float, default=0.0,
                            help="Start ratio (inclusive) of each pregrasp segment to keep for sampling")
        parser.add_argument("--sample_pregrasp_keep_end_ratio", type=float, default=0.5,
                            help="End ratio (exclusive) of each pregrasp segment to keep for sampling")
        parser.add_argument("--wm_corr_pregrasp_extra_enable", type=str2bool, default=False,
                            help="If true, use an extra pregrasp-only dataloader for WM correction samples")
        parser.add_argument("--wm_corr_pregrasp_extra_ratio", type=float, default=0.5,
                            help="Extra pregrasp dataloader batch ratio relative to base batch_size")

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
        parser.add_argument("--correction_force_generate", type=str2bool, default=False,
                            help="If true, always generate correction trajectory after rollout")
        parser.add_argument("--debug_correction_evac_rollout", type=str2bool, default=False,
                            help="If true, run EVAC once on final generated correction trajectory and save outputs.mp4")
        parser.add_argument("--rollout_exec_steps", type=int, default=None,
                            help="Number of actions executed per rollout step (prefix of chunk)")
        parser.add_argument("--orient_weight", type=float,
                            help="Weight for orientation distance in nearest-point matching (0=position only)")
        parser.add_argument("--gripper_penalty", type=float,
                            help="Penalty for gripper state mismatch in nearest-point matching (0=ignore gripper)")
        parser.add_argument("--debug_wm_correction", action="store_true",
                            help="Enable debug visualization for WM correction (saves images/stats to ckpt_dir/debug_wm)")
        parser.add_argument("--debug_loss_batch_projection", type=str2bool, default=False,
                            help="Save projection images for every sample in the final batch used for loss update.")
        parser.add_argument("--export_correction_dataset", type=str2bool, default=False,
                            help="Export successful correction samples as ACT-compatible hdf5 episodes")
        parser.add_argument("--export_correction_dir", type=str, default="",
                            help="Output directory for exported correction episodes (default: ckpt_dir/correction_dataset)")

        return parser
