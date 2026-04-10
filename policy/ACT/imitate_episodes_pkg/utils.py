from __future__ import annotations

import argparse
import numpy as np

PHASE_KEYS = ("approach", "pregrasp", "transport", "place")
PHASE_KEY_TO_ID = {k: i for i, k in enumerate(PHASE_KEYS)}
PHASE_ID_TO_KEY = {i: k for i, k in enumerate(PHASE_KEYS)}

ERROR_MODE_KEYS = ("translation", "rotation", "gripper_close")
ERROR_MODE_KEY_TO_ID = {k: i for i, k in enumerate(ERROR_MODE_KEYS)}
ERROR_MODE_ID_TO_KEY = {i: k for i, k in enumerate(ERROR_MODE_KEYS)}

ACTIVE_ARM_PATTERN_KEYS = ("left_only", "right_only", "both")
ACTIVE_ARM_PATTERN_KEY_TO_ID = {k: i for i, k in enumerate(ACTIVE_ARM_PATTERN_KEYS)}
ACTIVE_ARM_PATTERN_ID_TO_KEY = {i: k for i, k in enumerate(ACTIVE_ARM_PATTERN_KEYS)}

FAILURE_TRANSLATION_DIR_BINS = 5
FAILURE_TRANSLATION_MAG_BINS = 3
FAILURE_ROTATION_DIR_BINS = 6
FAILURE_ROTATION_MAG_BINS = 3


def set_failure_param_bins(
    translation_dir_bins=None,
    translation_mag_bins=None,
    rotation_dir_bins=None,
    rotation_mag_bins=None,
):
    global FAILURE_TRANSLATION_DIR_BINS, FAILURE_TRANSLATION_MAG_BINS
    global FAILURE_ROTATION_DIR_BINS, FAILURE_ROTATION_MAG_BINS
    if translation_dir_bins is not None:
        FAILURE_TRANSLATION_DIR_BINS = int(max(1, int(translation_dir_bins)))
    if translation_mag_bins is not None:
        FAILURE_TRANSLATION_MAG_BINS = int(max(1, int(translation_mag_bins)))
    if rotation_dir_bins is not None:
        FAILURE_ROTATION_DIR_BINS = int(max(1, int(rotation_dir_bins)))
    if rotation_mag_bins is not None:
        FAILURE_ROTATION_MAG_BINS = int(max(1, int(rotation_mag_bins)))
    return {
        "translation_dir_bins": int(FAILURE_TRANSLATION_DIR_BINS),
        "translation_mag_bins": int(FAILURE_TRANSLATION_MAG_BINS),
        "rotation_dir_bins": int(FAILURE_ROTATION_DIR_BINS),
        "rotation_mag_bins": int(FAILURE_ROTATION_MAG_BINS),
    }


def get_failure_param_bins():
    return {
        "translation_dir_bins": int(FAILURE_TRANSLATION_DIR_BINS),
        "translation_mag_bins": int(FAILURE_TRANSLATION_MAG_BINS),
        "rotation_dir_bins": int(FAILURE_ROTATION_DIR_BINS),
        "rotation_mag_bins": int(FAILURE_ROTATION_MAG_BINS),
    }


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


def phase_key_to_id(phase_key):
    key = str(phase_key).strip().lower()
    if key not in PHASE_KEY_TO_ID:
        raise ValueError(
            f"Invalid phase_key={phase_key!r}. Expected one of {list(PHASE_KEY_TO_ID.keys())}."
        )
    return int(PHASE_KEY_TO_ID[key])


def phase_id_to_key(phase_id):
    idx = int(phase_id)
    if idx not in PHASE_ID_TO_KEY:
        raise ValueError(
            f"Invalid phase_id={phase_id!r}. Expected one of {list(PHASE_ID_TO_KEY.keys())}."
        )
    return str(PHASE_ID_TO_KEY[idx])


def error_mode_key_to_id(error_mode_key):
    key = str(error_mode_key).strip().lower()
    if key not in ERROR_MODE_KEY_TO_ID:
        raise ValueError(
            f"Invalid error_mode_key={error_mode_key!r}. "
            f"Expected one of {list(ERROR_MODE_KEY_TO_ID.keys())}."
        )
    return int(ERROR_MODE_KEY_TO_ID[key])


def error_mode_id_to_key(error_mode_id):
    idx = int(error_mode_id)
    if idx not in ERROR_MODE_ID_TO_KEY:
        raise ValueError(
            f"Invalid error_mode_id={error_mode_id!r}. "
            f"Expected one of {list(ERROR_MODE_ID_TO_KEY.keys())}."
        )
    return str(ERROR_MODE_ID_TO_KEY[idx])


def active_arm_pattern_key_to_id(active_arm_pattern_key):
    key = str(active_arm_pattern_key).strip().lower()
    if key not in ACTIVE_ARM_PATTERN_KEY_TO_ID:
        raise ValueError(
            f"Invalid active_arm_pattern_key={active_arm_pattern_key!r}. "
            f"Expected one of {list(ACTIVE_ARM_PATTERN_KEY_TO_ID.keys())}."
        )
    return int(ACTIVE_ARM_PATTERN_KEY_TO_ID[key])


def active_arm_pattern_id_to_key(active_arm_pattern_id):
    idx = int(active_arm_pattern_id)
    if idx not in ACTIVE_ARM_PATTERN_ID_TO_KEY:
        raise ValueError(
            f"Invalid active_arm_pattern_id={active_arm_pattern_id!r}. "
            f"Expected one of {list(ACTIVE_ARM_PATTERN_ID_TO_KEY.keys())}."
        )
    return str(ACTIVE_ARM_PATTERN_ID_TO_KEY[idx])


def error_mode_bin_counts(error_mode_key):
    mode = str(error_mode_key).strip().lower()
    if mode == "translation":
        return int(FAILURE_TRANSLATION_DIR_BINS), int(FAILURE_TRANSLATION_MAG_BINS)
    if mode == "rotation":
        return int(FAILURE_ROTATION_DIR_BINS), int(FAILURE_ROTATION_MAG_BINS)
    if mode == "gripper_close":
        return 0, 0
    raise ValueError(
        f"Invalid error_mode_key={error_mode_key!r}. "
        f"Expected one of {list(ERROR_MODE_KEY_TO_ID.keys())}."
    )


def normalize_error_mode_dir_mag_bins(error_mode_key, dir_bin_id, mag_bin_id):
    mode = str(error_mode_key).strip().lower()
    if mode == "gripper_close":
        return -1, -1
    n_dir, n_mag = error_mode_bin_counts(mode)
    d = int(np.clip(int(dir_bin_id), 0, max(0, n_dir - 1)))
    m = int(np.clip(int(mag_bin_id), 0, max(0, n_mag - 1)))
    return d, m


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
        parser.add_argument("--sample_phase_window_len", type=int, default=16,
                            help="Window length used by shared phase inference during start_ts sampling")
        parser.add_argument("--failure_mode", type=str, default="off",
                            help="Failure pipeline mode: off|explore|train")
        parser.add_argument("--failure_table_path", type=str, default="",
                            help="Path to failure_table.json (required in failure_mode=train)")
        parser.add_argument("--failure_table_dir", type=str, default="",
                            help="Directory for explore outputs (default: ckpt_dir/failure_explore)")
        parser.add_argument("--failure_phase_bins", type=int, default=5,
                            help="Number of bins per phase for failure sampling")
        parser.add_argument("--failure_translation_dir_bins", type=int, default=5,
                            help="Number of translation direction bins for failure exploration")
        parser.add_argument("--failure_translation_mag_bins", type=int, default=3,
                            help="Number of translation magnitude bins for failure exploration")
        parser.add_argument("--failure_rotation_dir_bins", type=int, default=6,
                            help="Number of rotation direction bins for failure exploration")
        parser.add_argument("--failure_rotation_mag_bins", type=int, default=3,
                            help="Number of rotation magnitude bins for failure exploration")
        parser.add_argument("--failure_corr_batch_ratio", type=float, default=0.5,
                            help="Correction dataloader batch ratio relative to base batch size in failure_mode")
        parser.add_argument("--failure_explore_k", type=int, default=3,
                            help="Required unique samples (episode_id,start_ts) per failure unit before writing it to failure_table in explore mode")
        parser.add_argument("--failure_fail_recover_rate_thresh", type=float, default=0.5,
                            help="Mark a unit as failure when recover_rate <= this threshold")

        # Online perturbation
        parser.add_argument("--perturb_eef_fail_gain", type=float, default=0.03,
                            help="Directional EEF perturbation magnitude in meters")
        parser.add_argument("--perturb_rot_max_deg", type=float, default=15.0,
                            help="Max EEF rotation perturbation angle in degrees")
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
        parser.add_argument("--debug_correction_evac_rollout", type=str2bool, default=False,
                            help="If true, run EVAC once on final generated correction trajectory and save outputs.mp4")
        parser.add_argument("--rollout_exec_steps", type=int, default=None,
                            help="Number of actions executed per rollout step (prefix of chunk)")
        parser.add_argument("--recover_eval_enable", type=str2bool, default=False,
                            help="Evaluate if ACT can recover from perturbed observation at each rollout step")
        parser.add_argument("--recover_eval_save_video", type=str2bool, default=False,
                            help="If true, save EVAC rollout video for ACT recover-eval action sequence at each rollout step")
        parser.add_argument("--recover_eval_gripper_open_thresh", type=float, default=0.8,
                            help="Recoverability threshold for gripper_close: max open value in first rollout_exec_steps")
        parser.add_argument("--recover_eval_pos_thresh_m", type=float, default=0.03,
                            help="Recoverability threshold for translation: EEF position error at step rollout_exec_steps (meters)")
        parser.add_argument("--recover_eval_rot_thresh_deg", type=float, default=10.0,
                            help="Recoverability threshold for rotation: EEF orientation error at step rollout_exec_steps (degrees)")
        parser.add_argument("--recover_eval_nearest_window_radius", type=int, default=16,
                            help="Forward-only nearest matching radius for translation/rotation recover_eval around center=start_ts+rollout_exec_steps")
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
