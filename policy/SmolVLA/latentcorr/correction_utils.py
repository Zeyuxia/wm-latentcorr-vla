from __future__ import annotations

import argparse

import numpy as np


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_evac_infer_kwargs(cfg):
    infer_kwargs = {}
    if not bool(cfg.get("evac_use_dual_cache", False)):
        return infer_kwargs

    infer_kwargs["use_dual_cache"] = True
    dc_v_bounds = cfg.get("evac_dc_v_bounds", None)
    if dc_v_bounds is None:
        dc_v_bounds = []
    if isinstance(dc_v_bounds, str):
        dc_v_bounds = [part for part in dc_v_bounds.replace(",", " ").split() if part]
    dc_v_bounds = [int(bound) for bound in dc_v_bounds]
    if dc_v_bounds:
        infer_kwargs["dc_v_bounds"] = dc_v_bounds

    dc_budget = cfg.get("evac_dc_budget", None)
    if dc_budget is not None:
        dc_budget = float(dc_budget)
        if dc_budget >= 0.0:
            infer_kwargs["dc_budget"] = dc_budget

    infer_kwargs["dc_enc_start"] = int(cfg.get("evac_dc_enc_start", 999))
    infer_kwargs["dc_replay_step_noise"] = bool(cfg.get("evac_dc_replay_step_noise", False))
    infer_kwargs["dc_hf_metric"] = bool(cfg.get("evac_dc_hf_metric", False))
    infer_kwargs["dc_v_blur_on_reuse"] = bool(cfg.get("evac_dc_v_blur_on_reuse", False))
    infer_kwargs["dc_v_blur_kernel"] = int(cfg.get("evac_dc_v_blur_kernel", 3))
    infer_kwargs["dc_v_blur_strength"] = float(cfg.get("evac_dc_v_blur_strength", 0.15))
    return infer_kwargs


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
