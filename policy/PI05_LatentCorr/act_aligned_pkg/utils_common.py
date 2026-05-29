from __future__ import annotations

import numpy as np


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

