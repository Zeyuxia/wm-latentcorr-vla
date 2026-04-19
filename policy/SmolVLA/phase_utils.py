from __future__ import annotations

import numpy as np


def infer_phase_key_from_gt_window(left_grip_traj, right_grip_traj, window_len: int) -> str:
    left = np.asarray(left_grip_traj).reshape(-1)
    right = np.asarray(right_grip_traj).reshape(-1)
    n = int(max(1, min(len(left), len(right), int(window_len))))
    left = left[:n]
    right = right[:n]

    def smooth3(arr: np.ndarray) -> np.ndarray:
        if arr.size < 3:
            return arr
        kernel = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
        return np.convolve(arr, kernel, mode="same")

    left_smooth = smooth3(left)
    right_smooth = smooth3(right)

    def first_cross(arr: np.ndarray, open_event: bool) -> int | None:
        if arr.size < 2:
            return None
        if open_event:
            idx = np.where((arr[:-1] <= 0.5) & (arr[1:] > 0.5))[0]
        else:
            idx = np.where((arr[:-1] > 0.5) & (arr[1:] <= 0.5))[0]
        if idx.size == 0:
            return None
        return int(idx[0])

    open_left = first_cross(left_smooth, open_event=True)
    open_right = first_cross(right_smooth, open_event=True)
    close_left = first_cross(left_smooth, open_event=False)
    close_right = first_cross(right_smooth, open_event=False)

    open_events = [value for value in (open_left, open_right) if value is not None]
    close_events = [value for value in (close_left, close_right) if value is not None]
    first_open = min(open_events) if open_events else None
    first_close = min(close_events) if close_events else None

    if first_open is not None and first_close is not None:
        return "pregrasp" if first_close < first_open else "place"
    if first_close is not None:
        return "pregrasp"
    if first_open is not None:
        return "place"

    mean_left = float(np.mean(left_smooth))
    mean_right = float(np.mean(right_smooth))
    if mean_left > 0.5 and mean_right > 0.5:
        return "approach"
    if mean_left <= 0.5 or mean_right <= 0.5:
        return "transport"
    return "approach"

