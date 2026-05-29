import numpy as np


def infer_phase_key_from_gt_window(left_grip_traj, right_grip_traj, window_len):
    """
    Infer phase from the whole GT action window (prefix), not a single point.
    This function is intentionally shared by rollout logic and dataloader sampling.
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
