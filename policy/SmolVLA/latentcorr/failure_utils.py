from __future__ import annotations

import numpy as np

PHASE_KEYS = ("approach", "pregrasp", "transport", "place")
PHASE_KEY_TO_ID = {key: idx for idx, key in enumerate(PHASE_KEYS)}
PHASE_ID_TO_KEY = {idx: key for idx, key in enumerate(PHASE_KEYS)}

ERROR_MODE_KEYS = ("translation", "rotation", "gripper_close")
ERROR_MODE_KEY_TO_ID = {key: idx for idx, key in enumerate(ERROR_MODE_KEYS)}
ERROR_MODE_ID_TO_KEY = {idx: key for idx, key in enumerate(ERROR_MODE_KEYS)}

ACTIVE_ARM_PATTERN_KEYS = ("left_arm", "right_arm")
ACTIVE_ARM_PATTERN_KEY_TO_ID = {key: idx for idx, key in enumerate(ACTIVE_ARM_PATTERN_KEYS)}
ACTIVE_ARM_PATTERN_ID_TO_KEY = {idx: key for idx, key in enumerate(ACTIVE_ARM_PATTERN_KEYS)}

FAILURE_TRANSLATION_DIR_BINS = 5
FAILURE_TRANSLATION_MAG_BINS = 1
FAILURE_ROTATION_DIR_BINS = 6
FAILURE_ROTATION_MAG_BINS = 1


def set_failure_param_bins(
    translation_dir_bins: int,
    translation_mag_bins: int,
    rotation_dir_bins: int,
    rotation_mag_bins: int,
) -> dict[str, int]:
    global FAILURE_TRANSLATION_DIR_BINS, FAILURE_TRANSLATION_MAG_BINS
    global FAILURE_ROTATION_DIR_BINS, FAILURE_ROTATION_MAG_BINS
    FAILURE_TRANSLATION_DIR_BINS = int(max(1, int(translation_dir_bins)))
    FAILURE_TRANSLATION_MAG_BINS = int(max(1, int(translation_mag_bins)))
    FAILURE_ROTATION_DIR_BINS = int(max(1, int(rotation_dir_bins)))
    FAILURE_ROTATION_MAG_BINS = int(max(1, int(rotation_mag_bins)))
    return get_failure_param_bins()


def get_failure_param_bins() -> dict[str, int]:
    return {
        "translation_dir_bins": int(FAILURE_TRANSLATION_DIR_BINS),
        "translation_mag_bins": int(FAILURE_TRANSLATION_MAG_BINS),
        "rotation_dir_bins": int(FAILURE_ROTATION_DIR_BINS),
        "rotation_mag_bins": int(FAILURE_ROTATION_MAG_BINS),
    }


def phase_key_to_id(phase_key: str) -> int:
    key = str(phase_key).strip().lower()
    if key not in PHASE_KEY_TO_ID:
        raise ValueError(f"Invalid phase_key={phase_key!r}. Expected one of {list(PHASE_KEY_TO_ID.keys())}.")
    return int(PHASE_KEY_TO_ID[key])


def phase_id_to_key(phase_id: int) -> str:
    idx = int(phase_id)
    if idx not in PHASE_ID_TO_KEY:
        raise ValueError(f"Invalid phase_id={phase_id!r}. Expected one of {list(PHASE_ID_TO_KEY.keys())}.")
    return str(PHASE_ID_TO_KEY[idx])


def error_mode_key_to_id(error_mode_key: str) -> int:
    key = str(error_mode_key).strip().lower()
    if key not in ERROR_MODE_KEY_TO_ID:
        raise ValueError(
            f"Invalid error_mode_key={error_mode_key!r}. Expected one of {list(ERROR_MODE_KEY_TO_ID.keys())}."
        )
    return int(ERROR_MODE_KEY_TO_ID[key])


def error_mode_id_to_key(error_mode_id: int) -> str:
    idx = int(error_mode_id)
    if idx not in ERROR_MODE_ID_TO_KEY:
        raise ValueError(
            f"Invalid error_mode_id={error_mode_id!r}. Expected one of {list(ERROR_MODE_ID_TO_KEY.keys())}."
        )
    return str(ERROR_MODE_ID_TO_KEY[idx])


def canonicalize_active_arm_pattern_key(active_arm_pattern_key: str) -> str:
    key = str(active_arm_pattern_key).strip().lower()
    if key in {"left_arm", "left_only"}:
        return "left_arm"
    if key in {"right_arm", "right_only"}:
        return "right_arm"
    if key == "both":
        return "both"
    raise ValueError(
        "Invalid active_arm_pattern_key="
        f"{active_arm_pattern_key!r}. Expected one of left_arm/right_arm/both."
    )


def active_arm_pattern_key_to_id(active_arm_pattern_key: str) -> int:
    key = canonicalize_active_arm_pattern_key(active_arm_pattern_key)
    if key not in ACTIVE_ARM_PATTERN_KEY_TO_ID:
        raise ValueError(
            "Invalid active_arm_pattern_key="
            f"{active_arm_pattern_key!r}. Expected one of {list(ACTIVE_ARM_PATTERN_KEY_TO_ID.keys())}."
        )
    return int(ACTIVE_ARM_PATTERN_KEY_TO_ID[key])


def active_arm_pattern_id_to_key(active_arm_pattern_id: int) -> str:
    idx = int(active_arm_pattern_id)
    if idx not in ACTIVE_ARM_PATTERN_ID_TO_KEY:
        raise ValueError(
            "Invalid active_arm_pattern_id="
            f"{active_arm_pattern_id!r}. Expected one of {list(ACTIVE_ARM_PATTERN_ID_TO_KEY.keys())}."
        )
    return str(ACTIVE_ARM_PATTERN_ID_TO_KEY[idx])


def error_mode_bin_counts(error_mode_key: str) -> tuple[int, int]:
    mode = str(error_mode_key).strip().lower()
    if mode == "translation":
        return int(FAILURE_TRANSLATION_DIR_BINS), int(FAILURE_TRANSLATION_MAG_BINS)
    if mode == "rotation":
        return int(FAILURE_ROTATION_DIR_BINS), int(FAILURE_ROTATION_MAG_BINS)
    if mode == "gripper_close":
        return 0, 0
    raise ValueError(f"Invalid error_mode_key={error_mode_key!r}.")


def normalize_error_mode_dir_mag_bins(error_mode_key: str, dir_bin_id: int, mag_bin_id: int) -> tuple[int, int]:
    mode = str(error_mode_key).strip().lower()
    if mode == "gripper_close":
        return -1, -1
    num_dir_bins, num_mag_bins = error_mode_bin_counts(mode)
    dir_idx = int(np.clip(int(dir_bin_id), 0, max(0, num_dir_bins - 1)))
    mag_idx = int(np.clip(int(mag_bin_id), 0, max(0, num_mag_bins - 1)))
    return dir_idx, mag_idx
