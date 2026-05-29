from __future__ import annotations

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
    translation_dir_bins: int | None = None,
    translation_mag_bins: int | None = None,
    rotation_dir_bins: int | None = None,
    rotation_mag_bins: int | None = None,
) -> dict[str, int]:
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
        raise ValueError(
            f"Invalid phase_key={phase_key!r}. Expected one of {list(PHASE_KEY_TO_ID.keys())}."
        )
    return int(PHASE_KEY_TO_ID[key])


def phase_id_to_key(phase_id: int) -> str:
    idx = int(phase_id)
    if idx not in PHASE_ID_TO_KEY:
        raise ValueError(
            f"Invalid phase_id={phase_id!r}. Expected one of {list(PHASE_ID_TO_KEY.keys())}."
        )
    return str(PHASE_ID_TO_KEY[idx])


def error_mode_key_to_id(error_mode_key: str) -> int:
    key = str(error_mode_key).strip().lower()
    if key not in ERROR_MODE_KEY_TO_ID:
        raise ValueError(
            f"Invalid error_mode_key={error_mode_key!r}. "
            f"Expected one of {list(ERROR_MODE_KEY_TO_ID.keys())}."
        )
    return int(ERROR_MODE_KEY_TO_ID[key])


def error_mode_id_to_key(error_mode_id: int) -> str:
    idx = int(error_mode_id)
    if idx not in ERROR_MODE_ID_TO_KEY:
        raise ValueError(
            f"Invalid error_mode_id={error_mode_id!r}. "
            f"Expected one of {list(ERROR_MODE_ID_TO_KEY.keys())}."
        )
    return str(ERROR_MODE_ID_TO_KEY[idx])


def active_arm_pattern_key_to_id(active_arm_pattern_key: str) -> int:
    key = str(active_arm_pattern_key).strip().lower()
    if key not in ACTIVE_ARM_PATTERN_KEY_TO_ID:
        raise ValueError(
            f"Invalid active_arm_pattern_key={active_arm_pattern_key!r}. "
            f"Expected one of {list(ACTIVE_ARM_PATTERN_KEY_TO_ID.keys())}."
        )
    return int(ACTIVE_ARM_PATTERN_KEY_TO_ID[key])


def active_arm_pattern_id_to_key(active_arm_pattern_id: int) -> str:
    idx = int(active_arm_pattern_id)
    if idx not in ACTIVE_ARM_PATTERN_ID_TO_KEY:
        raise ValueError(
            f"Invalid active_arm_pattern_id={active_arm_pattern_id!r}. "
            f"Expected one of {list(ACTIVE_ARM_PATTERN_ID_TO_KEY.keys())}."
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
    raise ValueError(
        f"Invalid error_mode_key={error_mode_key!r}. "
        f"Expected one of {list(ERROR_MODE_KEY_TO_ID.keys())}."
    )


def normalize_error_mode_dir_mag_bins(error_mode_key: str, dir_bin_id: int, mag_bin_id: int) -> tuple[int, int]:
    mode = str(error_mode_key).strip().lower()
    if mode == "gripper_close":
        return -1, -1
    n_dir, n_mag = error_mode_bin_counts(mode)
    d = int(np.clip(int(dir_bin_id), 0, max(0, n_dir - 1)))
    m = int(np.clip(int(mag_bin_id), 0, max(0, n_mag - 1)))
    return d, m
