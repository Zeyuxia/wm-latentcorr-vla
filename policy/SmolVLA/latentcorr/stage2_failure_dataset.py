from __future__ import annotations

import hashlib
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from policy.SmolVLA.latentcorr.failure_utils import (
    ACTIVE_ARM_PATTERN_KEYS,
    ERROR_MODE_KEYS,
    active_arm_pattern_key_to_id,
    canonicalize_active_arm_pattern_key,
    error_mode_key_to_id,
    get_failure_param_bins,
    normalize_error_mode_dir_mag_bins,
    phase_key_to_id,
    set_failure_param_bins,
)
from policy.SmolVLA.latentcorr.latent_dataset_utils import get_norm_stats, list_valid_episode_ids, load_processed_episode_window
from policy.SmolVLA.latentcorr.phase_utils import infer_phase_key_from_gt_window
from policy.SmolVLA.latentcorr.correction_perturbation import (
    _mag_value_from_bin,
    _rotation_axis_from_bin,
    _rotation_sign_from_bin,
    _translation_dir_from_bin,
)

FAILURE_ERROR_MODES = tuple(ERROR_MODE_KEYS)
ACTIVE_ARM_PATTERNS = set(ACTIVE_ARM_PATTERN_KEYS)
EXPLORE_ACTIVE_ARM_PATTERN_SORT_KEYS = {
    "left_arm": 0,
    "right_arm": 1,
}
EXPLORE_PHASE_SORT_KEYS = {
    "pregrasp": 0,
    "approach": 1,
    "transport": 2,
    "place": 3,
}
EXPLORE_ERROR_MODE_SORT_KEYS = {
    "translation": 0,
    "rotation": 1,
    "gripper_close": 2,
}


def _phase_allows_explore_error_mode(phase_key: str, error_mode: str) -> bool:
    phase_key = str(phase_key).strip().lower()
    error_mode = str(error_mode).strip().lower()
    if phase_key in {"pregrasp", "approach"}:
        return error_mode in {"translation", "rotation", "gripper_close"}
    if phase_key in {"transport", "place"}:
        return error_mode in {"translation", "rotation"}
    return False


def _safe_float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _phase_unit_sort_key(unit: dict) -> tuple[int, int, int, int]:
    return (
        int(EXPLORE_PHASE_SORT_KEYS.get(str(unit["phase_key"]), int(phase_key_to_id(str(unit["phase_key"]))))),
        int(unit["phase_instance_idx"]),
        int(unit["phase_bin_id"]),
        int(EXPLORE_ACTIVE_ARM_PATTERN_SORT_KEYS.get(str(unit["active_arm_pattern"]), 999)),
    )


def _explore_unit_sort_key(unit: dict) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(EXPLORE_PHASE_SORT_KEYS.get(str(unit["phase_key"]), int(phase_key_to_id(str(unit["phase_key"]))))),
        int(unit["phase_instance_idx"]),
        int(unit["phase_bin_id"]),
        int(EXPLORE_ACTIVE_ARM_PATTERN_SORT_KEYS.get(str(unit["active_arm_pattern"]), 999)),
        int(EXPLORE_ERROR_MODE_SORT_KEYS.get(str(unit["error_mode"]), int(error_mode_key_to_id(str(unit["error_mode"]))))),
        int(unit["dir_bin_id"]),
        int(unit["mag_bin_id"]),
    )


class FailureAwareStage2Dataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        raw_data_dir: str | None,
        episode_ids: list[int],
        camera_names: list[str],
        norm_stats: dict[str, np.ndarray],
        act_chunk_size: int,
        prefix_steps: int,
        future_offset: int,
        sample_phase_window_len: int,
        start_margin: int,
        failure_mode: str,
        failure_table_path: str,
        failure_phase_bins: int,
        failure_translation_dir_bins: int,
        failure_translation_mag_bins: int,
        failure_rotation_dir_bins: int,
        failure_rotation_mag_bins: int,
        failure_explore_k: int,
        perturb_eef_fail_gain: float = 0.08,
        perturb_rot_max_deg: float = 15.0,
        evac_sample_size: tuple[int, int] | None = None,
        explore_phase_keys: list[str] | None = None,
        explore_error_modes: list[str] | None = None,
        explore_disable_phase_bin_skip: bool = False,
        explore_skip_open_laptop_transport: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.raw_data_dir = None if raw_data_dir is None else str(raw_data_dir).strip()
        self.episode_ids = [int(x) for x in episode_ids]
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.act_chunk_size = int(act_chunk_size)
        self.prefix_steps = int(prefix_steps)
        self.future_offset = int(future_offset)
        self.sample_phase_window_len = int(max(1, sample_phase_window_len))
        self.start_margin = int(max(0, start_margin))
        self.failure_mode = str(failure_mode).strip().lower()
        if self.failure_mode not in {"off", "train", "explore"}:
            raise ValueError(f"Invalid failure_mode={failure_mode!r}, expected off|train|explore")
        self.failure_table_path = str(failure_table_path).strip()
        self.failure_phase_bins = int(max(1, failure_phase_bins))
        self.failure_explore_k = int(max(1, failure_explore_k))
        self.perturb_eef_fail_gain = float(perturb_eef_fail_gain)
        self.perturb_rot_max_deg = float(perturb_rot_max_deg)
        self.evac_sample_size = None if evac_sample_size is None else (int(evac_sample_size[0]), int(evac_sample_size[1]))
        self.explore_phase_keys = None if not explore_phase_keys else {str(x).strip().lower() for x in explore_phase_keys if str(x).strip()}
        self.explore_error_modes = None if not explore_error_modes else {str(x).strip().lower() for x in explore_error_modes if str(x).strip()}
        self.explore_disable_phase_bin_skip = bool(explore_disable_phase_bin_skip)
        self.explore_skip_open_laptop_transport = bool(explore_skip_open_laptop_transport)
        set_failure_param_bins(
            translation_dir_bins=failure_translation_dir_bins,
            translation_mag_bins=failure_translation_mag_bins,
            rotation_dir_bins=failure_rotation_dir_bins,
            rotation_mag_bins=failure_rotation_mag_bins,
        )

        self._phase_scan_cache: dict[tuple[int, int, int, int], dict[int, dict]] = {}
        self._projection_visible_cache: dict[int, dict[int, dict[str, bool]]] = {}
        self._first_stage_len_cache: dict[tuple[int, int], int] = {}
        self._episode_len_cache: dict[int, int] = {}
        self._failure_entries: list[dict] = []
        self._failure_entries_by_mode: dict[str, list[dict]] = {}
        self._failure_mode_probs: dict[str, float] | None = None
        self._explore_units: list[dict] = []
        self._explore_curr_unit_idx = 0
        self._explore_k_local = int(self.failure_explore_k)
        self._explore_trial_count_local = 0
        self._explore_seen_samples: set[tuple[int, int]] = set()
        self._explore_completed_unit_count = 0
        self._explore_unit_candidate_counts: list[int] = []
        self._explore_unit_target_trials: list[int] = []
        self._explore_total_target_samples = 0
        self._init_failure_units()

    def _is_open_laptop_dataset(self) -> bool:
        dataset_key = os.path.normpath(self.dataset_dir).strip().lower()
        return "open_laptop" in dataset_key

    def _should_skip_explore_phase_bin(self, phase_key: str, phase_bin_id: int) -> bool:
        if bool(self.explore_disable_phase_bin_skip):
            return False
        phase_key = str(phase_key).strip().lower()
        phase_bin_id = int(phase_bin_id)
        if phase_key in {"approach", "transport"}:
            return phase_bin_id == 0
        if phase_key in {"pregrasp", "place"}:
            return phase_bin_id == int(self.failure_phase_bins) - 1
        return False

    def _resolve_start_bounds(self, episode_id: int) -> tuple[int, int] | None:
        max_start = self._resolve_max_start(int(episode_id))
        min_start = int(max(0, self.start_margin))
        if max_start < min_start:
            return None
        return min_start, max_start

    def _rebuild_explore_unit_targets(self) -> None:
        if len(self._explore_unit_candidate_counts) == len(self._explore_units):
            self._set_explore_unit_candidate_counts(self._explore_unit_candidate_counts)
            return
        counts: list[int] = []
        for unit in self._explore_units:
            count = int(self._count_unit_local_candidates(unit))
            counts.append(count)
        self._set_explore_unit_candidate_counts(counts)

    def _set_explore_unit_candidate_counts(self, counts: list[int]) -> None:
        counts = [int(max(0, int(x))) for x in list(counts)]
        self._explore_unit_candidate_counts = counts
        self._explore_unit_target_trials = [
            int(min(self._explore_k_local, int(count))) for count in counts
        ]
        self._explore_total_target_samples = int(sum(self._explore_unit_target_trials))

    def _advance_explore_unit(self) -> None:
        if not self._explore_units:
            self._explore_curr_unit_idx = 0
            self._explore_trial_count_local = 0
            self._explore_seen_samples = set()
            return
        self._explore_completed_unit_count += 1
        self._explore_curr_unit_idx = (int(self._explore_curr_unit_idx) + 1) % int(len(self._explore_units))
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()

    def _get_explore_unit_target_trials(self, unit_idx: int | None = None) -> int:
        if unit_idx is None:
            unit_idx = int(self._explore_curr_unit_idx)
        if unit_idx < 0 or unit_idx >= len(self._explore_unit_target_trials):
            return int(self._explore_k_local)
        return int(max(0, self._explore_unit_target_trials[unit_idx]))

    def __len__(self) -> int:
        return len(self.episode_ids)

    def _explore_cache_enabled(self) -> bool:
        value = str(os.environ.get("SMOLVLA_DISABLE_FAILURE_SCAN_CACHE", "")).strip().lower()
        return value not in {"1", "true", "yes", "y"}

    def _explore_cache_dir(self) -> str:
        override = str(os.environ.get("SMOLVLA_FAILURE_SCAN_CACHE_DIR", "")).strip()
        if override:
            return override
        return os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "outputs", "cache", "failure_scan"))

    @staticmethod
    def _file_signature(path: str) -> dict:
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            return {"path": path, "exists": False}
        stat = os.stat(path)
        return {
            "path": path,
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _build_explore_cache_key(self) -> dict:
        bin_cfg = get_failure_param_bins()
        episode_sigs = []
        for episode_id in self.episode_ids:
            episode_id = int(episode_id)
            processed_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
            raw_path = (
                None
                if not self.raw_data_dir
                else os.path.join(self.raw_data_dir, f"episode{episode_id}.hdf5")
            )
            episode_sigs.append(
                {
                    "episode_id": episode_id,
                    "processed": self._file_signature(processed_path),
                    "raw": None if raw_path is None else self._file_signature(raw_path),
                }
            )
        return {
            "version": 1,
            "dataset_dir": os.path.realpath(self.dataset_dir),
            "raw_data_dir": None if not self.raw_data_dir else os.path.realpath(self.raw_data_dir),
            "episode_ids": [int(x) for x in self.episode_ids],
            "episode_sigs": episode_sigs,
            "failure_phase_bins": int(self.failure_phase_bins),
            "failure_translation_dir_bins": int(bin_cfg["translation_dir_bins"]),
            "failure_translation_mag_bins": int(bin_cfg["translation_mag_bins"]),
            "failure_rotation_dir_bins": int(bin_cfg["rotation_dir_bins"]),
            "failure_rotation_mag_bins": int(bin_cfg["rotation_mag_bins"]),
            "future_offset": int(self.future_offset),
            "sample_phase_window_len": int(self.sample_phase_window_len),
            "start_margin": int(self.start_margin),
            "perturb_eef_fail_gain": float(self.perturb_eef_fail_gain),
            "perturb_rot_max_deg": float(self.perturb_rot_max_deg),
            "evac_sample_size": None if self.evac_sample_size is None else list(self.evac_sample_size),
            "explore_phase_keys": None if self.explore_phase_keys is None else sorted(self.explore_phase_keys),
            "explore_error_modes": None if self.explore_error_modes is None else sorted(self.explore_error_modes),
            "explore_disable_phase_bin_skip": bool(self.explore_disable_phase_bin_skip),
            "explore_skip_open_laptop_transport": bool(self.explore_skip_open_laptop_transport),
        }

    def _explore_cache_path_and_key(self) -> tuple[str, dict]:
        key = self._build_explore_cache_key()
        encoded = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        dataset_name = os.path.basename(os.path.normpath(self.dataset_dir)) or "dataset"
        safe_dataset_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset_name)
        cache_path = os.path.join(self._explore_cache_dir(), f"{safe_dataset_name}_{digest}.json")
        return cache_path, key

    def _load_explore_unit_cache(self) -> bool:
        if not self._explore_cache_enabled():
            return False
        try:
            cache_path, key = self._explore_cache_path_and_key()
            if not os.path.isfile(cache_path):
                return False
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("cache_key") != key:
                return False
            units = payload.get("explore_units")
            counts = payload.get("candidate_counts")
            if not isinstance(units, list) or not isinstance(counts, list) or len(units) != len(counts):
                return False
            self._explore_units = [dict(item) for item in units]
            self._set_explore_unit_candidate_counts([int(x) for x in counts])
            return True
        except Exception:
            return False

    def _save_explore_unit_cache(self) -> None:
        if not self._explore_cache_enabled():
            return
        if len(self._explore_unit_candidate_counts) != len(self._explore_units):
            return
        try:
            cache_path, key = self._explore_cache_path_and_key()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            payload = {
                "version": 1,
                "cache_key": key,
                "explore_units": [dict(item) for item in self._explore_units],
                "candidate_counts": [int(x) for x in self._explore_unit_candidate_counts],
            }
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, cache_path)
        except Exception:
            return

    def _init_failure_units(self) -> None:
        if self.failure_mode == "off":
            return
        if self.failure_mode == "explore":
            if self._load_explore_unit_cache():
                return
            units = []
            phase_units = self._collect_local_explore_phase_units()
            bin_cfg = get_failure_param_bins()
            trans_dir_bins = int(bin_cfg["translation_dir_bins"])
            trans_mag_bins = int(bin_cfg["translation_mag_bins"])
            rot_dir_bins = int(bin_cfg["rotation_dir_bins"])
            rot_mag_bins = int(bin_cfg["rotation_mag_bins"])
            def add_if_valid(unit: dict) -> None:
                if self._unit_has_local_candidate(unit):
                    units.append(unit)

            for phase_unit in phase_units:
                phase_key = str(phase_unit["phase_key"])
                phase_instance_idx = int(phase_unit["phase_instance_idx"])
                phase_bin_id = int(phase_unit["phase_bin_id"])
                active_arm_pattern = str(phase_unit["active_arm_pattern"])
                for mode in FAILURE_ERROR_MODES:
                    if self.explore_error_modes is not None and str(mode).strip().lower() not in self.explore_error_modes:
                        continue
                    if not _phase_allows_explore_error_mode(phase_key, mode):
                        continue
                    if mode == "translation":
                        for dir_bin_id in range(trans_dir_bins):
                            for mag_bin_id in range(trans_mag_bins):
                                add_if_valid(
                                    {
                                        "phase_key": phase_key,
                                        "phase_instance_idx": phase_instance_idx,
                                        "phase_bin_id": phase_bin_id,
                                        "error_mode": mode,
                                        "active_arm_pattern": active_arm_pattern,
                                        "dir_bin_id": dir_bin_id,
                                        "mag_bin_id": mag_bin_id,
                                        "weight": 1.0,
                                    }
                                )
                    elif mode == "rotation":
                        for dir_bin_id in range(rot_dir_bins):
                            for mag_bin_id in range(rot_mag_bins):
                                add_if_valid(
                                    {
                                        "phase_key": phase_key,
                                        "phase_instance_idx": phase_instance_idx,
                                        "phase_bin_id": phase_bin_id,
                                        "error_mode": mode,
                                        "active_arm_pattern": active_arm_pattern,
                                        "dir_bin_id": dir_bin_id,
                                        "mag_bin_id": mag_bin_id,
                                        "weight": 1.0,
                                    }
                                )
                    else:
                        add_if_valid(
                            {
                                "phase_key": phase_key,
                                "phase_instance_idx": phase_instance_idx,
                                "phase_bin_id": phase_bin_id,
                                "error_mode": mode,
                                "active_arm_pattern": active_arm_pattern,
                                "dir_bin_id": -1,
                                "mag_bin_id": -1,
                                "weight": 1.0,
                            }
                        )
            self._explore_units = sorted(units, key=_explore_unit_sort_key)
            self._rebuild_explore_unit_targets()
            self._save_explore_unit_cache()
            return

        if not self.failure_table_path:
            raise ValueError("failure_mode=train requires failure_table_path")
        path = self.failure_table_path
        if os.path.isdir(path):
            path = os.path.join(path, "failure_table.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"failure_table not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            table = json.load(f)
        entries = table.get("entries", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"failure_table has no entries: {path}")

        parsed = []
        for item in entries:
            mode = str(item["error_mode"]).strip().lower()
            if mode not in FAILURE_ERROR_MODES:
                continue
            phase_key = str(item["phase_key"]).strip().lower()
            phase_key_to_id(phase_key)
            phase_instance_idx = int(item["phase_instance_idx"])
            phase_bin_id = int(np.clip(int(item["phase_bin_id"]), 0, self.failure_phase_bins - 1))
            active_patterns = [None]
            active_pattern = item.get("active_arm_pattern")
            if active_pattern is not None:
                try:
                    active_pattern = canonicalize_active_arm_pattern_key(active_pattern)
                except ValueError:
                    continue
                if active_pattern == "both":
                    active_patterns = ["left_arm", "right_arm"]
                else:
                    active_patterns = [active_pattern]
            weight = float(item.get("weight", item.get("fail_rate", 1.0)))
            if (not np.isfinite(weight)) or weight <= 0.0:
                weight = 1.0
            dir_bin_id, mag_bin_id = normalize_error_mode_dir_mag_bins(
                mode, int(item.get("dir_bin_id", -1)), int(item.get("mag_bin_id", -1))
            )
            per_pattern_weight = float(weight) / float(len(active_patterns))
            for parsed_active_pattern in active_patterns:
                parsed.append(
                    {
                        "phase_key": phase_key,
                        "phase_instance_idx": phase_instance_idx,
                        "phase_bin_id": phase_bin_id,
                        "error_mode": mode,
                        "active_arm_pattern": parsed_active_pattern,
                        "dir_bin_id": int(dir_bin_id),
                        "mag_bin_id": int(mag_bin_id),
                        "weight": float(per_pattern_weight),
                        "n_trials": int(item.get("n_trials", 0)),
                        "n_recover": int(item.get("n_recover", 0)),
                        "fail_rate": float(item.get("fail_rate", max(1e-6, weight))),
                    }
                )
        if not parsed:
            raise ValueError(f"No valid entries in failure_table: {path}")

        self._failure_entries = parsed
        by_mode: dict[str, list[dict]] = {}
        mode_scores: list[tuple[str, float]] = []
        for mode in FAILURE_ERROR_MODES:
            mode_entries = [x for x in parsed if str(x.get("error_mode", "")) == mode]
            if not mode_entries:
                continue
            by_mode[mode] = mode_entries
            n_trials_sum = 0.0
            n_recover_sum = 0.0
            fail_rate_fallback = []
            for item in mode_entries:
                n_trials = float(item.get("n_trials", 0.0))
                n_recover = float(item.get("n_recover", 0.0))
                if n_trials > 0.0 and np.isfinite(n_trials) and np.isfinite(n_recover):
                    n_trials_sum += n_trials
                    n_recover_sum += n_recover
                else:
                    fail_rate_fallback.append(float(item.get("fail_rate", item.get("weight", 1.0))))
            if n_trials_sum > 0.0:
                recover_rate = n_recover_sum / max(1.0, n_trials_sum)
                score = float(np.clip(1.0 - recover_rate, 1e-6, 1.0))
            elif fail_rate_fallback:
                score = float(np.clip(np.mean(fail_rate_fallback), 1e-6, 1.0))
            else:
                score = 1.0
            mode_scores.append((mode, score))
        self._failure_entries_by_mode = by_mode
        if mode_scores:
            probs = np.asarray([score for _, score in mode_scores], dtype=np.float64)
            if (not np.all(np.isfinite(probs))) or float(np.sum(probs)) <= 1e-12:
                probs = np.ones(len(mode_scores), dtype=np.float64)
            probs = probs / np.sum(probs)
            self._failure_mode_probs = {mode_scores[i][0]: float(probs[i]) for i in range(len(mode_scores))}

    def _collect_local_explore_phase_units(self) -> list[dict]:
        unit_keys = set()
        units = []
        for episode_id in self.episode_ids:
            bounds = self._resolve_start_bounds(int(episode_id))
            if bounds is None:
                continue
            min_start, max_start = bounds
            scan = self._scan_phase_bins(int(episode_id), min_start, max_start)
            projection_visible = self._scan_start_projection_visibility(int(episode_id))
            for ts, meta in scan.items():
                phase_key = str(meta["phase_key"])
                if self.explore_phase_keys is not None and str(phase_key).strip().lower() not in self.explore_phase_keys:
                    continue
                phase_instance_idx = int(meta["phase_instance_idx"])
                phase_bin_id = int(meta["phase_bin_id"])
                raw_active_arm_pattern = str(meta.get("active_arm_pattern", "both")).strip().lower()
                try:
                    active_arm_pattern = canonicalize_active_arm_pattern_key(raw_active_arm_pattern)
                except ValueError:
                    continue
                if self._should_skip_explore_phase_bin(phase_key, phase_bin_id):
                    continue
                if self.explore_skip_open_laptop_transport and self._is_open_laptop_dataset() and phase_key == "transport":
                    continue
                expanded_patterns = (
                    ["left_arm", "right_arm"] if active_arm_pattern == "both" else [active_arm_pattern]
                )
                for expanded_pattern in expanded_patterns:
                    if (
                        projection_visible
                        and not bool(projection_visible.get(int(ts), {}).get(str(expanded_pattern), False))
                    ):
                        continue
                    unit_key = (phase_key, phase_instance_idx, phase_bin_id, expanded_pattern)
                    if unit_key in unit_keys:
                        continue
                    unit_keys.add(unit_key)
                    units.append(
                        {
                            "phase_key": phase_key,
                            "phase_instance_idx": phase_instance_idx,
                            "phase_bin_id": phase_bin_id,
                            "active_arm_pattern": expanded_pattern,
                        }
                    )
        return sorted(units, key=_phase_unit_sort_key)

    def _get_episode_len(self, episode_id: int) -> int:
        episode_id = int(episode_id)
        if episode_id in self._episode_len_cache:
            return int(self._episode_len_cache[episode_id])
        path = os.path.join(self.dataset_dir, f"episode_{int(episode_id)}.hdf5")
        with h5py.File(path, "r") as root:
            episode_len = int(root["/action"].shape[0])
        self._episode_len_cache[episode_id] = int(episode_len)
        return int(episode_len)

    def _resolve_max_start(self, episode_id: int) -> int:
        episode_len = self._get_episode_len(episode_id)
        max_start_future = episode_len - self.future_offset - 1
        return int(max(0, max_start_future))

    def _infer_active_arm_pattern_from_action_window(
        self,
        action: np.ndarray,
        ts: int,
        joint_delta_thresh: float = 0.02,
        gripper_delta_thresh: float = 0.05,
    ) -> str:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim != 2 or action_array.shape[0] <= 1 or action_array.shape[1] < 14:
            return "none"

        def bounds(num_steps: int) -> tuple[int, int]:
            if num_steps <= 1:
                return 0, 1
            start = int(np.clip(ts, 0, num_steps - 1))
            end = int(np.clip(start + max(2, self.sample_phase_window_len), start + 1, num_steps))
            return start, end

        def arm_active(arr: np.ndarray) -> bool:
            start, end = bounds(arr.shape[0])
            segment = arr[start:end]
            if segment.shape[0] <= 1:
                return False
            delta = np.diff(segment, axis=0)
            score = float(np.max(np.linalg.norm(delta, axis=1))) if delta.shape[0] > 0 else 0.0
            return bool(score >= joint_delta_thresh)

        def grip_active(arr: np.ndarray) -> bool:
            start, end = bounds(arr.shape[0])
            segment = arr[start:end]
            if segment.shape[0] <= 1:
                return False
            delta = np.diff(segment)
            score = float(np.max(np.abs(delta))) if delta.shape[0] > 0 else 0.0
            return bool(score >= gripper_delta_thresh)

        left_on = arm_active(action_array[:, 0:6]) or grip_active(action_array[:, 6])
        right_on = arm_active(action_array[:, 7:13]) or grip_active(action_array[:, 13])
        if left_on and right_on:
            return "both"
        if left_on:
            return "left_only"
        if right_on:
            return "right_only"
        return "none"

    def _scan_phase_bins(self, episode_id: int, min_start: int, max_start: int) -> dict[int, dict]:
        key = (int(episode_id), int(min_start), int(max_start), int(self.failure_phase_bins))
        if key in self._phase_scan_cache:
            return self._phase_scan_cache[key]

        info: dict[int, dict] = {}
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            action = root["/action"][()]
        left_gripper = np.asarray(action[:, 6], dtype=np.float32).reshape(-1)
        right_gripper = np.asarray(action[:, 13], dtype=np.float32).reshape(-1)
        ts_list = list(range(int(min_start), int(max_start) + 1))
        phase_list = [
            infer_phase_key_from_gt_window(left_gripper[ts:], right_gripper[ts:], self.sample_phase_window_len)
            for ts in ts_list
        ]

        index = 0
        num_steps = len(ts_list)
        phase_instance_cnt: dict[str, int] = {}
        while index < num_steps:
            end = index
            phase_key = str(phase_list[index]).strip().lower()
            while end + 1 < num_steps and str(phase_list[end + 1]).strip().lower() == phase_key:
                end += 1
            segment = ts_list[index : end + 1]
            seg_len = len(segment)
            phase_instance_cnt[phase_key] = int(phase_instance_cnt.get(phase_key, 0)) + 1
            phase_instance_idx = int(phase_instance_cnt[phase_key])
            for offset, ts in enumerate(segment):
                if seg_len <= 1:
                    bin_id = 0
                else:
                    progress = float(offset) / float(seg_len - 1)
                    bin_id = int(np.floor(progress * float(self.failure_phase_bins)))
                    bin_id = int(np.clip(bin_id, 0, self.failure_phase_bins - 1))
                active_pattern = self._infer_active_arm_pattern_from_action_window(action, ts)
                info[int(ts)] = {
                    "phase_key": phase_key,
                    "phase_instance_idx": phase_instance_idx,
                    "phase_bin_id": bin_id,
                    "seg_start": int(segment[0]),
                    "seg_end": int(segment[-1]),
                    "active_arm_pattern": active_pattern,
                }
            index = end + 1

        self._phase_scan_cache[key] = info
        return info

    @staticmethod
    def _pose_wxyz_to_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R

        pose = np.asarray(pose_wxyz, dtype=np.float32).reshape(7,)
        mat = np.eye(4, dtype=np.float32)
        q_wxyz = pose[3:7]
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
        mat[:3, :3] = R.from_quat(q_xyzw).as_matrix().astype(np.float32)
        mat[:3, 3] = pose[:3]
        return mat

    @staticmethod
    def _evac_gripper_keypoints_world(pose_wxyz: np.ndarray) -> np.ndarray:
        # Match EVAC get_traj(): pose_mat @ Gripper2EEFCvt @ EndEffectorPts.
        end_effector_pts = np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 1.0], [0.0, 0.1, 0.0, 1.0], [0.0, 0.0, 0.1, 1.0]],
            dtype=np.float32,
        ).T
        gripper_to_eef = np.asarray(
            [[1.0, 0.0, 0.0, 0.085], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        pts = FailureAwareStage2Dataset._pose_wxyz_to_matrix(pose_wxyz) @ gripper_to_eef @ end_effector_pts
        return pts[:3, :].T.astype(np.float32)

    @staticmethod
    def _project_point_in_image(point_3d: np.ndarray, intrinsic: np.ndarray, w2c: np.ndarray, width: int, height: int) -> bool:
        point_homo = np.append(np.asarray(point_3d, dtype=np.float32).reshape(3,), 1.0).astype(np.float32)
        w2c_arr = np.asarray(w2c, dtype=np.float32)
        if w2c_arr.shape == (3, 4):
            point_cam = w2c_arr @ point_homo
        elif w2c_arr.shape == (4, 4):
            point_cam = (w2c_arr @ point_homo)[:3]
        else:
            return False
        if point_cam.shape[0] < 3 or (not np.all(np.isfinite(point_cam))) or float(point_cam[2]) <= 1e-6:
            return False
        point_img = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3) @ point_cam[:3]
        if (not np.all(np.isfinite(point_img))) or abs(float(point_img[2])) <= 1e-6:
            return False
        u = float(point_img[0] / point_img[2])
        v = float(point_img[1] / point_img[2])
        return bool(0.0 <= u < float(width) and 0.0 <= v < float(height))

    @staticmethod
    def _evac_gripper_projection_visible(pose_wxyz: np.ndarray, intrinsic: np.ndarray, w2c: np.ndarray, width: int, height: int) -> bool:
        pts_world = FailureAwareStage2Dataset._evac_gripper_keypoints_world(pose_wxyz)
        return all(
            FailureAwareStage2Dataset._project_point_in_image(pt, intrinsic, w2c, width, height)
            for pt in pts_world
        )

    def _scan_start_projection_visibility(self, episode_id: int) -> dict[int, dict[str, bool | np.ndarray]]:
        episode_id = int(episode_id)
        if episode_id in self._projection_visible_cache:
            return self._projection_visible_cache[episode_id]
        raw_path = None
        if self.raw_data_dir:
            raw_path = os.path.join(self.raw_data_dir, f"episode{episode_id}.hdf5")
        if not raw_path or not os.path.isfile(raw_path):
            self._projection_visible_cache[episode_id] = {}
            return {}
        try:
            with h5py.File(raw_path, "r") as root:
                left_endpose = root["endpose/left_endpose"][()].astype(np.float32)
                right_endpose = root["endpose/right_endpose"][()].astype(np.float32)
                intrinsic = root["observation/head_camera/intrinsic_cv"][0].astype(np.float32)
                ext_cv = root["observation/head_camera/extrinsic_cv"][0].astype(np.float32)
                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :] = ext_cv
                rgb0 = root["observation/head_camera/rgb"][0]
                native_height, native_width = self._resolve_raw_rgb_hw(rgb0)
                if self.evac_sample_size is None:
                    height, width = int(native_height), int(native_width)
                else:
                    height, width = int(self.evac_sample_size[0]), int(self.evac_sample_size[1])
                intrinsic_scaled = intrinsic.copy()
                intrinsic_scaled[0, 0] *= float(width) / float(native_width)
                intrinsic_scaled[0, 2] *= float(width) / float(native_width)
                intrinsic_scaled[1, 1] *= float(height) / float(native_height)
                intrinsic_scaled[1, 2] *= float(height) / float(native_height)
        except Exception:
            self._projection_visible_cache[episode_id] = {}
            return {}
        visible = {}
        n = int(min(left_endpose.shape[0], right_endpose.shape[0]))
        for ts in range(n):
            visible[int(ts)] = {
                "left_arm": self._evac_gripper_projection_visible(left_endpose[ts], intrinsic_scaled, w2c, width, height),
                "right_arm": self._evac_gripper_projection_visible(right_endpose[ts], intrinsic_scaled, w2c, width, height),
                "_left_pose": left_endpose[ts].copy(),
                "_right_pose": right_endpose[ts].copy(),
                "_intrinsic": intrinsic_scaled,
                "_w2c": w2c,
                "_width": int(width),
                "_height": int(height),
            }
        self._projection_visible_cache[episode_id] = visible
        return visible

    @staticmethod
    def _resolve_raw_rgb_hw(rgb_frame) -> tuple[int, int]:
        if getattr(rgb_frame, "ndim", 0) >= 2:
            return int(rgb_frame.shape[0]), int(rgb_frame.shape[1])
        try:
            import cv2

            decoded = cv2.imdecode(np.frombuffer(bytes(rgb_frame), np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None and decoded.size > 0:
                return int(decoded.shape[0]), int(decoded.shape[1])
        except Exception:
            pass
        return 240, 320

    def _unit_projection_visible(self, projection_visible: dict, ts: int, unit: dict) -> bool:
        if not projection_visible:
            return not bool(self.raw_data_dir)
        item = projection_visible.get(int(ts), {})
        active_arm = str(unit.get("active_arm_pattern", "left_arm"))
        if active_arm not in {"left_arm", "right_arm"}:
            return False
        if not bool(item.get(active_arm, False)):
            return False

        mode = str(unit.get("error_mode", "")).strip().lower()
        if mode not in {"translation", "rotation"}:
            return True

        intrinsic = item.get("_intrinsic")
        w2c = item.get("_w2c")
        width = item.get("_width")
        height = item.get("_height")
        left_pose = item.get("_left_pose")
        right_pose = item.get("_right_pose")
        if left_pose is None or right_pose is None or intrinsic is None or w2c is None or width is None or height is None:
            return True

        target_left_pose = np.asarray(left_pose, dtype=np.float32).copy()
        target_right_pose = np.asarray(right_pose, dtype=np.float32).copy()
        active_pose = target_left_pose if active_arm == "left_arm" else target_right_pose
        bin_cfg = get_failure_param_bins()
        dir_bin_id = int(unit.get("dir_bin_id", -1))
        mag_bin_id = int(unit.get("mag_bin_id", -1))
        if mode == "translation":
            n_dir = int(max(1, bin_cfg["translation_dir_bins"]))
            n_mag = int(max(1, bin_cfg["translation_mag_bins"]))
            dir_bin_id = int(np.clip(dir_bin_id, 0, n_dir - 1))
            mag_bin_id = int(np.clip(mag_bin_id, 0, n_mag - 1))
            fail_gain, _ = _mag_value_from_bin(self.perturb_eef_fail_gain, mag_bin_id, n_mag)
            direction, _ = _translation_dir_from_bin(active_pose[3:7], dir_bin_id)
            active_pose[:3] = active_pose[:3] + float(fail_gain) * np.asarray(direction, dtype=np.float32)
        elif mode == "rotation":
            from scipy.spatial.transform import Rotation as R

            n_dir = int(max(1, bin_cfg["rotation_dir_bins"]))
            n_mag = int(max(1, bin_cfg["rotation_mag_bins"]))
            dir_bin_id = int(np.clip(dir_bin_id, 0, n_dir - 1))
            mag_bin_id = int(np.clip(mag_bin_id, 0, n_mag - 1))
            angle_max_deg, _ = _mag_value_from_bin(self.perturb_rot_max_deg, mag_bin_id, n_mag)
            rot_sign, _ = _rotation_sign_from_bin(dir_bin_id, n_dir)
            axis, _ = _rotation_axis_from_bin(active_pose[3:7], dir_bin_id)
            q_xyzw = np.array([active_pose[4], active_pose[5], active_pose[6], active_pose[3]], dtype=np.float32)
            q_new = (
                R.from_rotvec(np.asarray(axis, dtype=np.float32) * np.deg2rad(float(angle_max_deg) * float(rot_sign)))
                * R.from_quat(q_xyzw)
            ).as_quat()
            active_pose[3:7] = np.array([q_new[3], q_new[0], q_new[1], q_new[2]], dtype=np.float32)

        intrinsic_arr = np.asarray(intrinsic, dtype=np.float32)
        w2c_arr = np.asarray(w2c, dtype=np.float32)
        active_target_pose = target_left_pose if active_arm == "left_arm" else target_right_pose
        return bool(self._evac_gripper_projection_visible(active_target_pose, intrinsic_arr, w2c_arr, int(width), int(height)))

    def _select_failure_unit(self) -> tuple[dict, int] | None:
        if self.failure_mode == "explore":
            if not self._explore_units:
                raise RuntimeError("explore mode has no units")
            unit_idx = int(self._explore_curr_unit_idx) % int(len(self._explore_units))
            if self._get_explore_unit_target_trials(unit_idx) <= 0:
                self._advance_explore_unit()
                unit_idx = int(self._explore_curr_unit_idx) % int(len(self._explore_units))
            unit = dict(self._explore_units[unit_idx])
            unit["sampled_mode_prob"] = None
            unit["sampled_entry_prob_within_mode"] = None
            unit["sampled_unit_prob"] = None
            return unit, unit_idx
        if self.failure_mode != "train":
            return None
        if not self._failure_entries:
            raise RuntimeError("train mode has empty failure entries")

        mode_keys = []
        mode_probs = []
        for mode in FAILURE_ERROR_MODES:
            if mode in self._failure_entries_by_mode and self._failure_entries_by_mode[mode]:
                mode_keys.append(mode)
                mode_probs.append(float((self._failure_mode_probs or {}).get(mode, 0.0)))
        if not mode_keys:
            raise RuntimeError("train mode has no failure entries grouped by mode")

        mode_probs_array = np.asarray(mode_probs, dtype=np.float64)
        if (not np.all(np.isfinite(mode_probs_array))) or float(np.sum(mode_probs_array)) <= 1e-12:
            mode_probs_array = np.ones(len(mode_keys), dtype=np.float64)
        mode_probs_array = mode_probs_array / np.sum(mode_probs_array)
        mode_idx = int(np.random.choice(len(mode_keys), p=mode_probs_array))
        picked_mode = str(mode_keys[mode_idx])
        picked_mode_prob = float(mode_probs_array[mode_idx])
        entries = list(self._failure_entries_by_mode[picked_mode])
        weights = np.asarray([float(x.get("weight", 1.0)) for x in entries], dtype=np.float64)
        if (not np.all(np.isfinite(weights))) or float(np.sum(weights)) <= 1e-12:
            weights = np.ones(len(entries), dtype=np.float64)
        weights = weights / np.sum(weights)
        local_idx = int(np.random.choice(len(entries), p=weights))
        picked = dict(entries[local_idx])
        picked["sampled_mode_prob"] = picked_mode_prob
        picked["sampled_entry_prob_within_mode"] = float(weights[local_idx])
        picked["sampled_unit_prob"] = float(picked_mode_prob * float(weights[local_idx]))
        global_idx = int(self._failure_entries.index(entries[local_idx]))
        return picked, global_idx

    def _sample_start_ts_from_failure_unit(self, base_index: int) -> dict | None:
        output = self._select_failure_unit()
        if output is None:
            return None
        unit, unit_idx = output
        num_episodes = len(self.episode_ids)
        seen_samples = self._explore_seen_samples if self.failure_mode == "explore" else set()
        for offset in range(num_episodes):
            episode_id = int(self.episode_ids[(int(base_index) + offset) % num_episodes])
            bounds = self._resolve_start_bounds(episode_id)
            if bounds is None:
                continue
            min_start, max_start = bounds
            scan = self._scan_phase_bins(episode_id, min_start, max_start)
            projection_visible = self._scan_start_projection_visibility(episode_id)
            candidates = []
            unseen_candidates = []
            for ts, meta in scan.items():
                if str(meta["phase_key"]) != str(unit["phase_key"]):
                    continue
                target_phase_instance = unit.get("phase_instance_idx")
                if target_phase_instance is not None and int(meta.get("phase_instance_idx", -1)) != int(target_phase_instance):
                    continue
                if int(meta["phase_bin_id"]) != int(unit["phase_bin_id"]):
                    continue
                target_pattern = unit.get("active_arm_pattern")
                meta_pattern = None
                if target_pattern is not None:
                    meta_pattern = str(meta.get("active_arm_pattern", "both")).strip().lower()
                    try:
                        meta_pattern = canonicalize_active_arm_pattern_key(meta_pattern)
                    except ValueError:
                        continue
                    if meta_pattern != "both" and str(meta_pattern) != str(target_pattern):
                        continue
                    if not self._unit_projection_visible(projection_visible, int(ts), unit):
                        continue
                candidate_ts = int(ts)
                candidates.append(candidate_ts)
                if (episode_id, candidate_ts) not in seen_samples:
                    unseen_candidates.append(candidate_ts)
            if not candidates:
                continue
            choose_from = unseen_candidates if unseen_candidates else candidates
            ts = int(np.random.choice(choose_from))
            meta = scan.get(ts, {})
            meta_pattern = str(meta.get("active_arm_pattern", "both")).strip().lower()
            try:
                meta_pattern = canonicalize_active_arm_pattern_key(meta_pattern)
            except ValueError:
                meta_pattern = "both"
            target_pattern = unit.get("active_arm_pattern")
            if target_pattern is None:
                if meta_pattern == "both":
                    actual_pattern = str(np.random.choice(["left_arm", "right_arm"]))
                else:
                    actual_pattern = str(meta_pattern)
            else:
                actual_pattern = str(target_pattern)
            return {
                "episode_id": episode_id,
                "start_ts": ts,
                "seg_start": int(meta.get("seg_start", -1)),
                "seg_end": int(meta.get("seg_end", -1)),
                "phase_key": str(meta.get("phase_key", unit["phase_key"])),
                "phase_instance_idx": int(meta.get("phase_instance_idx", -1)),
                "phase_bin_id": int(meta.get("phase_bin_id", unit["phase_bin_id"])),
                "active_arm_pattern": str(actual_pattern),
                "original_active_arm_pattern": str(meta_pattern),
                "error_mode": str(unit["error_mode"]),
                "dir_bin_id": int(unit.get("dir_bin_id", -1)),
                "mag_bin_id": int(unit.get("mag_bin_id", -1)),
                "explore_unit_idx": int(unit_idx) if self.failure_mode == "explore" else -1,
                "sampled_mode_prob": unit.get("sampled_mode_prob", None),
                "sampled_entry_prob_within_mode": unit.get("sampled_entry_prob_within_mode", None),
                "sampled_unit_prob": unit.get("sampled_unit_prob", None),
            }
        raise RuntimeError(f"Failure unit has no local candidate: {unit} (dataset={self.dataset_dir})")

    def _iter_unit_candidate_ts(self, unit: dict, base_index: int = 0):
        num_episodes = len(self.episode_ids)
        for offset in range(num_episodes):
            episode_id = int(self.episode_ids[(int(base_index) + offset) % num_episodes])
            bounds = self._resolve_start_bounds(episode_id)
            if bounds is None:
                continue
            min_start, max_start = bounds
            scan = self._scan_phase_bins(episode_id, min_start, max_start)
            projection_visible = self._scan_start_projection_visibility(episode_id)
            for ts, meta in scan.items():
                if str(meta["phase_key"]) != str(unit["phase_key"]):
                    continue
                target_phase_instance = unit.get("phase_instance_idx")
                if target_phase_instance is not None and int(meta.get("phase_instance_idx", -1)) != int(target_phase_instance):
                    continue
                if int(meta["phase_bin_id"]) != int(unit["phase_bin_id"]):
                    continue
                target_pattern = unit.get("active_arm_pattern")
                if target_pattern is not None:
                    meta_pattern = str(meta.get("active_arm_pattern", "both")).strip().lower()
                    try:
                        meta_pattern = canonicalize_active_arm_pattern_key(meta_pattern)
                    except ValueError:
                        continue
                    if meta_pattern != "both" and str(meta_pattern) != str(target_pattern):
                        continue
                    if not self._unit_projection_visible(projection_visible, int(ts), unit):
                        continue
                yield episode_id, int(ts), meta

    def _unit_has_local_candidate(self, unit: dict) -> bool:
        return next(self._iter_unit_candidate_ts(unit, base_index=0), None) is not None

    def _count_unit_local_candidates(self, unit: dict) -> int:
        return int(sum(1 for _ in self._iter_unit_candidate_ts(unit, base_index=0)))

    def record_explore_trial(self, unit_idx: int, episode_id: int, start_ts: int) -> None:
        if self.failure_mode != "explore" or not self._explore_units:
            return
        if int(unit_idx) != int(self._explore_curr_unit_idx):
            return
        target_trials = int(self._get_explore_unit_target_trials(unit_idx))
        if target_trials <= 0:
            self._advance_explore_unit()
            return
        sample_uid = (int(episode_id), int(start_ts))
        if sample_uid in self._explore_seen_samples:
            return
        self._explore_seen_samples.add(sample_uid)
        self._explore_trial_count_local += 1
        if int(self._explore_trial_count_local) >= target_trials:
            self._advance_explore_unit()

    def set_explore_unit_idx(self, unit_idx: int) -> None:
        if not self._explore_units:
            self._explore_curr_unit_idx = 0
            return
        self._explore_curr_unit_idx = int(np.clip(int(unit_idx), 0, len(self._explore_units) - 1))
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()
        self._explore_completed_unit_count = 0

    def set_explore_units(self, units: list[dict] | None) -> None:
        self._explore_units = [dict(item) for item in list(units or [])]
        self._explore_unit_candidate_counts = []
        self._rebuild_explore_unit_targets()
        if not self._explore_units:
            self._explore_curr_unit_idx = 0
        else:
            self._explore_curr_unit_idx = int(np.clip(int(self._explore_curr_unit_idx), 0, len(self._explore_units) - 1))
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()
        self._explore_completed_unit_count = 0

    def set_explore_local_k(self, k_local: int) -> None:
        self._explore_k_local = int(max(1, int(k_local)))
        self._rebuild_explore_unit_targets()
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()
        self._explore_completed_unit_count = 0

    def set_episode_ids(self, episode_ids: list[int] | None) -> None:
        self.episode_ids = [int(x) for x in list(episode_ids or [])]
        self._phase_scan_cache = {}
        self._projection_visible_cache = {}
        self._first_stage_len_cache = {}
        self._episode_len_cache = {}
        self._failure_entries = []
        self._failure_entries_by_mode = {}
        self._failure_mode_probs = None
        self._explore_units = []
        self._explore_curr_unit_idx = 0
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()
        self._explore_completed_unit_count = 0
        self._explore_unit_candidate_counts = []
        self._explore_unit_target_trials = []
        self._explore_total_target_samples = 0
        self._init_failure_units()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        if self.failure_mode in {"train", "explore"}:
            sampled = self._sample_start_ts_from_failure_unit(index)
            assert sampled is not None
            episode_id = int(sampled["episode_id"])
            start_ts = int(sampled["start_ts"])
            sampled_phase_key = str(sampled.get("phase_key", "approach"))
            sampled_phase_instance_id = int(sampled.get("phase_instance_idx", -1))
            sampled_phase_bin_id = int(sampled.get("phase_bin_id", -1))
            forced_error_mode_id = error_mode_key_to_id(str(sampled.get("error_mode", "translation")))
            sampled_active_arm_pattern_id = active_arm_pattern_key_to_id(str(sampled.get("active_arm_pattern", "left_arm")))
            original_active_arm_pattern_id = active_arm_pattern_key_to_id(
                str(sampled.get("original_active_arm_pattern", sampled.get("active_arm_pattern", "left_arm")))
            )
            forced_dir_bin_id = int(sampled.get("dir_bin_id", -1))
            forced_mag_bin_id = int(sampled.get("mag_bin_id", -1))
            sampled_explore_unit_idx = int(sampled.get("explore_unit_idx", -1))
            sampled_mode_prob = _safe_float_or_nan(sampled.get("sampled_mode_prob", np.nan))
            sampled_entry_prob_within_mode = _safe_float_or_nan(sampled.get("sampled_entry_prob_within_mode", np.nan))
            sampled_unit_prob = _safe_float_or_nan(sampled.get("sampled_unit_prob", np.nan))
        else:
            episode_id = int(self.episode_ids[index])
            bounds = self._resolve_start_bounds(episode_id)
            if bounds is None:
                raise RuntimeError(f"No valid start bounds for episode_id={episode_id} in dataset={self.dataset_dir}")
            min_start, max_start = bounds
            start_ts = int(np.random.randint(min_start, max_start + 1))
            sampled_phase_key = infer_phase_key_from_gt_window(np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), 1)
            sampled_phase_instance_id = -1
            sampled_phase_bin_id = -1
            forced_error_mode_id = -1
            sampled_active_arm_pattern_id = -1
            original_active_arm_pattern_id = -1
            forced_dir_bin_id = -1
            forced_mag_bin_id = -1
            sampled_explore_unit_idx = -1
            sampled_mode_prob = np.nan
            sampled_entry_prob_within_mode = np.nan
            sampled_unit_prob = np.nan

        sample = load_processed_episode_window(
            dataset_dir=self.dataset_dir,
            episode_id=episode_id,
            camera_names=self.camera_names,
            norm_stats=self.norm_stats,
            act_chunk_size=self.act_chunk_size,
            prefix_steps=self.prefix_steps,
            future_offset=self.future_offset,
            start_ts=start_ts,
        )

        if self.failure_mode not in {"train", "explore"}:
            action_chunk_raw = sample["act_action_chunk_raw"].numpy()
            sampled_phase_key = infer_phase_key_from_gt_window(
                action_chunk_raw[:, 6],
                action_chunk_raw[:, 13],
                self.sample_phase_window_len,
            )
            sampled_phase_instance_id = 1
            if sample["ep_len"] <= 1:
                sampled_phase_bin_id = 0
            else:
                progress = float(start_ts) / float(max(1, int(sample["ep_len"]) - 1))
                sampled_phase_bin_id = int(np.floor(progress * float(self.failure_phase_bins)))
                sampled_phase_bin_id = int(np.clip(sampled_phase_bin_id, 0, self.failure_phase_bins - 1))

        sample["sampled_phase_id"] = torch.tensor(phase_key_to_id(sampled_phase_key), dtype=torch.int64)
        sample["sampled_phase_bin_id"] = torch.tensor(sampled_phase_bin_id, dtype=torch.int64)
        sample["sampled_phase_instance_id"] = torch.tensor(sampled_phase_instance_id, dtype=torch.int64)
        sample["forced_error_mode_id"] = torch.tensor(forced_error_mode_id, dtype=torch.int64)
        sample["sampled_active_arm_pattern_id"] = torch.tensor(sampled_active_arm_pattern_id, dtype=torch.int64)
        sample["original_active_arm_pattern_id"] = torch.tensor(original_active_arm_pattern_id, dtype=torch.int64)
        sample["forced_dir_bin_id"] = torch.tensor(forced_dir_bin_id, dtype=torch.int64)
        sample["forced_mag_bin_id"] = torch.tensor(forced_mag_bin_id, dtype=torch.int64)
        sample["sampled_explore_unit_idx"] = torch.tensor(sampled_explore_unit_idx, dtype=torch.int64)
        sample["sampled_mode_prob"] = torch.tensor(sampled_mode_prob, dtype=torch.float32)
        sample["sampled_entry_prob_within_mode"] = torch.tensor(sampled_entry_prob_within_mode, dtype=torch.float32)
        sample["sampled_unit_prob"] = torch.tensor(sampled_unit_prob, dtype=torch.float32)
        return sample


def build_failure_table_dataset(
    dataset_dir: str,
    num_episodes: int,
    camera_names: list[str],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
    sample_phase_window_len: int,
    start_margin: int,
    failure_mode: str,
    failure_table_path: str,
    failure_phase_bins: int,
    failure_translation_dir_bins: int,
    failure_translation_mag_bins: int,
    failure_rotation_dir_bins: int,
    failure_rotation_mag_bins: int,
    failure_explore_k: int,
    raw_data_dir: str | None = None,
    perturb_eef_fail_gain: float = 0.08,
    perturb_rot_max_deg: float = 15.0,
    evac_sample_size: tuple[int, int] | None = None,
    explore_phase_keys: list[str] | None = None,
    explore_error_modes: list[str] | None = None,
    explore_disable_phase_bin_skip: bool = False,
    explore_skip_open_laptop_transport: bool = False,
) -> tuple[FailureAwareStage2Dataset, dict[str, np.ndarray]]:
    stats = get_norm_stats(dataset_dir, num_episodes)
    valid_ids = list_valid_episode_ids(dataset_dir, num_episodes, future_offset)
    dataset = FailureAwareStage2Dataset(
        dataset_dir=dataset_dir,
        raw_data_dir=raw_data_dir,
        episode_ids=valid_ids,
        camera_names=camera_names,
        norm_stats=stats,
        act_chunk_size=act_chunk_size,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
        sample_phase_window_len=sample_phase_window_len,
        start_margin=start_margin,
        failure_mode=failure_mode,
        failure_table_path=failure_table_path,
        failure_phase_bins=failure_phase_bins,
        failure_translation_dir_bins=failure_translation_dir_bins,
        failure_translation_mag_bins=failure_translation_mag_bins,
        failure_rotation_dir_bins=failure_rotation_dir_bins,
        failure_rotation_mag_bins=failure_rotation_mag_bins,
        failure_explore_k=failure_explore_k,
        perturb_eef_fail_gain=perturb_eef_fail_gain,
        perturb_rot_max_deg=perturb_rot_max_deg,
        evac_sample_size=evac_sample_size,
        explore_phase_keys=explore_phase_keys,
        explore_error_modes=explore_error_modes,
        explore_disable_phase_bin_skip=explore_disable_phase_bin_skip,
        explore_skip_open_laptop_transport=explore_skip_open_laptop_transport,
    )
    return dataset, stats
