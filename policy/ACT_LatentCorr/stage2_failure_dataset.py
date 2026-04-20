from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .act_aligned_pkg.phase_utils import infer_phase_key_from_gt_window
from .failure_utils import (
    ACTIVE_ARM_PATTERN_KEYS,
    ERROR_MODE_KEYS,
    active_arm_pattern_key_to_id,
    error_mode_key_to_id,
    get_failure_param_bins,
    normalize_error_mode_dir_mag_bins,
    phase_key_to_id,
    set_failure_param_bins,
)
from .utils_latent import get_norm_stats, list_valid_episode_ids, load_processed_episode_window
from .utils_multitask_latent import MultiTaskSpec, get_multitask_norm_stats

_FAILURE_ERROR_MODES = tuple(ERROR_MODE_KEYS)
_ACTIVE_ARM_PATTERNS = set(ACTIVE_ARM_PATTERN_KEYS)


def _safe_float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class FailureAwareStage2Dataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        episode_ids: list[int],
        camera_names: list[str],
        norm_stats: dict[str, np.ndarray],
        act_chunk_size: int,
        prefix_steps: int,
        future_offset: int,
        sample_phase_window_len: int = 16,
        sample_skip_head_ratio: float | None = None,
        start_margin: int = 0,
        failure_mode: str = "off",
        failure_table_path: str = "",
        failure_phase_bins: int = 5,
        failure_translation_dir_bins: int = 6,
        failure_translation_mag_bins: int = 3,
        failure_rotation_dir_bins: int = 6,
        failure_rotation_mag_bins: int = 3,
        failure_explore_k: int = 1,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.episode_ids = [int(x) for x in episode_ids]
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.act_chunk_size = int(act_chunk_size)
        self.prefix_steps = int(prefix_steps)
        self.future_offset = int(future_offset)
        self.sample_phase_window_len = int(max(1, sample_phase_window_len))
        self.sample_skip_head_ratio = None if sample_skip_head_ratio is None else float(
            np.clip(sample_skip_head_ratio, 0.0, 0.99)
        )
        self.start_margin = int(max(0, start_margin))
        self.failure_mode = str(failure_mode).strip().lower()
        if self.failure_mode not in {"off", "train", "explore"}:
            raise ValueError(f"Invalid failure_mode={failure_mode!r}, expected off|train|explore")
        self.failure_table_path = str(failure_table_path).strip()
        self.failure_phase_bins = int(max(1, failure_phase_bins))
        self.failure_explore_k = int(max(1, failure_explore_k))
        self.failure_skip_approach_bins = 0
        if self.failure_mode == "explore":
            skip_bins = int(np.floor(float(self.sample_skip_head_ratio or 0.0) * float(self.failure_phase_bins)))
            self.failure_skip_approach_bins = int(np.clip(skip_bins, 0, self.failure_phase_bins))
        set_failure_param_bins(
            translation_dir_bins=failure_translation_dir_bins,
            translation_mag_bins=failure_translation_mag_bins,
            rotation_dir_bins=failure_rotation_dir_bins,
            rotation_mag_bins=failure_rotation_mag_bins,
        )

        self._phase_scan_cache: dict[tuple[int, int, int, int], dict[int, dict]] = {}
        self._first_stage_len_cache: dict[tuple[int, int], int] = {}
        self._failure_entries: list[dict] = []
        self._failure_entries_by_mode: dict[str, list[dict]] = {}
        self._failure_mode_probs: dict[str, float] | None = None
        self._explore_units: list[dict] = []
        self._explore_curr_unit_idx = 0
        self._explore_k_local = int(self.failure_explore_k)
        self._explore_trial_count_local = 0
        self._explore_seen_samples: set[tuple[int, int]] = set()
        self._explore_completed_unit_count = 0
        self._init_failure_units()

    def __len__(self) -> int:
        return len(self.episode_ids)

    def _init_failure_units(self) -> None:
        if self.failure_mode == "off":
            return
        if self.failure_mode == "explore":
            units = []
            phase_units = self._collect_local_explore_phase_units()
            bin_cfg = get_failure_param_bins()
            trans_dir_bins = int(max(1, bin_cfg["translation_dir_bins"]))
            trans_mag_bins = int(max(1, bin_cfg["translation_mag_bins"]))
            rot_dir_bins = int(max(1, bin_cfg["rotation_dir_bins"]))
            rot_mag_bins = int(max(1, bin_cfg["rotation_mag_bins"]))
            for phase_unit in phase_units:
                phase_key = str(phase_unit["phase_key"])
                phase_instance_idx = int(phase_unit["phase_instance_idx"])
                bin_id = int(phase_unit["phase_bin_id"])
                for mode in _FAILURE_ERROR_MODES:
                    if phase_key == "transport" and mode == "gripper_close":
                        continue
                    if mode == "translation":
                        for dir_bin_id in range(trans_dir_bins):
                            for mag_bin_id in range(trans_mag_bins):
                                units.append(
                                    {
                                        "phase_key": phase_key,
                                        "phase_instance_idx": phase_instance_idx,
                                        "phase_bin_id": bin_id,
                                        "error_mode": mode,
                                        "active_arm_pattern": None,
                                        "dir_bin_id": int(dir_bin_id),
                                        "mag_bin_id": int(mag_bin_id),
                                        "weight": 1.0,
                                    }
                                )
                    elif mode == "rotation":
                        for dir_bin_id in range(rot_dir_bins):
                            for mag_bin_id in range(rot_mag_bins):
                                units.append(
                                    {
                                        "phase_key": phase_key,
                                        "phase_instance_idx": phase_instance_idx,
                                        "phase_bin_id": bin_id,
                                        "error_mode": mode,
                                        "active_arm_pattern": None,
                                        "dir_bin_id": int(dir_bin_id),
                                        "mag_bin_id": int(mag_bin_id),
                                        "weight": 1.0,
                                    }
                                )
                    else:
                        units.append(
                            {
                                "phase_key": phase_key,
                                "phase_instance_idx": phase_instance_idx,
                                "phase_bin_id": bin_id,
                                "error_mode": mode,
                                "active_arm_pattern": None,
                                "dir_bin_id": -1,
                                "mag_bin_id": -1,
                                "weight": 1.0,
                            }
                        )
            self._explore_units = units
            return
        if not self.failure_table_path:
            raise ValueError("failure_mode=train requires failure_table_path")

        if os.path.isdir(self.failure_table_path):
            path = os.path.join(self.failure_table_path, "failure_table.json")
        else:
            path = self.failure_table_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"failure_table not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            table = json.load(f)
        entries = table.get("entries", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"failure_table has no entries: {path}")

        parsed: list[dict] = []
        for item in entries:
            try:
                mode = str(item["error_mode"]).strip().lower()
                if mode not in _FAILURE_ERROR_MODES:
                    continue
                phase_key = str(item["phase_key"]).strip().lower()
                phase_key_to_id(phase_key)
                phase_instance_idx = int(item["phase_instance_idx"])
                phase_bin_id = int(np.clip(int(item["phase_bin_id"]), 0, self.failure_phase_bins - 1))
                active_pattern = item.get("active_arm_pattern")
                if active_pattern is not None:
                    active_pattern = str(active_pattern).strip().lower()
                    if active_pattern not in _ACTIVE_ARM_PATTERNS:
                        continue
                weight = float(item.get("weight", item.get("fail_rate", 1.0)))
                if (not np.isfinite(weight)) or weight <= 0.0:
                    weight = 1.0
                dir_bin_id, mag_bin_id = normalize_error_mode_dir_mag_bins(
                    mode,
                    int(item.get("dir_bin_id", -1)),
                    int(item.get("mag_bin_id", -1)),
                )
                parsed.append(
                    {
                        "phase_key": phase_key,
                        "phase_instance_idx": phase_instance_idx,
                        "phase_bin_id": phase_bin_id,
                        "error_mode": mode,
                        "active_arm_pattern": active_pattern,
                        "dir_bin_id": int(dir_bin_id),
                        "mag_bin_id": int(mag_bin_id),
                        "weight": float(weight),
                        "n_trials": int(item.get("n_trials", 0)),
                        "n_recover": int(item.get("n_recover", 0)),
                        "fail_rate": float(item.get("fail_rate", max(1e-6, weight))),
                    }
                )
            except Exception:
                continue

        if not parsed:
            raise ValueError(f"No valid entries in failure_table: {path}")

        self._failure_entries = parsed
        by_mode: dict[str, list[dict]] = {}
        mode_scores: list[tuple[str, float]] = []
        for mode in _FAILURE_ERROR_MODES:
            mode_entries = [x for x in parsed if str(x.get("error_mode", "")) == mode]
            if not mode_entries:
                continue
            by_mode[mode] = mode_entries
            n_trials_sum = 0.0
            n_recover_sum = 0.0
            fail_rate_fallback = []
            for item in mode_entries:
                n_trials_i = float(item.get("n_trials", 0.0))
                n_recover_i = float(item.get("n_recover", 0.0))
                if n_trials_i > 0.0 and np.isfinite(n_trials_i) and np.isfinite(n_recover_i):
                    n_trials_sum += n_trials_i
                    n_recover_sum += n_recover_i
                else:
                    fail_rate_fallback.append(float(item.get("fail_rate", item.get("weight", 1.0))))
            if n_trials_sum > 0.0:
                recover_rate_mode = n_recover_sum / max(1.0, n_trials_sum)
                score = float(np.clip(1.0 - recover_rate_mode, 1e-6, 1.0))
            elif fail_rate_fallback:
                score = float(np.clip(np.mean(fail_rate_fallback), 1e-6, 1.0))
            else:
                score = 1.0
            mode_scores.append((mode, score))
        self._failure_entries_by_mode = by_mode
        if mode_scores:
            probs = np.asarray([float(x[1]) for x in mode_scores], dtype=np.float64)
            if (not np.all(np.isfinite(probs))) or float(np.sum(probs)) <= 1e-12:
                probs = np.ones(len(mode_scores), dtype=np.float64)
            probs = probs / np.sum(probs)
            self._failure_mode_probs = {str(mode_scores[i][0]): float(probs[i]) for i in range(len(mode_scores))}

    def _collect_local_explore_phase_units(self) -> list[dict]:
        unit_keys = set()
        units = []
        for episode_id in self.episode_ids:
            ep = int(episode_id)
            max_start = self._resolve_max_start(ep)
            scan = self._scan_phase_bins(ep, 0, max_start)
            for meta in scan.values():
                phase_key = str(meta["phase_key"])
                phase_instance_idx = int(meta["phase_instance_idx"])
                phase_bin_id = int(meta["phase_bin_id"])
                if phase_key == "approach" and phase_bin_id < int(self.failure_skip_approach_bins):
                    continue
                unit_key = (phase_key, phase_instance_idx, phase_bin_id)
                if unit_key in unit_keys:
                    continue
                unit_keys.add(unit_key)
                units.append(
                    {
                        "phase_key": phase_key,
                        "phase_instance_idx": phase_instance_idx,
                        "phase_bin_id": phase_bin_id,
                    }
                )
        return units

    def _get_episode_len(self, episode_id: int) -> int:
        path = os.path.join(self.dataset_dir, f"episode_{int(episode_id)}.hdf5")
        with h5py.File(path, "r") as root:
            return int(root["/action"].shape[0])

    def _resolve_max_start(self, episode_id: int) -> int:
        ep_len = self._get_episode_len(episode_id)
        max_start_future = ep_len - self.future_offset - 1
        max_start_margin = ep_len - 1 - self.start_margin
        return int(max(0, min(max_start_future, max_start_margin)))

    def _get_first_stage_len(self, episode_id: int, max_start: int) -> int:
        key = (int(episode_id), int(max_start))
        if key in self._first_stage_len_cache:
            return self._first_stage_len_cache[key]

        n_valid = int(max(1, max_start + 1))
        first_stage_len = n_valid
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        try:
            with h5py.File(dataset_path, "r") as root:
                action = root["/action"][()]
            if action.ndim == 2 and action.shape[1] >= 14 and action.shape[0] > 1:
                lg = np.asarray(action[:, 6], dtype=np.float32).reshape(-1)
                rg = np.asarray(action[:, 13], dtype=np.float32).reshape(-1)
                n_scan = min(n_valid, lg.shape[0], rg.shape[0])
                first_non_approach = n_scan
                for ts in range(n_scan):
                    phase_key = infer_phase_key_from_gt_window(
                        lg[ts:],
                        rg[ts:],
                        self.sample_phase_window_len,
                    )
                    if phase_key != "approach":
                        first_non_approach = ts
                        break
                first_stage_len = int(first_non_approach)
        except Exception:
            first_stage_len = n_valid

        first_stage_len = int(np.clip(first_stage_len, 0, n_valid))
        self._first_stage_len_cache[key] = first_stage_len
        return first_stage_len

    def _resolve_min_start(self, episode_id: int, max_start: int) -> int:
        max_start = int(max(0, max_start))
        if self.sample_skip_head_ratio is None:
            return 0
        first_stage_len = self._get_first_stage_len(episode_id, max_start)
        min_start_ratio = int(np.floor(float(first_stage_len) * float(self.sample_skip_head_ratio)))
        return int(np.clip(min_start_ratio, 0, max_start))

    def _infer_active_arm_pattern_from_action_window(
        self,
        action: np.ndarray,
        ts: int,
        joint_delta_thresh: float = 0.02,
        gripper_delta_thresh: float = 0.05,
    ) -> str:
        a = np.asarray(action, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] <= 1 or a.shape[1] < 14:
            return "both"

        def _bounds(n: int) -> tuple[int, int]:
            if n <= 1:
                return 0, 1
            s = int(np.clip(ts, 0, n - 1))
            e = int(np.clip(s + max(2, self.sample_phase_window_len), s + 1, n))
            return s, e

        def _arm_active(arr: np.ndarray) -> bool:
            s, e = _bounds(arr.shape[0])
            seg = arr[s:e]
            if seg.shape[0] <= 1:
                return False
            dq = np.diff(seg, axis=0)
            score = float(np.max(np.linalg.norm(dq, axis=1))) if dq.shape[0] > 0 else 0.0
            return bool(score >= float(joint_delta_thresh))

        def _grip_active(arr: np.ndarray) -> bool:
            s, e = _bounds(arr.shape[0])
            seg = arr[s:e]
            if seg.shape[0] <= 1:
                return False
            dg = np.diff(seg)
            score = float(np.max(np.abs(dg))) if dg.shape[0] > 0 else 0.0
            return bool(score >= float(gripper_delta_thresh))

        left_on = _arm_active(a[:, 0:6]) or _grip_active(a[:, 6])
        right_on = _arm_active(a[:, 7:13]) or _grip_active(a[:, 13])
        if left_on and right_on:
            return "both"
        if left_on:
            return "left_only"
        if right_on:
            return "right_only"
        return "both"

    def _scan_phase_bins(self, episode_id: int, min_start: int, max_start: int) -> dict[int, dict]:
        key = (int(episode_id), int(min_start), int(max_start), int(self.failure_phase_bins))
        if key in self._phase_scan_cache:
            return self._phase_scan_cache[key]

        info: dict[int, dict] = {}
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            action = root["/action"][()]
        lg = np.asarray(action[:, 6], dtype=np.float32).reshape(-1)
        rg = np.asarray(action[:, 13], dtype=np.float32).reshape(-1)
        ts_list = list(range(int(min_start), int(max_start) + 1))
        phase_list = []
        for ts in ts_list:
            phase_list.append(infer_phase_key_from_gt_window(lg[ts:], rg[ts:], self.sample_phase_window_len))

        i = 0
        n = len(ts_list)
        phase_instance_cnt: dict[str, int] = {}
        while i < n:
            j = i
            phase_key = str(phase_list[i]).strip().lower()
            while j + 1 < n and str(phase_list[j + 1]).strip().lower() == phase_key:
                j += 1
            seg = ts_list[i : j + 1]
            seg_len = len(seg)
            phase_instance_cnt[phase_key] = int(phase_instance_cnt.get(phase_key, 0)) + 1
            phase_instance_idx = int(phase_instance_cnt[phase_key])
            for k, ts in enumerate(seg):
                if seg_len <= 1:
                    bin_id = 0
                else:
                    progress = float(k) / float(seg_len - 1)
                    bin_id = int(np.floor(progress * float(self.failure_phase_bins)))
                    bin_id = int(np.clip(bin_id, 0, self.failure_phase_bins - 1))
                active_pattern = self._infer_active_arm_pattern_from_action_window(action, ts)
                info[int(ts)] = {
                    "phase_key": phase_key,
                    "phase_instance_idx": int(phase_instance_idx),
                    "phase_bin_id": int(bin_id),
                    "seg_start": int(seg[0]),
                    "seg_end": int(seg[-1]),
                    "active_arm_pattern": active_pattern,
                }
            i = j + 1
        self._phase_scan_cache[key] = info
        return info

    def _select_failure_unit(self) -> tuple[dict, int] | None:
        if self.failure_mode == "explore":
            if not self._explore_units:
                raise RuntimeError("explore mode has no units")
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
        for mode in _FAILURE_ERROR_MODES:
            if mode in self._failure_entries_by_mode and self._failure_entries_by_mode[mode]:
                mode_keys.append(mode)
                mode_probs.append(float((self._failure_mode_probs or {}).get(mode, 0.0)))
        if not mode_keys:
            raise RuntimeError("train mode has no failure entries grouped by mode")
        mode_probs_arr = np.asarray(mode_probs, dtype=np.float64)
        if (not np.all(np.isfinite(mode_probs_arr))) or float(np.sum(mode_probs_arr)) <= 1e-12:
            mode_probs_arr = np.ones(len(mode_keys), dtype=np.float64)
        mode_probs_arr = mode_probs_arr / np.sum(mode_probs_arr)
        mode_idx = int(np.random.choice(len(mode_keys), p=mode_probs_arr))
        picked_mode = str(mode_keys[mode_idx])
        picked_mode_prob = float(mode_probs_arr[mode_idx])
        entries = list(self._failure_entries_by_mode[picked_mode])
        weights = np.asarray([float(x.get("weight", 1.0)) for x in entries], dtype=np.float64)
        if (not np.all(np.isfinite(weights))) or float(np.sum(weights)) <= 1e-12:
            weights = np.ones(len(entries), dtype=np.float64)
        weights = weights / np.sum(weights)
        idx_local = int(np.random.choice(len(entries), p=weights))
        picked = dict(entries[idx_local])
        picked["sampled_mode_prob"] = picked_mode_prob
        picked["sampled_entry_prob_within_mode"] = float(weights[idx_local])
        picked["sampled_unit_prob"] = float(picked_mode_prob * float(weights[idx_local]))
        idx_global = int(self._failure_entries.index(entries[idx_local]))
        return picked, idx_global

    def _sample_start_ts_from_failure_unit(self, base_index: int) -> dict | None:
        out = self._select_failure_unit()
        if out is None:
            return None
        unit, unit_idx = out
        n_ep = len(self.episode_ids)
        for ofs in range(n_ep):
            ep = int(self.episode_ids[(int(base_index) + ofs) % n_ep])
            max_start = self._resolve_max_start(ep)
            min_start = 0 if self.failure_mode == "explore" else self._resolve_min_start(ep, max_start)
            scan = self._scan_phase_bins(ep, min_start, max_start)
            cand = []
            for ts, meta in scan.items():
                if str(meta["phase_key"]) != str(unit["phase_key"]):
                    continue
                target_phase_inst = unit.get("phase_instance_idx")
                if target_phase_inst is not None and int(meta.get("phase_instance_idx", -1)) != int(target_phase_inst):
                    continue
                if int(meta["phase_bin_id"]) != int(unit["phase_bin_id"]):
                    continue
                target_pat = unit.get("active_arm_pattern")
                if target_pat is not None and str(meta["active_arm_pattern"]) != str(target_pat):
                    continue
                cand.append(int(ts))
            if not cand:
                continue
            ts = int(np.random.choice(cand))
            meta = scan.get(ts, {})
            return {
                "episode_id": int(ep),
                "start_ts": int(ts),
                "seg_start": int(meta.get("seg_start", -1)),
                "seg_end": int(meta.get("seg_end", -1)),
                "phase_key": str(meta.get("phase_key", unit["phase_key"])),
                "phase_instance_idx": int(meta.get("phase_instance_idx", -1)),
                "phase_bin_id": int(meta.get("phase_bin_id", unit["phase_bin_id"])),
                "active_arm_pattern": str(meta.get("active_arm_pattern", "both")),
                "error_mode": str(unit["error_mode"]),
                "dir_bin_id": int(unit.get("dir_bin_id", -1)),
                "mag_bin_id": int(unit.get("mag_bin_id", -1)),
                "explore_unit_idx": int(unit_idx) if self.failure_mode == "explore" else -1,
                "sampled_mode_prob": unit.get("sampled_mode_prob", None),
                "sampled_entry_prob_within_mode": unit.get("sampled_entry_prob_within_mode", None),
                "sampled_unit_prob": unit.get("sampled_unit_prob", None),
            }
        raise RuntimeError(
            "Failure unit has no local candidate: "
            f"{unit} (dataset={self.dataset_dir})"
        )

    def _advance_explore_unit(self, count_as_completed: bool = False) -> None:
        if not self._explore_units:
            return
        if bool(count_as_completed):
            self._explore_completed_unit_count = min(
                int(len(self._explore_units)),
                int(self._explore_completed_unit_count) + 1,
            )
        prev_idx = int(self._explore_curr_unit_idx)
        self._explore_curr_unit_idx = (prev_idx + 1) % int(len(self._explore_units))
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()

    def set_explore_unit_idx(self, unit_idx: int) -> None:
        if not self._explore_units:
            self._explore_curr_unit_idx = 0
            return
        self._explore_curr_unit_idx = int(np.clip(int(unit_idx), 0, len(self._explore_units) - 1))
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()
        self._explore_completed_unit_count = 0

    def set_explore_units(self, units: list[dict] | None) -> None:
        self._explore_units = [dict(x) for x in list(units or [])]
        if not self._explore_units:
            self._explore_curr_unit_idx = 0
        else:
            self._explore_curr_unit_idx = int(
                np.clip(int(self._explore_curr_unit_idx), 0, len(self._explore_units) - 1)
            )
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()
        self._explore_completed_unit_count = 0

    def set_explore_local_k(self, k_local: int) -> None:
        self._explore_k_local = int(max(1, int(k_local)))
        self._explore_trial_count_local = 0
        self._explore_seen_samples = set()
        self._explore_completed_unit_count = 0

    def record_explore_trial(self, unit_idx: int, episode_id: int, start_ts: int) -> None:
        if self.failure_mode != "explore":
            return
        if not self._explore_units:
            return
        if int(unit_idx) != int(self._explore_curr_unit_idx):
            return
        sample_uid = (int(episode_id), int(start_ts))
        if sample_uid in self._explore_seen_samples:
            return
        self._explore_seen_samples.add(sample_uid)
        self._explore_trial_count_local += 1
        if int(self._explore_trial_count_local) >= int(self._explore_k_local):
            self._advance_explore_unit(count_as_completed=True)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        if self.failure_mode in {"train", "explore"}:
            sampled = self._sample_start_ts_from_failure_unit(index)
            assert sampled is not None
            episode_id = int(sampled["episode_id"])
            start_ts = int(sampled["start_ts"])
            pregrasp_seg_start = int(sampled.get("seg_start", -1))
            pregrasp_seg_end = int(sampled.get("seg_end", -1))
            sampled_phase_key = str(sampled.get("phase_key", "approach"))
            sampled_phase_instance_id = int(sampled.get("phase_instance_idx", -1))
            sampled_phase_bin_id = int(sampled.get("phase_bin_id", -1))
            forced_error_mode_id = error_mode_key_to_id(str(sampled.get("error_mode", "translation")))
            sampled_active_arm_pattern_id = active_arm_pattern_key_to_id(
                str(sampled.get("active_arm_pattern", "both"))
            )
            forced_dir_bin_id = int(sampled.get("dir_bin_id", -1))
            forced_mag_bin_id = int(sampled.get("mag_bin_id", -1))
            sampled_explore_unit_idx = int(sampled.get("explore_unit_idx", -1))
            sampled_mode_prob = _safe_float_or_nan(sampled.get("sampled_mode_prob", np.nan))
            sampled_entry_prob_within_mode = _safe_float_or_nan(
                sampled.get("sampled_entry_prob_within_mode", np.nan)
            )
            sampled_unit_prob = _safe_float_or_nan(sampled.get("sampled_unit_prob", np.nan))
        else:
            episode_id = int(self.episode_ids[index])
            max_start = self._resolve_max_start(episode_id)
            min_start = self._resolve_min_start(episode_id, max_start)
            start_ts = int(np.random.randint(min_start, max_start + 1))
            pregrasp_seg_start = -1
            pregrasp_seg_end = -1
            sampled_phase_key = infer_phase_key_from_gt_window(
                np.zeros((1,), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                1,
            )
            sampled_phase_instance_id = -1
            sampled_phase_bin_id = -1
            forced_error_mode_id = -1
            sampled_active_arm_pattern_id = -1
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
        sample["pregrasp_seg_start"] = torch.tensor(pregrasp_seg_start, dtype=torch.int64)
        sample["pregrasp_seg_end"] = torch.tensor(pregrasp_seg_end, dtype=torch.int64)
        sample["sampled_phase_bin_id"] = torch.tensor(sampled_phase_bin_id, dtype=torch.int64)
        sample["sampled_phase_instance_id"] = torch.tensor(sampled_phase_instance_id, dtype=torch.int64)
        sample["forced_error_mode_id"] = torch.tensor(forced_error_mode_id, dtype=torch.int64)
        sample["sampled_active_arm_pattern_id"] = torch.tensor(sampled_active_arm_pattern_id, dtype=torch.int64)
        sample["forced_dir_bin_id"] = torch.tensor(forced_dir_bin_id, dtype=torch.int64)
        sample["forced_mag_bin_id"] = torch.tensor(forced_mag_bin_id, dtype=torch.int64)
        sample["sampled_explore_unit_idx"] = torch.tensor(sampled_explore_unit_idx, dtype=torch.int64)
        sample["sampled_mode_prob"] = torch.tensor(sampled_mode_prob, dtype=torch.float32)
        sample["sampled_entry_prob_within_mode"] = torch.tensor(sampled_entry_prob_within_mode, dtype=torch.float32)
        sample["sampled_unit_prob"] = torch.tensor(sampled_unit_prob, dtype=torch.float32)
        return sample


class MultiTaskFailureAwareStage2Dataset(Dataset):
    """Thin multitask wrapper around per-task stage2 datasets.

    The per-task datasets keep all failure-table sampling semantics unchanged.
    This wrapper only adds task metadata so the trainer can load the correct
    raw episode and build task-scoped latent-cache keys.
    """

    def __init__(self, datasets: list[FailureAwareStage2Dataset], task_specs: list[MultiTaskSpec]):
        if len(datasets) != len(task_specs):
            raise ValueError("datasets and task_specs must have the same length")
        if not datasets:
            raise ValueError("MultiTaskFailureAwareStage2Dataset requires at least one dataset")
        self.datasets = list(datasets)
        self.task_specs = list(task_specs)
        self._cumulative = np.cumsum([len(ds) for ds in self.datasets]).astype(np.int64)

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    @property
    def norm_stats(self) -> dict[str, np.ndarray]:
        return self.datasets[0].norm_stats

    @norm_stats.setter
    def norm_stats(self, stats: dict[str, np.ndarray]) -> None:
        self.set_norm_stats(stats)

    @property
    def stats(self) -> dict[str, np.ndarray]:
        return self.norm_stats

    @stats.setter
    def stats(self, stats: dict[str, np.ndarray]) -> None:
        self.set_norm_stats(stats)

    def set_norm_stats(self, stats: dict[str, np.ndarray]) -> None:
        for ds in self.datasets:
            ds.norm_stats = stats

    def _locate(self, index: int) -> tuple[int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        task_idx = int(np.searchsorted(self._cumulative, index, side="right"))
        prev = 0 if task_idx == 0 else int(self._cumulative[task_idx - 1])
        return task_idx, index - prev

    def __getitem__(self, index: int):
        task_idx, local_index = self._locate(index)
        spec = self.task_specs[task_idx]
        sample = self.datasets[task_idx][local_index]
        sample["task_idx"] = torch.tensor(task_idx, dtype=torch.int64)
        sample["task_name"] = spec.task_name
        sample["raw_data_dir"] = "" if spec.raw_data_dir is None else spec.raw_data_dir
        return sample


def build_failure_table_dataset(
    dataset_dir: str,
    num_episodes: int,
    camera_names: list[str],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
    sample_phase_window_len: int,
    sample_skip_head_ratio: float | None,
    start_margin: int,
    failure_mode: str,
    failure_table_path: str,
    failure_phase_bins: int,
    failure_translation_dir_bins: int,
    failure_translation_mag_bins: int,
    failure_rotation_dir_bins: int,
    failure_rotation_mag_bins: int,
    failure_explore_k: int = 1,
) -> tuple[FailureAwareStage2Dataset, dict[str, np.ndarray]]:
    stats = get_norm_stats(dataset_dir, num_episodes)
    valid_ids = list_valid_episode_ids(dataset_dir, num_episodes, prefix_steps, future_offset)
    dataset = FailureAwareStage2Dataset(
        dataset_dir=dataset_dir,
        episode_ids=valid_ids,
        camera_names=camera_names,
        norm_stats=stats,
        act_chunk_size=act_chunk_size,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
        sample_phase_window_len=sample_phase_window_len,
        sample_skip_head_ratio=sample_skip_head_ratio,
        start_margin=start_margin,
        failure_mode=failure_mode,
        failure_table_path=failure_table_path,
        failure_phase_bins=failure_phase_bins,
        failure_translation_dir_bins=failure_translation_dir_bins,
        failure_translation_mag_bins=failure_translation_mag_bins,
        failure_rotation_dir_bins=failure_rotation_dir_bins,
        failure_rotation_mag_bins=failure_rotation_mag_bins,
        failure_explore_k=failure_explore_k,
    )
    return dataset, stats


def build_multitask_failure_table_dataset(
    task_specs: list[MultiTaskSpec],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
    sample_phase_window_len: int,
    sample_skip_head_ratio: float | None,
    start_margin: int,
    failure_mode: str,
    failure_table_paths: list[str],
    failure_phase_bins: int,
    failure_translation_dir_bins: int,
    failure_translation_mag_bins: int,
    failure_rotation_dir_bins: int,
    failure_rotation_mag_bins: int,
    failure_explore_k: int = 1,
) -> tuple[MultiTaskFailureAwareStage2Dataset, dict[str, np.ndarray]]:
    stats = get_multitask_norm_stats(task_specs)
    if failure_mode == "train":
        if not failure_table_paths:
            raise ValueError("failure_mode=train requires failure_table_paths for multitask stage2")
        if len(failure_table_paths) == 1 and len(task_specs) > 1:
            failure_table_paths = list(failure_table_paths) * len(task_specs)
        if len(failure_table_paths) != len(task_specs):
            raise ValueError(
                f"Expected 1 or {len(task_specs)} failure_table_paths, got {len(failure_table_paths)}"
            )
    else:
        failure_table_paths = list(failure_table_paths or [""] * len(task_specs))
        if len(failure_table_paths) == 1 and len(task_specs) > 1:
            failure_table_paths = list(failure_table_paths) * len(task_specs)
        if len(failure_table_paths) < len(task_specs):
            failure_table_paths.extend([""] * (len(task_specs) - len(failure_table_paths)))

    datasets: list[FailureAwareStage2Dataset] = []
    for idx, spec in enumerate(task_specs):
        valid_ids = list_valid_episode_ids(spec.dataset_dir, spec.num_episodes, prefix_steps, future_offset)
        datasets.append(
            FailureAwareStage2Dataset(
                dataset_dir=spec.dataset_dir,
                episode_ids=valid_ids,
                camera_names=list(spec.camera_names),
                norm_stats=stats,
                act_chunk_size=act_chunk_size,
                prefix_steps=prefix_steps,
                future_offset=future_offset,
                sample_phase_window_len=sample_phase_window_len,
                sample_skip_head_ratio=sample_skip_head_ratio,
                start_margin=start_margin,
                failure_mode=failure_mode,
                failure_table_path=failure_table_paths[idx],
                failure_phase_bins=failure_phase_bins,
                failure_translation_dir_bins=failure_translation_dir_bins,
                failure_translation_mag_bins=failure_translation_mag_bins,
                failure_rotation_dir_bins=failure_rotation_dir_bins,
                failure_rotation_mag_bins=failure_rotation_mag_bins,
                failure_explore_k=failure_explore_k,
            )
        )
    return MultiTaskFailureAwareStage2Dataset(datasets, task_specs), stats
