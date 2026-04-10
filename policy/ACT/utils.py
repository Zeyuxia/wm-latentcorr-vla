import numpy as np
import torch
import os
import json
import h5py
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, WeightedRandomSampler
from imitate_episodes_pkg.utils import (
    infer_phase_key_from_gt_window,
    phase_key_to_id,
    phase_id_to_key,
    error_mode_key_to_id,
    active_arm_pattern_key_to_id,
    normalize_error_mode_dir_mag_bins,
    get_failure_param_bins,
)

import IPython

e = IPython.embed

_FAILURE_ERROR_MODES = ("translation", "rotation", "gripper_close")


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len,
                 raw_data_dir=None, start_margin=0,
                 sample_skip_head_ratio=None,
                 sample_phase_window_len=16,
                 failure_mode="off",
                 failure_table_path="",
                 failure_phase_bins=5,
                 failure_explore_k=1):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        self.raw_data_dir = raw_data_dir
        self.start_margin = max(0, int(start_margin))
        self.sample_skip_head_ratio = None if sample_skip_head_ratio is None else float(sample_skip_head_ratio)
        if self.sample_skip_head_ratio is not None:
            self.sample_skip_head_ratio = float(np.clip(self.sample_skip_head_ratio, 0.0, 0.99))
        self.sample_phase_window_len = max(1, int(sample_phase_window_len))
        self.failure_mode = str(failure_mode).strip().lower()
        if self.failure_mode not in {"off", "explore", "train"}:
            raise ValueError(f"Invalid failure_mode={failure_mode!r}, expected off|explore|train")
        self.failure_table_path = str(failure_table_path).strip()
        self.failure_phase_bins = int(max(1, int(failure_phase_bins)))
        self.failure_explore_k = int(max(1, int(failure_explore_k)))
        self.failure_skip_approach_bins = 0
        if self.failure_mode == "explore":
            # In explore mode, skip early bins of the first stage directly in bin space
            # (instead of time-ratio min_start truncation).
            skip_bins = int(np.floor(float(self.sample_skip_head_ratio or 0.0) * float(self.failure_phase_bins)))
            self.failure_skip_approach_bins = int(np.clip(skip_bins, 0, self.failure_phase_bins))
        self._phase_scan_cache = {}
        self._first_stage_len_cache = {}
        self._failure_entries = []
        self._explore_units = []
        self._explore_curr_unit_idx = 0
        self._init_failure_units()
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def _init_failure_units(self):
        if self.failure_mode == "off":
            return
        if self.failure_mode == "explore":
            units = []
            bin_cfg = get_failure_param_bins()
            trans_dir_bins = int(max(1, bin_cfg["translation_dir_bins"]))
            trans_mag_bins = int(max(1, bin_cfg["translation_mag_bins"]))
            rot_dir_bins = int(max(1, bin_cfg["rotation_dir_bins"]))
            rot_mag_bins = int(max(1, bin_cfg["rotation_mag_bins"]))
            for phase_id in range(4):
                phase_key = phase_id_to_key(phase_id)
                for bin_id in range(self.failure_phase_bins):
                    if phase_key == "approach" and int(bin_id) < int(self.failure_skip_approach_bins):
                        continue
                    for mode in _FAILURE_ERROR_MODES:
                        if mode == "translation":
                            for dir_bin_id in range(trans_dir_bins):
                                for mag_bin_id in range(trans_mag_bins):
                                    units.append(
                                        {
                                            "phase_key": phase_key,
                                            "phase_instance_idx": None,
                                            "phase_bin_id": int(bin_id),
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
                                            "phase_instance_idx": None,
                                            "phase_bin_id": int(bin_id),
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
                                    "phase_instance_idx": None,
                                    "phase_bin_id": int(bin_id),
                                    "error_mode": mode,
                                    "active_arm_pattern": None,
                                    "dir_bin_id": -1,
                                    "mag_bin_id": -1,
                                    "weight": 1.0,
                                }
                            )
            self._explore_units = units
            return
        if self.failure_mode == "train":
            if not self.failure_table_path:
                raise ValueError("failure_mode=train requires failure_table_path")
            entries = []
            if os.path.isdir(self.failure_table_path):
                preferred_files = [
                    "failure_table_translation.json",
                    "failure_table_rotation.json",
                    "failure_table_gripper_close.json",
                ]
                loaded_any = False
                for name in preferred_files:
                    path_i = os.path.join(self.failure_table_path, name)
                    if not os.path.isfile(path_i):
                        continue
                    with open(path_i, "r") as f:
                        table_i = json.load(f)
                    entries.extend(table_i.get("entries", []))
                    loaded_any = True
                if not loaded_any:
                    path_i = os.path.join(self.failure_table_path, "failure_table.json")
                    if not os.path.isfile(path_i):
                        raise ValueError(
                            f"failure_mode=train expected table file(s) under directory: {self.failure_table_path}"
                        )
                    with open(path_i, "r") as f:
                        table_i = json.load(f)
                    entries = table_i.get("entries", [])
            else:
                with open(self.failure_table_path, "r") as f:
                    table = json.load(f)
                entries = table.get("entries", [])
            if not isinstance(entries, list) or len(entries) == 0:
                raise ValueError(f"failure_table has no entries: {self.failure_table_path}")
            parsed = []
            for it in entries:
                try:
                    mode = str(it["error_mode"]).strip().lower()
                    if mode not in _FAILURE_ERROR_MODES:
                        continue
                    phase_key = str(it["phase_key"]).strip().lower()
                    phase_key_to_id(phase_key)
                    if "phase_instance_idx" not in it:
                        continue
                    phase_instance_idx = int(it["phase_instance_idx"])
                    if phase_instance_idx < 1:
                        continue
                    bin_id = int(np.clip(int(it["phase_bin_id"]), 0, self.failure_phase_bins - 1))
                    if "active_arm_pattern" not in it:
                        continue
                    active_pattern = str(it["active_arm_pattern"]).strip().lower()
                    if active_pattern not in {"left_only", "right_only", "both"}:
                        continue
                    weight = float(it.get("weight", it.get("fail_rate", 1.0)))
                    if not np.isfinite(weight) or weight <= 0.0:
                        weight = 1.0
                    if mode in {"translation", "rotation"}:
                        if ("dir_bin_id" not in it) or ("mag_bin_id" not in it):
                            continue
                    dir_bin_id, mag_bin_id = normalize_error_mode_dir_mag_bins(
                        mode,
                        int(it.get("dir_bin_id", -1)),
                        int(it.get("mag_bin_id", -1)),
                    )
                    parsed.append(
                        {
                            "phase_key": phase_key,
                            "phase_instance_idx": int(phase_instance_idx),
                            "phase_bin_id": int(bin_id),
                            "error_mode": mode,
                            "active_arm_pattern": active_pattern,
                            "dir_bin_id": int(dir_bin_id),
                            "mag_bin_id": int(mag_bin_id),
                            "weight": float(weight),
                        }
                    )
                except Exception:
                    continue
            if len(parsed) == 0:
                raise ValueError(f"No valid entries in failure_table: {self.failure_table_path}")
            self._failure_entries = parsed

    def _infer_active_arm_pattern_from_action_window(self, action, ts, joint_delta_thresh=0.02):
        a = np.asarray(action, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] <= 1 or a.shape[1] < 14:
            return "both"
        s = int(np.clip(int(ts), 0, a.shape[0] - 1))
        e = int(np.clip(s + max(2, int(self.sample_phase_window_len)), s + 1, a.shape[0]))
        seg = a[s:e]
        if seg.shape[0] <= 1:
            return "both"
        dleft = np.diff(seg[:, 0:6], axis=0)
        dright = np.diff(seg[:, 7:13], axis=0)
        left_score = float(np.max(np.linalg.norm(dleft, axis=1))) if dleft.shape[0] > 0 else 0.0
        right_score = float(np.max(np.linalg.norm(dright, axis=1))) if dright.shape[0] > 0 else 0.0
        left_on = bool(left_score >= float(joint_delta_thresh))
        right_on = bool(right_score >= float(joint_delta_thresh))
        if left_on and right_on:
            return "both"
        if left_on:
            return "left_only"
        if right_on:
            return "right_only"
        return "both"

    def _scan_phase_bins(self, episode_id, min_start, max_start):
        key = (int(episode_id), int(min_start), int(max_start), int(self.failure_phase_bins))
        if key in self._phase_scan_cache:
            return self._phase_scan_cache[key]
        info = {}
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            action = root["/action"][()]
        lg = np.asarray(action[:, 6], dtype=np.float32).reshape(-1)
        rg = np.asarray(action[:, 13], dtype=np.float32).reshape(-1)
        ts_list = list(range(int(min_start), int(max_start) + 1))
        phase_list = []
        for ts in ts_list:
            phase_list.append(
                infer_phase_key_from_gt_window(
                    lg[ts:],
                    rg[ts:],
                    self.sample_phase_window_len,
                )
            )
        i = 0
        n = len(ts_list)
        phase_instance_cnt = {}
        while i < n:
            j = i
            phase_key = str(phase_list[i]).strip().lower()
            while j + 1 < n and str(phase_list[j + 1]).strip().lower() == phase_key:
                j += 1
            seg = ts_list[i:j + 1]
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

    def _advance_explore_unit(self):
        if len(self._explore_units) == 0:
            return
        self._explore_curr_unit_idx = (int(self._explore_curr_unit_idx) + 1) % int(len(self._explore_units))

    def set_explore_unit_idx(self, unit_idx):
        if len(self._explore_units) == 0:
            self._explore_curr_unit_idx = 0
            return
        self._explore_curr_unit_idx = int(np.clip(int(unit_idx), 0, len(self._explore_units) - 1))

    def _select_failure_unit(self):
        if self.failure_mode == "explore":
            if len(self._explore_units) == 0:
                raise RuntimeError("explore mode has no units")
            unit_idx = int(self._explore_curr_unit_idx) % int(len(self._explore_units))
            unit = self._explore_units[unit_idx]
            return dict(unit), int(unit_idx)
        if self.failure_mode == "train":
            if len(self._failure_entries) == 0:
                raise RuntimeError("train mode has empty failure entries")
            w = np.asarray([float(x.get("weight", 1.0)) for x in self._failure_entries], dtype=np.float64)
            if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 1e-12:
                w = np.ones(len(self._failure_entries), dtype=np.float64)
            w = w / np.sum(w)
            idx = int(np.random.choice(len(self._failure_entries), p=w))
            return dict(self._failure_entries[idx]), int(idx)
        return None

    def _sample_start_ts_from_failure_unit(self, base_index):
        if self.failure_mode == "explore":
            if len(self._explore_units) == 0:
                return None
            n_unit_try = int(len(self._explore_units))
        else:
            n_unit_try = 1

        for _u_try in range(n_unit_try):
            out = self._select_failure_unit()
            if out is None:
                return None
            unit, unit_idx = out
            n_ep = len(self.episode_ids)
            for ofs in range(n_ep):
                ep = int(self.episode_ids[(int(base_index) + ofs) % n_ep])
                dataset_path = os.path.join(self.dataset_dir, f"episode_{ep}.hdf5")
                with h5py.File(dataset_path, "r") as root:
                    ep_len = int(root["/action"].shape[0])
                max_start = max(0, ep_len - 1 - self.start_margin)
                if self.failure_mode == "explore":
                    min_start = 0
                else:
                    min_start = self._resolve_min_start(ep, max_start)
                scan = self._scan_phase_bins(ep, min_start, max_start)
                cand = []
                for ts, m in scan.items():
                    if str(m["phase_key"]) != str(unit["phase_key"]):
                        continue
                    target_phase_inst = unit.get("phase_instance_idx", None)
                    if target_phase_inst is not None:
                        if int(m.get("phase_instance_idx", -1)) != int(target_phase_inst):
                            continue
                    if int(m["phase_bin_id"]) != int(unit["phase_bin_id"]):
                        continue
                    target_pat = unit.get("active_arm_pattern", None)
                    if target_pat is not None and str(m["active_arm_pattern"]) != str(target_pat):
                        continue
                    cand.append(int(ts))
                if len(cand) == 0:
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
                }

            if self.failure_mode == "explore":
                # Current unit has no valid candidate in this dataset. Skip it.
                self._advance_explore_unit()

        raise RuntimeError(
            "Failed to sample start_ts for failure unit: "
            f"{unit if 'unit' in locals() else None} (dataset={self.dataset_dir})"
        )

    def _get_first_stage_len(self, episode_id, max_start):
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
                    ph = infer_phase_key_from_gt_window(
                        lg[ts:],
                        rg[ts:],
                        self.sample_phase_window_len,
                    )
                    if ph != "approach":
                        first_non_approach = ts
                        break
                first_stage_len = int(first_non_approach)
        except Exception:
            first_stage_len = n_valid

        first_stage_len = int(np.clip(first_stage_len, 0, n_valid))
        self._first_stage_len_cache[key] = first_stage_len
        return first_stage_len

    def _resolve_min_start(self, episode_id, max_start):
        max_start = int(max(0, max_start))
        if self.sample_skip_head_ratio is None:
            return 0
        first_stage_len = self._get_first_stage_len(episode_id, max_start)
        min_start_ratio = int(np.floor(float(first_stage_len) * float(self.sample_skip_head_ratio)))
        return int(np.clip(min_start_ratio, 0, max_start))

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        sampled_phase_bin_id = -1
        sampled_phase_instance_idx = -1
        forced_error_mode_key = None
        sampled_active_arm_pattern = "both"
        forced_dir_bin_id = -1
        forced_mag_bin_id = -1
        sampled_explore_unit_idx = -1
        forced_start_ts = None
        failure_sampled = None
        if (not sample_full_episode) and self.failure_mode in {"explore", "train"}:
            failure_sampled = self._sample_start_ts_from_failure_unit(index)
            episode_id = int(failure_sampled["episode_id"])
            forced_start_ts = int(failure_sampled["start_ts"])
            forced_error_mode_key = str(failure_sampled["error_mode"])
            sampled_phase_bin_id = int(failure_sampled["phase_bin_id"])
            sampled_phase_instance_idx = int(failure_sampled.get("phase_instance_idx", -1))
            sampled_active_arm_pattern = str(failure_sampled.get("active_arm_pattern", "both"))
            forced_dir_bin_id = int(failure_sampled.get("dir_bin_id", -1))
            forced_mag_bin_id = int(failure_sampled.get("mag_bin_id", -1))
            sampled_explore_unit_idx = int(failure_sampled.get("explore_unit_idx", -1))

        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = None
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
                pregrasp_seg_start = -1
                pregrasp_seg_end = -1
            else:
                # Avoid sampling too close to episode end so rollout/correction
                # still has enough future horizon.
                max_start = max(0, episode_len - 1 - self.start_margin)
                min_start = self._resolve_min_start(episode_id, max_start)
                if forced_start_ts is not None:
                    start_ts = int(np.clip(forced_start_ts, min_start, max_start))
                    pregrasp_seg_start = int(failure_sampled.get("seg_start", -1)) if failure_sampled is not None else -1
                    pregrasp_seg_end = int(failure_sampled.get("seg_end", -1)) if failure_sampled is not None else -1
                else:
                    start_ts = np.random.randint(min_start, max_start + 1)
                    pregrasp_seg_start = -1
                    pregrasp_seg_end = -1
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            # Keep action and qpos on the same timestamp origin.
            action_full = None
            if self.failure_mode in {"explore", "train"}:
                action_full = root["/action"][()]
            action = root["/action"][start_ts:]
            action_len = episode_len - start_ts

        sampled_phase = infer_phase_key_from_gt_window(
            action[:, 6],
            action[:, 13],
            self.sample_phase_window_len,
        )
        sampled_phase_id = phase_key_to_id(sampled_phase)
        if sampled_phase_bin_id < 0:
            if episode_len <= 1:
                sampled_phase_bin_id = 0
            else:
                progress = float(start_ts) / float(max(1, episode_len - 1))
                sampled_phase_bin_id = int(np.floor(progress * float(self.failure_phase_bins)))
                sampled_phase_bin_id = int(np.clip(sampled_phase_bin_id, 0, self.failure_phase_bins - 1))
        if sampled_phase_instance_idx < 0:
            sampled_phase_instance_idx = 1
        if self.failure_mode in {"explore", "train"}:
            sampled_active_arm_pattern = self._infer_active_arm_pattern_from_action_window(
                action_full,
                int(start_ts),
            )

        self.is_sim = is_sim

        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)  # 根据max_action_len初始化
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)  # 初始化为全1（True）
        is_pad[:action_len] = 0  # 前action_len个位置设置为0（False），表示非填充部分

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        if self.raw_data_dir is not None:
            if forced_error_mode_key is None:
                forced_error_mode_id = -1
            else:
                forced_error_mode_id = error_mode_key_to_id(forced_error_mode_key)
            sampled_active_arm_pattern_id = active_arm_pattern_key_to_id(sampled_active_arm_pattern)
            return (
                image_data,
                qpos_data,
                action_data,
                is_pad,
                torch.tensor(episode_id),
                torch.tensor(start_ts),
                torch.tensor(sampled_phase_id),
                torch.tensor(pregrasp_seg_start),
                torch.tensor(pregrasp_seg_end),
                torch.tensor(sampled_phase_bin_id),
                torch.tensor(sampled_phase_instance_idx),
                torch.tensor(forced_error_mode_id),
                torch.tensor(sampled_active_arm_pattern_id),
                torch.tensor(forced_dir_bin_id),
                torch.tensor(forced_mag_bin_id),
                torch.tensor(sampled_explore_unit_idx),
            )

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Pad all tensors to the maximum size
    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            # Pad with the last element
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action_data:
        current_len = action.size(0)
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
            action = torch.cat([action, pad], dim=0)
        padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    all_action_data = torch.stack(padded_action)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats, max_action_len


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
              raw_data_dir=None, start_margin=0, sample_skip_head_ratio=None,
              sample_phase_window_len=16,
              failure_mode="off",
              failure_table_path="",
              failure_phase_bins=5,
              failure_explore_k=1,
              dataset_dirs=None,
              num_episodes_list=None,
              task_weights=None):
    def _collect_valid_indices(_dataset_dir, _num_episodes):
        min_required_len = int(max(0, start_margin)) + 1
        _train_indices = []
        _skipped_short = []
        for ep in range(_num_episodes):
            dataset_path = os.path.join(_dataset_dir, f"episode_{ep}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                ep_len = int(root["/action"].shape[0])
            if ep_len >= min_required_len:
                _train_indices.append(ep)
            else:
                _skipped_short.append((ep, ep_len))
        if len(_train_indices) == 0:
            raise ValueError(
                f"No valid episodes: dataset={_dataset_dir}, require len >= {min_required_len} "
                f"(sample_skip_head_ratio={sample_skip_head_ratio}, "
                f"start_margin={int(max(0, start_margin))})."
            )
        if len(_skipped_short) > 0:
            print(
                f"[load_data] filtered short episodes: {len(_skipped_short)}/{_num_episodes} "
                f"(dataset={_dataset_dir}, min_required_len={min_required_len})"
            )
        return _train_indices

    def _get_norm_stats_multi(_dataset_dirs, _num_episodes_list):
        all_qpos_data = []
        all_action_data = []
        for ds_dir, n_ep in zip(_dataset_dirs, _num_episodes_list):
            for episode_idx in range(int(n_ep)):
                dataset_path = os.path.join(ds_dir, f"episode_{episode_idx}.hdf5")
                with h5py.File(dataset_path, "r") as root:
                    qpos = root["/observations/qpos"][()]
                    action = root["/action"][()]
                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(torch.from_numpy(action))

        max_qpos_len = max(q.size(0) for q in all_qpos_data)
        max_action_len = max(a.size(0) for a in all_action_data)

        padded_qpos = []
        for qpos in all_qpos_data:
            if qpos.size(0) < max_qpos_len:
                pad = qpos[-1:].repeat(max_qpos_len - qpos.size(0), 1)
                qpos = torch.cat([qpos, pad], dim=0)
            padded_qpos.append(qpos)

        padded_action = []
        for action in all_action_data:
            if action.size(0) < max_action_len:
                pad = action[-1:].repeat(max_action_len - action.size(0), 1)
                action = torch.cat([action, pad], dim=0)
            padded_action.append(action)

        all_qpos_data = torch.stack(padded_qpos)
        all_action_data = torch.stack(padded_action)
        action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
        action_std = all_action_data.std(dim=[0, 1], keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf)
        qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
        qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)
        stats = {
            "action_mean": action_mean.numpy().squeeze(),
            "action_std": action_std.numpy().squeeze(),
            "qpos_mean": qpos_mean.numpy().squeeze(),
            "qpos_std": qpos_std.numpy().squeeze(),
            "example_qpos": padded_qpos[-1],
        }
        return stats, max_action_len

    if dataset_dirs is None:
        print(f"\nData from: {dataset_dir}\n")
        train_indices = _collect_valid_indices(dataset_dir, num_episodes)
        norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)
        train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_action_len,
                                        raw_data_dir=raw_data_dir, start_margin=start_margin,
                                        sample_skip_head_ratio=sample_skip_head_ratio,
                                        sample_phase_window_len=sample_phase_window_len,
                                        failure_mode=failure_mode,
                                        failure_table_path=failure_table_path,
                                        failure_phase_bins=failure_phase_bins,
                                        failure_explore_k=failure_explore_k)
        _num_workers = 0 if str(failure_mode).strip().lower() == "explore" else 1
        _loader_kwargs = {
            "batch_size": batch_size_train,
            "shuffle": True,
            "pin_memory": True,
            "num_workers": _num_workers,
        }
        if _num_workers > 0:
            _loader_kwargs["prefetch_factor"] = 1
        train_dataloader = DataLoader(train_dataset, **_loader_kwargs)
        return train_dataloader, None, norm_stats, train_dataset.is_sim, max_action_len

    if raw_data_dir is not None:
        raise ValueError("Multi-task load_data currently does not support raw_data_dir")
    if num_episodes_list is None or len(dataset_dirs) != len(num_episodes_list):
        raise ValueError("dataset_dirs and num_episodes_list must have the same length")

    print("\nData from multiple tasks:")
    for _d in dataset_dirs:
        print(f"  - {_d}")
    print("")

    norm_stats, max_action_len = _get_norm_stats_multi(dataset_dirs, num_episodes_list)
    sub_datasets = []
    for ds_dir, n_ep in zip(dataset_dirs, num_episodes_list):
        train_indices = _collect_valid_indices(ds_dir, int(n_ep))
        ds = EpisodicDataset(train_indices, ds_dir, camera_names, norm_stats, max_action_len,
                             raw_data_dir=None, start_margin=start_margin,
                             sample_skip_head_ratio=sample_skip_head_ratio,
                             sample_phase_window_len=sample_phase_window_len,
                             failure_mode=failure_mode,
                             failure_table_path=failure_table_path,
                             failure_phase_bins=failure_phase_bins,
                             failure_explore_k=failure_explore_k)
        sub_datasets.append(ds)

    train_dataset = ConcatDataset(sub_datasets)
    sampler = None
    if task_weights is not None:
        if len(task_weights) != len(sub_datasets):
            raise ValueError("task_weights length must match number of datasets")
        sample_weights = []
        for w, ds in zip(task_weights, sub_datasets):
            n = max(1, len(ds))
            sample_weights.extend([float(w) / float(n)] * len(ds))
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )

    _num_workers_mt = 0 if str(failure_mode).strip().lower() == "explore" else 1
    _loader_kwargs_mt = {
        "batch_size": batch_size_train,
        "shuffle": (sampler is None),
        "sampler": sampler,
        "pin_memory": True,
        "num_workers": _num_workers_mt,
    }
    if _num_workers_mt > 0:
        _loader_kwargs_mt["prefetch_factor"] = 1
    train_dataloader = DataLoader(train_dataset, **_loader_kwargs_mt)

    is_sim = sub_datasets[0].is_sim if len(sub_datasets) > 0 else True
    return train_dataloader, None, norm_stats, is_sim, max_action_len


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
