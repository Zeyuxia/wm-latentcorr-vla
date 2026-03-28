import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from phase_utils import infer_phase_key_from_gt_window

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len,
                 raw_data_dir=None, start_margin=0,
                 sample_skip_head_ratio=None,
                 sample_pregrasp_bias_enable=False,
                 sample_pregrasp_prob=0.0,
                 sample_pregrasp_phase_window_len=16,
                 sample_pregrasp_keep_start_ratio=0.0,
                 sample_pregrasp_keep_end_ratio=0.5):
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
        self.sample_pregrasp_bias_enable = bool(sample_pregrasp_bias_enable)
        self.sample_pregrasp_prob = float(np.clip(sample_pregrasp_prob, 0.0, 1.0))
        self.sample_pregrasp_phase_window_len = max(1, int(sample_pregrasp_phase_window_len))
        self.sample_pregrasp_keep_start_ratio = float(np.clip(sample_pregrasp_keep_start_ratio, 0.0, 1.0))
        self.sample_pregrasp_keep_end_ratio = float(np.clip(sample_pregrasp_keep_end_ratio, 0.0, 1.0))
        if self.sample_pregrasp_keep_end_ratio < self.sample_pregrasp_keep_start_ratio:
            self.sample_pregrasp_keep_end_ratio = self.sample_pregrasp_keep_start_ratio
        self._pregrasp_start_cache = {}
        self._first_stage_len_cache = {}
        self._strict_pregrasp_calls = 0
        self._strict_pregrasp_skipped_total = 0
        self._strict_pregrasp_skipped_max = 0
        self._strict_pregrasp_log_every = 100
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

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
                        self.sample_pregrasp_phase_window_len,
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

    def _get_pregrasp_start_candidates(self, episode_id, min_start, max_start, apply_keep_ratio=True):
        key = (int(episode_id), int(min_start), int(max_start), bool(apply_keep_ratio))
        if key in self._pregrasp_start_cache:
            return self._pregrasp_start_cache[key]

        candidates = []
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        try:
            with h5py.File(dataset_path, "r") as root:
                action = root["/action"][()]
            if action.ndim != 2 or action.shape[1] < 14 or action.shape[0] <= 1:
                self._pregrasp_start_cache[key] = candidates
                return candidates

            lg = np.asarray(action[:, 6], dtype=np.float32).reshape(-1)
            rg = np.asarray(action[:, 13], dtype=np.float32).reshape(-1)
            for ts in range(int(min_start), int(max_start) + 1):
                ph = infer_phase_key_from_gt_window(
                    lg[ts:],
                    rg[ts:],
                    self.sample_pregrasp_phase_window_len,
                )
                if ph == "pregrasp":
                    candidates.append(int(ts))
            # Keep a configurable ratio-range from each contiguous pregrasp segment.
            # Example: [0.0, 0.5] means keep front half only.
            if apply_keep_ratio and len(candidates) > 0:
                kept = []
                seg_start = 0
                n = len(candidates)
                while seg_start < n:
                    seg_end = seg_start
                    while seg_end + 1 < n and candidates[seg_end + 1] == candidates[seg_end] + 1:
                        seg_end += 1
                    seg = candidates[seg_start:seg_end + 1]
                    seg_len = len(seg)
                    keep_l = int(np.floor(seg_len * self.sample_pregrasp_keep_start_ratio))
                    keep_r = int(np.ceil(seg_len * self.sample_pregrasp_keep_end_ratio))
                    keep_l = int(np.clip(keep_l, 0, max(0, seg_len - 1)))
                    keep_r = int(np.clip(keep_r, keep_l + 1, seg_len))
                    kept.extend(seg[keep_l:keep_r])
                    seg_start = seg_end + 1
                candidates = [int(ts) for ts in kept]
        except Exception:
            candidates = []

        self._pregrasp_start_cache[key] = candidates
        return candidates

    def _find_candidate_segment_bounds(self, candidates, ts):
        if candidates is None or len(candidates) == 0:
            return -1, -1
        arr = [int(x) for x in candidates]
        t = int(ts)
        try:
            pos = arr.index(t)
        except ValueError:
            return -1, -1
        l = pos
        r = pos
        while l - 1 >= 0 and arr[l - 1] == arr[l] - 1:
            l -= 1
        while r + 1 < len(arr) and arr[r + 1] == arr[r] + 1:
            r += 1
        return int(arr[l]), int(arr[r])

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        sample_pregrasp = False
        if self.sample_pregrasp_bias_enable and self.sample_pregrasp_prob > 0.0:
            sample_pregrasp = (np.random.rand() < self.sample_pregrasp_prob)
        strict_pregrasp = bool(sample_pregrasp and self.sample_pregrasp_prob >= (1.0 - 1e-8))

        # Strict mode: when pregrasp sampling is effectively mandatory (prob=1),
        # do not fall back to random start_ts; resample across episodes instead.
        forced_start_ts = None
        forced_cand = None
        if (not sample_full_episode) and strict_pregrasp:
            n_ep = len(self.episode_ids)
            found = False
            skipped_before_found = 0
            for ofs in range(n_ep):
                ep_try = self.episode_ids[(index + ofs) % n_ep]
                path_try = os.path.join(self.dataset_dir, f"episode_{ep_try}.hdf5")
                with h5py.File(path_try, "r") as root_try:
                    ep_len_try = int(root_try["/action"].shape[0])
                max_start_try = max(0, ep_len_try - 1 - self.start_margin)
                min_start_try = self._resolve_min_start(ep_try, max_start_try)
                cand_try = self._get_pregrasp_start_candidates(ep_try, min_start_try, max_start_try, apply_keep_ratio=True)
                if len(cand_try) > 0:
                    episode_id = ep_try
                    forced_cand = cand_try
                    forced_start_ts = int(np.random.choice(cand_try))
                    skipped_before_found = int(ofs)
                    found = True
                    break
            self._strict_pregrasp_calls += 1
            self._strict_pregrasp_skipped_total += int(skipped_before_found)
            self._strict_pregrasp_skipped_max = max(self._strict_pregrasp_skipped_max, int(skipped_before_found))
            if self._strict_pregrasp_calls % self._strict_pregrasp_log_every == 0:
                avg_skip = float(self._strict_pregrasp_skipped_total) / float(max(1, self._strict_pregrasp_calls))
                print(
                    "[EpisodicDataset] strict pregrasp resample stats: "
                    f"calls={self._strict_pregrasp_calls}, "
                    f"avg_skipped={avg_skip:.2f}, "
                    f"max_skipped={self._strict_pregrasp_skipped_max}"
                )
            if not found:
                raise RuntimeError(
                    "Strict pregrasp sampling enabled (sample_pregrasp_prob=1.0), "
                    "but no pregrasp candidates were found in any episode."
                )

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
                    cand_full = self._get_pregrasp_start_candidates(
                        episode_id, min_start, max_start, apply_keep_ratio=False
                    )
                    pregrasp_seg_start, pregrasp_seg_end = self._find_candidate_segment_bounds(cand_full, start_ts)
                else:
                    start_ts = np.random.randint(min_start, max_start + 1)
                    pregrasp_seg_start = -1
                    pregrasp_seg_end = -1
                    if sample_pregrasp:
                        cand = self._get_pregrasp_start_candidates(episode_id, min_start, max_start, apply_keep_ratio=True)
                        if len(cand) > 0:
                            start_ts = int(np.random.choice(cand))
                            cand_full = self._get_pregrasp_start_candidates(
                                episode_id, min_start, max_start, apply_keep_ratio=False
                            )
                            pregrasp_seg_start, pregrasp_seg_end = self._find_candidate_segment_bounds(cand_full, start_ts)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            # Keep action and qpos on the same timestamp origin.
            action = root["/action"][start_ts:]
            action_len = episode_len - start_ts

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
            return (
                image_data,
                qpos_data,
                action_data,
                is_pad,
                torch.tensor(episode_id),
                torch.tensor(start_ts),
                torch.tensor(pregrasp_seg_start),
                torch.tensor(pregrasp_seg_end),
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
              sample_pregrasp_bias_enable=False,
              sample_pregrasp_prob=0.0,
              sample_pregrasp_phase_window_len=16,
              sample_pregrasp_keep_start_ratio=0.0,
              sample_pregrasp_keep_end_ratio=0.5):
    print(f"\nData from: {dataset_dir}\n")
    # Filter episodes that are too short for the requested start sampling range.
    # Need at least one valid start_ts in [min_start, episode_len - 1 - start_margin].
    # For ratio-based skip-head, min_start depends on episode length, so only require
    # enough length for max_start >= 0.
    min_required_len = int(max(0, start_margin)) + 1
    train_indices = []
    skipped_short = []
    for ep in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{ep}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            ep_len = int(root["/action"].shape[0])
        if ep_len >= min_required_len:
            train_indices.append(ep)
        else:
            skipped_short.append((ep, ep_len))
    if len(train_indices) == 0:
        raise ValueError(
            f"No valid episodes: require len >= {min_required_len} "
            f"(sample_skip_head_ratio={sample_skip_head_ratio}, "
            f"start_margin={int(max(0, start_margin))})."
        )
    if len(skipped_short) > 0:
        print(
            f"[load_data] filtered short episodes: {len(skipped_short)}/{num_episodes} "
            f"(min_required_len={min_required_len})"
        )

    # obtain normalization stats for qpos and action
    norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_action_len,
                                    raw_data_dir=raw_data_dir, start_margin=start_margin,
                                    sample_skip_head_ratio=sample_skip_head_ratio,
                                    sample_pregrasp_bias_enable=sample_pregrasp_bias_enable,
                                    sample_pregrasp_prob=sample_pregrasp_prob,
                                    sample_pregrasp_phase_window_len=sample_pregrasp_phase_window_len,
                                    sample_pregrasp_keep_start_ratio=sample_pregrasp_keep_start_ratio,
                                    sample_pregrasp_keep_end_ratio=sample_pregrasp_keep_end_ratio)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, None, norm_stats, train_dataset.is_sim, max_action_len


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
