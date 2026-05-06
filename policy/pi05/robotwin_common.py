from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
POLICY_ROOT = THIS_DIR.parent
ROBOTWIN_ROOT = POLICY_ROOT.parent
PI05_ROOT = POLICY_ROOT / "pi05"
DEFAULT_OUTPUT_ROOT = THIS_DIR / "outputs"
DEFAULT_ASSETS_BASE_DIR = DEFAULT_OUTPUT_ROOT / "assets"
DEFAULT_CHECKPOINT_BASE_DIR = DEFAULT_OUTPUT_ROOT / "checkpoints"
DEFAULT_LOG_ROOT = DEFAULT_OUTPUT_ROOT / "logs"
DEFAULT_PROCESSED_DATA_ROOT = DEFAULT_OUTPUT_ROOT / "processed_data"
DEFAULT_CAMERA_MODE = "head_only"
DEFAULT_SECONDARY_CAMERA = "right_wrist"
SUPPORTED_CAMERA_MODES = ("head_only", "dual_view", "tri_view")
SUPPORTED_SECONDARY_CAMERAS = ("left_wrist", "right_wrist")
_PATCHED_TORCH_STACK = False
_PATCHED_LEROBOT_FAST_INIT = False
_PATCHED_LEROBOT_HF_COLUMN_QUERY = False
_PATCHED_OPENPI_RESIZE_WITH_PAD = False


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_secondary_camera(secondary_camera: str | None = None) -> str:
    resolved = str(secondary_camera or DEFAULT_SECONDARY_CAMERA).strip().lower()
    if resolved not in SUPPORTED_SECONDARY_CAMERAS:
        raise ValueError(
            f"Unsupported secondary_camera: {secondary_camera}. "
            f"Expected one of {SUPPORTED_SECONDARY_CAMERAS}."
        )
    return resolved


def normalize_camera_mode(camera_mode: str, secondary_camera: str | None = None) -> tuple[str, str]:
    resolved_mode = str(camera_mode).strip().lower()
    if resolved_mode not in SUPPORTED_CAMERA_MODES:
        raise ValueError(
            f"Unsupported camera_mode: {camera_mode}. "
            f"Expected one of {SUPPORTED_CAMERA_MODES}."
        )
    resolved_secondary = normalize_secondary_camera(secondary_camera)
    return resolved_mode, resolved_secondary


def camera_mode_dir_suffix(camera_mode: str, secondary_camera: str | None = None) -> str:
    resolved_mode, resolved_secondary = normalize_camera_mode(camera_mode, secondary_camera)
    if resolved_mode == "dual_view":
        return f"{resolved_mode}_{resolved_secondary}"
    return resolved_mode


def repo_camera_suffix(camera_mode: str, secondary_camera: str | None = None) -> str:
    resolved_mode, resolved_secondary = normalize_camera_mode(camera_mode, secondary_camera)
    if resolved_mode == "head_only":
        return "headonly"
    if resolved_mode == "dual_view":
        return f"dualview_{resolved_secondary}"
    return "triview"


def selected_cameras(camera_mode: str, secondary_camera: str | None = None) -> list[tuple[str, str]]:
    resolved_mode, resolved_secondary = normalize_camera_mode(camera_mode, secondary_camera)
    if resolved_mode == "head_only":
        return [("cam_high", "head_camera")]
    if resolved_mode == "dual_view":
        wrist_camera = "left_camera" if resolved_secondary == "left_wrist" else "right_camera"
        wrist_name = "cam_left_wrist" if resolved_secondary == "left_wrist" else "cam_right_wrist"
        return [
            ("cam_high", "head_camera"),
            (wrist_name, wrist_camera),
        ]
    return [
        ("cam_high", "head_camera"),
        ("cam_left_wrist", "left_camera"),
        ("cam_right_wrist", "right_camera"),
    ]


def prepare_openpi_imports() -> None:
    for path in (
        THIS_DIR,
        ROBOTWIN_ROOT,
        PI05_ROOT,
        PI05_ROOT / "src",
        PI05_ROOT / "packages" / "openpi-client" / "src",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    patch_torch_stack_for_hf_column()
    patch_openpi_data_loader_signature()
    patch_openpi_training_utils_forward_refs()
    patch_lerobot_fast_init()
    patch_lerobot_hf_column_query()
    patch_openpi_resize_with_pad_torch()


def patch_torch_stack_for_hf_column() -> None:
    global _PATCHED_TORCH_STACK
    if _PATCHED_TORCH_STACK:
        return

    try:
        import torch
        from datasets.arrow_dataset import Column
    except Exception:
        return

    original_stack = torch.stack

    def _stack_with_column_support(tensors: Any, *args: Any, **kwargs: Any):
        if isinstance(tensors, Column):
            tensors = list(tensors)
        return original_stack(tensors, *args, **kwargs)

    torch.stack = _stack_with_column_support  # type: ignore[assignment]
    _PATCHED_TORCH_STACK = True


def patch_openpi_data_loader_signature() -> None:
    try:
        from openpi.training import data_loader as openpi_data_loader
    except Exception:
        return

    create_data_loader = openpi_data_loader.create_data_loader
    if "num_workers" in inspect.signature(create_data_loader).parameters:
        return

    def _create_data_loader_compat(config: Any, *args: Any, num_workers: int | None = None, **kwargs: Any):
        return create_data_loader(config, *args, **kwargs)

    openpi_data_loader.create_data_loader = _create_data_loader_compat


def patch_openpi_training_utils_forward_refs() -> None:
    try:
        from openpi.shared import array_typing as at
        from openpi.training import utils as training_utils
    except Exception:
        return

    if not hasattr(training_utils, "ArrayTree"):
        training_utils.ArrayTree = at.PyTree


def patch_lerobot_fast_init() -> None:
    global _PATCHED_LEROBOT_FAST_INIT
    if _PATCHED_LEROBOT_FAST_INIT:
        return

    try:
        import numpy as np
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    except Exception:
        return

    original_init = lerobot_dataset.LeRobotDataset.__init__

    def _load_hf_dataset_without_transform(dataset_obj: Any):
        if dataset_obj.episodes is None:
            path = str(dataset_obj.root / "data")
            return lerobot_dataset.load_dataset("parquet", data_dir=path, split="train")
        files = [str(dataset_obj.root / dataset_obj.meta.get_data_file_path(ep_idx)) for ep_idx in dataset_obj.episodes]
        return lerobot_dataset.load_dataset("parquet", data_files=files, split="train")

    def _fast_init(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Any | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super(lerobot_dataset.LeRobotDataset, self).__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else lerobot_dataset.HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else lerobot_dataset.CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else lerobot_dataset.get_safe_default_codec()
        self.delta_indices = None
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = lerobot_dataset.LeRobotDatasetMetadata(
            self.repo_id,
            self.root,
            self.revision,
            force_cache_sync=force_cache_sync,
        )
        if self.episodes is not None and self.meta._version >= lerobot_dataset.packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = lerobot_dataset.aggregate_stats(episodes_stats)

        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = _load_hf_dataset_without_transform(self)
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = lerobot_dataset.get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = _load_hf_dataset_without_transform(self)

        self.episode_data_index = lerobot_dataset.get_episode_data_index(self.meta.episodes, self.episodes)

        timestamps = np.asarray(self.hf_dataset["timestamp"], dtype=np.float32)
        episode_indices = np.asarray(self.hf_dataset["episode_index"], dtype=np.int64)
        ep_data_index_np = {key: tensor.numpy() for key, tensor in self.episode_data_index.items()}
        lerobot_dataset.check_timestamps_sync(
            timestamps,
            episode_indices,
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        if self.delta_timestamps is not None:
            lerobot_dataset.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = lerobot_dataset.get_delta_indices(self.delta_timestamps, self.fps)

        self.hf_dataset.set_transform(lerobot_dataset.hf_transform_to_torch)

    if getattr(lerobot_dataset.LeRobotDataset.__init__, "_pi0_fast_init_patch", False):
        _PATCHED_LEROBOT_FAST_INIT = True
        return

    _fast_init._pi0_fast_init_patch = True  # type: ignore[attr-defined]
    _fast_init._original_init = original_init  # type: ignore[attr-defined]
    lerobot_dataset.LeRobotDataset.__init__ = _fast_init
    _PATCHED_LEROBOT_FAST_INIT = True


def patch_lerobot_hf_column_query() -> None:
    global _PATCHED_LEROBOT_HF_COLUMN_QUERY
    if _PATCHED_LEROBOT_HF_COLUMN_QUERY:
        return

    try:
        import torch
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        from datasets.arrow_dataset import Column
    except Exception:
        return

    if getattr(lerobot_dataset.LeRobotDataset, "_pi0_hf_column_query_patch", False):
        _PATCHED_LEROBOT_HF_COLUMN_QUERY = True
        return

    def _as_stackable(values: Any) -> Any:
        if isinstance(values, Column):
            return list(values)
        return values

    def _get_query_timestamps(
        self: Any,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(_as_stackable(timestamps)).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(self: Any, query_indices: dict[str, list[int]]) -> dict[str, Any]:
        return {
            key: torch.stack(_as_stackable(self.hf_dataset.select(q_idx)[key]))
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    lerobot_dataset.LeRobotDataset._get_query_timestamps = _get_query_timestamps
    lerobot_dataset.LeRobotDataset._query_hf_dataset = _query_hf_dataset
    setattr(lerobot_dataset.LeRobotDataset, "_pi0_hf_column_query_patch", True)
    _PATCHED_LEROBOT_HF_COLUMN_QUERY = True


def patch_openpi_resize_with_pad_torch() -> None:
    global _PATCHED_OPENPI_RESIZE_WITH_PAD
    if _PATCHED_OPENPI_RESIZE_WITH_PAD:
        return

    try:
        import torch
        import torch.nn.functional as F
        from openpi.models_pytorch import preprocessing_pytorch
        from openpi.shared import image_tools
    except Exception:
        return

    def _resize_with_pad_torch_fixed(
        images: torch.Tensor,
        height: int,
        width: int,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        original_ndim = images.dim()

        if images.shape[-1] <= 4:
            channels_last = True
            if original_ndim == 3:
                images = images.unsqueeze(0)
            images = images.permute(0, 3, 1, 2)
        else:
            channels_last = False
            if original_ndim == 3:
                images = images.unsqueeze(0)

        _, _, cur_height, cur_width = images.shape
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)

        resized_images = F.interpolate(
            images,
            size=(resized_height, resized_width),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )

        if images.dtype == torch.uint8:
            resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
            constant_value = 0
        elif images.dtype in {torch.float16, torch.float32, torch.bfloat16, torch.float64}:
            resized_images = resized_images.clamp(-1.0, 1.0)
            constant_value = -1.0
        else:
            raise ValueError(f"Unsupported image dtype: {images.dtype}")

        pad_h0, remainder_h = divmod(height - resized_height, 2)
        pad_h1 = pad_h0 + remainder_h
        pad_w0, remainder_w = divmod(width - resized_width, 2)
        pad_w1 = pad_w0 + remainder_w
        padded_images = F.pad(
            resized_images,
            (pad_w0, pad_w1, pad_h0, pad_h1),
            mode="constant",
            value=constant_value,
        )

        if channels_last:
            padded_images = padded_images.permute(0, 2, 3, 1)
        if original_ndim == 3:
            padded_images = padded_images.squeeze(0)
        return padded_images

    image_tools.resize_with_pad_torch = _resize_with_pad_torch_fixed
    preprocessing_pytorch.image_tools.resize_with_pad_torch = _resize_with_pad_torch_fixed
    _PATCHED_OPENPI_RESIZE_WITH_PAD = True


def load_module_from_path(module_name: str, file_path: str | Path) -> ModuleType:
    file_path = Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_experiment_dir(
    train_config_name: str,
    exp_name: str,
    checkpoint_base_dir: str | Path | None = None,
) -> Path:
    base_dir = Path(checkpoint_base_dir or DEFAULT_CHECKPOINT_BASE_DIR).expanduser().resolve()
    return base_dir / train_config_name / exp_name


def resolve_checkpoint_dir(
    train_config_name: str,
    exp_name: str,
    checkpoint_id: str | int | None = "latest",
    checkpoint_base_dir: str | Path | None = None,
) -> Path:
    experiment_dir = resolve_experiment_dir(train_config_name, exp_name, checkpoint_base_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    ckpt_str = "" if checkpoint_id is None else str(checkpoint_id).strip()
    if ckpt_str in {"", "latest"}:
        candidates = sorted(
            (
                path for path in experiment_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )
        if not candidates:
            raise FileNotFoundError(f"No numeric checkpoint directories found under: {experiment_dir}")
        return candidates[-1]

    ckpt_dir = experiment_dir / ckpt_str
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    return ckpt_dir


def infer_asset_repo_id_from_checkpoint(checkpoint_dir: str | Path) -> str | None:
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    assets_dir = checkpoint_dir / "assets"
    if not assets_dir.is_dir():
        return None
    candidates = sorted(path.name for path in assets_dir.iterdir() if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    return None


def build_train_config(
    train_config_name: str,
    *,
    repo_id: str,
    exp_name: str,
    camera_mode: str = DEFAULT_CAMERA_MODE,
    secondary_camera: str | None = None,
    asset_id: str | None = None,
    assets_base_dir: str | Path | None = None,
    checkpoint_base_dir: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    num_train_steps: int | None = None,
    log_interval: int | None = None,
    save_interval: int | None = None,
    keep_period: int | None = None,
    fsdp_devices: int | None = None,
    seed: int | None = None,
    overwrite: bool | None = None,
    resume: bool | None = None,
    wandb_enabled: bool | None = None,
    project_name: str | None = None,
    jax_params_path: str | None = None,
    pytorch_weight_path: str | None = None,
    pytorch_training_precision: str | None = None,
) -> Any:
    prepare_openpi_imports()
    from openpi.training import config as openpi_config
    from openpi.training import weight_loaders as openpi_weight_loaders
    import openpi.transforms as openpi_transforms

    cfg = openpi_config.get_config(train_config_name)
    data_factory = cfg.data
    image_mapping = {
        camera_name: f"observation.images.{camera_name}"
        for camera_name, _ in selected_cameras(camera_mode, secondary_camera)
    }

    repack_transforms = openpi_transforms.Group(
        inputs=[
            openpi_transforms.RepackTransform(
                {
                    "images": image_mapping,
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                }
            )
        ]
    )

    assets_cfg = dataclasses.replace(
        data_factory.assets,
        asset_id=(asset_id or repo_id),
    )
    data_factory = dataclasses.replace(
        data_factory,
        repo_id=repo_id,
        assets=assets_cfg,
        repack_transforms=repack_transforms,
    )

    cfg = dataclasses.replace(
        cfg,
        exp_name=exp_name,
        data=data_factory,
        assets_base_dir=str(Path(assets_base_dir or DEFAULT_ASSETS_BASE_DIR).expanduser().resolve()),
        checkpoint_base_dir=str(Path(checkpoint_base_dir or DEFAULT_CHECKPOINT_BASE_DIR).expanduser().resolve()),
    )

    if batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=int(batch_size))
    if num_workers is not None:
        cfg = dataclasses.replace(cfg, num_workers=int(num_workers))
    if num_train_steps is not None:
        cfg = dataclasses.replace(cfg, num_train_steps=int(num_train_steps))
    if log_interval is not None:
        cfg = dataclasses.replace(cfg, log_interval=int(log_interval))
    if save_interval is not None:
        cfg = dataclasses.replace(cfg, save_interval=int(save_interval))
    if keep_period is not None:
        cfg = dataclasses.replace(cfg, keep_period=int(keep_period))
    if fsdp_devices is not None:
        cfg = dataclasses.replace(cfg, fsdp_devices=int(fsdp_devices))
    if seed is not None:
        cfg = dataclasses.replace(cfg, seed=int(seed))
    if overwrite is not None:
        cfg = dataclasses.replace(cfg, overwrite=bool(overwrite))
    if resume is not None:
        cfg = dataclasses.replace(cfg, resume=bool(resume))
    if wandb_enabled is not None:
        cfg = dataclasses.replace(cfg, wandb_enabled=bool(wandb_enabled))
    if project_name is not None:
        cfg = dataclasses.replace(cfg, project_name=str(project_name))
    if jax_params_path is not None:
        if str(jax_params_path).startswith("gs://"):
            resolved_jax_params_path = str(jax_params_path)
        else:
            resolved_jax_params_path = str(Path(jax_params_path).expanduser().resolve())
        cfg = dataclasses.replace(
            cfg,
            weight_loader=openpi_weight_loaders.CheckpointWeightLoader(resolved_jax_params_path),
        )
    if pytorch_weight_path is not None:
        cfg = dataclasses.replace(cfg, pytorch_weight_path=str(pytorch_weight_path))
    if pytorch_training_precision is not None:
        cfg = dataclasses.replace(cfg, pytorch_training_precision=str(pytorch_training_precision))
    return cfg


def resolve_default_jax_params_path() -> Path | None:
    cache_roots = []

    env_cache_root = os.environ.get("OPENPI_DATA_HOME")
    if env_cache_root:
        cache_roots.append(Path(env_cache_root).expanduser())
    cache_roots.append(Path.home() / ".cache" / "openpi")

    relative_path = Path("openpi-assets") / "checkpoints" / "pi05_base" / "params"
    for cache_root in cache_roots:
        candidate = (cache_root / relative_path).resolve()
        if candidate.is_dir():
            return candidate
    return None


def ensure_norm_stats(
    cfg: Any,
    *,
    max_frames: int | None = None,
) -> Path:
    prepare_openpi_imports()
    import openpi.shared.normalize as normalize
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    try:
        import pyarrow.parquet as pq
    except Exception:
        pq = None

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("repo_id is required to compute norm stats")

    output_path = cfg.assets_dirs / repo_id
    try:
        normalize.load(output_path)
        print(f"[PI05_RobotWin] norm stats already exist: {output_path}")
        return output_path
    except FileNotFoundError:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[PI05_RobotWin] computing norm stats -> {output_path}")

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    parquet_root = HF_LEROBOT_HOME / repo_id / "data"
    parquet_files = sorted(parquet_root.glob("chunk-*/*.parquet"))

    if pq is not None and parquet_files:
        frames_seen = 0
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file, columns=["observation.state", "action"])
            state_batch = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
            action_batch = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
            if max_frames is not None and frames_seen + state_batch.shape[0] > max_frames:
                keep = max_frames - frames_seen
                if keep <= 0:
                    break
                state_batch = state_batch[:keep]
                action_batch = action_batch[:keep]
            if state_batch.size == 0:
                continue
            stats["state"].update(state_batch)
            stats["actions"].update(action_batch)
            frames_seen += state_batch.shape[0]
            if max_frames is not None and frames_seen >= max_frames:
                break
    else:
        compute_mod = load_module_from_path("pi05_compute_norm_stats", PI05_ROOT / "scripts" / "compute_norm_stats.py")
        if data_config.rlds_data_dir is not None:
            data_loader, num_batches = compute_mod.create_rlds_dataloader(
                data_config,
                cfg.model.action_horizon,
                cfg.batch_size,
                max_frames,
            )
        else:
            data_loader, num_batches = compute_mod.create_torch_dataloader(
                data_config,
                cfg.model.action_horizon,
                cfg.batch_size,
                cfg.model,
                cfg.num_workers,
                max_frames,
            )

        for batch in data_loader:
            for key in ("state", "actions"):
                stats[key].update(batch[key])
            num_batches -= 1
            if num_batches <= 0:
                break

    norm_stats = {key: stats[key].get_statistics() for key in ("state", "actions")}
    normalize.save(output_path, norm_stats)
    print(f"[PI05_RobotWin] wrote norm stats: {output_path}")
    return output_path


# Apply import path fixes and monkey patches at import time as well, so spawned
# dataloader workers inherit the same patched behavior as the parent process.
prepare_openpi_imports()
