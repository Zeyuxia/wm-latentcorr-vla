from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_model

from .common import DEFAULT_ASSETS_BASE_DIR, DEFAULT_CHECKPOINT_BASE_DIR, prepare_openpi_imports


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export Stage1 PI0 checkpoint to deployable open-loop checkpoint")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--train-config-name", default="pi05_aloha_full_base")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-mode", default="head_only", choices=["head_only", "tri_view"])
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--assets-base-dir", default=str(DEFAULT_ASSETS_BASE_DIR))
    parser.add_argument("--checkpoint-base-dir", default=str(DEFAULT_CHECKPOINT_BASE_DIR))
    return parser


def infer_checkpoint_id(stage1_ckpt: Path) -> str:
    stem = stage1_ckpt.stem
    if stem.startswith("stage1_step_"):
        return str(int(stem.split("_")[-1]))
    return "latest"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    prepare_openpi_imports()

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.training import config as openpi_config

    stage1_ckpt = Path(args.stage1_ckpt).expanduser().resolve()
    if not stage1_ckpt.is_file():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {stage1_ckpt}")

    checkpoint_id = str(args.checkpoint_id or infer_checkpoint_id(stage1_ckpt))
    checkpoint_dir = (
        Path(args.checkpoint_base_dir).expanduser().resolve() / args.train_config_name / args.exp_name / checkpoint_id
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(stage1_ckpt, map_location="cpu")
    stage1_state = ckpt["model"]
    base_state = {}
    prefix = "base_pi0."
    for key, value in stage1_state.items():
        if key.startswith(prefix):
            base_state[key[len(prefix) :]] = value
    if not base_state:
        raise ValueError(f"No base_pi0 weights found in checkpoint: {stage1_ckpt}")

    train_cfg = openpi_config.get_config(args.train_config_name)
    model = PI0Pytorch(config=train_cfg.model)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Failed to load exported base_pi0 state strictly. missing={list(missing)[:10]} unexpected={list(unexpected)[:10]}"
        )
    model = model.cpu()
    save_model(model, str(checkpoint_dir / "model.safetensors"))

    src_norm_stats = (
        Path(args.assets_base_dir).expanduser().resolve() / args.train_config_name / args.repo_id / "norm_stats.json"
    )
    if not src_norm_stats.is_file():
        raise FileNotFoundError(f"Norm stats not found: {src_norm_stats}")
    dst_norm_stats = checkpoint_dir / "assets" / args.repo_id / "norm_stats.json"
    dst_norm_stats.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_norm_stats, dst_norm_stats)

    write_json(
        checkpoint_dir / "export_meta.json",
        {
            "source_stage1_ckpt": str(stage1_ckpt),
            "train_config_name": args.train_config_name,
            "repo_id": args.repo_id,
            "camera_mode": args.camera_mode,
            "exp_name": args.exp_name,
            "checkpoint_id": checkpoint_id,
            "step": int(ckpt.get("step", 0)),
            "sample_count": int(ckpt.get("sample_count", 0)),
        },
    )
    print(f"[PI0_LatentCorr] exported deployable checkpoint -> {checkpoint_dir}")
    print(f"[PI0_LatentCorr] model -> {checkpoint_dir / 'model.safetensors'}")
    print(f"[PI0_LatentCorr] assets -> {dst_norm_stats.parent}")


if __name__ == "__main__":
    main()
