from __future__ import annotations

import argparse
from pathlib import Path

from .robotwin_common import (
    DEFAULT_ASSETS_BASE_DIR,
    DEFAULT_CAMERA_MODE,
    DEFAULT_CHECKPOINT_BASE_DIR,
    DEFAULT_SECONDARY_CAMERA,
    PI05_ROOT,
    build_train_config,
    ensure_norm_stats,
    load_module_from_path,
    prepare_openpi_imports,
    resolve_default_jax_params_path,
    timestamp,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train PI0.5 open-loop policy for RoboTwin")
    parser.add_argument("--framework", default="auto", choices=["auto", "jax", "pytorch"])
    parser.add_argument("--train-config-name", default="pi05_aloha_full_base")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id used by openpi data loader")
    parser.add_argument("--asset-id", default=None, help="Optional asset id for norm stats; defaults to repo-id")
    parser.add_argument("--camera-mode", default=DEFAULT_CAMERA_MODE, choices=["head_only", "dual_view", "tri_view"])
    parser.add_argument(
        "--secondary-camera",
        default=DEFAULT_SECONDARY_CAMERA,
        choices=["left_wrist", "right_wrist"],
        help="Used only when camera-mode=dual_view.",
    )
    parser.add_argument("--exp-name", default=f"pi05_openloop_{timestamp()}")
    parser.add_argument("--assets-base-dir", default=str(DEFAULT_ASSETS_BASE_DIR))
    parser.add_argument("--checkpoint-base-dir", default=str(DEFAULT_CHECKPOINT_BASE_DIR))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--keep-period", type=int, default=None)
    parser.add_argument("--fsdp-devices", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--wandb-enabled", type=str2bool, default=True)
    parser.add_argument("--project-name", default="RoboTwin_PI05_RobotWin")
    parser.add_argument("--jax-params-path", default=None)
    parser.add_argument("--pytorch-weight-path", default=None)
    parser.add_argument("--pytorch-training-precision", default=None, choices=[None, "bfloat16", "float32"])
    parser.add_argument("--gradient-checkpointing", type=str2bool, default=True)
    parser.add_argument("--find-unused-parameters", type=str2bool, default=True)
    parser.add_argument("--skip-norm-stats", type=str2bool, default=False)
    parser.add_argument("--norm-max-frames", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    prepare_openpi_imports()

    if args.jax_params_path is None:
        default_jax_params_path = resolve_default_jax_params_path()
        if default_jax_params_path is not None:
            args.jax_params_path = str(default_jax_params_path)

    framework = args.framework
    if framework == "auto":
        framework = "jax" if args.jax_params_path else ("pytorch" if args.pytorch_weight_path else "jax")

    print("[PI05_RobotWin] starting open-loop training")
    print(f"[PI05_RobotWin] train_config={args.train_config_name}")
    print(f"[PI05_RobotWin] repo_id={args.repo_id}")
    print(f"[PI05_RobotWin] camera_mode={args.camera_mode}")
    print(f"[PI05_RobotWin] exp_name={args.exp_name}")
    print(f"[PI05_RobotWin] framework={framework}")
    if args.jax_params_path:
        if str(args.jax_params_path).startswith("gs://"):
            print(f"[PI05_RobotWin] jax_params_path={args.jax_params_path}")
        else:
            print(f"[PI05_RobotWin] jax_params_path={Path(args.jax_params_path).expanduser().resolve()}")
    if args.pytorch_weight_path:
        print(f"[PI05_RobotWin] pytorch_weight_path={Path(args.pytorch_weight_path).expanduser().resolve()}")
    print(f"[PI05_RobotWin] assets_base_dir={Path(args.assets_base_dir).expanduser().resolve()}")
    print(f"[PI05_RobotWin] checkpoint_base_dir={Path(args.checkpoint_base_dir).expanduser().resolve()}")

    if framework == "pytorch":
        train_mod = load_module_from_path(
            "robotwin_pi05_train_pytorch_bridge",
            Path(__file__).resolve().parent / "train_robotwin_openloop_pytorch.py",
        )
        train_mod.main()
        return

    cfg = build_train_config(
        train_config_name=args.train_config_name,
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        camera_mode=args.camera_mode,
        secondary_camera=args.secondary_camera,
        asset_id=args.asset_id,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
        project_name=args.project_name,
        jax_params_path=args.jax_params_path,
        pytorch_weight_path=args.pytorch_weight_path,
        pytorch_training_precision=args.pytorch_training_precision,
    )

    if not args.skip_norm_stats:
        ensure_norm_stats(cfg, max_frames=args.norm_max_frames)

    train_mod = load_module_from_path("pi05_train", PI05_ROOT / "scripts" / "train.py")
    train_mod.main(cfg)


if __name__ == "__main__":
    main()
