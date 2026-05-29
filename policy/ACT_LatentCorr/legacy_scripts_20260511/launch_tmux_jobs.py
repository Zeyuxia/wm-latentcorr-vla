#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Launch multiple background jobs across GPUs with tmux")
    parser.add_argument("--commands_file", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--session_prefix", type=str, default="latentjob")
    parser.add_argument("--workdir", type=str, default="/data/zhenyangfan/RoboTwin")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/multi_jobs",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _load_commands(path: str) -> list[str]:
    commands = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            commands.append(text)
    return commands


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    commands = _load_commands(args.commands_file)
    if not commands:
        raise RuntimeError(f"No commands found in {args.commands_file}")

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise RuntimeError("No GPUs specified")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx, command in enumerate(commands):
        gpu = gpus[idx % len(gpus)]
        session = f"{args.session_prefix}_{stamp}_{idx:02d}_g{gpu}"
        log_path = os.path.join(args.log_dir, f"{session}.log")
        wrapped = (
            f"cd {args.workdir} && export CUDA_VISIBLE_DEVICES={gpu} && export PYTHONUNBUFFERED=1 && "
            f"{command} > {log_path} 2>&1"
        )
        print(f"[launch] session={session} gpu={gpu}")
        print(f"[launch] command={command}")
        print(f"[launch] log={log_path}")
        if args.dry_run:
            continue
        subprocess.run(["tmux", "new-session", "-d", "-s", session, wrapped], check=True)


if __name__ == "__main__":
    main()
