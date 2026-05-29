#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from .auto_pipeline_runner import (
    EVAL_ROOT,
    EvalResult,
    latest_ckpt,
    latest_result_under,
    log,
    parse_success_rate,
    run_parallel_logged,
    write_summary,
)


def wait_for_stage2(stage2_output_dir: str, target_epoch: int, poll_sec: int = 120) -> str:
    final_ckpt = os.path.join(stage2_output_dir, f"stage2_epoch_{target_epoch:04d}.pt")
    log(f"等待 stage2 完成，目标 checkpoint: {final_ckpt}")
    heartbeat = 0
    while True:
        if os.path.isfile(final_ckpt):
            log(f"检测到 stage2 最终 checkpoint: {final_ckpt}")
            return final_ckpt

        latest = latest_ckpt(stage2_output_dir, "stage2")
        latest_epoch = -1
        if latest:
            try:
                latest_epoch = int(os.path.basename(latest).split("_")[-1].split(".")[0])
            except Exception:
                latest_epoch = -1
        if heartbeat % max(1, 1800 // poll_sec) == 0:
            log(f"stage2 仍在等待 | latest_epoch={latest_epoch} | output_dir={stage2_output_dir}")
        heartbeat += 1
        time.sleep(poll_sec)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("等待 stage2 完成后自动跑成功率评测")
    parser.add_argument("--stage1-ckpt", type=str, required=True)
    parser.add_argument("--stage2-output-dir", type=str, required=True)
    parser.add_argument("--stage2-target-epoch", type=int, default=100)
    parser.add_argument("--summary-dir", type=str, required=True)
    parser.add_argument("--seed-file", type=str, default="/data/zhenyangfan/RoboTwin/data_eval/open_laptop/demo_clean_seed100k/seed.txt")
    parser.add_argument("--task-name", type=str, default="open_laptop")
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--eval-gpus", type=str, default="0,1,2,3")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    stage2_ckpt = wait_for_stage2(args.stage2_output_dir, args.stage2_target_epoch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_gpu_list = [int(x.strip()) for x in args.eval_gpus.split(",") if x.strip()]
    if len(eval_gpu_list) < 4:
        raise ValueError("--eval-gpus 至少提供 4 张卡")

    eval_specs = [
        ("pre_openloop_base", "base", args.stage1_ckpt, eval_gpu_list[0]),
        ("post_closedloop_teacher", "teacher", stage2_ckpt, eval_gpu_list[1]),
        ("post_closedloop_base", "base", stage2_ckpt, eval_gpu_list[2]),
        ("post_closedloop_bridge", "bridge", stage2_ckpt, eval_gpu_list[3]),
    ]

    jobs = []
    for name, mode, ckpt_path, gpu_id in eval_specs:
        setting_name = f"auto_{ts}_{name}"
        log_path = os.path.join(args.summary_dir, f"{name}.log")
        env = os.environ.copy()
        env.update(
            {
                "TASK_NAME": args.task_name,
                "TASK_CONFIG": args.task_config,
                "CKPT_SETTING": setting_name,
                "LATENT_CKPT_PATH": ckpt_path,
                "INFERENCE_MODE": mode,
                "SEED_FILE": args.seed_file,
                "GPU_ID": str(gpu_id),
            }
        )
        jobs.append(
            {
                "name": name,
                "gpu_id": gpu_id,
                "cmd": ["bash", "/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/eval_success.sh"],
                "env": env,
                "log_path": log_path,
            }
        )

    run_parallel_logged(jobs)

    eval_results: list[EvalResult] = []
    for name, mode, ckpt_path, _ in eval_specs:
        setting_name = f"auto_{ts}_{name}"
        result_path = latest_result_under(setting_name)
        success = parse_success_rate(result_path)
        eval_results.append(
            EvalResult(
                name=name,
                mode=mode,
                ckpt_path=ckpt_path,
                ckpt_setting=setting_name,
                result_path=result_path,
                success_rate=success,
            )
        )
        log(f"{name} 成功率: {success:.4f} | result={result_path}")

    write_summary(args.summary_dir, stage1_ckpt=args.stage1_ckpt, stage2_ckpt=stage2_ckpt, eval_results=eval_results)
    log("stage2 后处理评测全部完成。")


if __name__ == "__main__":
    main()
