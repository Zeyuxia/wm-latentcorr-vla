#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/data/zhenyangfan/RoboTwin")
POLICY_ROOT = REPO_ROOT / "policy" / "ACT_LatentCorr"
EVAL_ROOT = REPO_ROOT / "eval_result" / "open_laptop" / "ACT_LatentCorr" / "demo_clean"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def read_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def latest_ckpt(ckpt_dir: str | Path, prefix: str) -> str | None:
    pattern = os.path.join(str(ckpt_dir), f"{prefix}_epoch_*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def extract_epoch_from_ckpt(path: str | None) -> int:
    if not path:
        return -1
    m = re.search(r"_epoch_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def wait_for_stage1(stage1_output_dir: str, target_epoch: int, tmux_session: str | None, poll_sec: int = 120) -> str:
    final_ckpt = os.path.join(stage1_output_dir, f"stage1_epoch_{target_epoch:04d}.pt")
    log(f"等待 stage1 完成，目标 checkpoint: {final_ckpt}")
    heartbeat = 0
    while True:
        if os.path.isfile(final_ckpt):
            log(f"检测到 stage1 最终 checkpoint: {final_ckpt}")
            return final_ckpt

        latest = latest_ckpt(stage1_output_dir, "stage1")
        latest_epoch = extract_epoch_from_ckpt(latest)
        session_alive = tmux_session_exists(tmux_session) if tmux_session else True
        if heartbeat % max(1, 1800 // poll_sec) == 0:
            log(
                f"stage1 仍在等待 | latest_epoch={latest_epoch} "
                f"| tmux_alive={session_alive} | output_dir={stage1_output_dir}"
            )
        if (not session_alive) and latest_epoch < target_epoch:
            raise RuntimeError(
                f"stage1 tmux session `{tmux_session}` 已结束，但没有检测到目标 checkpoint。"
            )
        heartbeat += 1
        time.sleep(poll_sec)


def run_blocking(cmd: list[str], env: dict[str, str], log_path: str, cwd: str = str(REPO_ROOT)) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log(f"启动阻塞任务: {' '.join(cmd)}")
    log(f"日志写入: {log_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"命令失败({ret}): {' '.join(cmd)}")


def run_parallel_logged(jobs: list[dict]) -> None:
    processes = []
    files = []
    try:
        for job in jobs:
            os.makedirs(os.path.dirname(job["log_path"]), exist_ok=True)
            f = open(job["log_path"], "w", encoding="utf-8")
            files.append(f)
            log(f"启动评测: {job['name']} | gpu={job['gpu_id']} | log={job['log_path']}")
            p = subprocess.Popen(
                job["cmd"],
                cwd=job.get("cwd", str(REPO_ROOT)),
                env=job["env"],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((job, p))

        while processes:
            remaining = []
            for job, p in processes:
                ret = p.poll()
                if ret is None:
                    remaining.append((job, p))
                    continue
                if ret != 0:
                    raise RuntimeError(f"评测失败({ret}): {job['name']}")
                log(f"评测完成: {job['name']}")
            processes = remaining
            if processes:
                time.sleep(30)
    finally:
        for _, p in processes:
            if p.poll() is None:
                p.terminate()
        for f in files:
            f.close()


def latest_result_under(setting_name: str) -> str:
    result_dir = EVAL_ROOT / setting_name
    candidates = sorted(result_dir.glob("*/_result.txt"))
    if not candidates:
        raise FileNotFoundError(f"没有找到评测结果: {result_dir}")
    return str(candidates[-1])


def parse_success_rate(result_path: str) -> float:
    text = read_text(result_path)
    matches = re.findall(r"([-+]?\d+(?:\.\d+)?)", text)
    if not matches:
        raise ValueError(f"无法从结果文件解析成功率: {result_path}")
    return float(matches[-1])


@dataclass
class EvalResult:
    name: str
    mode: str
    ckpt_path: str
    ckpt_setting: str
    result_path: str
    success_rate: float


def write_summary(summary_dir: str, stage1_ckpt: str, stage2_ckpt: str, eval_results: list[EvalResult]) -> None:
    os.makedirs(summary_dir, exist_ok=True)
    result_map = {item.name: item for item in eval_results}
    pre = result_map["pre_openloop_base"]
    post = result_map["post_closedloop_teacher"]
    delta = post.success_rate - pre.success_rate

    txt_path = os.path.join(summary_dir, "FINAL_SUMMARY_CN.txt")
    json_path = os.path.join(summary_dir, "FINAL_SUMMARY.json")

    lines = [
        "自动实验流水线结果",
        "",
        f"stage1 最终 checkpoint: {stage1_ckpt}",
        f"stage2 最终 checkpoint: {stage2_ckpt}",
        "",
        "主对比结果：",
        f"1. 闭环前（stage1 + base）成功率: {pre.success_rate:.4f}",
        f"2. 闭环后（stage2 + teacher）成功率: {post.success_rate:.4f}",
        f"3. 成功率提升: {delta:+.4f}",
        "",
        "附加诊断结果：",
    ]
    for item in eval_results:
        lines.append(
            f"- {item.name}: mode={item.mode}, success={item.success_rate:.4f}, result={item.result_path}"
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    payload = {
        "stage1_ckpt": stage1_ckpt,
        "stage2_ckpt": stage2_ckpt,
        "main_comparison": {
            "pre_openloop_base": asdict(pre),
            "post_closedloop_teacher": asdict(post),
            "success_delta": delta,
        },
        "all_results": [asdict(x) for x in eval_results],
        "generated_at": now_str(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log(f"主结果摘要已写入: {txt_path}")
    log(f"结构化结果已写入: {json_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("自动等待 stage1 -> stage2 -> 成功率评测")
    parser.add_argument("--stage1-output-dir", type=str, required=True)
    parser.add_argument("--stage1-target-epoch", type=int, default=1000)
    parser.add_argument("--stage1-tmux-session", type=str, default="act_latent_stage1_ddp_8gpu_schedfix")
    parser.add_argument("--stage2-output-root", type=str, default=str(POLICY_ROOT / "outputs" / "formal_runs" / "teacher_mainline" / "stage2_auto"))
    parser.add_argument("--summary-dir", type=str, default=str(POLICY_ROOT / "outputs" / "formal_runs" / "teacher_mainline" / "auto_pipeline_summary"))
    parser.add_argument("--seed-file", type=str, default="/data/zhenyangfan/RoboTwin/data_eval/open_laptop/demo_clean_seed100k/seed.txt")
    parser.add_argument("--task-name", type=str, default="open_laptop")
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--stage2-num-epochs", type=int, default=100)
    parser.add_argument("--stage2-save-freq", type=int, default=10)
    parser.add_argument("--stage2-gpu", type=int, default=0)
    parser.add_argument("--eval-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--poll-sec", type=int, default=120)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    os.makedirs(args.summary_dir, exist_ok=True)
    stage1_ckpt = wait_for_stage1(
        stage1_output_dir=args.stage1_output_dir,
        target_epoch=args.stage1_target_epoch,
        tmux_session=args.stage1_tmux_session,
        poll_sec=args.poll_sec,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage2_log = os.path.join(args.summary_dir, f"stage2_auto_{ts}.log")
    stage2_env = os.environ.copy()
    stage2_env.update(
        {
            "STAGE1_CKPT": stage1_ckpt,
            "OUTPUT_ROOT": args.stage2_output_root,
            "NUM_EPOCHS": str(args.stage2_num_epochs),
            "SAVE_FREQ": str(args.stage2_save_freq),
            "CUDA_VISIBLE_DEVICES": str(args.stage2_gpu),
            "WANDB_RUN_NAME": f"teacher_mainline_stage2_auto_{ts}",
            "WANDB_GROUP": "teacher_mainline_stage2_auto",
            "WANDB_LOG_MODE": "auto",
        }
    )
    run_blocking(
        ["bash", str(POLICY_ROOT / "train_stage2.sh")],
        env=stage2_env,
        log_path=stage2_log,
    )

    stage2_dir_candidates = sorted(Path(args.stage2_output_root).glob("*"))
    if not stage2_dir_candidates:
        raise FileNotFoundError(f"stage2 输出目录为空: {args.stage2_output_root}")
    stage2_output_dir = str(stage2_dir_candidates[-1])
    stage2_ckpt = latest_ckpt(stage2_output_dir, "stage2")
    if stage2_ckpt is None:
        raise FileNotFoundError(f"没有找到 stage2 checkpoint: {stage2_output_dir}")
    log(f"stage2 完成，最终 checkpoint: {stage2_ckpt}")

    eval_gpu_list = [int(x.strip()) for x in args.eval_gpus.split(",") if x.strip()]
    if len(eval_gpu_list) < 4:
        raise ValueError("--eval-gpus 至少提供 4 张卡，当前脚本会并行跑 4 组评测")

    eval_specs = [
        ("pre_openloop_base", "base", stage1_ckpt, eval_gpu_list[0]),
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
                "cmd": ["bash", str(POLICY_ROOT / "eval_success.sh")],
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

    write_summary(args.summary_dir, stage1_ckpt=stage1_ckpt, stage2_ckpt=stage2_ckpt, eval_results=eval_results)
    log("自动流水线全部完成。")


if __name__ == "__main__":
    main()
