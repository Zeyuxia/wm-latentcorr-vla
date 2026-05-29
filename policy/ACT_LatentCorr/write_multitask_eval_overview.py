#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


TASKS = [
    "open_laptop",
    "pick_dual_bottles",
    "put_bottles_dustbin",
    "place_burger_fries",
    "handover_block",
]


def parse_summary(summary_path: Path) -> dict[str, str] | None:
    if not summary_path.exists():
        return None
    data: dict[str, str] = {}
    for line in summary_path.read_text(encoding="utf-8").splitlines():
      if ":" not in line:
          continue
      key, value = line.split(":", 1)
      data[key.strip()] = value.strip()
    return data


def derive_status(summary: dict[str, str] | None, total_success: dict[str, str] | None) -> str:
    if total_success is not None:
        completed = total_success.get("completed_count", "")
        total = total_success.get("total_seeds", "")
        if completed and total:
            if completed == total:
                return "completed"
            return "partial"
    if summary is None:
        return "pending"
    return summary.get("status", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-tag", required=True)
    parser.add_argument(
        "--log-root",
        default="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs",
    )
    parser.add_argument("--out-name", default="multitask_summary")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    out_dir = log_root / args.master_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / f"{args.out_name}.txt"
    out_tsv = out_dir / f"{args.out_name}.tsv"

    rows = []
    for task in TASKS:
        task_dir = log_root / f"{args.master_tag}_{task}"
        summary = parse_summary(task_dir / "summary.txt")
        total_success = parse_summary(task_dir / "total_success.txt")
        if summary is None:
            if total_success is None:
                rows.append((task, "pending", "", "", "", ""))
                continue
        status = derive_status(summary, total_success)
        if summary is None:
            rows.append(
                (
                    task,
                    status,
                    "",
                    "",
                    "" if total_success is None else total_success.get("total_success", ""),
                    "",
                )
            )
            continue
        rows.append(
            (
                task,
                status,
                summary.get("total", ""),
                summary.get("result_files", ""),
                "" if total_success is None else total_success.get("total_success", ""),
                summary.get("reference: ", summary.get("reference", "")),
            )
        )

    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("task\tstatus\ttotal\tresult_files\ttotal_success\treference\n")
        for row in rows:
            f.write("\t".join(row) + "\n")

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"master_tag: {args.master_tag}\n")
        for task, status, total, result_files, total_success, ref in rows:
            f.write(f"{task}: {status}")
            if total:
                f.write(f" | total={total}")
            if result_files:
                f.write(f" | result_files={result_files}")
            if total_success:
                f.write(f" | total_success={total_success}")
            if ref:
                f.write(f" | reference={ref}")
            f.write("\n")

    print(out_txt)


if __name__ == "__main__":
    main()
