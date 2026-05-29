#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from write_parallel_eval_summary import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-root", action="append", required=True)
    parser.add_argument(
        "--result-root",
        default="/data/zhenyangfan/RoboTwin/eval_result/open_laptop/PI0_LatentCorr/demo_clean",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_root = Path(args.result_root)

    group_rows = []
    total_num = 0
    total_den = 0
    completed_groups = 0

    for log_root_str in args.log_root:
        log_root = Path(log_root_str)
        run_tag = log_root.joinpath("run_tag.txt").read_text().strip()
        seed_dir = Path(log_root.joinpath("seed_dir.txt").read_text().strip())
        rows, num, den, complete = summarize(run_tag, log_root, seed_dir, result_root)
        total_num += num
        total_den += den
        completed_groups += 1 if complete == len(rows) else 0
        group_rows.append(
            {
                "run_tag": run_tag,
                "log_root": str(log_root),
                "num": num,
                "den": den,
                "rate": (num / den) if den else 0.0,
                "status": "completed" if complete == len(rows) else "partial",
            }
        )

    summary_tsv = output_dir / "summary.tsv"
    summary_txt = output_dir / "summary.txt"

    with summary_tsv.open("w", encoding="utf-8") as f:
        f.write("run_tag\tstatus\tnum\tden\trate\tlog_root\n")
        for row in group_rows:
            f.write(
                f"{row['run_tag']}\t{row['status']}\t{row['num']}\t{row['den']}\t{row['rate']:.6f}\t{row['log_root']}\n"
            )
        total_rate = (total_num / total_den) if total_den else 0.0
        overall_status = "completed" if completed_groups == len(group_rows) else "partial"
        f.write(f"TOTAL\t{overall_status}\t{total_num}\t{total_den}\t{total_rate:.6f}\t\n")

    with summary_txt.open("w", encoding="utf-8") as f:
        overall_status = "completed" if completed_groups == len(group_rows) else "partial"
        f.write(f"status: {overall_status}\n")
        f.write(f"groups: {len(group_rows)}\n")
        f.write(f"total: {total_num}/{total_den} = {(total_num / total_den) if total_den else 0.0:.2%}\n\n")
        for row in group_rows:
            f.write(f"{row['run_tag']}: {row['num']}/{row['den']} = {row['rate']:.2%} | {row['log_root']}\n")
   

if __name__ == "__main__":
    main()
