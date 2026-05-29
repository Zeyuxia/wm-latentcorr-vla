#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def extract_score(result_path: Path) -> float:
    lines = [line.strip() for line in result_path.read_text().splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return float(line)
        except ValueError:
            continue
    raise ValueError(f"no numeric score found in {result_path}")


def latest_result_for_shard(result_root: Path, run_tag: str, shard_idx: int) -> Path | None:
    pattern = f"{run_tag}_shard_{shard_idx}"
    candidates = sorted(result_root.glob(f"**/*{pattern}/_result.txt"))
    return candidates[-1] if candidates else None


def summarize(run_tag: str, log_root: Path, seed_dir: Path, result_root: Path) -> tuple[list[dict], int, int, int]:
    rows: list[dict] = []
    total_num = 0
    total_den = 0
    complete = 0
    log_root.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(seed_dir.glob("shard_*.txt"))
    for shard_file in shard_files:
        shard_idx = int(shard_file.stem.split("_")[-1])
        seeds = [line.strip() for line in shard_file.read_text().splitlines() if line.strip()]
        den = len(seeds)
        result_path = latest_result_for_shard(result_root, run_tag, shard_idx)
        if result_path is None:
            rows.append(
                {
                    "shard": shard_idx,
                    "status": "missing",
                    "num": "",
                    "den": den,
                    "rate": "",
                    "result_path": "",
                }
            )
            continue

        rate = extract_score(result_path)
        num = int(round(rate * den))
        total_num += num
        total_den += den
        complete += 1
        rows.append(
            {
                "shard": shard_idx,
                "status": "completed",
                "num": num,
                "den": den,
                "rate": rate,
                "result_path": str(result_path),
            }
        )

    summary_tsv = log_root / "summary.tsv"
    summary_txt = log_root / "summary.txt"

    with summary_tsv.open("w", encoding="utf-8") as f:
        f.write("run_tag\tshard\tstatus\tnum\tden\trate\tresult_path\n")
        for row in rows:
            rate_str = "" if row["rate"] == "" else f"{row['rate']:.6f}"
            f.write(
                f"{run_tag}\t{row['shard']}\t{row['status']}\t{row['num']}\t{row['den']}\t{rate_str}\t{row['result_path']}\n"
            )
        total_rate = (total_num / total_den) if total_den else 0.0
        overall_status = "completed" if complete == len(rows) else "partial"
        f.write(f"{run_tag}\tTOTAL\t{overall_status}\t{total_num}\t{total_den}\t{total_rate:.6f}\t\n")

    total_rate = (total_num / total_den) if total_den else 0.0
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"run_tag: {run_tag}\n")
        f.write(f"status: {'completed' if complete == len(rows) else 'partial'}\n")
        f.write(f"result_files: {complete}/{len(rows)}\n")
        f.write(f"total: {total_num}/{total_den} = {total_rate:.2%}\n\n")
        for row in rows:
            if row["status"] == "completed":
                f.write(
                    f"shard_{row['shard']}: {row['num']}/{row['den']} = {row['rate']:.2%} | {row['result_path']}\n"
                )
            else:
                f.write(f"shard_{row['shard']}: missing (den={row['den']})\n")

    return rows, total_num, total_den, complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument(
        "--result-root",
        default="/data/zhenyangfan/RoboTwin/eval_result/open_laptop/PI0_LatentCorr/demo_clean",
    )
    args = parser.parse_args()

    summarize(
        run_tag=args.run_tag,
        log_root=Path(args.log_root),
        seed_dir=Path(args.seed_dir),
        result_root=Path(args.result_root),
    )


if __name__ == "__main__":
    main()
