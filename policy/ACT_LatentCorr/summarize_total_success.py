#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PAT_SKIP = re.compile(r"skip seed due to .*?:\s*(\d+)")
PAT_CURRENT = re.compile(r"current seed:\s*(\d+)")


def parse_seed_order(seed_dir: Path) -> list[int]:
    shard_files = sorted(seed_dir.glob("shard_*.txt"))
    seed_order: list[int] = []
    for shard_file in shard_files:
        for line in shard_file.read_text().splitlines():
            line = line.strip()
            if line:
                seed_order.append(int(line))
    return sorted(seed_order)


def parse_log_statuses(log_root: Path) -> dict[int, str]:
    statuses: dict[int, str] = {}
    for log_file in sorted(log_root.glob("shard_*.log")):
        pending: str | None = None
        for raw_line in log_file.read_text(errors="ignore").splitlines():
            line = raw_line.strip()
            m_skip = PAT_SKIP.search(line)
            if m_skip:
                statuses[int(m_skip.group(1))] = "skip"
                pending = None
                continue
            if line == "Success!":
                pending = "success"
                continue
            if line == "Fail!":
                pending = "fail"
                continue
            m_curr = PAT_CURRENT.search(line)
            if m_curr and pending is not None:
                statuses[int(m_curr.group(1))] = pending
                pending = None
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--seed-dir", required=True)
    args = parser.parse_args()

    log_root = Path(args.log_root)
    seed_dir = Path(args.seed_dir)
    seed_order = parse_seed_order(seed_dir)
    statuses = parse_log_statuses(log_root)

    ordered = []
    for seed in seed_order:
        status = statuses.get(seed, "missing")
        ordered.append({"seed": seed, "status": status})

    success_count = sum(1 for row in ordered if row["status"] == "success")
    fail_count = sum(1 for row in ordered if row["status"] == "fail")
    skip_count = sum(1 for row in ordered if row["status"] == "skip")
    missing_count = sum(1 for row in ordered if row["status"] == "missing")
    completed_count = success_count + fail_count
    total = len(ordered)
    success_rate = (success_count / total) if total else 0.0
    completed_rate = (completed_count / total) if total else 0.0

    out = {
        "run_tag": args.run_tag,
        "total_seeds": total,
        "success_count": success_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "missing_count": missing_count,
        "completed_count": completed_count,
        "success_rate": success_rate,
        "completed_rate": completed_rate,
        "results": ordered,
    }

    out_json = log_root / "total_success.json"
    out_txt = log_root / "total_success.txt"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"run_tag: {args.run_tag}\n")
        f.write(f"total_seeds: {total}\n")
        f.write(f"success_count: {success_count}\n")
        f.write(f"fail_count: {fail_count}\n")
        f.write(f"skip_count: {skip_count}\n")
        f.write(f"missing_count: {missing_count}\n")
        f.write(f"completed_count: {completed_count}\n")
        f.write(f"total_success: {success_count}/{total} = {success_rate:.2%}\n")
        f.write(f"completed_ratio: {completed_count}/{total} = {completed_rate:.2%}\n")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
