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
    parser.add_argument("--take", type=int, default=25)
    args = parser.parse_args()

    log_root = Path(args.log_root)
    seed_dir = Path(args.seed_dir)
    seed_order = parse_seed_order(seed_dir)
    statuses = parse_log_statuses(log_root)

    ordered = []
    for seed in seed_order:
        status = statuses.get(seed, "missing")
        ordered.append({"seed": seed, "status": status})

    nonskip = [row for row in ordered if row["status"] in {"success", "fail"}]
    ref = nonskip[: int(args.take)]
    ref_success = sum(1 for row in ref if row["status"] == "success")
    ref_den = len(ref)
    ref_rate = (ref_success / ref_den) if ref_den else 0.0
    skip_count = sum(1 for row in ordered if row["status"] == "skip")
    missing_count = sum(1 for row in ordered if row["status"] == "missing")

    out = {
        "run_tag": args.run_tag,
        "take": int(args.take),
        "total_seeds": len(seed_order),
        "skip_count": skip_count,
        "missing_count": missing_count,
        "nonskip_count": len(nonskip),
        "reference_num": ref_success,
        "reference_den": ref_den,
        "reference_rate": ref_rate,
        "reference_seeds": ref,
    }

    out_json = log_root / "reference_first25_nonskip.json"
    out_txt = log_root / "reference_first25_nonskip.txt"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"run_tag: {args.run_tag}\n")
        f.write(f"take: {args.take}\n")
        f.write(f"total_seeds: {len(seed_order)}\n")
        f.write(f"skip_count: {skip_count}\n")
        f.write(f"missing_count: {missing_count}\n")
        f.write(f"nonskip_count: {len(nonskip)}\n")
        f.write(f"reference: {ref_success}/{ref_den} = {ref_rate:.2%}\n")
        f.write("reference_seeds:\n")
        for row in ref:
            f.write(f"  {row['seed']}: {row['status']}\n")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
