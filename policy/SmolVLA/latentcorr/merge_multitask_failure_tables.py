from __future__ import annotations

import argparse
import json
from pathlib import Path

from policy.SmolVLA.latentcorr.merge_failure_tables import merge_failure_dir


def _task_dir_name(task_name: str) -> str:
    if not task_name.startswith("sim-"):
        return task_name
    parts = task_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected multitask name format: {task_name}")
    return parts[1]


def merge_multitask_failure_dirs(
    source_root: str,
    output_root: str,
    task_names: list[str],
    failure_fail_recover_rate_thresh: float,
) -> str:
    src_root = Path(source_root).resolve()
    dst_root = Path(output_root).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"source_root not found: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    merged_tables: dict[str, str] = {}
    for task_name in task_names:
        task_dir_name = _task_dir_name(task_name)
        task_dir = src_root / task_dir_name
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task failure directory not found: {task_dir}")
        task_out_dir = dst_root / task_dir_name
        task_out_dir.mkdir(parents=True, exist_ok=True)
        merged_table = merge_failure_dir(
            failure_dir=str(task_dir),
            out_dir=str(task_out_dir),
            epoch=None,
            failure_fail_recover_rate_thresh=float(failure_fail_recover_rate_thresh),
        )
        merged_tables[task_name] = merged_table

    manifest = {
        "version": 1,
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "task_names": list(task_names),
        "failure_fail_recover_rate_thresh": float(failure_fail_recover_rate_thresh),
        "task_failure_tables": merged_tables,
    }
    manifest_path = dst_root / "multitask_failure_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return str(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-task failure explore outputs into a multitask manifest.")
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--task_names", nargs="+", required=True)
    parser.add_argument("--failure_fail_recover_rate_thresh", type=float, required=True)
    args = parser.parse_args()
    manifest_path = merge_multitask_failure_dirs(
        source_root=args.source_root,
        output_root=args.output_root,
        task_names=list(args.task_names),
        failure_fail_recover_rate_thresh=float(args.failure_fail_recover_rate_thresh),
    )
    print(f"multitask_failure_manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
