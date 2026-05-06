#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


RAW_SUBDIRS = ("data", "video", "instructions", "_traj_data")
STATE_ACTION_SUBDIRS = ("correction_data", "instructions", "metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge correction export rank shards.")
    parser.add_argument("--data-root", default="/data/zhenyangfan/RoboTwin/data")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--format", choices=["raw_episode", "state_action"], default="raw_episode")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def load_json_or_empty(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()
    task_dir = root / args.task
    if args.format == "state_action":
        shard_root = task_dir / f"{args.task_config}_state_action_shards"
        out_root = task_dir / f"{args.task_config}_state_action"
        subdirs = STATE_ACTION_SUBDIRS
    else:
        shard_root = task_dir / f"{args.task_config}_shards"
        out_root = task_dir / args.task_config
        subdirs = RAW_SUBDIRS
    if not shard_root.is_dir():
        raise FileNotFoundError(f"Missing shard root: {shard_root}")
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_root}; pass --overwrite to replace it")
        shutil.rmtree(out_root)
    for subdir in subdirs:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for rank_dir in sorted(path for path in shard_root.iterdir() if path.is_dir() and path.name.startswith("rank")):
        for record in read_manifest(rank_dir / "correction_manifest.jsonl"):
            record = dict(record)
            record["_rank_dir"] = str(rank_dir)
            all_records.append(record)
    all_records.sort(
        key=lambda item: (
            int(item.get("rank", 0)),
            int(item.get("episode_id", item.get("sample_id", 0))),
            str(item.get("export_path", item.get("path", ""))),
        )
    )

    merged_manifest = out_root / "correction_manifest.jsonl"
    with merged_manifest.open("w", encoding="utf-8") as manifest_f:
        for new_id, record in enumerate(all_records):
            rank_dir = Path(record.pop("_rank_dir"))
            if args.format == "state_action":
                old_id = int(record["sample_id"])
                for subdir, ext, prefix in (
                    ("correction_data", "npz", "sample"),
                    ("instructions", "json", "sample"),
                ):
                    src = rank_dir / subdir / f"{prefix}{old_id}.{ext}"
                    dst = out_root / subdir / f"sample{new_id}.{ext}"
                    if not src.is_file():
                        raise FileNotFoundError(f"Missing shard file: {src}")
                    shutil.copy2(src, dst)
                src_meta = rank_dir / "metadata" / f"sample{old_id}.json"
                dst_meta = out_root / "metadata" / f"sample{new_id}.json"
                metadata = load_json_or_empty(src_meta)
                if not metadata:
                    metadata = {
                        **record,
                        "note": "Backfilled during shard merge because source metadata/sampleN.json was missing.",
                    }
                record["merged_sample_id"] = int(new_id)
                record["merged_data_path"] = str(out_root / "correction_data" / f"sample{new_id}.npz")
                record["merged_instruction_path"] = str(out_root / "instructions" / f"sample{new_id}.json")
                record["merged_metadata_path"] = str(out_root / "metadata" / f"sample{new_id}.json")
                metadata.update(
                    {
                        "sample_id": int(new_id),
                        "path": record["merged_data_path"],
                        "instruction_path": record["merged_instruction_path"],
                        "metadata_path": record["merged_metadata_path"],
                        "source_shard_sample_id": int(old_id),
                        "source_shard_dir": str(rank_dir),
                    }
                )
                write_json(dst_meta, metadata)
            else:
                old_id = int(record["episode_id"])
                for subdir, ext in (
                    ("data", "hdf5"),
                    ("video", "mp4"),
                    ("instructions", "json"),
                    ("_traj_data", "pkl"),
                ):
                    src = rank_dir / subdir / f"episode{old_id}.{ext}"
                    dst = out_root / subdir / f"episode{new_id}.{ext}"
                    if not src.is_file():
                        raise FileNotFoundError(f"Missing shard file: {src}")
                    shutil.copy2(src, dst)
                record["merged_episode_id"] = int(new_id)
                record["merged_data_path"] = str(out_root / "data" / f"episode{new_id}.hdf5")
                record["merged_video_path"] = str(out_root / "video" / f"episode{new_id}.mp4")
            manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(
        out_root / "_dataset_info.json",
        {
            "version": 1,
            "format": str(args.format),
            "task": str(args.task),
            "task_config": str(args.task_config),
            "source_shard_root": str(shard_root),
            "num_samples": len(all_records),
            "manifest": str(merged_manifest),
        },
    )

    print(f"Merged {len(all_records)} records to {out_root}")
    print(f"Manifest: {merged_manifest}")


if __name__ == "__main__":
    main()
