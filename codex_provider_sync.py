#!/usr/bin/env python3
"""Sync Codex conversation history from one provider to another."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync Codex history provider labels from one provider to another. "
            "This updates both state_5.sqlite (threads.model_provider) and "
            "the session JSONL metadata line."
        )
    )
    parser.add_argument(
        "--codex-home",
        default=str(Path.home() / ".codex"),
        help="Codex home directory (default: ~/.codex)",
    )
    parser.add_argument("--from-provider", required=True, help="Source provider name")
    parser.add_argument("--to-provider", required=True, help="Target provider name")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional explicit thread IDs to migrate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of threads to migrate (by newest updated_at)",
    )
    parser.add_argument(
        "--only-unarchived",
        action="store_true",
        help="Only migrate threads where archived=0",
    )
    parser.add_argument(
        "--skip-jsonl",
        action="store_true",
        help="Only update SQLite, skip session JSONL metadata updates",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing anything",
    )
    return parser.parse_args()


def ensure_files(codex_home: Path) -> Path:
    db_path = codex_home / "state_5.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    return db_path


def fetch_threads(
    conn: sqlite3.Connection,
    from_provider: str,
    ids: list[str] | None,
    only_unarchived: bool,
    limit: int | None,
) -> list[sqlite3.Row]:
    clauses = ["model_provider = ?"]
    params: list[object] = [from_provider]

    if ids:
        placeholders = ",".join(["?"] * len(ids))
        clauses.append(f"id IN ({placeholders})")
        params.extend(ids)
    if only_unarchived:
        clauses.append("archived = 0")

    where_sql = " AND ".join(clauses)
    sql = (
        "SELECT id, rollout_path, title, updated_at, archived, model_provider "
        f"FROM threads WHERE {where_sql} ORDER BY updated_at DESC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    cur = conn.execute(sql, params)
    return list(cur.fetchall())


def build_backup_dir(codex_home: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = codex_home / "_provider_sync_backups" / f"sync-{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def backup_db(db_path: Path, backup_dir: Path) -> Path:
    dst = backup_dir / db_path.name
    shutil.copy2(db_path, dst)
    return dst


def backup_session_file(path: Path, codex_home: Path, backup_dir: Path) -> Path:
    backup_root = backup_dir / "sessions"
    try:
        rel = path.relative_to(codex_home)
        dst = backup_root / rel
    except ValueError:
        dst = backup_root / f"external-{path.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst


def update_jsonl_provider(path: Path, to_provider: str) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    tmp = path.with_suffix(path.suffix + ".tmp-provider-sync")
    try:
        with path.open("r", encoding="utf-8") as fin:
            first = fin.readline()
            if not first:
                return False, "empty"
            obj = json.loads(first)
            if obj.get("type") != "session_meta" or not isinstance(obj.get("payload"), dict):
                return False, "no_session_meta"
            obj["payload"]["model_provider"] = to_provider
            with tmp.open("w", encoding="utf-8") as fout:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                shutil.copyfileobj(fin, fout)
        tmp.replace(path)
        return True, "updated"
    except json.JSONDecodeError:
        if tmp.exists():
            tmp.unlink()
        return False, "bad_first_line"
    except OSError as exc:
        if tmp.exists():
            tmp.unlink()
        return False, f"io_error:{exc}"


def update_db_provider(conn: sqlite3.Connection, ids: Iterable[str], to_provider: str) -> int:
    rows = [(to_provider, thread_id) for thread_id in ids]
    conn.executemany("UPDATE threads SET model_provider = ? WHERE id = ?", rows)
    return conn.total_changes


def main() -> int:
    args = parse_args()
    codex_home = Path(args.codex_home).expanduser().resolve()
    db_path = ensure_files(codex_home)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        threads = fetch_threads(
            conn=conn,
            from_provider=args.from_provider,
            ids=args.ids,
            only_unarchived=args.only_unarchived,
            limit=args.limit,
        )
        if not threads:
            print("No threads matched. Nothing to do.")
            return 0

        print(f"Matched {len(threads)} thread(s).")
        for row in threads[:20]:
            updated_at = dt.datetime.utcfromtimestamp(row["updated_at"]).isoformat() + "Z"
            print(f"- {row['id']} | archived={row['archived']} | {updated_at} | {row['title'][:80]}")
        if len(threads) > 20:
            print(f"... ({len(threads) - 20} more)")

        if args.dry_run:
            print("Dry run enabled: no files were modified.")
            return 0

        backup_dir = build_backup_dir(codex_home)
        db_backup = backup_db(db_path, backup_dir)
        print(f"DB backup: {db_backup}")

        migrated_ids = [row["id"] for row in threads]
        changed = update_db_provider(conn, migrated_ids, args.to_provider)
        conn.commit()
        print(f"SQLite rows updated: {changed}")

        jsonl_updated = 0
        jsonl_skipped = 0
        if args.skip_jsonl:
            print("Skipped JSONL updates by --skip-jsonl.")
        else:
            for row in threads:
                session_path = Path(row["rollout_path"])
                if session_path.exists():
                    backup_session_file(session_path, codex_home, backup_dir)
                ok, reason = update_jsonl_provider(session_path, args.to_provider)
                if ok:
                    jsonl_updated += 1
                else:
                    jsonl_skipped += 1
                    print(f"JSONL skip: {row['id']} ({reason}) path={session_path}")
        print(f"JSONL updated: {jsonl_updated}, skipped: {jsonl_skipped}")
        print(f"Backup directory: {backup_dir}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
