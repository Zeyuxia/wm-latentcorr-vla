from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TASK_RULES: dict[str, dict[str, Any]] = {
    "sim-place_burger_fries-demo_clean-50": {
        "phase_key": "pregrasp",
        "phase_instance_idx": 1,
        "error_modes": ["translation", "rotation"],
    },
    "sim-open_laptop-demo_clean-50": {
        "phase_key": "pregrasp",
        "phase_instance_idx": 1,
        "error_modes": ["gripper_close"],
    },
    "sim-handover_block-demo_clean-50": {
        "phase_key": "pregrasp",
        "phase_instance_idx": 1,
        "error_modes": ["translation", "rotation"],
    },
    "sim-pick_dual_bottles-demo_clean-50": {
        "phase_key": "pregrasp",
        "phase_instance_idx": 1,
        "error_modes": ["translation", "rotation"],
    },
    "sim-put_bottles_dustbin-demo_clean-50": {
        "phase_key": "pregrasp",
        "phase_instance_idx": 1,
        "error_modes": ["translation", "rotation"],
    },
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _normalize_rules_payload(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Task rules JSON must be a dict[str, dict].")
    out: dict[str, dict[str, Any]] = {}
    for task_name, rule in payload.items():
        if not isinstance(rule, dict):
            raise ValueError(f"Rule for task {task_name!r} must be a dict.")
        phase_key = str(rule.get("phase_key", "pregrasp")).strip().lower()
        phase_instance_idx = int(rule.get("phase_instance_idx", 1))
        error_modes_raw = rule.get("error_modes", [])
        if isinstance(error_modes_raw, str):
            error_modes = [item.strip().lower() for item in error_modes_raw.replace(";", ",").split(",") if item.strip()]
        else:
            error_modes = [str(item).strip().lower() for item in list(error_modes_raw) if str(item).strip()]
        if not error_modes:
            raise ValueError(f"Rule for task {task_name!r} must specify non-empty error_modes.")
        out[str(task_name)] = {
            "phase_key": phase_key,
            "phase_instance_idx": phase_instance_idx,
            "error_modes": error_modes,
        }
    if not out:
        raise ValueError("No task rules provided.")
    return out


def _load_task_rules(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return dict(DEFAULT_TASK_RULES)
    return _normalize_rules_payload(_load_json(Path(path).resolve()))


def _filter_entries(entries: list[dict[str, Any]], *, phase_key: str, phase_instance_idx: int, error_modes: list[str]) -> list[dict[str, Any]]:
    allowed_modes = {str(item).strip().lower() for item in error_modes}
    kept: list[dict[str, Any]] = []
    for entry in entries:
        if str(entry.get("phase_key", "")).strip().lower() != phase_key:
            continue
        if int(entry.get("phase_instance_idx", -1)) != int(phase_instance_idx):
            continue
        if str(entry.get("error_mode", "")).strip().lower() not in allowed_modes:
            continue
        kept.append(entry)
    return kept


def _count_by(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(entry.get(key, "unknown"))
        counts[value] = int(counts.get(value, 0) + 1)
    return counts


def filter_failure_manifest(
    *,
    input_manifest: str,
    output_root: str,
    task_rules_json: str | None = None,
) -> str:
    manifest_path = Path(input_manifest).resolve()
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected dict manifest at {manifest_path}")
    task_failure_tables = manifest.get("task_failure_tables")
    if not isinstance(task_failure_tables, dict) or not task_failure_tables:
        raise ValueError(f"Missing task_failure_tables in {manifest_path}")

    task_rules = _load_task_rules(task_rules_json)
    output_root_path = Path(output_root).resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    filtered_paths: dict[str, str] = {}
    filtered_stats: dict[str, dict[str, Any]] = {}

    for task_name, rule in task_rules.items():
        if task_name not in task_failure_tables:
            raise KeyError(f"Task {task_name!r} not found in manifest {manifest_path}")
        source_table_path = Path(task_failure_tables[task_name]).resolve()
        table = _load_json(source_table_path)
        if not isinstance(table, dict):
            raise ValueError(f"Expected dict failure table at {source_table_path}")
        entries = table.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Missing entries list in {source_table_path}")

        kept_entries = _filter_entries(
            entries,
            phase_key=str(rule["phase_key"]),
            phase_instance_idx=int(rule["phase_instance_idx"]),
            error_modes=list(rule["error_modes"]),
        )
        if not kept_entries:
            raise ValueError(
                "No entries left after filtering "
                f"task={task_name!r} phase={rule['phase_key']!r} "
                f"phase_instance_idx={rule['phase_instance_idx']} error_modes={rule['error_modes']!r}"
            )

        task_dir = output_root_path / task_name.replace("/", "_")
        out_table_path = task_dir / "failure_table.json"
        out_payload = dict(table)
        out_payload["entries"] = kept_entries
        out_payload["source_manifest"] = str(manifest_path)
        out_payload["source_failure_table"] = str(source_table_path)
        out_payload["filter_rule"] = {
            "phase_key": str(rule["phase_key"]),
            "phase_instance_idx": int(rule["phase_instance_idx"]),
            "error_modes": list(rule["error_modes"]),
        }
        _dump_json(out_table_path, out_payload)

        filtered_paths[str(task_name)] = str(out_table_path)
        filtered_stats[str(task_name)] = {
            "num_entries": int(len(kept_entries)),
            "counts_by_phase": _count_by(kept_entries, "phase_key"),
            "counts_by_error_mode": _count_by(kept_entries, "error_mode"),
            "counts_by_active_arm": _count_by(kept_entries, "active_arm_pattern"),
        }

    out_manifest = {
        "version": 1,
        "source": "filtered_failure_manifest",
        "source_manifest": str(manifest_path),
        "task_failure_tables": filtered_paths,
        "task_stats": filtered_stats,
        "task_rules": task_rules,
    }
    out_manifest_path = output_root_path / "multitask_failure_manifest.json"
    _dump_json(out_manifest_path, out_manifest)
    return str(out_manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a multitask failure manifest down to selected task/phase/error-mode rules.")
    parser.add_argument("--input_manifest", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--task_rules_json", type=str, default="")
    args = parser.parse_args()

    manifest_path = filter_failure_manifest(
        input_manifest=str(args.input_manifest),
        output_root=str(args.output_root),
        task_rules_json=(None if not str(args.task_rules_json).strip() else str(args.task_rules_json)),
    )
    print(f"filtered_failure_manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
