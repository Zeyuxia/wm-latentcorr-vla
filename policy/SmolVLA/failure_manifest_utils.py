from __future__ import annotations

import json
from pathlib import Path


def load_failure_table_paths(path: str) -> dict[str, str]:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"failure table manifest/json not found: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {source_path}")
    if "task_failure_tables" in payload:
        mapping = payload["task_failure_tables"]
    else:
        mapping = payload
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected task->path mapping in {source_path}")
    normalized = {str(key): str(value) for key, value in mapping.items()}
    if not normalized:
        raise ValueError(f"Empty failure table mapping in {source_path}")
    return normalized

