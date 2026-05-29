from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .wandb_utils import finish_wandb, init_wandb_run, log_wandb, update_wandb_summary


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def wait_for_manifest(manifest_path: Path, timeout_s: float, poll_interval: float, min_mtime_ns: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if manifest_path.is_file() and manifest_path.stat().st_mtime_ns >= min_mtime_ns:
            manifest = read_json(manifest_path)
            if manifest is not None:
                return manifest
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for wandb manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser("Sync Stage1 metrics to wandb")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--start-timeout", type=float, default=600.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "stage1_wandb_manifest.json"
    metrics_path = output_dir / "stage1_metrics.jsonl"
    summary_path = output_dir / "stage1_wandb_summary.json"
    status_path = output_dir / "stage1_wandb_status.json"
    state_path = output_dir / "stage1_wandb_state.json"
    start_mtime_ns = time.time_ns()

    manifest = wait_for_manifest(manifest_path, args.start_timeout, args.poll_interval, start_mtime_ns)
    previous_state = read_json(state_path) or {}
    same_launch = previous_state.get("launch_id") == manifest.get("launch_id")
    run_id = previous_state.get("run_id") if same_launch else None

    run = init_wandb_run(
        enabled=True,
        project=str(manifest["project"]),
        entity=str(manifest.get("entity") or ""),
        run_name=str(manifest.get("run_name") or ""),
        group=str(manifest.get("group") or ""),
        tags=list(manifest.get("tags") or []),
        mode=str(manifest.get("mode") or "auto"),
        output_dir=str(output_dir),
        config=dict(manifest.get("config") or {}),
        run_id=run_id,
        resume="allow",
    )
    if run is None:
        return

    state = {
        "launch_id": manifest.get("launch_id"),
        "run_id": run.id,
        "run_name": run.name,
        "run_url": getattr(run, "url", None),
        "last_logged_step": int(previous_state.get("last_logged_step", -1)) if same_launch else -1,
        "metrics_offset": int(previous_state.get("metrics_offset", 0)) if same_launch else 0,
    }
    write_json(state_path, state)

    last_summary_mtime_ns = -1
    try:
        while True:
            last_logged_step = int(state.get("last_logged_step", -1))
            metrics_offset = int(state.get("metrics_offset", 0))

            if metrics_path.is_file():
                with metrics_path.open("r", encoding="utf-8") as f:
                    f.seek(metrics_offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        event = json.loads(line)
                        step = int(event.get("step", -1))
                        if step <= last_logged_step:
                            continue
                        metrics = dict(event.get("metrics") or {})
                        if metrics:
                            log_wandb(run, metrics, step=step)
                            last_logged_step = step
                    metrics_offset = f.tell()

                state["last_logged_step"] = last_logged_step
                state["metrics_offset"] = metrics_offset
                write_json(state_path, state)

            if summary_path.is_file():
                summary_mtime_ns = summary_path.stat().st_mtime_ns
                if summary_mtime_ns != last_summary_mtime_ns:
                    summary_payload = read_json(summary_path) or {}
                    update_wandb_summary(run, summary_payload)
                    last_summary_mtime_ns = summary_mtime_ns

            status_payload = read_json(status_path)
            if status_payload is not None:
                update_wandb_summary(run, status_payload)
                if status_payload.get("state") in {"completed", "failed"}:
                    break

            time.sleep(args.poll_interval)
    finally:
        finish_wandb(run)


if __name__ == "__main__":
    main()
