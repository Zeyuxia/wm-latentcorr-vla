from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _import_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def _has_local_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            text = netrc_path.read_text(encoding="utf-8", errors="ignore")
            if "api.wandb.ai" in text:
                return True
        except Exception:
            pass
    settings_path = Path.home() / ".config" / "wandb" / "settings"
    if settings_path.exists():
        return True
    return False


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    run_name: str | None,
    group: str | None,
    tags: list[str] | None,
    mode: str,
    output_dir: str,
    config: dict[str, Any],
    run_id: str | None = None,
    resume: str | None = None,
) -> Any | None:
    if not enabled:
        print("[wandb] disabled")
        return None

    wandb = _import_wandb()
    if wandb is None:
        print("[wandb] package not available, skip logging")
        return None

    wandb_dir = os.path.join(output_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    run_mode = mode
    if run_mode == "auto":
        run_mode = "online" if _has_local_wandb_credentials() else "offline"

    init_kwargs = dict(
        project=project,
        entity=entity or None,
        name=run_name or None,
        group=group or None,
        tags=tags or None,
        mode=run_mode,
        dir=wandb_dir,
        config=config,
    )
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = resume or "allow"

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        if run_mode == "online":
            print(f"[wandb] online init failed ({exc}), falling back to offline mode")
            init_kwargs["mode"] = "offline"
            run = wandb.init(**init_kwargs)
        else:
            raise

    print(f"[wandb] mode={run.settings.mode} dir={wandb_dir}")
    if getattr(run, "url", None):
        print(f"[wandb] url={run.url}")
    return run


def log_wandb(run: Any | None, metrics: dict[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    payload = {k: v for k, v in metrics.items() if v is not None}
    if not payload:
        return
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=step)


def update_wandb_summary(run: Any | None, summary: dict[str, Any]) -> None:
    if run is None:
        return
    for key, value in summary.items():
        run.summary[key] = value


def finish_wandb(run: Any | None) -> None:
    if run is None:
        return
    run.finish()
