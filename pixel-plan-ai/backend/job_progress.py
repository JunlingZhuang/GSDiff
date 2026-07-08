from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


PROGRESS_FILE_ENV = "PIXEL_PLAN_PROGRESS_FILE"

# append_job_event writes a typed event timeline beside the snapshot for replay/debugging;
# the snapshot file stays the UI contract (docs/claude-code-lessons.md #6).
EVENTS_SUFFIX = ".events.jsonl"


def publish_job_progress(
    iterations: list[dict[str, Any]],
    *,
    phase: str,
    result: dict[str, Any] | None = None,
    code: str | None = None,
    source: str | None = None,
    model: str | None = None,
    message: str | None = None,
) -> None:
    """Atomically publish the latest agent checkpoint for the parent server."""

    raw_path = os.environ.get(PROGRESS_FILE_ENV, "")
    if not raw_path:
        return
    path = Path(raw_path)
    preview: dict[str, Any] | None = None
    if code is not None or result is not None:
        preview = {
            "code": code,
            "source": source,
            "model": model,
        }
        if result is not None:
            preview["plan"] = result.get("plan")
            preview["validation"] = result.get("validation")
    payload = {
        "status": "running",
        "phase": phase,
        "message": message or "",
        "iterations": iterations,
        "preview": preview,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_job_event(event: dict[str, Any]) -> None:
    """Append one typed event to the per-job JSONL timeline for replay/debugging.

    No-op when PIXEL_PLAN_PROGRESS_FILE is unset (same convention as
    publish_job_progress). Each event is stamped with "t": time.time() rounded
    to 3 decimals. The snapshot file remains the UI contract; this is additive.
    """

    raw_path = os.environ.get(PROGRESS_FILE_ENV, "")
    if not raw_path:
        return
    events_path = Path(raw_path + EVENTS_SUFFIX)
    payload = {**event, "t": round(time.time(), 3)}
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
