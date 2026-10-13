"""Atomic, file-backed progress state for video-processing jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STATUS_FILENAME = "status.json"
PROCESSING_STATE_FILENAME = "processing_state.json"


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    temporary_path.replace(path)


def read_job_status(job_directory: Path) -> dict[str, Any]:
    """Read the latest complete status snapshot."""
    status_path = job_directory / STATUS_FILENAME
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError("Video job status is not available.") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Video job status is temporarily unavailable.") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Video job status is invalid.")
    return payload


def write_job_status(
    job_directory: Path,
    *,
    status: str,
    stage: str,
    current_frame: int = 0,
    total_frames: int = 0,
    progress_percent: float = 0,
    message: str = "",
    error: str | None = None,
    completed_stages: list[str] | None = None,
) -> dict[str, Any]:
    """Persist one monotonic status snapshot and retain the legacy state file."""
    existing: dict[str, Any] = {}
    try:
        existing = read_job_status(job_directory)
    except (FileNotFoundError, RuntimeError):
        pass

    previous_progress = float(existing.get("progress_percent", 0) or 0)
    progress = max(previous_progress, min(100.0, float(progress_percent)))
    payload: dict[str, Any] = {
        "job_id": job_directory.name,
        "status": status,
        "stage": stage,
        "current_stage": stage,
        "current_frame": max(0, int(current_frame)),
        "total_frames": max(0, int(total_frames)),
        "progress_percent": round(progress, 1),
        "message": message,
        "completed_stages": list(completed_stages or existing.get("completed_stages", [])),
    }
    if error:
        payload["error"] = error
    _atomic_write(job_directory / STATUS_FILENAME, payload)
    _atomic_write(
        job_directory / PROCESSING_STATE_FILENAME,
        {
            "current_stage": stage,
            "current_frame": payload["current_frame"],
            "total_frames": payload["total_frames"],
            **({"error": error} if error else {}),
        },
    )
    return payload
