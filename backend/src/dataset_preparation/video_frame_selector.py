"""Interactive, label-neutral extraction of original-resolution video frames."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np

from .video_review_history import (
    ReviewHistoryStore,
    format_source_frame_warning,
    image_content_sha256,
    load_label_links,
)


EXTRACTED_FRAME_COLUMNS = [
    "source_video",
    "frame_number",
    "timestamp_seconds",
    "saved_filename",
]
VIDEO_REVIEW_STATUS_COLUMNS = [
    "video_filename",
    "status",
    "saved_frame_count",
    "last_reviewed_timestamp",
    "notes",
]
ALLOWED_VIDEO_STATUSES = frozenset({"not_reviewed", "in_progress", "reviewed"})
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4"})
RIGHT_ARROW_KEYS = frozenset({2555904, 65363})
LEFT_ARROW_KEYS = frozenset({2424832, 65361})
DEFAULT_WINDOW_SIZE = (1500, 950)
DISPLAY_MARGIN = 16
FRAME_BORDER_COLOUR = (120, 180, 220)


@dataclass(frozen=True)
class VideoMetadata:
    source_path: Path
    total_frames: int
    fps: float


def count_saved_frames(extracted_frames_csv: str | Path, video_filename: str) -> int:
    """Count existing extraction records for one source video without changing them."""

    csv_path = Path(extracted_frames_csv)
    if not csv_path.exists():
        return 0
    return sum(row["source_video"] == video_filename for row in _read_existing_records(csv_path))


def load_video_review_status(status_csv: str | Path) -> dict[str, dict[str, str]]:
    """Load and validate persistent video statuses keyed by exact filename."""

    path = Path(status_csv)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != VIDEO_REVIEW_STATUS_COLUMNS:
            raise ValueError("video_review_status.csv columns do not match the required schema")
        rows = list(reader)
    statuses: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = row["video_filename"].strip()
        status = row["status"].strip()
        if not filename or status not in ALLOWED_VIDEO_STATUSES:
            raise ValueError("video_review_status.csv contains an invalid filename or status")
        if filename in statuses:
            raise ValueError(f"Duplicate video status row: {filename}")
        statuses[filename] = dict(row)
    return statuses


def _write_video_review_status(status_csv: Path, rows: Mapping[str, Mapping[str, object]]) -> None:
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_csv.with_name(f".{status_csv.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary status output already exists: {temporary}")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=VIDEO_REVIEW_STATUS_COLUMNS)
            writer.writeheader()
            for filename in sorted(rows, key=lambda value: (value.casefold(), value)):
                writer.writerow(rows[filename])
        temporary.replace(status_csv)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_video_review_status(
    status_csv: str | Path,
    videos: list[Path],
    extracted_frames_csv: str | Path,
) -> dict[str, dict[str, str]]:
    """Create missing not-reviewed entries and synchronize saved-frame counts."""

    path = Path(status_csv)
    rows = load_video_review_status(path)
    for video in videos:
        filename = video.name
        count = str(count_saved_frames(extracted_frames_csv, filename))
        if filename not in rows:
            rows[filename] = {
                "video_filename": filename,
                "status": "not_reviewed",
                "saved_frame_count": count,
                "last_reviewed_timestamp": "",
                "notes": "",
            }
        else:
            rows[filename]["saved_frame_count"] = count
    _write_video_review_status(path, rows)
    return rows


def mark_video_status(
    status_csv: str | Path,
    video_filename: str,
    status: str,
    extracted_frames_csv: str | Path,
) -> dict[str, str]:
    """Persist one allowed review status while retaining notes and extraction records."""

    if status not in ALLOWED_VIDEO_STATUSES:
        raise ValueError(f"Unsupported video review status: {status}")
    path = Path(status_csv)
    rows = load_video_review_status(path)
    existing = rows.get(
        video_filename,
        {
            "video_filename": video_filename,
            "status": "not_reviewed",
            "saved_frame_count": "0",
            "last_reviewed_timestamp": "",
            "notes": "",
        },
    )
    updated = {
        **existing,
        "video_filename": video_filename,
        "status": status,
        "saved_frame_count": str(count_saved_frames(extracted_frames_csv, video_filename)),
        "last_reviewed_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    rows[video_filename] = updated
    _write_video_review_status(path, rows)
    return updated


def refresh_saved_frame_count(
    status_csv: str | Path,
    video_filename: str,
    extracted_frames_csv: str | Path,
) -> None:
    rows = load_video_review_status(status_csv)
    if video_filename not in rows:
        return
    rows[video_filename]["saved_frame_count"] = str(count_saved_frames(extracted_frames_csv, video_filename))
    _write_video_review_status(Path(status_csv), rows)


def discover_video_files(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a video directory: {root}")
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES),
        key=lambda path: (path.name.casefold(), path.name),
    )


def select_review_videos(
    videos: list[Path],
    statuses: Mapping[str, Mapping[str, str]],
    *,
    include_reviewed: bool = False,
) -> list[Path]:
    if include_reviewed:
        return list(videos)
    return [
        video for video in videos
        if statuses.get(video.name, {}).get(
            "status", statuses.get(video.name, {}).get("review_status", "not_reviewed")
        ) not in {"reviewed", "COMPLETED"}
    ]


def frame_timestamp_seconds(frame_number: int, fps: float) -> float:
    """Convert a zero-based frame number to its presentation time."""

    if frame_number < 0:
        raise ValueError("frame_number cannot be negative")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    return frame_number / float(fps)


def seconds_to_frame_delta(seconds: float, fps: float) -> int:
    """Convert a navigation interval into at least one decoded frame."""

    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    return max(1, round(seconds * fps))


def jump_frame_number(frame_number: int, total_frames: int, seconds: float, fps: float) -> int:
    """Apply a signed time jump and clamp it to the available frame range."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if not 0 <= frame_number < total_frames:
        raise ValueError("frame_number is outside the video")
    if not np.isfinite(seconds) or seconds == 0:
        raise ValueError("seconds must be a non-zero finite value")
    delta = seconds_to_frame_delta(abs(seconds), fps)
    target = frame_number + (delta if seconds > 0 else -delta)
    return min(total_frames - 1, max(0, target))


def resume_frame_number(video_history: Mapping[str, str], total_frames: int, *, resume: bool) -> int:
    """Choose a safe start frame without changing or clearing existing history."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if not resume or not str(video_history.get("furthest_reviewed_frame", "")).strip():
        return 0
    return min(total_frames - 1, max(0, int(video_history["furthest_reviewed_frame"])))


def _safe_video_stem(video_path: str | Path) -> str:
    stem = Path(video_path).stem.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return safe or "video"


def extracted_frame_filename(video_path: str | Path, frame_number: int, fps: float) -> str:
    """Create a deterministic PNG filename without assigning a particle label."""

    timestamp = frame_timestamp_seconds(frame_number, fps)
    return f"{_safe_video_stem(video_path)}_frame_{frame_number}_time_{timestamp:.3f}.png"


def _read_existing_records(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXTRACTED_FRAME_COLUMNS:
            raise ValueError("extracted_frames.csv columns do not match the required schema")
        return list(reader)


def _append_extraction_record(
    csv_path: Path,
    record: Mapping[str, object],
    *,
    allow_duplicate: bool = False,
) -> None:
    existing = _read_existing_records(csv_path)
    source_video = str(record["source_video"])
    frame_number = str(record["frame_number"])
    saved_filename = str(record["saved_filename"])
    if not allow_duplicate and any(
        (row["source_video"] == source_video and row["frame_number"] == frame_number)
        or row["saved_filename"] == saved_filename
        for row in existing
    ):
        raise FileExistsError(f"Frame extraction is already recorded: {source_video} frame {frame_number}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    mode = "x" if write_header else "a"
    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTRACTED_FRAME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def save_original_frame(
    frame: np.ndarray,
    video_path: str | Path,
    frame_number: int,
    fps: float,
    output_dir: str | Path,
    *,
    allow_duplicate: bool = False,
) -> dict[str, object]:
    """Losslessly save the decoded frame and append provenance without overwriting."""

    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        raise ValueError("Cannot save an empty or invalid decoded frame")
    timestamp = frame_timestamp_seconds(frame_number, fps)
    output = Path(output_dir)
    base_filename = extracted_frame_filename(video_path, frame_number, fps)
    filename = base_filename
    destination = output / filename
    csv_path = output / "extracted_frames.csv"
    existing = _read_existing_records(csv_path)
    duplicate_exists = destination.exists() or any(
        (row["source_video"] == Path(video_path).name and row["frame_number"] == str(frame_number))
        or row["saved_filename"] == base_filename
        for row in existing
    )
    if duplicate_exists and not allow_duplicate:
        raise FileExistsError(f"Refusing to overwrite existing extracted frame: {destination}")
    if duplicate_exists:
        stem, suffix = Path(base_filename).stem, Path(base_filename).suffix
        index = 1
        while True:
            filename = f"{stem}_override_{index:03d}{suffix}"
            destination = output / filename
            if not destination.exists() and all(row["saved_filename"] != filename for row in existing):
                break
            index += 1

    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise OSError("OpenCV could not encode the current frame as lossless PNG")
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "source_video": Path(video_path).name,
        "frame_number": frame_number,
        "timestamp_seconds": f"{timestamp:.6f}",
        "saved_filename": filename,
    }
    try:
        with destination.open("xb") as handle:
            handle.write(encoded.tobytes())
        _append_extraction_record(csv_path, record, allow_duplicate=allow_duplicate)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return record


def aspect_fit_size(
    source_width: int,
    source_height: int,
    available_width: int,
    available_height: int,
) -> tuple[int, int]:
    """Return the largest aspect-preserving size contained by the bounds."""

    if min(source_width, source_height, available_width, available_height) <= 0:
        raise ValueError("Source and available display dimensions must be positive")
    scale = min(available_width / source_width, available_height / source_height)
    width = min(available_width, max(1, round(source_width * scale)))
    height = min(available_height, max(1, round(source_height * scale)))
    return width, height


def _fit_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    size = aspect_fit_size(width, height, max_width, max_height)
    if size == (width, height):
        return frame.copy()
    interpolation = cv2.INTER_AREA if size[0] < width or size[1] < height else cv2.INTER_LINEAR
    return cv2.resize(frame, size, interpolation=interpolation)


def _window_content_size(window: str) -> tuple[int, int]:
    """Return the live OpenCV content size, with a safe initial fallback."""

    try:
        _, _, width, height = cv2.getWindowImageRect(window)
    except (cv2.error, AttributeError):
        return DEFAULT_WINDOW_SIZE
    if width < 320 or height < 240:
        return DEFAULT_WINDOW_SIZE
    return int(width), int(height)


def _draw_fitted_text(
    canvas: np.ndarray,
    text: str,
    y: int,
    *,
    preferred_scale: float = 0.58,
    colour: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw one complete status line, shrinking it only when the window requires it."""

    maximum_width = max(1, canvas.shape[1] - 24)
    measured_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, preferred_scale, 1)[0][0]
    scale = preferred_scale if measured_width <= maximum_width else preferred_scale * maximum_width / measured_width
    cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def build_display_frame(
    frame: np.ndarray,
    metadata: VideoMetadata,
    frame_number: int,
    *,
    video_status: str | None = None,
    video_position: tuple[int, int] | None = None,
    frame_history: Mapping[str, str] | None = None,
    linked_track_ids: list[str] | None = None,
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
) -> np.ndarray:
    """Create a full-window, aspect-fit display copy; ``frame`` is never changed."""

    if frame.ndim == 2:
        display_source = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        display_source = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        display_source = frame
    history_enabled = frame_history is not None or linked_track_ids is not None
    window_width, window_height = window_size
    if window_width < 320 or window_height < 240:
        raise ValueError("Display window must be at least 320 x 240 pixels")
    header_height = 76 if video_status is not None else 44
    footer_height = 112 if history_enabled else 50
    image_width = window_width - 2 * DISPLAY_MARGIN
    image_height = window_height - header_height - footer_height - 2 * DISPLAY_MARGIN
    display = _fit_display(display_source, image_width, image_height)
    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    display_height, display_width = display.shape[:2]
    image_left = (window_width - display_width) // 2
    image_top = header_height + DISPLAY_MARGIN + (image_height - display_height) // 2
    canvas[image_top:image_top + display_height, image_left:image_left + display_width] = display
    cv2.rectangle(
        canvas,
        (image_left - 2, image_top - 2),
        (image_left + display_width + 1, image_top + display_height + 1),
        FRAME_BORDER_COLOUR,
        2,
        cv2.LINE_AA,
    )
    timestamp = frame_timestamp_seconds(frame_number, metadata.fps)
    _draw_fitted_text(
        canvas,
        f"{metadata.source_path.name} | Frame {frame_number} / {metadata.total_frames - 1} | Total {metadata.total_frames} | FPS {metadata.fps:.3f} | Time {timestamp:.3f}s",
        30,
    )
    if video_status is not None:
        position = f" | Video {video_position[0]} / {video_position[1]}" if video_position else ""
        _draw_fitted_text(
            canvas,
            f"Status: {video_status}{position} | C/M mark complete | N next video | P previous video",
            62,
        )
    if history_enabled:
        historical = bool(
            frame_history
            and frame_history.get("saved") == "YES"
            and frame_history.get("history_origin") == "EXISTING_DATA_IMPORT"
        )
        reviewed = bool(frame_history and frame_history.get("reviewed_at")) or historical
        saved = bool(frame_history and frame_history.get("saved") == "YES")
        review_text = (
            "PREVIOUSLY REVIEWED / HISTORICAL"
            if historical
            else ("PREVIOUSLY REVIEWED" if reviewed else "NEW FRAME")
        )
        save_text = "NOT SAVED"
        if saved:
            save_text = (
                f"ALREADY SAVED: {frame_history.get('particle_label') or 'Unlabelled'} | "
                f"{frame_history.get('saved_filename', '')}"
            )
        _draw_fitted_text(
            canvas,
            f"REVIEW STATUS: {review_text} | SAVE STATUS: {save_text}",
            window_height - 82,
            preferred_scale=.55,
            colour=(0, 220, 255) if saved else (255, 255, 255),
        )
        crops = ", ".join(linked_track_ids or []) or "NONE"
        _draw_fitted_text(
            canvas,
            f"Existing labelled crops from this source frame: {crops}",
            window_height - 50,
            preferred_scale=.55,
        )
    _draw_fitted_text(
        canvas,
        "Right/Left frame | D/A +/-1s | W/X +/-0.1s | S unlabelled | 1 Alpha | 2 Electron | Q quit",
        window_height - 17,
        preferred_scale=.55,
    )
    return canvas


def _read_frame(capture: cv2.VideoCapture, frame_number: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = capture.read()
    if not success or frame is None or frame.size == 0:
        raise OSError(f"Could not decode frame {frame_number}")
    return frame


def _run_video_session(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    status_csv: str | Path | None = None,
    video_position: tuple[int, int] | None = None,
    history_dir: str | Path | None = None,
    labels_path: str | Path | None = None,
    source_dataset: str = "Lecturer",
    near_timestamp_seconds: float = .10,
    input_func: Callable[[str], str] = input,
) -> str:
    """Navigate one video and return a folder-navigation command."""

    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {source}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0 or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video must report a positive frame count and FPS")
    metadata = VideoMetadata(source, total_frames, fps)
    frame_number = 0
    extracted_csv = Path(output_dir) / "extracted_frames.csv"
    history = ReviewHistoryStore(history_dir) if history_dir is not None else None
    label_links = load_label_links(labels_path) if labels_path is not None else {}
    if history is not None:
        duration = total_frames / fps
        video_history = history.ensure_video(source.name, source_dataset, duration)
        print("\n" + "-" * 50 + "\nVIDEO REVIEW HISTORY\n")
        print(f"Video:\n{source.name}\n\nStatus:\n{video_history['review_status']}")
        furthest = video_history.get("furthest_reviewed_frame", "")
        last_frame = video_history.get("last_reviewed_frame", "")
        if furthest:
            print(f"\nPreviously reviewed:\nframes 0 -> {furthest}")
        if last_frame:
            print(f"\nLast position:\nframe {last_frame}\n{float(video_history['last_reviewed_timestamp_seconds']):.3f} s")
        print(
            "\nSaved from this video:\n"
            f"Alpha: {video_history['alpha_frames_saved']}\n"
            f"Electron: {video_history['electron_frames_saved']}\n"
            f"Unlabelled: {video_history['unlabelled_frames_saved']}"
        )
        if video_history["review_status"] == "COMPLETED":
            print("\nWARNING: This video was previously marked COMPLETED. Completion history is retained.")
        if furthest:
            resume = input_func("\nResume from previous furthest position? [Y/n] ").strip().lower()
            frame_number = resume_frame_number(video_history, total_frames, resume=resume not in {"n", "no"})
        print("-" * 50)
        current_status = video_history["review_status"]
    else:
        current_status = (
            load_video_review_status(status_csv).get(source.name, {}).get("status", "not_reviewed")
            if status_csv is not None else None
        )
    window = "Lecturer video frame selector (no automatic label)"
    try:
        frame = _read_frame(capture, frame_number)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(window, *DEFAULT_WINDOW_SIZE)
        while True:
            timestamp = frame_timestamp_seconds(frame_number, fps)
            prior_history = history.frame_record(source.name, frame_number) if history is not None else None
            linked = label_links.get(extracted_frame_filename(source, frame_number, fps), [])
            if history is not None:
                history.record_inspection(source.name, source_dataset, frame_number, timestamp)
                current_status = history.load_videos()[source.name]["review_status"]
            previous_window_size = None
            while True:
                window_size = _window_content_size(window)
                if window_size != previous_window_size:
                    cv2.imshow(
                        window,
                        build_display_frame(
                            frame,
                            metadata,
                            frame_number,
                            video_status=current_status,
                            video_position=video_position,
                            frame_history=prior_history if history is not None else None,
                            linked_track_ids=[row["track_id"] for row in linked] if history is not None else None,
                            window_size=window_size,
                        ),
                    )
                    previous_window_size = window_size
                key = cv2.waitKeyEx(50)
                if key >= 0:
                    break
            target = frame_number
            if key in RIGHT_ARROW_KEYS:
                target = min(total_frames - 1, frame_number + 1)
            elif key in LEFT_ARROW_KEYS:
                target = max(0, frame_number - 1)
            elif key in (ord("d"), ord("D")):
                target = jump_frame_number(frame_number, total_frames, 1.0, fps)
            elif key in (ord("a"), ord("A")):
                target = jump_frame_number(frame_number, total_frames, -1.0, fps)
            elif key in (ord("w"), ord("W")):
                target = jump_frame_number(frame_number, total_frames, 0.1, fps)
            elif key in (ord("x"), ord("X")):
                target = jump_frame_number(frame_number, total_frames, -0.1, fps)
            elif key in (ord("s"), ord("S"), ord("1"), ord("2")):
                particle_label = "Alpha" if key == ord("1") else ("Electron" if key == ord("2") else "Unlabelled")
                try:
                    digest = image_content_sha256(frame)
                    duplicates = history.exact_duplicates(source.name, frame_number, digest) if history is not None else []
                    allow_duplicate = False
                    duplicate_reference = ""
                    if duplicates:
                        duplicate_reference = ";".join(
                            f"{row['source_video']} frame {row['frame_number']} ({row['saved_filename']}, {row['particle_label'] or 'Unlabelled'})"
                            for row in duplicates
                        )
                        print("\nWARNING:\nThis frame appears to have already been saved.\n\nExisting:")
                        print(duplicate_reference)
                        allow_duplicate = input_func("Create an explicit duplicate override? [y/N] ").strip().lower() == "y"
                        if not allow_duplicate:
                            print("Duplicate save cancelled (default).")
                            continue
                    near = history.near_timestamp_saved(
                        source.name, timestamp, threshold=near_timestamp_seconds, exclude_frame=frame_number
                    ) if history is not None else []
                    if near:
                        print("\nPOSSIBLE SAME EVENT")
                        print(f"Current:\n{timestamp:.3f} s\n\nPreviously saved:")
                        for row in near:
                            print(f"{float(row['timestamp_seconds']):.3f} s - {row['saved_filename']}")
                        print("Please visually confirm whether this is the same physical particle event.")
                    record = save_original_frame(
                        frame, source, frame_number, fps, output_dir, allow_duplicate=allow_duplicate
                    )
                    print(f"Saved {record['saved_filename']} ({particle_label} frame metadata only).")
                    if linked:
                        print(format_source_frame_warning(labels_path, extracted_frame_filename(source, frame_number, fps)))
                    if history is not None:
                        history.record_saved_frame(
                            source.name, source_dataset, frame_number, timestamp,
                            str(record["saved_filename"]), particle_label, digest,
                            possible_duplicate=bool(duplicates), duplicate_reference=duplicate_reference,
                            linked_tracks=linked,
                        )
                    if status_csv is not None:
                        refresh_saved_frame_count(status_csv, source.name, extracted_csv)
                except FileExistsError as exc:
                    print(exc)
                continue
            elif history is not None and key in (ord("c"), ord("C"), ord("m"), ord("M")):
                confirmed = input_func("Mark this video as fully reviewed? [Y/N] ").strip().lower() == "y"
                if history.mark_completed(source.name, confirmed=confirmed):
                    current_status = "COMPLETED"
                    print(f"Marked {source.name} as COMPLETED.")
                else:
                    print("Completion cancelled; status unchanged.")
                continue
            elif history is None and status_csv is not None and key in (ord("m"), ord("M")):
                current_status = mark_video_status(status_csv, source.name, "reviewed", extracted_csv)["status"]
                print(f"Marked {source.name} as reviewed.")
                continue
            elif history is None and status_csv is not None and key in (ord("i"), ord("I")):
                current_status = mark_video_status(status_csv, source.name, "in_progress", extracted_csv)["status"]
                print(f"Marked {source.name} as in_progress.")
                continue
            elif (status_csv is not None or history is not None) and key in (ord("n"), ord("N")):
                return "n"
            elif (status_csv is not None or history is not None) and key in (ord("p"), ord("P")):
                return "p"
            elif key in (ord("q"), ord("Q"), 27):
                return "q"
            if target != frame_number:
                frame = _read_frame(capture, target)
                frame_number = target
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_video_frame_selector(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    history_dir: str | Path | None = None,
    labels_path: str | Path | None = None,
    source_dataset: str = "Lecturer",
    near_timestamp_seconds: float = .10,
) -> int:
    """Run one video, with persistent history when a history directory is supplied."""

    _run_video_session(
        video_path, output_dir, history_dir=history_dir, labels_path=labels_path,
        source_dataset=source_dataset, near_timestamp_seconds=near_timestamp_seconds,
    )
    return 0


def run_video_folder_selector(
    folder: str | Path,
    output_dir: str | Path,
    status_csv: str | Path,
    *,
    include_reviewed: bool = False,
    history_dir: str | Path | None = None,
    labels_path: str | Path | None = None,
    source_dataset: str = "Lecturer",
    near_timestamp_seconds: float = .10,
) -> int:
    """Review a folder of MP4 files, skipping completed videos by default."""

    videos = discover_video_files(folder)
    if not videos:
        raise ValueError(f"No MP4 videos found in directory: {Path(folder).expanduser().resolve()}")
    extracted_csv = Path(output_dir) / "extracted_frames.csv"
    if history_dir is not None:
        history = ReviewHistoryStore(history_dir)
        for video in videos:
            capture = cv2.VideoCapture(str(video))
            try:
                frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                duration = frames / fps if frames > 0 and fps > 0 else ""
            finally:
                capture.release()
            history.ensure_video(video.name, source_dataset, duration)
        statuses = history.load_videos()
    else:
        statuses = ensure_video_review_status(status_csv, videos, extracted_csv)
    review_queue = select_review_videos(videos, statuses, include_reviewed=include_reviewed)
    if not review_queue:
        print("No videos require review. Use --include-reviewed to revisit completed videos.")
        return 0
    index = 0
    while True:
        command = _run_video_session(
            review_queue[index],
            output_dir,
            status_csv=status_csv,
            video_position=(index + 1, len(review_queue)),
            history_dir=history_dir,
            labels_path=labels_path,
            source_dataset=source_dataset,
            near_timestamp_seconds=near_timestamp_seconds,
        )
        if command == "q":
            return 0
        if command == "n":
            next_index = min(len(review_queue) - 1, index + 1)
            if next_index == index:
                print("Already at the last video in the review queue.")
            index = next_index
        elif command == "p":
            previous_index = max(0, index - 1)
            if previous_index == index:
                print("Already at the first video in the review queue.")
            index = previous_index


def run_video_frame_selector_path(
    source: str | Path,
    output_dir: str | Path,
    status_csv: str | Path,
    *,
    include_reviewed: bool = False,
    history_dir: str | Path | None = None,
    labels_path: str | Path | None = None,
    source_dataset: str = "Lecturer",
    near_timestamp_seconds: float = .10,
) -> int:
    """Dispatch a file to single mode or a directory to persistent folder mode."""

    path = Path(source).expanduser()
    if path.is_dir():
        if history_dir is None and labels_path is None and source_dataset == "Lecturer" and near_timestamp_seconds == .10:
            return run_video_folder_selector(path, output_dir, status_csv, include_reviewed=include_reviewed)
        return run_video_folder_selector(
            path, output_dir, status_csv, include_reviewed=include_reviewed,
            history_dir=history_dir, labels_path=labels_path, source_dataset=source_dataset,
            near_timestamp_seconds=near_timestamp_seconds,
        )
    if history_dir is None and labels_path is None and source_dataset == "Lecturer" and near_timestamp_seconds == .10:
        return run_video_frame_selector(path, output_dir)
    return run_video_frame_selector(
        path, output_dir, history_dir=history_dir, labels_path=labels_path,
        source_dataset=source_dataset, near_timestamp_seconds=near_timestamp_seconds,
    )
