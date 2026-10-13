"""Persistent, restart-safe metadata for manual video/frame review.

The history is deliberately separate from labels.csv and from all image data.
Rows are keyed by source_video (video history) and source_video+frame_number
(frame history), and every update rewrites its CSV atomically.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


VIDEO_HISTORY_COLUMNS = [
    "source_video", "source_dataset", "video_duration_seconds", "review_status",
    "last_reviewed_frame", "last_reviewed_timestamp_seconds", "furthest_reviewed_frame",
    "furthest_reviewed_timestamp_seconds", "frames_inspected", "frames_saved",
    "alpha_frames_saved", "electron_frames_saved", "unlabelled_frames_saved",
    "review_started_at", "last_reviewed_at", "review_completed_at", "notes",
]
FRAME_HISTORY_COLUMNS = [
    "source_video", "source_dataset", "frame_number", "timestamp_seconds", "reviewed_at",
    "review_decision", "saved", "saved_filename", "particle_label", "possible_duplicate",
    "duplicate_reference", "notes", "history_origin", "image_sha256", "linked_track_ids",
    "linked_event_ids",
]
ALLOWED_REVIEW_STATUSES = frozenset({"NOT_STARTED", "IN_PROGRESS", "COMPLETED"})
ALLOWED_REVIEW_DECISIONS = frozenset({
    "VIEWED_NO_SAVE", "SAVED_ALPHA", "SAVED_ELECTRON", "SAVED_UNLABELLED", "SKIPPED",
    "POSSIBLE_DUPLICATE",
})
ALLOWED_PARTICLE_LABELS = frozenset({"", "Alpha", "Electron", "Unlabelled"})
HISTORICAL_REVIEW_NOTE = "PREVIOUSLY_REVIEWED / HISTORICAL"
FRAME_FILENAME_PATTERN = re.compile(
    r"^(?P<stem>.+)_frame_(?P<frame>\d+)_time_(?P<time>\d+(?:\.\d+)?)\.(?:png|jpe?g)$",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_rows(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise ValueError(f"{path.name} columns do not match the required schema")
        return [dict(row) for row in reader]


def _atomic_write(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def image_content_sha256(image: np.ndarray) -> str:
    """Hash decoded pixels plus shape/dtype so encoding differences do not matter."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Cannot hash an empty image")
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def image_file_sha256(path: str | Path) -> str:
    image = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not decode image for hashing: {path}")
    return image_content_sha256(image)


def load_label_links(labels_path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Read labels.csv as linkage metadata without changing it."""
    path = Path(labels_path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"track_id", "class", "source_image", "event_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("labels.csv lacks source-image linkage columns")
        links: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            source_image = row.get("source_image", "").strip()
            if source_image:
                links[source_image].append({name: row.get(name, "") for name in required})
    for source_image in links:
        links[source_image].sort(key=lambda row: (row["track_id"].casefold(), row["track_id"]))
    return dict(links)


def format_source_frame_warning(labels_path: str | Path, source_image: str) -> str:
    links = load_label_links(labels_path).get(source_image, [])
    if not links:
        return "Existing labelled crops: NONE"
    lines = ["SOURCE FRAME ALREADY USED", "Existing tracks:"]
    lines.extend(f"{row['track_id']} - {row['event_id']} ({row['class']})" for row in links)
    lines.append("Check carefully before adding another track.")
    return "\n".join(lines)


class ReviewHistoryStore:
    """Atomic two-CSV review history with unique stable identities."""

    def __init__(self, history_dir: str | Path):
        self.root = Path(history_dir)
        self.video_path = self.root / "video_review_history.csv"
        self.frame_path = self.root / "frame_review_history.csv"

    def load_videos(self) -> dict[str, dict[str, str]]:
        rows = _read_rows(self.video_path, VIDEO_HISTORY_COLUMNS)
        output: dict[str, dict[str, str]] = {}
        for row in rows:
            key = row["source_video"].strip()
            if not key or key in output or row["review_status"] not in ALLOWED_REVIEW_STATUSES:
                raise ValueError("video_review_history.csv contains an invalid or duplicate video row")
            output[key] = row
        return output

    def load_frames(self) -> dict[tuple[str, int], dict[str, str]]:
        rows = _read_rows(self.frame_path, FRAME_HISTORY_COLUMNS)
        output: dict[tuple[str, int], dict[str, str]] = {}
        for row in rows:
            try:
                key = (row["source_video"].strip(), int(row["frame_number"]))
            except ValueError as exc:
                raise ValueError("frame_review_history.csv contains an invalid frame number") from exc
            if not key[0] or key[1] < 0 or key in output:
                raise ValueError("frame_review_history.csv contains an invalid or duplicate frame row")
            if row["review_decision"] not in ALLOWED_REVIEW_DECISIONS:
                raise ValueError("frame_review_history.csv contains an invalid review decision")
            if row["particle_label"] not in ALLOWED_PARTICLE_LABELS or row["saved"] not in {"YES", "NO"}:
                raise ValueError("frame_review_history.csv contains invalid save metadata")
            output[key] = row
        return output

    def _write_videos(self, videos: Mapping[str, Mapping[str, Any]]) -> None:
        ordered = [videos[key] for key in sorted(videos, key=lambda value: (value.casefold(), value))]
        _atomic_write(self.video_path, VIDEO_HISTORY_COLUMNS, ordered)

    def _write_frames(self, frames: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
        ordered = [frames[key] for key in sorted(frames, key=lambda value: (value[0].casefold(), value[0], value[1]))]
        _atomic_write(self.frame_path, FRAME_HISTORY_COLUMNS, ordered)

    @staticmethod
    def _blank_video(source_video: str, source_dataset: str, duration: float | str = "") -> dict[str, Any]:
        return {
            "source_video": source_video, "source_dataset": source_dataset,
            "video_duration_seconds": f"{float(duration):.6f}" if duration != "" else "",
            "review_status": "NOT_STARTED", "last_reviewed_frame": "",
            "last_reviewed_timestamp_seconds": "", "furthest_reviewed_frame": "",
            "furthest_reviewed_timestamp_seconds": "", "frames_inspected": 0, "frames_saved": 0,
            "alpha_frames_saved": 0, "electron_frames_saved": 0, "unlabelled_frames_saved": 0,
            "review_started_at": "", "last_reviewed_at": "", "review_completed_at": "", "notes": "",
        }

    @staticmethod
    def _blank_frame(source_video: str, source_dataset: str, frame_number: int,
                     timestamp_seconds: float) -> dict[str, Any]:
        return {
            "source_video": source_video, "source_dataset": source_dataset,
            "frame_number": frame_number, "timestamp_seconds": f"{timestamp_seconds:.6f}",
            "reviewed_at": "", "review_decision": "VIEWED_NO_SAVE", "saved": "NO",
            "saved_filename": "", "particle_label": "", "possible_duplicate": "NO",
            "duplicate_reference": "", "notes": "", "history_origin": "LIVE_REVIEW",
            "image_sha256": "", "linked_track_ids": "", "linked_event_ids": "",
        }

    def ensure_video(self, source_video: str, source_dataset: str,
                     video_duration_seconds: float | str = "") -> dict[str, str]:
        videos = self.load_videos()
        row = videos.get(source_video, self._blank_video(source_video, source_dataset, video_duration_seconds))
        row["source_dataset"] = row.get("source_dataset") or source_dataset
        if video_duration_seconds != "":
            row["video_duration_seconds"] = f"{float(video_duration_seconds):.6f}"
        videos[source_video] = row
        self._recount(row, self.load_frames())
        self._write_videos(videos)
        return {key: str(value) for key, value in row.items()}

    @staticmethod
    def _recount(video: dict[str, Any], frames: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
        selected = [row for (name, _), row in frames.items() if name == video["source_video"]]
        inspected = [
            row
            for row in selected
            if row.get("reviewed_at")
            or (
                row.get("saved") == "YES"
                and row.get("history_origin") == "EXISTING_DATA_IMPORT"
            )
        ]
        saved = [row for row in selected if row.get("saved") == "YES"]
        counts = Counter(row.get("particle_label") or "Unlabelled" for row in saved)
        video["frames_inspected"] = len(inspected)
        video["frames_saved"] = len(saved)
        video["alpha_frames_saved"] = counts["Alpha"]
        video["electron_frames_saved"] = counts["Electron"]
        video["unlabelled_frames_saved"] = counts["Unlabelled"]

    def record_inspection(self, source_video: str, source_dataset: str, frame_number: int,
                          timestamp_seconds: float, *, decision: str = "VIEWED_NO_SAVE",
                          when: str | None = None) -> dict[str, str]:
        if decision not in ALLOWED_REVIEW_DECISIONS:
            raise ValueError(f"Unsupported review decision: {decision}")
        now = when or utc_now()
        frames = self.load_frames()
        key = (source_video, int(frame_number))
        row = frames.get(key, self._blank_frame(source_video, source_dataset, frame_number, timestamp_seconds))
        row.update(source_dataset=source_dataset, timestamp_seconds=f"{timestamp_seconds:.6f}", reviewed_at=now)
        if not (
            row.get("saved") == "YES"
            and row.get("history_origin") == "EXISTING_DATA_IMPORT"
        ):
            row["history_origin"] = "LIVE_REVIEW"
        if row.get("saved") != "YES":
            row["review_decision"] = decision
        frames[key] = row
        self._write_frames(frames)

        videos = self.load_videos()
        video = videos.get(source_video, self._blank_video(source_video, source_dataset))
        if video["review_status"] != "COMPLETED":
            video["review_status"] = "IN_PROGRESS"
        video["review_started_at"] = video.get("review_started_at") or now
        video["last_reviewed_frame"] = frame_number
        video["last_reviewed_timestamp_seconds"] = f"{timestamp_seconds:.6f}"
        current_furthest = int(video["furthest_reviewed_frame"]) if str(video.get("furthest_reviewed_frame", "")) else -1
        if frame_number > current_furthest:
            video["furthest_reviewed_frame"] = frame_number
            video["furthest_reviewed_timestamp_seconds"] = f"{timestamp_seconds:.6f}"
        video["last_reviewed_at"] = now
        self._recount(video, frames)
        videos[source_video] = video
        self._write_videos(videos)
        return {key: str(value) for key, value in row.items()}

    def record_saved_frame(self, source_video: str, source_dataset: str, frame_number: int,
                           timestamp_seconds: float, saved_filename: str, particle_label: str,
                           image_sha256: str, *, possible_duplicate: bool = False,
                           duplicate_reference: str = "", linked_tracks: Sequence[Mapping[str, str]] = (),
                           notes: str = "", when: str | None = None) -> dict[str, str]:
        if particle_label not in {"Alpha", "Electron", "Unlabelled"}:
            raise ValueError(f"Unsupported particle label: {particle_label}")
        now = when or utc_now()
        self.record_inspection(source_video, source_dataset, frame_number, timestamp_seconds, when=now)
        frames = self.load_frames()
        key = (source_video, int(frame_number))
        row = frames[key]
        previous_names = [name for name in row.get("saved_filename", "").split(";") if name]
        if saved_filename not in previous_names:
            previous_names.append(saved_filename)
        row.update(
            review_decision=f"SAVED_{particle_label.upper()}", saved="YES",
            saved_filename=";".join(previous_names), particle_label=particle_label,
            possible_duplicate="YES" if possible_duplicate else "NO",
            duplicate_reference=duplicate_reference, notes=notes, image_sha256=image_sha256,
            linked_track_ids=";".join(link.get("track_id", "") for link in linked_tracks if link.get("track_id")),
            linked_event_ids=";".join(link.get("event_id", "") for link in linked_tracks if link.get("event_id")),
        )
        frames[key] = row
        self._write_frames(frames)
        videos = self.load_videos()
        video = videos[source_video]
        self._recount(video, frames)
        videos[source_video] = video
        self._write_videos(videos)
        return {key: str(value) for key, value in row.items()}

    def mark_completed(self, source_video: str, *, confirmed: bool, when: str | None = None) -> bool:
        if not confirmed:
            return False
        videos = self.load_videos()
        if source_video not in videos:
            raise KeyError(f"Unknown video: {source_video}")
        now = when or utc_now()
        videos[source_video]["review_status"] = "COMPLETED"
        videos[source_video]["review_completed_at"] = now
        videos[source_video]["last_reviewed_at"] = now
        self._write_videos(videos)
        return True

    def exact_duplicates(self, source_video: str, frame_number: int, image_sha256: str = "") -> list[dict[str, str]]:
        matches = []
        for key, row in self.load_frames().items():
            same_identity = key == (source_video, int(frame_number)) and row.get("saved") == "YES"
            same_hash = bool(image_sha256 and row.get("image_sha256") == image_sha256 and row.get("saved") == "YES")
            if same_identity or same_hash:
                matches.append(row)
        return matches

    def near_timestamp_saved(self, source_video: str, timestamp_seconds: float, *, threshold: float = .10,
                             exclude_frame: int | None = None) -> list[dict[str, str]]:
        if threshold < 0:
            raise ValueError("Near-timestamp threshold cannot be negative")
        matches = []
        for (name, frame_number), row in self.load_frames().items():
            if name != source_video or row.get("saved") != "YES" or frame_number == exclude_frame:
                continue
            if abs(float(row["timestamp_seconds"]) - timestamp_seconds) <= threshold + 1e-12:
                matches.append(row)
        return sorted(matches, key=lambda row: float(row["timestamp_seconds"]))

    def frame_record(self, source_video: str, frame_number: int) -> dict[str, str] | None:
        return self.load_frames().get((source_video, int(frame_number)))

    def summary(self) -> dict[str, Any]:
        videos = self.load_videos()
        frames = self.load_frames()
        status_counts = Counter(row["review_status"] for row in videos.values())
        saved = [row for row in frames.values() if row["saved"] == "YES"]
        label_counts = Counter(row.get("particle_label") or "Unlabelled" for row in saved)
        exact_hash_groups = Counter(row["image_sha256"] for row in saved if row.get("image_sha256"))
        return {
            "videos": videos, "frames": frames, "total_videos": len(videos),
            "completed": status_counts["COMPLETED"], "in_progress": status_counts["IN_PROGRESS"],
            "not_started": status_counts["NOT_STARTED"], "frames_saved": len(saved),
            "alpha": label_counts["Alpha"], "electron": label_counts["Electron"],
            "unlabelled": label_counts["Unlabelled"],
            "exact_duplicates": sum(count - 1 for count in exact_hash_groups.values() if count > 1),
            "linked_saved_frames": sum(bool(row.get("linked_track_ids")) for row in saved),
        }


def _video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0.0
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return frames / fps if frames > 0 and fps > 0 else 0.0
    finally:
        capture.release()


def discover_existing_review_data(videos_root: str | Path, frames_root: str | Path,
                                  labels_path: str | Path) -> dict[str, Any]:
    videos = sorted(Path(videos_root).rglob("*.mp4"), key=lambda path: (path.name.casefold(), path.name))
    links = load_label_links(labels_path)
    images = []
    for path in sorted(Path(frames_root).rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file():
            continue
        match = FRAME_FILENAME_PATTERN.match(path.name)
        if match:
            images.append({
                "path": path, "source_video": f"{match.group('stem')}.mp4",
                "frame_number": int(match.group("frame")), "timestamp_seconds": float(match.group("time")),
                "links": links.get(path.name, []),
            })
    identity_counts = Counter((row["source_video"], row["frame_number"]) for row in images)
    return {
        "videos": videos, "frames": images,
        "labelled_source_frames": sum(bool(row["links"]) for row in images),
        "potential_conflicts": sum(count - 1 for count in identity_counts.values() if count > 1),
    }


def bootstrap_existing_data(store: ReviewHistoryStore, videos_root: str | Path, frames_root: str | Path,
                            labels_path: str | Path, *, source_dataset: str = "Lecturer") -> dict[str, int]:
    """Idempotently import only filename/image/labels facts that can be proven."""
    discovered = discover_existing_review_data(videos_root, frames_root, labels_path)
    videos = store.load_videos()
    frames = store.load_frames()
    for path in discovered["videos"]:
        videos.setdefault(path.name, store._blank_video(path.name, source_dataset, _video_duration(path)))
    imported = 0
    hashes: Counter[str] = Counter()
    for item in discovered["frames"]:
        key = (item["source_video"], item["frame_number"])
        digest = image_file_sha256(item["path"])
        hashes[digest] += 1
        links = item["links"]
        if key in frames:
            row = frames[key]
        else:
            row = store._blank_frame(
                item["source_video"], source_dataset, item["frame_number"], item["timestamp_seconds"]
            )
            imported += 1

        existing_label = str(row.get("particle_label", ""))
        existing_decision = str(row.get("review_decision", ""))
        if existing_label in {"Alpha", "Electron"}:
            particle_label = existing_label
            review_decision = f"SAVED_{existing_label.upper()}"
        elif existing_decision in {"SAVED_ALPHA", "SAVED_ELECTRON"}:
            review_decision = existing_decision
            particle_label = "Alpha" if existing_decision == "SAVED_ALPHA" else "Electron"
        else:
            particle_label = "Unlabelled"
            review_decision = "SAVED_UNLABELLED"

        saved_names = [name for name in str(row.get("saved_filename", "")).split(";") if name]
        if item["path"].name not in saved_names:
            saved_names.append(item["path"].name)
        linked_track_ids = [value for value in str(row.get("linked_track_ids", "")).split(";") if value]
        linked_event_ids = [value for value in str(row.get("linked_event_ids", "")).split(";") if value]
        for link in links:
            if link["track_id"] not in linked_track_ids:
                linked_track_ids.append(link["track_id"])
            if link["event_id"] and link["event_id"] not in linked_event_ids:
                linked_event_ids.append(link["event_id"])
        notes = str(row.get("notes", "")).strip()
        if HISTORICAL_REVIEW_NOTE not in notes:
            notes = "; ".join(value for value in (notes, HISTORICAL_REVIEW_NOTE) if value)
        row.update(
            source_dataset=row.get("source_dataset") or source_dataset,
            timestamp_seconds=f"{float(item['timestamp_seconds']):.6f}",
            review_decision=review_decision,
            saved="YES",
            saved_filename=";".join(saved_names),
            particle_label=particle_label,
            notes=notes,
            history_origin="EXISTING_DATA_IMPORT",
            image_sha256=digest,
            linked_track_ids=";".join(linked_track_ids),
            linked_event_ids=";".join(linked_event_ids),
        )
        frames[key] = row
        videos.setdefault(item["source_video"], store._blank_video(item["source_video"], source_dataset))
    for video in videos.values():
        store._recount(video, frames)
    store._write_frames(frames)
    store._write_videos(videos)
    near_pairs = 0
    by_video: dict[str, list[float]] = defaultdict(list)
    for row in frames.values():
        if row["saved"] == "YES":
            by_video[row["source_video"]].append(float(row["timestamp_seconds"]))
    for timestamps in by_video.values():
        timestamps.sort()
        near_pairs += sum(b - a <= .10 + 1e-12 for a, b in zip(timestamps, timestamps[1:]))
    return {
        "videos_discovered": len(discovered["videos"]), "saved_frames_discovered": len(discovered["frames"]),
        "labelled_source_frames_discovered": discovered["labelled_source_frames"],
        "potential_conflicts": discovered["potential_conflicts"], "historical_saved_frames_imported": imported,
        "exact_duplicates_detected": sum(count - 1 for count in hashes.values() if count > 1),
        "potential_near_timestamp_duplicates": near_pairs,
    }


def format_review_status(store: ReviewHistoryStore) -> str:
    summary = store.summary()
    lines = [f"{'Video':36} {'Status':12} {'Progress':10} {'Saved':>6}"]
    for name, row in summary["videos"].items():
        duration = float(row["video_duration_seconds"] or 0)
        furthest = float(row["furthest_reviewed_timestamp_seconds"] or 0)
        progress = 100.0 if row["review_status"] == "COMPLETED" else (100.0 * furthest / duration if duration else 0.0)
        lines.append(f"{name[:36]:36} {row['review_status']:12} {progress:8.1f}% {int(row['frames_saved']):6d}")
    lines.extend([
        "", f"Total videos: {summary['total_videos']}", f"Completed: {summary['completed']}",
        f"In progress: {summary['in_progress']}", f"Not started: {summary['not_started']}", "",
        f"Total frames saved: {summary['frames_saved']}", f"Alpha: {summary['alpha']}",
        f"Electron: {summary['electron']}", f"Unlabelled: {summary['unlabelled']}",
    ])
    return "\n".join(lines)
