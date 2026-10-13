"""Resumable annotation queue for authoritative saved Lecturer frames.

This ledger is intentionally separate from video/frame review history.  A
saved frame becomes ANNOTATED only when it was already linked by labels.csv at
the first import, or when an annotator explicitly finishes it here.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .video_review_history import FRAME_HISTORY_COLUMNS, image_content_sha256, load_label_links


ANNOTATION_HISTORY_COLUMNS = [
    "source_video", "source_dataset", "frame_number", "timestamp_seconds",
    "saved_filename", "annotation_status", "linked_track_ids", "alpha_track_ids",
    "electron_track_ids", "uncertain_track_ids", "status_origin", "status_updated_at",
    "notes",
]
ALLOWED_ANNOTATION_STATUSES = frozenset({"UNANNOTATED", "ANNOTATED", "SKIPPED"})


@dataclass(frozen=True)
class SavedLecturerFrame:
    source_video: str
    source_dataset: str
    frame_number: int
    timestamp_seconds: float
    saved_filename: str
    path: Path

    @property
    def key(self) -> tuple[str, int]:
        return self.source_video, self.frame_number


@dataclass(frozen=True)
class DuplicateAssessment:
    existing_track_ids: tuple[str, ...]
    overlapping_track_ids: tuple[str, ...]
    matching_crop_track_ids: tuple[str, ...]

    @property
    def likely_duplicate(self) -> bool:
        return bool(self.overlapping_track_ids or self.matching_crop_track_ids)


def _read_csv(path: Path, required_columns: Sequence[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(required_columns):
            raise ValueError(f"{path.name} columns do not match the required schema")
        return [dict(row) for row in reader]


def _atomic_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ANNOTATION_HISTORY_COLUMNS, extrasaction="ignore")
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_saved_lecturer_frames(
    frame_history_path: str | Path,
    frames_root: str | Path,
) -> list[SavedLecturerFrame]:
    """Resolve only saved Lecturer rows from the authoritative frame history."""

    history_rows = _read_csv(Path(frame_history_path), FRAME_HISTORY_COLUMNS)
    root = Path(frames_root).expanduser().resolve()
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            by_name[path.name].append(path.resolve())

    frames: list[SavedLecturerFrame] = []
    seen: set[tuple[str, int]] = set()
    for row in history_rows:
        if row["saved"] != "YES" or row["source_dataset"].strip().casefold() != "lecturer":
            continue
        filename = Path(row["saved_filename"].strip()).name
        key = (row["source_video"].strip(), int(row["frame_number"]))
        if not filename or key in seen:
            raise ValueError("Saved Lecturer frame history contains a blank filename or duplicate key")
        matches = by_name.get(filename, [])
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one saved Lecturer image named {filename}; found {len(matches)}"
            )
        seen.add(key)
        frames.append(SavedLecturerFrame(
            source_video=key[0],
            source_dataset=row["source_dataset"].strip(),
            frame_number=key[1],
            timestamp_seconds=float(row["timestamp_seconds"]),
            saved_filename=filename,
            path=matches[0],
        ))
    return sorted(frames, key=lambda item: (item.source_video.casefold(), item.source_video, item.frame_number))


def _joined_ids(links: Sequence[Mapping[str, str]], particle_class: str | None = None) -> str:
    ids = [
        row["track_id"].strip()
        for row in links
        if row.get("track_id", "").strip()
        and (particle_class is None or row.get("class", "").strip().casefold() == particle_class.casefold())
    ]
    return ";".join(sorted(set(ids), key=lambda value: (value.casefold(), value)))


class LecturerAnnotationHistoryStore:
    """Atomic, unique-keyed frame annotation completion ledger."""

    def __init__(self, history_path: str | Path):
        self.path = Path(history_path)

    def load(self) -> dict[tuple[str, int], dict[str, str]]:
        rows = _read_csv(self.path, ANNOTATION_HISTORY_COLUMNS)
        output: dict[tuple[str, int], dict[str, str]] = {}
        for row in rows:
            try:
                key = (row["source_video"].strip(), int(row["frame_number"]))
            except ValueError as exc:
                raise ValueError("Annotation history contains an invalid frame number") from exc
            if not key[0] or key[1] < 0 or key in output:
                raise ValueError("Annotation history contains an invalid or duplicate frame key")
            if row["annotation_status"] not in ALLOWED_ANNOTATION_STATUSES:
                raise ValueError("Annotation history contains an invalid status")
            output[key] = row
        return output

    def write(self, rows: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
        ordered = [rows[key] for key in sorted(rows, key=lambda item: (item[0].casefold(), item[0], item[1]))]
        _atomic_write(self.path, ordered)

    @staticmethod
    def _apply_links(row: dict[str, str], links: Sequence[Mapping[str, str]]) -> None:
        row["linked_track_ids"] = _joined_ids(links)
        row["alpha_track_ids"] = _joined_ids(links, "Alpha")
        row["electron_track_ids"] = _joined_ids(links, "Electron")
        row["uncertain_track_ids"] = _joined_ids(links, "Uncertain")

    def reconcile(
        self,
        frames: Sequence[SavedLecturerFrame],
        labels_path: str | Path,
    ) -> dict[tuple[str, int], dict[str, str]]:
        rows = self.load()
        links_by_image = load_label_links(labels_path)
        for frame in frames:
            links = links_by_image.get(frame.saved_filename, [])
            row = rows.get(frame.key)
            if row is None:
                annotated = bool(links)
                row = {
                    "source_video": frame.source_video,
                    "source_dataset": frame.source_dataset,
                    "frame_number": str(frame.frame_number),
                    "timestamp_seconds": f"{frame.timestamp_seconds:.6f}",
                    "saved_filename": frame.saved_filename,
                    "annotation_status": "ANNOTATED" if annotated else "UNANNOTATED",
                    "linked_track_ids": "", "alpha_track_ids": "", "electron_track_ids": "",
                    "uncertain_track_ids": "",
                    "status_origin": "EXISTING_LABELS_IMPORT" if annotated else "SAVED_FRAME_IMPORT",
                    "status_updated_at": "", "notes": "",
                }
                rows[frame.key] = row
            else:
                row.update(
                    source_dataset=frame.source_dataset,
                    timestamp_seconds=f"{frame.timestamp_seconds:.6f}",
                    saved_filename=frame.saved_filename,
                )
                # Labels discovered outside an active unfinished session are
                # authoritative. LIVE_IN_PROGRESS deliberately remains queued
                # until the user explicitly finishes the frame.
                if links and not row["status_origin"].startswith("LIVE_"):
                    row["annotation_status"] = "ANNOTATED"
                    row["status_origin"] = "EXISTING_LABELS_IMPORT"
            self._apply_links(row, links)
        self.write(rows)
        return rows

    def begin_frame(self, key: tuple[str, int]) -> None:
        rows = self.load()
        row = rows[key]
        if row["annotation_status"] != "ANNOTATED":
            row["status_origin"] = "LIVE_IN_PROGRESS"
            self.write(rows)

    def refresh_links(self, key: tuple[str, int], labels_path: str | Path) -> None:
        rows = self.load()
        row = rows[key]
        self._apply_links(row, load_label_links(labels_path).get(row["saved_filename"], []))
        self.write(rows)

    def mark(self, key: tuple[str, int], status: str) -> None:
        if status not in {"ANNOTATED", "SKIPPED"}:
            raise ValueError("A live frame can only be marked ANNOTATED or SKIPPED")
        rows = self.load()
        row = rows[key]
        row["annotation_status"] = status
        row["status_origin"] = "LIVE_FINISH" if status == "ANNOTATED" else "LIVE_SKIP"
        row["status_updated_at"] = utc_now()
        self.write(rows)


def unresolved_queue(
    frames: Sequence[SavedLecturerFrame],
    rows: Mapping[tuple[str, int], Mapping[str, str]],
) -> list[SavedLecturerFrame]:
    unresolved = [frame for frame in frames if rows[frame.key]["annotation_status"] != "ANNOTATED"]
    return sorted(
        unresolved,
        key=lambda frame: (
            rows[frame.key]["annotation_status"] == "SKIPPED",
            frame.source_video.casefold(), frame.source_video, frame.frame_number,
        ),
    )


def annotation_summary(
    frames: Sequence[SavedLecturerFrame],
    rows: Mapping[tuple[str, int], Mapping[str, str]],
) -> dict[str, Any]:
    counts = Counter(rows[frame.key]["annotation_status"] for frame in frames)
    track_ids: set[str] = set()
    alpha_ids: set[str] = set()
    electron_ids: set[str] = set()
    by_video: dict[str, Counter[str]] = defaultdict(Counter)
    for frame in frames:
        row = rows[frame.key]
        track_ids.update(value for value in row["linked_track_ids"].split(";") if value)
        frame_alpha = {value for value in row["alpha_track_ids"].split(";") if value}
        frame_electron = {value for value in row["electron_track_ids"].split(";") if value}
        alpha_ids.update(frame_alpha)
        electron_ids.update(frame_electron)
        by_video[frame.source_video][row["annotation_status"]] += 1
        by_video[frame.source_video]["ALPHA_TRACKS"] += len(frame_alpha)
        by_video[frame.source_video]["ELECTRON_TRACKS"] += len(frame_electron)
    return {
        "total": len(frames), "annotated": counts["ANNOTATED"],
        "unannotated": counts["UNANNOTATED"], "skipped": counts["SKIPPED"],
        "linked_tracks": len(track_ids), "alpha_tracks": len(alpha_ids),
        "electron_tracks": len(electron_ids), "by_video": dict(by_video),
    }


def format_annotation_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"TOTAL SAVED LECTURER FRAMES: {summary['total']}",
        f"ANNOTATED FRAMES: {summary['annotated']}",
        f"UNANNOTATED FRAMES: {summary['unannotated']}",
        f"SKIPPED FRAMES (UNRESOLVED): {summary['skipped']}",
        f"TOTAL EXISTING LABELLED TRACKS FROM THESE FRAMES: {summary['linked_tracks']}",
        f"ALPHA TRACKS: {summary['alpha_tracks']}",
        f"ELECTRON TRACKS: {summary['electron_tracks']}",
        "BY SOURCE VIDEO:",
    ]
    for video, counts in sorted(summary["by_video"].items(), key=lambda item: item[0].casefold()):
        lines.append(
            f"  {video}: ANNOTATED {counts['ANNOTATED']}, "
            f"UNANNOTATED {counts['UNANNOTATED']}, SKIPPED {counts['SKIPPED']}, "
            f"ALPHA TRACKS {counts['ALPHA_TRACKS']}, ELECTRON TRACKS {counts['ELECTRON_TRACKS']}"
        )
    return "\n".join(lines)


def _intersection_metrics(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> tuple[float, float]:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = overlap_width * overlap_height
    first_area, second_area = aw * ah, bw * bh
    union = first_area + second_area - intersection
    return (intersection / union if union else 0.0, intersection / min(first_area, second_area) if min(first_area, second_area) else 0.0)


def assess_duplicate_annotation(
    source_image: str,
    roi: tuple[int, int, int, int],
    crop: np.ndarray,
    labels_path: str | Path,
    crops_root: str | Path,
) -> DuplicateAssessment:
    labels_file = Path(labels_path)
    links: list[dict[str, str]] = []
    if labels_file.exists():
        with labels_file.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"track_id", "class", "source_image", "crop_filename", "x", "y", "width", "height"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("labels.csv lacks duplicate-protection columns")
            links = [dict(row) for row in reader if row.get("source_image", "").strip() == source_image]
    overlapping: list[str] = []
    matching: list[str] = []
    candidate_hash = image_content_sha256(crop)
    root = Path(crops_root)
    for row in links:
        track_id = row["track_id"].strip()
        try:
            existing_roi = tuple(int(row[name]) for name in ("x", "y", "width", "height"))
            iou, smaller_coverage = _intersection_metrics(roi, existing_roi)  # type: ignore[arg-type]
            if iou >= 0.80 or smaller_coverage >= 0.90:
                overlapping.append(track_id)
        except (KeyError, TypeError, ValueError):
            pass
        crop_filename = row.get("crop_filename", "").strip()
        particle_class = row.get("class", "").strip().casefold()
        existing_path = root / particle_class / crop_filename
        existing = cv2.imread(str(existing_path), cv2.IMREAD_UNCHANGED) if existing_path.is_file() else None
        if existing is not None and image_content_sha256(existing) == candidate_hash:
            matching.append(track_id)
    return DuplicateAssessment(
        existing_track_ids=tuple(row["track_id"].strip() for row in links),
        overlapping_track_ids=tuple(overlapping),
        matching_crop_track_ids=tuple(matching),
    )


def confirm_duplicate_annotation(
    assessment: DuplicateAssessment,
    *,
    input_func=input,
    output_func=print,
) -> bool:
    """Default-deny likely duplicates while permitting distinct second tracks."""

    if assessment.existing_track_ids:
        output_func("SOURCE FRAME ALREADY HAS TRACKS: " + ", ".join(assessment.existing_track_ids))
    if not assessment.likely_duplicate:
        return True
    reasons: list[str] = []
    if assessment.overlapping_track_ids:
        reasons.append("overlapping ROI: " + ", ".join(assessment.overlapping_track_ids))
    if assessment.matching_crop_track_ids:
        reasons.append("exact crop pixels: " + ", ".join(assessment.matching_crop_track_ids))
    output_func("POSSIBLE DUPLICATE - " + "; ".join(reasons))
    response = input_func("Save duplicate anyway? [y/N]: ").strip().casefold()
    if response not in {"y", "yes"}:
        output_func("Duplicate was not saved.")
        return False
    output_func("Explicit duplicate override accepted.")
    return True


def run_lecturer_annotation_queue(
    *,
    frames_root: str | Path,
    frame_history_path: str | Path,
    annotation_history_path: str | Path,
    output_root: str | Path,
    include_annotated: bool = False,
    input_func=input,
    output_func=print,
    session_runner=None,
) -> int:
    """Run the existing crop annotator over the persistent Lecturer queue."""

    from .track_annotation_tool import (
        AnnotationDefaults,
        _read_source_image,
        _run_image_session,
        ensure_dataset_structure,
    )

    output = Path(output_root)
    labels_path = output / "labels.csv"
    frames = discover_saved_lecturer_frames(frame_history_path, frames_root)
    if not frames:
        raise ValueError("No authoritative saved Lecturer frames were found")
    ensure_dataset_structure(output)
    store = LecturerAnnotationHistoryStore(annotation_history_path)
    rows = store.reconcile(frames, labels_path)
    summary = annotation_summary(frames, rows)
    output_func(format_annotation_summary(summary))
    output_func(f"Already annotated frames excluded by default: {summary['annotated']}")
    if not include_annotated:
        for frame in frames:
            row = rows[frame.key]
            if row["annotation_status"] == "ANNOTATED":
                output_func(
                    f"ALREADY ANNOTATED: {frame.saved_filename} | "
                    f"track IDs: {row['linked_track_ids'] or 'NONE'}"
                )

    queue = list(frames) if include_annotated else unresolved_queue(frames, rows)
    if not queue:
        output_func("No unresolved saved Lecturer frames remain.")
        return 0
    defaults = AnnotationDefaults("Lecturer", "", "unknown")
    runner = session_runner or (
        lambda path, image, root, configured, guard: _run_image_session(
            path,
            image,
            root,
            configured,
            saved_frame_mode=True,
            before_save=guard,
        )
    )

    for index, frame in enumerate(queue, start=1):
        rows = store.load()
        row = rows[frame.key]
        output_func("")
        output_func(f"ANNOTATION: {index} / {len(queue)} unresolved")
        output_func(f"Source video: {frame.source_video}")
        output_func(f"Frame: {frame.frame_number} at {frame.timestamp_seconds:.3f} seconds")
        output_func(f"Saved file: {frame.saved_filename}")
        output_func(f"Status: {row['annotation_status']}")
        output_func(f"Existing track IDs: {row['linked_track_ids'] or 'NONE'}")
        if row["annotation_status"] == "ANNOTATED":
            output_func("ALREADY ANNOTATED")
            if input_func("Inspect this annotated frame? [y/N]: ").strip().casefold() not in {"y", "yes"}:
                continue
        else:
            store.begin_frame(frame.key)

        resolved_path, image = _read_source_image(frame.path)

        def duplicate_guard(roi: tuple[int, int, int, int], crop: np.ndarray) -> bool:
            assessment = assess_duplicate_annotation(
                frame.saved_filename, roi, crop, labels_path, output,
            )
            return confirm_duplicate_annotation(
                assessment, input_func=input_func, output_func=output_func,
            )

        command = runner(resolved_path, image, output, defaults, duplicate_guard)
        store.refresh_links(frame.key, labels_path)
        if command == "f":
            store.mark(frame.key, "ANNOTATED")
            output_func("Frame explicitly marked ANNOTATED.")
        elif command == "k":
            store.mark(frame.key, "SKIPPED")
            output_func("Frame marked SKIPPED and remains unresolved.")
        elif command == "q":
            output_func("Annotation queue quit safely; unfinished frame remains unresolved.")
            return 0
        else:
            raise ValueError(f"Unexpected saved-frame session command: {command}")

    final_rows = store.reconcile(frames, labels_path)
    output_func("")
    output_func(format_annotation_summary(annotation_summary(frames, final_rows)))
    return 0
