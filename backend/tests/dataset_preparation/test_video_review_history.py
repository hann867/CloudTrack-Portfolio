from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.dataset_preparation.video_frame_selector import resume_frame_number
from src.dataset_preparation.video_review_history import (
    FRAME_HISTORY_COLUMNS,
    VIDEO_HISTORY_COLUMNS,
    ReviewHistoryStore,
    bootstrap_existing_data,
    format_source_frame_warning,
    image_content_sha256,
    load_label_links,
)


def _write_png(path: Path, value: int = 50) -> np.ndarray:
    image = np.full((12, 18, 3), value, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return image


def _write_labels(path: Path, rows: list[dict[str, str]]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "class", "source_image", "event_id"])
        writer.writeheader(); writer.writerows(rows)
    return path.read_bytes()


def test_first_time_video_and_restart_persist_progress(tmp_path: Path) -> None:
    store = ReviewHistoryStore(tmp_path / "history")
    video = store.ensure_video("v.mp4", "Lecturer", 10.0)
    assert video["review_status"] == "NOT_STARTED"
    store.record_inspection("v.mp4", "Lecturer", 12, .4, when="2026-01-01T00:00:00+00:00")

    restarted = ReviewHistoryStore(tmp_path / "history")
    video = restarted.load_videos()["v.mp4"]
    assert video["review_status"] == "IN_PROGRESS"
    assert video["last_reviewed_frame"] == video["furthest_reviewed_frame"] == "12"
    assert video["frames_inspected"] == "1"
    assert resume_frame_number(video, 100, resume=True) == 12
    assert resume_frame_number(video, 100, resume=False) == 0
    assert restarted.frame_record("v.mp4", 12)["reviewed_at"]


def test_revisiting_frame_updates_one_stable_row_and_preserves_furthest(tmp_path: Path) -> None:
    store = ReviewHistoryStore(tmp_path)
    store.ensure_video("v.mp4", "Lecturer", 10)
    store.record_inspection("v.mp4", "Lecturer", 20, 2.0, when="first")
    store.record_inspection("v.mp4", "Lecturer", 5, .5, when="second")
    store.record_inspection("v.mp4", "Lecturer", 5, .5, when="third")
    assert len(store.load_frames()) == 2
    assert store.frame_record("v.mp4", 5)["reviewed_at"] == "third"
    video = store.load_videos()["v.mp4"]
    assert video["last_reviewed_frame"] == "5"
    assert video["furthest_reviewed_frame"] == "20"


def test_saved_frame_exact_identity_hash_and_near_timestamp_detection(tmp_path: Path) -> None:
    store = ReviewHistoryStore(tmp_path)
    store.ensure_video("v.mp4", "Lecturer", 10)
    image = np.full((3, 4, 3), 7, dtype=np.uint8)
    digest = image_content_sha256(image)
    store.record_saved_frame("v.mp4", "Lecturer", 10, 1.0, "a.png", "Alpha", digest, when="saved")
    assert store.frame_record("v.mp4", 10)["saved"] == "YES"
    assert store.exact_duplicates("v.mp4", 10, digest)
    assert store.exact_duplicates("other.mp4", 99, digest)
    store.record_saved_frame("v.mp4", "Lecturer", 11, 1.08, "b.png", "Electron", "different", when="saved2")
    assert [row["frame_number"] for row in store.near_timestamp_saved("v.mp4", 1.0, threshold=.1, exclude_frame=10)] == ["11"]
    assert store.near_timestamp_saved("v.mp4", 1.0, threshold=.05, exclude_frame=10) == []


def test_completed_video_reopens_without_losing_completion_and_requires_confirmation(tmp_path: Path) -> None:
    store = ReviewHistoryStore(tmp_path)
    store.ensure_video("v.mp4", "Lecturer", 10)
    assert store.mark_completed("v.mp4", confirmed=False) is False
    assert store.load_videos()["v.mp4"]["review_status"] == "NOT_STARTED"
    assert store.mark_completed("v.mp4", confirmed=True, when="done") is True
    store.ensure_video("v.mp4", "Lecturer", 10)
    store.record_inspection("v.mp4", "Lecturer", 2, .2, when="reopened")
    row = store.load_videos()["v.mp4"]
    assert row["review_status"] == "COMPLETED"
    assert row["review_completed_at"] == "done"


def test_bootstrap_existing_frames_is_idempotent_and_marks_historical_review(tmp_path: Path) -> None:
    videos = tmp_path / "videos"; videos.mkdir()
    (videos / "VID_one.mp4").write_bytes(b"")
    frames = tmp_path / "frames"
    filename = "VID_one_frame_25_time_1.000.png"
    _write_png(frames / filename)
    labels = tmp_path / "labels.csv"
    original_labels = _write_labels(labels, [{"track_id": "A001", "class": "Alpha", "source_image": filename, "event_id": "EVENT_1"}])
    store = ReviewHistoryStore(tmp_path / "history")

    first = bootstrap_existing_data(store, videos, frames, labels)
    second = bootstrap_existing_data(store, videos, frames, labels)

    assert first["historical_saved_frames_imported"] == 1
    assert second["historical_saved_frames_imported"] == 0
    assert len(store.load_frames()) == 1
    frame = store.frame_record("VID_one.mp4", 25)
    assert frame["history_origin"] == "EXISTING_DATA_IMPORT"
    assert frame["reviewed_at"] == ""
    assert frame["review_decision"] == "SAVED_UNLABELLED"
    assert frame["saved"] == "YES"
    assert frame["notes"] == "PREVIOUSLY_REVIEWED / HISTORICAL"
    assert frame["linked_track_ids"] == "A001"
    video = store.load_videos()["VID_one.mp4"]
    assert video["review_status"] == "NOT_STARTED"
    assert video["frames_inspected"] == "1" and video["frames_saved"] == "1"
    assert labels.read_bytes() == original_labels


def test_bootstrap_preserves_specific_saved_label_and_video_status(tmp_path: Path) -> None:
    videos = tmp_path / "videos"; videos.mkdir()
    (videos / "VID_one.mp4").write_bytes(b"")
    frames = tmp_path / "frames"
    filename = "VID_one_frame_25_time_1.000.png"
    _write_png(frames / filename)
    labels = tmp_path / "labels.csv"; _write_labels(labels, [])
    store = ReviewHistoryStore(tmp_path / "history")
    store.ensure_video("VID_one.mp4", "Lecturer", 10)
    store.record_saved_frame(
        "VID_one.mp4", "Lecturer", 25, 1.0, filename, "Alpha", "old-hash", when="known"
    )
    status_before = store.load_videos()["VID_one.mp4"]["review_status"]

    bootstrap_existing_data(store, videos, frames, labels)
    bootstrap_existing_data(store, videos, frames, labels)

    frame = store.frame_record("VID_one.mp4", 25)
    assert frame["particle_label"] == "Alpha"
    assert frame["review_decision"] == "SAVED_ALPHA"
    assert frame["history_origin"] == "EXISTING_DATA_IMPORT"
    assert frame["notes"] == "PREVIOUSLY_REVIEWED / HISTORICAL"
    assert len(store.load_frames()) == 1
    assert store.load_videos()["VID_one.mp4"]["review_status"] == status_before


def test_bootstrap_detects_equal_image_hashes_without_deleting_anything(tmp_path: Path) -> None:
    videos = tmp_path / "videos"; videos.mkdir()
    frames = tmp_path / "frames"
    _write_png(frames / "V_frame_1_time_0.010.png", 80)
    _write_png(frames / "V_frame_2_time_0.020.png", 80)
    labels = tmp_path / "labels.csv"; _write_labels(labels, [])
    result = bootstrap_existing_data(ReviewHistoryStore(tmp_path / "history"), videos, frames, labels)
    assert result["exact_duplicates_detected"] == 1
    assert result["potential_near_timestamp_duplicates"] == 1
    assert len(list(frames.glob("*.png"))) == 2


def test_labels_linkage_is_read_only_and_formats_event_warning(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    original = _write_labels(labels, [
        {"track_id": "A017", "class": "Alpha", "source_image": "frame.png", "event_id": "EVENT_A"},
        {"track_id": "E018", "class": "Electron", "source_image": "frame.png", "event_id": "EVENT_E"},
    ])
    links = load_label_links(labels)
    warning = format_source_frame_warning(labels, "frame.png")
    assert [row["track_id"] for row in links["frame.png"]] == ["A017", "E018"]
    assert "A017 - EVENT_A" in warning and "E018 - EVENT_E" in warning
    assert labels.read_bytes() == original


def test_history_csv_schemas_and_interrupted_progress_are_valid(tmp_path: Path) -> None:
    store = ReviewHistoryStore(tmp_path)
    store.ensure_video("v.mp4", "Lecturer", 20)
    store.record_inspection("v.mp4", "Lecturer", 4, .4, when="before-quit")
    # No close/completion call: the atomic per-frame update is already durable.
    with store.video_path.open(newline="", encoding="utf-8") as handle:
        video_reader = csv.DictReader(handle); list(video_reader)
    with store.frame_path.open(newline="", encoding="utf-8") as handle:
        frame_reader = csv.DictReader(handle); rows = list(frame_reader)
    assert video_reader.fieldnames == VIDEO_HISTORY_COLUMNS
    assert frame_reader.fieldnames == FRAME_HISTORY_COLUMNS
    assert len(rows) == 1
    assert ReviewHistoryStore(tmp_path).load_videos()["v.mp4"]["review_status"] == "IN_PROGRESS"
