from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.dataset_preparation.dataset_metadata import LABEL_COLUMNS, build_label_record
from src.dataset_preparation.lecturer_annotation_history import (
    ANNOTATION_HISTORY_COLUMNS,
    DuplicateAssessment,
    LecturerAnnotationHistoryStore,
    annotation_summary,
    assess_duplicate_annotation,
    confirm_duplicate_annotation,
    discover_saved_lecturer_frames,
    run_lecturer_annotation_queue,
    unresolved_queue,
)
from src.dataset_preparation.video_review_history import FRAME_HISTORY_COLUMNS


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _png(path: Path, value: int) -> np.ndarray:
    image = np.full((20, 30, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return image


def _frame_row(filename: str, number: int) -> dict[str, object]:
    row = {column: "" for column in FRAME_HISTORY_COLUMNS}
    row.update(
        source_video="video.mp4", source_dataset="Lecturer", frame_number=number,
        timestamp_seconds=f"{number / 10:.6f}", review_decision="SAVED_UNLABELLED",
        saved="YES", saved_filename=filename, particle_label="Unlabelled",
    )
    return row


def _label(track_id: str, particle_class: str, source_image: str, roi=(1, 2, 8, 9)) -> dict[str, object]:
    return build_label_record(
        track_id=track_id,
        label=particle_class,
        source_dataset="Lecturer",
        source_image=source_image,
        roi=roi,
        event_id=f"EVENT_{track_id}",
    )


def _fixture(tmp_path: Path):
    frames_root = tmp_path / "frames"
    first = "video_frame_10_time_1.000.png"
    second = "video_frame_20_time_2.000.png"
    _png(frames_root / "alpha_frames" / first, 20)
    _png(frames_root / "unlabelled_frames" / second, 80)
    frame_history = tmp_path / "frame_review_history.csv"
    _write_csv(frame_history, FRAME_HISTORY_COLUMNS, [_frame_row(first, 10), _frame_row(second, 20)])
    output = tmp_path / "labelled_tracks"
    _write_csv(output / "labels.csv", LABEL_COLUMNS, [_label("A001", "alpha", first)])
    history = tmp_path / "lecturer_annotation_history.csv"
    return frames_root, frame_history, output, history, first, second


def test_existing_labels_bootstrap_annotated_and_default_queue_excludes_them(tmp_path: Path) -> None:
    frames_root, frame_history, output, history, _, _ = _fixture(tmp_path)
    frames = discover_saved_lecturer_frames(frame_history, frames_root)
    store = LecturerAnnotationHistoryStore(history)
    rows = store.reconcile(frames, output / "labels.csv")

    assert annotation_summary(frames, rows) | {"by_video": None} == {
        "total": 2, "annotated": 1, "unannotated": 1, "skipped": 0,
        "linked_tracks": 1, "alpha_tracks": 1, "electron_tracks": 0, "by_video": None,
    }
    assert rows[("video.mp4", 10)]["linked_track_ids"] == "A001"
    assert [frame.frame_number for frame in unresolved_queue(frames, rows)] == [20]
    store.reconcile(frames, output / "labels.csv")
    with history.open(newline="", encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert len(read) == 2 and list(read[0]) == ANNOTATION_HISTORY_COLUMNS


def test_live_tracks_remain_unresolved_until_explicit_finish_and_multiple_ids_are_kept(tmp_path: Path) -> None:
    frames_root, frame_history, output, history, _, second = _fixture(tmp_path)
    frames = discover_saved_lecturer_frames(frame_history, frames_root)
    store = LecturerAnnotationHistoryStore(history)
    store.reconcile(frames, output / "labels.csv")
    key = ("video.mp4", 20)
    store.begin_frame(key)
    _write_csv(
        output / "labels.csv", LABEL_COLUMNS,
        [_label("A001", "alpha", frames[0].saved_filename), _label("A002", "alpha", second), _label("E002", "electron", second, (15, 2, 8, 9))],
    )
    rows = store.reconcile(frames, output / "labels.csv")
    assert rows[key]["annotation_status"] == "UNANNOTATED"
    assert rows[key]["linked_track_ids"] == "A002;E002"
    store.mark(key, "ANNOTATED")
    assert store.load()[key]["annotation_status"] == "ANNOTATED"


def test_skipped_remains_unresolved(tmp_path: Path) -> None:
    frames_root, frame_history, output, history, _, _ = _fixture(tmp_path)
    frames = discover_saved_lecturer_frames(frame_history, frames_root)
    store = LecturerAnnotationHistoryStore(history)
    rows = store.reconcile(frames, output / "labels.csv")
    store.mark(("video.mp4", 20), "SKIPPED")
    _write_csv(
        output / "labels.csv", LABEL_COLUMNS,
        [_label("A001", "alpha", frames[0].saved_filename), _label("E002", "electron", frames[1].saved_filename)],
    )
    reconciled = store.reconcile(frames, output / "labels.csv")
    assert reconciled[("video.mp4", 20)]["annotation_status"] == "SKIPPED"
    assert [frame.frame_number for frame in unresolved_queue(frames, reconciled)] == [20]


def test_duplicate_detection_defaults_to_no_but_distinct_second_track_is_allowed(tmp_path: Path) -> None:
    output = tmp_path / "tracks"
    source = "frame.png"
    crop = np.full((9, 8, 3), 42, dtype=np.uint8)
    _write_csv(output / "labels.csv", LABEL_COLUMNS, [_label("A001", "alpha", source)])
    (output / "alpha").mkdir(parents=True)
    assert cv2.imwrite(str(output / "alpha" / "A001.png"), crop)

    duplicate = assess_duplicate_annotation(source, (1, 2, 8, 9), crop, output / "labels.csv", output)
    assert duplicate.likely_duplicate
    assert not confirm_duplicate_annotation(duplicate, input_func=lambda _: "", output_func=lambda _: None)
    assert confirm_duplicate_annotation(duplicate, input_func=lambda _: "yes", output_func=lambda _: None)

    distinct = assess_duplicate_annotation(
        source, (18, 2, 8, 9), np.full_like(crop, 99), output / "labels.csv", output,
    )
    assert distinct.existing_track_ids == ("A001",)
    assert not distinct.likely_duplicate
    assert confirm_duplicate_annotation(distinct, input_func=lambda _: (_ for _ in ()).throw(AssertionError()), output_func=lambda _: None)


def test_queue_uses_authoritative_unresolved_frame_and_preserves_labels_and_crops(tmp_path: Path) -> None:
    frames_root, frame_history, output, history, _, second = _fixture(tmp_path)
    crop_path = output / "alpha" / "A001.png"
    original_crop = _png(crop_path, 11).tobytes()
    labels_before = (output / "labels.csv").read_bytes()
    visited: list[str] = []

    def runner(path, _image, _root, _defaults, _guard):
        visited.append(path.name)
        return "q"

    result = run_lecturer_annotation_queue(
        frames_root=frames_root,
        frame_history_path=frame_history,
        annotation_history_path=history,
        output_root=output,
        session_runner=runner,
        output_func=lambda _: None,
    )
    assert result == 0
    assert visited == [second]
    assert (output / "labels.csv").read_bytes() == labels_before
    assert cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED).tobytes() == original_crop
    assert LecturerAnnotationHistoryStore(history).load()[("video.mp4", 20)]["annotation_status"] == "UNANNOTATED"
