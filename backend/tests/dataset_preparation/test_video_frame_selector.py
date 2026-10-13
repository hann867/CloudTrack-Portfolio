"""GUI-independent tests for original-resolution video frame extraction."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import src.dataset_preparation.video_frame_selector as frame_selector
from src.dataset_preparation.video_frame_selector import (
    EXTRACTED_FRAME_COLUMNS,
    FRAME_BORDER_COLOUR,
    VIDEO_REVIEW_STATUS_COLUMNS,
    VideoMetadata,
    aspect_fit_size,
    build_display_frame,
    ensure_video_review_status,
    extracted_frame_filename,
    frame_timestamp_seconds,
    jump_frame_number,
    load_video_review_status,
    mark_video_status,
    save_original_frame,
    seconds_to_frame_delta,
    select_review_videos,
)


class VideoFrameSelectorTests(unittest.TestCase):
    def test_landscape_frame_aspect_fits_landscape_window(self):
        self.assertEqual(aspect_fit_size(1920, 1080, 1400, 700), (1244, 700))

    def test_portrait_frame_aspect_fits_landscape_window(self):
        self.assertEqual(aspect_fit_size(1080, 1920, 1400, 700), (394, 700))

    def test_aspect_fit_never_exceeds_bounds_and_preserves_ratio(self):
        for source, available in [((1920, 1080), (901, 517)), ((1080, 1920), (1200, 633))]:
            fitted = aspect_fit_size(*source, *available)
            self.assertLessEqual(fitted[0], available[0])
            self.assertLessEqual(fitted[1], available[1])
            self.assertAlmostEqual(fitted[0] / fitted[1], source[0] / source[1], delta=0.003)

    def test_display_scaling_does_not_modify_original_frame(self):
        frame = np.arange(90 * 160 * 3, dtype=np.uint8).reshape(90, 160, 3)
        original = frame.copy()
        display = build_display_frame(
            frame, VideoMetadata(Path("wide.mp4"), 100, 25.0), 10,
            video_status="NOT_STARTED", window_size=(900, 600),
        )

        self.assertEqual(display.shape, (600, 900, 3))
        self.assertTrue(np.any(np.all(display == FRAME_BORDER_COLOUR, axis=2)))
        np.testing.assert_array_equal(frame, original)

    def test_historical_saved_frame_display_is_not_new(self):
        metadata = VideoMetadata(Path("lecture.mp4"), 100, 25.0)
        history = {
            "saved": "YES",
            "history_origin": "EXISTING_DATA_IMPORT",
            "reviewed_at": "",
            "particle_label": "Alpha",
            "saved_filename": "lecture_frame_10_time_0.400.png",
        }
        texts = []
        original_put_text = cv2.putText

        def capture_text(image, text, *args, **kwargs):
            texts.append(text)
            return original_put_text(image, text, *args, **kwargs)

        with mock.patch.object(cv2, "putText", side_effect=capture_text):
            build_display_frame(
                np.zeros((20, 30, 3), dtype=np.uint8), metadata, 10,
                video_status="NOT_STARTED", frame_history=history, linked_track_ids=["A001"],
            )

        display = "\n".join(texts)
        self.assertIn("PREVIOUSLY REVIEWED / HISTORICAL", display)
        self.assertIn("ALREADY SAVED: Alpha", display)
        self.assertIn("lecture_frame_10_time_0.400.png", display)
        self.assertIn("A001", display)
        self.assertNotIn("NEW FRAME", display)

    def test_status_file_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "videos" / "video_review_status.csv"
            statuses = ensure_video_review_status(
                status_path,
                [root / "b.mp4", root / "a.mp4"],
                root / "frames" / "extracted_frames.csv",
            )
            with status_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, VIDEO_REVIEW_STATUS_COLUMNS)
        self.assertEqual([row["video_filename"] for row in rows], ["a.mp4", "b.mp4"])
        self.assertEqual({row["status"] for row in rows}, {"not_reviewed"})
        self.assertEqual(statuses["a.mp4"]["saved_frame_count"], "0")

    def test_marking_video_reviewed_is_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "video_review_status.csv"
            updated = mark_video_status(status_path, "lecture.mp4", "reviewed", root / "extracted_frames.csv")
            reloaded = load_video_review_status(status_path)

        self.assertEqual(updated["status"], "reviewed")
        self.assertEqual(reloaded["lecture.mp4"]["status"], "reviewed")
        self.assertTrue(reloaded["lecture.mp4"]["last_reviewed_timestamp"])

    def test_marking_video_in_progress_is_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "video_review_status.csv"
            mark_video_status(status_path, "lecture.mp4", "reviewed", root / "extracted_frames.csv")
            mark_video_status(status_path, "lecture.mp4", "in_progress", root / "extracted_frames.csv")

            self.assertEqual(load_video_review_status(status_path)["lecture.mp4"]["status"], "in_progress")

    def test_reviewed_videos_are_skipped_unless_included(self):
        videos = [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]
        statuses = {
            "a.mp4": {"status": "reviewed"},
            "b.mp4": {"status": "in_progress"},
            "c.mp4": {"status": "not_reviewed"},
        }
        self.assertEqual(select_review_videos(videos, statuses), videos[1:])
        self.assertEqual(select_review_videos(videos, statuses, include_reviewed=True), videos)

    def test_saved_frame_count_reads_existing_extraction_csv_without_modifying_it(self):
        frame = np.full((10, 14, 3), 50, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            save_original_frame(frame, "a.mp4", 0, 20.0, frames)
            save_original_frame(frame, "a.mp4", 1, 20.0, frames)
            save_original_frame(frame, "b.mp4", 0, 20.0, frames)
            extracted_csv = frames / "extracted_frames.csv"
            original_csv = extracted_csv.read_bytes()

            statuses = ensure_video_review_status(
                root / "videos" / "video_review_status.csv",
                [root / "a.mp4", root / "b.mp4"],
                extracted_csv,
            )

            self.assertEqual(statuses["a.mp4"]["saved_frame_count"], "2")
            self.assertEqual(statuses["b.mp4"]["saved_frame_count"], "1")
            self.assertEqual(extracted_csv.read_bytes(), original_csv)

    def test_existing_single_video_mode_still_dispatches_normally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "one.mp4"
            output = root / "frames"
            status = root / "video_review_status.csv"
            with mock.patch.object(frame_selector, "run_video_frame_selector", return_value=0) as run_single, mock.patch.object(
                frame_selector, "run_video_folder_selector"
            ) as run_folder:
                result = frame_selector.run_video_frame_selector_path(video, output, status, include_reviewed=True)

            self.assertEqual(result, 0)
            run_single.assert_called_once_with(video, output)
            run_folder.assert_not_called()

    def test_frame_number_and_timestamp_conversion(self):
        self.assertAlmostEqual(frame_timestamp_seconds(25, 25.0), 1.0)
        self.assertAlmostEqual(frame_timestamp_seconds(3, 29.97), 3 / 29.97)
        self.assertEqual(seconds_to_frame_delta(1.0, 25.0), 25)
        self.assertEqual(seconds_to_frame_delta(0.1, 25.0), 2)
        with self.assertRaises(ValueError):
            frame_timestamp_seconds(0, 0)

    def test_backward_point_one_second_navigation_is_clamped(self):
        self.assertEqual(jump_frame_number(20, 100, -0.1, 25.0), 18)
        self.assertEqual(jump_frame_number(1, 100, -0.1, 25.0), 0)
        self.assertEqual(jump_frame_number(0, 100, -0.1, 25.0), 0)
        self.assertEqual(jump_frame_number(99, 100, 0.1, 25.0), 99)

    def test_filename_generation_is_deterministic(self):
        self.assertEqual(
            extracted_frame_filename(Path("Lecturer Video.mp4"), 25, 25.0),
            "Lecturer_Video_frame_25_time_1.000.png",
        )

    def test_saved_png_preserves_original_resolution_and_pixels(self):
        frame = np.arange(37 * 53 * 3, dtype=np.uint8).reshape(37, 53, 3)
        original = frame.copy()
        with tempfile.TemporaryDirectory() as directory:
            record = save_original_frame(frame, "lecture.mp4", 7, 25.0, directory)
            saved = cv2.imread(str(Path(directory) / str(record["saved_filename"])), cv2.IMREAD_UNCHANGED)

        self.assertEqual(saved.shape, frame.shape)
        np.testing.assert_array_equal(saved, original)
        np.testing.assert_array_equal(frame, original)

    def test_csv_records_are_appended_without_particle_label(self):
        frame = np.full((12, 18, 3), 91, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_original_frame(frame, "lecture.mp4", 0, 20.0, root)
            save_original_frame(frame, "lecture.mp4", 20, 20.0, root)
            with (root / "extracted_frames.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, EXTRACTED_FRAME_COLUMNS)
        self.assertEqual([row["frame_number"] for row in rows], ["0", "20"])
        self.assertEqual([row["timestamp_seconds"] for row in rows], ["0.000000", "1.000000"])
        self.assertEqual({row["source_video"] for row in rows}, {"lecture.mp4"})
        self.assertNotIn("class", reader.fieldnames or [])
        self.assertNotIn("label", reader.fieldnames or [])

    def test_existing_frame_and_csv_record_are_not_overwritten(self):
        first = np.full((10, 14, 3), 30, dtype=np.uint8)
        replacement = np.full((10, 14, 3), 220, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = save_original_frame(first, "lecture.mp4", 4, 20.0, root)
            destination = root / str(record["saved_filename"])
            original_png = destination.read_bytes()
            original_csv = (root / "extracted_frames.csv").read_bytes()

            with self.assertRaises(FileExistsError):
                save_original_frame(replacement, "lecture.mp4", 4, 20.0, root)

            self.assertEqual(destination.read_bytes(), original_png)
            self.assertEqual((root / "extracted_frames.csv").read_bytes(), original_csv)
            saved = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(saved, first)

    def test_explicit_duplicate_override_creates_new_file_without_overwriting(self):
        frame = np.full((10, 14, 3), 30, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_original_frame(frame, "lecture.mp4", 4, 20.0, root)
            original = (root / str(first["saved_filename"])).read_bytes()
            duplicate = save_original_frame(
                frame, "lecture.mp4", 4, 20.0, root, allow_duplicate=True
            )
            with (root / "extracted_frames.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertIn("_override_001.png", str(duplicate["saved_filename"]))
            self.assertEqual((root / str(first["saved_filename"])).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
