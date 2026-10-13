from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from backend.src.video_processing.video_builder import (
    ClassifiedSequenceError,
    build_classified_video,
    finalise_classified_sequence,
    finalise_video_job,
)


def mappings(count):
    return [
        {
            "output_frame_index": index,
            "source_frame_index": index,
            "timestamp_seconds": index / 30,
            "filename": f"frame_{index + 1:06d}.png",
        }
        for index in range(count)
    ]


def base_results(count):
    return {
        "job_id": "job123",
        "current_stage": "CLASSIFYING",
        "current_frame": count,
        "target_fps": 30,
        "total_frames": count,
        "processed_frames": count,
        "preprocessed_frames": count,
        "classified_frames": count,
        "failed_frames": 0,
        "unique_source_frames_processed": count,
        "particles": {"total_tracks": 0, "alpha": 0, "electron": 0},
        "frames": [
            {
                **mapping,
                "status": "success",
                "error": None,
                "tracks_detected": 0,
                "alpha_count": 0,
                "electron_count": 0,
                "detection_outputs": {"annotation": f"detection/annotations/{mapping['filename']}"},
                "tracks": [],
            }
            for mapping in mappings(count)
        ],
    }


class VideoBuilderTests(unittest.TestCase):
    def make_job(self, root, count=3, missing_classified=()):
        job = root / "job123"
        raw_directory = job / "frames"
        classified_directory = job / "classified"
        raw_directory.mkdir(parents=True)
        classified_directory.mkdir()
        for index, mapping in enumerate(mappings(count)):
            image = np.full((48, 64, 3), 30 + index * 60, dtype=np.uint8)
            cv2.imwrite(str(raw_directory / mapping["filename"]), image)
            if index not in missing_classified:
                cv2.imwrite(str(classified_directory / mapping["filename"]), image)
        return job

    def test_missing_classified_slot_uses_raw_fallback_without_shifting_timeline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job = self.make_job(Path(temporary_directory), missing_classified=(1,))
            results = base_results(3)
            ordered = finalise_classified_sequence(
                job, mappings(3), results, width=64, height=48
            )

            self.assertEqual([path.name for path in ordered], [f"frame_{i:06d}.png" for i in range(1, 4)])
            self.assertTrue((job / "classified" / "frame_000002.png").is_file())
            self.assertTrue(results["frames"][1]["used_fallback_frame"])
            self.assertEqual(results["frames"][1]["status"], "failed")
            self.assertEqual(results["sequence_validation"]["valid_frames"], 3)

    def test_invalid_classified_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job = self.make_job(Path(temporary_directory))
            cv2.imwrite(
                str(job / "classified" / "frame_2.png"),
                np.zeros((48, 64, 3), dtype=np.uint8),
            )
            with self.assertRaisesRegex(ClassifiedSequenceError, "invalid frame filename"):
                finalise_classified_sequence(
                    job, mappings(3), base_results(3), width=64, height=48
                )

    def test_mp4_is_encoded_once_in_numeric_order_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job = self.make_job(Path(temporary_directory))
            ordered = [job / "classified" / item["filename"] for item in mappings(3)]
            output = build_classified_video(job, ordered, width=64, height=48)

            self.assertEqual(output["fps"], 30.0)
            self.assertEqual(output["frame_count"], 3)
            self.assertEqual((output["width"], output["height"]), (64, 48))
            self.assertAlmostEqual(output["duration_seconds"], 0.1, places=3)
            self.assertEqual(output["codec"], "h264")
            self.assertEqual(output["pixel_format"], "yuv420p")
            encoded = (job / output["relative_path"]).read_bytes()
            self.assertIn(b"avc1", encoded)
            self.assertLess(encoded.index(b"moov"), encoded.index(b"mdat"))

            capture = cv2.VideoCapture(str(job / output["relative_path"]))
            means = []
            while True:
                readable, frame = capture.read()
                if not readable:
                    break
                means.append(float(frame.mean()))
            capture.release()
            self.assertEqual(len(means), 3)
            self.assertLess(means[0], means[1])
            self.assertLess(means[1], means[2])

    def test_invalid_sequence_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job = self.make_job(Path(temporary_directory))
            ordered = [job / "classified" / item["filename"] for item in mappings(3)]
            output = build_classified_video(job, ordered, width=64, height=48)
            before = (job / output["relative_path"]).read_bytes()
            with self.assertRaisesRegex(ClassifiedSequenceError, "numerical sequence"):
                build_classified_video(job, list(reversed(ordered)), width=64, height=48)
            self.assertEqual((job / output["relative_path"]).read_bytes(), before)
            self.assertEqual(list((job / "output").glob(".encoding-*.mp4")), [])

    def test_final_result_updates_existing_analysis_and_marks_completed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job = self.make_job(Path(temporary_directory))
            results = finalise_video_job(
                job,
                mappings(3),
                base_results(3),
                width=64,
                height=48,
                source_duration_seconds=0.1,
            )

            self.assertEqual(results["current_stage"], "COMPLETED")
            self.assertEqual(results["processing"]["video_frames_ready"], 3)
            self.assertEqual(results["summary"]["total_output_frames"], 3)
            self.assertIn("not unique physical particles", results["summary"]["scientific_limitation"])
            self.assertTrue((job / "results.json").is_file())
            self.assertTrue((job / "output" / "classified_video.mp4").is_file())


if __name__ == "__main__":
    unittest.main()
