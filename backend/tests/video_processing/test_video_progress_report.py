import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from backend.src.video_processing.job_status import read_job_status, write_job_status
from backend.src.video_processing.video_report import REPORT_FILENAME, generate_video_report


class VideoProgressAndReportTests(unittest.TestCase):
    def test_status_progress_is_monotonic_and_records_failure_context(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_directory = Path(temporary_directory) / "a1b2c3d4e5f6"
            write_job_status(job_directory, status="PROCESSING", stage="EXTRACTING_FRAMES",
                             current_frame=4, total_frames=10, progress_percent=9, message="Extracting")
            write_job_status(job_directory, status="PROCESSING", stage="PREPROCESSING",
                             current_frame=1, total_frames=10, progress_percent=8, message="Preprocessing")
            write_job_status(job_directory, status="FAILED", stage="PREPROCESSING",
                             current_frame=1, total_frames=10, progress_percent=8,
                             message="Failed", error="Unable to preprocess frame 1.")

            status = read_job_status(job_directory)
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["stage"], "PREPROCESSING")
            self.assertEqual(status["progress_percent"], 9.0)
            self.assertEqual(status["error"], "Unable to preprocess frame 1.")
            self.assertTrue((job_directory / "processing_state.json").is_file())

    def test_report_uses_saved_results_and_reuses_current_pdf(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_directory = Path(temporary_directory) / "a1b2c3d4e5f6"
            classified = job_directory / "classified"
            classified.mkdir(parents=True)
            frames = []
            for index in range(5):
                filename = f"frame_{index + 1:06d}.png"
                image = np.full((180, 320, 3), 25 + index * 30, dtype=np.uint8)
                cv2.putText(image, f"Frame {index + 1}", (38, 98), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (255, 255, 255), 2, cv2.LINE_AA)
                self.assertTrue(cv2.imwrite(str(classified / filename), image))
                frames.append({
                    "output_frame_index": index,
                    "source_frame_index": index,
                    "timestamp_seconds": index / 30,
                    "filename": filename,
                    "tracks_detected": index + 1,
                    "alpha_count": index % 2,
                    "electron_count": index + 1 - (index % 2),
                    "tracks": [],
                })
            (job_directory / "metadata.json").write_text(json.dumps({
                "filename": "cloud_chamber.mp4", "duration_seconds": 5.0,
                "original_fps": 60.0, "target_fps": 30, "width": 320, "height": 180,
                "original_frames": 300,
            }), encoding="utf-8")
            (job_directory / "frames.json").write_text(json.dumps(frames), encoding="utf-8")
            (job_directory / "results.json").write_text(json.dumps({
                "total_frames": 5, "unique_source_frames_processed": 5, "frames": frames,
                "particles": {"total_tracks": 15, "alpha": 2, "electron": 13},
                "processing": {"extracted_frames": 5, "preprocessed_frames": 5,
                               "detected_frames": 5, "classified_frames": 5, "failed_frames": 0},
                "output": {"frame_count": 5},
            }), encoding="utf-8")

            report_path = generate_video_report(job_directory)
            first_modified = report_path.stat().st_mtime_ns
            reused_path = generate_video_report(job_directory)

            self.assertEqual(report_path.name, REPORT_FILENAME)
            self.assertEqual(report_path.read_bytes()[:4], b"%PDF")
            self.assertGreater(report_path.stat().st_size, 5_000)
            self.assertEqual(reused_path.stat().st_mtime_ns, first_modified)


if __name__ == "__main__":
    unittest.main()
