import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
from fastapi.testclient import TestClient
import numpy as np

from backend.app.main import app
from backend.src.video_processing.video_builder import finalise_video_job


class VideoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_missing_and_unsupported_uploads_return_clear_errors(self):
        missing = self.client.post("/api/video/process")
        unsupported = self.client.post(
            "/api/video/process", files={"file": ("notes.txt", b"not video", "text/plain")}
        )

        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.json()["detail"], "No video file was supplied.")
        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(unsupported.json()["detail"], "Unsupported video format.")

    def test_process_endpoint_persists_job_metadata_and_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.avi"
            writer = cv2.VideoWriter(
                str(source_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for index in range(30):
                writer.write(np.full((48, 64, 3), index * 6, dtype=np.uint8))
            writer.release()

            output_root = root / "outputs"
            clock = [100.0]

            def finish_encoding(*args, **kwargs):
                output = finalise_video_job(*args, **kwargs)
                clock[0] = 142.5  # Encoding and validation finish before the end timestamp.
                return output

            with (
                patch("backend.app.routes.video.OUTPUT_VIDEO_DIR", output_root),
                patch("backend.app.routes.video.perf_counter", side_effect=lambda: clock[0]),
                patch("backend.app.routes.video.finalise_video_job", side_effect=finish_encoding),
            ):
                response = self.client.post(
                    "/api/video/process",
                    files={"file": ("cloud_chamber.avi", source_path.read_bytes(), "video/x-msvideo")},
                )
                job_id = response.json()["job_id"]
                status = self.client.get(f"/api/video/status/{job_id}")
                result_response = self.client.get(f"/api/video/result/{job_id}")
                frame_detail = self.client.get(f"/api/video/result/{job_id}/frame/1")
                download = self.client.get(f"/api/video/{job_id}/download/video")
                report = self.client.post(f"/api/video/{job_id}/report")
                report_download = self.client.get(f"/api/video/{job_id}/download/report")
                results_download = self.client.get(f"/api/video/{job_id}/download/results")

            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["status"], "QUEUED")
            self.assertEqual(status.json()["status"], "COMPLETED")
            self.assertEqual(status.json()["progress_percent"], 100)
            self.assertEqual(status.json()["filename"], "cloud_chamber.avi")
            self.assertGreater(status.json()["size"], 0)
            self.assertTrue(status.json()["original_video_url"].endswith(f"/{job_id}/original/cloud_chamber.avi"))
            payload = result_response.json()
            job_directory = output_root / payload["job_id"]
            self.assertTrue((job_directory / "original" / "cloud_chamber.avi").is_file())
            self.assertTrue((job_directory / "metadata.json").is_file())
            self.assertTrue((job_directory / "frames.json").is_file())
            self.assertEqual(len(payload["preview_frames"]), 5)
            self.assertEqual(payload["processing"]["preprocessed_frames"], 30)
            self.assertEqual(payload["processing"]["classified_frames"], 30)
            self.assertEqual(payload["processing"]["failed_frames"], 0)
            self.assertEqual(payload["tracks"]["total_track_detections"], 0)
            self.assertEqual(payload["video"]["target_fps"], 30)
            self.assertEqual(payload["video"]["output_frames"], 30)
            self.assertEqual(payload["video"]["processing_duration_seconds"], 42.5)
            self.assertEqual(payload["video"]["duration_seconds"], 1.0)
            saved = json.loads((job_directory / "results.json").read_text())
            self.assertEqual(payload["video"]["processing_duration_seconds"], saved["processing_duration_seconds"])
            self.assertIn("?v=", payload["classified_video_url"])
            self.assertEqual(frame_detail.status_code, 200)
            self.assertEqual(frame_detail.json()["frame_index"], 1)
            self.assertIn("tracks", frame_detail.json())
            self.assertEqual(download.status_code, 200)
            self.assertEqual(download.headers["content-type"], "video/mp4")
            self.assertEqual(report.status_code, 200, report.text)
            self.assertTrue(report.json()["download_url"].endswith(f"/{job_id}/download/report"))
            self.assertEqual(report_download.headers["content-type"], "application/pdf")
            self.assertEqual(report_download.content[:4], b"%PDF")
            self.assertEqual(results_download.headers["content-type"], "application/json")
            self.assertEqual(len(json.loads((job_directory / "frames.json").read_text())), 30)
            self.assertEqual(len(list((job_directory / "preprocessed").glob("frame_*.png"))), 30)
            self.assertEqual(len(list((job_directory / "classified").glob("frame_*.png"))), 30)
            self.assertTrue((job_directory / "results.json").is_file())
            self.assertTrue((job_directory / "output" / "classified_video.mp4").is_file())

    def test_invalid_video_content_does_not_crash_or_leave_a_job(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs"
            with patch("backend.app.routes.video.OUTPUT_VIDEO_DIR", output_root):
                response = self.client.post(
                    "/api/video/process",
                    files={"file": ("broken.mp4", b"not a real video", "video/mp4")},
                )

            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"], "Unable to open uploaded video.")
            self.assertEqual(list(output_root.glob("*")), [])

    def test_missing_job_is_404_but_existing_job_with_unavailable_status_is_retryable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            (output_root / "aaaaaaaaaaaa").mkdir()
            with patch("backend.app.routes.video.OUTPUT_VIDEO_DIR", output_root):
                self.assertEqual(self.client.get("/api/video/status/bbbbbbbbbbbb").status_code, 404)
                self.assertEqual(self.client.get("/api/video/status/aaaaaaaaaaaa").status_code, 503)

    def test_five_second_24_fps_source_produces_complete_30_fps_sequence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source_24fps.avi"
            writer = cv2.VideoWriter(
                str(source_path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for index in range(120):
                writer.write(np.full((48, 64, 3), index % 20, dtype=np.uint8))
            writer.release()

            output_root = root / "outputs"
            with patch("backend.app.routes.video.OUTPUT_VIDEO_DIR", output_root):
                response = self.client.post(
                    "/api/video/process",
                    files={"file": ("source_24fps.avi", source_path.read_bytes(), "video/x-msvideo")},
                )
                result_response = self.client.get(
                    f"/api/video/result/{response.json()['job_id']}"
                )

            self.assertEqual(response.status_code, 202, response.text)
            payload = result_response.json()
            job_directory = output_root / payload["job_id"]
            results = json.loads((job_directory / "results.json").read_text())
            self.assertEqual(payload["processing"]["unique_source_frames_processed"], 120)
            self.assertEqual(payload["video"]["target_fps"], 30)
            self.assertEqual(payload["video"]["output_frames"], 150)
            self.assertEqual(len(list((job_directory / "preprocessed").glob("frame_*.png"))), 150)
            self.assertEqual(len(list((job_directory / "classified").glob("frame_*.png"))), 150)
            self.assertEqual(sum(frame["is_duplicate_source_frame"] for frame in results["frames"]), 30)

    def test_five_second_30_and_60_fps_sources_reconstruct_at_30_fps(self):
        for source_fps in (30.0, 60.0):
            with self.subTest(source_fps=source_fps), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                source_path = root / f"source_{int(source_fps)}fps.avi"
                source_frame_count = int(source_fps * 5)
                writer = cv2.VideoWriter(
                    str(source_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    source_fps,
                    (64, 48),
                )
                self.assertTrue(writer.isOpened())
                for index in range(source_frame_count):
                    writer.write(np.full((48, 64, 3), index % 20, dtype=np.uint8))
                writer.release()

                output_root = root / "outputs"
                with patch("backend.app.routes.video.OUTPUT_VIDEO_DIR", output_root):
                    response = self.client.post(
                        "/api/video/process",
                        files={
                            "file": (
                                source_path.name,
                                source_path.read_bytes(),
                                "video/x-msvideo",
                            )
                        },
                    )
                    result_response = self.client.get(
                        f"/api/video/result/{response.json()['job_id']}"
                    )

                self.assertEqual(response.status_code, 202, response.text)
                payload = result_response.json()
                self.assertEqual(payload["video"]["original_frames"], source_frame_count)
                self.assertEqual(payload["video"]["target_fps"], 30)
                self.assertEqual(payload["video"]["output_frames"], 150)


if __name__ == "__main__":
    unittest.main()
