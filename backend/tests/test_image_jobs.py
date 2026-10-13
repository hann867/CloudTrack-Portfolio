import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from backend.app.routes import image_jobs
from backend.src.video_processing.job_status import write_job_status


class ImageJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patcher = patch.object(image_jobs, "OUTPUT_IMAGE_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        detection = patch("backend.app.routes.detection.MODULE2_OUTPUT_DIR", self.root / "module2_outputs")
        detection.start()
        self.addCleanup(detection.stop)
        app = FastAPI()
        app.include_router(image_jobs.router, prefix="/api")
        app.mount("/api/static", StaticFiles(directory=self.root), name="static")
        self.client = TestClient(app)
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        cv2.line(image, (15, 25), (110, 75), (255, 255, 255), 3)
        self.png = cv2.imencode(".png", image)[1].tobytes()

    def start(self, workflow=False, files=None, calibration=None):
        return self.client.post("/api/image/process", files=files or [("files", ("particle.png", self.png, "image/png"))],
                                data={"workflow": str(workflow).lower(), "calibrations_json": json.dumps(calibration or {})})

    def test_real_preprocessing_and_full_workflow_remain_queryable_and_serve_originals(self):
        for workflow in (False, True):
            with self.subTest(workflow=workflow):
                response = self.start(workflow)
                self.assertEqual(response.status_code, 202, response.text)
                job_id = response.json()["job_id"]
                status = self.client.get(f"/api/image/status/{job_id}").json()
                self.assertEqual(status["status"], "COMPLETED", status)
                self.assertEqual(status["progress_percent"], 100)
                self.assertEqual(status["files"][0]["name"], "particle.png")
                self.assertEqual(self.client.get(status["files"][0]["original_image_url"]).content, self.png)
                result = self.client.get(f"/api/image/result/{job_id}").json()
                self.assertEqual(result["preprocessing"]["successful_count"], 1)
                self.assertEqual(self.client.get(result["preprocessing"]["results"][0]["processed_image_url"]).status_code, 200)
                if workflow:
                    self.assertEqual(result["detection"]["successful_count"], 1)
                    self.assertIsNotNone(result["classificationSummary"])
                # A second client represents a reload; querying cannot rerun modules.
                with patch.object(image_jobs, "run_preprocessing_pipeline", side_effect=AssertionError("must not rerun")):
                    reloaded = TestClient(self.client.app)
                    self.assertEqual(reloaded.get(f"/api/image/result/{job_id}").json(), result)

    def test_queued_job_survives_request_and_worker_is_independent_of_observers(self):
        with patch.object(image_jobs, "_run_job"):
            job_id = self.start().json()["job_id"]
        directory = self.root / "jobs" / job_id
        metadata = json.loads((directory / "metadata.json").read_text())
        entered, release = threading.Event(), threading.Event()

        def delayed_pipeline(path):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test worker timeout")
            return {"final_image": np.zeros((16, 16), dtype=np.uint8)}

        with patch.object(image_jobs, "run_preprocessing_pipeline", side_effect=delayed_pipeline):
            worker = threading.Thread(target=image_jobs._run_job, args=(directory, metadata, "http://testserver"))
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                status = self.client.get(f"/api/image/status/{job_id}").json()
                self.assertEqual(status["status"], "PROCESSING")
                self.assertEqual(self.client.get(f"/api/image/result/{job_id}").status_code, 409)
                # No polling or connected page while the worker finishes.
            finally:
                release.set()
                worker.join(5)
        self.assertEqual(self.client.get(f"/api/image/status/{job_id}").json()["status"], "COMPLETED")

    def test_failed_job_retains_stage_message_original_and_partial_results(self):
        with patch.object(image_jobs, "_process_path", side_effect=RuntimeError("Detector unavailable")):
            job_id = self.start(True).json()["job_id"]
        status = self.client.get(f"/api/image/status/{job_id}").json()
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["stage"], "DETECTING")
        self.assertIn("Detector unavailable", status["error"])
        result = self.client.get(f"/api/image/result/{job_id}").json()
        self.assertEqual(result["preprocessing"]["successful_count"], 1)
        self.assertEqual(result["detection"]["failed_count"], 1)

    def test_mixed_images_keep_successes_and_all_bad_images_fail(self):
        mixed = self.start(files=[("files", ("good.png", self.png, "image/png")), ("files", ("bad.png", b"bad", "image/png"))])
        job_id = mixed.json()["job_id"]
        self.assertEqual(self.client.get(f"/api/image/status/{job_id}").json()["status"], "COMPLETED")
        result = self.client.get(f"/api/image/result/{job_id}").json()["preprocessing"]
        self.assertEqual((result["successful_count"], result["failed_count"]), (1, 1))
        bad = self.start(files=[("files", ("bad.png", b"bad", "image/png"))]).json()["job_id"]
        self.assertEqual(self.client.get(f"/api/image/status/{bad}").json()["status"], "FAILED")

    def test_calibrated_mask_handoff_reuses_classification_and_retains_provenance(self):
        mask_path = self.root / "mask.png"
        mask_path.write_bytes(self.png)
        detection = {"status": "success", "run_id": "run", "tracks": [{"track_id": "track", "roi_mask_url": "http://testserver/api/static/mask.png", "bbox_x": 7}]}
        calibration = {"particle.png": {"status": "calibrated", "data": {"mm_per_pixel": 0.2, "measurement_homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}}

        async def classify(files, provenance):
            supplied = json.loads(provenance)[0]
            self.assertEqual(supplied["request_id"], "run:track")
            self.assertEqual(supplied["bbox_x"], 7)
            self.assertEqual(supplied["mm_per_pixel"], 0.2)
            self.assertEqual(files[0].file.read(), self.png)
            return {"results": [{"processing_status": "success", "predicted_class": "Alpha"}]}

        with patch.object(image_jobs, "_process_path", return_value=detection), patch.object(image_jobs, "classify_masks_batch", side_effect=classify):
            job_id = self.start(True, calibration=calibration).json()["job_id"]
        result = self.client.get(f"/api/image/result/{job_id}").json()
        self.assertEqual(result["classificationSummary"]["alpha_count"], 1)
        self.assertEqual(result["calibrations"], calibration)

    def test_missing_ids_are_404_but_corrupt_status_is_retryable(self):
        self.assertEqual(self.client.get("/api/image/status/aaaaaaaaaaaa").status_code, 404)
        self.assertEqual(self.client.get("/api/image/status/not-a-job").status_code, 404)
        directory = self.root / "jobs" / "bbbbbbbbbbbb"
        directory.mkdir(parents=True)
        write_job_status(directory, status="PROCESSING", stage="PREPROCESSING")
        (directory / "status.json").write_text("{broken")
        self.assertEqual(self.client.get("/api/image/status/bbbbbbbbbbbb").status_code, 503)


if __name__ == "__main__":
    unittest.main()
