"""Isolated backend for frontend/e2e/realProcessingPersistence.cjs.

Run with: python -m backend.tests.browser_job_server
All generated fixtures and outputs live in a temporary directory.
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.routes import detection, image_jobs, video


def main():
    with tempfile.TemporaryDirectory(prefix="cloudtrack_browser_jobs_") as temporary:
        root = Path(temporary)
        images, videos, fixtures = root / "images", root / "videos", root / "fixtures"
        for directory in (images, videos, fixtures):
            directory.mkdir()
        image_jobs.OUTPUT_IMAGE_DIR = images
        detection.MODULE2_OUTPUT_DIR = images / "module2_outputs"
        video.OUTPUT_VIDEO_DIR = videos
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.line(frame, (30, 50), (280, 190), (255, 255, 255), 3)
        cv2.imwrite(str(fixtures / "particle.png"), frame)
        writer = cv2.VideoWriter(str(fixtures / "chamber.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
        if not writer.isOpened():
            raise RuntimeError("Unable to create browser video fixture")
        for _ in range(150):
            writer.write(frame)
        writer.release()
        app = FastAPI()
        app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5174"], allow_methods=["*"], allow_headers=["*"])
        app.include_router(image_jobs.router, prefix="/api")
        app.include_router(video.router, prefix="/api")
        app.mount("/api/static", StaticFiles(directory=images))
        app.mount("/api/video/static", StaticFiles(directory=videos))
        app.mount("/fixtures", StaticFiles(directory=fixtures))

        @app.get("/api/calibration/profiles")
        def profiles():
            return {"profiles": []}

        uvicorn.run(app, host="127.0.0.1", port=8011)


if __name__ == "__main__":
    main()
