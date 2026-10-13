"""Asynchronous video-job coordination, results, reports, and safe downloads."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import shutil
from time import perf_counter
from typing import Any, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute

from backend.src.calibration.calibration_metadata import load_calibration_metadata
from backend.src.utils.path_utils import OUTPUT_VIDEO_DIR
from backend.src.video_processing import (
    SUPPORTED_VIDEO_FORMATS,
    TARGET_FPS,
    VideoValidationError,
    extract_frames,
    finalise_video_job,
    process_extracted_frames,
    read_video_metadata,
)
from backend.src.video_processing.job_status import read_job_status, write_job_status
from backend.src.video_processing.video_report import REPORT_FILENAME, generate_video_report


class VideoTimingRoute(APIRoute):
    """Start timing before FastAPI parses the uploaded multipart body."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def timed_handler(request: Request):
            request.state.video_processing_started = perf_counter()
            return await handler(request)

        return timed_handler


router = APIRouter(prefix="/video", tags=["video"], route_class=VideoTimingRoute)
logger = logging.getLogger(__name__)
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


def _safe_filename(filename: str) -> str:
    basename = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(basename).stem).strip("._") or "video"
    return f"{stem}{Path(basename).suffix.lower()}"


def _job_directory(job_id: str) -> Path:
    if JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise HTTPException(status_code=404, detail="Video job not found.")
    job_directory = OUTPUT_VIDEO_DIR / job_id
    if not job_directory.is_dir():
        raise HTTPException(status_code=404, detail="Video job not found.")
    return job_directory


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{path.name} is not available.") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"{path.name} could not be read.") from error


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _preview_indices(total_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    return sorted({round((total_frames - 1) * fraction) for fraction in (0, 0.25, 0.5, 0.75, 1)})


def _video_calibration(filename: str) -> float | None:
    try:
        metadata = load_calibration_metadata(f"{Path(filename).name}_calibration.json")
    except FileNotFoundError:
        return None
    return float(metadata["mm_per_pixel"])


def _run_video_job(job_directory: Path, video_path: Path, original_filename: str, metadata: Any,
                   processing_started: float) -> None:
    """Run the existing pipeline after the upload response has been sent."""
    total_frames = 0
    completed = ["UPLOADING", "READING_METADATA"]
    try:
        logger.info("[VIDEO] Job %s", job_directory.name)
        total_frames = metadata.expected_extracted_frames

        frames_directory = job_directory / "frames"
        write_job_status(job_directory, status="PROCESSING", stage="EXTRACTING_FRAMES",
                         total_frames=total_frames, progress_percent=5,
                         message=f"Extracting frame 0 of {total_frames}", completed_stages=completed)

        def extraction_progress(current: int, total: int) -> None:
            write_job_status(job_directory, status="PROCESSING", stage="EXTRACTING_FRAMES",
                             current_frame=current, total_frames=total,
                             progress_percent=5 + (10 * current / max(total, 1)),
                             message=f"Extracting frame {current} of {total}", completed_stages=completed)

        frame_mapping = extract_frames(video_path, frames_directory, metadata,
                                       progress_callback=extraction_progress)
        logger.info("[VIDEO] Extracted %d frames", len(frame_mapping))
        metadata_payload = _read_json(job_directory / "metadata.json")
        metadata_payload["extracted_frames"] = len(frame_mapping)
        _write_json(job_directory / "metadata.json", metadata_payload)
        completed.append("EXTRACTING_FRAMES")

        def analysis_progress(stage: str, current: int, total: int, operation_index: int) -> None:
            completed_work = ((current - 1) * 3) + operation_index
            labels = {"PREPROCESSING": "Preprocessing", "DETECTING": "Detecting tracks in",
                      "CLASSIFYING": "Classifying particles in"}
            write_job_status(job_directory, status="PROCESSING", stage=stage,
                             current_frame=current, total_frames=total,
                             progress_percent=15 + (75 * completed_work / max(total * 3, 1)),
                             message=f"{labels[stage]} frame {current} of {total}",
                             completed_stages=completed)

        processing_results = process_extracted_frames(
            job_directory, frame_mapping, mm_per_pixel=_video_calibration(original_filename),
            progress_callback=analysis_progress,
        )
        completed.extend(["PREPROCESSING", "DETECTING", "CLASSIFYING"])
        logger.info("[VIDEO] Classified %d frames", processing_results["classified_frames"])

        def video_progress(current: int, total: int) -> None:
            write_job_status(job_directory, status="PROCESSING", stage="BUILDING_VIDEO",
                             current_frame=current, total_frames=total,
                             progress_percent=90 + (8 * current / max(total, 1)),
                             message=f"Encoding classified video: frame {current} of {total}",
                             completed_stages=completed)

        write_job_status(job_directory, status="PROCESSING", stage="BUILDING_VIDEO",
                         total_frames=len(frame_mapping), progress_percent=90,
                         message="Encoding classified video...", completed_stages=completed)
        processing_results = finalise_video_job(
            job_directory, frame_mapping, processing_results, width=metadata.width, height=metadata.height,
            source_duration_seconds=metadata.duration_seconds, target_fps=TARGET_FPS,
            progress_callback=video_progress,
        )
        completed.append("BUILDING_VIDEO")
        logger.info("[VIDEO] Output video: %d frames", processing_results["output"]["frame_count"])
        write_job_status(job_directory, status="PROCESSING", stage="FINALISING",
                         current_frame=len(frame_mapping), total_frames=len(frame_mapping),
                         progress_percent=99, message="Finalising video analysis results",
                         completed_stages=completed)
        completed.append("FINALISING")
        # Includes upload parsing, queue time, all modules, encoder flush/faststart,
        # full decode validation and atomic publication of the finished MP4.
        processing_results["processing_duration_seconds"] = perf_counter() - processing_started
        _write_json(job_directory / "results.json", processing_results)
        write_job_status(job_directory, status="COMPLETED", stage="COMPLETED",
                         current_frame=len(frame_mapping), total_frames=len(frame_mapping),
                         progress_percent=100, message="Video analysis complete", completed_stages=completed)
        logger.info("[VIDEO] Completed job %s", job_directory.name)
    except Exception as error:
        logger.exception("[VIDEO] Job %s failed", job_directory.name)
        error_message = str(error) or type(error).__name__
        try:
            current = read_job_status(job_directory)
        except (FileNotFoundError, RuntimeError):
            current = {}
        write_job_status(job_directory, status="FAILED", stage=str(current.get("stage", "FAILED")),
                         current_frame=int(current.get("current_frame", 0) or 0),
                         total_frames=int(current.get("total_frames", total_frames) or total_frames),
                         progress_percent=float(current.get("progress_percent", 0) or 0),
                         message="Video processing failed", error=error_message,
                         completed_stages=completed)


def _result_payload(job_id: str, base_url: str) -> dict[str, Any]:
    job_directory = _job_directory(job_id)
    status = read_job_status(job_directory)
    if status.get("status") != "COMPLETED":
        raise HTTPException(status_code=409, detail="Video processing is not complete.")
    metadata = _read_json(job_directory / "metadata.json")
    results = _read_json(job_directory / "results.json")
    frame_results = results.get("frames", [])
    if not isinstance(frame_results, list):
        raise HTTPException(status_code=500, detail="Frame results are invalid.")

    stored_filename = metadata.get("stored_filename") or _safe_filename(str(metadata.get("filename", "video.mp4")))
    processing = results.get("processing", {})
    particles = results.get("particles", {})
    output = results.get("output", {})
    output_file = job_directory / "output" / "classified_video.mp4"
    output_version = output_file.stat().st_mtime_ns if output_file.is_file() else 0
    preview_frames = []
    for index in _preview_indices(len(frame_results)):
        frame = frame_results[index]
        preview_frames.append({key: value for key, value in frame.items()
                               if key not in {"tracks", "detection_outputs"}} | {
            "classified_frame_url": f"{base_url}/api/video/static/{job_id}/classified/{frame['filename']}"
        })
    report_available = (job_directory / "report" / REPORT_FILENAME).is_file()
    total_tracks = int(particles.get("total_tracks", particles.get("total_track_detections", 0)) or 0)
    return {
        "success": True, "job_id": job_id,
        "video": {
            "filename": metadata.get("filename"), "duration_seconds": metadata.get("duration_seconds"),
            "original_fps": metadata.get("original_fps"), "target_fps": metadata.get("target_fps", TARGET_FPS),
            "width": metadata.get("width"), "height": metadata.get("height"),
            "original_frames": metadata.get("original_frames"),
            "output_frames": output.get("frame_count", len(frame_results)),
            "processing_duration_seconds": results.get("processing_duration_seconds"),
        },
        "processing": {**processing, "unique_source_frames_processed": results.get("unique_source_frames_processed", 0)},
        "tracks": {"total_track_detections": total_tracks, "alpha": int(particles.get("alpha", 0) or 0),
                   "electron": int(particles.get("electron", 0) or 0)},
        "original_video_url": f"{base_url}/api/video/static/{job_id}/original/{stored_filename}",
        "classified_video_url": f"{base_url}/api/video/static/{job_id}/output/classified_video.mp4?v={output_version}",
        "preview_frames": preview_frames,
        "downloads": {
            "video": f"{base_url}/api/video/{job_id}/download/video",
            "results": f"{base_url}/api/video/{job_id}/download/results",
            "report": f"{base_url}/api/video/{job_id}/download/report" if report_available else None,
        },
        "report_available": report_available,
    }


@router.post("/process", status_code=202)
async def process_video(request: Request, background_tasks: BackgroundTasks, file: Optional[UploadFile] = File(None)):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No video file was supplied.")
    if Path(file.filename).suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    job_id = uuid.uuid4().hex[:12]
    job_directory = OUTPUT_VIDEO_DIR / job_id
    original_directory = job_directory / "original"
    video_path = original_directory / _safe_filename(file.filename)
    try:
        original_directory.mkdir(parents=True, exist_ok=False)
        write_job_status(job_directory, status="PROCESSING", stage="UPLOADING", progress_percent=1,
                         message="Uploading video")
        with video_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
        if video_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Video contains no readable frames.")
        write_job_status(job_directory, status="PROCESSING", stage="READING_METADATA", progress_percent=3,
                         message="Reading video metadata", completed_stages=["UPLOADING"])
        try:
            metadata = read_video_metadata(video_path, file.filename, TARGET_FPS)
        except VideoValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        _write_json(job_directory / "metadata.json", {
            **metadata.to_dict(), "job_id": job_id, "extracted_frames": 0,
            "stored_filename": video_path.name,
            "size": video_path.stat().st_size, "type": file.content_type,
        })
        write_job_status(job_directory, status="QUEUED", stage="EXTRACTING_FRAMES", progress_percent=5,
                         message="Video uploaded; frame extraction queued",
                         completed_stages=["UPLOADING", "READING_METADATA"])
        background_tasks.add_task(_run_video_job, job_directory, video_path, file.filename, metadata,
                                  request.state.video_processing_started)
        return {"success": True, "job_id": job_id, "status": "QUEUED"}
    except HTTPException:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Unable to save uploaded video: {error}") from error
    finally:
        await file.close()


@router.get("/status/{job_id}")
def video_status(job_id: str, request: Request):
    job_directory = _job_directory(job_id)
    try:
        status = read_job_status(job_directory)
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    try:
        metadata = _read_json(job_directory / "metadata.json")
    except HTTPException as error:
        raise HTTPException(status_code=503, detail="Video metadata is temporarily unavailable.") from error
    stored = metadata.get("stored_filename") or _safe_filename(str(metadata.get("filename", "video.mp4")))
    return {"success": True, **status, "filename": metadata.get("filename"),
            "size": metadata.get("size"), "type": metadata.get("type"),
            "original_video_url": f"{str(request.base_url).rstrip('/')}/api/video/static/{job_id}/original/{stored}"}


@router.get("/result/{job_id}")
def video_result(job_id: str, request: Request):
    return _result_payload(job_id, str(request.base_url).rstrip("/"))


@router.get("/result/{job_id}/frame/{frame_index}")
def video_frame_result(job_id: str, frame_index: int, request: Request):
    job_directory = _job_directory(job_id)
    if read_job_status(job_directory).get("status") != "COMPLETED":
        raise HTTPException(status_code=409, detail="Video processing is not complete.")
    results = _read_json(job_directory / "results.json")
    frames = results.get("frames", [])
    if frame_index < 1 or frame_index > len(frames):
        raise HTTPException(status_code=404, detail="Video frame result not found.")
    frame = dict(frames[frame_index - 1])
    if int(frame.get("output_frame_index", -1)) != frame_index - 1:
        raise HTTPException(status_code=500, detail="Video frame results are not in sequence.")
    base_url = str(request.base_url).rstrip("/")
    return {
        "success": True, "job_id": job_id, "frame_index": frame_index,
        "timestamp_seconds": frame.get("timestamp_seconds", 0),
        "classified_frame_url": f"{base_url}/api/video/static/{job_id}/classified/{frame['filename']}",
        "tracks_detected": frame.get("tracks_detected", 0), "alpha": frame.get("alpha_count", 0),
        "electron": frame.get("electron_count", 0), "status": frame.get("status"),
        "error": frame.get("error"), "tracks": frame.get("tracks", []),
        "previous_frame": frame_index - 1 if frame_index > 1 else None,
        "next_frame": frame_index + 1 if frame_index < len(frames) else None,
    }


@router.post("/{job_id}/report")
def create_video_report(job_id: str, request: Request):
    job_directory = _job_directory(job_id)
    if read_job_status(job_directory).get("status") != "COMPLETED":
        raise HTTPException(status_code=409, detail="Video processing is not complete.")
    try:
        report_path = generate_video_report(job_directory)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        logger.exception("[VIDEO] Report generation failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"Unable to generate video report: {error}") from error
    return {"success": True, "job_id": job_id, "filename": report_path.name,
            "download_url": f"{str(request.base_url).rstrip('/')}/api/video/{job_id}/download/report"}


def _download(job_id: str, relative_path: Path, *, media_type: str, filename: str) -> FileResponse:
    output_path = _job_directory(job_id) / relative_path
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} is not available.")
    return FileResponse(output_path, media_type=media_type, filename=filename)


@router.get("/{job_id}/download/video")
@router.get("/{job_id}/download", include_in_schema=False)
def download_classified_video(job_id: str):
    return _download(job_id, Path("output/classified_video.mp4"), media_type="video/mp4",
                     filename="classified_video.mp4")


@router.get("/{job_id}/download/report")
def download_video_report(job_id: str):
    return _download(job_id, Path("report") / REPORT_FILENAME, media_type="application/pdf",
                     filename=REPORT_FILENAME)


@router.get("/{job_id}/download/results")
def download_video_results(job_id: str):
    return _download(job_id, Path("results.json"), media_type="application/json",
                     filename=f"CloudTrack_{job_id}_results.json")
