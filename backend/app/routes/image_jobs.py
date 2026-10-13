"""Durable image job coordination using the existing Module 1–3 implementations."""

import asyncio
import json
import re
import shutil
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from backend.app.routes.classification import classify_masks_batch
from backend.app.routes.detection import _process_path
from backend.src.preprocessing.image_loader import SUPPORTED_IMAGE_FORMATS
from backend.src.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline
from backend.src.utils.path_utils import OUTPUT_IMAGE_DIR
from backend.src.video_processing.job_status import _atomic_write, read_job_status, write_job_status

router = APIRouter(prefix="/image", tags=["image jobs"])


def _directory(job_id):
    directory = OUTPUT_IMAGE_DIR / "jobs" / job_id
    if not re.fullmatch(r"[a-f0-9]{12}", job_id) or not directory.is_dir():
        raise HTTPException(404, "Image job not found.")
    return directory


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Image job data is temporarily unavailable.") from error


def _batch(results):
    successful = sum(item["status"] == "success" for item in results)
    return {"total_images": len(results), "successful_count": successful,
            "failed_count": len(results) - successful, "results": results}


def _local_output(url):
    # URLs here come exclusively from Module 2, never from client input.
    relative = unquote(urlsplit(url).path).removeprefix("/api/static/")
    path = (OUTPUT_IMAGE_DIR / relative).resolve()
    if not path.is_relative_to(OUTPUT_IMAGE_DIR.resolve()):
        raise ValueError("Invalid module output path.")
    return path


def _run_job(directory, metadata, base_url):
    started = time.perf_counter()
    stage = "PREPROCESSING"
    snapshot = {"preprocessing": None, "detection": None, "classifications": {},
                "classificationSummary": None, "calibrations": metadata["calibrations"],
                "handoffErrors": [], "warning": "", "durationMs": None}
    total = len(metadata["files"])

    def progress(current, percent):
        write_job_status(directory, status="PROCESSING", stage=stage,
                         current_frame=current, total_frames=total, progress_percent=percent,
                         message=f"{stage.title()}: image {current} of {total}")

    def save():
        snapshot["durationMs"] = round((time.perf_counter() - started) * 1000)
        _atomic_write(directory / "results.json", snapshot)

    try:
        processed = []
        for index, file in enumerate(metadata["files"]):
            progress(index, 5 + 30 * index / total)
            item = {"original_filename": file["name"], "original_image_url": file["original_image_url"],
                    "status": "error", "processed_image_url": None, "error_message": None}
            try:
                final = run_preprocessing_pipeline(directory / file["stored_filename"]).get("final_image")
                if final is None:
                    raise ValueError("Pipeline did not return final_image.")
                output = directory / f"processed_{index}.png"
                if not cv2.imwrite(str(output), final):
                    raise ValueError("Failed to save processed image.")
                item.update(status="success", processed_image_url=f"{base_url}/api/static/jobs/{directory.name}/{output.name}")
            except Exception as error:
                item["error_message"] = str(error)
            processed.append(item)
        snapshot["preprocessing"] = _batch(processed)
        save()
        if not snapshot["preprocessing"]["successful_count"]:
            raise ValueError(processed[0]["error_message"] or "Every image failed preprocessing.")

        if metadata["workflow"]:
            stage = "DETECTING"
            detected = []
            for index, item in enumerate(processed):
                progress(index, 35 + 30 * index / total)
                if item["status"] != "success":
                    continue
                stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(item["original_filename"]).stem)
                filename = f"module1_{stem or 'image'}_{index + 1}.png"
                try:
                    result = _process_path(directory / f"processed_{index}.png", filename, base_url)
                except Exception as error:
                    result = {"original_filename": filename, "status": "error", "error_message": str(error)}
                result["source_original_filename"] = item["original_filename"]
                detected.append(result)
            snapshot["detection"] = _batch(detected)
            save()
            if not snapshot["detection"]["successful_count"]:
                raise ValueError(detected[0].get("error_message") or "Every image failed detection.")

            stage = "CLASSIFYING"
            progress(total, 65)
            tracks = []
            for item in detected:
                calibration = metadata["calibrations"].get(item["source_original_filename"], {}).get("data", {})
                for track in item.get("tracks", []):
                    key = f"{item['run_id']}:{track['track_id']}"
                    if not track.get("roi_mask_url"):
                        snapshot["classifications"][key] = {"status": "unavailable"}
                        continue
                    fields = ("track_id", "component_id", "label_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "roi_x0", "roi_y0")
                    provenance = {field: track[field] for field in fields if track.get(field) is not None}
                    provenance["request_id"] = key
                    for field in ("mm_per_pixel", "measurement_homography"):
                        if calibration.get(field) is not None:
                            provenance[field] = calibration[field]
                    tracks.append((key, track["roi_mask_url"], provenance))
            all_results = []
            for offset in range(0, len(tracks), 32):
                chunk = tracks[offset:offset + 32]
                with ExitStack() as stack:
                    uploads = [UploadFile(file=stack.enter_context(_local_output(url).open("rb")), filename=f"track_{offset + i}.png")
                               for i, (_, url, _) in enumerate(chunk)]
                    batch = asyncio.run(classify_masks_batch(uploads, json.dumps([entry[2] for entry in chunk])))
                if not isinstance(batch, dict):
                    payload = json.loads(batch.body)
                    raise ValueError(payload.get("error_message", "Classification failed."))
                for (key, _, _), result in zip(chunk, batch["results"]):
                    snapshot["classifications"][key] = ({"status": "success", "result": result}
                        if result["processing_status"] == "success" else {"status": "error", "message": result.get("error_message")})
                all_results.extend(batch["results"])
                progress(total, 65 + 34 * len(all_results) / max(len(tracks), 1))
            successful = [item for item in all_results if item["processing_status"] == "success"]
            snapshot["classificationSummary"] = {
                "total_tracks": len(all_results), "successful_tracks": len(successful),
                "failed_tracks": len(all_results) - len(successful),
                "alpha_count": sum(item["predicted_class"] == "Alpha" for item in successful),
                "electron_count": sum(item["predicted_class"] == "Electron" for item in successful),
            }
            failures = snapshot["detection"]["failed_count"] + snapshot["classificationSummary"]["failed_tracks"]
            if failures:
                snapshot["warning"] = f"{failures} image/track operations failed; successful results remain available."
        save()
        write_job_status(directory, status="COMPLETED", stage="COMPLETED", current_frame=total,
                         total_frames=total, progress_percent=100, message="Image analysis complete")
    except Exception as error:
        save()
        write_job_status(directory, status="FAILED", stage=stage, total_frames=total,
                         error=str(error) or type(error).__name__, message="Image processing failed")


@router.post("/process", status_code=202)
def start_image_job(request: Request, background_tasks: BackgroundTasks,
                    files: list[UploadFile] = File(...), workflow: bool = Form(False),
                    calibrations_json: str = Form("{}")):
    if not files or any(Path(file.filename or "").suffix.lower() not in SUPPORTED_IMAGE_FORMATS for file in files):
        raise HTTPException(400, "Select supported image files.")
    try:
        calibrations = json.loads(calibrations_json)
        if not isinstance(calibrations, dict) or any(not isinstance(value, dict) for value in calibrations.values()):
            raise ValueError
    except (ValueError, TypeError) as error:
        raise HTTPException(400, "Invalid image calibration metadata.") from error
    directory = OUTPUT_IMAGE_DIR / "jobs" / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    base = str(request.base_url).rstrip("/")
    metadata = {"job_id": directory.name, "workflow": workflow, "calibrations": calibrations, "files": []}
    try:
        for index, file in enumerate(files):
            stored = f"original_{index}{Path(file.filename).suffix.lower()}"
            with (directory / stored).open("wb") as output:
                shutil.copyfileobj(file.file, output)
            metadata["files"].append({"name": file.filename, "type": file.content_type,
                "size": (directory / stored).stat().st_size, "stored_filename": stored,
                "original_image_url": f"{base}/api/static/jobs/{directory.name}/{stored}"})
        _atomic_write(directory / "metadata.json", metadata)
        write_job_status(directory, status="QUEUED", stage="PREPROCESSING", total_frames=len(files), message="Image processing queued")
        background_tasks.add_task(_run_job, directory, metadata, base)
        return {"job_id": directory.name, "status": "QUEUED"}
    except Exception as error:
        # Only this newly allocated, server-generated directory is removed.
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(500, "Unable to save image upload.") from error
    finally:
        for file in files:
            file.file.close()


@router.get("/status/{job_id}")
def image_status(job_id: str):
    directory = _directory(job_id)
    try:
        status = read_job_status(directory)
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(503, "Image status is temporarily unavailable.") from error
    metadata = _read(directory / "metadata.json")
    return {**status, "files": metadata["files"], "calibrations": metadata["calibrations"]}


@router.get("/result/{job_id}")
def image_result(job_id: str):
    directory = _directory(job_id)
    status = image_status(job_id)
    if status["status"] not in {"COMPLETED", "FAILED"}:
        raise HTTPException(409, "Image processing is not complete.")
    return {"job_id": job_id, **_read(directory / "results.json")}
