"""API endpoints for Module 2 particle-track detection and segmentation."""

from __future__ import annotations

import math
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.src.detection_segmentation import (
    CONFIG,
    CSV_COLUMNS,
    process_image,
    save_png,
    validate_config,
)
from backend.src.utils.path_utils import MODULE2_OUTPUT_DIR


router = APIRouter(prefix="/detection", tags=["detection"])

SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}
CSV_WRITE_LOCK = threading.Lock()
TRACK_RESPONSE_FIELDS = (
    "track_id",
    "label_id",
    "source_component_ids",
    "fragment_count",
    "trajectory_linked",
    "trajectory_link_count",
    "curve_fitted",
    "curve_model",
    "curve_score",
    "curve_contrast",
    "area_pixels",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "roi_x0",
    "roi_y0",
    "roi_width",
    "roi_height",
)


def _safe_stem(filename: str) -> str:
    """Return a filesystem-safe source stem while keeping it recognisable."""
    stem = Path(filename or "image").stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned[:100] or "image"


def _public_url(request_base: str, relative_path: str) -> str:
    relative_path = relative_path.replace("\\", "/").lstrip("/")
    return f"{request_base.rstrip('/')}/api/static/{relative_path}"


def _json_value(value):
    """Convert NumPy/pandas scalars and non-finite floats for strict JSON."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _save_result(result: dict, original_filename: str, request_base: str) -> dict:
    """Persist one Module 2 result and return its API representation."""
    run_id = uuid.uuid4().hex[:12]
    source_stem = _safe_stem(original_filename)
    output_stem = f"{source_stem}_{run_id}"

    output_files = {
        "binary_mask": MODULE2_OUTPUT_DIR / "binary_masks" / f"{output_stem}_binary.png",
        "cleaned_mask": MODULE2_OUTPUT_DIR / "cleaned_masks" / f"{output_stem}_cleaned.png",
        "labelled_mask": MODULE2_OUTPUT_DIR / "labelled_masks" / f"{output_stem}_labelled.png",
        "annotated_image": MODULE2_OUTPUT_DIR / "annotated_images" / f"{output_stem}_annotated.png",
    }
    save_png(output_files["binary_mask"], result["binary"])
    save_png(output_files["cleaned_mask"], result["filtered_mask"])

    maximum_label = int(result["accepted_labels"].max()) if result["accepted_labels"].size else 0
    if maximum_label > np.iinfo(np.uint16).max:
        raise OverflowError("Accepted-label count exceeds the uint16 PNG limit.")
    save_png(output_files["labelled_mask"], result["accepted_labels"].astype(np.uint16))
    save_png(output_files["annotated_image"], result["annotated"])

    csv_records = []
    tracks = []
    for record in result["rois"]:
        track_id = str(record["track_id"])
        roi_output_id = f"{track_id}_{run_id}"
        roi_image_path = MODULE2_OUTPUT_DIR / "rois" / "images" / f"{roi_output_id}.png"
        roi_mask_path = MODULE2_OUTPUT_DIR / "rois" / "masks" / f"{roi_output_id}_mask.png"
        save_png(roi_image_path, record["_image_roi"])
        save_png(roi_mask_path, record["_mask_roi"])

        csv_record = {column: record.get(column) for column in CSV_COLUMNS}
        csv_record["source_filename"] = original_filename
        csv_record["source_stem"] = source_stem
        csv_record["roi_image_file"] = roi_image_path.relative_to(MODULE2_OUTPUT_DIR).as_posix()
        csv_record["roi_mask_file"] = roi_mask_path.relative_to(MODULE2_OUTPUT_DIR).as_posix()
        csv_records.append(csv_record)

        track = {field: _json_value(record.get(field)) for field in TRACK_RESPONSE_FIELDS}
        track["roi_image_url"] = _public_url(
            request_base, roi_image_path.relative_to(MODULE2_OUTPUT_DIR.parent).as_posix())
        track["roi_mask_url"] = _public_url(
            request_base, roi_mask_path.relative_to(MODULE2_OUTPUT_DIR.parent).as_posix())
        tracks.append(track)

    region_table = pd.DataFrame(csv_records, columns=CSV_COLUMNS)
    master_csv_path = MODULE2_OUTPUT_DIR / "track_regions.csv"
    MODULE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_WRITE_LOCK:
        master_exists = master_csv_path.exists() and master_csv_path.stat().st_size > 0
        region_table.to_csv(
            master_csv_path,
            mode="a" if master_exists else "w",
            header=not master_exists,
            index=False,
            float_format="%.4f",
        )

    relative_outputs = {
        name: path.relative_to(MODULE2_OUTPUT_DIR.parent).as_posix()
        for name, path in output_files.items()
    }
    return {
        "run_id": run_id,
        "original_filename": original_filename,
        "status": "success",
        "segmentation_mode": result["segmentation_mode"],
        "raw_component_count": int(result["raw_component_count"]),
        "trajectory_group_count": int(result["trajectory_group_count"]),
        "linked_group_count": int(result["linked_group_count"]),
        "trajectory_link_count": int(result["trajectory_link_count"]),
        "accepted_track_count": int(result["accepted_component_count"]),
        "foreground_percent": {
            "adaptive": float(result["adaptive_foreground_percent"]),
            "opened": float(result["opened_foreground_percent"]),
            "closed": float(result["closed_foreground_percent"]),
            "final": float(result["final_foreground_percent"]),
        },
        "outputs": {
            name: _public_url(request_base, relative_path)
            for name, relative_path in relative_outputs.items()
        },
        "csv_url": _public_url(
            request_base, master_csv_path.relative_to(MODULE2_OUTPUT_DIR.parent).as_posix()),
        "master_csv_url": _public_url(
            request_base, master_csv_path.relative_to(MODULE2_OUTPUT_DIR.parent).as_posix()),
        "tracks": tracks,
    }


def _process_path(temp_path: Path, original_filename: str, request_base: str) -> dict:
    validate_config(CONFIG)
    result = process_image(temp_path, CONFIG)
    return _save_result(result, original_filename, request_base)


async def _detect_upload(file: UploadFile, request_base: str) -> dict:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in SUPPORTED_IMAGE_FORMATS:
        await file.close()
        raise ValueError("Module 2 accepts PNG and JPEG images only.")

    temp_dir = Path(tempfile.mkdtemp(prefix="cloudtrack_module2_"))
    temp_path = temp_dir / f"{_safe_stem(file.filename or 'image')}{extension}"
    try:
        with temp_path.open("wb") as temporary:
            shutil.copyfileobj(file.file, temporary)
        return await run_in_threadpool(
            _process_path, temp_path, file.filename or temp_path.name, request_base)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        await file.close()


@router.post("/detect")
async def detect_tracks(request: Request, file: UploadFile = File(...)):
    """Detect, segment, and extract track ROIs from one preprocessed image."""
    try:
        return await _detect_upload(file, str(request.base_url))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Module 2 processing failed: {exc}") from exc


@router.post("/batch")
async def detect_tracks_batch(request: Request, files: list[UploadFile] = File(...)):
    """Run Module 2 independently for each uploaded preprocessed image."""
    results = []
    successful_count = 0
    for file in files:
        try:
            result = await _detect_upload(file, str(request.base_url))
            successful_count += 1
        except Exception as exc:
            result = {
                "original_filename": file.filename,
                "status": "error",
                "error_message": str(exc),
            }
        results.append(result)
    return {
        "total_images": len(files),
        "successful_count": successful_count,
        "failed_count": len(files) - successful_count,
        "results": results,
    }
