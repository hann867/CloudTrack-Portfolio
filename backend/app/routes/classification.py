"""API boundary for Module 3 single-mask geometric classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
from fastapi import APIRouter, File, Form, UploadFile
import numpy as np
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from backend.src.geometric_classification.production_inference import (
    BATCH_STATUS_FAILED,
    classify_module2_mask,
    classify_module2_masks,
    summarize_module3_results,
)


router = APIRouter(prefix="/classification", tags=["classification"])

MAX_MASK_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_CLASSIFICATION_BATCH_TRACKS = 32
SUPPORTED_MASK_SUFFIXES = {".png"}


class ClassificationProbabilities(BaseModel):
    Alpha: float
    Electron: float


class ClassificationFeatures(BaseModel):
    track_length_px: float
    mean_width_px: float
    median_width_px: float
    max_width_px: float
    mean_curvature: float
    max_curvature: float
    tortuosity: float


class ClassificationPhysicalFeatures(BaseModel):
    track_length_mm: float
    mean_width_mm: float
    median_width_mm: float
    max_width_mm: float


class ClassificationModel(BaseModel):
    model_version: str
    model_stage: Literal["development"]


class ClassificationSuccess(BaseModel):
    processing_status: Literal["success"]
    predicted_class: Literal["Alpha", "Electron"]
    confidence: float
    probabilities: ClassificationProbabilities
    features: ClassificationFeatures
    physical_features: ClassificationPhysicalFeatures | None = None
    model: ClassificationModel
    provenance: dict[str, Any]


class ClassificationFailure(BaseModel):
    processing_status: Literal["failed"] = "failed"
    error_code: str
    error_message: str


class ClassificationBatchSummary(BaseModel):
    total_tracks: int
    successful_tracks: int
    failed_tracks: int
    alpha_count: int
    electron_count: int
    success_rate: float


class ClassificationBatchResponse(BaseModel):
    processing_status: Literal["SUCCESS", "PARTIAL_SUCCESS", "FAILED"]
    error_code: str | None = None
    error_message: str | None = None
    summary: ClassificationBatchSummary
    results: list[dict[str, Any]]


PUBLIC_FAILURES: dict[str, tuple[int, str]] = {
    "missing_mask": (400, "A mask image is required."),
    "malformed_mask": (400, "The uploaded mask is not a valid 2-D grayscale mask."),
    "empty_mask": (400, "The uploaded mask is empty."),
    "no_foreground": (422, "The uploaded mask contains no foreground track."),
    "invalid_calibration": (422, "mm_per_pixel must be a positive finite number."),
    "invalid_provenance": (400, "The supplied provenance is invalid."),
    "geometric_extraction_failure": (422, "Geometric features could not be extracted from this mask."),
    "non_finite_feature_vector": (422, "The extracted geometric feature vector is invalid."),
    "feature_contract_mismatch": (503, "The Module 3 model feature contract is unavailable."),
    "class_contract_mismatch": (503, "The Module 3 model class contract is unavailable."),
    "missing_model_artifact": (503, "The Module 3 development model is unavailable."),
    "missing_model_metadata": (503, "The Module 3 development model metadata is unavailable."),
    "corrupt_model_artifact": (503, "The Module 3 development model could not be loaded."),
    "corrupt_model_metadata": (503, "The Module 3 development model metadata is invalid."),
    "model_type_mismatch": (503, "The Module 3 development model type is invalid."),
    "model_stage_mismatch": (503, "The required Module 3 development model is unavailable."),
    "model_version_mismatch": (503, "The required Module 3 model version is unavailable."),
    "artifact_access_failure": (503, "The Module 3 development model is unavailable."),
    "artifact_load_failure": (503, "The Module 3 development model could not be loaded."),
    "predict_failure": (500, "Module 3 classification failed."),
    "malformed_prediction_output": (500, "Module 3 returned an invalid prediction."),
    "malformed_probability_output": (500, "Module 3 returned invalid probabilities."),
}


def _failure_response(error_code: str) -> JSONResponse:
    status_code, message = PUBLIC_FAILURES.get(
        error_code, (500, "An unexpected Module 3 inference failure occurred.")
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "processing_status": "failed",
            "error_code": error_code,
            "error_message": message,
        },
    )


def _empty_batch_payload(error_code: str, error_message: str) -> dict[str, Any]:
    return {
        "processing_status": BATCH_STATUS_FAILED,
        "error_code": error_code,
        "error_message": error_message,
        "summary": {
            "total_tracks": 0,
            "successful_tracks": 0,
            "failed_tracks": 0,
            "alpha_count": 0,
            "electron_count": 0,
            "success_rate": 0.0,
        },
        "results": [],
    }


def _batch_track_failure(
    error_code: str,
    error_message: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Represent an upload-boundary failure with the production result shape."""
    return {
        "processing_status": "failed",
        "error_code": error_code,
        "error_message": error_message,
        "predicted_class": None,
        "confidence": None,
        "probability_alpha": None,
        "probability_electron": None,
        "probabilities": {},
        "features": {},
        "physical_features": {},
        "model_version": None,
        "model_stage": None,
        "model": {},
        "provenance": provenance,
    }


async def _close_uploads(files: list[UploadFile]) -> None:
    for file in files:
        await file.close()


def _normalise_batch_metadata(
    raw_metadata: str | None,
    files: list[UploadFile],
) -> tuple[list[dict[str, Any]] | None, JSONResponse | None]:
    if raw_metadata is None or not raw_metadata.strip():
        supplied: list[Any] = [{} for _ in files]
    else:
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return None, JSONResponse(
                status_code=400,
                content=_empty_batch_payload(
                    "invalid_batch_provenance",
                    "Batch provenance must be a valid JSON array.",
                ),
            )
        if not isinstance(parsed, list) or len(parsed) != len(files):
            return None, JSONResponse(
                status_code=400,
                content=_empty_batch_payload(
                    "invalid_batch_provenance",
                    "Batch provenance must contain exactly one object per uploaded mask.",
                ),
            )
        supplied = parsed

    allowed = {
        "request_id",
        "track_id",
        "component_id",
        "label_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "roi_x0",
        "roi_y0",
        "mm_per_pixel",
        "measurement_homography",
    }
    integer_fields = {
        "component_id",
        "label_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "roi_x0",
        "roi_y0",
    }
    metadata: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, (entry, file) in enumerate(zip(supplied, files), start=1):
        if not isinstance(entry, dict) or any(key not in allowed for key in entry):
            return None, JSONResponse(
                status_code=400,
                content=_empty_batch_payload(
                    "invalid_batch_provenance",
                    f"Batch provenance item {index} is not a supported provenance object.",
                ),
            )
        normalised: dict[str, Any] = {}
        try:
            for key, value in entry.items():
                if value is None or value == "":
                    continue
                if key in integer_fields:
                    if isinstance(value, bool):
                        raise ValueError
                    normalised[key] = int(value)
                elif key == "mm_per_pixel":
                    if isinstance(value, bool):
                        raise ValueError
                    normalised[key] = float(value)
                elif key == "measurement_homography":
                    matrix = np.asarray(value, dtype=float)
                    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                        raise ValueError
                    normalised[key] = matrix.tolist()
                elif key in {"request_id", "track_id"}:
                    normalised[key] = str(value)
        except (TypeError, ValueError, OverflowError):
            return None, JSONResponse(
                status_code=400,
                content=_empty_batch_payload(
                    "invalid_batch_provenance",
                    f"Batch provenance item {index} contains an invalid field value.",
                ),
            )

        filename = _safe_upload_name(file.filename)
        identity_value = str(
            normalised.get("request_id") or normalised.get("track_id") or filename
        ).strip().casefold()
        if not identity_value or identity_value in identities:
            return None, JSONResponse(
                status_code=400,
                content=_empty_batch_payload(
                    "duplicate_batch_identity",
                    "Every uploaded mask must have a unique request_id, track_id, or filename identity.",
                ),
            )
        identities.add(identity_value)
        normalised["source_filename"] = filename
        metadata.append(normalised)
    return metadata, None


def _safe_upload_name(filename: str | None) -> str:
    return Path(filename or "mask.png").name


async def _decode_mask_upload(file: UploadFile) -> tuple[np.ndarray | None, JSONResponse | None]:
    filename = _safe_upload_name(file.filename)
    suffix = Path(filename).suffix.lower()
    try:
        if suffix not in SUPPORTED_MASK_SUFFIXES:
            return None, JSONResponse(
                status_code=400,
                content={
                    "processing_status": "failed",
                    "error_code": "unsupported_mask_format",
                    "error_message": "Module 3 accepts lossless PNG mask uploads only.",
                },
            )
        content = await file.read(MAX_MASK_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if not content:
        return None, _failure_response("empty_mask")
    if len(content) > MAX_MASK_UPLOAD_BYTES:
        return None, JSONResponse(
            status_code=400,
            content={
                "processing_status": "failed",
                "error_code": "mask_too_large",
                "error_message": "The uploaded mask exceeds the 16 MiB limit.",
            },
        )
    encoded = np.frombuffer(content, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None, JSONResponse(
            status_code=400,
            content={
                "processing_status": "failed",
                "error_code": "unreadable_mask",
                "error_message": "The uploaded PNG could not be decoded as an image.",
            },
        )
    if decoded.ndim != 2:
        return None, _failure_response("malformed_mask")
    return decoded, None


@router.post(
    "/classify",
    response_model=ClassificationSuccess,
    responses={
        400: {"model": ClassificationFailure},
        422: {"model": ClassificationFailure},
        500: {"model": ClassificationFailure},
        503: {"model": ClassificationFailure},
    },
)
async def classify_mask(
    file: UploadFile = File(...),
    track_id: str | None = Form(default=None),
    component_id: int | None = Form(default=None),
    label_id: int | None = Form(default=None),
    bbox_x: int | None = Form(default=None),
    bbox_y: int | None = Form(default=None),
    bbox_width: int | None = Form(default=None),
    bbox_height: int | None = Form(default=None),
    roi_x0: int | None = Form(default=None),
    roi_y0: int | None = Form(default=None),
    mm_per_pixel: float | None = Form(default=None),
):
    """Classify one uploaded lossless Module 2 mask without re-segmentation."""
    mask, decode_failure = await _decode_mask_upload(file)
    if decode_failure is not None:
        return decode_failure
    provenance: dict[str, Any] = {"source_filename": _safe_upload_name(file.filename)}
    optional_provenance = {
        "track_id": track_id,
        "component_id": component_id,
        "label_id": label_id,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "roi_x0": roi_x0,
        "roi_y0": roi_y0,
    }
    provenance.update({key: value for key, value in optional_provenance.items() if value is not None})
    try:
        result = await run_in_threadpool(
            classify_module2_mask,
            mask,
            provenance=provenance,
            mm_per_pixel=mm_per_pixel,
        )
    except Exception:
        return _failure_response("unexpected_inference_failure")
    if result.get("processing_status") != "success":
        return _failure_response(str(result.get("error_code") or "unexpected_inference_failure"))
    return {
        "processing_status": "success",
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "features": result["features"],
        "physical_features": result["physical_features"] or None,
        "model": result["model"],
        "provenance": result["provenance"],
    }


@router.post(
    "/classify-batch",
    response_model=ClassificationBatchResponse,
    responses={
        400: {"model": ClassificationBatchResponse},
        413: {"model": ClassificationBatchResponse},
    },
)
async def classify_masks_batch(
    files: list[UploadFile] | None = File(default=None),
    provenance_json: str | None = Form(default=None),
):
    """Classify up to 32 existing Module 2 PNG masks without segmentation.

    Upload and decode failures become ordered per-track failure records. Valid
    masks are delegated to the reusable batch service, which invokes the
    authoritative ``classify_module2_mask`` function once per track.
    """
    uploads = list(files or [])
    if not uploads:
        return JSONResponse(
            status_code=400,
            content=_empty_batch_payload(
                "empty_batch",
                "At least one Module 2 mask is required for batch classification.",
            ),
        )
    if len(uploads) > MAX_CLASSIFICATION_BATCH_TRACKS:
        await _close_uploads(uploads)
        return JSONResponse(
            status_code=413,
            content=_empty_batch_payload(
                "batch_too_large",
                f"A Module 3 batch may contain at most {MAX_CLASSIFICATION_BATCH_TRACKS} masks.",
            ),
        )

    metadata, metadata_failure = _normalise_batch_metadata(provenance_json, uploads)
    if metadata_failure is not None:
        await _close_uploads(uploads)
        return metadata_failure
    assert metadata is not None

    ordered_results: list[dict[str, Any] | None] = [None] * len(uploads)
    valid_masks: list[np.ndarray] = []
    valid_provenances: list[dict[str, Any]] = []
    valid_scales: list[float | None] = []
    valid_indices: list[int] = []

    for index, (file, supplied) in enumerate(zip(uploads, metadata)):
        provenance = {key: value for key, value in supplied.items() if key != "mm_per_pixel"}
        scale = supplied.get("mm_per_pixel")
        mask, decode_failure = await _decode_mask_upload(file)
        if decode_failure is not None:
            try:
                payload = json.loads(bytes(decode_failure.body).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {
                    "error_code": "malformed_mask",
                    "error_message": "The uploaded mask could not be decoded.",
                }
            ordered_results[index] = _batch_track_failure(
                str(payload.get("error_code") or "malformed_mask"),
                str(payload.get("error_message") or "The uploaded mask is invalid."),
                provenance,
            )
            continue
        assert mask is not None
        valid_indices.append(index)
        valid_masks.append(mask)
        valid_provenances.append(provenance)
        valid_scales.append(scale)

    if valid_masks:
        try:
            valid_batch = await run_in_threadpool(
                classify_module2_masks,
                valid_masks,
                provenances=valid_provenances,
                mm_per_pixels=valid_scales,
            )
            valid_results = valid_batch["results"]
        except Exception:
            valid_results = [
                _batch_track_failure(
                    "unexpected_inference_failure",
                    "An unexpected Module 3 inference failure occurred.",
                    provenance,
                )
                for provenance in valid_provenances
            ]
        for index, result in zip(valid_indices, valid_results):
            ordered_results[index] = result

    complete_results = [result for result in ordered_results if result is not None]
    if len(complete_results) != len(uploads):
        complete_results = [
            result
            if result is not None
            else _batch_track_failure(
                "unexpected_inference_failure",
                "An unexpected Module 3 inference failure occurred.",
                {"source_filename": _safe_upload_name(uploads[index].filename)},
            )
            for index, result in enumerate(ordered_results)
        ]
    batch = summarize_module3_results(complete_results)
    batch["error_code"] = None
    batch["error_message"] = None
    return batch
