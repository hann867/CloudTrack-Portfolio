import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.src.calibration.calibration_metadata import (
    list_calibration_profiles,
    load_calibration_metadata,
    load_calibration_profile,
    save_calibration_metadata,
    save_calibration_profile,
)

router = APIRouter(prefix="/calibration", tags=["calibration"])


class CalibrationRequest(BaseModel):
    point_1_x: float = Field(..., description="Top-left chamber corner X in source-image pixels")
    point_1_y: float = Field(..., description="Top-left chamber corner Y in source-image pixels")
    point_2_x: float = Field(..., description="Top-right chamber corner X in source-image pixels")
    point_2_y: float = Field(..., description="Top-right chamber corner Y in source-image pixels")
    point_3_x: float = Field(..., description="Bottom-right chamber corner X in source-image pixels")
    point_3_y: float = Field(..., description="Bottom-right chamber corner Y in source-image pixels")
    point_4_x: float = Field(..., description="Bottom-left chamber corner X in source-image pixels")
    point_4_y: float = Field(..., description="Bottom-left chamber corner Y in source-image pixels")
    chamber_width_mm: float = Field(..., gt=0)
    chamber_height_mm: float = Field(..., gt=0)
    profile_name: str = Field(..., min_length=1)
    source_image_width: int = Field(..., gt=0)
    source_image_height: int = Field(..., gt=0)
    image_filename: Optional[str] = None


def _four_point_metadata(
    points: list[tuple[float, float]],
    chamber_width_mm: float,
    chamber_height_mm: float,
) -> dict:
    if not all(math.isfinite(value) for point in points for value in point):
        raise HTTPException(status_code=422, detail="Calibration points must be finite.")
    if not all(math.isfinite(value) and value > 0 for value in (chamber_width_mm, chamber_height_mm)):
        raise HTTPException(status_code=422, detail="Chamber width and height must be positive finite values.")
    if any(
        math.dist(points[left], points[right]) <= 1e-6
        for left in range(4)
        for right in range(left + 1, 4)
    ):
        raise HTTPException(status_code=400, detail="Each chamber corner must be a distinct point.")

    source = np.asarray(points, dtype=np.float32)
    contour = source.reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) <= 1:
        raise HTTPException(status_code=400, detail="Chamber corners must form a non-crossing A, B, C, D region.")
    destination = np.asarray([
        [0.0, 0.0],
        [chamber_width_mm, 0.0],
        [chamber_width_mm, chamber_height_mm],
        [0.0, chamber_height_mm],
    ], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, destination)
    if not np.isfinite(homography).all():
        raise HTTPException(status_code=400, detail="The selected corners do not define a valid measurement transform.")

    width_px = (math.dist(points[0], points[1]) + math.dist(points[3], points[2])) / 2
    height_px = (math.dist(points[0], points[3]) + math.dist(points[1], points[2])) / 2
    horizontal_scale = chamber_width_mm / width_px
    vertical_scale = chamber_height_mm / height_px
    return {
        "calibration_type": "four_point_measurement",
        "selected_corners": {
            label: {"x": point[0], "y": point[1]}
            for label, point in zip(("A", "B", "C", "D"), points)
        },
        "chamber_width_mm": chamber_width_mm,
        "chamber_height_mm": chamber_height_mm,
        "reference_width_px": width_px,
        "reference_height_px": height_px,
        "horizontal_mm_per_pixel": horizontal_scale,
        "vertical_mm_per_pixel": vertical_scale,
        "mm_per_pixel": (horizontal_scale + vertical_scale) / 2,
        "measurement_homography": homography.tolist(),
    }


def _normalise_profile(metadata: dict) -> dict | None:
    """Expose older four-corner profiles through the measurement-only contract."""
    if metadata.get("calibration_type") == "four_point_measurement":
        return None
    corners = metadata.get("selected_corners")
    if not isinstance(corners, dict):
        corners = {
            label: metadata.get(f"point_{index}")
            for index, label in enumerate(("A", "B", "C", "D"), start=1)
        }
    try:
        points = [(float(corners[label]["x"]), float(corners[label]["y"])) for label in ("A", "B", "C", "D")]
        measurement = _four_point_metadata(
            points,
            float(metadata["chamber_width_mm"]),
            float(metadata["chamber_height_mm"]),
        )
    except (KeyError, TypeError, ValueError, HTTPException):
        return metadata
    identity = {
        key: metadata[key]
        for key in ("profile_name", "profile_id", "source_image_resolution", "image_filename", "created_at", "updated_at")
        if key in metadata
    }
    return {**measurement, **identity}


@router.post("/calculate")
async def calculate_calibration(req: CalibrationRequest):
    profile_name = req.profile_name.strip()
    if not profile_name:
        raise HTTPException(status_code=422, detail="Calibration profile name must not be empty.")
    points = [
        (req.point_1_x, req.point_1_y),
        (req.point_2_x, req.point_2_y),
        (req.point_3_x, req.point_3_y),
        (req.point_4_x, req.point_4_y),
    ]
    metadata = {
        **_four_point_metadata(points, req.chamber_width_mm, req.chamber_height_mm),
        "profile_name": profile_name,
        "source_image_resolution": {"width": req.source_image_width, "height": req.source_image_height},
    }
    if req.image_filename:
        metadata["image_filename"] = Path(req.image_filename).name
    filename = f"{req.image_filename}_calibration.json" if req.image_filename else "calibration_metadata.json"
    try:
        metadata = save_calibration_profile(metadata)
        save_calibration_metadata(metadata, filename)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {error}") from error
    return metadata


@router.get("/profiles")
async def get_calibration_profiles():
    profiles = [_normalise_profile(profile) for profile in list_calibration_profiles()]
    return {"profiles": [profile for profile in profiles if profile is not None]}


@router.get("/profiles/{profile_id}")
async def get_calibration_profile(profile_id: str):
    try:
        profile = _normalise_profile(load_calibration_profile(profile_id))
        if profile is None:
            raise HTTPException(status_code=400, detail="This is not a reusable four-point chamber profile.")
        return profile
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Calibration profile was not found.")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.get("/metadata")
async def get_calibration_metadata(image_filename: str):
    safe_image_filename = Path(image_filename).name
    if not safe_image_filename:
        raise HTTPException(status_code=400, detail="Image filename must not be empty.")
    try:
        metadata = load_calibration_metadata(f"{safe_image_filename}_calibration.json")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No saved calibration exists for this image.")
    except ValueError as error:
        raise HTTPException(status_code=500, detail=f"Saved calibration metadata is invalid: {error}")
    normalised = _normalise_profile(metadata)
    if normalised is None:
        raise HTTPException(status_code=400, detail="This image does not have four-point chamber calibration.")
    return {**normalised, "image_filename": safe_image_filename}
