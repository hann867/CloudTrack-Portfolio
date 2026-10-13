"""GUI-independent ROI coordinate and cropping helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


ROI = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass(frozen=True)
class RotatedROI:
    """An oriented rectangle expressed in original-image coordinates.

    Corners are ordered top-left, top-right, bottom-right, bottom-left relative
    to the user-defined width axis. The angle is in OpenCV image coordinates:
    positive angles rotate clockwise because image y increases downwards.
    """

    center_x: float
    center_y: float
    width: float
    height: float
    angle_degrees: float
    corners: tuple[Point, Point, Point, Point]


def rotated_roi_from_center(
    center: Point,
    width: float,
    height: float,
    angle_degrees: float,
) -> RotatedROI:
    """Construct an oriented ROI without inspecting or classifying image data."""

    if not all(math.isfinite(value) for value in (*center, width, height, angle_degrees)):
        raise ValueError("Rotated ROI values must be finite")
    if width < 2 or height < 2:
        raise ValueError("Rotated ROI width and height must each be at least 2 pixels")
    angle_radians = math.radians(angle_degrees)
    unit_x = (math.cos(angle_radians), math.sin(angle_radians))
    unit_y = (-unit_x[1], unit_x[0])
    half_width = width / 2.0
    half_height = height / 2.0
    center_x, center_y = center

    corners = tuple(
        (
            center_x + width_sign * half_width * unit_x[0] + height_sign * half_height * unit_y[0],
            center_y + width_sign * half_width * unit_x[1] + height_sign * half_height * unit_y[1],
        )
        for width_sign, height_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    )
    return RotatedROI(center_x, center_y, width, height, angle_degrees, corners)  # type: ignore[arg-type]


def rotated_roi_from_axis(start: Point, end: Point, half_height: float) -> RotatedROI:
    """Create an oriented ROI from a user-drawn centreline and half-height."""

    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    width = math.hypot(delta_x, delta_y)
    if not math.isfinite(half_height) or half_height < 1:
        raise ValueError("Rotated ROI background half-height must be at least 1 pixel")
    center = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return rotated_roi_from_center(center, width, 2.0 * half_height, angle)


def validate_rotated_roi(roi: RotatedROI, image_width: int, image_height: int) -> RotatedROI:
    """Require the complete oriented rectangle to lie inside the source image."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if roi.width < 2 or roi.height < 2 or len(roi.corners) != 4:
        raise ValueError("Rotated ROI must be a non-empty four-corner rectangle")
    for x, y in roi.corners:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Rotated ROI coordinates must be finite")
        if x < 0 or y < 0 or x > image_width - 1 or y > image_height - 1:
            raise ValueError("Rotated ROI extends outside the image")
    return roi


def rotated_roi_bounding_box(roi: RotatedROI, image_width: int, image_height: int) -> ROI:
    """Return a compatible axis-aligned bounding box for legacy x/y fields."""

    validate_rotated_roi(roi, image_width, image_height)
    left = max(0, math.floor(min(point[0] for point in roi.corners)))
    top = max(0, math.floor(min(point[1] for point in roi.corners)))
    right = min(image_width, math.ceil(max(point[0] for point in roi.corners)) + 1)
    bottom = min(image_height, math.ceil(max(point[1] for point in roi.corners)) + 1)
    return validate_roi((left, top, right - left, bottom - top), image_width, image_height)


def extract_rotated_crop(image: np.ndarray, roi: RotatedROI) -> np.ndarray:
    """Extract one oriented crop with one perspective-warp image operation.

    Unlike an axis-aligned slice, an oriented raster crop cannot generally be
    formed from unchanged source pixels. ``cv2.warpPerspective`` performs
    unavoidable bilinear interpolation while mapping the four source corners
    directly to the output rectangle. No separate resize, rotation, filtering,
    enhancement, thresholding, or other preprocessing is performed.
    """

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Image must be a 2D or 3D NumPy array")
    validate_rotated_roi(roi, image.shape[1], image.shape[0])
    output_width = max(2, round(roi.width))
    output_height = max(2, round(roi.height))
    source_points = np.asarray(roi.corners, dtype=np.float32)
    destination_points = np.asarray(
        (
            (0, 0),
            (output_width - 1, 0),
            (output_width - 1, output_height - 1),
            (0, output_height - 1),
        ),
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_points, destination_points)
    return cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def validate_roi(roi: Sequence[int | float], image_width: int, image_height: int) -> ROI:
    """Validate and normalize an ``(x, y, width, height)`` ROI.

    Coordinates must describe a non-empty rectangle entirely inside the image.
    Floating point values are accepted only when they represent whole pixels.
    """

    if len(roi) != 4:
        raise ValueError("ROI must contain x, y, width, and height")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    values = tuple(int(value) for value in roi)
    if any(float(original) != normalized for original, normalized in zip(roi, values)):
        raise ValueError("ROI coordinates must map to whole pixels")
    x, y, width, height = values
    if x < 0 or y < 0:
        raise ValueError("ROI origin cannot be negative")
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    if x + width > image_width or y + height > image_height:
        raise ValueError("ROI extends outside the image")
    return values


def map_display_roi_to_original(
    display_roi: Sequence[int | float],
    original_size: tuple[int, int],
    display_size: tuple[int, int],
    *,
    original_offset: tuple[float, float] = (0.0, 0.0),
    viewport_original_size: tuple[float, float] | None = None,
) -> ROI:
    """Map a display-space ROI back to original full-resolution pixels.

    Sizes are ``(width, height)``. For a zoomed/panned display, provide the
    viewport's top-left ``original_offset`` and its visible
    ``viewport_original_size``. Start coordinates are floored and end
    coordinates are ceiled so fractional display scaling cannot trim content.
    """

    original_width, original_height = original_size
    display_width, display_height = display_size
    validate_roi(display_roi, display_width, display_height)
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Original image dimensions must be positive")

    x, y, width, height = (float(value) for value in display_roi)
    offset_x, offset_y = original_offset
    viewport_width, viewport_height = viewport_original_size or original_size
    if viewport_width <= 0 or viewport_height <= 0 or offset_x < 0 or offset_y < 0:
        raise ValueError("Viewport dimensions and offset must be valid")
    if offset_x + viewport_width > original_width or offset_y + viewport_height > original_height:
        raise ValueError("Viewport extends outside the original image")
    scale_x = viewport_width / display_width
    scale_y = viewport_height / display_height

    original_x = math.floor(offset_x + x * scale_x)
    original_y = math.floor(offset_y + y * scale_y)
    original_right = math.ceil(offset_x + (x + width) * scale_x)
    original_bottom = math.ceil(offset_y + (y + height) * scale_y)
    mapped = (
        original_x,
        original_y,
        original_right - original_x,
        original_bottom - original_y,
    )
    return validate_roi(mapped, original_width, original_height)


def crop_original_pixels(image: np.ndarray, roi: Sequence[int | float]) -> np.ndarray:
    """Return an unprocessed copy of an ROI from the original image array."""

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Image must be a 2D or 3D NumPy array")
    x, y, width, height = validate_roi(roi, image.shape[1], image.shape[0])
    return image[y : y + height, x : x + width].copy()
