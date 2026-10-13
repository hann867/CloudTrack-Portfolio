"""Detection and cropping of manually drawn red ROI rectangles."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


RED_LOW_1 = (0, 65, 55)
RED_HIGH_1 = (12, 255, 255)
RED_LOW_2 = (165, 65, 55)
RED_HIGH_2 = (179, 255, 255)
MIN_RECTANGLE_SIDE = 12


def _red_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1)
    high = cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    mask = cv2.bitwise_or(low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def detect_red_rectangle(image: np.ndarray) -> dict[str, Any]:
    """Return the most likely red annotation rectangle and its HSV mask."""
    if image is None or image.size == 0:
        return {"processing_status": "failed", "error_message": "Image is empty."}

    mask = _red_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_h, image_w = image.shape[:2]
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < MIN_RECTANGLE_SIDE or height < MIN_RECTANGLE_SIDE:
            continue
        box_area = float(width * height)
        if box_area > 0.98 * image_w * image_h:
            continue

        thickness = max(2, min(8, round(min(width, height) * 0.08)))
        box_mask = mask[y : y + height, x : x + width]
        edge = np.zeros_like(box_mask, dtype=bool)
        edge[:thickness, :] = True
        edge[-thickness:, :] = True
        edge[:, :thickness] = True
        edge[:, -thickness:] = True
        edge_support = float(np.count_nonzero(box_mask[edge])) / max(1, int(edge.sum()))
        if edge_support < 0.06:
            continue

        perimeter = max(1.0, 2.0 * (width + height))
        contour_support = min(1.0, cv2.arcLength(contour, True) / perimeter)
        size_score = np.sqrt(box_area / (image_w * image_h))
        score = 4.0 * edge_support + contour_support + size_score
        candidates.append((score, (x, y, width, height)))

    if not candidates:
        return {
            "processing_status": "failed",
            "error_message": "No plausible red annotation rectangle was detected.",
            "red_mask": mask,
        }

    _, (x, y, width, height) = max(candidates, key=lambda item: item[0])
    return {
        "x": int(x), "y": int(y), "width": int(width), "height": int(height),
        "x2": int(x + width), "y2": int(y + height),
        "processing_status": "success", "error_message": None, "red_mask": mask,
    }


def crop_inside_rectangle(
    image: np.ndarray, rectangle: dict[str, Any], inward_margin: int = 4
) -> np.ndarray:
    """Crop inside a detected rectangle, excluding its red border."""
    if rectangle.get("processing_status") != "success":
        raise ValueError(rectangle.get("error_message") or "Invalid rectangle")
    margin = max(1, int(inward_margin))
    x, y = int(rectangle["x"]), int(rectangle["y"])
    x2_outer, y2_outer = int(rectangle["x2"]), int(rectangle["y2"])
    width, height = x2_outer - x, y2_outer - y
    left_inner, right_inner = 0, width - 1
    top_inner, bottom_inner = 0, height - 1
    red_mask = rectangle.get("red_mask")
    if isinstance(red_mask, np.ndarray):
        box = red_mask[y:y2_outer, x:x2_outer] > 0
        column_support = box.sum(axis=0)
        row_support = box.sum(axis=1)
        vertical = np.flatnonzero(column_support >= max(3, round(0.25 * height)))
        horizontal = np.flatnonzero(row_support >= max(3, round(0.25 * width)))
        left = vertical[vertical < width / 2]
        right = vertical[vertical >= width / 2]
        top = horizontal[horizontal < height / 2]
        bottom = horizontal[horizontal >= height / 2]
        if left.size and right.size:
            left_inner, right_inner = int(left.max()), int(right.min())
        if top.size and bottom.size:
            top_inner, bottom_inner = int(top.max()), int(bottom.min())
    x1 = x + left_inner + margin
    y1 = y + top_inner + margin
    x2 = x + right_inner - margin + 1
    y2 = y + bottom_inner - margin + 1
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detected rectangle is too small after applying inward margin.")
    return image[y1:y2, x1:x2].copy()
