"""Temporary ROI segmentation used until Module 2 supplies track masks."""

from __future__ import annotations

import cv2
import numpy as np


ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = -3
MIN_COMPONENT_AREA = 12
MAX_COMPONENT_FRACTION = 0.55


def _best_component(candidate: np.ndarray) -> tuple[np.ndarray | None, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    best_label, best_score = None, -np.inf
    max_area = candidate.size * MAX_COMPONENT_FRACTION
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA or area > max_area:
            continue
        x, y, width, height = map(int, stats[label, :4])
        elongation = max(width, height) / max(1.0, min(width, height))
        continuity = area / max(1.0, width * height)
        touches = x == 0 or y == 0 or x + width == candidate.shape[1] or y + height == candidate.shape[0]
        border_penalty = 0.45 if touches else 1.0
        score = border_penalty * np.sqrt(area) * min(elongation, 25.0) * (0.35 + continuity)
        if score > best_score:
            best_label, best_score = label, score
    return (labels == best_label) if best_label is not None else None, best_score


def segment_track(roi: np.ndarray) -> dict[str, object]:
    """Segment the most plausible bright elongated component in an ROI."""
    if roi is None or roi.size == 0:
        return {"processing_status": "failed", "error_message": "ROI is empty.", "mask": None}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.astype(np.uint8)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    block = min(ADAPTIVE_BLOCK_SIZE, min(enhanced.shape) // 2 * 2 - 1)
    block = max(3, block)
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, ADAPTIVE_C)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    percentile = np.percentile(enhanced, 82)
    bright = np.where(enhanced >= percentile, 255, 0).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = [cv2.morphologyEx(value, cv2.MORPH_CLOSE, close_kernel, iterations=1) for value in (adaptive, otsu, bright)]
    choices = [_best_component(value) for value in candidates]
    best_mask, score = max(choices, key=lambda item: item[1])
    if best_mask is None or not np.isfinite(score):
        return {"processing_status": "failed", "error_message": "No plausible bright track component was found in the ROI.", "mask": None}
    return {"processing_status": "success", "error_message": None, "mask": best_mask.astype(np.uint8) * 255, "enhanced_grayscale": enhanced}
