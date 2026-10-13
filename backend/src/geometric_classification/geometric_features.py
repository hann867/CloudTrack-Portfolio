"""Measurements derived from a binary track mask and ordered centreline."""

from __future__ import annotations

import cv2
import numpy as np


def path_length(path: np.ndarray) -> float:
    differences = np.diff(np.asarray(path, dtype=float), axis=0)
    return float(np.linalg.norm(differences, axis=1).sum())


def width_statistics(mask: np.ndarray, path: np.ndarray) -> dict[str, float]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    distances = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    coordinates = np.asarray(path, dtype=int)
    widths = 2.0 * distances[coordinates[:, 0], coordinates[:, 1]]
    return {"mean_width_px": float(np.mean(widths)), "median_width_px": float(np.median(widths)), "max_width_px": float(np.max(widths))}


def curvature_statistics(path: np.ndarray, smoothing_window: int = 5) -> dict[str, float]:
    """Measure curvature after arc-length resampling and light smoothing."""
    coordinates = np.asarray(path, dtype=float)
    if len(coordinates) < 5:
        return {"mean_curvature": 0.0, "max_curvature": 0.0}
    steps = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    sample_s = np.arange(0.0, cumulative[-1] + 1e-9, 1.0)
    if len(sample_s) < 5:
        sample_s = np.linspace(0.0, cumulative[-1], 5)
    y = np.interp(sample_s, cumulative, coordinates[:, 0])
    x = np.interp(sample_s, cumulative, coordinates[:, 1])
    window = min(smoothing_window, len(sample_s) if len(sample_s) % 2 else len(sample_s) - 1)
    if window >= 3:
        kernel = np.ones(window) / window
        pad = window // 2
        y = np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")
        x = np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")
    dx, dy = np.gradient(x, sample_s), np.gradient(y, sample_s)
    ddx, ddy = np.gradient(dx, sample_s), np.gradient(dy, sample_s)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(np.abs(dx * ddy - dy * ddx), denominator, out=np.zeros_like(denominator), where=denominator > 1e-8)
    core = curvature[2:-2] if len(curvature) > 6 else curvature
    return {"mean_curvature": float(np.mean(core)) if core.size else 0.0, "max_curvature": float(np.max(core)) if core.size else 0.0}


def tortuosity(path: np.ndarray, length: float | None = None) -> tuple[float, float]:
    coordinates = np.asarray(path, dtype=float)
    endpoint_distance = float(np.linalg.norm(coordinates[-1] - coordinates[0]))
    if endpoint_distance <= 1e-12:
        return float("nan"), endpoint_distance
    actual_length = path_length(coordinates) if length is None else float(length)
    return actual_length / endpoint_distance, endpoint_distance
