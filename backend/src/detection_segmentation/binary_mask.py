"""Adaptive-threshold binary segmentation."""

import warnings

import cv2
import numpy as np


def adaptive_segment(gray, config):
    """Segment a 2-D uint8 image, preserving masks that are already binary."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("adaptive_segment expects a 2-D uint8 grayscale image.")
    if config.get("preserve_near_binary", True):
        tolerance = int(config.get("near_binary_tolerance", 5))
        endpoint = (gray <= tolerance) | (gray >= 255 - tolerance)
        low_fraction = float(np.mean(gray <= tolerance))
        high_fraction = float(np.mean(gray >= 255 - tolerance))
        if (
                float(np.mean(endpoint)) >= float(config.get("near_binary_required_fraction", 0.995))
                and low_fraction >= float(config.get("near_binary_min_endpoint_fraction", 0.0001))
                and high_fraction >= float(config.get("near_binary_min_endpoint_fraction", 0.0001))):
            if config["foreground_polarity"].lower() == "bright":
                return np.where(gray >= 128, 255, 0).astype(np.uint8)
            return np.where(gray < 128, 255, 0).astype(np.uint8)
    working = gray
    blur_size = config["blur_kernel_size"]
    if blur_size is not None:
        working = cv2.GaussianBlur(working, (blur_size, blur_size), sigmaX=0)
    method_map = {
        "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    }
    threshold_type = (
        cv2.THRESH_BINARY if config["foreground_polarity"].lower() == "bright"
        else cv2.THRESH_BINARY_INV
    )
    mask = cv2.adaptiveThreshold(
        working, 255, method_map[config["adaptive_method"].lower()], threshold_type,
        config["adaptive_block_size"], float(config["adaptive_C"]),
    )
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    foreground_percent = 100.0 * np.count_nonzero(mask) / mask.size
    if foreground_percent > float(config["max_foreground_percent"]):
        warnings.warn(
            f"Adaptive mask is {foreground_percent:.2f}% foreground. This exceeds the "
            f"configured {config['max_foreground_percent']:.2f}% QC limit; check C and "
            "foreground polarity before trusting this result."
        )
    return mask
