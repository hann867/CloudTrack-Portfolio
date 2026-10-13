"""Noise reduction filters for grayscale cloud chamber images.

Median filtering is the recommended default because it reduces noise while
preserving particle track boundaries.
"""

from numbers import Integral

import cv2
import numpy as np


DEFAULT_NOISE_REDUCTION_METHOD = "median"


def _validate_grayscale_image(image: np.ndarray) -> None:
    """Validate that an input is a non-empty, two-dimensional image."""
    if image is None:
        raise ValueError("Input grayscale image must not be empty or None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Input grayscale image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Input grayscale image must not be empty.")
    if image.ndim != 2:
        raise ValueError("Input image must be a two-dimensional grayscale image.")


def _validate_odd_kernel_size(kernel_size: int, minimum: int) -> int:
    """Validate and normalize an odd OpenCV kernel size."""
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, Integral):
        raise TypeError("Kernel size must be an integer.")

    normalized_size = int(kernel_size)
    if normalized_size < minimum or normalized_size % 2 == 0:
        raise ValueError(
            f"Kernel size must be an odd integer greater than or equal to {minimum}."
        )

    return normalized_size


def _ensure_dimensions_preserved(
    original_image: np.ndarray, filtered_image: np.ndarray
) -> np.ndarray:
    """Ensure a filtering operation has preserved the input dimensions."""
    if filtered_image.shape != original_image.shape:
        raise RuntimeError("Noise reduction changed the image dimensions unexpectedly.")
    return filtered_image


def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply the recommended boundary-preserving median noise filter."""
    _validate_grayscale_image(image)
    validated_kernel_size = _validate_odd_kernel_size(kernel_size, minimum=3)
    filtered_image = cv2.medianBlur(image, validated_kernel_size)
    return _ensure_dimensions_preserved(image, filtered_image)


def apply_gaussian_filter(
    image: np.ndarray, kernel_size: int = 5, sigma: float = 0
) -> np.ndarray:
    """Apply Gaussian smoothing to a grayscale image."""
    _validate_grayscale_image(image)
    validated_kernel_size = _validate_odd_kernel_size(kernel_size, minimum=1)
    filtered_image = cv2.GaussianBlur(
        image,
        (validated_kernel_size, validated_kernel_size),
        sigmaX=sigma,
    )
    return _ensure_dimensions_preserved(image, filtered_image)


def apply_bilateral_filter(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """Apply edge-preserving bilateral filtering to a grayscale image."""
    _validate_grayscale_image(image)
    filtered_image = cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space,
    )
    return _ensure_dimensions_preserved(image, filtered_image)
