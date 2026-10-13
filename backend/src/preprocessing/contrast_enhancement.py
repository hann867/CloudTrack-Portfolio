"""Contrast enhancement methods for grayscale cloud chamber images.

CLAHE is the recommended default because it improves faint local track detail
while handling uneven illumination.
"""

from numbers import Integral, Real

import cv2
import numpy as np


DEFAULT_CONTRAST_ENHANCEMENT_METHOD = "clahe"


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


def _ensure_dimensions_preserved(
    original_image: np.ndarray, enhanced_image: np.ndarray
) -> np.ndarray:
    """Ensure an enhancement operation has preserved the input dimensions."""
    if enhanced_image.shape != original_image.shape:
        raise RuntimeError(
            "Contrast enhancement changed the image dimensions unexpectedly."
        )
    return enhanced_image


def _validate_clahe_parameters(
    clip_limit: float, tile_grid_size: tuple[int, int]
) -> tuple[float, tuple[int, int]]:
    """Validate and normalize CLAHE configuration values."""
    if isinstance(clip_limit, bool) or not isinstance(clip_limit, Real):
        raise TypeError("CLAHE clip limit must be a number.")
    if clip_limit <= 0:
        raise ValueError("CLAHE clip limit must be greater than zero.")

    if not isinstance(tile_grid_size, tuple) or len(tile_grid_size) != 2:
        raise TypeError("CLAHE tile grid size must be a tuple of two integers.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in tile_grid_size
    ):
        raise ValueError("CLAHE tile grid dimensions must be positive integers.")

    normalized_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))
    return float(clip_limit), normalized_grid_size


def apply_histogram_equalisation(image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalisation to a grayscale image."""
    _validate_grayscale_image(image)
    enhanced_image = cv2.equalizeHist(image)
    return _ensure_dimensions_preserved(image, enhanced_image)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply the recommended local contrast enhancement method."""
    _validate_grayscale_image(image)
    validated_clip_limit, validated_grid_size = _validate_clahe_parameters(
        clip_limit, tile_grid_size
    )
    clahe = cv2.createCLAHE(
        clipLimit=validated_clip_limit,
        tileGridSize=validated_grid_size,
    )
    enhanced_image = clahe.apply(image)
    return _ensure_dimensions_preserved(image, enhanced_image)


def apply_gamma_correction(
    image: np.ndarray, gamma: float = 1.2
) -> np.ndarray:
    """Apply lookup-table gamma correction to a grayscale image."""
    _validate_grayscale_image(image)
    if isinstance(gamma, bool) or not isinstance(gamma, Real):
        raise TypeError("Gamma must be a number.")
    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")

    inverse_gamma = 1.0 / float(gamma)
    lookup_table = np.array(
        [((value / 255.0) ** inverse_gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    enhanced_image = cv2.LUT(image, lookup_table)
    return _ensure_dimensions_preserved(image, enhanced_image)
