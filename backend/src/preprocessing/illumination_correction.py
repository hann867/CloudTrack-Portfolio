"""Uneven illumination correction for grayscale cloud chamber images."""

from numbers import Integral

import cv2
import numpy as np


DEFAULT_ILLUMINATION_CORRECTION_METHOD = "gaussian"
SUPPORTED_ILLUMINATION_CORRECTION_METHODS = frozenset({"gaussian", "top_hat"})


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


def _validate_kernel_size(kernel_size: int) -> int:
    """Validate and normalize an odd background-estimation kernel size."""
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, Integral):
        raise TypeError("Kernel size must be an integer.")

    normalized_size = int(kernel_size)
    if normalized_size < 3 or normalized_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer greater than or equal to 3.")

    return normalized_size


def correct_uneven_illumination(
    image: np.ndarray,
    method: str = DEFAULT_ILLUMINATION_CORRECTION_METHOD,
    kernel_size: int = 51,
) -> np.ndarray:
    """Correct uneven lighting while retaining bright particle track detail.

    ``gaussian`` estimates a slowly varying background, subtracts it, and
    restores contrast with min-max normalisation. ``top_hat`` uses a white
    morphological top-hat to retain small bright traces while suppressing
    broad background haze. Gaussian correction remains the default.
    """
    _validate_grayscale_image(image)

    if not isinstance(method, str):
        raise TypeError("Illumination correction method must be a string.")
    normalized_method = method.strip().lower()
    if normalized_method not in SUPPORTED_ILLUMINATION_CORRECTION_METHODS:
        supported = ", ".join(sorted(SUPPORTED_ILLUMINATION_CORRECTION_METHODS))
        raise ValueError(
            f"Unsupported illumination correction method '{method}'. "
            f"Supported methods are: {supported}"
        )

    validated_kernel_size = _validate_kernel_size(kernel_size)

    if normalized_method == "gaussian":
        background_illumination = cv2.GaussianBlur(
            image,
            (validated_kernel_size, validated_kernel_size),
            sigmaX=0,
        )
        foreground_detail = cv2.subtract(image, background_illumination)
        corrected_image = cv2.normalize(
            foreground_detail,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )
    else:
        top_hat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (validated_kernel_size, validated_kernel_size),
        )
        corrected_image = cv2.morphologyEx(
            image,
            cv2.MORPH_TOPHAT,
            top_hat_kernel,
        )

    if corrected_image.shape != image.shape:
        raise RuntimeError(
            "Illumination correction changed the image dimensions unexpectedly."
        )

    return corrected_image
