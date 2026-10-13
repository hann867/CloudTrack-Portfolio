"""Image loading utilities for CloudTrack Module 1."""

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_FORMATS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)


def validate_image_path(image_path: str | Path) -> Path:
    """Validate that an image path exists and points to a file."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Image path does not point to a file: {path}")

    return path


def is_supported_image_format(image_path: str | Path) -> bool:
    """Return whether an image path has a supported file extension."""
    return Path(image_path).suffix.lower() in SUPPORTED_IMAGE_FORMATS


def convert_bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB for display."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def convert_bgr_to_grayscale(bgr_image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to grayscale for processing."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


def load_image(image_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an image and return its BGR, RGB, and grayscale representations."""
    path = validate_image_path(image_path)

    if not is_supported_image_format(path):
        extension = path.suffix.lower() or "<no extension>"
        supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise ValueError(
            f"Unsupported image format '{extension}' for file: {path}. "
            f"Supported formats are: {supported}"
        )

    bgr_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(
            f"Unable to read image file: {path}. "
            "The file may be corrupted or contain invalid image data."
        )

    rgb_image = convert_bgr_to_rgb(bgr_image)
    grayscale_image = convert_bgr_to_grayscale(bgr_image)

    return bgr_image, rgb_image, grayscale_image
