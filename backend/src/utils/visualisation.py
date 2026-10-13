"""Matplotlib display and comparison utilities for CloudTrack Module 1."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .path_utils import OUTPUT_IMAGE_DIR


def _validate_image(image: np.ndarray, expected_channels: int | None = None) -> None:
    """Validate an image before displaying or saving it."""
    if image is None:
        raise ValueError("Image must not be empty or None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Image must not be empty.")

    if expected_channels is None:
        if image.ndim == 2:
            return
        if image.ndim == 3 and image.shape[2] == 3:
            return
        raise ValueError("Image must be a grayscale or three-channel RGB image.")

    if expected_channels == 1 and image.ndim != 2:
        raise ValueError("Grayscale image must be a two-dimensional array.")
    if expected_channels == 3 and not (
        image.ndim == 3 and image.shape[2] == 3
    ):
        raise ValueError("RGB image must have shape (height, width, 3).")


def _render_image(axis, image: np.ndarray, title: str) -> None:
    """Render a validated grayscale or RGB image on a Matplotlib axis."""
    if image.ndim == 2:
        axis.imshow(image, cmap="gray")
    else:
        axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")


def display_grayscale_image(
    image: np.ndarray, title: str = "Grayscale Image"
) -> None:
    """Display one grayscale image in a notebook or Python script."""
    _validate_image(image, expected_channels=1)
    figure, axis = plt.subplots(figsize=(8, 6))
    _render_image(axis, image, title)
    figure.tight_layout()
    plt.show()


def display_rgb_image(image: np.ndarray, title: str = "RGB Image") -> None:
    """Display one RGB image in a notebook or Python script."""
    _validate_image(image, expected_channels=3)
    figure, axis = plt.subplots(figsize=(8, 6))
    _render_image(axis, image, title)
    figure.tight_layout()
    plt.show()


def display_images_side_by_side(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Display two or more grayscale or RGB images in one horizontal row."""
    image_list = list(images)
    if len(image_list) < 2:
        raise ValueError("At least two images are required for side-by-side display.")
    for image in image_list:
        _validate_image(image)

    if titles is None:
        title_list = [f"Image {index}" for index in range(1, len(image_list) + 1)]
    else:
        title_list = list(titles)
        if len(title_list) != len(image_list):
            raise ValueError("The number of titles must match the number of images.")

    resolved_figsize = figsize or (5 * len(image_list), 5)
    figure, axes = plt.subplots(
        1,
        len(image_list),
        figsize=resolved_figsize,
        squeeze=False,
    )
    for axis, image, title in zip(axes[0], image_list, title_list):
        _render_image(axis, image, title)

    figure.tight_layout()
    plt.show()


def save_before_after_comparison(
    before_image: np.ndarray,
    after_image: np.ndarray,
    filename: str | Path = "before_after_comparison.png",
    before_title: str = "Before",
    after_title: str = "After",
) -> Path:
    """Save a two-panel comparison image in the configured output directory."""
    _validate_image(before_image)
    _validate_image(after_image)

    output_name = Path(filename).name
    if not output_name:
        raise ValueError("Comparison filename must not be empty.")
    if not Path(output_name).suffix:
        output_name = f"{output_name}.png"

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_IMAGE_DIR / output_name

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)
    _render_image(axes[0][0], before_image, before_title)
    _render_image(axes[0][1], after_image, after_title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_path
