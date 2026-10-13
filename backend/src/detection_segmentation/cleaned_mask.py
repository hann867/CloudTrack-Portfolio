"""Morphological opening and closing stages."""

import cv2


def make_morphology_kernel(shape, size):
    """Create an OpenCV morphology kernel from a named shape and size."""
    shape_map = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
    return cv2.getStructuringElement(shape_map[shape.lower()], tuple(map(int, size)))


def morphological_open(binary_mask, config):
    """Apply opening, or return an unchanged copy when iterations is zero."""
    if config["opening_iterations"] == 0:
        return binary_mask.copy()
    kernel = make_morphology_kernel(config["kernel_shape"], config["opening_kernel_size"])
    return cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel, iterations=config["opening_iterations"])


def clean_mask(binary_mask, config):
    """Return the mask after opening and then after closing."""
    opened = morphological_open(binary_mask, config)
    kernel = make_morphology_kernel(config["kernel_shape"], config["closing_kernel_size"])
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=config["closing_iterations"])
    return opened, closed

