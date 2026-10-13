"""Interactive two-point manual calibration for CloudTrack Module 1."""

from math import hypot, isfinite
from pathlib import Path

import cv2
import numpy as np

from ..preprocessing.image_loader import load_image


_WINDOW_NAME = "CloudTrack Manual Calibration"


def _select_two_points(image: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Collect two image coordinates using OpenCV left-button clicks."""
    selected_points: list[tuple[int, int]] = []
    window_created = False

    def handle_mouse_click(event, x, y, _flags, _parameters) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(selected_points) >= 2:
            return

        selected_points.append((int(x), int(y)))
        annotated_image = image.copy()
        for point in selected_points:
            cv2.circle(annotated_image, point, 5, (0, 0, 255), thickness=-1)
        if len(selected_points) == 2:
            cv2.line(
                annotated_image,
                selected_points[0],
                selected_points[1],
                (0, 255, 0),
                thickness=2,
            )
        cv2.imshow(_WINDOW_NAME, annotated_image)

    print("Select two calibration points with the left mouse button.")
    print("Press Esc or Q to cancel.")

    try:
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        window_created = True
        cv2.setMouseCallback(_WINDOW_NAME, handle_mouse_click)
        cv2.imshow(_WINDOW_NAME, image)

        while len(selected_points) < 2:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                raise RuntimeError(
                    "Manual calibration was cancelled before two points were selected."
                )
            if (
                len(selected_points) < 2
                and cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            ):
                raise RuntimeError(
                    "Calibration window was closed before two points were selected."
                )
    finally:
        if window_created:
            try:
                cv2.destroyWindow(_WINDOW_NAME)
            except cv2.error:
                pass

    return selected_points[0], selected_points[1]


def _prompt_for_distance_mm() -> float:
    """Prompt until the user enters a finite, positive distance in millimetres."""
    while True:
        response = input(
            "Enter the real-world distance between the selected points "
            "in millimetres: "
        ).strip()
        try:
            distance_mm = float(response)
        except ValueError:
            print("Invalid distance. Enter a positive number in millimetres.")
            continue

        if not isfinite(distance_mm) or distance_mm <= 0:
            print("Invalid distance. Enter a positive number in millimetres.")
            continue

        return distance_mm


def run_manual_calibration(image_path: str | Path) -> dict[str, object]:
    """Run interactive two-point calibration and return the calculated scale."""
    original_bgr, _original_rgb, _grayscale = load_image(image_path)
    point_1, point_2 = _select_two_points(original_bgr)

    x_1, y_1 = point_1
    x_2, y_2 = point_2
    pixel_distance = hypot(x_2 - x_1, y_2 - y_1)
    if pixel_distance == 0:
        raise ValueError("The two calibration points must be different.")

    real_world_distance_mm = _prompt_for_distance_mm()
    millimetres_per_pixel = real_world_distance_mm / pixel_distance
    pixels_per_millimetre = pixel_distance / real_world_distance_mm

    return {
        "point_1": {"x": x_1, "y": y_1},
        "point_2": {"x": x_2, "y": y_2},
        "pixel_distance": pixel_distance,
        "real_world_distance_mm": real_world_distance_mm,
        "mm_per_pixel": millimetres_per_pixel,
        "pixels_per_mm": pixels_per_millimetre,
    }
