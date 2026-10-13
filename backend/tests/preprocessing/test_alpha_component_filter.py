import cv2
import numpy as np

from backend.src.preprocessing.preprocessing_pipeline import (
    CLEAN_PIPELINE_CONFIG,
    _filter_alpha_proton_components,
)


def _filter(binary: np.ndarray, **overrides: object) -> np.ndarray:
    height, width = binary.shape
    roi_mask = np.full_like(binary, 255)
    roi_details = {
        "x1": 0,
        "x2": width,
        "y1": 0,
        "y2": height,
        "width": width,
        "height": height,
    }
    config = {
        **CLEAN_PIPELINE_CONFIG["alpha_proton"]["component_filter"],
        **overrides,
    }
    return _filter_alpha_proton_components(
        binary,
        roi_mask,
        roi_details,
        binary.shape,
        config,
    )


def test_default_filter_preserves_compact_valid_alpha_candidate() -> None:
    binary = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(binary, (100, 100), 8, 255, thickness=-1)

    filtered = _filter(binary)

    assert CLEAN_PIPELINE_CONFIG["alpha_proton"]["component_filter"][
        "min_aspect_ratio"
    ] is None
    assert np.count_nonzero(filtered) == np.count_nonzero(binary)


def test_optional_legacy_aspect_filter_can_still_be_enabled() -> None:
    binary = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(binary, (100, 100), 8, 255, thickness=-1)

    filtered = _filter(binary, min_aspect_ratio=2.5)

    assert np.count_nonzero(filtered) == 0


def test_area_and_image_border_protections_remain_active() -> None:
    binary = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(binary, (0, 80), (20, 89), 255, thickness=-1)
    cv2.rectangle(binary, (80, 80), (84, 89), 255, thickness=-1)

    filtered = _filter(binary)

    assert np.count_nonzero(filtered) == 0
