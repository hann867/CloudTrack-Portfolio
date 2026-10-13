"""Manual known-source track dataset preparation utilities.

This package deliberately does not infer particle classes from image geometry.
Labels are supplied explicitly by the annotator.
"""

from .crop_utils import (
    RotatedROI,
    crop_original_pixels,
    extract_rotated_crop,
    map_display_roi_to_original,
    validate_roi,
)
from .dataset_metadata import LABEL_COLUMNS, create_event_id

__all__ = [
    "LABEL_COLUMNS",
    "RotatedROI",
    "create_event_id",
    "crop_original_pixels",
    "extract_rotated_crop",
    "map_display_roi_to_original",
    "validate_roi",
]
