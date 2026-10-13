"""Geometric track analysis and optional particle classification."""

from .annotation_roi import crop_inside_rectangle, detect_red_rectangle
from .feature_extractor import (
    analyse_annotated_image,
    analyse_track_mask,
    extract_geometric_features,
    process_annotated_dataset,
)
from .module2_adapter import adapt_module2_result
from .module2_bridge_diagnostics import export_module2_bridge_diagnostics
from .model_contract import AUTHORIZED_CLASS_NAMES, AUTHORIZED_FEATURE_NAMES
from .production_inference import classify_module2_mask, classify_module2_masks

__all__ = [
    "analyse_annotated_image",
    "analyse_track_mask",
    "AUTHORIZED_CLASS_NAMES",
    "AUTHORIZED_FEATURE_NAMES",
    "adapt_module2_result",
    "crop_inside_rectangle",
    "classify_module2_mask",
    "classify_module2_masks",
    "detect_red_rectangle",
    "extract_geometric_features",
    "export_module2_bridge_diagnostics",
    "process_annotated_dataset",
]
