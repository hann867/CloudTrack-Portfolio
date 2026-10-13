"""Module 2 particle-track detection and segmentation pipeline.

Adapted from the standalone ``Cloud Chamber Assignment`` implementation so it
can be called by the CloudTrack API as well as from the command line.
"""

from pathlib import Path
import argparse
import sys
import warnings

import cv2
import numpy as np
import pandas as pd

from .binary_mask import adaptive_segment
from .cleaned_mask import clean_mask
from .labelled_mask import analyse_components, filter_components, link_track_fragments
from .load_preprocessed import (
    discover_images, input_intensity_record, is_nearly_binary, load_grayscale_image)
from .roi_extraction import extract_contours_and_rois

MODULE_TITLE = "Module 2: Particle Track Detection and Segmentation"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
PREPROCESSED_DIR = DEFAULT_PREPROCESSED_DIR
OUTPUT_ROOT = PROJECT_ROOT / "module2_outputs"


def _build_output_paths(root):
    root = Path(root)
    return {
        "root": root,
        "binary_masks": root / "binary_masks",
        "cleaned_masks": root / "cleaned_masks",
        "labelled_masks": root / "labelled_masks",
        "annotated_images": root / "annotated_images",
        "roi_images": root / "rois" / "images",
        "roi_masks": root / "rois" / "masks",
        "csv": root / "track_regions.csv",
    }


OUTPUT_PATHS = _build_output_paths(OUTPUT_ROOT)

CONFIG = {
    "adaptive_method": "gaussian", "adaptive_block_size": 61, "adaptive_C": -2.0,
    "blur_kernel_size": 3, "foreground_polarity": "bright", "preserve_near_binary": True,
    "kernel_shape": "ellipse", "opening_kernel_size": (3, 3),
    "closing_kernel_size": (5, 5), "opening_iterations": 1, "closing_iterations": 1,
    "connectivity": 8, "min_component_area": 100, "max_component_area": None,
    "min_bbox_long_side": 60, "min_bbox_short_side": 1, "aspect_ratio_range": None,
    "trajectory_linking_enabled": True,
    "trajectory_protect_individual_candidates": True,
    "trajectory_min_fragment_area": 3, "trajectory_max_pairwise_fragments": 2000,
    "trajectory_max_centroid_gap": 28.0, "trajectory_tangent_radius": 60.0,
    "trajectory_min_local_points": 3, "trajectory_min_local_linearity": 0.76,
    "trajectory_max_alignment_degrees": 28.0,
    "trajectory_max_tangent_change_degrees": 35.0,
    "trajectory_min_group_fragments": 4,
    "trajectory_min_group_link_length": 60.0,
    "trajectory_max_mean_link_gap": 20.0, "trajectory_max_step_ratio": 2.5,
    "curve_linking_enabled": True,
    "curve_min_fragment_area": 2, "curve_max_fragment_area": 150,
    "curve_hough_threshold": 12, "curve_hough_min_line_length": 35,
    "curve_hough_max_line_gap": 35, "curve_max_hough_lines": 160,
    "curve_min_axis_span": 180.0, "curve_max_seed_fit_rmse": 8.0,
    "curve_max_seed_tangent_error_degrees": 22.0,
    "curve_min_total_turn_degrees": 8.0,
    "curve_residual_tolerance": 14.0, "curve_context_radius": 35.0,
    "curve_max_component_gap": 65.0, "curve_min_components": 8,
    "curve_bin_size": 25.0, "curve_min_bin_coverage": 0.50,
    "curve_min_contrast": 0.58, "curve_max_candidates": 6,
    "curve_max_member_overlap": 0.55, "curve_max_bbox_overlap": 0.30,
    "curve_min_unclaimed_fraction": 0.80,
    "curve_max_protected_overlap": 0.50,
    "curve_min_score": 550.0,
    "polyline_linking_enabled": True,
    "polyline_hough_threshold": 10, "polyline_hough_min_line_length": 25,
    "polyline_hough_max_line_gap": 35, "polyline_max_hough_lines": 160,
    "polyline_min_fragment_area": 2, "polyline_max_fragment_area": 300,
    "polyline_seed_tube_radius": 11.0, "polyline_seed_extension": 15.0,
    "polyline_min_seed_components": 4, "polyline_max_step": 45.0,
    "polyline_max_turn_degrees": 28.0, "polyline_direction_history": 5,
    "polyline_min_components": 8, "polyline_min_length": 150.0,
    "polyline_residual_tolerance": 14.0, "polyline_context_radius": 35.0,
    "polyline_bin_size": 25.0, "polyline_min_bin_coverage": 0.52,
    "polyline_min_contrast": 0.55, "polyline_min_score": 550.0,
    "polyline_max_candidates": 6, "polyline_turn_penalty": 0.35,
    "polyline_max_low_contrast_turn_density": 1.50,
    "polyline_high_turn_min_contrast": 0.60,
    "extended_polyline_enabled": True,
    "extended_polyline_max_step": 55.0,
    "extended_polyline_max_turn_degrees": 40.0,
    "extended_polyline_min_member_overlap": 0.30,
    "polyline_fragment_extension_enabled": True,
    "polyline_fragment_min_components": 6,
    "polyline_fragment_min_length": 100.0,
    "polyline_fragment_min_score": 300.0,
    "polyline_fragment_min_base_contrast": 0.68,
    "polyline_fragment_max_axis_gap": 60.0,
    "polyline_fragment_max_endpoint_gap": 75.0,
    "polyline_fragment_max_join_turn_degrees": 45.0,
    "polyline_fragment_min_orthogonal_overlap": 0.65,
    "polyline_fragment_min_axis_extension": 100.0,
    "beam_polyline_enabled": True,
    "beam_polyline_hough_seeds": 30, "beam_polyline_width": 30,
    "beam_polyline_branch_factor": 4, "beam_polyline_growth_steps": 15,
    "beam_polyline_checkpoint_steps": (7, 11, 14),
    "beam_polyline_max_step": 65.0, "beam_polyline_max_turn_degrees": 55.0,
    "beam_polyline_min_length": 100.0, "beam_polyline_min_contrast": 0.55,
    "beam_polyline_min_components": 20,
    "beam_polyline_min_efficiency": 0.75, "beam_polyline_turn_scale": 1200.0,
    "beam_polyline_min_quality": 180.0,
    "beam_polyline_selection_multiplier": 2.5,
    "beam_polyline_max_candidates": 1,
    "beam_polyline_max_bbox_overlap": 0.20,
    "arc_linking_enabled": True, "arc_hough_dp": 1.5,
    "arc_hough_scale": 1.0, "arc_hough_dilation": 1,
    "arc_hough_min_distance": 60.0, "arc_hough_param1": 50.0,
    "arc_hough_param2": 15.0, "arc_min_radius": 70,
    "arc_max_radius": 250, "arc_max_hough_circles": 48,
    "arc_max_windows_per_circle": 5,
    "arc_residual_tolerance": 22.0, "arc_context_radius": 50.0,
    "arc_max_component_gap": 75.0, "arc_min_components": 12,
    "arc_max_components": 90, "arc_min_turn_degrees": 90.0,
    "arc_max_turn_degrees": 260.0, "arc_bin_size": 25.0,
    "arc_min_bin_coverage": 0.60, "arc_min_contrast": 0.60,
    "arc_min_score": 850.0, "arc_max_candidates": 5,
    "arc_prefer_compact": True,
    "large_arc_linking_enabled": True,
    "large_arc_min_radius": 140, "large_arc_max_radius": 650,
    "large_arc_max_hough_circles": 64,
    "large_arc_max_components": 220,
    "large_arc_min_turn_degrees": 55.0,
    "large_arc_max_turn_degrees": 170.0,
    "large_arc_search_min_contrast": 0.60,
    "large_arc_min_contrast": 0.65, "large_arc_min_score": 800.0,
    "large_arc_min_final_score": 1100.0,
    "large_arc_min_piecewise_score": 1500.0,
    "large_arc_max_candidates": 2,
    "arc_join_max_endpoint_gap": 90.0,
    "arc_join_min_orthogonal_overlap": 0.55,
    "arc_join_min_axis_extension": 80.0,
    "arc_join_max_orthogonal_extension_ratio": 0.25,
    "roi_padding": 8, "recursive_discovery": True, "max_foreground_percent": 10.0,
    "near_binary_tolerance": 5, "near_binary_required_fraction": 0.995,
    "near_binary_min_endpoint_fraction": 0.0001,
}

CSV_COLUMNS = [
    "source_filename", "source_path", "source_stem", "track_id", "component_id", "label_id",
    "source_component_ids", "fragment_count", "trajectory_linked", "trajectory_link_count",
    "curve_fitted", "curve_model", "curve_score", "curve_contrast", "curve_axis_span_pixels",
    "curve_total_turn_degrees",
    "trajectory_link_length_pixels", "trajectory_mean_link_gap_pixels",
    "trajectory_mean_local_linearity", "trajectory_max_link_alignment_degrees",
    "trajectory_max_tangent_change_degrees", "trajectory_step_ratio", "area_pixels",
    "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
    "bbox_long_side", "bbox_short_side", "bbox_x2_exclusive", "bbox_y2_exclusive",
    "touches_boundary", "contour_count", "roi_padding", "roi_x0", "roi_y0",
    "roi_x1_exclusive", "roi_y1_exclusive", "roi_width", "roi_height",
    "roi_clipped_to_boundary", "roi_image_file", "roi_mask_file",
]


def _is_plain_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate_kernel_size(name, value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item tuple/list.")
    if not all(_is_plain_int(v) and v >= 1 for v in value):
        raise ValueError(f"{name} values must be positive integers.")


def validate_config(config):
    """Validate configuration values before any image is processed."""
    block_size = config["adaptive_block_size"]
    if not _is_plain_int(block_size) or block_size <= 1 or block_size % 2 == 0:
        raise ValueError("adaptive_block_size must be an odd integer greater than 1.")
    if config["adaptive_method"].lower() not in {"gaussian", "mean"}:
        raise ValueError("adaptive_method must be 'gaussian' or 'mean'.")
    if config["foreground_polarity"].lower() not in {"bright", "dark"}:
        raise ValueError("foreground_polarity must be 'bright' or 'dark'.")
    if not np.isfinite(float(config["adaptive_C"])):
        raise ValueError("adaptive_C must be finite.")
    blur_size = config["blur_kernel_size"]
    if blur_size is not None and (
            not _is_plain_int(blur_size) or blur_size <= 1 or blur_size % 2 == 0):
        raise ValueError("blur_kernel_size must be None or an odd integer greater than 1.")
    if config["kernel_shape"].lower() not in {"ellipse", "rect", "cross"}:
        raise ValueError("kernel_shape must be 'ellipse', 'rect', or 'cross'.")
    _validate_kernel_size("opening_kernel_size", config["opening_kernel_size"])
    _validate_kernel_size("closing_kernel_size", config["closing_kernel_size"])
    if not _is_plain_int(config["opening_iterations"]) or config["opening_iterations"] < 0:
        raise ValueError("opening_iterations must be a non-negative integer.")
    if not _is_plain_int(config["closing_iterations"]) or config["closing_iterations"] < 1:
        raise ValueError("closing_iterations must be an integer of at least 1.")
    if config["connectivity"] not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8.")
    for name in ("min_component_area", "min_bbox_long_side", "min_bbox_short_side"):
        if not _is_plain_int(config[name]) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config["min_bbox_long_side"] < config["min_bbox_short_side"]:
        raise ValueError("min_bbox_long_side must be >= min_bbox_short_side.")
    max_area = config["max_component_area"]
    if max_area is not None and (not _is_plain_int(max_area) or max_area < config["min_component_area"]):
        raise ValueError("max_component_area must be None or >= min_component_area.")
    ratio_range = config["aspect_ratio_range"]
    if ratio_range is not None:
        if not isinstance(ratio_range, (tuple, list)) or len(ratio_range) != 2:
            raise ValueError("aspect_ratio_range must be None or a two-item pair.")
        ratio_min, ratio_max = map(float, ratio_range)
        if ratio_min <= 0 or ratio_max < ratio_min:
            raise ValueError("aspect_ratio_range must be positive and ordered.")
    for name in ("trajectory_linking_enabled", "trajectory_protect_individual_candidates"):
        if not isinstance(config[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be Boolean.")
    if not isinstance(config["curve_linking_enabled"], (bool, np.bool_)):
        raise ValueError("curve_linking_enabled must be Boolean.")
    if not isinstance(config["polyline_linking_enabled"], (bool, np.bool_)):
        raise ValueError("polyline_linking_enabled must be Boolean.")
    if not isinstance(config["beam_polyline_enabled"], (bool, np.bool_)):
        raise ValueError("beam_polyline_enabled must be Boolean.")
    if not isinstance(config["extended_polyline_enabled"], (bool, np.bool_)):
        raise ValueError("extended_polyline_enabled must be Boolean.")
    if not isinstance(config["polyline_fragment_extension_enabled"], (bool, np.bool_)):
        raise ValueError("polyline_fragment_extension_enabled must be Boolean.")
    if not isinstance(config["arc_linking_enabled"], (bool, np.bool_)):
        raise ValueError("arc_linking_enabled must be Boolean.")
    if not isinstance(config["arc_prefer_compact"], (bool, np.bool_)):
        raise ValueError("arc_prefer_compact must be Boolean.")
    if not isinstance(config["large_arc_linking_enabled"], (bool, np.bool_)):
        raise ValueError("large_arc_linking_enabled must be Boolean.")
    for name in ("trajectory_min_fragment_area", "trajectory_max_pairwise_fragments",
                 "trajectory_min_local_points", "trajectory_min_group_fragments"):
        if not _is_plain_int(config[name]) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config["trajectory_min_local_points"] < 3:
        raise ValueError("trajectory_min_local_points must be at least 3.")
    if config["trajectory_min_group_fragments"] < config["trajectory_min_local_points"]:
        raise ValueError("trajectory_min_group_fragments must be >= trajectory_min_local_points.")
    for name in ("trajectory_max_centroid_gap", "trajectory_tangent_radius",
                 "trajectory_min_group_link_length", "trajectory_max_mean_link_gap"):
        if not np.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    if config["trajectory_tangent_radius"] < config["trajectory_max_centroid_gap"]:
        raise ValueError("trajectory_tangent_radius must be >= trajectory_max_centroid_gap.")
    if config["trajectory_max_mean_link_gap"] > config["trajectory_max_centroid_gap"]:
        raise ValueError("trajectory_max_mean_link_gap must be <= trajectory_max_centroid_gap.")
    if not 0.0 <= float(config["trajectory_min_local_linearity"]) <= 1.0:
        raise ValueError("trajectory_min_local_linearity must be in [0, 1].")
    for name in ("trajectory_max_alignment_degrees", "trajectory_max_tangent_change_degrees"):
        if not 0.0 < float(config[name]) <= 90.0:
            raise ValueError(f"{name} must be in (0, 90].")
    if not np.isfinite(float(config["trajectory_max_step_ratio"])) or float(config["trajectory_max_step_ratio"]) < 1:
        raise ValueError("trajectory_max_step_ratio must be finite and >= 1.")
    for name in ("curve_min_fragment_area", "curve_max_fragment_area", "curve_hough_threshold",
                 "curve_hough_min_line_length", "curve_hough_max_line_gap",
                 "curve_max_hough_lines", "curve_min_components", "curve_max_candidates"):
        if not _is_plain_int(config[name]) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("polyline_hough_threshold", "polyline_hough_min_line_length",
                 "polyline_hough_max_line_gap", "polyline_max_hough_lines",
                 "polyline_min_fragment_area", "polyline_max_fragment_area",
                 "polyline_min_seed_components", "polyline_direction_history",
                 "polyline_min_components", "polyline_max_candidates",
                 "polyline_fragment_min_components",
                 "beam_polyline_hough_seeds", "beam_polyline_width",
                 "beam_polyline_branch_factor", "beam_polyline_growth_steps",
                 "beam_polyline_min_components", "beam_polyline_max_candidates"):
        if not _is_plain_int(config[name]) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    checkpoints = config["beam_polyline_checkpoint_steps"]
    if (not isinstance(checkpoints, (tuple, list)) or not checkpoints
            or any(not _is_plain_int(step) or step < 0 for step in checkpoints)):
        raise ValueError(
            "beam_polyline_checkpoint_steps must contain non-negative integers.")
    if max(checkpoints) >= config["beam_polyline_growth_steps"]:
        raise ValueError(
            "beam_polyline_checkpoint_steps must be below beam_polyline_growth_steps.")
    for name in ("arc_min_radius", "arc_max_radius", "arc_max_hough_circles",
                 "arc_max_windows_per_circle",
                 "arc_hough_dilation",
                 "arc_min_components", "arc_max_components", "arc_max_candidates"):
        if not _is_plain_int(config[name]) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config["arc_max_radius"] < config["arc_min_radius"]:
        raise ValueError("arc_max_radius must be >= arc_min_radius.")
    if config["arc_max_components"] < config["arc_min_components"]:
        raise ValueError("arc_max_components must be >= arc_min_components.")
    for name in ("large_arc_min_radius", "large_arc_max_radius",
                 "large_arc_max_hough_circles", "large_arc_max_components",
                 "large_arc_max_candidates"):
        if not _is_plain_int(config[name]) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config["large_arc_max_radius"] < config["large_arc_min_radius"]:
        raise ValueError("large_arc_max_radius must be >= large_arc_min_radius.")
    if config["large_arc_max_components"] < config["arc_min_components"]:
        raise ValueError("large_arc_max_components must be >= arc_min_components.")
    if float(config["arc_hough_scale"]) > 1.0:
        raise ValueError("arc_hough_scale must be <= 1.0.")
    if config["curve_max_fragment_area"] < config["curve_min_fragment_area"]:
        raise ValueError("curve_max_fragment_area must be >= curve_min_fragment_area.")
    for name in ("curve_min_axis_span", "curve_max_seed_fit_rmse",
                 "curve_max_seed_tangent_error_degrees", "curve_min_total_turn_degrees",
                 "curve_residual_tolerance",
                 "curve_context_radius", "curve_max_component_gap", "curve_bin_size",
                 "curve_min_score"):
        if not np.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    for name in ("extended_polyline_max_step", "extended_polyline_max_turn_degrees",
                 "polyline_fragment_min_length", "polyline_fragment_min_score",
                 "polyline_fragment_max_axis_gap",
                 "polyline_fragment_max_endpoint_gap",
                 "polyline_fragment_max_join_turn_degrees",
                 "polyline_fragment_min_axis_extension",
                 "polyline_max_low_contrast_turn_density",
                 "beam_polyline_max_step", "beam_polyline_max_turn_degrees",
                 "beam_polyline_min_length", "beam_polyline_turn_scale",
                 "beam_polyline_min_quality", "beam_polyline_selection_multiplier"):
        if not np.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    if not np.isfinite(float(config["polyline_turn_penalty"])) or float(
            config["polyline_turn_penalty"]) < 0:
        raise ValueError("polyline_turn_penalty must be a non-negative finite number.")
    if not 0.0 < float(config["beam_polyline_max_turn_degrees"]) <= 90.0:
        raise ValueError("beam_polyline_max_turn_degrees must be in (0, 90].")
    if not 0.0 < float(config["extended_polyline_max_turn_degrees"]) <= 90.0:
        raise ValueError("extended_polyline_max_turn_degrees must be in (0, 90].")
    if not 0.0 < float(config["polyline_fragment_max_join_turn_degrees"]) <= 90.0:
        raise ValueError(
            "polyline_fragment_max_join_turn_degrees must be in (0, 90].")
    for name in ("arc_hough_dp", "arc_hough_scale", "arc_hough_min_distance", "arc_hough_param1",
                 "arc_hough_param2", "arc_residual_tolerance", "arc_context_radius",
                 "arc_max_component_gap", "arc_min_turn_degrees", "arc_max_turn_degrees",
                 "arc_bin_size", "arc_min_score", "arc_join_max_endpoint_gap",
                 "arc_join_min_axis_extension"):
        if not np.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    if config["arc_max_turn_degrees"] <= config["arc_min_turn_degrees"]:
        raise ValueError("arc_max_turn_degrees must be greater than arc_min_turn_degrees.")
    for name in ("large_arc_min_turn_degrees", "large_arc_max_turn_degrees",
                 "large_arc_min_score", "large_arc_min_final_score",
                 "large_arc_min_piecewise_score"):
        if not np.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    if config["large_arc_max_turn_degrees"] <= config["large_arc_min_turn_degrees"]:
        raise ValueError(
            "large_arc_max_turn_degrees must be greater than large_arc_min_turn_degrees.")
    for name in ("curve_min_bin_coverage", "curve_min_contrast", "curve_max_member_overlap",
                 "curve_max_bbox_overlap", "curve_min_unclaimed_fraction",
                 "curve_max_protected_overlap"):
        if not 0.0 < float(config[name]) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1].")
    for name in ("extended_polyline_min_member_overlap",
                 "polyline_fragment_min_base_contrast",
                 "polyline_fragment_min_orthogonal_overlap",
                 "polyline_high_turn_min_contrast",
                 "beam_polyline_min_contrast", "beam_polyline_min_efficiency",
                 "beam_polyline_max_bbox_overlap"):
        if not 0.0 < float(config[name]) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1].")
    for name in ("arc_min_bin_coverage", "arc_min_contrast",
                 "arc_join_min_orthogonal_overlap",
                 "arc_join_max_orthogonal_extension_ratio"):
        if not 0.0 < float(config[name]) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1].")
    for name in ("large_arc_search_min_contrast", "large_arc_min_contrast"):
        if not 0.0 < float(config[name]) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1].")
    if not _is_plain_int(config["roi_padding"]) or config["roi_padding"] < 0:
        raise ValueError("roi_padding must be a non-negative integer.")
    if not 0.0 < float(config["max_foreground_percent"]) <= 100.0:
        raise ValueError("max_foreground_percent must be in (0, 100].")


def _foreground_percent(mask):
    return 100.0 * np.count_nonzero(mask) / mask.size


def process_image(path, config):
    """Run the complete Module 2 pipeline for one image."""
    path = Path(path)
    gray, source_bgr, metadata = load_grayscale_image(path)
    binary_input = is_nearly_binary(
        gray, config["near_binary_tolerance"], config["near_binary_required_fraction"],
        config["near_binary_min_endpoint_fraction"])
    binary = adaptive_segment(gray, config)
    opened, closed = clean_mask(binary, config)
    raw_labels, raw_components = analyse_components(closed, path.name, config["connectivity"])
    grouped_labels, components, links, link_diagnostics = link_track_fragments(
        raw_labels, raw_components, config)
    decisions, accepted, accepted_labels, filtered_mask = filter_components(
        grouped_labels, components, config)
    annotated, rois = extract_contours_and_rois(
        source_bgr, accepted_labels, accepted, path, config["roi_padding"])
    if len(rois) != len(accepted):
        raise RuntimeError(
            f"ROI extraction returned {len(rois)} regions for {len(accepted)} accepted "
            f"components in {path.name}.")
    return {
        "path": path, "metadata": metadata,
        "segmentation_mode": (
            "binary-preserving threshold"
            if binary_input and config["preserve_near_binary"] else "adaptive threshold"),
        "gray": gray, "source_bgr": source_bgr, "binary": binary, "opened": opened,
        "closed": closed, "raw_labels": raw_labels, "raw_components": raw_components,
        "grouped_labels": grouped_labels, "components": components,
        "trajectory_links": links, "link_diagnostics": link_diagnostics,
        "decisions": decisions, "accepted": accepted, "accepted_labels": accepted_labels,
        "filtered_mask": filtered_mask, "annotated": annotated, "rois": rois,
        "raw_component_count": len(raw_components), "trajectory_group_count": len(components),
        "linked_group_count": link_diagnostics["linked_groups"],
        "trajectory_link_count": link_diagnostics["accepted_mutual_links"],
        "accepted_component_count": len(accepted),
        "adaptive_foreground_percent": _foreground_percent(binary),
        "opened_foreground_percent": _foreground_percent(opened),
        "closed_foreground_percent": _foreground_percent(closed),
        "final_foreground_percent": _foreground_percent(filtered_mask),
    }


def save_png(path, image):
    """Encode and save a checked PNG with Windows-safe path handling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise IOError(f"OpenCV could not encode PNG output: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise IOError(f"Could not save PNG output {path}: {exc}") from exc
    if not path.exists() or path.stat().st_size == 0:
        raise IOError(f"PNG output was not created correctly: {path}")


def _confirm_module2_folder(folder):
    root, resolved = OUTPUT_ROOT.resolve(), Path(folder).resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing cleanup outside Module 2 output root: {resolved}")


def ensure_output_directories():
    for key, path in OUTPUT_PATHS.items():
        if key != "csv":
            path.mkdir(parents=True, exist_ok=True)


def clear_generated_outputs():
    """Remove only known Module 2 files, leaving directories and unrelated files intact."""
    suffixes = {"binary_masks": "_binary.png", "cleaned_masks": "_cleaned.png",
                "labelled_masks": "_labelled.png", "annotated_images": "_annotated.png"}
    for key, suffix in suffixes.items():
        folder = OUTPUT_PATHS[key]
        _confirm_module2_folder(folder)
        for candidate in folder.iterdir():
            if candidate.is_file() and candidate.name.endswith(suffix):
                candidate.unlink()
    for key in ("roi_images", "roi_masks"):
        folder = OUTPUT_PATHS[key]
        _confirm_module2_folder(folder)
        for candidate in folder.iterdir():
            if candidate.is_file() and candidate.suffix.lower() == ".png" and "_track_" in candidate.name:
                candidate.unlink()
    _confirm_module2_folder(OUTPUT_PATHS["csv"])
    if OUTPUT_PATHS["csv"].is_file():
        OUTPUT_PATHS["csv"].unlink()


def remove_outputs_for_stem(source_stem):
    """Remove partial outputs for a source after a failed process/save operation."""
    exact = [
        OUTPUT_PATHS["binary_masks"] / f"{source_stem}_binary.png",
        OUTPUT_PATHS["cleaned_masks"] / f"{source_stem}_cleaned.png",
        OUTPUT_PATHS["labelled_masks"] / f"{source_stem}_labelled.png",
        OUTPUT_PATHS["annotated_images"] / f"{source_stem}_annotated.png",
    ]
    for candidate in exact:
        _confirm_module2_folder(candidate)
        if candidate.is_file():
            candidate.unlink()
    prefix = f"{source_stem}_track_"
    for key in ("roi_images", "roi_masks"):
        _confirm_module2_folder(OUTPUT_PATHS[key])
        for candidate in OUTPUT_PATHS[key].iterdir():
            if candidate.is_file() and candidate.name.startswith(prefix):
                candidate.unlink()


def save_image_result(result):
    """Save masks, uint16 labels, annotation, and non-empty ROI pairs."""
    stem = result["path"].stem
    save_png(OUTPUT_PATHS["binary_masks"] / f"{stem}_binary.png", result["binary"])
    save_png(OUTPUT_PATHS["cleaned_masks"] / f"{stem}_cleaned.png", result["filtered_mask"])
    maximum_label = int(result["accepted_labels"].max()) if result["accepted_labels"].size else 0
    if maximum_label > np.iinfo(np.uint16).max:
        raise OverflowError(f"{stem} has {maximum_label} labels, exceeding the uint16 PNG limit.")
    save_png(OUTPUT_PATHS["labelled_masks"] / f"{stem}_labelled.png",
             result["accepted_labels"].astype(np.uint16))
    save_png(OUTPUT_PATHS["annotated_images"] / f"{stem}_annotated.png", result["annotated"])
    csv_records = []
    for record in result["rois"]:
        if record["_image_roi"].size == 0 or record["_mask_roi"].size == 0 or not np.any(record["_mask_roi"]):
            warnings.warn(f"Skipping empty ROI output: {record['track_id']}")
            continue
        save_png(OUTPUT_ROOT / record["roi_image_file"], record["_image_roi"])
        save_png(OUTPUT_ROOT / record["roi_mask_file"], record["_mask_roi"])
        csv_records.append({column: record[column] for column in CSV_COLUMNS})
    return csv_records


def snapshot_sources(paths):
    return {str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def validate_outputs(processed_names, region_table, expected_regions):
    missing = []
    for name in processed_names:
        stem = Path(name).stem
        for path in (
            OUTPUT_PATHS["binary_masks"] / f"{stem}_binary.png",
            OUTPUT_PATHS["cleaned_masks"] / f"{stem}_cleaned.png",
            OUTPUT_PATHS["labelled_masks"] / f"{stem}_labelled.png",
            OUTPUT_PATHS["annotated_images"] / f"{stem}_annotated.png",
        ):
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing or not OUTPUT_PATHS["csv"].is_file():
        raise RuntimeError(f"Final output validation failed; missing outputs: {missing}")
    if len(region_table) != expected_regions:
        raise RuntimeError("CSV row count does not match accepted-component total.")
    if not region_table.empty and not region_table["track_id"].is_unique:
        raise RuntimeError("Track IDs are not unique.")
    for row in region_table.itertuples(index=False):
        if not (OUTPUT_ROOT / row.roi_image_file).is_file() or not (OUTPUT_ROOT / row.roi_mask_file).is_file():
            raise RuntimeError(f"Missing ROI pair for {row.track_id}.")


def _parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect and segment particle tracks in preprocessed cloud-chamber images.")
    parser.add_argument("--input", type=Path, default=PREPROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv=None):
    global PREPROCESSED_DIR, OUTPUT_ROOT, OUTPUT_PATHS
    arguments = _parse_arguments(argv)
    PREPROCESSED_DIR = arguments.input.expanduser().resolve()
    OUTPUT_ROOT = arguments.output.expanduser().resolve()
    OUTPUT_PATHS = _build_output_paths(OUTPUT_ROOT)
    print(MODULE_TITLE)
    print("Scope: segmentation and ROI generation only; classification is reserved for a later module.\n")
    validate_config(CONFIG)
    if not PREPROCESSED_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {PREPROCESSED_DIR}")
    if not PREPROCESSED_DIR.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {PREPROCESSED_DIR}")
    try:
        OUTPUT_ROOT.resolve().relative_to(PREPROCESSED_DIR.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("The Module 2 output folder must not be inside the input folder.")
    ensure_output_directories()
    image_paths = discover_images(PREPROCESSED_DIR, CONFIG["recursive_discovery"])
    if not image_paths:
        raise FileNotFoundError(f"No supported PNG/JPEG images were found in {PREPROCESSED_DIR}")
    source_snapshot = snapshot_sources(image_paths)
    print(f"Input:\n{PREPROCESSED_DIR}\n\nFound {len(image_paths)} supported images.\n")
    for path in image_paths:
        record = input_intensity_record(path, CONFIG)
        inferred = record["inferred_foreground_polarity"]
        if inferred != "undetermined" and inferred != CONFIG["foreground_polarity"]:
            warnings.warn(f"Configured foreground polarity conflicts with diagnostic for {path.name}.")
    clear_generated_outputs()
    all_records, summaries, failures, zero_candidates = [], [], [], []
    for number, image_path in enumerate(image_paths, start=1):
        print(f"[{number}/{len(image_paths)}] {image_path.name}")
        try:
            result = process_image(image_path, CONFIG)
            saved_records = save_image_result(result)
        except Exception as exc:
            remove_outputs_for_stem(image_path.stem)
            failures.append({"source_filename": image_path.name, "error": str(exc)})
            warnings.warn(f"Skipping {image_path.name}: {exc}")
            print(f"  Failed: {exc}\n")
            continue
        all_records.extend(saved_records)
        if result["accepted_component_count"] == 0:
            zero_candidates.append(image_path.name)
            warnings.warn(f"No accepted track candidates in {image_path.name}")
        summaries.append({
            "source_filename": image_path.name,
            "raw_components": result["raw_component_count"],
            "trajectory_groups": result["trajectory_group_count"],
            "linked_groups": result["linked_group_count"],
            "trajectory_links": result["trajectory_link_count"],
            "accepted_components": result["accepted_component_count"],
            "adaptive_foreground_percent": result["adaptive_foreground_percent"],
            "opened_foreground_percent": result["opened_foreground_percent"],
            "closed_foreground_percent": result["closed_foreground_percent"],
            "final_foreground_percent": result["final_foreground_percent"],
        })
        print(f"  Raw components: {result['raw_component_count']}")
        print(f"  Trajectory groups: {result['trajectory_group_count']}")
        print(f"  Links: {result['trajectory_link_count']}")
        print(f"  Accepted candidates: {result['accepted_component_count']}")
        print(f"  Final foreground: {result['final_foreground_percent']:.4f}%\n")
    region_table = pd.DataFrame(all_records, columns=CSV_COLUMNS)
    if not region_table.empty:
        region_table = region_table.sort_values(["source_filename", "label_id"], kind="stable").reset_index(drop=True)
    region_table.to_csv(OUTPUT_PATHS["csv"], index=False, float_format="%.4f")
    expected_regions = sum(row["accepted_components"] for row in summaries)
    validate_outputs([row["source_filename"] for row in summaries], region_table, expected_regions)
    source_unchanged = snapshot_sources(image_paths) == source_snapshot
    if not source_unchanged:
        raise RuntimeError("One or more Module 1 input files changed during Module 2 processing.")
    print("Module 2 completed.\n")
    print(f"Images discovered: {len(image_paths)}")
    print(f"Images processed: {len(summaries)}")
    print(f"Images skipped/failed: {len(failures)}")
    print(f"Images with no accepted candidates: {len(zero_candidates)}")
    print(f"Accepted track-candidate regions: {len(region_table)}")
    print(f"Source images unchanged: {source_unchanged}")
    print(f"\nCSV:\n{OUTPUT_PATHS['csv']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Fatal Module 2 error: {exc}", file=sys.stderr)
        sys.exit(2)
