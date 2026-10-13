"""Module-3-side adapter for already-segmented production Module 2 ROIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .feature_extractor import extract_geometric_features


MODULE2_PROVENANCE_FIELDS = (
    "source_path",
    "source_filename",
    "source_stem",
    "component_id",
    "label_id",
    "source_component_ids",
    "fragment_count",
    "trajectory_linked",
    "trajectory_link_count",
    "curve_fitted",
    "curve_model",
    "curve_score",
    "curve_contrast",
    "curve_axis_span_pixels",
    "curve_total_turn_degrees",
    "trajectory_link_length_pixels",
    "trajectory_mean_link_gap_pixels",
    "trajectory_mean_local_linearity",
    "trajectory_max_link_alignment_degrees",
    "trajectory_max_tangent_change_degrees",
    "trajectory_step_ratio",
    "area_pixels",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "bbox_long_side",
    "bbox_short_side",
    "bbox_x2_exclusive",
    "bbox_y2_exclusive",
    "touches_boundary",
    "contour_count",
    "roi_padding",
    "roi_x0",
    "roi_y0",
    "roi_x1_exclusive",
    "roi_y1_exclusive",
    "roi_width",
    "roi_height",
    "roi_clipped_to_boundary",
    "roi_image_file",
    "roi_mask_file",
)

GEOMETRIC_FEATURE_FIELDS = (
    "track_length_px",
    "track_length_mm",
    "mean_width_px",
    "median_width_px",
    "max_width_px",
    "mean_curvature",
    "max_curvature",
    "tortuosity",
    "endpoint_distance_px",
    "mask_area_px",
    "skeleton_pixels",
)


def _failure(track_id: Any, provenance: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "processing_status": "failed",
        "error_message": message,
        "track_id": track_id,
        "module2": provenance,
        "features": {},
    }


def _provenance(roi: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only non-pixel Module 2 provenance that is actually present."""

    return {field: roi[field] for field in MODULE2_PROVENANCE_FIELDS if field in roi}


def _calibration(
    module2_result: Mapping[str, Any], roi: Mapping[str, Any]
) -> tuple[float | None, str | None]:
    """Return an upstream physical calibration, without inventing one."""

    candidates = []
    if "mm_per_pixel" in roi:
        candidates.append(roi["mm_per_pixel"])
    if "mm_per_pixel" in module2_result:
        candidates.append(module2_result["mm_per_pixel"])
    metadata = module2_result.get("metadata")
    if isinstance(metadata, Mapping) and "mm_per_pixel" in metadata:
        candidates.append(metadata["mm_per_pixel"])
    if not candidates:
        return None, None
    try:
        value = float(candidates[0])
    except (TypeError, ValueError):
        return None, "Upstream mm_per_pixel must be a positive finite number."
    if not np.isfinite(value) or value <= 0:
        return None, "Upstream mm_per_pixel must be a positive finite number."
    return value, None


def adapt_module2_result(
    module2_result: Mapping[str, Any],
    *,
    geometric_extractor: Callable[..., Mapping[str, Any]] = extract_geometric_features,
) -> list[dict[str, Any]]:
    """Extract Module 3 geometry from every existing Module 2 ROI mask.

    Each ROI produces exactly one success or failure record. The adapter never
    invokes Module 3 temporary segmentation and never mutates Module 2 data.
    """

    if not isinstance(module2_result, Mapping):
        raise TypeError("Module 2 result must be a mapping.")
    rois = module2_result.get("rois")
    if not isinstance(rois, (list, tuple)):
        raise ValueError("Module 2 result must contain a list or tuple in 'rois'.")

    records: list[dict[str, Any]] = []
    for index, roi in enumerate(rois, start=1):
        if not isinstance(roi, Mapping):
            records.append(_failure(None, {}, f"ROI {index} must be a mapping."))
            continue
        track_id = roi.get("track_id")
        provenance = _provenance(roi)
        mm_per_pixel, calibration_error = _calibration(module2_result, roi)
        if mm_per_pixel is not None:
            provenance["mm_per_pixel"] = mm_per_pixel
        if calibration_error:
            records.append(_failure(track_id, provenance, calibration_error))
            continue
        if "_mask_roi" not in roi or roi["_mask_roi"] is None:
            records.append(_failure(track_id, provenance, "Module 2 ROI mask '_mask_roi' is missing."))
            continue
        try:
            mask = np.asarray(roi["_mask_roi"])
        except (TypeError, ValueError) as exc:
            records.append(_failure(track_id, provenance, f"Module 2 ROI mask is not array-like: {exc}"))
            continue
        if mask.ndim != 2:
            records.append(
                _failure(track_id, provenance, f"Module 2 ROI mask must be 2-D; received shape {mask.shape}.")
            )
            continue
        if mask.size == 0:
            records.append(_failure(track_id, provenance, "Module 2 ROI mask is empty."))
            continue
        try:
            has_foreground = bool(np.any(mask > 0))
        except (TypeError, ValueError) as exc:
            records.append(_failure(track_id, provenance, f"Module 2 ROI mask values are invalid: {exc}"))
            continue
        if not has_foreground:
            records.append(_failure(track_id, provenance, "Module 2 ROI mask contains no foreground."))
            continue
        try:
            geometry = geometric_extractor(mask.copy(), mm_per_pixel=mm_per_pixel)
        except Exception as exc:
            records.append(
                _failure(track_id, provenance, f"Module 3 geometric extraction raised {type(exc).__name__}: {exc}")
            )
            continue
        if not isinstance(geometry, Mapping):
            records.append(_failure(track_id, provenance, "Module 3 geometric extraction returned no result mapping."))
            continue
        if geometry.get("processing_status") != "success":
            message = str(geometry.get("error_message") or "Module 3 geometric extraction failed.")
            records.append(_failure(track_id, provenance, message))
            continue
        records.append(
            {
                "processing_status": "success",
                "error_message": None,
                "track_id": track_id,
                "module2": provenance,
                "features": {
                    field: geometry[field]
                    for field in GEOMETRIC_FEATURE_FIELDS
                    if field in geometry
                },
            }
        )
    return records
