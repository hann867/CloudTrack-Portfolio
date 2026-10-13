"""Per-label contour detection and padded ROI extraction."""

from pathlib import Path
import warnings

import cv2
import numpy as np

from .annotated_image import annotate_components


def detect_component_contours(accepted_label_image, accepted_components):
    """Extract external contours separately for every accepted label."""
    records = []
    for row in accepted_components.itertuples(index=False):
        component_mask = np.where(accepted_label_image == int(row.label_id), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            warnings.warn(f"No contour found for accepted component {row.component_id}; it will be skipped.")
            continue
        records.append({
            "component_id": int(row.component_id), "label_id": int(row.label_id),
            "curve_fitted": bool(row.curve_fitted),
            "contours": contours, "contour_count": len(contours),
            "bbox_x": int(row.bbox_x), "bbox_y": int(row.bbox_y),
            "bbox_width": int(row.bbox_width), "bbox_height": int(row.bbox_height),
        })
    return records


def extract_contours_and_rois(source_bgr, accepted_label_image, accepted_components, source_path, padding):
    """Annotate accepted components and extract clipped image/mask ROI pairs."""
    source_path = Path(source_path)
    source_stem = source_path.stem
    image_height, image_width = accepted_label_image.shape
    contour_records = detect_component_contours(accepted_label_image, accepted_components)
    annotated = annotate_components(source_bgr, contour_records)
    contours_by_label = {record["label_id"]: record for record in contour_records}
    roi_records = []
    for row in accepted_components.itertuples(index=False):
        label_id = int(row.label_id)
        contour_record = contours_by_label.get(label_id)
        if contour_record is None:
            continue
        x, y = int(row.bbox_x), int(row.bbox_y)
        width, height = int(row.bbox_width), int(row.bbox_height)
        ideal_x0, ideal_y0 = x - padding, y - padding
        ideal_x1, ideal_y1 = x + width + padding, y + height + padding
        roi_x0, roi_y0 = max(0, ideal_x0), max(0, ideal_y0)
        roi_x1, roi_y1 = min(image_width, ideal_x1), min(image_height, ideal_y1)
        component_mask = np.where(accepted_label_image == label_id, 255, 0).astype(np.uint8)
        image_roi = source_bgr[roi_y0:roi_y1, roi_x0:roi_x1].copy()
        mask_roi = component_mask[roi_y0:roi_y1, roi_x0:roi_x1].copy()
        if image_roi.size == 0 or mask_roi.size == 0 or not np.any(mask_roi):
            warnings.warn(f"Skipping empty ROI for {source_path.name}, component {row.component_id}.")
            continue
        track_id = f"{source_stem}_track_{label_id:04d}"
        roi_records.append({
            "source_filename": source_path.name, "source_path": str(source_path.resolve()),
            "source_stem": source_stem, "track_id": track_id,
            "component_id": int(row.component_id), "label_id": label_id,
            "source_component_ids": str(row.source_component_ids),
            "fragment_count": int(row.fragment_count),
            "trajectory_linked": bool(row.trajectory_linked),
            "trajectory_link_count": int(row.trajectory_link_count),
            "curve_fitted": bool(row.curve_fitted),
            "curve_model": str(row.curve_model),
            "curve_score": float(row.curve_score),
            "curve_contrast": float(row.curve_contrast),
            "curve_axis_span_pixels": float(row.curve_axis_span_pixels),
            "curve_total_turn_degrees": float(row.curve_total_turn_degrees),
            "trajectory_link_length_pixels": float(row.trajectory_link_length_pixels),
            "trajectory_mean_link_gap_pixels": float(row.trajectory_mean_link_gap_pixels),
            "trajectory_mean_local_linearity": float(row.trajectory_mean_local_linearity),
            "trajectory_max_link_alignment_degrees": float(row.trajectory_max_link_alignment_degrees),
            "trajectory_max_tangent_change_degrees": float(row.trajectory_max_tangent_change_degrees),
            "trajectory_step_ratio": float(row.trajectory_step_ratio),
            "area_pixels": int(row.area_pixels), "centroid_x": float(row.centroid_x),
            "centroid_y": float(row.centroid_y), "bbox_x": x, "bbox_y": y,
            "bbox_width": width, "bbox_height": height,
            "bbox_long_side": int(row.bbox_long_side), "bbox_short_side": int(row.bbox_short_side),
            "bbox_x2_exclusive": x + width, "bbox_y2_exclusive": y + height,
            "touches_boundary": bool(row.touches_boundary),
            "contour_count": int(contour_record["contour_count"]), "roi_padding": int(padding),
            "roi_x0": roi_x0, "roi_y0": roi_y0, "roi_x1_exclusive": roi_x1,
            "roi_y1_exclusive": roi_y1, "roi_width": roi_x1 - roi_x0,
            "roi_height": roi_y1 - roi_y0,
            "roi_clipped_to_boundary": bool(
                roi_x0 != ideal_x0 or roi_y0 != ideal_y0 or roi_x1 != ideal_x1 or roi_y1 != ideal_y1),
            "roi_image_file": (Path("rois") / "images" / f"{track_id}.png").as_posix(),
            "roi_mask_file": (Path("rois") / "masks" / f"{track_id}_mask.png").as_posix(),
            "_image_roi": image_roi, "_mask_roi": mask_roi,
        })
    return annotated, roi_records
