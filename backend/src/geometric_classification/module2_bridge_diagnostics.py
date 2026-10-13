"""Analysis-only export of in-memory Module 2 ROIs and Module 3 results."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MANIFEST_COLUMNS = [
    "track_id",
    "source_filename",
    "component_id",
    "label_id",
    "area_pixels",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "roi_x0",
    "roi_y0",
    "roi_width",
    "roi_height",
    "touches_boundary",
    "trajectory_linked",
    "curve_fitted",
    "processing_status",
    "error_message",
    "diagnostic_image_file",
    "diagnostic_mask_file",
    "diagnostic_overlay_file",
    "track_length_px",
    "mean_width_px",
    "median_width_px",
    "max_width_px",
    "mean_curvature",
    "max_curvature",
    "tortuosity",
]


def _safe_filename(value: Any, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return cleaned[:140] or fallback


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Could not encode diagnostic PNG: {path.name}")
    encoded.tofile(str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Diagnostic PNG was not created: {path}")


def _display_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"ROI image must be grayscale, BGR, or BGRA; received shape {image.shape}.")


def _overlay(image: np.ndarray, mask: np.ndarray, track_id: Any) -> np.ndarray:
    display = _display_bgr(image)
    if display.shape[:2] != mask.shape:
        raise ValueError(
            f"ROI image and mask dimensions differ: {display.shape[:2]} versus {mask.shape}."
        )
    foreground = mask > 0
    colour = np.zeros_like(display)
    colour[:, :] = (0, 180, 255)
    blended = display.copy()
    blended[foreground] = cv2.addWeighted(
        display[foreground], 0.55, colour[foreground], 0.45, 0
    )
    contours, _ = cv2.findContours(
        foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 1)
    label = str(track_id or "unknown track")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(
        blended,
        (0, 0),
        (min(blended.shape[1] - 1, text_width + 8), min(blended.shape[0] - 1, text_height + baseline + 8)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(blended, label, (4, text_height + 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return blended


def _manifest_row(
    roi: Mapping[str, Any], result: Mapping[str, Any], image_file: str, mask_file: str, overlay_file: str
) -> dict[str, Any]:
    features = result.get("features") if isinstance(result.get("features"), Mapping) else {}
    row = {column: "" for column in MANIFEST_COLUMNS}
    for field in (
        "track_id", "source_filename", "component_id", "label_id", "area_pixels",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height", "roi_x0", "roi_y0",
        "roi_width", "roi_height", "touches_boundary", "trajectory_linked", "curve_fitted",
    ):
        if field in roi:
            row[field] = roi[field]
    row["track_id"] = result.get("track_id", roi.get("track_id", ""))
    row["processing_status"] = result.get("processing_status", "failed")
    row["error_message"] = result.get("error_message") or ""
    row["diagnostic_image_file"] = image_file
    row["diagnostic_mask_file"] = mask_file
    row["diagnostic_overlay_file"] = overlay_file
    for feature in (
        "track_length_px", "mean_width_px", "median_width_px", "max_width_px",
        "mean_curvature", "max_curvature", "tortuosity",
    ):
        if feature in features:
            row[feature] = features[feature]
    return row


def export_module2_bridge_diagnostics(
    module2_result: Mapping[str, Any],
    track_results: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    *,
    source_stem: str | None = None,
) -> dict[str, Any]:
    """Export copies of Module 2 ROI arrays and a Module 3 result manifest."""

    rois = module2_result.get("rois") if isinstance(module2_result, Mapping) else None
    if not isinstance(rois, (list, tuple)):
        raise ValueError("Module 2 result must contain a list or tuple in 'rois'.")
    if len(rois) != len(track_results):
        raise ValueError("Module 2 ROI count and Module 3 result count must match exactly.")
    inferred_stem = source_stem
    if not inferred_stem and rois and isinstance(rois[0], Mapping):
        inferred_stem = str(rois[0].get("source_stem") or "")
    if not inferred_stem:
        path = module2_result.get("path")
        inferred_stem = Path(path).stem if path else "unknown_source"
    source_dir = Path(output_root) / _safe_filename(inferred_stem, "unknown_source")
    images_dir, masks_dir, overlays_dir = (
        source_dir / "images", source_dir / "masks", source_dir / "overlays"
    )
    for directory in (images_dir, masks_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (roi, result) in enumerate(zip(rois, track_results), start=1):
        if not isinstance(roi, Mapping) or not isinstance(result, Mapping):
            raise ValueError(f"ROI/result pair {index} must contain mappings.")
        track_id = result.get("track_id", roi.get("track_id"))
        file_id = _safe_filename(track_id, f"roi_{index:04d}")
        image_relative = (Path("images") / f"{file_id}.png").as_posix()
        mask_relative = (Path("masks") / f"{file_id}_mask.png").as_posix()
        overlay_relative = (Path("overlays") / f"{file_id}_overlay.png").as_posix()
        try:
            image = np.asarray(roi["_image_roi"]).copy()
            mask = np.asarray(roi["_mask_roi"]).copy()
            if image.size == 0:
                raise ValueError("Module 2 ROI image is empty.")
            if mask.ndim != 2 or mask.size == 0:
                raise ValueError("Module 2 ROI mask must be a nonempty 2-D array.")
            overlay = _overlay(image, mask, track_id)
            _write_png(source_dir / image_relative, image)
            _write_png(source_dir / mask_relative, mask)
            _write_png(source_dir / overlay_relative, overlay)
            row = _manifest_row(roi, result, image_relative, mask_relative, overlay_relative)
        except (KeyError, TypeError, ValueError, OSError, cv2.error) as exc:
            failed_result = dict(result)
            failed_result["processing_status"] = "failed"
            existing = str(result.get("error_message") or "").strip()
            export_error = f"Diagnostic export failed: {exc}"
            failed_result["error_message"] = f"{existing}; {export_error}" if existing else export_error
            row = _manifest_row(roi, failed_result, "", "", "")
        rows.append(row)

    manifest = source_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "diagnostic_root": str(source_dir),
        "manifest": str(manifest),
        "roi_count": len(rows),
        "exported_count": sum(bool(row["diagnostic_mask_file"]) for row in rows),
        "failed_count": sum(not bool(row["diagnostic_mask_file"]) for row in rows),
    }

