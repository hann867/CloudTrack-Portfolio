"""Batch Module 3 feature extraction for audited labelled track crops."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np

from backend.src.geometric_classification.feature_extractor import extract_geometric_features
from backend.src.geometric_classification.track_segmentation import segment_track

from .dataset_audit import _write_csv


SEGMENTATION_SOURCE = "temporary_module3_development"
GEOMETRIC_FEATURE_COLUMNS = [
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
]
FEATURE_COLUMNS = [
    "track_id",
    "class",
    "crop_filename",
    "source_dataset",
    "source_image",
    "source_isotope",
    "physical_origin",
    "ground_truth_quality",
    "event_id",
    "roi_type",
    "rotation_angle_degrees",
    "crop_width_px",
    "crop_height_px",
    "segmentation_source",
    "processing_status",
    "error_message",
] + GEOMETRIC_FEATURE_COLUMNS
EXTRACTION_SUMMARY_COLUMNS = ["class", "total_samples", "success", "failed", "success_rate", "failure_reasons"]
DESCRIPTIVE_FEATURES = [
    "track_length_px",
    "mean_width_px",
    "median_width_px",
    "max_width_px",
    "mean_curvature",
    "max_curvature",
    "tortuosity",
]
DESCRIPTIVE_COLUMNS = ["class", "feature", "count", "mean", "median", "standard_deviation", "min", "max"]


def validate_track_mask(mask: np.ndarray) -> str | None:
    """Return a QC failure message, or ``None`` for a plausible binary mask."""

    if mask is None or np.asarray(mask).size == 0:
        return "Segmentation mask is empty."
    binary = np.asarray(mask) > 0
    foreground = int(binary.sum())
    if foreground == 0:
        return "Segmentation mask contains no foreground."
    fraction = foreground / binary.size
    if fraction >= 0.9:
        return f"Segmentation mask is almost entirely foreground ({fraction:.3f})."
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    components = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] > 0]
    if not components:
        return "Segmentation mask has no connected foreground component."
    if len(components) > 5:
        return f"Segmentation mask has implausibly many foreground components ({len(components)})."
    largest_fraction = max(components) / foreground
    if largest_fraction < 0.75:
        return f"Segmentation foreground is too fragmented (largest component fraction {largest_fraction:.3f})."
    return None


def _base_record(record: Mapping[str, str]) -> dict[str, Any]:
    return {
        field: record.get(field, "")
        for field in (
            "track_id",
            "class",
            "crop_filename",
            "source_dataset",
            "source_image",
            "source_isotope",
            "physical_origin",
            "ground_truth_quality",
            "event_id",
            "roi_type",
            "rotation_angle_degrees",
        )
    }


def _failure(base: dict[str, Any], message: str, width: int | str = "", height: int | str = "") -> dict[str, Any]:
    return {
        **base,
        "crop_width_px": width,
        "crop_height_px": height,
        "segmentation_source": SEGMENTATION_SOURCE,
        "processing_status": "failed",
        "error_message": message,
    }


def extract_labelled_geometric_features(
    dataset_root: str | Path,
    records: Iterable[Mapping[str, str]],
    output_dir: str | Path | None = None,
    *,
    segmenter: Callable[[np.ndarray], Mapping[str, Any]] = segment_track,
    geometric_extractor: Callable[[np.ndarray], Mapping[str, Any]] = extract_geometric_features,
) -> list[dict[str, Any]]:
    """Extract unchanged Module 3 features; one bad sample never stops the batch."""

    root = Path(dataset_root).expanduser().resolve()
    output_rows: list[dict[str, Any]] = []
    for record in records:
        class_name = record.get("class", "").strip()
        if class_name not in {"Alpha", "Electron"}:
            continue
        base = _base_record(record)
        crop_path = root / class_name.lower() / record.get("crop_filename", "").strip()
        try:
            crop = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
            if crop is None or crop.size == 0:
                output_rows.append(_failure(base, f"Could not read crop: {crop_path}"))
                continue
            height, width = map(int, crop.shape[:2])
            segmentation_input = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR) if crop.ndim == 3 and crop.shape[2] == 4 else crop
            segmentation = segmenter(segmentation_input)
            if segmentation.get("processing_status") != "success":
                output_rows.append(_failure(base, str(segmentation.get("error_message") or "Segmentation failed."), width, height))
                continue
            mask = np.asarray(segmentation.get("mask"))
            mask_error = validate_track_mask(mask)
            if mask_error:
                output_rows.append(_failure(base, mask_error, width, height))
                continue
            features = dict(geometric_extractor(mask))
            if features.get("processing_status") != "success":
                output_rows.append(_failure(base, str(features.get("error_message") or "Geometric extraction failed."), width, height))
                continue
            output_rows.append(
                {
                    **base,
                    "crop_width_px": width,
                    "crop_height_px": height,
                    "segmentation_source": SEGMENTATION_SOURCE,
                    "processing_status": "success",
                    "error_message": "",
                    **{column: features.get(column, "") for column in GEOMETRIC_FEATURE_COLUMNS},
                }
            )
        except Exception as exc:
            output_rows.append(_failure(base, f"{type(exc).__name__}: {exc}"))

    if output_dir is not None:
        output = Path(output_dir)
        _write_csv(output / "labelled_geometric_features.csv", FEATURE_COLUMNS, output_rows)
        _write_csv(output / "feature_extraction_summary.csv", EXTRACTION_SUMMARY_COLUMNS, build_extraction_summary(output_rows))
        _write_csv(output / "feature_summary_by_class.csv", DESCRIPTIVE_COLUMNS, build_descriptive_summary(output_rows))
    return output_rows


def build_extraction_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = list(rows)
    summary: list[dict[str, Any]] = []
    for class_name in ("Alpha", "Electron"):
        class_rows = [row for row in records if row.get("class") == class_name]
        success = sum(row.get("processing_status") == "success" for row in class_rows)
        failed = len(class_rows) - success
        reasons = Counter(str(row.get("error_message", "")) for row in class_rows if row.get("processing_status") != "success")
        summary.append(
            {
                "class": class_name,
                "total_samples": len(class_rows),
                "success": success,
                "failed": failed,
                "success_rate": f"{success / len(class_rows):.6f}" if class_rows else "0.000000",
                "failure_reasons": "; ".join(f"{reason} ({count})" for reason, count in sorted(reasons.items())),
            }
        )
    return summary


def build_descriptive_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("processing_status") == "success"]
    summary: list[dict[str, Any]] = []
    for class_name in ("Alpha", "Electron"):
        class_rows = [row for row in successful if row.get("class") == class_name]
        for feature in DESCRIPTIVE_FEATURES:
            values = np.asarray([float(row[feature]) for row in class_rows if row.get(feature, "") not in ("", None)], dtype=float)
            summary.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "count": int(values.size),
                    "mean": float(np.mean(values)) if values.size else "",
                    "median": float(np.median(values)) if values.size else "",
                    "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else (0.0 if values.size else ""),
                    "min": float(np.min(values)) if values.size else "",
                    "max": float(np.max(values)) if values.size else "",
                }
            )
    return summary

