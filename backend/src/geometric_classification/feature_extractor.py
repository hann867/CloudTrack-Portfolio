"""Reusable mask analysis and annotated-dataset batch processing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .annotation_roi import crop_inside_rectangle, detect_red_rectangle
from .geometric_features import curvature_statistics, tortuosity, width_statistics
from .skeleton_analysis import principal_skeleton_path, skeletonise_mask
from .track_segmentation import segment_track


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FEATURE_COLUMNS = [
    "track_id", "source_image", "relative_path", "video_folder",
    "roi_x", "roi_y", "roi_width", "roi_height", "track_length_px", "track_length_mm",
    "mean_width_px", "median_width_px", "max_width_px", "mean_curvature", "max_curvature",
    "tortuosity", "endpoint_distance_px", "mask_area_px", "skeleton_pixels", "label",
    "predicted_class", "confidence", "probability_alpha", "probability_electron",
    "processing_status", "error_message",
]


def _failed(message: str) -> dict[str, Any]:
    return {"processing_status": "failed", "error_message": message}


def extract_geometric_features(mask: np.ndarray, mm_per_pixel: float | None = None) -> dict[str, Any]:
    """Extract length, width, curvature and tortuosity from a binary mask."""
    if mask is None or np.asarray(mask).size == 0 or not np.any(mask):
        return _failed("Track mask is empty.")
    binary = np.asarray(mask) > 0
    skeleton = skeletonise_mask(binary)
    path_result = principal_skeleton_path(skeleton)
    if path_result["processing_status"] != "success":
        result = _failed(str(path_result["error_message"]))
        result.update({"mask_area_px": int(binary.sum()), "skeleton_pixels": int(skeleton.sum())})
        return result
    path = np.asarray(path_result["path"])
    length = float(path_result["track_length_px"])
    widths = width_statistics(binary, path)
    curvatures = curvature_statistics(path)
    track_tortuosity, endpoint_distance = tortuosity(path, length)
    return {
        "track_length_px": length,
        "track_length_mm": length * mm_per_pixel if mm_per_pixel is not None else None,
        **widths, **curvatures, "tortuosity": float(track_tortuosity),
        "endpoint_distance_px": endpoint_distance, "mask_area_px": int(binary.sum()),
        "skeleton_pixels": int(skeleton.sum()), "processing_status": "success",
        "error_message": None, "skeleton": skeleton, "principal_path": path,
        "endpoints": path_result["endpoints"],
    }


analyse_track_mask = extract_geometric_features


def _video_folder(relative_path: Path) -> str:
    return relative_path.parts[0] if len(relative_path.parts) > 1 else relative_path.stem.split(".mp4")[0]


def analyse_annotated_image(image_path: str | Path, dataset_root: str | Path | None = None, inward_margin: int = 4) -> dict[str, Any]:
    """Run annotation detection, temporary segmentation and mask analysis."""
    path = Path(image_path)
    root = Path(dataset_root) if dataset_root is not None else path.parent
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(path.name)
    base = {"source_image": path.name, "relative_path": relative.as_posix(), "video_folder": _video_folder(relative)}
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {**base, **_failed(f"Could not load image: {path}")}
    rectangle = detect_red_rectangle(image)
    if rectangle["processing_status"] != "success":
        return {**base, **_failed(str(rectangle["error_message"])), "image": image}
    roi_fields = {"roi_x": rectangle["x"], "roi_y": rectangle["y"], "roi_width": rectangle["width"], "roi_height": rectangle["height"]}
    try:
        roi = crop_inside_rectangle(image, rectangle, inward_margin)
    except ValueError as exc:
        return {**base, **roi_fields, **_failed(str(exc)), "image": image, "rectangle": rectangle}
    segmentation = segment_track(roi)
    if segmentation["processing_status"] != "success":
        return {**base, **roi_fields, **_failed(str(segmentation["error_message"])), "image": image, "rectangle": rectangle, "roi": roi}
    features = extract_geometric_features(np.asarray(segmentation["mask"]))
    return {**base, **roi_fields, **features, "image": image, "rectangle": rectangle, "roi": roi, "mask": segmentation["mask"]}


def find_annotated_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def load_labels(label_file: str | Path | None) -> dict[str, str]:
    if label_file is None or not Path(label_file).exists():
        return {}
    labels: dict[str, str] = {}
    with Path(label_file).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_path, label = row.get("relative_path", "").strip(), row.get("label", "").strip()
            if relative_path and label:
                labels[Path(relative_path).as_posix()] = label
    return labels


def _serialisable_record(record: dict[str, Any]) -> dict[str, Any]:
    return {column: record.get(column) for column in FEATURE_COLUMNS}


def process_annotated_dataset(input_root: str | Path, output_csv: str | Path, label_file: str | Path | None = None, diagnostics_dir: str | Path | None = None, diagnostic_limit: int = 0) -> pd.DataFrame:
    """Recursively process every supported image and save one CSV row per image."""
    from .visualisation import save_diagnostic_figure
    root = Path(input_root)
    labels = load_labels(label_file)
    records: list[dict[str, Any]] = []
    diagnostics_saved = 0
    for index, path in enumerate(find_annotated_images(root), start=1):
        record = analyse_annotated_image(path, root)
        record["track_id"] = f"track_{index:05d}"
        record["label"] = labels.get(record["relative_path"])
        records.append(_serialisable_record(record))
        if diagnostics_dir is not None and diagnostics_saved < diagnostic_limit and record.get("processing_status") == "success":
            target = Path(diagnostics_dir) / f"{record['track_id']}_{path.stem}.png"
            save_diagnostic_figure(record, target)
            diagnostics_saved += 1
    frame = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
