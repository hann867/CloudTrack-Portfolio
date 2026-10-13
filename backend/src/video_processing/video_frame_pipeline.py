"""Sequential Steps 4-6 coordinator built from the existing image modules."""

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from backend.src.detection_segmentation import CONFIG, process_image, save_png, validate_config
from backend.src.geometric_classification.production_inference import classify_module2_masks
from backend.src.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline


CLASS_COLOURS_BGR = {
    "Alpha": (252, 132, 192),
    "Electron": (250, 165, 96),
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(payload), handle, indent=2)


def _track_record(roi: Mapping[str, Any], classification: Mapping[str, Any], frame_name: str) -> dict:
    label_id = int(roi["label_id"])
    local_track_id = f"track_{label_id:04d}"
    public_fields = {
        key: _json_value(value)
        for key, value in roi.items()
        if not str(key).startswith("_")
        and key not in {"source_path", "source_filename", "source_stem", "roi_image_file", "roi_mask_file"}
    }
    public_fields.update(
        {
            "track_id": local_track_id,
            "roi_image_file": f"detection/rois/images/{Path(frame_name).stem}_{local_track_id}.png",
            "roi_mask_file": f"detection/rois/masks/{Path(frame_name).stem}_{local_track_id}_mask.png",
            "classification": _json_value(classification),
        }
    )
    return public_fields


def _save_detection_outputs(job_directory: Path, frame_name: str, result: Mapping[str, Any]) -> dict:
    frame_stem = Path(frame_name).stem
    relative_outputs = {
        "annotation": f"detection/annotations/{frame_name}",
        "binary_mask": f"detection/masks/binary/{frame_name}",
        "cleaned_mask": f"detection/masks/cleaned/{frame_name}",
        "labelled_mask": f"detection/masks/labelled/{frame_name}",
    }
    save_png(job_directory / relative_outputs["annotation"], result["annotated"])
    save_png(job_directory / relative_outputs["binary_mask"], result["binary"])
    save_png(job_directory / relative_outputs["cleaned_mask"], result["filtered_mask"])
    maximum_label = int(result["accepted_labels"].max()) if result["accepted_labels"].size else 0
    if maximum_label > np.iinfo(np.uint16).max:
        raise OverflowError("Accepted-label count exceeds the uint16 PNG limit.")
    save_png(job_directory / relative_outputs["labelled_mask"], result["accepted_labels"].astype(np.uint16))

    for roi in result["rois"]:
        local_track_id = f"track_{int(roi['label_id']):04d}"
        save_png(
            job_directory / "detection" / "rois" / "images" / f"{frame_stem}_{local_track_id}.png",
            roi["_image_roi"],
        )
        save_png(
            job_directory / "detection" / "rois" / "masks" / f"{frame_stem}_{local_track_id}_mask.png",
            roi["_mask_roi"],
        )
    return relative_outputs


def _annotate_classifications(source_bgr: np.ndarray, tracks: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Mirror the existing browser annotation colours and label content."""
    annotated = source_bgr.copy()
    image_height, image_width = annotated.shape[:2]
    font_scale = max(0.45, image_width * 0.0007)
    font_thickness = max(1, round(image_width * 0.0012))
    line_thickness = max(2, round(image_width * 0.0015))

    for track in tracks:
        classification = track.get("classification", {})
        if classification.get("processing_status") != "success":
            continue
        class_name = str(classification.get("predicted_class"))
        colour = CLASS_COLOURS_BGR.get(class_name)
        if colour is None:
            continue
        x = int(track["bbox_x"])
        y = int(track["bbox_y"])
        width = int(track["bbox_width"])
        height = int(track["bbox_height"])
        confidence = float(classification["confidence"]) * 100
        label = f"Track {track['label_id']} | {class_name} {confidence:.1f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        plaque_height = text_height + baseline + 10
        plaque_width = min(image_width, max(width, text_width + 12))
        plaque_x = max(0, min(x, image_width - plaque_width))
        plaque_y = y - plaque_height - 3 if y >= plaque_height + 3 else y
        plaque_y = max(0, min(plaque_y, image_height - plaque_height))
        cv2.rectangle(annotated, (x, y), (x + width, y + height), colour, line_thickness)
        cv2.rectangle(
            annotated,
            (plaque_x, plaque_y),
            (plaque_x + plaque_width, plaque_y + plaque_height),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (plaque_x + 6, plaque_y + text_height + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (7, 9, 12),
            font_thickness,
            cv2.LINE_AA,
        )
    return annotated


def _copy_file(job_directory: Path, source_relative: str, destination_relative: str) -> None:
    source = job_directory / source_relative
    if not source.is_file():
        return
    destination = job_directory / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _duplicate_frame_result(
    job_directory: Path,
    template: Mapping[str, Any],
    frame_mapping: Mapping[str, Any],
) -> dict:
    duplicate = deepcopy(template)
    old_name = str(template["filename"])
    new_name = str(frame_mapping["filename"])
    _copy_file(job_directory, f"preprocessed/{old_name}", f"preprocessed/{new_name}")
    _copy_file(job_directory, f"classified/{old_name}", f"classified/{new_name}")

    old_outputs = template.get("detection_outputs", {})
    new_outputs = {
        key: str(path).replace(old_name, new_name)
        for key, path in old_outputs.items()
    }
    for key, old_path in old_outputs.items():
        _copy_file(job_directory, old_path, new_outputs[key])

    old_stem = Path(old_name).stem
    new_stem = Path(new_name).stem
    for track in duplicate.get("tracks", []):
        for field in ("roi_image_file", "roi_mask_file"):
            old_path = track[field]
            new_path = old_path.replace(old_stem, new_stem, 1)
            _copy_file(job_directory, old_path, new_path)
            track[field] = new_path
        provenance = track.get("classification", {}).get("provenance", {})
        provenance.update(
            {
                "output_frame_index": frame_mapping["output_frame_index"],
                "source_frame_index": frame_mapping["source_frame_index"],
                "frame_filename": new_name,
            }
        )

    duplicate.update(
        {
            "output_frame_index": frame_mapping["output_frame_index"],
            "source_frame_index": frame_mapping["source_frame_index"],
            "timestamp_seconds": frame_mapping["timestamp_seconds"],
            "filename": new_name,
            "is_duplicate_source_frame": True,
            "detection_outputs": new_outputs,
        }
    )
    return duplicate


def process_extracted_frames(
    job_directory: Path,
    frame_mapping: Sequence[Mapping[str, Any]],
    *,
    mm_per_pixel: float | None = None,
    progress_callback: Callable[[str, int, int, int], None] | None = None,
) -> dict:
    """Run existing Modules 1-3 once per unique source frame, in order."""
    ordered_frames = list(frame_mapping)
    total_frames = len(ordered_frames)
    validate_config(CONFIG)
    for directory in (
        job_directory / "preprocessed",
        job_directory / "detection" / "annotations",
        job_directory / "detection" / "masks" / "binary",
        job_directory / "detection" / "masks" / "cleaned",
        job_directory / "detection" / "masks" / "labelled",
        job_directory / "detection" / "rois" / "images",
        job_directory / "detection" / "rois" / "masks",
        job_directory / "classified",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    cache: dict[int, dict] = {}
    frame_results: list[dict] = []
    unique_totals = {"total_tracks": 0, "alpha": 0, "electron": 0}

    for one_based_index, mapping in enumerate(ordered_frames, start=1):
        output_index = int(mapping["output_frame_index"])
        if output_index != one_based_index - 1:
            raise ValueError("Frame mapping is not in contiguous output-frame order.")
        source_index = int(mapping["source_frame_index"])
        frame_name = str(mapping["filename"])

        if source_index in cache:
            result = _duplicate_frame_result(job_directory, cache[source_index], mapping)
            frame_results.append(result)
            if progress_callback is not None:
                for operation_index, stage in enumerate(("PREPROCESSING", "DETECTING", "CLASSIFYING"), start=1):
                    progress_callback(stage, one_based_index, total_frames, operation_index)
            continue

        raw_path = job_directory / "frames" / frame_name
        classified_path = job_directory / "classified" / frame_name
        base_result = {
            "output_frame_index": output_index,
            "source_frame_index": source_index,
            "timestamp_seconds": mapping["timestamp_seconds"],
            "filename": frame_name,
            "is_duplicate_source_frame": False,
        }
        try:
            raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                raise ValueError(f"Extracted frame is unreadable: {frame_name}")

            if progress_callback is not None:
                progress_callback("PREPROCESSING", one_based_index, total_frames, 1)
            preprocessing = run_preprocessing_pipeline(raw_path)
            preprocessed = preprocessing.get("final_image")
            if preprocessed is None:
                raise RuntimeError("Module 1 did not return final_image.")
            preprocessed_path = job_directory / "preprocessed" / frame_name
            save_png(preprocessed_path, preprocessed)

            if progress_callback is not None:
                progress_callback("DETECTING", one_based_index, total_frames, 2)
            detection = process_image(preprocessed_path, CONFIG)
            detection_outputs = _save_detection_outputs(job_directory, frame_name, detection)

            if progress_callback is not None:
                progress_callback("CLASSIFYING", one_based_index, total_frames, 3)
            provenances = [
                {
                    "track_id": f"track_{int(roi['label_id']):04d}",
                    "component_id": int(roi["component_id"]),
                    "label_id": int(roi["label_id"]),
                    "bbox_x": int(roi["bbox_x"]),
                    "bbox_y": int(roi["bbox_y"]),
                    "bbox_width": int(roi["bbox_width"]),
                    "bbox_height": int(roi["bbox_height"]),
                    "output_frame_index": output_index,
                    "source_frame_index": source_index,
                    "frame_filename": frame_name,
                }
                for roi in detection["rois"]
            ]
            classification_batch = (
                classify_module2_masks(
                    [roi["_mask_roi"] for roi in detection["rois"]],
                    provenances=provenances,
                    mm_per_pixels=[mm_per_pixel] * len(provenances),
                )
                if provenances
                else {"processing_status": "SUCCESS", "results": []}
            )
            if len(classification_batch["results"]) != len(detection["rois"]):
                raise RuntimeError("Module 3 result count does not match the Module 2 track count.")
            tracks = [
                _track_record(roi, classification, frame_name)
                for roi, classification in zip(detection["rois"], classification_batch["results"])
            ]
            alpha_count = sum(
                track["classification"].get("predicted_class") == "Alpha"
                and track["classification"].get("processing_status") == "success"
                for track in tracks
            )
            electron_count = sum(
                track["classification"].get("predicted_class") == "Electron"
                and track["classification"].get("processing_status") == "success"
                for track in tracks
            )
            failed_track_classifications = sum(
                track["classification"].get("processing_status") != "success" for track in tracks
            )
            classification_errors = [
                str(track["classification"].get("error_message") or "Classification failed.")
                for track in tracks
                if track["classification"].get("processing_status") != "success"
            ]
            save_png(classified_path, _annotate_classifications(raw_bgr, tracks))
            result = {
                **base_result,
                "status": "failed" if failed_track_classifications else "success",
                "error": "; ".join(classification_errors) if classification_errors else None,
                "tracks_detected": len(tracks),
                "alpha_count": alpha_count,
                "electron_count": electron_count,
                "failed_track_classifications": failed_track_classifications,
                "classification_status": classification_batch["processing_status"],
                "detection_outputs": detection_outputs,
                "tracks": tracks,
            }
            unique_totals["total_tracks"] += len(tracks)
            unique_totals["alpha"] += alpha_count
            unique_totals["electron"] += electron_count
        except Exception as error:
            if raw_path.is_file():
                shutil.copy2(raw_path, classified_path)
            result = {
                **base_result,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "tracks_detected": 0,
                "alpha_count": 0,
                "electron_count": 0,
                "failed_track_classifications": 0,
                "classification_status": "FAILED",
                "detection_outputs": {},
                "tracks": [],
            }

        cache[source_index] = result
        frame_results.append(result)

    failed_frames = sum(frame["status"] == "failed" for frame in frame_results)
    preprocessed_frames = sum((job_directory / "preprocessed" / frame["filename"]).is_file() for frame in frame_results)
    classified_frames = sum((job_directory / "classified" / frame["filename"]).is_file() for frame in frame_results)
    results = {
        "job_id": job_directory.name,
        "current_stage": "CLASSIFYING",
        "current_frame": total_frames,
        "target_fps": 30,
        "total_frames": total_frames,
        "processed_frames": total_frames,
        "preprocessed_frames": preprocessed_frames,
        "classified_frames": classified_frames,
        "failed_frames": failed_frames,
        "unique_source_frames_processed": len(cache),
        "particles": unique_totals,
        "frames": frame_results,
    }
    _write_json(job_directory / "results.json", results)
    return results
