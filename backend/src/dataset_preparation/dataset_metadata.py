"""Metadata, naming, and persistence for manually labelled track crops."""

from __future__ import annotations

import csv
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .crop_utils import (
    RotatedROI,
    crop_original_pixels,
    extract_rotated_crop,
    rotated_roi_bounding_box,
    validate_roi,
    validate_rotated_roi,
)


LEGACY_LABEL_COLUMNS = [
    "track_id",
    "class",
    "crop_filename",
    "source_dataset",
    "source_image",
    "source_isotope",
    "physical_origin",
    "source_panel",
    "x",
    "y",
    "width",
    "height",
    "label_basis",
    "ground_truth_quality",
    "event_id",
    "notes",
]

ROI_PROVENANCE_COLUMNS = [
    "roi_type",
    "center_x",
    "center_y",
    "roi_width",
    "roi_height",
    "rotation_angle_degrees",
    "corner_1_x",
    "corner_1_y",
    "corner_2_x",
    "corner_2_y",
    "corner_3_x",
    "corner_3_y",
    "corner_4_x",
    "corner_4_y",
]

LABEL_COLUMNS = LEGACY_LABEL_COLUMNS + ROI_PROVENANCE_COLUMNS

SOURCE_METADATA_COLUMNS = [
    "source_dataset",
    "source_reference",
    "source_isotope",
    "particle_class",
    "physical_origin",
    "notes",
]

AWAN_IMAGE_NAME = "Particle_Tracks_in_AWAN_Expansion_Cloud_Chamber.jpg"
AWAN_PANEL_METADATA = {
    "left": {"source_isotope": "Am-241", "physical_origin": "alpha_decay"},
    "right": {"source_isotope": "Sr-90/Y-90", "physical_origin": "beta_minus_decay"},
}


@dataclass(frozen=True)
class LabelSpec:
    class_name: str
    prefix: str
    directory: str
    label_basis: str
    ground_truth_quality: str


LABEL_SPECS = {
    "alpha": LabelSpec("Alpha", "A", "alpha", "known_source_manual_track_selection", "high"),
    "electron": LabelSpec("Electron", "E", "electron", "known_source_manual_track_selection", "high"),
    "uncertain": LabelSpec("Uncertain", "U", "uncertain", "uncertain", "low"),
}


def normalize_label(label: str) -> str:
    aliases = {"a": "alpha", "alpha": "alpha", "e": "electron", "electron": "electron", "u": "uncertain", "uncertain": "uncertain"}
    try:
        return aliases[label.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("Label must be Alpha, Electron, or Uncertain") from exc


def class_metadata(label: str) -> dict[str, str]:
    """Return manual-label defaults without examining image measurements."""

    spec = LABEL_SPECS[normalize_label(label)]
    return {
        "class": spec.class_name,
        "label_basis": spec.label_basis,
        "ground_truth_quality": spec.ground_truth_quality,
    }


def create_event_id(source_dataset: str = "EVENT") -> str:
    """Create a unique event group ID when no known grouping is supplied."""

    stem = re.sub(r"[^A-Za-z0-9]+", "_", source_dataset.strip().upper()).strip("_") or "EVENT"
    return f"{stem}_EVENT_{uuid.uuid4().hex[:12].upper()}"


def ensure_dataset_structure(output_root: Path) -> None:
    """Create class folders and CSV templates without replacing existing data."""

    output_root = Path(output_root)
    for spec in LABEL_SPECS.values():
        (output_root / spec.directory).mkdir(parents=True, exist_ok=True)
    _create_csv_if_missing(output_root / "labels.csv", LABEL_COLUMNS, [])
    _create_csv_if_missing(
        output_root / "source_metadata.csv",
        SOURCE_METADATA_COLUMNS,
        [
            {
                "source_dataset": "AWAN",
                "source_reference": "[enter official source reference]",
                "source_isotope": "Am-241",
                "particle_class": "Alpha candidate",
                "physical_origin": "alpha_decay",
                "notes": "Left panel; manual track-level label still required.",
            },
            {
                "source_dataset": "AWAN",
                "source_reference": "[enter official source reference]",
                "source_isotope": "Sr-90/Y-90",
                "particle_class": "Electron candidate",
                "physical_origin": "beta_minus_decay",
                "notes": "Right panel; manual track-level label still required.",
            },
            {
                "source_dataset": "McGill",
                "source_reference": "[enter official source reference]",
                "source_isotope": "Am-241",
                "particle_class": "Alpha candidate",
                "physical_origin": "alpha_decay",
                "notes": "Manual track-level label still required.",
            },
            {
                "source_dataset": "McGill",
                "source_reference": "[enter official source reference]",
                "source_isotope": "Pb-210",
                "particle_class": "Electron candidate",
                "physical_origin": "beta_minus_decay",
                "notes": "Manual track-level label still required.",
            },
        ],
    )


def _format_coordinate(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def axis_aligned_roi_metadata(roi: tuple[int, int, int, int]) -> dict[str, object]:
    """Create oriented-provenance fields for a legacy rectangular ROI."""

    x, y, width, height = roi
    corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
    metadata: dict[str, object] = {
        "roi_type": "axis_aligned",
        "center_x": _format_coordinate(x + width / 2.0),
        "center_y": _format_coordinate(y + height / 2.0),
        "roi_width": width,
        "roi_height": height,
        "rotation_angle_degrees": "0",
    }
    for index, (corner_x, corner_y) in enumerate(corners, start=1):
        metadata[f"corner_{index}_x"] = corner_x
        metadata[f"corner_{index}_y"] = corner_y
    return metadata


def rotated_roi_metadata(roi: RotatedROI) -> dict[str, object]:
    """Serialize complete reproducible rotated-rectangle provenance."""

    metadata: dict[str, object] = {
        "roi_type": "rotated",
        "center_x": _format_coordinate(roi.center_x),
        "center_y": _format_coordinate(roi.center_y),
        "roi_width": _format_coordinate(roi.width),
        "roi_height": _format_coordinate(roi.height),
        "rotation_angle_degrees": _format_coordinate(roi.angle_degrees),
    }
    for index, (corner_x, corner_y) in enumerate(roi.corners, start=1):
        metadata[f"corner_{index}_x"] = _format_coordinate(corner_x)
        metadata[f"corner_{index}_y"] = _format_coordinate(corner_y)
    return metadata


def _create_csv_if_missing(path: Path, columns: list[str], rows: list[Mapping[str, object]]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def ensure_labels_schema(labels_path: Path) -> None:
    """Upgrade the legacy schema additively while preserving old field values."""

    labels_path = Path(labels_path)
    if not labels_path.exists():
        _create_csv_if_missing(labels_path, LABEL_COLUMNS, [])
        return
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if fieldnames == LABEL_COLUMNS:
        return
    if fieldnames != LEGACY_LABEL_COLUMNS:
        raise ValueError("labels.csv columns do not match a supported schema")

    upgraded_rows: list[dict[str, object]] = []
    for row in rows:
        upgraded = dict(row)
        try:
            legacy_roi = tuple(int(row[column]) for column in ("x", "y", "width", "height"))
            upgraded.update(axis_aligned_roi_metadata(legacy_roi))  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot migrate ROI metadata for track {row.get('track_id', '')}") from exc
        upgraded_rows.append(upgraded)

    temporary_path = labels_path.with_name(f".{labels_path.name}.schema-upgrade.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Schema-upgrade temporary file already exists: {temporary_path}")
    try:
        with temporary_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            writer.writerows(upgraded_rows)
        temporary_path.replace(labels_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _existing_ids(output_root: Path, label: str) -> set[str]:
    spec = LABEL_SPECS[normalize_label(label)]
    ids: set[str] = set()
    labels_path = Path(output_root) / "labels.csv"
    if labels_path.exists():
        with labels_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "track_id" in reader.fieldnames:
                ids.update(row["track_id"].strip().upper() for row in reader if row.get("track_id"))
    class_directory = Path(output_root) / spec.directory
    if class_directory.exists():
        ids.update(path.stem.upper() for path in class_directory.glob(f"{spec.prefix}*.png"))
    return ids


def next_track_id(output_root: Path, label: str) -> str:
    """Return the next unused class-specific ID using files and labels.csv."""

    spec = LABEL_SPECS[normalize_label(label)]
    pattern = re.compile(rf"^{spec.prefix}(\d+)$", re.IGNORECASE)
    numbers = [int(match.group(1)) for item in _existing_ids(output_root, label) if (match := pattern.match(item))]
    candidate = max(numbers, default=0) + 1
    existing = _existing_ids(output_root, label)
    while True:
        track_id = f"{spec.prefix}{candidate:03d}"
        if track_id not in existing:
            return track_id
        candidate += 1


def build_label_record(
    *,
    track_id: str,
    label: str,
    source_dataset: str,
    source_image: str,
    roi: tuple[int, int, int, int],
    event_id: str,
    source_isotope: str = "",
    physical_origin: str = "unknown",
    source_panel: str = "",
    label_basis: str | None = None,
    ground_truth_quality: str | None = None,
    notes: str = "",
    rotated_roi: RotatedROI | None = None,
) -> dict[str, object]:
    spec = LABEL_SPECS[normalize_label(label)]
    x, y, width, height = roi
    if not event_id.strip():
        raise ValueError("event_id cannot be blank")
    record: dict[str, object] = {
        "track_id": track_id,
        "class": spec.class_name,
        "crop_filename": f"{track_id}.png",
        "source_dataset": source_dataset,
        "source_image": source_image,
        "source_isotope": source_isotope,
        "physical_origin": physical_origin or "unknown",
        "source_panel": source_panel,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "label_basis": label_basis or spec.label_basis,
        "ground_truth_quality": ground_truth_quality or spec.ground_truth_quality,
        "event_id": event_id,
        "notes": notes,
    }
    record.update(rotated_roi_metadata(rotated_roi) if rotated_roi else axis_aligned_roi_metadata(roi))
    return record


def append_label_record(labels_path: Path, record: Mapping[str, object]) -> None:
    """Append exactly one complete, non-duplicate record to labels.csv."""

    prepared_record = dict(record)
    missing = [column for column in LABEL_COLUMNS if column not in prepared_record]
    if missing and set(missing) == set(ROI_PROVENANCE_COLUMNS) and all(
        column in prepared_record for column in LEGACY_LABEL_COLUMNS
    ):
        try:
            legacy_roi = tuple(int(prepared_record[column]) for column in ("x", "y", "width", "height"))
            prepared_record.update(axis_aligned_roi_metadata(legacy_roi))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("Legacy axis-aligned ROI values are invalid") from exc
        missing = []
    if missing:
        raise ValueError(f"Missing label columns: {', '.join(missing)}")
    labels_path = Path(labels_path)
    ensure_labels_schema(labels_path)
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LABEL_COLUMNS:
            raise ValueError("labels.csv columns do not match the required schema")
        if any(row["track_id"] == str(prepared_record["track_id"]) for row in reader):
            raise FileExistsError(f"Track ID already exists: {prepared_record['track_id']}")
    with labels_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, extrasaction="ignore").writerow(prepared_record)


def save_annotation(
    *,
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    label: str,
    output_root: Path,
    source_dataset: str,
    source_image: str,
    event_id: str | None = None,
    source_isotope: str = "",
    physical_origin: str = "unknown",
    source_panel: str = "",
    label_basis: str | None = None,
    ground_truth_quality: str | None = None,
    notes: str = "",
    rotated_roi: RotatedROI | None = None,
    prepared_crop: np.ndarray | None = None,
) -> dict[str, object]:
    """Save one previewed crop and append its manual label metadata.

    Axis-aligned crops are exact source-array slices. Rotated crops use the one
    perspective-warp extraction documented by :func:`extract_rotated_crop` and
    therefore contain geometrically interpolated pixels.
    """

    output_root = Path(output_root)
    ensure_dataset_structure(output_root)
    normalized = normalize_label(label)
    spec = LABEL_SPECS[normalized]
    if rotated_roi is None:
        valid_roi = validate_roi(roi, image.shape[1], image.shape[0])
        expected_shape = (valid_roi[3], valid_roi[2])
    else:
        validate_rotated_roi(rotated_roi, image.shape[1], image.shape[0])
        valid_roi = rotated_roi_bounding_box(rotated_roi, image.shape[1], image.shape[0])
        expected_shape = (max(2, round(rotated_roi.height)), max(2, round(rotated_roi.width)))
    track_id = next_track_id(output_root, normalized)
    destination = output_root / spec.directory / f"{track_id}.png"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    if prepared_crop is None:
        crop = crop_original_pixels(image, valid_roi) if rotated_roi is None else extract_rotated_crop(image, rotated_roi)
    else:
        if not isinstance(prepared_crop, np.ndarray) or prepared_crop.shape[:2] != expected_shape:
            raise ValueError("Prepared crop dimensions do not match the selected ROI")
        if prepared_crop.dtype != image.dtype:
            raise ValueError("Prepared crop dtype does not match the source image")
        crop = prepared_crop
    if not cv2.imwrite(str(destination), crop):
        raise OSError(f"OpenCV could not write {destination}")
    record = build_label_record(
        track_id=track_id,
        label=normalized,
        source_dataset=source_dataset,
        source_image=source_image,
        roi=valid_roi,
        event_id=event_id or create_event_id(source_dataset),
        source_isotope=source_isotope,
        physical_origin=physical_origin,
        source_panel=source_panel,
        label_basis=label_basis,
        ground_truth_quality=ground_truth_quality,
        notes=notes,
        rotated_roi=rotated_roi,
    )
    try:
        append_label_record(output_root / "labels.csv", record)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return record
