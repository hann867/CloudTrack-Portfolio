"""Read-only quality-control audit for manually labelled track crops."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import cv2


ALLOWED_CLASSES = {"Alpha": "A", "Electron": "E", "Uncertain": "U"}
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
STATUS_ORDER = {"PASS": 0, "REVIEW": 1, "ERROR": 2}

ISSUE_COLUMNS = ["track_id", "status", "issue_code", "message"]
DUPLICATE_COLUMNS = ["duplicate_type", "value", "track_ids", "classes", "details"]
SOURCE_DISTRIBUTION_COLUMNS = [
    "class",
    "source_dataset",
    "source_image",
    "source_isotope",
    "physical_origin",
    "sample_count",
    "crop_width_min",
    "crop_width_max",
    "crop_height_min",
    "crop_height_max",
]
SUMMARY_COLUMNS = ["section", "class", "metric", "value"]


@dataclass
class DatasetAuditResult:
    records: list[dict[str, str]]
    record_statuses: dict[int, str]
    issues: list[dict[str, str]]
    duplicates: list[dict[str, str]]
    summary: list[dict[str, object]]
    source_distribution: list[dict[str, object]]
    source_confound: bool

    def structurally_valid_records(self, classes: Iterable[str] = ("Alpha", "Electron")) -> list[dict[str, str]]:
        allowed = set(classes)
        return [
            record
            for index, record in enumerate(self.records)
            if record.get("class", "").strip() in allowed and self.record_statuses[index] != "ERROR"
        ]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_status(statuses: dict[int, str], index: int, status: str) -> None:
    if STATUS_ORDER[status] > STATUS_ORDER[statuses[index]]:
        statuses[index] = status


def _issue(
    issues: list[dict[str, str]],
    statuses: dict[int, str],
    index: int,
    record: Mapping[str, str],
    status: str,
    code: str,
    message: str,
) -> None:
    _set_status(statuses, index, status)
    issues.append(
        {
            "track_id": record.get("track_id", "").strip(),
            "status": status,
            "issue_code": code,
            "message": message,
        }
    )


def _crop_path(root: Path, record: Mapping[str, str]) -> Path:
    class_directory = record.get("class", "").strip().lower()
    return root / class_directory / record.get("crop_filename", "").strip()


def _required_rotated_values(record: Mapping[str, str]) -> list[str]:
    fields = ["center_x", "center_y", "roi_width", "roi_height", "rotation_angle_degrees"]
    fields.extend(f"corner_{index}_{axis}" for index in range(1, 5) for axis in ("x", "y"))
    return [field for field in fields if not record.get(field, "").strip()]


def _detect_source_confound(records: list[dict[str, str]]) -> bool:
    dominant: dict[str, tuple[str, float]] = {}
    for class_name in ("Alpha", "Electron"):
        sources = [row.get("source_dataset", "").strip() for row in records if row.get("class", "").strip() == class_name]
        sources = [source for source in sources if source]
        if not sources:
            continue
        source, count = Counter(sources).most_common(1)[0]
        dominant[class_name] = (source, count / len(sources))
    return (
        len(dominant) == 2
        and dominant["Alpha"][0] != dominant["Electron"][0]
        and dominant["Alpha"][1] >= 0.8
        and dominant["Electron"][1] >= 0.8
    )


def audit_labelled_dataset(dataset_root: str | Path, output_dir: str | Path | None = None) -> DatasetAuditResult:
    """Audit labels and crops without modifying either source."""

    root = Path(dataset_root).expanduser().resolve()
    labels_path = root / "labels.csv"
    _, records = _read_csv(labels_path)
    source_metadata_path = root / "source_metadata.csv"
    source_metadata = _read_csv(source_metadata_path)[1] if source_metadata_path.is_file() else []
    known_provenance = {
        (
            row.get("source_dataset", "").strip(),
            row.get("source_isotope", "").strip(),
            row.get("physical_origin", "").strip(),
        )
        for row in source_metadata
    }

    statuses = {index: "PASS" for index in range(len(records))}
    issues: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    hashes: defaultdict[str, list[int]] = defaultdict(list)
    event_ids: defaultdict[str, list[int]] = defaultdict(list)
    roi_references: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    dimensions: dict[int, tuple[int, int]] = {}

    for index, record in enumerate(records):
        track_id = record.get("track_id", "").strip()
        class_name = record.get("class", "").strip()
        crop_filename = record.get("crop_filename", "").strip()
        if not track_id:
            _issue(issues, statuses, index, record, "ERROR", "blank_track_id", "track_id is blank")
        if class_name not in ALLOWED_CLASSES:
            _issue(issues, statuses, index, record, "ERROR", "invalid_class", f"Unsupported class: {class_name or '[blank]'}")
        else:
            prefix = ALLOWED_CLASSES[class_name]
            expected_track = len(track_id) == 4 and track_id.startswith(prefix) and track_id[1:].isdigit()
            if not expected_track:
                _issue(issues, statuses, index, record, "ERROR", "track_id_mismatch", f"{class_name} track_id must match {prefix}xxx")
            if crop_filename != f"{track_id}.png":
                _issue(issues, statuses, index, record, "ERROR", "crop_filename_mismatch", f"crop_filename must be {track_id}.png")

        crop_path = _crop_path(root, record)
        if not crop_filename or not crop_path.is_file():
            _issue(issues, statuses, index, record, "ERROR", "missing_crop", f"Crop file does not exist: {crop_path}")
        else:
            image = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
                _issue(issues, statuses, index, record, "ERROR", "unreadable_crop", f"Crop is unreadable or empty: {crop_path}")
            else:
                dimensions[index] = (int(image.shape[1]), int(image.shape[0]))
                hashes[hashlib.sha256(crop_path.read_bytes()).hexdigest()].append(index)

        event_id = record.get("event_id", "").strip()
        if not event_id:
            _issue(issues, statuses, index, record, "ERROR", "blank_event_id", "event_id is blank")
        else:
            event_ids[event_id].append(index)

        for field in ("source_dataset", "source_image"):
            if not record.get(field, "").strip():
                _issue(issues, statuses, index, record, "REVIEW", f"missing_{field}", f"{field} is blank")
        provenance = tuple(record.get(field, "").strip() for field in ("source_dataset", "source_isotope", "physical_origin"))
        if all(provenance) and known_provenance and provenance not in known_provenance:
            _issue(issues, statuses, index, record, "REVIEW", "unlisted_source_metadata", "Provenance combination is not present in source_metadata.csv")

        quality = record.get("ground_truth_quality", "").strip().lower()
        if quality in {"medium", "low"}:
            _issue(issues, statuses, index, record, "REVIEW", f"{quality}_ground_truth_quality", f"Ground-truth quality is {quality}")
        elif quality not in {"high", "medium", "low"}:
            _issue(issues, statuses, index, record, "REVIEW", "missing_ground_truth_quality", "Ground-truth quality is missing or invalid")

        roi_type = record.get("roi_type", "").strip()
        if roi_type not in {"axis_aligned", "rotated"}:
            _issue(issues, statuses, index, record, "ERROR", "invalid_roi_type", f"Unsupported ROI type: {roi_type or '[blank]'}")
        elif roi_type == "rotated":
            missing_rotated = _required_rotated_values(record)
            if missing_rotated:
                _issue(issues, statuses, index, record, "ERROR", "incomplete_rotated_roi", f"Missing rotated ROI fields: {', '.join(missing_rotated)}")

        roi_key = tuple(record.get(field, "").strip() for field in ("source_dataset", "source_image", "x", "y", "width", "height"))
        if roi_key[1] and all(roi_key[2:]):
            roi_references[roi_key].append(index)

    for event_id, indexes in event_ids.items():
        if len(indexes) < 2:
            continue
        classes = sorted({records[index].get("class", "").strip() for index in indexes})
        details = "Same event_id occurs across different labels" if len(classes) > 1 else "Repeated event_id"
        duplicates.append({"duplicate_type": "event_id", "value": event_id, "track_ids": ";".join(records[index].get("track_id", "") for index in indexes), "classes": ";".join(classes), "details": details})
        for index in indexes:
            _issue(issues, statuses, index, records[index], "REVIEW", "duplicate_event_id", details)

    for digest, indexes in hashes.items():
        if len(indexes) < 2:
            continue
        duplicates.append({"duplicate_type": "exact_crop_sha256", "value": digest, "track_ids": ";".join(records[index].get("track_id", "") for index in indexes), "classes": ";".join(sorted({records[index].get("class", "") for index in indexes})), "details": "Crop files are byte-for-byte identical"})
        for index in indexes:
            _issue(issues, statuses, index, records[index], "REVIEW", "duplicate_crop_hash", "Crop is an exact SHA-256 duplicate")

    for roi_key, indexes in roi_references.items():
        if len(indexes) < 2:
            continue
        duplicates.append({"duplicate_type": "source_image_roi", "value": "|".join(roi_key), "track_ids": ";".join(records[index].get("track_id", "") for index in indexes), "classes": ";".join(sorted({records[index].get("class", "") for index in indexes})), "details": "Same source image and axis-aligned ROI coordinates"})
        for index in indexes:
            _issue(issues, statuses, index, records[index], "REVIEW", "duplicate_source_roi", "Source image and ROI coordinates are duplicated")

    referenced = {record.get("crop_filename", "").strip() for record in records if record.get("crop_filename", "").strip()}
    for class_name in ALLOWED_CLASSES:
        class_dir = root / class_name.lower()
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.name not in referenced:
                issues.append({"track_id": "", "status": "REVIEW", "issue_code": "orphan_crop", "message": f"Unreferenced crop file: {path}"})

    grouped: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = tuple(record.get(field, "").strip() for field in ("class", "source_dataset", "source_image", "source_isotope", "physical_origin"))
        grouped[key].append(index)
    source_distribution: list[dict[str, object]] = []
    for key, indexes in sorted(grouped.items()):
        sizes = [dimensions[index] for index in indexes if index in dimensions]
        widths = [size[0] for size in sizes]
        heights = [size[1] for size in sizes]
        source_distribution.append(dict(zip(SOURCE_DISTRIBUTION_COLUMNS[:5], key)) | {
            "sample_count": len(indexes),
            "crop_width_min": min(widths) if widths else "",
            "crop_width_max": max(widths) if widths else "",
            "crop_height_min": min(heights) if heights else "",
            "crop_height_max": max(heights) if heights else "",
        })

    summary: list[dict[str, object]] = []
    for class_name in (*ALLOWED_CLASSES, "Invalid"):
        class_records = [row for row in records if row.get("class", "").strip() == class_name] if class_name != "Invalid" else [row for row in records if row.get("class", "").strip() not in ALLOWED_CLASSES]
        summary.append({"section": "dataset_count", "class": class_name, "metric": "records", "value": len(class_records)})
    for class_name in ALLOWED_CLASSES:
        for quality in ("high", "medium", "low", "missing"):
            count = sum(1 for row in records if row.get("class", "").strip() == class_name and ((row.get("ground_truth_quality", "").strip().lower() or "missing") == quality))
            summary.append({"section": "ground_truth_quality", "class": class_name, "metric": quality, "value": count})
    for status in ("PASS", "REVIEW", "ERROR"):
        summary.append({"section": "audit_status", "class": "ALL", "metric": status, "value": sum(value == status for value in statuses.values())})
    for roi_type in ("axis_aligned", "rotated", "missing_or_invalid"):
        count = sum(1 for row in records if (row.get("roi_type", "").strip() if row.get("roi_type", "").strip() in {"axis_aligned", "rotated"} else "missing_or_invalid") == roi_type)
        summary.append({"section": "roi_type", "class": "ALL", "metric": roi_type, "value": count})
    summary.append({"section": "warning", "class": "ALL", "metric": "potential_source_domain_confound", "value": "YES" if _detect_source_confound(records) else "NO"})
    summary.append({"section": "warning", "class": "ALL", "metric": "pixel_scale_effects", "value": "Pixel-based geometric features may include source-scale effects."})

    result = DatasetAuditResult(records, statuses, issues, duplicates, summary, source_distribution, _detect_source_confound(records))
    if output_dir is not None:
        output = Path(output_dir)
        _write_csv(output / "dataset_audit_summary.csv", SUMMARY_COLUMNS, summary)
        _write_csv(output / "dataset_audit_issues.csv", ISSUE_COLUMNS, issues)
        _write_csv(output / "dataset_source_distribution.csv", SOURCE_DISTRIBUTION_COLUMNS, source_distribution)
        _write_csv(output / "duplicate_report.csv", DUPLICATE_COLUMNS, duplicates)
    return result

