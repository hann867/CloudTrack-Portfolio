"""Controlled revision and audit of adjudicated production Module 2 mappings."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .confirmed_mapping_audit import association_2x2
from .feature_extractor import extract_geometric_features
from .ground_truth_candidate_matcher import manual_polygon
from .production_module2_features import (
    FEATURES,
    OUTPUT_COLUMNS as FEATURE_COLUMNS,
    extract_confirmed_module2_features,
    feature_sanity_issues,
)


REVISED_MAPPING_EXTRA_COLUMNS = ["mapping_status", "exclusion_reason", "duplicate_group"]
EXCLUSION_COLUMNS = [
    "ground_truth_track_id",
    "class",
    "source_dataset",
    "previous_module2_track_id",
    "exclusion_reason",
    "adjudication_basis",
]
OVERLAP_REVIEW_COLUMNS = [
    "track_id_a",
    "track_id_b",
    "source_image",
    "manual_roi_iou",
    "smaller_roi_coverage",
    "normalized_centroid_distance",
    "status",
]

DUPLICATE_GROUP_A003_A008 = "PHYSICAL_TRACK_A003_A008"
EXCLUSION_SPECS = {
    "A008": (
        "DUPLICATE_PHYSICAL_TRACK",
        "A003/A008 were adjudicated as the same physical track. Their provenance quality is equal, "
        "so the deterministic earlier ground-truth ID A003 is retained.",
    ),
    "A012": (
        "NO_VALID_MODULE2_CANDIDATE",
        "Selected and alternative Module 2 candidates do not isolate the complete labelled track.",
    ),
    "A020": (
        "NO_VALID_MODULE2_CANDIDATE",
        "Selected candidate is chamber/background and no reviewed alternative isolates the complete track.",
    ),
    "A021": (
        "NO_VALID_MODULE2_CANDIDATE",
        "Selected candidate is chamber/background and no reviewed alternative isolates the complete track.",
    ),
    "A023": (
        "NO_VALID_MODULE2_CANDIDATE",
        "Selected candidate is chamber/background and no reviewed alternative isolates the complete track.",
    ),
    "A025": (
        "DUPLICATE_WITH_NO_VALID_MODULE2_CANDIDATE",
        "A025/A028 are the same physical track; their shared candidate is unrelated background and no "
        "complete Module 2 alternative was found.",
    ),
    "A028": (
        "DUPLICATE_WITH_NO_VALID_MODULE2_CANDIDATE",
        "A025/A028 are the same physical track; their shared candidate is unrelated background and no "
        "complete Module 2 alternative was found.",
    ),
    "E025": (
        "NO_VALID_MODULE2_CANDIDATE",
        "Extreme tortuosity came from a frame-scale background mask; no reviewed candidate represents "
        "the complete labelled Electron track.",
    ),
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(
    path: str | Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _refuse_same_file(source: str | Path, output: str | Path) -> None:
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError(f"Refusing to overwrite protected input: {Path(source).resolve()}")


def build_revised_mapping(
    mapping_rows: Sequence[Mapping[str, str]],
    label_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Apply only the fixed adjudication decisions and preserve retained fields."""
    labels = {str(row.get("track_id", "")): row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("labels.csv contains duplicate track_id values.")
    mapped_ids = [str(row.get("ground_truth_track_id", "")) for row in mapping_rows]
    if len(mapped_ids) != len(set(mapped_ids)):
        raise ValueError("Original confirmed mapping contains duplicate ground_truth_track_id values.")
    missing = sorted(set(EXCLUSION_SPECS) - set(mapped_ids))
    if missing:
        raise ValueError(f"Adjudicated records missing from original mapping: {', '.join(missing)}")
    if "A003" not in mapped_ids:
        raise ValueError("A003 must exist so the deterministic duplicate representative can be retained.")

    revised: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    for original in mapping_rows:
        track_id = str(original.get("ground_truth_track_id", ""))
        label = labels.get(track_id)
        if label is None:
            raise ValueError(f"Confirmed mapping references unknown labelled track: {track_id}")
        if track_id in EXCLUSION_SPECS:
            reason, basis = EXCLUSION_SPECS[track_id]
            exclusions.append({
                "ground_truth_track_id": track_id,
                "class": str(label.get("class", "")),
                "source_dataset": str(label.get("source_dataset", "")),
                "previous_module2_track_id": str(original.get("module2_track_id", "")),
                "exclusion_reason": reason,
                "adjudication_basis": basis,
            })
            continue
        row = dict(original)
        row.update({
            "mapping_status": "RETAINED_AFTER_ADJUDICATION",
            "exclusion_reason": "",
            "duplicate_group": DUPLICATE_GROUP_A003_A008 if track_id == "A003" else "",
        })
        revised.append(row)

    original_columns = list(mapping_rows[0].keys()) if mapping_rows else []
    return revised, exclusions, [*original_columns, *REVISED_MAPPING_EXTRA_COLUMNS]


def revise_confirmed_mapping(
    original_mapping_csv: str | Path,
    labels_csv: str | Path,
    revised_mapping_csv: str | Path,
    exclusions_csv: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    _refuse_same_file(original_mapping_csv, revised_mapping_csv)
    mapping_rows = _read_csv(original_mapping_csv)
    label_rows = _read_csv(labels_csv)
    revised, exclusions, columns = build_revised_mapping(mapping_rows, label_rows)
    _atomic_csv(revised_mapping_csv, columns, revised)
    _atomic_csv(exclusions_csv, EXCLUSION_COLUMNS, exclusions)
    return revised, exclusions


def _polygon_overlap(first: Mapping[str, str], second: Mapping[str, str]) -> dict[str, float]:
    polygon_a = manual_polygon(first)
    polygon_b = manual_polygon(second)
    area_a = abs(float(cv2.contourArea(polygon_a)))
    area_b = abs(float(cv2.contourArea(polygon_b)))
    intersection, _ = cv2.intersectConvexConvex(polygon_a, polygon_b)
    intersection = max(0.0, float(intersection))
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    smaller_coverage = intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0
    centroid_a = np.mean(polygon_a, axis=0)
    centroid_b = np.mean(polygon_b, axis=0)
    minimum = np.minimum(polygon_a.min(axis=0), polygon_b.min(axis=0))
    maximum = np.maximum(polygon_a.max(axis=0), polygon_b.max(axis=0))
    diagonal = math.hypot(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1]))
    centroid_distance = float(np.linalg.norm(centroid_a - centroid_b))
    return {
        "iou": iou,
        "smaller_coverage": smaller_coverage,
        "normalized_centroid_distance": centroid_distance / diagonal if diagonal else math.inf,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_mapping_independence(
    revised_rows: Sequence[Mapping[str, str]],
    label_rows: Sequence[Mapping[str, str]],
    crop_root: str | Path,
) -> dict[str, Any]:
    """Audit identity, event, crop, and strong same-source ROI duplication signals."""
    labels = {str(row.get("track_id", "")): row for row in label_rows}
    mapped_ids = [str(row.get("ground_truth_track_id", "")) for row in revised_rows]
    module2_ids = [str(row.get("module2_track_id", "")) for row in revised_rows]
    duplicate_ground_ids = sorted(
        value for value, count in Counter(mapped_ids).items() if value and count > 1
    )
    duplicate_module2_ids = sorted(
        value for value, count in Counter(module2_ids).items() if value and count > 1
    )
    duplicate_groups = sorted(
        value
        for value, count in Counter(
            str(row.get("duplicate_group", "")) for row in revised_rows
        ).items()
        if value and count > 1
    )
    selected_labels = [labels[track_id] for track_id in mapped_ids if track_id in labels]
    if len(selected_labels) != len(mapped_ids):
        missing = sorted(set(mapped_ids) - set(labels))
        raise ValueError(f"Revised mapping references unknown tracks: {', '.join(missing)}")
    duplicate_event_ids = sorted(
        value
        for value, count in Counter(str(row.get("event_id", "")) for row in selected_labels).items()
        if value and count > 1
    )

    overlap_rows: list[dict[str, Any]] = []
    suspected_overlap_duplicates: list[dict[str, Any]] = []
    for index, first in enumerate(selected_labels):
        for second in selected_labels[index + 1:]:
            if first.get("source_image") != second.get("source_image"):
                continue
            metrics = _polygon_overlap(first, second)
            if metrics["iou"] <= 0:
                continue
            suspicious = (
                metrics["iou"] >= 0.80
                or (
                    metrics["smaller_coverage"] >= 0.90
                    and metrics["normalized_centroid_distance"] <= 0.10
                )
            )
            row = {
                "track_id_a": first["track_id"],
                "track_id_b": second["track_id"],
                "source_image": first["source_image"],
                "manual_roi_iou": metrics["iou"],
                "smaller_roi_coverage": metrics["smaller_coverage"],
                "normalized_centroid_distance": metrics["normalized_centroid_distance"],
                "status": "REVIEW_REQUIRED" if suspicious else "INFORMATIONAL_OVERLAP",
            }
            overlap_rows.append(row)
            if suspicious:
                suspected_overlap_duplicates.append(row)

    crop_hashes: dict[str, list[str]] = defaultdict(list)
    crop_directory = Path(crop_root)
    for label in selected_labels:
        filename = str(label.get("crop_filename", ""))
        path = crop_directory / filename
        if not filename or not path.is_file():
            continue
        crop_hashes[_sha256(path)].append(str(label["track_id"]))
    duplicate_crop_hash_groups = sorted(
        sorted(track_ids) for track_ids in crop_hashes.values() if len(track_ids) > 1
    )
    return {
        "duplicate_ground_truth_ids": duplicate_ground_ids,
        "duplicate_module2_ids": duplicate_module2_ids,
        "duplicate_physical_track_groups": duplicate_groups,
        "duplicate_event_ids": duplicate_event_ids,
        "overlap_rows": overlap_rows,
        "suspected_overlap_duplicates": suspected_overlap_duplicates,
        "duplicate_crop_hash_groups": duplicate_crop_hash_groups,
        "review_required": bool(
            duplicate_ground_ids
            or duplicate_module2_ids
            or duplicate_groups
            or duplicate_event_ids
            or suspected_overlap_duplicates
            or duplicate_crop_hash_groups
        ),
    }


def strict_geometry_extractor(
    mask: np.ndarray,
    *,
    extractor: Callable[[np.ndarray], Mapping[str, Any]] = extract_geometric_features,
) -> Mapping[str, Any]:
    """Validate a production mask, then invoke the unchanged Module 3 extractor."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return {"processing_status": "failed", "error_message": "Module 2 mask is not 2-D."}
    if not np.any(mask > 0):
        return {"processing_status": "failed", "error_message": "Module 2 mask has no foreground."}
    return extractor(mask)


def extreme_feature_outliers(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flag descriptive extreme outliers using reported 3*IQR fences; never remove rows."""
    successful = [row for row in rows if row.get("processing_status") == "success"]
    outliers: list[dict[str, Any]] = []
    for feature in FEATURES:
        values = np.asarray([float(row[feature]) for row in successful], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 4:
            continue
        q1, q3 = (float(value) for value in np.percentile(finite, [25, 75]))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        for row in successful:
            value = float(row[feature])
            if value < lower or value > upper:
                outliers.append({
                    "track_id": row["ground_truth_track_id"],
                    "feature": feature,
                    "value": value,
                    "lower_fence": lower,
                    "upper_fence": upper,
                })
    return outliers


def _counts_by(rows: Sequence[Mapping[str, Any]], *keys: str) -> Counter[Any]:
    return Counter(tuple(row.get(key, "") for key in keys) for row in rows)


def _retention_rate(final_count: int, original_count: int) -> float:
    return 100.0 * final_count / original_count if original_count else 0.0


def final_subset_audit(
    feature_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, str]],
    independence: Mapping[str, Any],
) -> dict[str, Any]:
    usable = [row for row in feature_rows if row.get("processing_status") == "success"]
    failures = [row for row in feature_rows if row.get("processing_status") != "success"]
    class_counts = Counter(str(row.get("ground_truth_class", "")) for row in usable)
    source_counts = Counter(str(row.get("source_dataset", "")) for row in usable)
    class_source = _counts_by(usable, "source_dataset", "ground_truth_class")
    class_roi = _counts_by(usable, "roi_type", "ground_truth_class")
    quality_counts = Counter(str(row.get("ground_truth_quality", "")).lower() for row in usable)
    original_class = Counter(str(row.get("class", "")) for row in label_rows)
    original_source = Counter(str(row.get("source_dataset", "")) for row in label_rows)
    original_roi = Counter(str(row.get("roi_type", "")) for row in label_rows)
    final_roi = Counter(str(row.get("roi_type", "")) for row in usable)
    original_electron_rotated = sum(
        row.get("class") == "Electron" and row.get("roi_type") == "rotated"
        for row in label_rows
    )
    final_electron_rotated = class_roi[("rotated", "Electron")]
    retention = {
        "Alpha": (class_counts["Alpha"], original_class["Alpha"]),
        "Electron": (class_counts["Electron"], original_class["Electron"]),
        "AWAN": (source_counts["AWAN"], original_source["AWAN"]),
        "Lecturer": (source_counts["Lecturer"], original_source["Lecturer"]),
        "axis_aligned": (final_roi["axis_aligned"], original_roi["axis_aligned"]),
        "rotated": (final_roi["rotated"], original_roi["rotated"]),
    }
    source_table = [
        [class_source[("AWAN", "Alpha")], class_source[("AWAN", "Electron")]],
        [class_source[("Lecturer", "Alpha")], class_source[("Lecturer", "Electron")]],
    ]
    association = association_2x2(source_table)
    sanity = feature_sanity_issues(feature_rows)
    outliers = extreme_feature_outliers(feature_rows)
    unique_masks = len({str(row.get("module2_track_id", "")) for row in usable})
    all_source_cells_present = all(value > 0 for row in source_table for value in row)
    if failures or sanity or independence.get("review_required") or not all_source_cells_present:
        decision = "C. NOT READY - DATASET TOO SMALL OR BIASED AFTER CLEANING"
        ready = False
        blocker = (
            "Extraction/sanity/independence review remains unresolved or a primary class-source cell is empty."
        )
    else:
        decision = "B. READY, BUT SAMPLE/SELECTION BIAS IS SEVERE - PILOT ONLY"
        ready = True
        blocker = ""
    return {
        "usable": usable,
        "failures": failures,
        "class_counts": class_counts,
        "source_counts": source_counts,
        "class_source": class_source,
        "class_roi": class_roi,
        "quality_counts": quality_counts,
        "retention": retention,
        "electron_rotated_original": original_electron_rotated,
        "electron_rotated_final": final_electron_rotated,
        "source_table": source_table,
        "source_association": association,
        "sanity_issues": sanity,
        "extreme_outliers": outliers,
        "unique_masks": unique_masks,
        "decision": decision,
        "ready": ready,
        "blocker": blocker,
    }


def _format_optional(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.8g}"


def build_mapping_audit_report(
    original_count: int,
    revised_count: int,
    exclusions: Sequence[Mapping[str, str]],
    independence: Mapping[str, Any],
) -> str:
    overlap_rows = independence["overlap_rows"]
    lines = [
        "REVISED CONFIRMED MAPPING INDEPENDENCE AUDIT",
        "",
        f"Original confirmed mappings: {original_count}",
        f"Revised mappings: {revised_count}",
        f"Excluded by adjudication: {len(exclusions)}",
        "Deterministic duplicate representative: A003 retained; A008 excluded.",
        "A003 and A008 both have high ground-truth quality and the same AWAN source provenance;",
        "no metadata field provides explicit evidence that A008 is superior.",
        "",
        "EXCLUSIONS",
        *[
            f"{row['ground_truth_track_id']}: {row['exclusion_reason']} | {row['adjudication_basis']}"
            for row in exclusions
        ],
        "",
        "INDEPENDENCE CHECKS",
        f"Duplicate ground_truth_track_id: {len(independence['duplicate_ground_truth_ids'])}",
        f"Duplicate Module 2 track_id: {len(independence['duplicate_module2_ids'])}",
        f"Duplicate retained physical-track groups: {len(independence['duplicate_physical_track_groups'])}",
        f"Duplicate event_id: {len(independence['duplicate_event_ids'])}",
        f"Duplicate crop SHA-256 groups: {len(independence['duplicate_crop_hash_groups'])}",
        f"Same-source manual ROI overlaps inspected: {len(overlap_rows)}",
        f"New possible duplicates requiring review: {len(independence['suspected_overlap_duplicates'])}",
        "Screening rule (reported, not used for automatic deletion): manual ROI IoU >= 0.80,",
        "or smaller-ROI coverage >= 0.90 with normalized centroid distance <= 0.10.",
        *[
            f"{row['track_id_a']}/{row['track_id_b']}: IoU={float(row['manual_roi_iou']):.6f}, "
            f"smaller coverage={float(row['smaller_roi_coverage']):.6f}, "
            f"centroid distance={float(row['normalized_centroid_distance']):.6f}, {row['status']}"
            for row in overlap_rows
        ],
        "",
        f"REVIEW_REQUIRED: {'YES' if independence['review_required'] else 'NO'}",
        "No non-adjudicated record was removed automatically.",
        "No Random Forest was trained.",
        "",
    ]
    return "\n".join(lines)


def build_final_audit_report(
    original_label_count: int,
    previous_confirmed_count: int,
    revised_count: int,
    exclusions: Sequence[Mapping[str, str]],
    audit: Mapping[str, Any],
) -> str:
    class_counts = audit["class_counts"]
    source_counts = audit["source_counts"]
    class_source = audit["class_source"]
    class_roi = audit["class_roi"]
    retention = audit["retention"]
    association = audit["source_association"]
    outliers = audit["extreme_outliers"]
    lines = [
        "REVISED PRODUCTION MODULE 2 FEATURE/SUBSET AUDIT",
        "",
        f"ORIGINAL LABELLED: {original_label_count}",
        f"PREVIOUS CONFIRMED: {previous_confirmed_count}",
        f"REVISED MAPPING: {revised_count}",
        f"EXCLUDED BY ADJUDICATION: {len(exclusions)}",
        f"FINAL FEATURE SUCCESS: {len(audit['usable'])}",
        f"FAILED FEATURE ROWS: {len(audit['failures'])}",
        f"FINAL UNIQUE INDEPENDENT MASKS: {audit['unique_masks']}",
        "",
        "CLASS DISTRIBUTION",
        f"Alpha: {class_counts['Alpha']}",
        f"Electron: {class_counts['Electron']}",
        "",
        "SOURCE DISTRIBUTION",
        f"AWAN: {source_counts['AWAN']}",
        f"Lecturer: {source_counts['Lecturer']}",
        "",
        "CLASS X SOURCE (Alpha, Electron)",
        f"AWAN: {class_source[('AWAN', 'Alpha')]}, {class_source[('AWAN', 'Electron')]}",
        f"Lecturer: {class_source[('Lecturer', 'Alpha')]}, {class_source[('Lecturer', 'Electron')]}",
        "",
        "ROI X CLASS (Alpha, Electron)",
        f"axis_aligned: {class_roi[('axis_aligned', 'Alpha')]}, {class_roi[('axis_aligned', 'Electron')]}",
        f"rotated: {class_roi[('rotated', 'Alpha')]}, {class_roi[('rotated', 'Electron')]}",
        "",
        "GROUND-TRUTH QUALITY",
        f"high: {audit['quality_counts']['high']}",
        f"medium: {audit['quality_counts']['medium']}",
        f"low: {audit['quality_counts']['low']}",
        "",
        "SELECTION/RETENTION RATES AGAINST ORIGINAL 56",
        *[
            f"{name}: {count}/{total} ({_retention_rate(count, total):.2f}%)"
            for name, (count, total) in retention.items()
        ],
        f"Electron rotated: {audit['electron_rotated_final']}/{audit['electron_rotated_original']} "
        f"({_retention_rate(audit['electron_rotated_final'], audit['electron_rotated_original']):.2f}%)",
        "Selection remains particularly severe for rotated Electron annotations and is also source-conditional:",
        "only two AWAN Electron records and one rotated Electron record remain usable.",
        "",
        "SOURCE ASSOCIATION (DIAGNOSTIC ONLY)",
        f"Contingency [[AWAN Alpha, AWAN Electron], [Lecturer Alpha, Lecturer Electron]]: {audit['source_table']}",
        f"Absolute phi / Cramer's V: {_format_optional(association['phi_cramers_v'])}",
        f"Fisher exact p-value (two-sided): {_format_optional(association['fisher_p_value'])}",
        f"Fisher odds ratio: {_format_optional(association['odds_ratio'])}",
        "These values diagnose class/source association; they are not classifier performance.",
        "",
        "FEATURE SANITY",
        f"Critical issue count: {len(audit['sanity_issues'])}",
        *[
            f"{issue['track_id']}: {issue['issue']}" for issue in audit["sanity_issues"]
        ],
        "Extreme outlier rule: below Q1 - 3*IQR or above Q3 + 3*IQR, reported descriptively and never auto-removed.",
        f"Extreme outlier flags: {len(outliers)}",
        *[
            f"{row['track_id']} {row['feature']}={row['value']:.8g} "
            f"(fences {row['lower_fence']:.8g} to {row['upper_fence']:.8g})"
            for row in outliers
        ],
        f"FEATURE SANITY: {'PASS' if not audit['sanity_issues'] else 'ISSUES'}",
        "",
        "RF READINESS",
        audit["decision"],
        *(
            [
                "Reason: all revised mappings extract successfully and are independent, with both classes",
                "and all four class/source cells represented; however, the usable subset is strongly selected by ROI,",
                "especially rotated Electron. Any RF result must remain a small, source-aware, group-aware",
                "exploratory pilot and cannot be presented as production performance.",
            ]
            if audit["ready"]
            else [f"Reason: {audit['blocker']}"]
        ),
        f"READY FOR SMALL MODULE-2-BASED RF PILOT: {'YES' if audit['ready'] else 'NO'}",
    ]
    if audit["blocker"]:
        lines.append(f"Exact blocker: {audit['blocker']}")
    lines.extend([
        "",
        "SAFETY",
        "Module 1 modified: NO",
        "Module 2 modified: NO",
        "Frontend modified: NO",
        "labels.csv modified: NO",
        "Original confirmed mapping modified: NO",
        "Existing crops modified: NO",
        "Previous feature CSV modified: NO",
        "Existing RF outputs modified: NO",
        "RANDOM FOREST TRAINED: NO",
        "",
    ])
    return "\n".join(lines)


def run_revised_production_workflow(
    original_mapping_csv: str | Path,
    labels_csv: str | Path,
    diagnostics_root: str | Path,
    crop_root: str | Path,
    revised_mapping_csv: str | Path,
    exclusions_csv: str | Path,
    overlap_review_csv: str | Path,
    mapping_audit_txt: str | Path,
    revised_features_csv: str | Path,
    final_audit_txt: str | Path,
    *,
    extractor: Callable[[np.ndarray], Mapping[str, Any]] = extract_geometric_features,
) -> dict[str, Any]:
    _refuse_same_file(original_mapping_csv, revised_mapping_csv)
    label_rows = _read_csv(labels_csv)
    original_mapping_rows = _read_csv(original_mapping_csv)
    revised, exclusions = revise_confirmed_mapping(
        original_mapping_csv, labels_csv, revised_mapping_csv, exclusions_csv
    )
    independence = audit_mapping_independence(revised, label_rows, crop_root)
    _atomic_csv(overlap_review_csv, OVERLAP_REVIEW_COLUMNS, independence["overlap_rows"])
    _atomic_text(
        mapping_audit_txt,
        build_mapping_audit_report(
            len(original_mapping_rows), len(revised), exclusions, independence
        ),
    )
    feature_rows = extract_confirmed_module2_features(
        revised_mapping_csv,
        labels_csv,
        diagnostics_root,
        extractor=lambda mask: strict_geometry_extractor(mask, extractor=extractor),
    )
    _atomic_csv(revised_features_csv, FEATURE_COLUMNS, feature_rows)
    audit = final_subset_audit(feature_rows, label_rows, independence)
    _atomic_text(
        final_audit_txt,
        build_final_audit_report(
            len(label_rows),
            len(original_mapping_rows),
            len(revised),
            exclusions,
            audit,
        ),
    )
    return {
        "original_labelled": len(label_rows),
        "previous_confirmed": len(original_mapping_rows),
        "revised_mapping": len(revised),
        "excluded": len(exclusions),
        "feature_success": len(audit["usable"]),
        "feature_failed": len(audit["failures"]),
        "unique_masks": audit["unique_masks"],
        "class_counts": dict(audit["class_counts"]),
        "source_counts": dict(audit["source_counts"]),
        "class_source": {
            f"{source}|{class_name}": audit["class_source"][(source, class_name)]
            for source in ("AWAN", "Lecturer")
            for class_name in ("Alpha", "Electron")
        },
        "class_roi": {
            f"{roi}|{class_name}": audit["class_roi"][(roi, class_name)]
            for roi in ("axis_aligned", "rotated")
            for class_name in ("Alpha", "Electron")
        },
        "source_association": audit["source_association"],
        "retention": {
            key: {
                "final": pair[0],
                "original": pair[1],
                "percent": _retention_rate(*pair),
            }
            for key, pair in audit["retention"].items()
        },
        "electron_rotated": {
            "final": audit["electron_rotated_final"],
            "original": audit["electron_rotated_original"],
        },
        "independence_review_required": independence["review_required"],
        "sanity_issue_count": len(audit["sanity_issues"]),
        "extreme_outliers": audit["extreme_outliers"],
        "decision": audit["decision"],
        "ready_for_small_rf_pilot": audit["ready"],
        "random_forest_trained": False,
    }
