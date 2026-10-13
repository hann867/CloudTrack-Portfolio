"""Read-only audit of confirmed ground-truth to Module 2 mappings."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import fisher_exact


CLASSES = ("Alpha", "Electron")
SOURCES = ("AWAN", "Lecturer", "Other/blank")
ROI_TYPES = ("axis_aligned", "rotated")
QUALITY_LEVELS = ("high", "medium", "low")
METHODS = ("AUTO_HIGH", "MANUAL_REVIEW")

DISTRIBUTION_COLUMNS = ["distribution", "category", "class", "count", "denominator", "percentage"]
EXCLUDED_COLUMNS = [
    "track_id", "class", "source_dataset", "matcher_status", "manual_review_status", "reason"
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: str | Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path: str | Path, text: str) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
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


def source_group(value: object) -> str:
    text = str(value or "").strip()
    return text if text in {"AWAN", "Lecturer"} else "Other/blank"


def association_2x2(table: Sequence[Sequence[int]]) -> dict[str, float | None]:
    values = np.asarray(table, dtype=int)
    if values.shape != (2, 2):
        raise ValueError("Fisher/phi association requires a 2x2 table.")
    if values.sum() == 0 or np.any(values.sum(axis=0) == 0) or np.any(values.sum(axis=1) == 0):
        return {"odds_ratio": None, "fisher_p_value": None, "phi_cramers_v": None}
    odds, p_value = fisher_exact(values, alternative="two-sided")
    a, b, c, d = (float(value) for value in values.ravel())
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return {
        "odds_ratio": float(odds),
        "fisher_p_value": float(p_value),
        "phi_cramers_v": abs((a * d - b * c) / denominator) if denominator else None,
    }


def _percentage(count: int, denominator: int) -> float:
    return 100.0 * count / denominator if denominator else 0.0


def _distribution_row(distribution: str, category: str, class_name: str,
                      count: int, denominator: int) -> dict[str, Any]:
    return {"distribution": distribution, "category": category, "class": class_name,
            "count": count, "denominator": denominator, "percentage": _percentage(count, denominator)}


def _status_index(candidate_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    statuses: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        statuses[str(row.get("labelled_track_id", ""))].add(str(row.get("confidence_status", "")))
    invalid = {track: values for track, values in statuses.items() if len(values) != 1}
    if invalid:
        raise ValueError(f"Inconsistent matcher statuses: {invalid}")
    return {track: next(iter(values)) for track, values in statuses.items()}


def _reason_for_exclusion(track_id: str, matcher_status: str, manual_status: str) -> str:
    if manual_status == "NONE":
        return "MANUAL_NONE"
    if matcher_status == "NO_MATCH":
        return "NO_MATCH"
    if manual_status == "SKIPPED":
        return "REVIEW_SKIPPED"
    if matcher_status == "REVIEW" and not manual_status:
        return "REVIEW_UNRESOLVED"
    if manual_status == "CONFIRMED":
        return "CONFIRMED_DECISION_ABSENT_FROM_MAPPING"
    if matcher_status == "HIGH":
        return "AUTO_HIGH_ABSENT_FROM_MAPPING"
    return "UNRESOLVED_STATUS"


def audit_confirmed_mapping(
    labels_csv: str | Path,
    mapping_csv: str | Path,
    candidate_matching_csv: str | Path,
    manual_review_csv: str | Path,
    report_txt: str | Path,
    distribution_csv: str | Path,
    excluded_csv: str | Path,
) -> dict[str, Any]:
    labels = [row for row in _read_csv(labels_csv) if row.get("class") in CLASSES]
    mapping = _read_csv(mapping_csv)
    candidates = _read_csv(candidate_matching_csv)
    reviews = _read_csv(manual_review_csv)
    label_index = {row["track_id"]: row for row in labels}
    if len(label_index) != len(labels):
        raise ValueError("labels.csv contains duplicate valid track IDs.")
    mapped_ids = [row.get("ground_truth_track_id", "") for row in mapping]
    if len(set(mapped_ids)) != len(mapped_ids):
        raise ValueError("Confirmed mapping contains duplicate ground-truth track IDs.")
    for row in mapping:
        track_id = row.get("ground_truth_track_id", "")
        if track_id not in label_index:
            raise ValueError(f"Confirmed mapping references unknown track: {track_id}")
        label = label_index[track_id]
        if row.get("ground_truth_class") != label.get("class") or row.get("source_image") != label.get("source_image"):
            raise ValueError(f"Confirmed mapping provenance disagrees with labels.csv for {track_id}.")
        if row.get("match_method") not in METHODS:
            raise ValueError(f"Invalid match_method for {track_id}: {row.get('match_method')}")
    mapped_set = set(mapped_ids)
    confirmed = [label_index[track_id] for track_id in mapped_ids]

    class_counts = Counter(row["class"] for row in confirmed)
    source_counts = Counter(source_group(row.get("source_dataset")) for row in confirmed)
    class_source = {(source, cls): sum(source_group(row.get("source_dataset")) == source and row["class"] == cls
                                       for row in confirmed) for source in SOURCES for cls in CLASSES}
    method_class = {(method, cls): sum(row["match_method"] == method and row["ground_truth_class"] == cls
                                       for row in mapping) for method in METHODS for cls in CLASSES}
    roi_class = {(roi, cls): sum(row.get("roi_type") == roi and row["class"] == cls for row in confirmed)
                 for roi in ROI_TYPES for cls in CLASSES}
    quality_class = {(quality, cls): sum(str(row.get("ground_truth_quality", "")).strip().lower() == quality
                                         and row["class"] == cls for row in confirmed)
                     for quality in QUALITY_LEVELS for cls in CLASSES}

    matcher_status = _status_index(candidates)
    review_status = {row["ground_truth_track_id"]: row["review_status"] for row in reviews}
    excluded = []
    for row in labels:
        track_id = row["track_id"]
        if track_id in mapped_set:
            continue
        match_status = matcher_status.get(track_id, "")
        manual_status = review_status.get(track_id, "")
        excluded.append({
            "track_id": track_id, "class": row["class"],
            "source_dataset": source_group(row.get("source_dataset")),
            "matcher_status": match_status, "manual_review_status": manual_status,
            "reason": _reason_for_exclusion(track_id, match_status, manual_status),
        })

    distribution_rows: list[dict[str, Any]] = []
    total_confirmed = len(confirmed)
    for cls in CLASSES:
        distribution_rows.append(_distribution_row("CLASS", cls, "ALL", class_counts[cls], total_confirmed))
    for source in SOURCES:
        distribution_rows.append(_distribution_row("SOURCE", source, "ALL", source_counts[source], total_confirmed))
        for cls in CLASSES:
            distribution_rows.append(_distribution_row("CLASS_X_SOURCE", source, cls,
                                                        class_source[(source, cls)], class_counts[cls]))
    for method in METHODS:
        for cls in CLASSES:
            distribution_rows.append(_distribution_row("MATCH_METHOD", method, cls,
                                                        method_class[(method, cls)], class_counts[cls]))
    for roi in ROI_TYPES:
        for cls in CLASSES:
            distribution_rows.append(_distribution_row("ROI_TYPE", roi, cls,
                                                        roi_class[(roi, cls)], class_counts[cls]))
    for quality in QUALITY_LEVELS:
        for cls in CLASSES:
            distribution_rows.append(_distribution_row("GROUND_TRUTH_QUALITY", quality, cls,
                                                        quality_class[(quality, cls)], class_counts[cls]))

    all_class = Counter(row["class"] for row in labels)
    all_source = Counter(source_group(row.get("source_dataset")) for row in labels)
    all_roi = Counter(row.get("roi_type", "") for row in labels)
    confirmed_roi = Counter(row.get("roi_type", "") for row in confirmed)
    confirmation_class = {cls: (class_counts[cls], all_class[cls]) for cls in CLASSES}
    confirmation_source = {source: (source_counts[source], all_source[source]) for source in SOURCES}
    confirmation_roi = {roi: (confirmed_roi[roi], all_roi[roi]) for roi in ROI_TYPES}
    confirmation_class_roi = {
        (cls, roi): (
            sum(row["class"] == cls and row.get("roi_type") == roi for row in confirmed),
            sum(row["class"] == cls and row.get("roi_type") == roi for row in labels),
        ) for cls in CLASSES for roi in ROI_TYPES
    }
    for cls, (count, total) in confirmation_class.items():
        distribution_rows.append(_distribution_row("CONFIRMATION_RATE_CLASS", cls, "ALL", count, total))
    for source, (count, total) in confirmation_source.items():
        distribution_rows.append(_distribution_row("CONFIRMATION_RATE_SOURCE", source, "ALL", count, total))
    for roi, (count, total) in confirmation_roi.items():
        distribution_rows.append(_distribution_row("CONFIRMATION_RATE_ROI", roi, "ALL", count, total))
    for (cls, roi), (count, total) in confirmation_class_roi.items():
        distribution_rows.append(_distribution_row("CONFIRMATION_RATE_CLASS_X_ROI", roi, cls, count, total))

    source_table = [[class_source[("AWAN", cls)] for cls in CLASSES],
                    [class_source[("Lecturer", cls)] for cls in CLASSES]]
    source_association = association_2x2(source_table)
    selection_class_table = [[class_counts["Alpha"], all_class["Alpha"] - class_counts["Alpha"]],
                             [class_counts["Electron"], all_class["Electron"] - class_counts["Electron"]]]
    selection_source_table = [[source_counts["AWAN"], all_source["AWAN"] - source_counts["AWAN"]],
                              [source_counts["Lecturer"], all_source["Lecturer"] - source_counts["Lecturer"]]]
    selection_roi_table = [[confirmed_roi["axis_aligned"], all_roi["axis_aligned"] - confirmed_roi["axis_aligned"]],
                           [confirmed_roi["rotated"], all_roi["rotated"] - confirmed_roi["rotated"]]]
    selection_associations = {
        "class": association_2x2(selection_class_table),
        "source": association_2x2(selection_source_table),
        "roi": association_2x2(selection_roi_table),
    }
    selection_class_roi_associations = {}
    for cls in CLASSES:
        axis_pair = confirmation_class_roi[(cls, "axis_aligned")]
        rotated_pair = confirmation_class_roi[(cls, "rotated")]
        selection_class_roi_associations[cls] = association_2x2([
            [axis_pair[0], axis_pair[1] - axis_pair[0]],
            [rotated_pair[0], rotated_pair[1] - rotated_pair[0]],
        ])

    def rate(pair: tuple[int, int]) -> float:
        return _percentage(*pair)

    class_gap = abs(rate(confirmation_class["Alpha"]) - rate(confirmation_class["Electron"]))
    source_gap = abs(rate(confirmation_source["AWAN"]) - rate(confirmation_source["Lecturer"]))
    roi_gap = abs(rate(confirmation_roi["axis_aligned"]) - rate(confirmation_roi["rotated"]))
    electron_rotated = confirmation_class_roi[("Electron", "rotated")]
    electron_axis = confirmation_class_roi[("Electron", "axis_aligned")]
    all_primary_cells_present = all(class_source[(source, cls)] > 0
                                    for source in ("AWAN", "Lecturer") for cls in CLASSES)
    both_classes_present = all(class_counts[cls] > 0 for cls in CLASSES)
    if not both_classes_present or not all_primary_cells_present:
        recommendation = "C. DO NOT PROCEED — CONFIRMED SUBSET IS TOO IMBALANCED/BIASED"
        recommendation_reason = (
            "At least one class or primary class/source cell is absent from the confirmed subset, so the next RF "
            "could not separate particle class from source coverage."
        )
    elif class_gap == 0 and source_gap == 0 and roi_gap == 0:
        recommendation = "A. PROCEED TO MODULE-2-BASED FEATURE EXTRACTION"
        recommendation_reason = "Both classes and all primary class/source cells are represented without differential confirmation by class, source, or ROI type."
    else:
        recommendation = "B. PROCEED, BUT TREAT THE NEXT RF AS A SMALL EXPLORATORY PILOT"
        recommendation_reason = (
            "Both classes, both primary sources, and all four class/source cells remain represented, so Module-2-based "
            "feature extraction is informative. However, confirmation varies by class and/or ROI type; any RF result "
            "must be described as conditional on Module 2's usable-candidate subset, with grouped/source-aware validation "
            "and no production-performance claim."
        )

    lines = [
        "CONFIRMED GROUND TRUTH <-> MODULE 2 MAPPING AUDIT",
        "",
        f"TOTAL CONFIRMED MAPPINGS: {total_confirmed}",
        "",
        "CLASS DISTRIBUTION",
        f"Alpha: {class_counts['Alpha']} ({_percentage(class_counts['Alpha'], total_confirmed):.2f}%)",
        f"Electron: {class_counts['Electron']} ({_percentage(class_counts['Electron'], total_confirmed):.2f}%)",
        "",
        "SOURCE DISTRIBUTION",
        *[f"{source}: {source_counts[source]} ({_percentage(source_counts[source], total_confirmed):.2f}%)" for source in SOURCES],
        "",
        "CLASS X SOURCE (Alpha, Electron)",
        *[f"{source}: {class_source[(source, 'Alpha')]}, {class_source[(source, 'Electron')]}" for source in SOURCES],
        "",
        "MATCH METHOD BY CLASS (Alpha, Electron)",
        *[f"{method}: {method_class[(method, 'Alpha')]}, {method_class[(method, 'Electron')]}" for method in METHODS],
        "",
        "ROI TYPE BY CLASS (Alpha, Electron)",
        *[f"{roi}: {roi_class[(roi, 'Alpha')]}, {roi_class[(roi, 'Electron')]}" for roi in ROI_TYPES],
        "",
        "GROUND-TRUTH QUALITY BY CLASS (Alpha, Electron)",
        *[f"{quality}: {quality_class[(quality, 'Alpha')]}, {quality_class[(quality, 'Electron')]}" for quality in QUALITY_LEVELS],
        "",
        "CONFIRMATION RATES",
        *[f"{cls}: {confirmation_class[cls][0]} / {confirmation_class[cls][1]} ({rate(confirmation_class[cls]):.2f}%)" for cls in CLASSES],
        *[f"{source}: {confirmation_source[source][0]} / {confirmation_source[source][1]} ({rate(confirmation_source[source]):.2f}%)" for source in ("AWAN", "Lecturer")],
        *[f"{roi}: {confirmation_roi[roi][0]} / {confirmation_roi[roi][1]} ({rate(confirmation_roi[roi]):.2f}%)" for roi in ROI_TYPES],
        "Class x ROI confirmation:",
        *[f"{cls} {roi}: {confirmation_class_roi[(cls, roi)][0]} / {confirmation_class_roi[(cls, roi)][1]} ({rate(confirmation_class_roi[(cls, roi)]):.2f}%)"
          for cls in CLASSES for roi in ROI_TYPES],
        "",
        "CONFIRMED CLASS X SOURCE ASSOCIATION",
        f"Contingency [[AWAN Alpha, AWAN Electron], [Lecturer Alpha, Lecturer Electron]]: {source_table}",
        f"Absolute phi / Cramer's V: {source_association['phi_cramers_v']:.6f}",
        f"Fisher exact p-value (two-sided): {source_association['fisher_p_value']:.6g}",
        f"Fisher odds ratio: {source_association['odds_ratio']:.6g}",
        "The Fisher p-value measures class/source association in this subset; it is not classifier performance.",
        "",
        "EXCLUSIONS",
        f"Total excluded: {len(excluded)}",
        f"MANUAL_NONE: {sum(row['reason'] == 'MANUAL_NONE' for row in excluded)}",
        f"NO_MATCH: {sum(row['reason'] == 'NO_MATCH' for row in excluded)}",
        *[f"{row['track_id']} | {row['class']} | {row['source_dataset']} | matcher={row['matcher_status']} | manual={row['manual_review_status'] or 'blank'} | {row['reason']}"
          for row in excluded],
        "",
        "SELECTION-BIAS ASSESSMENT",
        f"Class confirmation gap: {class_gap:.2f} percentage points.",
        f"Source confirmation gap: {source_gap:.2f} percentage points.",
        f"ROI confirmation gap: {roi_gap:.2f} percentage points.",
        f"All four AWAN/Lecturer x Alpha/Electron cells represented: {'YES' if all_primary_cells_present else 'NO'}.",
        f"Rotated Electron ROI: {electron_rotated[0]}/{electron_rotated[1]} confirmed ({rate(electron_rotated):.2f}%).",
        f"Axis-aligned Electron ROI: {electron_axis[0]}/{electron_axis[1]} confirmed ({rate(electron_axis):.2f}%).",
        "Selection-outcome association diagnostics (confirmed vs excluded):",
        f"Class (Alpha vs Electron): |phi|={selection_associations['class']['phi_cramers_v']:.6f}, Fisher p={selection_associations['class']['fisher_p_value']:.6g}.",
        f"Source (AWAN vs Lecturer): |phi|={selection_associations['source']['phi_cramers_v']:.6f}, Fisher p={selection_associations['source']['fisher_p_value']:.6g}.",
        f"ROI (axis-aligned vs rotated): |phi|={selection_associations['roi']['phi_cramers_v']:.6f}, Fisher p={selection_associations['roi']['fisher_p_value']:.6g}.",
        f"Alpha ROI selection: |phi|={selection_class_roi_associations['Alpha']['phi_cramers_v']:.6f}, Fisher p={selection_class_roi_associations['Alpha']['fisher_p_value']:.6g}.",
        f"Electron ROI selection: |phi|={selection_class_roi_associations['Electron']['phi_cramers_v']:.6f}, Fisher p={selection_class_roi_associations['Electron']['fisher_p_value']:.6g}.",
        "These selection p-values diagnose mapping inclusion patterns; they are not classifier performance metrics.",
        "The observed confirmation-rate differences are selection effects and must be retained as limitations of the confirmed subset.",
        "",
        "FINAL RECOMMENDATION",
        recommendation,
        f"Reason: {recommendation_reason}",
        "No geometric features were regenerated and no Random Forest was trained.",
    ]
    _atomic_csv(distribution_csv, DISTRIBUTION_COLUMNS, distribution_rows)
    _atomic_csv(excluded_csv, EXCLUDED_COLUMNS, excluded)
    _atomic_text(report_txt, "\n".join(lines) + "\n")
    return {
        "total_confirmed": total_confirmed,
        "class_counts": dict(class_counts),
        "source_counts": dict(source_counts),
        "class_source": class_source,
        "method_class": method_class,
        "roi_class": roi_class,
        "quality_class": quality_class,
        "excluded": excluded,
        "confirmation_class": confirmation_class,
        "confirmation_source": confirmation_source,
        "confirmation_roi": confirmation_roi,
        "confirmation_class_roi": confirmation_class_roi,
        "source_association": source_association,
        "selection_associations": selection_associations,
        "selection_class_roi_associations": selection_class_roi_associations,
        "recommendation": recommendation,
    }
