"""Extract Module 3 geometry directly from confirmed production Module 2 masks."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .feature_extractor import extract_geometric_features


FEATURES = [
    "track_length_px", "mean_width_px", "median_width_px", "max_width_px",
    "mean_curvature", "max_curvature", "tortuosity", "endpoint_distance_px",
    "mask_area_px", "skeleton_pixels",
]
PRIORITY_FEATURES = {"mean_width_px", "track_length_px", "mean_curvature", "tortuosity"}
OUTPUT_COLUMNS = [
    "ground_truth_track_id", "ground_truth_class", "source_dataset", "source_image",
    "event_id", "roi_type", "ground_truth_quality", "match_method", "module2_track_id",
    "module2_component_id", "module2_label_id", "segmentation_source", "module2_mask_file",
    "processing_status", "error_message", *FEATURES,
]
COMPARISON_COLUMNS = [
    "class", "feature", "paired_count", "production_mean", "production_median",
    "temporary_mean", "temporary_median", "mean_difference_production_minus_temporary",
    "median_difference_production_minus_temporary", "relative_mean_shift",
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


def load_module2_mask_index(diagnostics_root: str | Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for manifest in sorted(Path(diagnostics_root).rglob("manifest.csv"), key=lambda path: path.as_posix().casefold()):
        for row in _read_csv(manifest):
            track_id = str(row.get("track_id", "")).strip()
            if not track_id:
                continue
            if track_id in index:
                raise ValueError(f"Duplicate Module 2 track_id across diagnostic manifests: {track_id}")
            relative = str(row.get("diagnostic_mask_file", "")).strip()
            indexed = dict(row)
            indexed["mask_path"] = manifest.parent / relative if relative else None
            indexed["manifest_path"] = manifest
            index[track_id] = indexed
    return index


def _decode_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def feature_sanity_issues(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    ground_ids = [str(row.get("ground_truth_track_id", "")) for row in rows]
    module2_ids = [str(row.get("module2_track_id", "")) for row in rows]
    for name, values in (("ground_truth_track_id", ground_ids), ("module2_track_id", module2_ids)):
        duplicates = sorted(value for value, count in Counter(values).items() if value and count > 1)
        for value in duplicates:
            issues.append({"track_id": value, "issue": f"duplicate {name}"})
    for row in rows:
        track_id = str(row.get("ground_truth_track_id", ""))
        if row.get("processing_status") != "success":
            message = str(row.get("error_message", "") or "extraction failure")
            issue = "missing Module 2 mask" if "mask" in message.lower() else "extraction failure"
            issues.append({"track_id": track_id, "issue": f"{issue}: {message}"})
            continue
        numeric = {}
        for feature in FEATURES:
            try:
                numeric[feature] = float(row[feature])
            except (TypeError, ValueError):
                issues.append({"track_id": track_id, "issue": f"NaN/non-numeric {feature}"})
                continue
            if not math.isfinite(numeric[feature]):
                issues.append({"track_id": track_id, "issue": f"NaN/infinity {feature}"})
        if numeric.get("track_length_px", 0) <= 0:
            issues.append({"track_id": track_id, "issue": "invalid length"})
        widths = [numeric.get(name, 0) for name in ("mean_width_px", "median_width_px", "max_width_px")]
        if any(width <= 0 for width in widths) or (len(widths) == 3 and max(widths[:2]) > widths[2]):
            issues.append({"track_id": track_id, "issue": "invalid width"})
        if "tortuosity" in numeric and numeric["tortuosity"] < 1.0:
            issues.append({"track_id": track_id, "issue": "tortuosity < 1"})
    return issues


def extract_confirmed_module2_features(
    mapping_csv: str | Path,
    labels_csv: str | Path,
    diagnostics_root: str | Path,
    *,
    extractor: Callable[[np.ndarray], Mapping[str, Any]] = extract_geometric_features,
) -> list[dict[str, Any]]:
    mapping = _read_csv(mapping_csv)
    labels = {row["track_id"]: row for row in _read_csv(labels_csv)}
    ground_ids = [row.get("ground_truth_track_id", "") for row in mapping]
    if len(set(ground_ids)) != len(ground_ids):
        raise ValueError("Duplicate ground_truth_track_id in confirmed mapping.")
    mask_index = load_module2_mask_index(diagnostics_root)
    output = []
    for match in mapping:
        ground_id = match["ground_truth_track_id"]
        if ground_id not in labels:
            raise ValueError(f"Confirmed mapping references unknown ground-truth track: {ground_id}")
        label = labels[ground_id]
        if match.get("ground_truth_class") != label.get("class") or match.get("source_image") != label.get("source_image"):
            raise ValueError(f"Confirmed mapping provenance disagrees with labels.csv for {ground_id}.")
        row = {column: "" for column in OUTPUT_COLUMNS}
        row.update({
            "ground_truth_track_id": ground_id, "ground_truth_class": label.get("class", ""),
            "source_dataset": label.get("source_dataset", ""), "source_image": label.get("source_image", ""),
            "event_id": label.get("event_id", ""), "roi_type": label.get("roi_type", ""),
            "ground_truth_quality": label.get("ground_truth_quality", ""),
            "match_method": match.get("match_method", ""), "module2_track_id": match.get("module2_track_id", ""),
            "segmentation_source": "production_module2", "processing_status": "failed",
        })
        diagnostic = mask_index.get(str(match.get("module2_track_id", "")))
        if diagnostic is None:
            row["error_message"] = "Confirmed Module 2 track is absent from diagnostic manifests."
            output.append(row); continue
        row["module2_component_id"] = diagnostic.get("component_id", "")
        row["module2_label_id"] = diagnostic.get("label_id", "")
        mask_path = diagnostic.get("mask_path")
        row["module2_mask_file"] = str(Path(mask_path).resolve()) if mask_path else ""
        if diagnostic.get("source_filename") != label.get("source_image"):
            row["error_message"] = "Diagnostic source_filename does not match ground-truth source_image."
            output.append(row); continue
        mask = _decode_mask(Path(mask_path)) if mask_path else None
        if mask is None:
            row["error_message"] = "Module 2 diagnostic mask is missing or unreadable."
            output.append(row); continue
        try:
            result = dict(extractor(mask))
        except Exception as exc:
            row["error_message"] = f"Feature extraction raised {type(exc).__name__}: {exc}"
            output.append(row); continue
        if result.get("processing_status") != "success":
            row["error_message"] = str(result.get("error_message") or "Feature extraction failed.")
            output.append(row); continue
        row["processing_status"] = "success"; row["error_message"] = ""
        for feature in FEATURES:
            row[feature] = result.get(feature, "")
        output.append(row)
    return output


def compare_with_temporary(production_rows: Sequence[Mapping[str, Any]], temporary_csv: str | Path) -> list[dict[str, Any]]:
    temporary = {row["track_id"]: row for row in _read_csv(temporary_csv)
                 if row.get("processing_status") == "success"}
    comparisons = []
    for class_name in ("Alpha", "Electron"):
        selected = [row for row in production_rows if row.get("processing_status") == "success"
                    and row.get("ground_truth_class") == class_name and row.get("ground_truth_track_id") in temporary]
        for feature in FEATURES:
            paired = []
            for row in selected:
                try:
                    production_value = float(row[feature])
                    temporary_value = float(temporary[row["ground_truth_track_id"]][feature])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(production_value) and math.isfinite(temporary_value):
                    paired.append((production_value, temporary_value))
            production_values = np.asarray([value[0] for value in paired], dtype=float)
            temporary_values = np.asarray([value[1] for value in paired], dtype=float)
            production_mean = float(production_values.mean()) if len(paired) else math.nan
            temporary_mean = float(temporary_values.mean()) if len(paired) else math.nan
            production_median = float(np.median(production_values)) if len(paired) else math.nan
            temporary_median = float(np.median(temporary_values)) if len(paired) else math.nan
            difference = production_mean - temporary_mean
            comparisons.append({
                "class": class_name, "feature": feature, "paired_count": len(paired),
                "production_mean": production_mean, "production_median": production_median,
                "temporary_mean": temporary_mean, "temporary_median": temporary_median,
                "mean_difference_production_minus_temporary": difference,
                "median_difference_production_minus_temporary": production_median - temporary_median,
                "relative_mean_shift": abs(difference) / abs(temporary_mean) if temporary_mean else math.inf,
            })
    return comparisons


def classify_feature_shift(comparisons: Sequence[Mapping[str, Any]]) -> str:
    shifts = [float(row["relative_mean_shift"]) for row in comparisons
              if row.get("feature") in PRIORITY_FEATURES and math.isfinite(float(row["relative_mean_shift"]))]
    if not shifts:
        return "UNAVAILABLE"
    if sum(shift >= 1.0 for shift in shifts) >= 2 or max(shifts) >= 2.0:
        return "HIGH"
    if max(shifts) >= 0.25:
        return "MODERATE"
    return "LOW"


def run_production_feature_extraction(
    mapping_csv: str | Path, labels_csv: str | Path, diagnostics_root: str | Path,
    temporary_features_csv: str | Path, output_csv: str | Path, comparison_csv: str | Path,
    audit_txt: str | Path, *, extractor: Callable[[np.ndarray], Mapping[str, Any]] = extract_geometric_features,
) -> dict[str, Any]:
    rows = extract_confirmed_module2_features(mapping_csv, labels_csv, diagnostics_root, extractor=extractor)
    issues = feature_sanity_issues(rows)
    comparisons = compare_with_temporary(rows, temporary_features_csv)
    shift = classify_feature_shift(comparisons)
    _atomic_csv(output_csv, OUTPUT_COLUMNS, rows)
    _atomic_csv(comparison_csv, COMPARISON_COLUMNS, comparisons)
    successes = [row for row in rows if row["processing_status"] == "success"]
    class_counts = Counter(row["ground_truth_class"] for row in successes)
    source_counts = Counter(row["source_dataset"] for row in successes)
    ready = len(successes) == len(rows) and not issues
    lines = [
        "PRODUCTION MODULE 2 GEOMETRIC FEATURE EXTRACTION AUDIT", "",
        f"EXPECTED CONFIRMED: {len(rows)}", f"SUCCESSFUL: {len(successes)}",
        f"FAILED: {len(rows) - len(successes)}", "",
        f"Alpha success: {class_counts['Alpha']}", f"Electron success: {class_counts['Electron']}",
        f"AWAN success: {source_counts['AWAN']}", f"Lecturer success: {source_counts['Lecturer']}", "",
        f"FEATURE SANITY: {'PASS' if not issues else 'ISSUES'}",
        f"Duplicate ground_truth_track_id: {sum('duplicate ground_truth_track_id' in issue['issue'] for issue in issues)}",
        f"Duplicate module2_track_id: {sum('duplicate module2_track_id' in issue['issue'] for issue in issues)}",
        f"Sanity issue count: {len(issues)}",
        *[f"{issue['track_id']}: {issue['issue']}" for issue in issues], "",
        "SEGMENTATION SOURCE: production_module2", "No millimetre values were created.", "",
        "TEMPORARY VS PRODUCTION FEATURE DISTRIBUTIONS",
        "Differences are production minus temporary. Relative shift is abs(mean difference)/abs(temporary mean).",
        *[f"{row['class']} {row['feature']}: n={row['paired_count']}, production mean/median={row['production_mean']:.6g}/{row['production_median']:.6g}, "
          f"temporary mean/median={row['temporary_mean']:.6g}/{row['temporary_median']:.6g}, "
          f"mean/median difference={row['mean_difference_production_minus_temporary']:.6g}/{row['median_difference_production_minus_temporary']:.6g}"
          for row in comparisons], "",
        f"TEMPORARY VS PRODUCTION FEATURE SHIFT: {shift}",
        "Shift flag rule (reported, descriptive only): LOW if all four priority-feature relative mean shifts are <25%; "
        "MODERATE if any is >=25%; HIGH if at least two are >=100% or any is >=200%.",
        "Feature shifts reflect segmentation changes and are not classifier performance.", "",
        f"READY FOR MODULE-2-BASED RF PILOT: {'YES' if ready else 'NO'}",
    ]
    if ready:
        lines.extend([
            "This remains a small exploratory pilot because n is small, Electron count is lower, rotated Electron has only one confirmed sample, and Module 2 candidate usability introduces selection bias.",
            "No Random Forest was trained in this task.",
        ])
    else:
        lines.append("Resolve the extraction or feature-sanity issues before an RF pilot.")
    _atomic_text(audit_txt, "\n".join(lines) + "\n")
    return {"expected": len(rows), "successful": len(successes), "failed": len(rows)-len(successes),
            "class_counts": dict(class_counts), "source_counts": dict(source_counts), "issues": issues,
            "comparisons": comparisons, "feature_shift": shift, "ready_for_rf_pilot": ready}
