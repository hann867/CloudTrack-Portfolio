"""Generate audited Module 3 geometry from human-approved production Module 2 masks."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .feature_extractor import extract_geometric_features


FEATURES = [
    "track_length_px", "mean_width_px", "median_width_px", "max_width_px",
    "mean_curvature", "max_curvature", "tortuosity",
]
FEATURE_COLUMNS = [
    "ground_truth_track_id", "ground_truth_class", "source_dataset", "source_image",
    "source_video", "frame_number", "timestamp_seconds", "event_id", "roi_type",
    "physical_track_group_id", "dependency_group_id", "module2_event_id",
    "module2_component_id", "module2_label_id", "selected_candidate", "module2_mask_file",
    "segmentation_source", "mapping_decision", "mask_quality_decision", "reviewed_at",
    "review_method", "human_review_notes", "processing_status", "error_message", *FEATURES,
]
FAILURE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in FEATURES]
FLAG_COLUMNS = ["ground_truth_track_id", "flag_type", "feature", "value", "threshold", "severity", "notes"]
INDEPENDENCE_COLUMNS = [
    "ground_truth_track_id", "ground_truth_class", "source_dataset", "source_image", "source_video",
    "event_id", "physical_track_group_id", "module2_mask_id", "dependency_group_id",
    "dependency_group_size", "source_image_group_size", "source_video_group_size",
    "duplicate_ground_truth_id", "duplicate_module2_mask_id", "excluded_physical_duplicate_present",
    "processing_status", "independence_status",
]
DIST_COLUMNS = ["dataset", "dimension", "category_1", "category_2", "count"]


def read_csv(path: str | Path) -> list[dict[str, str]]:
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
        try: os.unlink(temporary)
        except OSError: pass
        raise


def _atomic_text(path: str | Path, text: str) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle: handle.write(text)
        os.replace(temporary, output)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def _unique_index(rows: Sequence[Mapping[str, str]], key: str) -> dict[str, dict[str, str]]:
    output = {}
    for raw in rows:
        value = str(raw.get(key, "")).strip()
        if not value or value in output:
            raise ValueError(f"Missing or duplicate {key}: {value or '<blank>'}")
        output[value] = dict(raw)
    return output


def authorized_history(history_csv: str | Path) -> list[dict[str, str]]:
    rows = read_csv(history_csv)
    if len(rows) != 114 or len({row["track_id"] for row in rows}) != 114:
        raise ValueError("Human adjudication must contain exactly 114 unique independent tracks.")
    mapping = Counter(row["mapping_decision"] for row in rows)
    if mapping != Counter({"CONFIRMED": 108, "REJECTED": 6}):
        raise ValueError(f"Human mapping adjudication is not at the verified completed state: {dict(mapping)}")
    confirmed = [row for row in rows if row["mapping_decision"] == "CONFIRMED"]
    mask = Counter(row["mask_quality_decision"] for row in confirmed)
    if mask != Counter({"PASS": 35, "FAIL": 73}):
        raise ValueError(f"Confirmed-mask adjudication is not complete or expected: {dict(mask)}")
    selected = [row for row in confirmed if row["mask_quality_decision"] == "PASS"]
    if len(selected) != 35:
        raise ValueError("Expected exactly 35 human-authorized Module 3 inputs.")
    counts = Counter(row["ground_truth_class"] for row in selected)
    sources = Counter(row["source_dataset"] for row in selected)
    if counts != Counter({"Alpha": 16, "Electron": 19}) or sources != Counter({"AWAN": 7, "Lecturer": 28}):
        raise ValueError(f"Authorized class/source counts disagree with verified state: {dict(counts)} / {dict(sources)}")
    if {"A018", "A027"} & {row["track_id"] for row in selected}:
        raise ValueError("Excluded physical duplicate observations entered the authorized subset.")
    return selected


def _manifest_index(batch_manifest_csv: str | Path) -> dict[str, dict[str, Any]]:
    batch = read_csv(batch_manifest_csv)
    paths = sorted({Path(row["diagnostic_manifest_path"]) for row in batch}, key=lambda p: p.as_posix().casefold())
    output = {}
    for manifest in paths:
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing Module 2 manifest: {manifest}")
        for raw in read_csv(manifest):
            track_id = raw.get("track_id", "")
            if track_id in output:
                raise ValueError(f"Duplicate Module 2 mask identity: {track_id}")
            row: dict[str, Any] = dict(raw)
            row["mask_path"] = manifest.parent / raw.get("diagnostic_mask_file", "")
            row["manifest_path"] = manifest
            output[track_id] = row
    return output


def _dependency_groups(labels: Sequence[Mapping[str, str]]) -> dict[str, str]:
    parent = {row["track_id"]: row["track_id"] for row in labels}
    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]; item = parent[item]
        return item
    def union(first: str, second: str) -> None:
        a, b = find(first), find(second)
        if a != b: parent[max(a, b)] = min(a, b)
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in labels:
        event = row.get("event_id", "").strip()
        if event and set(event) != {"+"}:
            buckets[("event", event)].append(row["track_id"])
        buckets[("source_image", row.get("source_image", ""))].append(row["track_id"])
        group = row.get("physical_track_group_id", "").strip()
        if group: buckets[("physical", group)].append(row["track_id"])
    for members in buckets.values():
        for member in members[1:]: union(members[0], member)
    components: dict[str, list[str]] = defaultdict(list)
    for track_id in parent: components[find(track_id)].append(track_id)
    return {member: f"DEP_{min(members)}" for members in components.values() for member in members}


def extract_authorized_features(history_csv: str | Path, labels_csv: str | Path,
                                batch_manifest_csv: str | Path) -> list[dict[str, Any]]:
    authorized = authorized_history(history_csv)
    labels = _unique_index(read_csv(labels_csv), "track_id")
    batch = _unique_index(read_csv(batch_manifest_csv), "ground_truth_track_id")
    enriched = []
    for review in authorized:
        track_id = review["track_id"]
        label, manifest = labels[track_id], batch[track_id]
        if review["ground_truth_class"] != label["class"] or review["source"] != label["source_image"]:
            raise ValueError(f"Human adjudication provenance disagrees with independent labels for {track_id}.")
        enriched.append({**label, **review, **{
            "source_image": label["source_image"], "physical_track_group_id": manifest.get("physical_track_group_id", ""),
        }})
    dependency = _dependency_groups(enriched)
    masks = _manifest_index(batch_manifest_csv)
    output = []
    for item in enriched:
        track_id, candidate_id = item["track_id"], item["selected_candidate"]
        row = {column: "" for column in FEATURE_COLUMNS}
        row.update({
            "ground_truth_track_id": track_id, "ground_truth_class": item["class"],
            "source_dataset": item["source_dataset"], "source_image": item["source_image"],
            "source_video": item.get("source_video", ""), "frame_number": item.get("frame_number", ""),
            "timestamp_seconds": item.get("timestamp_seconds", ""), "event_id": item.get("event_id", ""),
            "roi_type": item.get("roi_type", ""), "physical_track_group_id": item.get("physical_track_group_id", ""),
            "dependency_group_id": dependency[track_id], "module2_event_id": item.get("module2_event_id", ""),
            "module2_component_id": item.get("module2_component_id", ""), "selected_candidate": candidate_id,
            "segmentation_source": "production_module2_human_selected", "mapping_decision": item["mapping_decision"],
            "mask_quality_decision": item["mask_quality_decision"], "reviewed_at": item.get("reviewed_at", ""),
            "review_method": item.get("review_method", ""), "human_review_notes": item.get("notes", ""),
            "processing_status": "FAILURE",
        })
        diagnostic = masks.get(candidate_id)
        if diagnostic is None:
            row["error_message"] = "Human-selected Module 2 component is absent from diagnostic manifests."
            output.append(row); continue
        row["module2_label_id"] = diagnostic.get("label_id", "")
        row["module2_component_id"] = diagnostic.get("component_id", row["module2_component_id"])
        mask_path = Path(diagnostic["mask_path"])
        row["module2_mask_file"] = str(mask_path.resolve())
        if diagnostic.get("source_filename") != item["source_image"]:
            row["error_message"] = "Selected Module 2 mask source does not match ground-truth source image."
            output.append(row); continue
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        if mask is None:
            row["error_message"] = "Human-selected Module 2 mask is missing or unreadable."
            output.append(row); continue
        try:
            result = dict(extract_geometric_features(mask))
        except Exception as exc:
            row["error_message"] = f"Production feature extractor raised {type(exc).__name__}: {exc}"
            output.append(row); continue
        if result.get("processing_status") != "success":
            row["error_message"] = str(result.get("error_message") or "Production feature extraction failed.")
            output.append(row); continue
        row["processing_status"] = "SUCCESS"
        for feature in FEATURES: row[feature] = result.get(feature, "")
        output.append(row)
    return output


def feature_flags(successes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    numeric: dict[str, dict[str, float]] = {}
    for row in successes:
        track_id = row["ground_truth_track_id"]; numeric[track_id] = {}
        for feature in FEATURES:
            try: value = float(row[feature])
            except (TypeError, ValueError): value = math.nan
            numeric[track_id][feature] = value
            if not math.isfinite(value):
                flags.append({"ground_truth_track_id": track_id, "flag_type": "NONFINITE", "feature": feature,
                              "value": value, "threshold": "finite", "severity": "INVALID", "notes": "NaN/infinity/non-numeric"})
        values = numeric[track_id]
        for feature in ("track_length_px", "mean_width_px", "median_width_px", "max_width_px"):
            if math.isfinite(values[feature]) and values[feature] <= 0:
                flags.append({"ground_truth_track_id": track_id, "flag_type": "NON_POSITIVE", "feature": feature,
                              "value": values[feature], "threshold": "> 0", "severity": "INVALID", "notes": "Geometry must be positive."})
        if math.isfinite(values["tortuosity"]) and values["tortuosity"] < 1:
            flags.append({"ground_truth_track_id": track_id, "flag_type": "TORTUOSITY_LT_ONE", "feature": "tortuosity",
                          "value": values["tortuosity"], "threshold": ">= 1", "severity": "INVALID", "notes": "Geometrically impossible."})
        if all(math.isfinite(values[f]) for f in ("mean_width_px", "median_width_px", "max_width_px")) and max(values["mean_width_px"], values["median_width_px"]) > values["max_width_px"]:
            flags.append({"ground_truth_track_id": track_id, "flag_type": "WIDTH_ORDER", "feature": "width",
                          "value": "", "threshold": "mean/median <= max", "severity": "INVALID", "notes": "Width summary ordering is inconsistent."})
        if math.isfinite(values["max_width_px"]) and math.isfinite(values["track_length_px"]) and values["max_width_px"] > values["track_length_px"]:
            flags.append({"ground_truth_track_id": track_id, "flag_type": "WIDTH_EXCEEDS_LENGTH", "feature": "max_width_px",
                          "value": values["max_width_px"], "threshold": values["track_length_px"], "severity": "REVIEW", "notes": "Potential implausible geometry."})
        if math.isfinite(values["tortuosity"]) and values["tortuosity"] > 10:
            flags.append({"ground_truth_track_id": track_id, "flag_type": "EXTREME_TORTUOSITY", "feature": "tortuosity",
                          "value": values["tortuosity"], "threshold": "<= 10 descriptive", "severity": "REVIEW", "notes": "Extreme geometry; do not auto-remove."})
    # Descriptive Tukey flags are class-specific and never invalidate a row automatically.
    by_class = defaultdict(list)
    for row in successes: by_class[row["ground_truth_class"]].append(row)
    for class_name, rows in by_class.items():
        for feature in FEATURES:
            values = np.asarray([numeric[row["ground_truth_track_id"]][feature] for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) < 4: continue
            q1, q3 = np.percentile(values, [25, 75]); iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            for row in rows:
                value = numeric[row["ground_truth_track_id"]][feature]
                if math.isfinite(value) and (value < low or value > high):
                    flags.append({"ground_truth_track_id": row["ground_truth_track_id"], "flag_type": "TUKEY_OUTLIER",
                                  "feature": feature, "value": value, "threshold": f"{class_name} [{low:.8g}, {high:.8g}]",
                                  "severity": "DESCRIPTIVE", "notes": "Flag only; not automatically excluded."})
    return flags


def independence_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gt_counts = Counter(row["ground_truth_track_id"] for row in rows)
    mask_counts = Counter(row["selected_candidate"] for row in rows)
    dep_counts = Counter(row["dependency_group_id"] for row in rows)
    image_counts = Counter(row["source_image"] for row in rows)
    video_counts = Counter((row.get("source_video") or "UNAVAILABLE") for row in rows)
    output = []
    for row in rows:
        duplicate_gt = gt_counts[row["ground_truth_track_id"]] > 1
        duplicate_mask = mask_counts[row["selected_candidate"]] > 1
        excluded = row["ground_truth_track_id"] in {"A018", "A027"}
        output.append({
            "ground_truth_track_id": row["ground_truth_track_id"], "ground_truth_class": row["ground_truth_class"],
            "source_dataset": row["source_dataset"], "source_image": row["source_image"],
            "source_video": row.get("source_video", ""), "event_id": row.get("event_id", ""),
            "physical_track_group_id": row.get("physical_track_group_id", ""), "module2_mask_id": row["selected_candidate"],
            "dependency_group_id": row["dependency_group_id"], "dependency_group_size": dep_counts[row["dependency_group_id"]],
            "source_image_group_size": image_counts[row["source_image"]],
            "source_video_group_size": video_counts[row["source_video"]] if row.get("source_video") else "",
            "duplicate_ground_truth_id": "YES" if duplicate_gt else "NO",
            "duplicate_module2_mask_id": "YES" if duplicate_mask else "NO",
            "excluded_physical_duplicate_present": "YES" if excluded else "NO",
            "processing_status": row["processing_status"],
            "independence_status": "FAIL" if duplicate_gt or duplicate_mask or excluded else "PASS",
        })
    return output


def distribution_rows(new_rows: Sequence[Mapping[str, Any]], previous_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output = []
    def add(dataset: str, rows: Sequence[Mapping[str, Any]]) -> None:
        dimensions = [
            ("class", lambda r: (r.get("ground_truth_class", ""), "")),
            ("source", lambda r: (r.get("source_dataset", ""), "")),
            ("class_x_source", lambda r: (r.get("ground_truth_class", ""), r.get("source_dataset", ""))),
            ("roi_type_x_class", lambda r: (r.get("roi_type", "") or "UNAVAILABLE", r.get("ground_truth_class", ""))),
            ("source_video", lambda r: (r.get("source_video", "") or "UNAVAILABLE", "")),
            ("dependency_group", lambda r: (r.get("dependency_group_id", "") or "UNAVAILABLE", "")),
        ]
        for dimension, key in dimensions:
            for (first, second), count in sorted(Counter(key(row) for row in rows).items()):
                output.append({"dataset": dataset, "dimension": dimension, "category_1": first,
                               "category_2": second, "count": count})
    add("expanded_human_approved", new_rows)
    add("previous_production_29", previous_rows)
    return output


def run_expanded_feature_stage(history_csv: str | Path, labels_csv: str | Path, batch_manifest_csv: str | Path,
                               previous_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = extract_authorized_features(history_csv, labels_csv, batch_manifest_csv)
    successes = [row for row in rows if row["processing_status"] == "SUCCESS"]
    failures = [row for row in rows if row["processing_status"] != "SUCCESS"]
    flags = feature_flags(successes)
    independence = independence_audit(rows)
    previous = [row for row in read_csv(previous_csv) if row.get("processing_status") == "success"]
    if len(previous) != 29: raise ValueError(f"Expected previous production dataset to contain 29 successful rows, found {len(previous)}.")
    distributions = distribution_rows(successes, previous)
    _atomic_csv(output / "expanded_module3_features.csv", FEATURE_COLUMNS, successes)
    _atomic_csv(output / "expanded_feature_failures.csv", FAILURE_COLUMNS, failures)
    _atomic_csv(output / "expanded_feature_sanity_flags.csv", FLAG_COLUMNS, flags)
    _atomic_csv(output / "expanded_dataset_distribution.csv", DIST_COLUMNS, distributions)
    _atomic_csv(output / "expanded_independence_audit.csv", INDEPENDENCE_COLUMNS, independence)

    invalid_ids = {flag["ground_truth_track_id"] for flag in flags if flag["severity"] == "INVALID"}
    review_flags = [flag for flag in flags if flag["severity"] == "REVIEW"]
    duplicate_masks = sum(row["duplicate_module2_mask_id"] == "YES" for row in independence)
    mask_counts = Counter(row["selected_candidate"] for row in rows)
    duplicate_mask_redundancy = sum(count - 1 for count in mask_counts.values() if count > 1)
    independence_failures = sum(row["independence_status"] == "FAIL" for row in independence)
    class_counts = Counter(row["ground_truth_class"] for row in successes)
    source_counts = Counter(row["source_dataset"] for row in successes)
    class_source = Counter((row["ground_truth_class"], row["source_dataset"]) for row in successes)
    missing_class_source = [
        (class_name, source_name)
        for class_name in ("Alpha", "Electron") for source_name in ("AWAN", "Lecturer")
        if class_source[(class_name, source_name)] == 0
    ]
    dependency_counts = Counter(row["dependency_group_id"] for row in successes)
    groups_by_class = {name: len({row["dependency_group_id"] for row in successes if row["ground_truth_class"] == name})
                       for name in ("Alpha", "Electron")}
    if independence_failures or invalid_ids or min(class_counts.values(), default=0) < 5 or min(groups_by_class.values(), default=0) < 2:
        readiness = "C. NOT READY — EXTRACTION/INDEPENDENCE PROBLEM"
    elif failures or review_flags or missing_class_source:
        readiness = "B. READY AFTER SMALL DATA CLEANUP"
    else:
        readiness = "A. READY FOR GROUPED RF RE-EVALUATION"
    clean_count = len(successes) - len(invalid_ids) - duplicate_mask_redundancy
    report = [
        "EXPANDED HUMAN-APPROVED MODULE 3 FEATURE AUDIT", "",
        "AUTHORIZED MODULE-3 INPUT TRACKS: 35", f"FEATURE EXTRACTION SUCCESS: {len(successes)}",
        f"FEATURE EXTRACTION FAILURE: {len(failures)}", "",
        f"SUCCESSFUL ALPHA: {class_counts['Alpha']}", f"SUCCESSFUL ELECTRON: {class_counts['Electron']}",
        f"AWAN: {source_counts['AWAN']}", f"LECTURER: {source_counts['Lecturer']}",
        f"DEPENDENCY GROUPS: {len(dependency_counts)}", "",
        f"INVALID FEATURE ROWS: {len(invalid_ids)}", f"DUPLICATE MASK ROWS: {duplicate_masks}",
        f"OUTLIER FLAGS: {len(flags)} (descriptive={sum(f['severity']=='DESCRIPTIVE' for f in flags)}, review={len(review_flags)})",
        f"INDEPENDENCE FAILURES: {independence_failures}", "",
        "CLASS x SOURCE:",
        *[f"  {cls} x {source}: {count}" for (cls, source), count in sorted(class_source.items())], "",
        f"CLASS-SOURCE BALANCE FLAGS: {len(missing_class_source)}",
        *[f"  MISSING {cls} x {source}" for cls, source in missing_class_source],
        ("The missing class-source cell creates partial source/class confounding and must be retained in grouped evaluation caveats."
         if missing_class_source else "All class-source cells are represented."), "",
        f"DEPENDENCY GROUPS BY CLASS: Alpha={groups_by_class['Alpha']}; Electron={groups_by_class['Electron']}",
        "Source-image, event, physical-track, and source-video grouping columns are preserved for grouped evaluation.", "",
        "PREVIOUS PRODUCTION DATASET: 29 rows", f"NEW EXTRACTED DATASET: {len(successes)} rows",
        (f"NEW CLEAN DATASET: {clean_count} rows maximum after duplicate-mask adjudication; NOT FINALIZED"
         if duplicate_mask_redundancy else f"NEW CLEAN DATASET: {clean_count} rows"), "",
        f"RF READINESS: {readiness}",
        "Descriptive statistical outliers were flagged but not automatically removed.",
        "No Random Forest evaluation or training was performed.",
    ]
    if failures:
        report.extend(["", "EXTRACTION FAILURES:", *[f"  {row['ground_truth_track_id']}: {row['error_message']}" for row in failures]])
    if flags:
        report.extend(["", "FEATURE FLAGS:", *[f"  {f['ground_truth_track_id']} {f['severity']} {f['flag_type']} {f['feature']}: {f['value']}" for f in flags]])
    _atomic_text(output / "expanded_feature_audit.txt", "\n".join(report) + "\n")
    return {
        "authorized": len(rows), "successful": len(successes), "failed": len(failures),
        "class_counts": dict(class_counts), "source_counts": dict(source_counts),
        "dependency_groups": len(dependency_counts), "invalid_feature_rows": len(invalid_ids),
        "duplicate_mask_rows": duplicate_masks, "duplicate_mask_redundancy": duplicate_mask_redundancy,
        "outlier_flags": len(flags), "class_source_balance_flags": len(missing_class_source), "clean_rows": clean_count,
        "readiness": readiness,
    }
