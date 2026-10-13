"""Audit labelled crops and extract existing Module 3 geometric features."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from backend.src.dataset_preparation.dataset_audit import audit_labelled_dataset
from backend.src.dataset_preparation.labelled_feature_extraction import (
    SEGMENTATION_SOURCE,
    build_descriptive_summary,
    build_extraction_summary,
    extract_labelled_geometric_features,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = BACKEND_ROOT / "data" / "processed" / "labelled_tracks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only labelled-dataset audit and pilot Module 3 feature extraction")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def _count_issues(issues: list[dict[str, str]], code: str) -> int:
    return sum(issue["issue_code"] == code for issue in issues)


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    audit_dir = root / "audit"
    feature_dir = root / "features"
    try:
        audit = audit_labelled_dataset(root, audit_dir)
        eligible = audit.structurally_valid_records()
        features = extract_labelled_geometric_features(root, eligible, feature_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    class_counts = Counter(row.get("class", "").strip() for row in audit.records)
    quality_counts = Counter(
        (row.get("class", "").strip(), row.get("ground_truth_quality", "").strip().lower() or "missing")
        for row in audit.records
    )
    status_counts = Counter(audit.record_statuses.values())
    extraction = {row["class"]: row for row in build_extraction_summary(features)}
    descriptive = build_descriptive_summary(features)
    feature_lookup = {(row["class"], row["feature"]): row for row in descriptive}
    duplicate_crop_groups = sum(row["duplicate_type"] == "exact_crop_sha256" for row in audit.duplicates)
    duplicate_event_groups = sum(row["duplicate_type"] == "event_id" for row in audit.duplicates)

    print("ACTUAL DATASET COUNTS")
    print(f"Alpha: {class_counts['Alpha']}")
    print(f"Electron: {class_counts['Electron']}")
    print(f"Uncertain: {class_counts['Uncertain']}")
    print(f"Total: {len(audit.records)}")
    print("\nGROUND-TRUTH QUALITY COUNTS")
    for class_name in ("Alpha", "Electron"):
        for quality in ("high", "medium", "low", "missing"):
            print(f"{class_name} {quality}: {quality_counts[(class_name, quality)]}")

    print("\nAUDIT")
    print(f"Records checked: {len(audit.records)}")
    for status in ("PASS", "REVIEW", "ERROR"):
        print(f"{status}: {status_counts[status]}")
    print(f"Exact duplicate crops: {duplicate_crop_groups}")
    print(f"Duplicate event IDs: {duplicate_event_groups}")
    print(f"Missing crops: {_count_issues(audit.issues, 'missing_crop')}")
    print(f"Invalid labels: {_count_issues(audit.issues, 'invalid_class')}")

    print("\nFEATURE EXTRACTION")
    for class_name in ("Alpha", "Electron"):
        row = extraction[class_name]
        print(f"{class_name} processed: {row['total_samples']}")
        print(f"{class_name} success: {row['success']}")
        print(f"{class_name} failed: {row['failed']}")
    print(f"Segmentation source used: {SEGMENTATION_SOURCE}")
    print(f"Output feature CSV: {feature_dir / 'labelled_geometric_features.csv'}")

    print("\nFEATURE DESCRIPTIVE FINDINGS")
    for feature in ("track_length_px", "mean_width_px", "mean_curvature", "tortuosity"):
        alpha = feature_lookup[("Alpha", feature)]
        electron = feature_lookup[("Electron", feature)]
        print(f"{feature}: Alpha mean={alpha['mean']}, Electron mean={electron['mean']} (descriptive only)")
    print("Pixel-based geometric features may include source-scale effects.")

    print("\nSOURCE CONFOUND")
    print(f"Potential source-domain confounding: {'YES' if audit.source_confound else 'NO'}")
    if audit.source_confound:
        print("Potential source-domain confounding exists. Differences in camera, background, resolution, or acquisition setup may correlate with class label.")
    else:
        print("No strong class-specific dominant source_dataset pattern was detected by the audit threshold.")

    all_features_succeeded = all(row["failed"] == 0 for row in extraction.values())
    ready = status_counts["ERROR"] == 0 and all_features_succeeded and not audit.source_confound
    print("\nRANDOM FOREST TRAINED:")
    print("NO")
    print("\nPilot dataset audited: YES")
    print(f"Geometric feature extraction completed: {'YES' if features else 'NO'}")
    print(f"Ready for first classification experiment: {'YES' if ready else 'NO'}")
    if not ready:
        blockers = []
        if status_counts["ERROR"]:
            blockers.append(f"{status_counts['ERROR']} audit ERROR record(s)")
        if not all_features_succeeded:
            blockers.append("one or more feature extractions failed")
        if audit.source_confound:
            blockers.append("potential source-domain confounding")
        print(f"Blocking issue: {', '.join(blockers)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

