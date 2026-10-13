"""Run spatial-only matching of labelled ground truth to Module 2 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.geometric_classification.ground_truth_candidate_matcher import match_dataset


BACKEND_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match manual ground truth to exact-source Module 2 candidates.")
    parser.add_argument("--labels", type=Path, default=BACKEND_ROOT / "data/processed/labelled_tracks/labels.csv")
    parser.add_argument("--diagnostics-root", type=Path, default=BACKEND_ROOT / "data/processed/geometric_classification/module2_bridge_diagnostics")
    parser.add_argument("--source-root", type=Path, default=BACKEND_ROOT / "data/raw/labelled_sources")
    parser.add_argument("--output-dir", type=Path, default=BACKEND_ROOT / "data/processed/geometric_classification/ground_truth_candidate_matching")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = match_dataset(args.labels, args.diagnostics_root, args.source_root, args.output_dir)
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2))
        return 1
    summary = {key: value for key, value in result.items() if key != "track_summaries"}
    a022 = result["track_summaries"].get("A022")
    if a022:
        summary["A022_validation"] = {
            key: a022.get(key) for key in (
                "confidence_status", "candidate_track_id", "intersection_area", "iou",
                "manual_roi_coverage", "candidate_bbox_coverage", "centroid_distance_px",
                "normalized_centroid_distance", "spatial_score", "dominance_margin",
            )
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
