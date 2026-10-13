"""Extract geometry for confirmed mappings from existing production Module 2 masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.geometric_classification.production_module2_features import run_production_feature_extraction


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MATCH_ROOT = BACKEND_ROOT / "data/processed/geometric_classification/ground_truth_candidate_matching"
OUTPUT_ROOT = BACKEND_ROOT / "data/processed/geometric_classification/production_module2_features"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract Module 3 geometry from confirmed production Module 2 masks.")
    parser.add_argument("--mapping", type=Path, default=MATCH_ROOT / "confirmed_ground_truth_module2_mapping.csv")
    parser.add_argument("--labels", type=Path, default=BACKEND_ROOT / "data/processed/labelled_tracks/labels.csv")
    parser.add_argument("--diagnostics-root", type=Path,
                        default=BACKEND_ROOT / "data/processed/geometric_classification/module2_bridge_diagnostics")
    parser.add_argument("--temporary-features", type=Path,
                        default=BACKEND_ROOT / "data/processed/labelled_tracks/features/labelled_geometric_features.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "confirmed_module2_geometric_features.csv")
    parser.add_argument("--comparison", type=Path, default=OUTPUT_ROOT / "temporary_vs_production_feature_comparison.csv")
    parser.add_argument("--audit", type=Path, default=OUTPUT_ROOT / "production_module2_feature_audit.txt")
    args = parser.parse_args(argv)
    try:
        result = run_production_feature_extraction(
            args.mapping, args.labels, args.diagnostics_root, args.temporary_features,
            args.output, args.comparison, args.audit,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2)); return 1
    summary = {key: value for key, value in result.items() if key not in {"issues", "comparisons"}}
    summary["feature_sanity"] = "PASS" if not result["issues"] else "ISSUES"
    summary["output"] = str(args.output.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
