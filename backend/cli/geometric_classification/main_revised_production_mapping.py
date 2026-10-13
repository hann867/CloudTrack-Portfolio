"""Create and audit the adjudication-controlled production mapping revision."""

from __future__ import annotations

import json
from pathlib import Path

from backend.src.geometric_classification.revised_production_mapping import (
    run_revised_production_workflow,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MATCH_ROOT = BACKEND_ROOT / "data/processed/geometric_classification/ground_truth_candidate_matching"
FEATURE_ROOT = BACKEND_ROOT / "data/processed/geometric_classification/production_module2_features"


def main() -> int:
    try:
        result = run_revised_production_workflow(
            MATCH_ROOT / "confirmed_ground_truth_module2_mapping.csv",
            BACKEND_ROOT / "data/processed/labelled_tracks/labels.csv",
            BACKEND_ROOT / "data/processed/geometric_classification/module2_bridge_diagnostics",
            BACKEND_ROOT / "data/processed/labelled_tracks",
            MATCH_ROOT / "confirmed_ground_truth_module2_mapping_revised.csv",
            MATCH_ROOT / "revised_mapping_exclusions.csv",
            MATCH_ROOT / "revised_mapping_independence_review.csv",
            MATCH_ROOT / "revised_mapping_audit.txt",
            FEATURE_ROOT / "revised_confirmed_module2_geometric_features.csv",
            FEATURE_ROOT / "revised_confirmed_module2_feature_audit.txt",
        )
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2))
        return 1
    print(json.dumps({"status": "success", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
