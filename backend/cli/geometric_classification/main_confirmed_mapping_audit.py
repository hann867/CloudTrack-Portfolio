"""Generate the isolated audit of confirmed ground-truth to Module 2 mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.geometric_classification.confirmed_mapping_audit import audit_confirmed_mapping


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MATCH_ROOT = BACKEND_ROOT / "data/processed/geometric_classification/ground_truth_candidate_matching"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the confirmed ground-truth to Module 2 subset.")
    parser.add_argument("--labels", type=Path, default=BACKEND_ROOT / "data/processed/labelled_tracks/labels.csv")
    parser.add_argument("--mapping", type=Path, default=MATCH_ROOT / "confirmed_ground_truth_module2_mapping.csv")
    parser.add_argument("--matches", type=Path, default=MATCH_ROOT / "candidate_matching.csv")
    parser.add_argument("--reviews", type=Path, default=MATCH_ROOT / "manual_review_results.csv")
    parser.add_argument("--report", type=Path, default=MATCH_ROOT / "confirmed_mapping_audit.txt")
    parser.add_argument("--distribution", type=Path, default=MATCH_ROOT / "confirmed_mapping_distribution.csv")
    parser.add_argument("--excluded", type=Path, default=MATCH_ROOT / "excluded_tracks.csv")
    args = parser.parse_args(argv)
    try:
        result = audit_confirmed_mapping(args.labels, args.mapping, args.matches, args.reviews,
                                         args.report, args.distribution, args.excluded)
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2)); return 1
    summary = {
        "total_confirmed": result["total_confirmed"],
        "alpha": result["class_counts"].get("Alpha", 0),
        "electron": result["class_counts"].get("Electron", 0),
        "excluded": len(result["excluded"]),
        "absolute_phi_cramers_v": result["source_association"]["phi_cramers_v"],
        "fisher_p_value": result["source_association"]["fisher_p_value"],
        "recommendation": result["recommendation"],
        "report": str(args.report.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
