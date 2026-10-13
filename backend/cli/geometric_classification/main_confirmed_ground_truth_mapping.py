"""Generate the non-destructive confirmed ground-truth to Module 2 mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.geometric_classification.manual_candidate_review import generate_confirmed_mapping


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MATCH_ROOT = BACKEND_ROOT / "data/processed/geometric_classification/ground_truth_candidate_matching"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine AUTO_HIGH and MANUAL_REVIEW matches only.")
    parser.add_argument("--labels", type=Path, default=BACKEND_ROOT / "data/processed/labelled_tracks/labels.csv")
    parser.add_argument("--matches", type=Path, default=MATCH_ROOT / "candidate_matching.csv")
    parser.add_argument("--results", type=Path, default=MATCH_ROOT / "manual_review_results.csv")
    parser.add_argument("--output", type=Path, default=MATCH_ROOT / "confirmed_ground_truth_module2_mapping.csv")
    args = parser.parse_args(argv)
    try:
        rows = generate_confirmed_mapping(args.labels, args.matches, args.results, args.output)
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2)); return 1
    methods = {method: sum(row["match_method"] == method for row in rows)
               for method in ("AUTO_HIGH", "MANUAL_REVIEW")}
    print(json.dumps({"output": str(args.output.resolve()), "total_confirmed": len(rows), **methods}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
