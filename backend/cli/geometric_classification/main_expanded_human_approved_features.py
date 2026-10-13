"""Run the isolated human-approved expanded Module 3 feature stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.geometric_classification.expanded_human_approved_features import run_expanded_feature_stage


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "data/processed/geometric_classification/expanded_ground_truth_module2_batch"
DEFAULT_OUTPUT = ROOT / "data/processed/geometric_classification/expanded_human_approved_module3_features"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract seven Module 3 features from 35 human-approved Module 2 masks.")
    parser.add_argument("--history", type=Path, default=BATCH / "human_review/module2_human_adjudication.csv")
    parser.add_argument("--labels", type=Path, default=BATCH / "independent_ground_truth_labels.csv")
    parser.add_argument("--batch-manifest", type=Path, default=BATCH / "module2_batch_manifest.csv")
    parser.add_argument("--previous-features", type=Path, default=ROOT / "data/processed/geometric_classification/production_module2_features/revised_confirmed_module2_geometric_features.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run_expanded_feature_stage(args.history, args.labels, args.batch_manifest,
                                            args.previous_features, args.output_dir)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2)); return 1
    print(json.dumps({"status": "success", **result, "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
