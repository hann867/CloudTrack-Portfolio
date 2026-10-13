"""Run diagnostic class/source analysis on existing labelled geometric features."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.src.dataset_preparation.source_confound_analysis import analyse_source_confound


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = BACKEND_ROOT / "data" / "processed" / "labelled_tracks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose source-domain confounding without training a classifier")
    parser.add_argument("--labels", type=Path, default=DATASET_ROOT / "labels.csv")
    parser.add_argument("--features", type=Path, default=DATASET_ROOT / "features" / "labelled_geometric_features.csv")
    parser.add_argument("--output-dir", type=Path, default=DATASET_ROOT / "analysis" / "source_confound")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyse_source_confound(args.labels, args.features, args.output_dir)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 1
    print(result["report"], end="")
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

