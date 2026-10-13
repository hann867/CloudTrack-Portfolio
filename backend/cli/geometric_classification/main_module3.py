"""Development CLI for CloudChamber Module 3 geometric analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.src.geometric_classification.feature_extractor import (
    find_annotated_images,
    process_annotated_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "geometric_classification"


def create_output_directories(output_root: Path) -> None:
    for name in ("features", "diagnostics", "models"):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometric analysis of cloud-chamber tracks")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--diagnostics", type=int, default=0, metavar="N", help="Save diagnostics for N successes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"CloudTrack Module 3\n\nInput directory not found: {args.input}")
        return 1
    create_output_directories(args.output_root)
    images = find_annotated_images(args.input)
    output_csv = args.output_root / "features" / "geometric_features.csv"
    label_file = args.input / "particle_labels.csv"
    frame = process_annotated_dataset(
        args.input,
        output_csv,
        label_file=label_file if label_file.exists() else None,
        diagnostics_dir=args.output_root / "diagnostics",
        diagnostic_limit=max(0, args.diagnostics),
    )
    successes = int((frame["processing_status"] == "success").sum()) if not frame.empty else 0
    print("CloudTrack Module 3")
    print(f"\nImages found: {len(images)}")
    print(f"Successfully processed: {successes}")
    print(f"Failed: {len(frame) - successes}")
    print(f"\nFeature table: {output_csv}")
    if not label_file.exists():
        print("\nRandom Forest training on real project data: NOT PERFORMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
