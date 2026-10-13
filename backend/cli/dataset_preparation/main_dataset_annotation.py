"""Launch the manual known-source particle-track annotation tool."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.src.dataset_preparation.track_annotation_tool import run_interactive


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = BACKEND_ROOT / "data" / "processed" / "labelled_tracks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually crop and label tracks from one source image or an image folder (no automatic classification)"
    )
    parser.add_argument("image", nargs="?", type=Path, help="Source image or folder path; prompted when omitted")
    parser.add_argument("--source-dataset", default="", help="Source dataset name, such as AWAN or McGill")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT, help="Labelled-track output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_interactive(args.image, args.output_root, args.source_dataset)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
