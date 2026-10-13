"""Launch the lecturer-video frame extraction utility."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.src.dataset_preparation.video_frame_selector import run_video_frame_selector_path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = BACKEND_ROOT / "data" / "raw" / "labelled_sources" / "lecturer" / "alpha_frames"
DEFAULT_STATUS = BACKEND_ROOT / "data" / "raw" / "labelled_sources" / "lecturer" / "videos" / "video_review_status.csv"
DEFAULT_HISTORY = BACKEND_ROOT / "data" / "processed" / "labelled_tracks" / "review_history"
DEFAULT_LABELS = BACKEND_ROOT / "data" / "processed" / "labelled_tracks" / "labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract explicit original-resolution video frames without assigning particle labels"
    )
    parser.add_argument("video", type=Path, help="Lecturer-provided MP4 video or folder")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-reviewed", action="store_true", help="Include videos already marked reviewed in folder mode")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Read-only crop linkage metadata")
    parser.add_argument("--source-dataset", default="Lecturer")
    parser.add_argument("--near-timestamp-seconds", type=float, default=.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_video_frame_selector_path(
            args.video,
            args.output_dir,
            DEFAULT_STATUS,
            include_reviewed=args.include_reviewed,
            history_dir=args.history_dir,
            labels_path=args.labels,
            source_dataset=args.source_dataset,
            near_timestamp_seconds=args.near_timestamp_seconds,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
