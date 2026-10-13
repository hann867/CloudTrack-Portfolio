"""Bootstrap and summarize persistent video/frame review history."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.src.dataset_preparation.video_review_history import (
    ReviewHistoryStore,
    bootstrap_existing_data,
    discover_existing_review_data,
    format_review_status,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = BACKEND_ROOT / "data" / "processed" / "labelled_tracks" / "review_history"
DEFAULT_VIDEOS = BACKEND_ROOT / "data" / "raw" / "labelled_sources" / "lecturer" / "videos"
DEFAULT_FRAMES = BACKEND_ROOT / "data" / "raw" / "labelled_sources"
DEFAULT_LABELS = BACKEND_ROOT / "data" / "processed" / "labelled_tracks" / "labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show or bootstrap persistent video-review history")
    parser.add_argument("--bootstrap-existing", action="store_true", help="Import provable existing frame metadata")
    parser.add_argument("--yes", action="store_true", help="Confirm bootstrap non-interactively")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = ReviewHistoryStore(args.history_dir)
    if args.bootstrap_existing:
        discovered = discover_existing_review_data(args.videos_dir, args.frames_root, args.labels)
        print(f"Videos discovered: {len(discovered['videos'])}")
        print(f"Saved frames discovered: {len(discovered['frames'])}")
        print(f"Labelled source frames discovered: {discovered['labelled_source_frames']}")
        print(f"Potential conflicts: {discovered['potential_conflicts']}")
        confirmed = args.yes or input("Import these proven facts into review history? [y/N] ").strip().lower() == "y"
        if not confirmed:
            print("Bootstrap cancelled; nothing was changed.")
            return 0
        result = bootstrap_existing_data(store, args.videos_dir, args.frames_root, args.labels)
        print(f"Historical saved frames imported this run: {result['historical_saved_frames_imported']}")
        print(f"Exact duplicates detected: {result['exact_duplicates_detected']}")
        print(f"Potential near-timestamp duplicates: {result['potential_near_timestamp_duplicates']}")
        print()
    print(format_review_status(store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
