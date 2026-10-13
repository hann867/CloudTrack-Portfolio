"""Launch or inspect the resumable saved-Lecturer-frame annotation queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.src.dataset_preparation.lecturer_annotation_history import (
    LecturerAnnotationHistoryStore,
    annotation_summary,
    discover_saved_lecturer_frames,
    format_annotation_summary,
    run_lecturer_annotation_queue,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_ROOT = BACKEND_ROOT / "data" / "raw" / "labelled_sources" / "lecturer"
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "data" / "processed" / "labelled_tracks"
DEFAULT_REVIEW_ROOT = DEFAULT_OUTPUT_ROOT / "review_history"
DEFAULT_FRAME_HISTORY = DEFAULT_REVIEW_ROOT / "frame_review_history.csv"
DEFAULT_ANNOTATION_HISTORY = DEFAULT_REVIEW_ROOT / "lecturer_annotation_history.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume annotation of authoritative saved Lecturer frames without rescanning unrelated files"
    )
    parser.add_argument("--status-only", action="store_true", help="Reconcile and print queue status without opening images")
    parser.add_argument(
        "--include-annotated", action="store_true",
        help="Offer already annotated frames for explicit inspection (excluded by default)",
    )
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--frame-history", type=Path, default=DEFAULT_FRAME_HISTORY)
    parser.add_argument("--annotation-history", type=Path, default=DEFAULT_ANNOTATION_HISTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.status_only:
            frames = discover_saved_lecturer_frames(args.frame_history, args.frames_root)
            store = LecturerAnnotationHistoryStore(args.annotation_history)
            rows = store.reconcile(frames, args.output_root / "labels.csv")
            print(format_annotation_summary(annotation_summary(frames, rows)))
            print(f"ANNOTATION HISTORY: {args.annotation_history.resolve()}")
            return 0
        return run_lecturer_annotation_queue(
            frames_root=args.frames_root,
            frame_history_path=args.frame_history,
            annotation_history_path=args.annotation_history,
            output_root=args.output_root,
            include_annotated=args.include_annotated,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
