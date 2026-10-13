"""Interactive review of pre-ranked ground-truth to Module 2 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from backend.src.geometric_classification.manual_candidate_review import (
    confirmation_screen, diagnostic_overlay_index, ensure_review_results, exact_file_index,
    generate_confirmed_mapping, load_review_queue, progress_summary, record_review_decision,
    render_review_screen,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MATCH_ROOT = BACKEND_ROOT / "data/processed/geometric_classification/ground_truth_candidate_matching"
DEFAULT_LABELS = BACKEND_ROOT / "data/processed/labelled_tracks/labels.csv"
DEFAULT_MATCHES = MATCH_ROOT / "candidate_matching.csv"
DEFAULT_RESULTS = MATCH_ROOT / "manual_review_results.csv"
DEFAULT_MAPPING = MATCH_ROOT / "confirmed_ground_truth_module2_mapping.csv"
DEFAULT_DIAGNOSTICS = BACKEND_ROOT / "data/processed/geometric_classification/module2_bridge_diagnostics"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually confirm REVIEW-ranked Module 2 candidates.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--diagnostics-root", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--save-montages", action="store_true", help="Save a display montage only after an item is reviewed")
    parser.add_argument("--summary-only", action="store_true", help="Initialize outputs and print progress without opening OpenCV")
    return parser.parse_args(argv)


def _confirm(window: str, screen, message: str) -> str:
    cv2.imshow(window, confirmation_screen(screen, message))
    while True:
        key = cv2.waitKeyEx(0) & 0xFF
        if key in (ord("y"), ord("Y")):
            return "yes"
        if key in (ord("n"), ord("N")):
            return "no"
        if key in (ord("q"), ord("Q")):
            return "quit"


def _print_summary(args: argparse.Namespace) -> None:
    print(json.dumps(progress_summary(args.labels, args.matches, args.results), indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ensure_review_results(args.results)
        generate_confirmed_mapping(args.labels, args.matches, args.results, args.mapping_output)
        queue = load_review_queue(args.labels, args.matches, args.results)
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2))
        return 1
    if args.summary_only:
        _print_summary(args)
        return 0
    if not queue:
        print("No unresolved REVIEW tracks remain.")
        _print_summary(args)
        return 0
    wanted_sources = {item["annotation"]["source_image"] for item in queue}
    wanted_crops = {item["annotation"]["crop_filename"] for item in queue}
    sources = exact_file_index([BACKEND_ROOT / "data/raw/labelled_sources", BACKEND_ROOT / "data/raw/cloud_chamber"], wanted_sources)
    crops = exact_file_index([BACKEND_ROOT / "data/processed/labelled_tracks"], wanted_crops)
    overlays = diagnostic_overlay_index(args.diagnostics_root)
    window = "Ground truth <-> Module 2 manual review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL); cv2.resizeWindow(window, 1500, 1020)
    index = 0
    try:
        while 0 <= index < len(queue):
            item = queue[index]; annotation = item["annotation"]
            screen = render_review_screen(item, sources.get(annotation["source_image"]),
                                          crops.get(annotation["crop_filename"]), overlays)
            cv2.imshow(window, screen)
            key = cv2.waitKeyEx(0) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("p"), ord("P")):
                index = max(0, index - 1); continue
            if key in (ord("s"), ord("S")):
                status, rank, message = "SKIPPED", None, f"{annotation['track_id']} -> SKIPPED"
            elif key in (ord("n"), ord("N")):
                status, rank, message = "NONE", None, f"{annotation['track_id']} -> NONE of these candidates"
            elif key in (ord("1"), ord("2"), ord("3")):
                rank = int(chr(key))
                candidate = next((row for row in item["candidates"] if int(row["rank"]) == rank), None)
                if candidate is None:
                    continue
                status, message = "CONFIRMED", f"{annotation['track_id']} -> {candidate['candidate_track_id']}"
            else:
                continue
            answer = _confirm(window, screen, message)
            if answer == "quit":
                break
            if answer == "no":
                continue
            try:
                record_review_decision(args.results, item, status, selected_rank=rank,
                                       replace_existing=item.get("existing_result") is not None)
                generate_confirmed_mapping(args.labels, args.matches, args.results, args.mapping_output)
                if args.save_montages:
                    montage = MATCH_ROOT / "manual_review_montages" / f"{annotation['track_id']}.png"
                    montage.parent.mkdir(parents=True, exist_ok=True)
                    ok, encoded = cv2.imencode(".png", screen)
                    if ok:
                        encoded.tofile(str(montage))
            except (OSError, TypeError, ValueError, FileExistsError) as exc:
                print(f"Could not save {annotation['track_id']}: {exc}")
                continue
            index += 1
    finally:
        cv2.destroyAllWindows()
    _print_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
