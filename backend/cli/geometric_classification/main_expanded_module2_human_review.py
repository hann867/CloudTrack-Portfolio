"""Launch or inspect the resumable expanded Module 2 human review queue."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from backend.src.geometric_classification.expanded_module2_human_review import (
    build_review_queue,
    cycle_candidate,
    format_status,
    initialize_history,
    is_complete,
    next_unresolved_index,
    previous_index,
    record_mapping_decision,
    record_mask_decision,
    render_review_screen,
    status_summary,
)
from backend.src.geometric_classification.manual_candidate_review import exact_file_index


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = BACKEND_ROOT / "data/processed/geometric_classification/expanded_ground_truth_module2_batch"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume human mapping and mask-quality adjudication for 114 independent tracks.")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--history", type=Path, default=None, help="Isolated review CSV; defaults inside batch-root/human_review")
    parser.add_argument("--status-only", action="store_true", help="Initialize/reconcile history and print status without opening a GUI")
    return parser.parse_args(argv)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.batch_root.resolve()
    return {
        "root": root,
        "manifest": root / "module2_batch_manifest.csv",
        "mapping": root / "candidate_mapping.csv",
        "quality": root / "mask_quality_review.csv",
        "candidates": root / "candidate_matching_work/candidate_matching.csv",
        "diagnostics": root / "module2_diagnostics",
        "history": (args.history or root / "human_review/module2_human_adjudication.csv").resolve(),
    }


def _confirm(screen, message: str) -> bool | None:
    prompt = screen.copy()
    cv2.rectangle(prompt, (0, 0), (prompt.shape[1], 70), (0, 0, 120), -1)
    cv2.putText(prompt, f"{message}  Y=yes N=no Q=quit", (15, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    cv2.imshow("Expanded Module 2 human adjudication", prompt)
    while True:
        key = cv2.waitKeyEx(0) & 0xFF
        if key in (ord("y"), ord("Y")):
            return True
        if key in (ord("n"), ord("N")):
            return False
        if key in (ord("q"), ord("Q"), 27):
            return None


def _print_status(paths: dict[str, Path]) -> None:
    print(format_status(status_summary(paths["manifest"], paths["mapping"], paths["quality"], paths["history"])))
    print(f"\nHUMAN REVIEW HISTORY: {paths['history']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _paths(args)
    try:
        initialize_history(paths["manifest"], paths["mapping"], paths["quality"], paths["history"])
        summary = status_summary(paths["manifest"], paths["mapping"], paths["quality"], paths["history"])
        if args.status_only:
            queue = build_review_queue(paths["manifest"], paths["mapping"], paths["quality"],
                                       paths["candidates"], paths["diagnostics"], paths["history"])
            if len(queue) != summary["human_review_required"]:
                raise ValueError("Review queue count does not match status summary.")
            _print_status(paths)
            return 0
        queue = build_review_queue(paths["manifest"], paths["mapping"], paths["quality"],
                                   paths["candidates"], paths["diagnostics"], paths["history"])
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"HUMAN REVIEW REQUIRED BEFORE START: {summary['human_review_required']}")
    if summary["human_review_remaining"] == 0:
        print("No unresolved cases remain.")
        _print_status(paths)
        return 0

    wanted_sources = {item["annotation"]["source_image"] for item in queue}
    wanted_crops = {item["annotation"]["crop_filename"] for item in queue}
    source_index = exact_file_index([BACKEND_ROOT / "data/raw/labelled_sources"], wanted_sources)
    crop_index = exact_file_index([BACKEND_ROOT / "data/processed/labelled_tracks"], wanted_crops)
    index = next((i for i, item in enumerate(queue) if not is_complete(item["history"])), 0)
    candidate_index, zoom_roi, stage = 0, False, "mapping"
    window = "Expanded Module 2 human adjudication"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 1000)
    try:
        while 0 <= index < len(queue):
            item = queue[index]
            if item["history"]["mapping_decision"] == "CONFIRMED":
                stage = "mask"
            candidate_index = min(candidate_index, max(0, len(item["candidates"]) - 1))
            screen = render_review_screen(
                item, candidate_index,
                source_index.get(item["annotation"]["source_image"]),
                crop_index.get(item["annotation"]["crop_filename"]),
                zoom_roi=zoom_roi,
            )
            cv2.putText(screen, f"ACTIVE DECISION STAGE: {stage.upper()}", (1350, 1090),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 1, cv2.LINE_AA)
            cv2.imshow(window, screen)
            key = cv2.waitKeyEx(0) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("["), ord(",")):
                candidate_index = cycle_candidate(candidate_index, len(item["candidates"]), -1); continue
            if key in (ord("]"), ord(".")):
                candidate_index = cycle_candidate(candidate_index, len(item["candidates"]), 1); continue
            if key in (ord("z"), ord("Z")):
                zoom_roi = not zoom_roi; continue
            if key == ord("0"):
                zoom_roi = False; continue
            if key in (ord("m"), ord("M")):
                stage = "mapping"; continue
            if key in (ord("b"), ord("B")):
                index = previous_index(index, len(queue)); candidate_index = 0; stage = "mapping"; continue
            if key in (ord("n"), ord("N")):
                target = next_unresolved_index(queue, index)
                if target is not None:
                    index, candidate_index, stage = target, 0, "mapping"
                continue

            try:
                if stage == "mapping" and key in (ord("c"), ord("C"), ord("r"), ord("R"), ord("u"), ord("U")):
                    decision = {ord("c"): "CONFIRMED", ord("r"): "REJECTED", ord("u"): "UNCERTAIN"}[ord(chr(key).lower())]
                    candidate = item["candidates"][candidate_index] if decision == "CONFIRMED" and item["candidates"] else None
                    existing = item["history"]["mapping_decision"] != "UNREVIEWED"
                    if existing:
                        answer = _confirm(screen, f"Replace existing mapping decision for {item['annotation']['track_id']}?")
                        if answer is None: break
                        if not answer: continue
                    updated = record_mapping_decision(paths["history"], item["annotation"]["track_id"], decision,
                                                      candidate, replace_existing=existing)
                    item["history"] = updated
                    if decision == "CONFIRMED":
                        stage = "mask"
                    else:
                        target = next_unresolved_index(queue, index)
                        if target is not None: index, candidate_index, stage = target, 0, "mapping"
                    continue
                if stage == "mask" and key in (ord("p"), ord("P"), ord("f"), ord("F"), ord("u"), ord("U")):
                    decision = {ord("p"): "PASS", ord("f"): "FAIL", ord("u"): "UNCERTAIN"}[ord(chr(key).lower())]
                    existing = item["history"]["mask_quality_decision"] != "UNREVIEWED"
                    if existing:
                        answer = _confirm(screen, f"Replace existing mask decision for {item['annotation']['track_id']}?")
                        if answer is None: break
                        if not answer: continue
                    updated = record_mask_decision(paths["history"], item["annotation"]["track_id"], decision,
                                                   replace_existing=existing)
                    item["history"] = updated
                    target = next_unresolved_index(queue, index)
                    if target is not None: index, candidate_index, stage = target, 0, "mapping"
                    continue
            except (OSError, ValueError, FileExistsError, KeyError) as exc:
                print(f"Could not save decision: {exc}")
    finally:
        cv2.destroyAllWindows()
    _print_status(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
