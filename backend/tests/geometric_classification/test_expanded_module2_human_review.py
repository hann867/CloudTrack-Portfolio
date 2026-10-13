from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.geometric_classification.expanded_module2_human_review import (
    build_review_queue,
    cycle_candidate,
    initialize_history,
    is_complete,
    load_history,
    next_unresolved_index,
    previous_index,
    record_mapping_decision,
    record_mask_decision,
    render_review_screen,
    status_summary,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _fixture(root: Path) -> dict[str, Path]:
    track_ids = [f"A{i:03d}" for i in range(1, 54) if i not in {18, 27}] + [f"E{i:03d}" for i in range(1, 64)]
    assert len(track_ids) == 114
    manifests, mappings, qualities, labels = [], [], [], []
    review = {"A001": ("AMBIGUOUS", "REVIEW"), "A002": ("HIGH_CONFIDENCE_CANDIDATE", "PASS"),
              "E001": ("NO_MATCH", "FAIL")}
    diagnostics = root / "module2_diagnostics"
    candidate_rows = []
    for index, track_id in enumerate(track_ids):
        class_name = "Alpha" if track_id.startswith("A") else "Electron"
        source = f"source_{track_id}.png"
        mapping_status, quality_status = review.get(track_id, ("CONFIRMED_EXISTING", "PASS"))
        selected = f"{source[:-4]}_track_0001"
        manifests.append({
            "ground_truth_track_id": track_id, "ground_truth_class": class_name,
            "source_dataset": "AWAN" if index < 16 else "Lecturer", "source_image": source,
            "source_video": "video.mp4" if index >= 16 else "", "frame_number": str(index),
            "timestamp_seconds": str(index / 10), "diagnostic_manifest_path": str(diagnostics/source[:-4]/"manifest.csv"),
        })
        mappings.append({
            "ground_truth_track_id": track_id, "mapping_status": mapping_status,
            "candidate_track_id": selected, "component_id": "1",
        })
        qualities.append({
            "ground_truth_track_id": track_id, "mask_quality_status": quality_status,
            "quality_flags": "NONE" if quality_status == "PASS" else quality_status,
        })
        labels.append({
            "track_id": track_id, "class": class_name, "source_image": source,
            "source_dataset": manifests[-1]["source_dataset"], "crop_filename": f"{track_id}.png",
            "roi_type": "axis_aligned", "x": "10", "y": "10", "width": "20", "height": "30",
        })
        if track_id in review:
            folder = diagnostics / source[:-4]
            rows = []
            for number in range(1, 4):
                candidate_id = f"{source[:-4]}_track_{number:04d}"
                rows.append({
                    "track_id": candidate_id, "source_filename": source, "component_id": str(number),
                    "label_id": str(number), "bbox_x": str(8 + 25 * (number - 1)), "bbox_y": "8",
                    "bbox_width": "22", "bbox_height": "34", "roi_x0": str(5 + 25 * (number - 1)),
                    "roi_y0": "5", "roi_width": "30", "roi_height": "42",
                    "diagnostic_mask_file": f"masks/{candidate_id}_mask.png",
                    "diagnostic_image_file": f"images/{candidate_id}.png",
                    "diagnostic_overlay_file": f"overlays/{candidate_id}_overlay.png",
                })
            _write(folder / "manifest.csv", rows)
            matched_rows = rows[:1] if track_id == "E001" else rows
            for rank, diagnostic in enumerate(matched_rows, 1):
                candidate_rows.append({
                    "labelled_track_id": track_id, "confidence_status": "NO_MATCH" if track_id == "E001" else "REVIEW",
                    "rank": str(rank), "candidate_track_id": diagnostic["track_id"],
                    "component_id": diagnostic["component_id"], "label_id": diagnostic["label_id"],
                    "iou": ".5", "manual_roi_coverage": ".6", "candidate_bbox_coverage": ".7",
                    "normalized_centroid_distance": ".1", "spatial_score": ".7",
                })
    paths = {
        "manifest": root / "module2_batch_manifest.csv", "mapping": root / "candidate_mapping.csv",
        "quality": root / "mask_quality_review.csv", "labels": root / "independent_ground_truth_labels.csv",
        "candidates": root / "candidate_matching_work/candidate_matching.csv", "diagnostics": diagnostics,
        "history": root / "human_review/module2_human_adjudication.csv",
    }
    for key, data in (("manifest", manifests), ("mapping", mappings), ("quality", qualities),
                      ("labels", labels), ("candidates", candidate_rows)):
        _write(paths[key], data)
    return paths


def _initialize(paths: dict[str, Path]) -> None:
    initialize_history(paths["manifest"], paths["mapping"], paths["quality"], paths["history"])


def test_queue_preserves_confirmed_pass_and_includes_high_ambiguous_and_no_match(tmp_path: Path) -> None:
    paths = _fixture(tmp_path); _initialize(paths)
    history = load_history(paths["history"])
    assert sum(row["auto_accepted_existing"] == "YES" for row in history.values()) == 111
    assert history["A003"]["mapping_decision"] == "CONFIRMED"
    assert history["A003"]["mask_quality_decision"] == "PASS"
    assert history["A003"]["reviewed_at"] == ""
    queue = build_review_queue(paths["manifest"], paths["mapping"], paths["quality"],
                               paths["candidates"], paths["diagnostics"], paths["history"])
    assert [item["annotation"]["track_id"] for item in queue] == ["A001", "A002", "E001"]
    no_match = next(item for item in queue if item["annotation"]["track_id"] == "E001")
    assert len(no_match["candidates"]) == 3  # all source detections, not only matcher top rows


def test_resume_candidate_selection_and_separate_mask_decision(tmp_path: Path) -> None:
    paths = _fixture(tmp_path); _initialize(paths)
    queue = build_review_queue(paths["manifest"], paths["mapping"], paths["quality"],
                               paths["candidates"], paths["diagnostics"], paths["history"])
    item = queue[0]; selected = item["candidates"][1]
    saved = record_mapping_decision(paths["history"], "A001", "CONFIRMED", selected,
                                    timestamp="2026-01-01T00:00:00+00:00")
    assert saved["selected_candidate"] == selected["candidate_track_id"]
    assert saved["module2_component_id"] == selected["component_id"]
    assert saved["mask_quality_decision"] == "UNREVIEWED"
    resumed = build_review_queue(paths["manifest"], paths["mapping"], paths["quality"],
                                 paths["candidates"], paths["diagnostics"], paths["history"])
    assert resumed[0]["history"]["mapping_decision"] == "CONFIRMED"
    assert resumed[0]["candidates"][0]["candidate_track_id"] == selected["candidate_track_id"]
    complete = record_mask_decision(paths["history"], "A001", "UNCERTAIN",
                                    timestamp="2026-01-01T00:01:00+00:00")
    assert is_complete(complete)


def test_rejected_skips_mask_and_overwrite_requires_confirmation_flag(tmp_path: Path) -> None:
    paths = _fixture(tmp_path); _initialize(paths)
    rejected = record_mapping_decision(paths["history"], "E001", "REJECTED", timestamp="2026-01-01T00:00:00+00:00")
    assert rejected["mask_quality_decision"] == "UNREVIEWED"
    assert is_complete(rejected)
    with pytest.raises(ValueError, match="only be decided"):
        record_mask_decision(paths["history"], "E001", "PASS")
    with pytest.raises(FileExistsError, match="explicit replacement"):
        record_mapping_decision(paths["history"], "E001", "UNCERTAIN")
    changed = record_mapping_decision(paths["history"], "E001", "UNCERTAIN", replace_existing=True)
    assert changed["mapping_decision"] == "UNCERTAIN"


def test_confirmed_allows_every_mask_decision(tmp_path: Path) -> None:
    for decision in ("PASS", "FAIL", "UNCERTAIN"):
        case = tmp_path / decision; paths = _fixture(case); _initialize(paths)
        queue = build_review_queue(paths["manifest"], paths["mapping"], paths["quality"],
                                   paths["candidates"], paths["diagnostics"], paths["history"])
        record_mapping_decision(paths["history"], "A002", "CONFIRMED", queue[1]["candidates"][0])
        assert record_mask_decision(paths["history"], "A002", decision)["mask_quality_decision"] == decision


def test_navigation_status_counts_and_provisional_breakdown(tmp_path: Path) -> None:
    paths = _fixture(tmp_path); _initialize(paths)
    queue = build_review_queue(paths["manifest"], paths["mapping"], paths["quality"],
                               paths["candidates"], paths["diagnostics"], paths["history"])
    assert cycle_candidate(0, 3, -1) == 2
    assert cycle_candidate(2, 3, 1) == 0
    assert previous_index(2, 3) == 1
    assert next_unresolved_index(queue, 0) == 1
    record_mapping_decision(paths["history"], "A001", "REJECTED")
    queue[0]["history"] = load_history(paths["history"])["A001"]
    summary = status_summary(paths["manifest"], paths["mapping"], paths["quality"], paths["history"])
    assert summary["total_independent_gt"] == 114
    assert summary["auto_accepted_existing"] == 111
    assert summary["human_review_required"] == 3
    assert summary["human_review_completed"] == 1
    assert summary["human_review_remaining"] == 2
    assert summary["provisional_module3_usable"] == 111
    assert sum(summary["provisional_by_class"].values()) == 111


def test_render_and_history_writes_do_not_modify_inputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path); _initialize(paths)
    protected = [paths[key] for key in ("manifest", "mapping", "quality", "labels", "candidates")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    queue = build_review_queue(paths["manifest"], paths["mapping"], paths["quality"],
                               paths["candidates"], paths["diagnostics"], paths["history"])
    source = tmp_path / queue[0]["annotation"]["source_image"]
    crop = tmp_path / queue[0]["annotation"]["crop_filename"]
    cv2.imwrite(str(source), np.full((100, 120, 3), 100, np.uint8)); cv2.imwrite(str(crop), np.full((30, 20, 3), 180, np.uint8))
    screen = render_review_screen(queue[0], 0, source, crop)
    assert screen.shape == (1120, 1800, 3)
    record_mapping_decision(paths["history"], "A001", "UNCERTAIN")
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
