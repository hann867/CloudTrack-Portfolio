from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.geometric_classification.ground_truth_candidate_matcher import (
    confidence_status,
    manual_polygon,
    match_dataset,
    rank_candidates,
    spatial_metrics,
)


def _annotation(track_id: str = "A999", source: str = "frame.png") -> dict[str, str]:
    return {
        "track_id": track_id, "class": "Alpha", "source_image": source,
        "roi_type": "axis_aligned", "x": "10", "y": "20", "width": "100", "height": "200",
    }


def _candidate(track_id: str, x: int, y: int, width: int, height: int) -> dict[str, str]:
    return {
        "track_id": track_id, "component_id": "1", "label_id": "1",
        "bbox_x": str(x), "bbox_y": str(y), "bbox_width": str(width), "bbox_height": str(height),
        "roi_x0": str(x), "roi_y0": str(y), "roi_width": str(width), "roi_height": str(height),
        "_resolved_mask_file": "",
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def test_manual_regions_use_axis_fields_and_exact_rotated_corners() -> None:
    axis = manual_polygon(_annotation())
    assert cv2.contourArea(axis) == 20_000
    rotated = _annotation()
    rotated.update({"roi_type": "rotated", "corner_1_x": "0", "corner_1_y": "5",
                    "corner_2_x": "5", "corner_2_y": "0", "corner_3_x": "10", "corner_3_y": "5",
                    "corner_4_x": "5", "corner_4_y": "10"})
    np.testing.assert_array_equal(manual_polygon(rotated), [[0, 5], [5, 0], [10, 5], [5, 10]])


def test_spatial_metrics_have_expected_identical_box_values() -> None:
    metrics = spatial_metrics(manual_polygon(_annotation()), _candidate("track_1", 10, 20, 100, 200))
    assert metrics["intersection_area"] == 20_000
    assert metrics["iou"] == 1
    assert metrics["manual_roi_coverage"] == 1
    assert metrics["candidate_bbox_coverage"] == 1
    assert metrics["normalized_centroid_distance"] == 0
    assert metrics["spatial_score"] == 1


def test_ranking_is_spatial_only_and_returns_deterministic_top_five() -> None:
    annotation = _annotation()
    candidates = [_candidate(f"track_{index}", 10 + index * 3, 20, 100, 200) for index in range(7)]
    first = rank_candidates(annotation, candidates)[:5]
    annotation["class"] = "Electron"
    for index, candidate in enumerate(candidates):
        candidate.update({"mean_width_px": str(999 - index), "track_length_px": str(index),
                          "rf_prediction": "Electron" if index % 2 else "Alpha"})
    second = rank_candidates(annotation, reversed(candidates))[:5]
    assert [row["track_id"] for row in first] == [row["track_id"] for row in second]
    assert [row["track_id"] for row in first] == [f"track_{index}" for index in range(5)]
    assert [row["spatial_score"] for row in first] == [row["spatial_score"] for row in second]


def test_confidence_rules_cover_high_review_and_no_match() -> None:
    annotation = _annotation()
    high = rank_candidates(annotation, [_candidate("strong", 10, 20, 100, 200),
                                        _candidate("far", 1000, 1000, 20, 20)])
    assert confidence_status(high)[0] == "HIGH"
    ambiguous = rank_candidates(annotation, [_candidate("one", 10, 20, 100, 200),
                                             _candidate("two", 11, 20, 100, 200)])
    assert confidence_status(ambiguous)[0] == "REVIEW"
    none = rank_candidates(annotation, [_candidate("far", 1000, 1000, 20, 20)])
    assert confidence_status(none)[0] == "NO_MATCH"


def test_dataset_matcher_enforces_exact_source_and_writes_required_outputs(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    label_rows = [_annotation("A001", "frame.png"), _annotation("E001", "other.png")]
    _write_csv(labels, list(label_rows[0]), label_rows)
    diagnostics = tmp_path / "diagnostics" / "frame"
    candidate = _candidate("track_0040", 10, 20, 100, 200)
    candidate.update({"source_filename": "frame.png", "diagnostic_mask_file": "masks/track_0040_mask.png",
                      "diagnostic_image_file": "", "diagnostic_overlay_file": ""})
    manifest_columns = [key for key in candidate if not key.startswith("_")]
    _write_csv(diagnostics / "manifest.csv", manifest_columns,
               [{key: value for key, value in candidate.items() if not key.startswith("_")}])
    mask = np.full((200, 100), 255, np.uint8)
    (diagnostics / "masks").mkdir()
    assert cv2.imwrite(str(diagnostics / "masks" / "track_0040_mask.png"), mask)
    source_root = tmp_path / "sources"; source_root.mkdir()
    assert cv2.imwrite(str(source_root / "frame.png"), np.zeros((260, 140, 3), np.uint8))
    result = match_dataset(labels, tmp_path / "diagnostics", source_root, tmp_path / "output")
    assert result["total_labelled_tracks"] == 2
    assert result["high_confidence_matches"] == 1
    assert result["no_match"] == 1
    assert result["track_summaries"]["A001"]["candidate_track_id"] == "track_0040"
    assert result["track_summaries"]["E001"]["confidence_status"] == "NO_MATCH"
    for name in ("candidate_matching.csv", "high_confidence_matches.csv", "review_matches.csv",
                 "no_matches.csv", "matching_method.txt", "visual_confirmations/A001_confirmation.png"):
        assert (tmp_path / "output" / name).is_file()
