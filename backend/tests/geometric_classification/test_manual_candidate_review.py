from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.geometric_classification.manual_candidate_review import (
    MAPPING_COLUMNS, REVIEW_RESULT_COLUMNS, ensure_review_results, generate_confirmed_mapping,
    load_review_queue, load_review_results, record_review_decision,
)


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _fixture(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    label_rows = [
        {"track_id": "A001", "class": "Alpha", "source_image": "high.png", "crop_filename": "A001.png"},
        {"track_id": "E001", "class": "Electron", "source_image": "review.png", "crop_filename": "E001.png"},
        {"track_id": "A002", "class": "Alpha", "source_image": "none.png", "crop_filename": "A002.png"},
        {"track_id": "E002", "class": "Electron", "source_image": "skip.png", "crop_filename": "E002.png"},
        {"track_id": "E003", "class": "Electron", "source_image": "no-match.png", "crop_filename": "E003.png"},
    ]
    _write(labels, list(label_rows[0]), label_rows)
    matches = tmp_path / "candidate_matching.csv"
    match_rows = []

    def candidate(gt, source, status, rank, candidate_id, score):
        return {"labelled_track_id": gt, "source_image": source, "confidence_status": status,
                "rank": str(rank), "candidate_track_id": candidate_id, "iou": str(score),
                "manual_roi_coverage": str(score + 0.01), "candidate_bbox_coverage": str(score + 0.02),
                "normalized_centroid_distance": str(1 - score), "spatial_score": str(score + 0.03)}

    match_rows.append(candidate("A001", "high.png", "HIGH", 1, "high_candidate", 0.90))
    match_rows.extend([candidate("E001", "review.png", "REVIEW", rank, f"review_candidate_{rank}", 0.8-rank/10)
                       for rank in (3, 1, 2)])
    match_rows.extend([candidate("A002", "none.png", "REVIEW", rank, f"none_candidate_{rank}", 0.7-rank/10)
                       for rank in (1, 2)])
    match_rows.append(candidate("E002", "skip.png", "REVIEW", 1, "skip_candidate_1", 0.5))
    match_rows.append(candidate("E003", "no-match.png", "NO_MATCH", 1, "irrelevant", 0.0))
    _write(matches, list(match_rows[0]), match_rows)
    return labels, matches


def test_review_queue_excludes_high_and_no_match_and_preserves_rank_order(tmp_path: Path) -> None:
    labels, matches = _fixture(tmp_path)
    queue = load_review_queue(labels, matches)
    assert [item["annotation"]["track_id"] for item in queue] == ["E001", "A002", "E002"]
    assert [row["candidate_track_id"] for row in queue[0]["candidates"]] == [
        "review_candidate_1", "review_candidate_2", "review_candidate_3"]


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_selecting_candidate_rank_records_exact_candidate(tmp_path: Path, rank: int) -> None:
    labels, matches = _fixture(tmp_path); results = tmp_path / f"results_{rank}.csv"
    item = load_review_queue(labels, matches)[0]
    row = record_review_decision(results, item, "CONFIRMED", selected_rank=rank,
                                 timestamp="2026-01-01T00:00:00+00:00")
    assert row["selected_candidate_track_id"] == f"review_candidate_{rank}"
    assert row["selected_rank"] == rank
    assert row["selected_candidate_iou"] == next(c["iou"] for c in item["candidates"] if int(c["rank"]) == rank)


def test_none_is_recorded_without_candidate_and_resume_skips_completed(tmp_path: Path) -> None:
    labels, matches = _fixture(tmp_path); results = tmp_path / "results.csv"
    queue = load_review_queue(labels, matches)
    record_review_decision(results, queue[1], "NONE", timestamp="2026-01-01T00:00:00+00:00")
    row = load_review_results(results)["A002"]
    assert row["review_status"] == "NONE"
    assert row["selected_candidate_track_id"] == ""
    resumed = load_review_queue(labels, matches, results)
    assert [item["annotation"]["track_id"] for item in resumed] == ["E001", "E002"]


def test_skipped_remains_unresolved_and_prior_decision_needs_explicit_replacement(tmp_path: Path) -> None:
    labels, matches = _fixture(tmp_path); results = tmp_path / "results.csv"
    item = load_review_queue(labels, matches)[0]
    record_review_decision(results, item, "SKIPPED", timestamp="2026-01-01T00:00:00+00:00")
    assert "E001" in [entry["annotation"]["track_id"] for entry in load_review_queue(labels, matches, results)]
    with pytest.raises(FileExistsError):
        record_review_decision(results, item, "CONFIRMED", selected_rank=2)
    replacement = record_review_decision(results, item, "CONFIRMED", selected_rank=2,
                                         replace_existing=True, timestamp="2026-01-02T00:00:00+00:00")
    assert replacement["selected_candidate_track_id"] == "review_candidate_2"


def test_final_mapping_contains_auto_high_and_manual_confirmed_only(tmp_path: Path) -> None:
    labels, matches = _fixture(tmp_path); results = tmp_path / "results.csv"; output = tmp_path / "mapping.csv"
    queue = load_review_queue(labels, matches)
    record_review_decision(results, queue[0], "CONFIRMED", selected_rank=3)
    record_review_decision(results, queue[1], "NONE")
    record_review_decision(results, queue[2], "SKIPPED")
    rows = generate_confirmed_mapping(labels, matches, results, output)
    assert [(row["ground_truth_track_id"], row["match_method"]) for row in rows] == [
        ("A001", "AUTO_HIGH"), ("E001", "MANUAL_REVIEW")]
    assert rows[1]["module2_track_id"] == "review_candidate_3"
    with output.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == MAPPING_COLUMNS


def test_review_writes_leave_labels_and_diagnostics_unchanged(tmp_path: Path) -> None:
    labels, matches = _fixture(tmp_path); results = tmp_path / "results.csv"
    diagnostic = tmp_path / "diagnostics" / "manifest.csv"
    diagnostic.parent.mkdir(); diagnostic.write_bytes(b"module2 diagnostic sentinel")
    labels_before, diagnostic_before = labels.read_bytes(), diagnostic.read_bytes()
    ensure_review_results(results)
    record_review_decision(results, load_review_queue(labels, matches)[0], "CONFIRMED", selected_rank=1)
    assert labels.read_bytes() == labels_before
    assert diagnostic.read_bytes() == diagnostic_before
    with results.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == REVIEW_RESULT_COLUMNS
