from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.geometric_classification.confirmed_mapping_audit import (
    DISTRIBUTION_COLUMNS, EXCLUDED_COLUMNS, association_2x2, audit_confirmed_mapping,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _fixture(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    label_rows = [
        {"track_id": "A1", "class": "Alpha", "source_dataset": "AWAN", "source_image": "a1.png", "roi_type": "axis_aligned", "ground_truth_quality": "high"},
        {"track_id": "E1", "class": "Electron", "source_dataset": "AWAN", "source_image": "e1.png", "roi_type": "rotated", "ground_truth_quality": "medium"},
        {"track_id": "A2", "class": "Alpha", "source_dataset": "Lecturer", "source_image": "a2.png", "roi_type": "rotated", "ground_truth_quality": "low"},
        {"track_id": "E2", "class": "Electron", "source_dataset": "Lecturer", "source_image": "e2.png", "roi_type": "axis_aligned", "ground_truth_quality": "high"},
        {"track_id": "A3", "class": "Alpha", "source_dataset": "AWAN", "source_image": "a3.png", "roi_type": "axis_aligned", "ground_truth_quality": "high"},
        {"track_id": "E3", "class": "Electron", "source_dataset": "Lecturer", "source_image": "e3.png", "roi_type": "rotated", "ground_truth_quality": "high"},
    ]
    _write(labels, label_rows)
    mapping = tmp_path / "mapping.csv"
    mapping_rows = [
        {"ground_truth_track_id": "A1", "ground_truth_class": "Alpha", "source_image": "a1.png", "match_method": "AUTO_HIGH", "module2_track_id": "m1"},
        {"ground_truth_track_id": "E1", "ground_truth_class": "Electron", "source_image": "e1.png", "match_method": "MANUAL_REVIEW", "module2_track_id": "m2"},
        {"ground_truth_track_id": "A2", "ground_truth_class": "Alpha", "source_image": "a2.png", "match_method": "MANUAL_REVIEW", "module2_track_id": "m3"},
        {"ground_truth_track_id": "E2", "ground_truth_class": "Electron", "source_image": "e2.png", "match_method": "AUTO_HIGH", "module2_track_id": "m4"},
    ]
    _write(mapping, mapping_rows)
    matches = tmp_path / "candidate_matching.csv"
    match_rows = [
        {"labelled_track_id": "A1", "confidence_status": "HIGH"},
        {"labelled_track_id": "E1", "confidence_status": "REVIEW"},
        {"labelled_track_id": "A2", "confidence_status": "REVIEW"},
        {"labelled_track_id": "E2", "confidence_status": "HIGH"},
        {"labelled_track_id": "A3", "confidence_status": "REVIEW"},
        {"labelled_track_id": "E3", "confidence_status": "NO_MATCH"},
    ]
    _write(matches, match_rows)
    reviews = tmp_path / "reviews.csv"
    review_rows = [
        {"ground_truth_track_id": "E1", "review_status": "CONFIRMED"},
        {"ground_truth_track_id": "A2", "review_status": "CONFIRMED"},
        {"ground_truth_track_id": "A3", "review_status": "NONE"},
    ]
    _write(reviews, review_rows)
    return labels, mapping, matches, reviews


def test_known_2x2_association() -> None:
    result = association_2x2([[10, 3], [12, 12]])
    assert result["phi_cramers_v"] == pytest.approx(0.2617851572)
    assert result["odds_ratio"] == pytest.approx(10 / 3)
    assert 0 <= result["fisher_p_value"] <= 1


def test_audit_distributions_exclusions_and_outputs(tmp_path: Path) -> None:
    labels, mapping, matches, reviews = _fixture(tmp_path)
    report, distribution, excluded = tmp_path / "audit.txt", tmp_path / "distribution.csv", tmp_path / "excluded.csv"
    result = audit_confirmed_mapping(labels, mapping, matches, reviews, report, distribution, excluded)
    assert result["total_confirmed"] == 4
    assert result["class_counts"] == {"Alpha": 2, "Electron": 2}
    assert result["class_source"][("AWAN", "Alpha")] == 1
    assert result["class_source"][("Lecturer", "Electron")] == 1
    assert {row["track_id"]: row["reason"] for row in result["excluded"]} == {
        "A3": "MANUAL_NONE", "E3": "NO_MATCH"}
    assert report.is_file() and "FINAL RECOMMENDATION" in report.read_text(encoding="utf-8")
    with distribution.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == DISTRIBUTION_COLUMNS
    with excluded.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == EXCLUDED_COLUMNS


def test_audit_does_not_modify_any_input(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in inputs}
    audit_confirmed_mapping(*inputs, tmp_path / "audit.txt", tmp_path / "distribution.csv", tmp_path / "excluded.csv")
    assert {path: path.read_bytes() for path in inputs} == before


def test_duplicate_confirmed_track_is_rejected(tmp_path: Path) -> None:
    labels, mapping, matches, reviews = _fixture(tmp_path)
    rows = list(csv.DictReader(mapping.open(encoding="utf-8", newline="")))
    _write(mapping, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        audit_confirmed_mapping(labels, mapping, matches, reviews,
                                tmp_path / "audit.txt", tmp_path / "distribution.csv", tmp_path / "excluded.csv")
