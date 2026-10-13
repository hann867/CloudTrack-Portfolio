from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.geometric_classification.production_mapping_adjudication import (
    ALLOWED_ACTIONS,
    OUTPUT_COLUMNS,
    PROBLEM_TRACKS,
    build_adjudication_outputs,
    mask_diagnostics,
    write_adjudication,
)


def _metrics() -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for index, track_id in enumerate(PROBLEM_TRACKS):
        values[track_id] = {
            "_module2_track_id": f"module2_{index}",
            "mask_area_pixels": 1000 + index,
            "foreground_component_count": 1,
            "skeleton_endpoint_count": 200,
            "skeleton_branch_pixel_count": 800,
            "mask_fraction_inside_manual_roi": 0.0,
            "manual_roi_foreground_fraction": 0.0,
            "principal_path_status": "failed",
            "principal_path_error": "Skeleton is implausibly branch-heavy.",
        }
    values["E025"].update({
        "principal_path_status": "success",
        "principal_path_error": "",
        "principal_path_length_px": 930.7,
        "principal_endpoint_distance_px": 10.0,
        "computed_tortuosity": 93.07,
    })
    return values


def test_mask_diagnostics_reports_area_endpoints_and_manual_overlap() -> None:
    mask = np.zeros((40, 40), np.uint8)
    mask[8:32, 20] = 255
    annotation = {"roi_type": "axis_aligned", "x": "10", "y": "5", "width": "20", "height": "30"}
    diagnostic = {"roi_x0": "0", "roi_y0": "0", "area_pixels": "24"}

    result = mask_diagnostics(mask, annotation, diagnostic)

    assert result["mask_area_pixels"] == result["manifest_area_pixels"] == 24
    assert result["foreground_component_count"] == 1
    assert result["skeleton_endpoint_count"] == 2
    assert result["skeleton_branch_pixel_count"] == 0
    assert result["mask_fraction_inside_manual_roi"] == pytest.approx(1.0)
    assert result["principal_path_status"] == "success"


def test_adjudication_outputs_cover_exact_problem_set_and_allowed_actions() -> None:
    rows, report = build_adjudication_outputs(_metrics())

    assert {row["ground_truth_track_id"] for row in rows} == set(PROBLEM_TRACKS)
    assert len(rows) == len(PROBLEM_TRACKS)
    assert all(row["recommended_action"] in ALLOWED_ACTIONS for row in rows)
    assert all(row["geometry_review_needed"] == "NO" for row in rows)
    assert "Current successful unique masks: 31" in report
    assert "Expected unique masks after adjudication: 29 (estimate only)" in report
    assert "Ready to train Random Forest: NO" in report


def test_duplicate_and_extreme_geometry_recommendations_are_conservative() -> None:
    rows, _ = build_adjudication_outputs(_metrics())
    by_id = {row["ground_truth_track_id"]: row for row in rows}

    assert by_id["A003"]["recommended_action"] == "REVIEW_GROUND_TRUTH"
    assert by_id["A008"]["recommended_action"] == "REVIEW_GROUND_TRUTH"
    assert by_id["A025"]["manual_reselection_needed"] == "YES"
    assert by_id["A028"]["manual_reselection_needed"] == "YES"
    assert by_id["E025"]["recommended_action"] == "EXCLUDE_NO_VALID_MODULE2_CANDIDATE"
    assert "93.070000" in by_id["E025"]["notes"]


def test_write_adjudication_uses_exact_schema_and_leaves_inputs_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "protected.csv"
    source.write_text("protected\n", encoding="utf-8")
    before = source.read_bytes()
    rows, report = build_adjudication_outputs(_metrics())
    output_csv = tmp_path / "new" / "production_mapping_adjudication.csv"
    output_txt = tmp_path / "new" / "diagnostic_report.txt"

    write_adjudication(output_csv, output_txt, rows, report)

    with output_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == OUTPUT_COLUMNS
        assert len(list(reader)) == len(PROBLEM_TRACKS)
    assert output_txt.read_text(encoding="utf-8") == report
    assert source.read_bytes() == before


def test_write_adjudication_rejects_unknown_action(tmp_path: Path) -> None:
    rows, report = build_adjudication_outputs(_metrics())
    rows[0]["recommended_action"] = "AUTOMATICALLY_CHANGE_MAPPING"

    with pytest.raises(ValueError, match="Invalid recommended action"):
        write_adjudication(tmp_path / "out.csv", tmp_path / "out.txt", rows, report)
    assert not (tmp_path / "out.csv").exists()
