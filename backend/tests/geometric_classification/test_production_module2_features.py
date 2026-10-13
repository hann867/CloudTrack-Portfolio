from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.geometric_classification.production_module2_features import (
    FEATURES, OUTPUT_COLUMNS, extract_confirmed_module2_features, feature_sanity_issues,
    run_production_feature_extraction,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _feature_result(scale: float = 1.0) -> dict:
    values = {
        "track_length_px": 20.0 * scale, "mean_width_px": 3.0 * scale,
        "median_width_px": 3.0 * scale, "max_width_px": 4.0 * scale,
        "mean_curvature": 0.1 * scale, "max_curvature": 0.2 * scale,
        "tortuosity": 1.1, "endpoint_distance_px": 18.0 * scale,
        "mask_area_px": 50 * scale, "skeleton_pixels": 20 * scale,
    }
    return {**values, "processing_status": "success", "error_message": None,
            "track_length_mm": None, "skeleton": np.ones((2, 2)), "principal_path": np.ones((2, 2))}


def _fixture(tmp_path: Path, *, missing_second_mask: bool = False):
    labels = tmp_path / "labels.csv"
    label_rows = [
        {"track_id": "A1", "class": "Alpha", "source_dataset": "AWAN", "source_image": "one.png",
         "event_id": "event1", "roi_type": "axis_aligned", "ground_truth_quality": "high"},
        {"track_id": "E1", "class": "Electron", "source_dataset": "Lecturer", "source_image": "two.png",
         "event_id": "event2", "roi_type": "rotated", "ground_truth_quality": "high"},
    ]
    _write(labels, label_rows)
    mapping = tmp_path / "mapping.csv"
    _write(mapping, [
        {"ground_truth_track_id": "A1", "ground_truth_class": "Alpha", "source_image": "one.png",
         "match_method": "AUTO_HIGH", "module2_track_id": "m1"},
        {"ground_truth_track_id": "E1", "ground_truth_class": "Electron", "source_image": "two.png",
         "match_method": "MANUAL_REVIEW", "module2_track_id": "m2"},
    ])
    diagnostics = tmp_path / "diagnostics"
    for track, source, component in (("m1", "one.png", "11"), ("m2", "two.png", "22")):
        root = diagnostics / track
        mask_name = f"masks/{track}_mask.png"
        _write(root / "manifest.csv", [{"track_id": track, "source_filename": source,
                                         "component_id": component, "label_id": component,
                                         "diagnostic_mask_file": mask_name}])
        if not (missing_second_mask and track == "m2"):
            (root / "masks").mkdir()
            mask = np.zeros((12, 14), np.uint8); mask[3:9, 5:10] = 255
            assert cv2.imwrite(str(root / mask_name), mask)
    temporary = tmp_path / "temporary.csv"
    temporary_rows = []
    for track, class_name in (("A1", "Alpha"), ("E1", "Electron")):
        row = {"track_id": track, "class": class_name, "processing_status": "success"}
        row.update({feature: _feature_result()[feature] for feature in FEATURES})
        temporary_rows.append(row)
    _write(temporary, temporary_rows)
    return mapping, labels, diagnostics, temporary


def test_existing_module2_masks_are_passed_directly_and_metadata_is_preserved(tmp_path: Path) -> None:
    mapping, labels, diagnostics, _ = _fixture(tmp_path)
    received = []

    def extractor(mask):
        received.append(mask.copy())
        return _feature_result()

    rows = extract_confirmed_module2_features(mapping, labels, diagnostics, extractor=extractor)
    assert len(received) == 2
    assert set(np.unique(received[0])) == {0, 255}
    assert [row["segmentation_source"] for row in rows] == ["production_module2"] * 2
    assert rows[0]["module2_component_id"] == "11"
    assert rows[1]["roi_type"] == "rotated"
    assert "track_length_mm" not in rows[0]


def test_missing_mask_fails_one_mapping_without_aborting_batch(tmp_path: Path) -> None:
    mapping, labels, diagnostics, _ = _fixture(tmp_path, missing_second_mask=True)
    rows = extract_confirmed_module2_features(mapping, labels, diagnostics, extractor=lambda _: _feature_result())
    assert rows[0]["processing_status"] == "success"
    assert rows[1]["processing_status"] == "failed"
    assert "mask" in rows[1]["error_message"].lower()


def test_duplicate_mapping_ids_are_rejected(tmp_path: Path) -> None:
    mapping, labels, diagnostics, _ = _fixture(tmp_path)
    rows = list(csv.DictReader(mapping.open(encoding="utf-8", newline="")))
    _write(mapping, rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate ground_truth_track_id"):
        extract_confirmed_module2_features(mapping, labels, diagnostics)


def test_feature_sanity_detects_nan_invalid_width_length_and_tortuosity() -> None:
    row = {"ground_truth_track_id": "A1", "module2_track_id": "m1", "processing_status": "success"}
    row.update(_feature_result())
    row.update({"track_length_px": -1, "mean_width_px": float("nan"), "median_width_px": 0,
                "max_width_px": 0, "tortuosity": 0.9})
    duplicate = dict(row); duplicate["ground_truth_track_id"] = "A2"
    issues = [issue["issue"] for issue in feature_sanity_issues([row, duplicate])]
    assert "duplicate module2_track_id" in issues
    assert any("NaN/infinity" in issue for issue in issues)
    assert "invalid length" in issues
    assert "invalid width" in issues
    assert "tortuosity < 1" in issues


def test_full_run_creates_new_outputs_and_preserves_all_inputs(tmp_path: Path) -> None:
    mapping, labels, diagnostics, temporary = _fixture(tmp_path)
    protected = [mapping, labels, temporary, *diagnostics.rglob("*")]
    protected = [path for path in protected if path.is_file()]
    before = {path: path.read_bytes() for path in protected}
    output, comparison, audit = tmp_path / "new/features.csv", tmp_path / "new/comparison.csv", tmp_path / "new/audit.txt"
    result = run_production_feature_extraction(
        mapping, labels, diagnostics, temporary, output, comparison, audit,
        extractor=lambda mask: _feature_result(float(np.count_nonzero(mask)) / 30),
    )
    assert result["expected"] == result["successful"] == 2
    assert result["failed"] == 0
    assert result["ready_for_rf_pilot"] is True
    assert {path: path.read_bytes() for path in protected} == before
    with output.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == OUTPUT_COLUMNS
    assert comparison.is_file() and audit.is_file()
