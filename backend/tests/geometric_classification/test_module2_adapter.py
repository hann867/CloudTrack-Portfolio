from __future__ import annotations

import copy

import cv2
import numpy as np

from src.geometric_classification.module2_adapter import adapt_module2_result


def _line_mask(offset: int = 0) -> np.ndarray:
    mask = np.zeros((60, 120), dtype=np.uint8)
    cv2.line(mask, (10, 30 + offset), (105, 30 + offset), 255, 5)
    return mask


def _roi(track_id: str, mask: np.ndarray) -> dict:
    return {
        "track_id": track_id,
        "source_path": "C:/input/frame.png",
        "source_filename": "frame.png",
        "component_id": 17,
        "label_id": 2,
        "source_component_ids": "4;9;12",
        "centroid_x": 58.5,
        "centroid_y": 30.0,
        "area_pixels": int(np.count_nonzero(mask)),
        "bbox_x": 10,
        "bbox_y": 27,
        "bbox_width": 96,
        "bbox_height": 7,
        "roi_x0": 2,
        "roi_y0": 19,
        "roi_x1_exclusive": 114,
        "roi_y1_exclusive": 46,
        "trajectory_linked": True,
        "trajectory_link_count": 2,
        "curve_fitted": False,
        "curve_model": "none",
        "_image_roi": np.zeros((60, 120, 3), dtype=np.uint8),
        "_mask_roi": mask,
    }


def test_valid_module2_mask_produces_geometry_and_preserves_provenance() -> None:
    result = adapt_module2_result({"rois": [_roi("frame_track_0002", _line_mask())]})
    assert len(result) == 1
    record = result[0]
    assert record["processing_status"] == "success"
    assert record["track_id"] == "frame_track_0002"
    assert record["features"]["track_length_px"] > 90
    assert record["features"]["mean_width_px"] > 0
    assert record["features"]["track_length_mm"] is None
    assert record["module2"]["component_id"] == 17
    assert record["module2"]["label_id"] == 2
    assert record["module2"]["source_component_ids"] == "4;9;12"
    assert "_mask_roi" not in record["module2"]
    assert "_image_roi" not in record["module2"]


def test_multiple_rois_produce_independent_records() -> None:
    result = adapt_module2_result(
        {"rois": [_roi("track_1", _line_mask(-5)), _roi("track_2", _line_mask(5))]}
    )
    assert [record["track_id"] for record in result] == ["track_1", "track_2"]
    assert all(record["processing_status"] == "success" for record in result)


def test_missing_mask_fails_gracefully() -> None:
    roi = _roi("missing", _line_mask())
    del roi["_mask_roi"]
    record = adapt_module2_result({"rois": [roi]})[0]
    assert record["processing_status"] == "failed"
    assert "missing" in record["error_message"].lower()


def test_empty_mask_fails_gracefully() -> None:
    record = adapt_module2_result({"rois": [_roi("empty", np.empty((0, 0), np.uint8))]})[0]
    assert record["processing_status"] == "failed"
    assert "empty" in record["error_message"].lower()


def test_mask_without_foreground_fails_gracefully() -> None:
    record = adapt_module2_result({"rois": [_roi("blank", np.zeros((20, 30), np.uint8))]})[0]
    assert record["processing_status"] == "failed"
    assert "no foreground" in record["error_message"].lower()


def test_malformed_mask_dimensions_fail_gracefully() -> None:
    record = adapt_module2_result({"rois": [_roi("colour", np.zeros((20, 30, 3), np.uint8))]})[0]
    assert record["processing_status"] == "failed"
    assert "must be 2-d" in record["error_message"].lower()


def test_failure_of_one_roi_does_not_abort_valid_roi() -> None:
    records = adapt_module2_result(
        {"rois": [_roi("bad", np.zeros((20, 30), np.uint8)), _roi("good", _line_mask())]}
    )
    assert [record["processing_status"] for record in records] == ["failed", "success"]


def test_geometric_extraction_failure_is_isolated() -> None:
    calls = 0

    def extractor(mask, mm_per_pixel=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"processing_status": "failed", "error_message": "synthetic geometry failure"}
        return {
            "processing_status": "success",
            "error_message": None,
            "track_length_px": 1.0,
            "track_length_mm": None,
        }

    records = adapt_module2_result(
        {"rois": [_roi("geometry_bad", _line_mask()), _roi("geometry_good", _line_mask())]},
        geometric_extractor=extractor,
    )
    assert [record["processing_status"] for record in records] == ["failed", "success"]
    assert "synthetic geometry failure" in records[0]["error_message"]


def test_temporary_segmentation_is_not_invoked(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("temporary segment_track must not be called")

    monkeypatch.setattr("src.geometric_classification.feature_extractor.segment_track", forbidden)
    record = adapt_module2_result({"rois": [_roi("direct_mask", _line_mask())]})[0]
    assert record["processing_status"] == "success"


def test_input_result_and_masks_are_not_modified_in_place() -> None:
    module2_result = {"metadata": {"source_kind": "grayscale"}, "rois": [_roi("stable", _line_mask())]}
    original = copy.deepcopy(module2_result)
    adapt_module2_result(module2_result)
    assert module2_result.keys() == original.keys()
    assert module2_result["metadata"] == original["metadata"]
    for key, value in module2_result["rois"][0].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, original["rois"][0][key])
        else:
            assert value == original["rois"][0][key]


def test_valid_upstream_scale_is_used_without_inventing_calibration() -> None:
    uncalibrated = adapt_module2_result({"rois": [_roi("px", _line_mask())]})[0]
    calibrated = adapt_module2_result(
        {"metadata": {"mm_per_pixel": 0.25}, "rois": [_roi("mm", _line_mask())]}
    )[0]
    assert uncalibrated["features"]["track_length_mm"] is None
    assert calibrated["module2"]["mm_per_pixel"] == 0.25
    assert calibrated["features"]["track_length_mm"] == calibrated["features"]["track_length_px"] * 0.25
