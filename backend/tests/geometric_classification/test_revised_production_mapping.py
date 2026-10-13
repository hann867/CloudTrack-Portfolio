from __future__ import annotations

import copy
import csv
from pathlib import Path

import cv2
import numpy as np

from src.geometric_classification.revised_production_mapping import (
    EXCLUSION_SPECS,
    build_revised_mapping,
    audit_mapping_independence,
    extreme_feature_outliers,
    run_revised_production_workflow,
    strict_geometry_extractor,
)
from src.geometric_classification.production_module2_features import FEATURES


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _label(track_id: str, *, class_name: str = "Alpha", source: str = "AWAN") -> dict[str, str]:
    return {
        "track_id": track_id,
        "class": class_name,
        "crop_filename": f"{track_id}.png",
        "source_dataset": source,
        "source_image": f"{track_id}.jpg",
        "event_id": f"event_{track_id}",
        "roi_type": "axis_aligned",
        "ground_truth_quality": "high",
        "x": "10",
        "y": "10",
        "width": "20",
        "height": "30",
        "corner_1_x": "",
        "corner_1_y": "",
        "corner_2_x": "",
        "corner_2_y": "",
        "corner_3_x": "",
        "corner_3_y": "",
        "corner_4_x": "",
        "corner_4_y": "",
    }


def _mapping(track_id: str, *, class_name: str = "Alpha") -> dict[str, str]:
    return {
        "ground_truth_track_id": track_id,
        "ground_truth_class": class_name,
        "source_image": f"{track_id}.jpg",
        "match_method": "MANUAL_REVIEW",
        "module2_track_id": f"module2_{track_id}",
        "iou": "0.5",
        "manual_roi_coverage": "0.8",
        "candidate_coverage": "0.7",
        "normalized_centroid_distance": "0.1",
        "spatial_score": "0.7",
    }


def _feature_result() -> dict[str, object]:
    return {
        "processing_status": "success",
        "error_message": "",
        "track_length_px": 20.0,
        "mean_width_px": 3.0,
        "median_width_px": 3.0,
        "max_width_px": 4.0,
        "mean_curvature": 0.1,
        "max_curvature": 0.2,
        "tortuosity": 1.1,
        "endpoint_distance_px": 18.0,
        "mask_area_px": 50,
        "skeleton_pixels": 20,
    }


def test_revised_mapping_applies_only_fixed_adjudication_and_keeps_a003() -> None:
    track_ids = ["A003", *EXCLUSION_SPECS, "E001"]
    labels = [_label(track_id) for track_id in track_ids]
    mappings = [_mapping(track_id) for track_id in track_ids]
    before = copy.deepcopy(mappings)

    revised, excluded, columns = build_revised_mapping(mappings, labels)

    assert mappings == before
    assert [row["ground_truth_track_id"] for row in revised] == ["A003", "E001"]
    assert {row["ground_truth_track_id"] for row in excluded} == set(EXCLUSION_SPECS)
    assert next(row for row in excluded if row["ground_truth_track_id"] == "A008")[
        "exclusion_reason"
    ] == "DUPLICATE_PHYSICAL_TRACK"
    assert next(row for row in revised if row["ground_truth_track_id"] == "A003")[
        "duplicate_group"
    ] == "PHYSICAL_TRACK_A003_A008"
    assert columns[-3:] == ["mapping_status", "exclusion_reason", "duplicate_group"]


def test_independence_audit_flags_duplicate_identity_event_group_crop_and_roi(tmp_path: Path) -> None:
    labels = [_label("A1"), _label("A2")]
    labels[1].update({"source_image": labels[0]["source_image"], "event_id": labels[0]["event_id"]})
    revised = [
        {"ground_truth_track_id": "A1", "module2_track_id": "m1", "duplicate_group": "g1"},
        {"ground_truth_track_id": "A2", "module2_track_id": "m1", "duplicate_group": "g1"},
    ]
    (tmp_path / "A1.png").write_bytes(b"identical crop")
    (tmp_path / "A2.png").write_bytes(b"identical crop")

    result = audit_mapping_independence(revised, labels, tmp_path)

    assert result["duplicate_module2_ids"] == ["m1"]
    assert result["duplicate_physical_track_groups"] == ["g1"]
    assert result["duplicate_event_ids"] == ["event_A1"]
    assert result["duplicate_crop_hash_groups"] == [["A1", "A2"]]
    assert result["suspected_overlap_duplicates"][0]["status"] == "REVIEW_REQUIRED"
    assert result["review_required"] is True


def test_strict_geometry_validation_precedes_unchanged_extractor() -> None:
    calls: list[np.ndarray] = []

    def extractor(mask: np.ndarray) -> dict[str, object]:
        calls.append(mask.copy())
        return _feature_result()

    assert strict_geometry_extractor(np.zeros((3, 3, 1), np.uint8), extractor=extractor)[
        "processing_status"
    ] == "failed"
    assert strict_geometry_extractor(np.zeros((3, 3), np.uint8), extractor=extractor)[
        "processing_status"
    ] == "failed"
    mask = np.zeros((5, 5), np.uint8)
    mask[2, 2] = 255
    before = mask.copy()
    assert strict_geometry_extractor(mask, extractor=extractor)["processing_status"] == "success"
    assert len(calls) == 1
    assert np.array_equal(mask, before)


def test_extreme_outliers_are_descriptive_and_do_not_remove_rows() -> None:
    rows = []
    for index, value in enumerate([10.0] * 7 + [1000.0]):
        row: dict[str, object] = {
            "ground_truth_track_id": f"T{index}",
            "processing_status": "success",
        }
        row.update({feature: 2.0 for feature in FEATURES})
        row["track_length_px"] = value
        rows.append(row)
    before = copy.deepcopy(rows)

    outliers = extreme_feature_outliers(rows)

    assert rows == before
    assert outliers == []  # zero IQR is intentionally not treated as a reliable fence


def test_full_workflow_creates_new_outputs_without_changing_inputs(tmp_path: Path) -> None:
    track_ids = ["A003", *EXCLUSION_SPECS, "E001"]
    labels = [_label(track_id) for track_id in track_ids]
    labels[-1].update({"class": "Electron", "source_dataset": "Lecturer"})
    mappings = [_mapping(track_id) for track_id in track_ids]
    mappings[-1].update({"ground_truth_class": "Electron"})
    labels_csv = tmp_path / "labels.csv"
    mapping_csv = tmp_path / "mapping.csv"
    _write_csv(labels_csv, labels)
    _write_csv(mapping_csv, mappings)
    crop_root = tmp_path / "crops"
    crop_root.mkdir()
    for label in labels:
        (crop_root / label["crop_filename"]).write_bytes(label["track_id"].encode())
    diagnostics = tmp_path / "diagnostics"
    for track_id in ("A003", "E001"):
        root = diagnostics / track_id
        mask_relative = f"masks/module2_{track_id}_mask.png"
        _write_csv(root / "manifest.csv", [{
            "track_id": f"module2_{track_id}",
            "source_filename": f"{track_id}.jpg",
            "component_id": "1",
            "label_id": "1",
            "diagnostic_mask_file": mask_relative,
        }])
        (root / "masks").mkdir()
        mask = np.zeros((20, 20), np.uint8)
        mask[4:16, 9:12] = 255
        assert cv2.imwrite(str(root / mask_relative), mask)
    protected = [labels_csv, mapping_csv, *[path for path in diagnostics.rglob("*") if path.is_file()]]
    before = {path: path.read_bytes() for path in protected}
    outputs = {
        "revised": tmp_path / "new/revised.csv",
        "excluded": tmp_path / "new/excluded.csv",
        "overlap": tmp_path / "new/overlap.csv",
        "mapping_audit": tmp_path / "new/mapping_audit.txt",
        "features": tmp_path / "new/features.csv",
        "feature_audit": tmp_path / "new/feature_audit.txt",
    }

    result = run_revised_production_workflow(
        mapping_csv,
        labels_csv,
        diagnostics,
        crop_root,
        outputs["revised"],
        outputs["excluded"],
        outputs["overlap"],
        outputs["mapping_audit"],
        outputs["features"],
        outputs["feature_audit"],
        extractor=lambda _: _feature_result(),
    )

    assert result["revised_mapping"] == result["feature_success"] == 2
    assert result["excluded"] == 8
    assert result["unique_masks"] == 2
    assert {path: path.read_bytes() for path in protected} == before
    assert all(path.is_file() for path in outputs.values())
