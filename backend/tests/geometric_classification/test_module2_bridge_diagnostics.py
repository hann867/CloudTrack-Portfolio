from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from backend.cli.geometric_classification import main_module2_module3_bridge as runner
from backend.src.geometric_classification.module2_adapter import adapt_module2_result
from backend.src.geometric_classification.module2_bridge_diagnostics import export_module2_bridge_diagnostics


def _roi(track_id: str, offset: int = 0) -> dict:
    image = np.full((50, 90, 3), 25 + offset, np.uint8)
    mask = np.zeros((50, 90), np.uint8)
    cv2.line(mask, (8, 25), (80, 25), 255, 5)
    return {
        "track_id": track_id,
        "source_filename": "real_frame.jpg",
        "source_stem": "real_frame",
        "component_id": 10 + offset,
        "label_id": 1 + offset,
        "area_pixels": int(np.count_nonzero(mask)),
        "bbox_x": 8,
        "bbox_y": 22,
        "bbox_width": 73,
        "bbox_height": 7,
        "roi_x0": 0,
        "roi_y0": 14,
        "roi_width": 90,
        "roi_height": 22,
        "touches_boundary": False,
        "trajectory_linked": False,
        "curve_fitted": False,
        "_image_roi": image,
        "_mask_roi": mask,
    }


def _export(tmp_path: Path, rois: list[dict]):
    module2_result = {"path": Path("real_frame.jpg"), "rois": rois}
    results = adapt_module2_result(module2_result)
    summary = export_module2_bridge_diagnostics(
        module2_result, results, tmp_path, source_stem="real_frame"
    )
    return module2_result, results, summary


def test_diagnostic_export_is_disabled_by_default() -> None:
    arguments = runner.parse_args(["real_frame.jpg"])
    assert arguments.export_diagnostics is False


def test_export_creates_image_mask_overlay_and_manifest(tmp_path: Path) -> None:
    _, _, summary = _export(tmp_path, [_roi("track_0001")])
    root = Path(summary["diagnostic_root"])
    assert (root / "images" / "track_0001.png").is_file()
    assert (root / "masks" / "track_0001_mask.png").is_file()
    assert (root / "overlays" / "track_0001_overlay.png").is_file()
    assert (root / "manifest.csv").is_file()


def test_exported_mask_preserves_dimensions_and_values(tmp_path: Path) -> None:
    roi = _roi("track_0001")
    original = roi["_mask_roi"].copy()
    _, _, summary = _export(tmp_path, [roi])
    exported = cv2.imread(
        str(Path(summary["diagnostic_root"]) / "masks" / "track_0001_mask.png"),
        cv2.IMREAD_UNCHANGED,
    )
    assert exported.shape == original.shape
    np.testing.assert_array_equal(exported, original)
    assert set(np.unique(exported)) == {0, 255}


def test_export_does_not_modify_source_module2_arrays(tmp_path: Path) -> None:
    roi = _roi("track_0001")
    before = copy.deepcopy(roi)
    _export(tmp_path, [roi])
    np.testing.assert_array_equal(roi["_image_roi"], before["_image_roi"])
    np.testing.assert_array_equal(roi["_mask_roi"], before["_mask_roi"])


def test_multiple_rois_export_independently_and_manifest_matches_ids(tmp_path: Path) -> None:
    _, _, summary = _export(tmp_path, [_roi("track_0001"), _roi("track_0002", 1)])
    root = Path(summary["diagnostic_root"])
    assert summary["exported_count"] == 2
    assert (root / "images" / "track_0001.png").is_file()
    assert (root / "images" / "track_0002.png").is_file()
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["track_id"] for row in rows] == ["track_0001", "track_0002"]
    assert all(row["processing_status"] == "success" for row in rows)
    assert all(row["track_length_px"] for row in rows)


def test_existing_runner_behavior_is_unchanged_without_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    image = tmp_path / "real_frame.jpg"
    image.write_bytes(b"placeholder")
    module2_result = {"accepted_component_count": 1, "rois": [_roi("track_0001")]}
    expected_tracks = adapt_module2_result(module2_result)
    monkeypatch.setattr(runner, "validate_config", lambda _config: None)
    monkeypatch.setattr(runner, "process_image", lambda _path, _config: module2_result)
    monkeypatch.setattr(runner, "adapt_module2_result", lambda _result: expected_tracks)

    def forbidden_export(*_args, **_kwargs):
        raise AssertionError("diagnostic exporter must stay disabled")

    monkeypatch.setattr(runner, "export_module2_bridge_diagnostics", forbidden_export)
    assert runner.main([str(image)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert "diagnostic_export" not in output
    assert output["tracks"] == expected_tracks

