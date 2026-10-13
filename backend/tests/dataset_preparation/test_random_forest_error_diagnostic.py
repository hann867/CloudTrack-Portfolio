from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.dataset_preparation.random_forest_error_diagnostic import (
    build_misclassified_table,
    error_summary,
    feature_typicality,
    load_diagnostic_data,
    width_summary,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    labels, features, predictions = [], [], []
    errors = {"A001", "E001"}
    for class_name, prefix, base in (("Alpha", "A", 10.0), ("Electron", "E", 2.0)):
        for number in range(1, 5):
            track_id = f"{prefix}{number:03d}"
            labels.append({
                "track_id": track_id, "class": class_name, "crop_filename": f"{track_id}.png",
                "source_dataset": "AWAN" if number < 3 else "Lecturer", "source_image": f"{track_id}.jpg",
                "roi_type": "axis_aligned" if number % 2 else "rotated", "ground_truth_quality": "high",
                "event_id": f"EVENT_{track_id}",
            })
            features.append({
                "track_id": track_id, "crop_width_px": 20 + number, "crop_height_px": 40 + number,
                "segmentation_source": "temporary_module3_development", "track_length_px": base + number,
                "mean_width_px": base + number / 10, "median_width_px": base + number / 10,
                "max_width_px": base + number / 10, "mean_curvature": base / 100,
                "max_curvature": base / 50, "tortuosity": 1 + base / 100,
            })
            predicted = ("Electron" if class_name == "Alpha" else "Alpha") if track_id in errors else class_name
            predictions.append({
                "track_id": track_id, "true_label": class_name, "predicted_label": predicted,
                "probability_alpha": 0.2 if predicted == "Electron" else 0.8,
                "probability_electron": 0.8 if predicted == "Electron" else 0.2, "fold": 1,
            })
    paths = tmp_path / "labels.csv", tmp_path / "features.csv", tmp_path / "predictions.csv"
    for path, rows in zip(paths, (labels, features, predictions)):
        pd.DataFrame(rows).to_csv(path, index=False)
    return paths


def test_loading_is_read_only_and_validates_expected_error_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    before = tuple(_hash(path) for path in paths)
    monkeypatch.setattr("src.dataset_preparation.random_forest_error_diagnostic.MISCLASSIFIED_IDS", ["A001", "E001"])
    frame = load_diagnostic_data(*paths)
    assert len(frame) == 8
    assert tuple(_hash(path) for path in paths) == before


def test_summaries_and_relative_typicality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr("src.dataset_preparation.random_forest_error_diagnostic.MISCLASSIFIED_IDS", ["A001", "E001"])
    frame = load_diagnostic_data(*paths)
    table = build_misclassified_table(frame)
    assert set(table.track_id) == {"A001", "E001"}
    assert "mean_width_px_true_class_percentile" in table
    assert feature_typicality(frame, "Alpha", "mean_width_px", 2.2) == "Electron"
    source = error_summary(frame, "source_dataset")
    assert source["n"].sum() == 8
    widths = width_summary(frame)
    assert set(widths.feature) == {"mean_width_px", "median_width_px", "max_width_px"}
    assert len(widths) == 12

