from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_preparation.random_forest_pilot import (
    MODEL_FEATURES,
    build_dependency_groups,
    choose_group_splits,
    load_pilot_dataset,
    run_grouped_oof,
    run_random_forest_pilot,
    stratified_metrics,
)


EXPECTED_FEATURES = [
    "track_length_px",
    "mean_width_px",
    "median_width_px",
    "max_width_px",
    "mean_curvature",
    "max_curvature",
    "tortuosity",
]


def _write_fixture(root: Path, count_per_class: int = 10) -> tuple[Path, Path]:
    labels, features = [], []
    for class_name, prefix, offset in (("Alpha", "A", 5.0), ("Electron", "E", 1.0)):
        for number in range(1, count_per_class + 1):
            track_id = f"{prefix}{number:03d}"
            labels.append(
                {
                    "track_id": track_id,
                    "class": class_name,
                    "event_id": f"EVENT_{track_id}",
                    "source_dataset": "AWAN" if number % 2 else "Lecturer",
                    "source_image": f"source_{track_id}.png",
                    "roi_type": "axis_aligned" if number % 2 else "rotated",
                    "ground_truth_quality": "high",
                }
            )
            features.append(
                {
                    "track_id": track_id,
                    "segmentation_source": "temporary_module3_development",
                    "processing_status": "success",
                    **{
                        feature: offset + number / 100 + feature_index / 1000
                        for feature_index, feature in enumerate(EXPECTED_FEATURES)
                    },
                }
            )
    labels_path, features_path = root / "labels.csv", root / "features.csv"
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    pd.DataFrame(features).to_csv(features_path, index=False)
    return labels_path, features_path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_feature_whitelist_excludes_all_metadata() -> None:
    assert MODEL_FEATURES == EXPECTED_FEATURES
    forbidden = {
        "source_dataset", "roi_type", "event_id", "source_image", "crop_filename",
        "ground_truth_quality", "track_id", "image_pixels", "background_statistics",
    }
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_feature_label_alignment_and_mismatch(tmp_path: Path) -> None:
    labels_path, features_path = _write_fixture(tmp_path)
    frame = load_pilot_dataset(labels_path, features_path)
    assert len(frame) == 20
    altered = pd.read_csv(features_path).iloc[:-1]
    altered.to_csv(features_path, index=False)
    with pytest.raises(ValueError, match="do not align"):
        load_pilot_dataset(labels_path, features_path)


def test_dependency_groups_prevent_event_and_source_image_leakage(tmp_path: Path) -> None:
    labels_path, features_path = _write_fixture(tmp_path)
    frame = load_pilot_dataset(labels_path, features_path)
    frame.loc[1, "event_id"] = frame.loc[0, "event_id"]
    frame.loc[3, "source_image"] = frame.loc[2, "source_image"]
    groups = build_dependency_groups(frame)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[2] == groups.iloc[3]
    for train, test in choose_group_splits(frame, groups):
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])


def test_oof_coverage_probability_validity_determinism_and_source_metrics(tmp_path: Path) -> None:
    labels_path, features_path = _write_fixture(tmp_path)
    frame = load_pilot_dataset(labels_path, features_path)
    first = run_grouped_oof(frame, bootstrap=False)
    second = run_grouped_oof(frame, bootstrap=False)
    assert len(first.predictions) == len(frame)
    assert first.predictions["track_id"].is_unique
    assert (first.predictions["fold"] > 0).all()
    probabilities = first.predictions[["probability_alpha", "probability_electron"]].to_numpy()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    source = stratified_metrics(first.predictions, "source_dataset")
    assert set(source["source_dataset"]) == {"AWAN", "Lecturer"}
    assert source["n"].sum() == len(frame)


def test_full_pilot_does_not_modify_input_csvs(tmp_path: Path) -> None:
    labels_path, features_path = _write_fixture(tmp_path)
    before = (_digest(labels_path), _digest(features_path))
    run_random_forest_pilot(labels_path, features_path, tmp_path / "outputs")
    assert (_digest(labels_path), _digest(features_path)) == before

