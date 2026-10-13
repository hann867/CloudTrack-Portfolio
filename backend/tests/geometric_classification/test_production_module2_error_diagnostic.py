from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.dataset_preparation.random_forest_pilot import MODEL_FEATURES
from src.geometric_classification.production_module2_error_diagnostic import (
    ERROR_IDS,
    build_feature_profiles,
    choose_comparators,
    classify_percentiles,
    load_diagnostic_frame,
)


def test_percentile_classification_rules_are_explicit() -> None:
    assert classify_percentiles(50, 60) == "OVERLAP"
    assert classify_percentiles(50, 96) == "ALPHA_TYPICAL"
    assert classify_percentiles(2, 4) == "EXTREME"


def test_profiles_have_all_seven_features_and_two_exact_percentiles() -> None:
    rows = []
    for class_name, prefix in (("Alpha", "A"), ("Electron", "E")):
        for index in range(5):
            row = {"ground_truth_track_id": f"{prefix}{index}", "true_class": class_name,
                   "predicted_class": class_name}
            row.update({feature: float(index + (0 if class_name == "Alpha" else 1)) for feature in MODEL_FEATURES})
            rows.append(row)
    frame = pd.DataFrame(rows)
    # Re-use the required IDs for the target-only selector.
    frame.loc[[0, 1, 5], "ground_truth_track_id"] = list(ERROR_IDS)
    profiles = build_feature_profiles(frame)
    assert len(profiles) == 3 * 7
    assert set(profiles.feature) == set(MODEL_FEATURES)
    assert profiles[["alpha_percentile_midrank", "electron_percentile_midrank"]].notna().all().all()


def test_comparators_use_only_correct_rows_and_return_three_per_class() -> None:
    rows = []
    for class_name, prefix in (("Alpha", "A"), ("Electron", "E")):
        for index in range(8):
            sample_id = f"{prefix}{index:02d}"
            row = {"ground_truth_track_id": sample_id, "true_class": class_name, "correct": True,
                   "source_dataset": "S", "roi_type": "axis_aligned"}
            row.update({feature: float(index + (0 if class_name == "Alpha" else 10)) for feature in MODEL_FEATURES})
            rows.append(row)
    frame = pd.DataFrame(rows)
    for index, sample_id in enumerate(ERROR_IDS):
        frame.loc[index, "ground_truth_track_id"] = sample_id
        frame.loc[index, "correct"] = False
    comparators = choose_comparators(frame)
    assert len(comparators) == 18
    assert set(comparators.comparator_track_id).isdisjoint(ERROR_IDS)
    assert (comparators.groupby(["misclassified_track_id", "comparator_class"]).size() == 3).all()
    assert (comparators.standardized_euclidean_distance >= 0).all()


def test_real_saved_inputs_are_read_only_and_expected_errors_match() -> None:
    backend = Path(__file__).resolve().parents[2]
    processed = backend / "data" / "processed"
    geometric = processed / "geometric_classification"
    paths = [
        geometric / "production_module2_features" / "revised_confirmed_module2_geometric_features.csv",
        geometric / "production_module2_rf_pilot" / "production_rf_oof_predictions.csv",
        processed / "labelled_tracks" / "labels.csv",
        geometric / "ground_truth_candidate_matching" / "confirmed_ground_truth_module2_mapping_revised.csv",
    ]
    before = [path.read_bytes() for path in paths]
    frame = load_diagnostic_frame(*paths)
    after = [path.read_bytes() for path in paths]
    assert before == after
    errors = frame.loc[~frame.correct]
    assert set(errors.ground_truth_track_id) == set(ERROR_IDS)
    expected = {"A018": (.45, .55), "A024": (.37, .63), "E022": (.94, .06)}
    for sample_id, probabilities in expected.items():
        row = errors.loc[errors.ground_truth_track_id == sample_id].iloc[0]
        assert np.allclose([row.probability_alpha, row.probability_electron], probabilities)
