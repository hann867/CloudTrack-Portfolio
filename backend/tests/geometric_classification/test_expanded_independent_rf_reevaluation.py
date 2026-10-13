from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.geometric_classification.expanded_independent_rf_reevaluation import (
    FEATURES,
    RF_PARAMETERS,
    choose_splits,
    load_final_dataset,
    run_grouped_reevaluation,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/geometric_classification/expanded_human_approved_module3_features/final_independent/final_independent_module3_features.csv"


def test_final_dataset_contract_and_exact_predictors() -> None:
    frame = load_final_dataset(INPUT)
    assert len(frame) == 34
    assert frame.ground_truth_class.value_counts().to_dict() == {"Electron": 19, "Alpha": 15}
    assert frame.dependency_group_id.nunique() == 29
    assert frame.selected_candidate.nunique() == 34
    assert FEATURES == ["track_length_px", "mean_width_px", "median_width_px", "max_width_px",
                        "mean_curvature", "max_curvature", "tortuosity"]
    assert RF_PARAMETERS == {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}


def test_largest_sensible_splits_have_complete_group_isolation() -> None:
    frame = load_final_dataset(INPUT); splits = choose_splits(frame)
    assert len(splits) == 5
    coverage = np.zeros(len(frame), int)
    for train, test in splits:
        coverage[test] += 1
        assert set(frame.iloc[train].ground_truth_class) == {"Alpha", "Electron"}
        assert set(frame.iloc[test].ground_truth_class) == {"Alpha", "Electron"}
        assert set(frame.iloc[train].dependency_group_id).isdisjoint(frame.iloc[test].dependency_group_id)
        assert set(frame.iloc[train].selected_candidate).isdisjoint(frame.iloc[test].selected_candidate)
    assert np.all(coverage == 1)


def test_isolated_evaluation_writes_oof_outputs_without_mutating_input(tmp_path: Path) -> None:
    before = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    result = run_grouped_reevaluation(INPUT, tmp_path, bootstrap_repetitions=100, permutation_repeats=3)
    assert result["samples"] == 34 and result["folds"] == 5
    assert result["leakage_detected"] is False
    predictions = pd.read_csv(tmp_path / "oof_predictions.csv")
    assert len(predictions) == 34 and predictions.ground_truth_track_id.is_unique
    np.testing.assert_allclose(predictions[["probability_alpha","probability_electron"]].sum(axis=1), 1)
    for name in ("rf_reevaluation_summary.txt", "fold_assignments.csv", "confusion_matrix.csv",
                 "classification_report.csv", "feature_importance.csv", "misclassifications.csv",
                 "source_diagnostics.csv", "roi_diagnostics.csv", "source_video_diagnostics.csv"):
        assert (tmp_path / name).is_file()
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == before


def test_duplicate_mask_is_rejected_before_evaluation(tmp_path: Path) -> None:
    frame = pd.read_csv(INPUT); frame.loc[1, "selected_candidate"] = frame.loc[0, "selected_candidate"]
    altered = tmp_path / "altered.csv"; frame.to_csv(altered, index=False)
    with pytest.raises(ValueError, match="Duplicate Module 2 masks"):
        load_final_dataset(altered)

