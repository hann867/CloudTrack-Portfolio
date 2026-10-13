from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.dataset_preparation.random_forest_pilot import MODEL_FEATURES, RF_PARAMETERS
from src.geometric_classification.production_module2_rf_pilot import (
    MISCLASSIFICATION_COLUMNS,
    load_production_feature_dataset,
    production_source_heldout,
    run_production_grouped_oof,
    run_production_module2_rf_pilot,
)


def _dataset(rows_per_class: int = 6) -> pd.DataFrame:
    rows = []
    for class_name, prefix in (("Alpha", "A"), ("Electron", "E")):
        for index in range(rows_per_class):
            source = "AWAN" if index < rows_per_class // 2 else "Lecturer"
            roi = "rotated" if index == 0 else "axis_aligned"
            scale = 1.0 if class_name == "Alpha" else 2.0
            rows.append({
                "ground_truth_track_id": f"{prefix}{index:03d}",
                "ground_truth_class": class_name,
                "source_dataset": source,
                "source_image": f"source_{prefix}{index:03d}.png",
                "event_id": f"event_{prefix}{index:03d}",
                "roi_type": roi,
                "ground_truth_quality": "high",
                "module2_track_id": f"module2_{prefix}{index:03d}",
                "segmentation_source": "production_module2",
                "processing_status": "success",
                "track_length_px": scale * (20 + index),
                "mean_width_px": scale * 3,
                "median_width_px": scale * 3.1,
                "max_width_px": scale * 4,
                "mean_curvature": scale * 0.05,
                "max_curvature": scale * 0.1,
                "tortuosity": 1.05 + scale * 0.05,
            })
    return pd.DataFrame(rows)


def test_loader_uses_exact_seven_predictors_and_production_rows_only(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _dataset().to_csv(path, index=False)

    frame = load_production_feature_dataset(path)

    assert MODEL_FEATURES == [
        "track_length_px", "mean_width_px", "median_width_px", "max_width_px",
        "mean_curvature", "max_curvature", "tortuosity",
    ]
    assert frame[MODEL_FEATURES].shape == (12, 7)
    assert set(frame["class"]) == {"Alpha", "Electron"}
    assert RF_PARAMETERS == {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}


def test_loader_rejects_nonproduction_or_failed_rows(tmp_path: Path) -> None:
    frame = _dataset()
    frame.loc[0, "segmentation_source"] = "temporary_module3_development"
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="segmentation_source"):
        load_production_feature_dataset(path)

    frame.loc[0, "segmentation_source"] = "production_module2"
    frame.loc[0, "processing_status"] = "failed"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="processing_status"):
        load_production_feature_dataset(path)


def test_largest_grouped_split_has_both_classes_no_leakage_and_exact_oof_coverage(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _dataset().to_csv(path, index=False)
    frame = load_production_feature_dataset(path)

    result = run_production_grouped_oof(
        frame, bootstrap_repetitions=100, permutation_repeats=2
    )

    assert result.group_count == 12
    assert result.fold_count == 6
    assert result.leakage_detected is False
    assert (result.fold_audit[["test_alpha", "test_electron"]] > 0).all().all()
    assert len(result.predictions) == result.predictions["ground_truth_track_id"].nunique() == 12
    assert np.allclose(
        result.predictions[["probability_alpha", "probability_electron"]].sum(axis=1), 1.0
    )


def test_source_heldout_reports_both_directions(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _dataset().to_csv(path, index=False)
    frame = load_production_feature_dataset(path)

    heldout = production_source_heldout(frame)

    assert list(zip(heldout["train_source"], heldout["test_source"])) == [
        ("AWAN", "Lecturer"), ("Lecturer", "AWAN")
    ]
    assert set(["balanced_accuracy", "macro_f1", "alpha_as_alpha", "electron_as_electron"]).issubset(heldout)


def test_full_pilot_writes_only_new_outputs_and_preserves_input(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    _dataset().to_csv(features, index=False)
    before = features.read_bytes()
    output = tmp_path / "new_pilot"

    result = run_production_module2_rf_pilot(
        features,
        output,
        bootstrap_repetitions=100,
        permutation_repeats=2,
    )

    expected = {
        "production_rf_pilot_summary.txt",
        "production_rf_metrics.csv",
        "production_rf_confusion_matrix.csv",
        "production_rf_oof_predictions.csv",
        "production_rf_source_metrics.csv",
        "production_rf_roi_metrics.csv",
        "production_rf_source_heldout.csv",
        "production_rf_feature_importance.csv",
        "production_rf_misclassifications.csv",
        "plots/confusion_matrix.png",
        "plots/feature_importance.png",
    }
    assert {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} == expected
    assert features.read_bytes() == before
    oof = pd.read_csv(output / "production_rf_oof_predictions.csv")
    assert len(oof) == oof["ground_truth_track_id"].nunique() == 12
    errors = pd.read_csv(output / "production_rf_misclassifications.csv")
    assert errors.columns.tolist() == MISCLASSIFICATION_COLUMNS
    assert result["primary"].leakage_detected is False
    for plot in ("confusion_matrix.png", "feature_importance.png"):
        image = cv2.imread(str(output / "plots" / plot))
        assert image is not None and image.ndim == 3
