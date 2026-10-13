from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from src.geometric_classification.classifier import ClassifierArtifact
from src.geometric_classification.final_assignment_rf_trainer import (
    METADATA_ARTIFACT,
    MODEL_ARTIFACT,
    MODEL_STAGE,
    MODEL_VERSION,
)
from src.geometric_classification.model_contract import AUTHORIZED_CLASS_NAMES, AUTHORIZED_FEATURE_NAMES
from src.geometric_classification.production_inference import ModelArtifactCache, classify_module2_mask


def line_mask() -> np.ndarray:
    mask = np.zeros((80, 140), dtype=np.uint8)
    cv2.line(mask, (15, 40), (125, 40), 255, 5)
    return mask


@pytest.fixture(scope="module")
def real_cache() -> ModelArtifactCache:
    return ModelArtifactCache()


@pytest.fixture(scope="module")
def successful_result(real_cache: ModelArtifactCache) -> dict:
    return classify_module2_mask(line_mask(), artifact_cache=real_cache)


def test_valid_synthetic_mask_returns_successful_development_prediction(successful_result: dict) -> None:
    assert successful_result["processing_status"] == "success"
    assert successful_result["error_code"] is None
    assert successful_result["predicted_class"] in AUTHORIZED_CLASS_NAMES
    assert successful_result["model_version"] == MODEL_VERSION
    assert successful_result["model_stage"] == MODEL_STAGE
    assert successful_result["model"] == {
        "model_version": MODEL_VERSION,
        "model_stage": "development",
    }


def test_result_contains_exactly_seven_authoritative_features(successful_result: dict) -> None:
    assert list(successful_result["features"]) == AUTHORIZED_FEATURE_NAMES
    assert len(successful_result["features"]) == 7


def test_probabilities_are_ordered_normalized_and_define_confidence(successful_result: dict) -> None:
    probabilities = successful_result["probabilities"]
    assert tuple(probabilities) == AUTHORIZED_CLASS_NAMES
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert successful_result["probability_alpha"] == probabilities["Alpha"]
    assert successful_result["probability_electron"] == probabilities["Electron"]
    assert successful_result["confidence"] == probabilities[successful_result["predicted_class"]]


def test_module2_mask_is_not_modified_in_place(real_cache: ModelArtifactCache) -> None:
    mask = line_mask()
    original = mask.copy()
    result = classify_module2_mask(mask, artifact_cache=real_cache)
    assert result["processing_status"] == "success"
    np.testing.assert_array_equal(mask, original)


def test_temporary_segmentation_is_never_invoked(monkeypatch, real_cache: ModelArtifactCache) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("temporary segment_track must not be called")

    monkeypatch.setattr("src.geometric_classification.feature_extractor.segment_track", forbidden)
    assert classify_module2_mask(line_mask(), artifact_cache=real_cache)["processing_status"] == "success"


@pytest.mark.parametrize(
    ("mask", "error_code"),
    [
        (None, "missing_mask"),
        (np.empty((0, 0), dtype=np.uint8), "empty_mask"),
        (np.zeros((10, 10), dtype=np.uint8), "no_foreground"),
        (np.zeros((10, 10, 3), dtype=np.uint8), "malformed_mask"),
    ],
)
def test_invalid_masks_return_distinct_structured_failures(mask, error_code: str) -> None:
    result = classify_module2_mask(mask, artifact_cache=ModelArtifactCache())
    assert result["processing_status"] == "failed"
    assert result["error_code"] == error_code
    assert result["predicted_class"] is None
    assert result["probabilities"] == {}


def test_geometric_extraction_failure_is_structured() -> None:
    def failed_geometry(_mask, mm_per_pixel=None):
        return {"processing_status": "failed", "error_message": "synthetic geometry failure"}

    result = classify_module2_mask(line_mask(), geometric_extractor=failed_geometry)
    assert result["error_code"] == "geometric_extraction_failure"
    assert "synthetic geometry failure" in result["error_message"]


def test_non_finite_feature_vector_fails_before_prediction() -> None:
    def non_finite_geometry(_mask, mm_per_pixel=None):
        result = {name: float(index + 1) for index, name in enumerate(AUTHORIZED_FEATURE_NAMES)}
        result.update({"processing_status": "success", "max_curvature": np.inf})
        return result

    result = classify_module2_mask(line_mask(), geometric_extractor=non_finite_geometry)
    assert result["error_code"] == "non_finite_feature_vector"


def test_missing_extracted_feature_fails_contract() -> None:
    def incomplete_geometry(_mask, mm_per_pixel=None):
        return {"processing_status": "success", "track_length_px": 1.0}

    result = classify_module2_mask(line_mask(), geometric_extractor=incomplete_geometry)
    assert result["error_code"] == "feature_contract_mismatch"


def test_missing_model_artifact_fails_clearly(tmp_path: Path) -> None:
    result = classify_module2_mask(
        line_mask(),
        model_path=tmp_path / "missing.joblib",
        metadata_path=METADATA_ARTIFACT,
        artifact_cache=ModelArtifactCache(),
    )
    assert result["error_code"] == "missing_model_artifact"


def test_missing_metadata_fails_clearly(tmp_path: Path) -> None:
    result = classify_module2_mask(
        line_mask(),
        model_path=MODEL_ARTIFACT,
        metadata_path=tmp_path / "missing.metadata.json",
        artifact_cache=ModelArtifactCache(),
    )
    assert result["error_code"] == "missing_model_metadata"


def test_corrupt_model_artifact_fails_clearly(tmp_path: Path) -> None:
    model_path = tmp_path / "corrupt.joblib"
    model_path.write_bytes(b"not a joblib artifact")
    result = classify_module2_mask(
        line_mask(),
        model_path=model_path,
        metadata_path=METADATA_ARTIFACT,
        artifact_cache=ModelArtifactCache(),
    )
    assert result["error_code"] == "corrupt_model_artifact"


@pytest.mark.parametrize("content", [b"{not json", b"\xff\xfe\x00"])
def test_corrupt_metadata_fails_clearly(tmp_path: Path, content: bytes) -> None:
    metadata_path = tmp_path / "corrupt.metadata.json"
    metadata_path.write_bytes(content)
    result = classify_module2_mask(
        line_mask(),
        model_path=MODEL_ARTIFACT,
        metadata_path=metadata_path,
        artifact_cache=ModelArtifactCache(),
    )
    assert result["error_code"] == "corrupt_model_metadata"


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("feature_names", list(reversed(AUTHORIZED_FEATURE_NAMES)), "feature_contract_mismatch"),
        ("class_names", ["Electron", "Alpha"], "class_contract_mismatch"),
    ],
)
def test_sidecar_contract_mismatch_fails_clearly(
    tmp_path: Path, field: str, value: object, expected_code: str
) -> None:
    model_path = tmp_path / f"{field}.joblib"
    metadata_path = tmp_path / f"{field}.metadata.json"
    shutil.copy2(MODEL_ARTIFACT, model_path)
    metadata = json.loads(METADATA_ARTIFACT.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = classify_module2_mask(
        line_mask(),
        model_path=model_path,
        metadata_path=metadata_path,
        artifact_cache=ModelArtifactCache(),
    )
    assert result["error_code"] == expected_code


def test_supplied_provenance_and_calibration_are_preserved(real_cache: ModelArtifactCache) -> None:
    provenance = {
        "track_id": "frame_track_0007",
        "source_filename": "frame.png",
        "component_id": 17,
        "label_id": 2,
        "bbox": [10, 20, 100, 8],
    }
    original = copy.deepcopy(provenance)
    result = classify_module2_mask(
        line_mask(), provenance=provenance, mm_per_pixel=0.25, artifact_cache=real_cache
    )
    assert result["processing_status"] == "success"
    assert result["provenance"] == {**original, "mm_per_pixel": 0.25}
    assert provenance == original


def test_calibration_adds_reporting_units_without_changing_classifier_features(
    real_cache: ModelArtifactCache,
) -> None:
    uncalibrated = classify_module2_mask(line_mask(), artifact_cache=real_cache)
    calibrated = classify_module2_mask(
        line_mask(), mm_per_pixel=0.2, artifact_cache=real_cache
    )
    assert calibrated["features"] == uncalibrated["features"]
    assert calibrated["predicted_class"] == uncalibrated["predicted_class"]
    assert calibrated["confidence"] == uncalibrated["confidence"]
    physical = calibrated["physical_features"]
    assert physical["track_length_mm"] == pytest.approx(
        calibrated["features"]["track_length_px"] * 0.2
    )
    assert physical["mean_width_mm"] == pytest.approx(
        calibrated["features"]["mean_width_px"] * 0.2
    )
    assert "mean_curvature_per_mm" not in physical
    assert "max_curvature_per_mm" not in physical
    assert uncalibrated["physical_features"] == {}


def test_four_point_measurement_homography_does_not_change_classifier_features(
    real_cache: ModelArtifactCache,
) -> None:
    uncalibrated = classify_module2_mask(line_mask(), artifact_cache=real_cache)
    calibrated = classify_module2_mask(
        line_mask(),
        mm_per_pixel=0.15,
        provenance={
            "measurement_homography": [[0.2, 0, 0], [0, 0.1, 0], [0, 0, 1]],
            "roi_x0": 120,
            "roi_y0": 80,
        },
        artifact_cache=real_cache,
    )
    assert calibrated["features"] == uncalibrated["features"]
    assert calibrated["predicted_class"] == uncalibrated["predicted_class"]
    assert calibrated["physical_features"]["track_length_mm"] == pytest.approx(
        calibrated["features"]["track_length_px"] * 0.2,
        rel=0.03,
    )
    assert calibrated["confidence"] == uncalibrated["confidence"]
    assert calibrated["probabilities"] == uncalibrated["probabilities"]


def test_lazy_loading_reuses_one_validated_artifact(monkeypatch) -> None:
    import src.geometric_classification.production_inference as inference

    actual_loader = inference.load_classifier
    calls = 0

    def counting_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual_loader(*args, **kwargs)

    monkeypatch.setattr(inference, "load_classifier", counting_loader)
    cache = ModelArtifactCache()
    first = classify_module2_mask(line_mask(), artifact_cache=cache)
    second = classify_module2_mask(line_mask(), artifact_cache=cache)
    assert first == second
    assert calls == 1


class _StaticCache:
    def __init__(self, model):
        self.artifact = ClassifierArtifact(
            model=model,
            metadata={"model_version": MODEL_VERSION, "model_stage": MODEL_STAGE},
        )

    def load(self, *_args, **_kwargs):
        return self.artifact


class _PredictFailureModel:
    def predict(self, _frame):
        raise RuntimeError("synthetic predict failure")

    def predict_proba(self, _frame):
        raise AssertionError("predict_proba should not be reached")


class _MalformedProbabilityModel:
    def predict(self, _frame):
        return np.asarray(["Alpha"])

    def predict_proba(self, _frame):
        return np.asarray([[0.7, 0.7]])


def test_predict_failure_is_structured() -> None:
    result = classify_module2_mask(line_mask(), artifact_cache=_StaticCache(_PredictFailureModel()))
    assert result["error_code"] == "predict_failure"


def test_malformed_probability_output_is_structured() -> None:
    result = classify_module2_mask(line_mask(), artifact_cache=_StaticCache(_MalformedProbabilityModel()))
    assert result["error_code"] == "malformed_probability_output"
