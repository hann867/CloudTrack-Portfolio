from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.dataset_preparation.random_forest_pilot import MODEL_FEATURES
from src.geometric_classification.classifier import (
    CLASSIFIER_FEATURES,
    validate_classifier_features,
)
from src.geometric_classification.model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
    AUTHORIZED_MODEL_TYPE,
    REQUIRED_MODEL_METADATA_FIELDS,
    validate_artifact_metadata,
    validate_class_names,
    validate_feature_names,
    validate_model_contract,
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


def valid_features() -> dict[str, float]:
    return {name: float(index + 1) for index, name in enumerate(EXPECTED_FEATURES)}


def valid_metadata() -> dict[str, object]:
    return {
        "model_type": AUTHORIZED_MODEL_TYPE,
        "model_version": "test-contract-v1",
        "model_stage": "development",
        "feature_names": list(EXPECTED_FEATURES),
        "class_names": list(AUTHORIZED_CLASS_NAMES),
        "random_state": 42,
        "n_estimators": 100,
        "class_weight": "balanced",
        "training_timestamp": "2026-08-31T00:00:00+00:00",
        "training_dataset_identifier": "test-dataset",
        "training_sample_count": 4,
        "training_class_counts": {"Alpha": 2, "Electron": 2},
        "training_source_counts": {"source-a": 3, "source-b": 1},
        "training_code_version": "test-commit",
    }


def model_declaration(*, features=EXPECTED_FEATURES, classes=AUTHORIZED_CLASS_NAMES):
    model_type = type(AUTHORIZED_MODEL_TYPE, (), {})
    model = model_type()
    model.feature_names_in_ = np.asarray(features, dtype=object)
    model.classes_ = np.asarray(classes, dtype=object)
    return model


def test_authorized_contract_has_exactly_seven_features_in_exact_order() -> None:
    assert AUTHORIZED_FEATURE_NAMES == EXPECTED_FEATURES
    assert len(AUTHORIZED_FEATURE_NAMES) == 7
    assert validate_feature_names(AUTHORIZED_FEATURE_NAMES) == tuple(EXPECTED_FEATURES)


def test_reordered_feature_contract_is_rejected() -> None:
    reordered = [EXPECTED_FEATURES[1], EXPECTED_FEATURES[0], *EXPECTED_FEATURES[2:]]
    with pytest.raises(ValueError, match="exactly, in order"):
        validate_feature_names(reordered)
    with pytest.raises(ValueError, match="exactly, in order"):
        validate_classifier_features({name: 1.0 for name in reordered})


def test_classifier_no_longer_accepts_legacy_four_feature_input() -> None:
    legacy = {
        "track_length_px": 1.0,
        "mean_width_px": 2.0,
        "mean_curvature": 3.0,
        "tortuosity": 4.0,
    }
    with pytest.raises(ValueError, match="Missing required Module 3 feature"):
        validate_classifier_features(legacy)


def test_extra_unauthorized_feature_is_rejected() -> None:
    features = valid_features()
    features["crop_area_px"] = 99.0
    with pytest.raises(ValueError, match="Unexpected Module 3 feature.*crop_area_px"):
        validate_classifier_features(features)


@pytest.mark.parametrize("missing", EXPECTED_FEATURES)
def test_each_missing_authorized_feature_is_rejected(missing: str) -> None:
    features = valid_features()
    del features[missing]
    with pytest.raises(ValueError, match=missing):
        validate_classifier_features(features)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_non_finite_feature_values_are_rejected(invalid: float) -> None:
    features = valid_features()
    features["max_curvature"] = invalid
    with pytest.raises(ValueError, match="finite"):
        validate_classifier_features(features)


def test_model_with_unexpected_feature_contract_is_rejected() -> None:
    model = model_declaration(features=[*EXPECTED_FEATURES[:-1], "unexpected_feature"])
    with pytest.raises(ValueError, match="Unexpected Module 3 feature contract"):
        validate_model_contract(model)


def test_unexpected_class_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected Module 3 class contract"):
        validate_class_names(["Alpha", "Gamma"])
    with pytest.raises(ValueError, match="Unexpected Module 3 class contract"):
        validate_model_contract(model_declaration(classes=["Alpha", "Gamma"]))


def test_classifier_and_pilot_share_the_authoritative_feature_object() -> None:
    assert CLASSIFIER_FEATURES is AUTHORIZED_FEATURE_NAMES
    assert MODEL_FEATURES is AUTHORIZED_FEATURE_NAMES


def test_complete_future_metadata_contract_is_accepted() -> None:
    metadata = valid_metadata()
    assert tuple(metadata) == REQUIRED_MODEL_METADATA_FIELDS
    assert validate_artifact_metadata(metadata) == metadata


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("model_type", "OtherClassifier"),
        ("model_stage", ""),
        ("feature_names", list(reversed(EXPECTED_FEATURES))),
        ("class_names", ["Electron", "Alpha"]),
        ("training_timestamp", "not-a-timestamp"),
        ("training_sample_count", 0),
        ("training_code_version", ""),
    ],
)
def test_malformed_future_metadata_is_rejected(field: str, invalid: object) -> None:
    metadata = valid_metadata()
    metadata[field] = invalid
    with pytest.raises(ValueError):
        validate_artifact_metadata(metadata)


def test_metadata_missing_any_required_field_is_rejected() -> None:
    for field in REQUIRED_MODEL_METADATA_FIELDS:
        metadata = deepcopy(valid_metadata())
        del metadata[field]
        with pytest.raises(ValueError, match=field):
            validate_artifact_metadata(metadata)
