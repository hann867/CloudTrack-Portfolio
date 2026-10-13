from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.geometric_classification.classifier import load_classifier
from src.geometric_classification.development_rf_trainer import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_SOURCE_COUNTS,
    MODEL_STAGE,
    MODEL_VERSION,
    RF_PARAMETERS,
    build_metadata,
    load_authorized_training_data,
    persist_development_artifacts,
    train_development_model,
    validate_saved_artifact,
)
from src.geometric_classification.model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
    REQUIRED_MODEL_METADATA_FIELDS,
)


@pytest.fixture(scope="module")
def development_outputs(tmp_path_factory: pytest.TempPathFactory):
    frame = load_authorized_training_data()
    model = train_development_model(frame)
    metadata = build_metadata(
        frame,
        training_timestamp="2026-08-31T00:00:00+00:00",
        training_code_version="test-commit",
    )
    directory = tmp_path_factory.mktemp("module3-development-rf")
    outputs = persist_development_artifacts(
        model,
        metadata,
        frame,
        model_path=directory / "model.joblib",
        metadata_path=directory / "model.metadata.json",
        summary_path=directory / "training_summary.txt",
    )
    return frame, model, metadata, outputs


def test_authorized_training_data_has_required_rows_classes_and_sources() -> None:
    frame = load_authorized_training_data()
    assert len(frame) == 29
    assert frame["ground_truth_class"].value_counts().to_dict() == EXPECTED_CLASS_COUNTS
    assert frame["source_dataset"].value_counts().to_dict() == {
        "Lecturer": EXPECTED_SOURCE_COUNTS["Lecturer"],
        "AWAN": EXPECTED_SOURCE_COUNTS["AWAN"],
    }


def test_training_uses_exact_authoritative_feature_order_and_parameters(development_outputs) -> None:
    _, model, _, _ = development_outputs
    assert list(model.feature_names_in_) == AUTHORIZED_FEATURE_NAMES
    assert len(model.feature_names_in_) == 7
    assert model.n_estimators == RF_PARAMETERS["n_estimators"]
    assert model.random_state == RF_PARAMETERS["random_state"]
    assert model.class_weight == RF_PARAMETERS["class_weight"]


def test_training_classes_match_authoritative_contract(development_outputs) -> None:
    _, model, metadata, _ = development_outputs
    assert tuple(model.classes_) == AUTHORIZED_CLASS_NAMES
    assert tuple(metadata["class_names"]) == AUTHORIZED_CLASS_NAMES


def test_metadata_is_complete_and_preserves_development_version(development_outputs) -> None:
    _, _, metadata, _ = development_outputs
    assert set(REQUIRED_MODEL_METADATA_FIELDS).issubset(metadata)
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["model_stage"] == MODEL_STAGE
    assert metadata["training_sample_count"] == 29
    assert metadata["training_class_counts"] == EXPECTED_CLASS_COUNTS
    assert metadata["training_source_counts"] == EXPECTED_SOURCE_COUNTS


def test_saved_artifact_reloads_and_returns_ordered_probabilities(development_outputs) -> None:
    frame, _, _, outputs = development_outputs
    artifact = validate_saved_artifact(outputs.model_path, outputs.metadata_path, frame.iloc[:4])
    probabilities = artifact.model.predict_proba(frame.loc[:3, AUTHORIZED_FEATURE_NAMES])
    assert probabilities.shape == (4, 2)
    assert tuple(artifact.model.classes_) == AUTHORIZED_CLASS_NAMES
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert artifact.metadata["model_version"] == MODEL_VERSION


def test_missing_metadata_sidecar_fails_clearly(development_outputs, tmp_path: Path) -> None:
    _, _, _, outputs = development_outputs
    with pytest.raises(FileNotFoundError, match="metadata sidecar not found"):
        load_classifier(outputs.model_path, tmp_path / "missing.metadata.json")


def test_corrupted_metadata_sidecar_fails_clearly(development_outputs, tmp_path: Path) -> None:
    _, _, _, outputs = development_outputs
    corrupt = tmp_path / "corrupt.metadata.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable or malformed"):
        load_classifier(outputs.model_path, corrupt)


def test_wrong_feature_contract_in_metadata_fails_clearly(development_outputs, tmp_path: Path) -> None:
    _, _, metadata, outputs = development_outputs
    wrong = dict(metadata)
    wrong["feature_names"] = list(reversed(AUTHORIZED_FEATURE_NAMES))
    sidecar = tmp_path / "wrong-features.metadata.json"
    sidecar.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected Module 3 feature contract"):
        load_classifier(outputs.model_path, sidecar)


def test_rf_predictions_are_deterministic_for_fixed_data_and_parameters(development_outputs) -> None:
    frame, first_model, _, _ = development_outputs
    second_model = train_development_model(frame)
    features = frame.loc[:, AUTHORIZED_FEATURE_NAMES]
    assert np.array_equal(first_model.predict(features), second_model.predict(features))
    assert np.array_equal(first_model.predict_proba(features), second_model.predict_proba(features))


def test_existing_artifact_is_not_silently_overwritten(development_outputs) -> None:
    frame, model, metadata, outputs = development_outputs
    before = hashlib.sha256(outputs.model_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        persist_development_artifacts(
            model,
            metadata,
            frame,
            model_path=outputs.model_path,
            metadata_path=outputs.metadata_path,
            summary_path=outputs.summary_path,
        )
    assert hashlib.sha256(outputs.model_path.read_bytes()).hexdigest() == before


def test_embedded_metadata_cannot_be_replaced_by_an_unrelated_sidecar(
    development_outputs, tmp_path: Path
) -> None:
    _, _, metadata, outputs = development_outputs
    unrelated = dict(metadata)
    unrelated["model_version"] = "module3_rf_development_v2"
    sidecar = tmp_path / "unrelated.metadata.json"
    sidecar.write_text(json.dumps(unrelated), encoding="utf-8")
    with pytest.raises(ValueError, match="embedded metadata and JSON sidecar disagree"):
        load_classifier(outputs.model_path, sidecar)
