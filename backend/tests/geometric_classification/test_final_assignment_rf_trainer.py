from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.geometric_classification.final_assignment_rf_trainer import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_DEPENDENCY_GROUPS,
    EXPECTED_SOURCE_CLASS_COUNTS,
    EXPECTED_SOURCE_COUNTS,
    MODEL_STAGE,
    MODEL_VERSION,
    RF_PARAMETERS,
    build_metadata,
    load_final_assignment_training_data,
    persist_final_assignment_artifacts,
    train_final_assignment_model,
    validate_saved_artifact,
)
from src.geometric_classification.model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
)


@pytest.fixture(scope="module")
def v2_outputs(tmp_path_factory: pytest.TempPathFactory):
    frame = load_final_assignment_training_data()
    model = train_final_assignment_model(frame)
    metadata = build_metadata(
        frame,
        training_timestamp="2026-09-01T00:00:00+00:00",
        training_code_version="test-commit",
    )
    directory = tmp_path_factory.mktemp("module3-development-rf-v2")
    outputs = persist_final_assignment_artifacts(
        model,
        metadata,
        frame,
        model_path=directory / "model.joblib",
        metadata_path=directory / "model.metadata.json",
        summary_path=directory / "training_summary.txt",
    )
    return frame, model, metadata, outputs


def test_frozen_final_assignment_dataset_contract() -> None:
    frame = load_final_assignment_training_data()
    assert len(frame) == 34
    assert frame["ground_truth_class"].value_counts().to_dict() == {
        "Electron": EXPECTED_CLASS_COUNTS["Electron"],
        "Alpha": EXPECTED_CLASS_COUNTS["Alpha"],
    }
    assert frame["source_dataset"].value_counts().to_dict() == {
        "Lecturer": EXPECTED_SOURCE_COUNTS["Lecturer"],
        "AWAN": EXPECTED_SOURCE_COUNTS["AWAN"],
    }
    source_class = {
        f"{source}|{class_name}": int(
            ((frame.source_dataset == source) & (frame.ground_truth_class == class_name)).sum()
        )
        for source in ("AWAN", "Lecturer")
        for class_name in AUTHORIZED_CLASS_NAMES
    }
    assert source_class == EXPECTED_SOURCE_CLASS_COUNTS
    assert frame["dependency_group_id"].nunique() == EXPECTED_DEPENDENCY_GROUPS
    assert frame["ground_truth_track_id"].is_unique
    assert frame["selected_candidate"].is_unique
    assert frame["mapping_decision"].eq("CONFIRMED").all()
    assert frame["mask_quality_decision"].eq("PASS").all()


def test_v2_uses_exact_feature_class_and_rf_contract(v2_outputs) -> None:
    _, model, _, _ = v2_outputs
    assert list(model.feature_names_in_) == AUTHORIZED_FEATURE_NAMES
    assert tuple(model.classes_) == AUTHORIZED_CLASS_NAMES
    assert model.n_estimators == RF_PARAMETERS["n_estimators"]
    assert model.random_state == RF_PARAMETERS["random_state"]
    assert model.class_weight == RF_PARAMETERS["class_weight"]


def test_v2_metadata_contains_reproducibility_and_limitations(v2_outputs) -> None:
    frame, _, metadata, _ = v2_outputs
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["model_stage"] == MODEL_STAGE
    assert metadata["training_sample_count"] == 34
    assert metadata["training_class_counts"] == EXPECTED_CLASS_COUNTS
    assert metadata["training_source_counts"] == EXPECTED_SOURCE_COUNTS
    assert metadata["dependency_group_count"] == EXPECTED_DEPENDENCY_GROUPS
    assert metadata["training_gt_ids"] == sorted(frame["ground_truth_track_id"].tolist())
    assert len(metadata["training_dataset_sha256"]) == 64
    assert metadata["preprocessing"] == "none"
    assert any("AWAN Electron = 0" in value for value in metadata["known_limitations"])
    assert any("source-generalizable" in value for value in metadata["known_limitations"])


def test_v2_saved_artifact_reloads_and_predicts_two_probabilities(v2_outputs) -> None:
    frame, _, metadata, outputs = v2_outputs
    artifact = validate_saved_artifact(outputs.model_path, outputs.metadata_path, frame)
    values = frame.loc[:, AUTHORIZED_FEATURE_NAMES]
    predictions = artifact.model.predict(values)
    probabilities = artifact.model.predict_proba(values)
    assert predictions.shape == (34,)
    assert probabilities.shape == (34, 2)
    assert np.isfinite(probabilities).all()
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert artifact.metadata == metadata


def test_v2_summary_does_not_present_training_accuracy(v2_outputs) -> None:
    _, _, _, outputs = v2_outputs
    summary = outputs.summary_path.read_text(encoding="utf-8")
    assert "Training-set accuracy was not calculated or reported." in summary
    assert "AWAN Electron = 0" in summary
    assert "Artifact reload validation: PASS" in summary


def test_v2_outputs_cannot_be_overwritten(v2_outputs) -> None:
    frame, model, metadata, outputs = v2_outputs
    before = hashlib.sha256(outputs.model_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        persist_final_assignment_artifacts(
            model,
            metadata,
            frame,
            model_path=outputs.model_path,
            metadata_path=outputs.metadata_path,
            summary_path=outputs.summary_path,
        )
    assert hashlib.sha256(outputs.model_path.read_bytes()).hexdigest() == before
