"""Controlled trainer for the versioned Module 3 development RF artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .classifier import ClassifierArtifact, load_classifier, save_classifier, train_classifier
from .model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
    AUTHORIZED_MODEL_TYPE,
    validate_artifact_metadata,
    validate_class_names,
    validate_feature_names,
    validate_model_contract,
)


MODEL_VERSION = "module3_rf_development_v1"
MODEL_STAGE = "development"
RF_PARAMETERS = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}
EXPECTED_CLASS_COUNTS = {"Alpha": 15, "Electron": 14}
EXPECTED_SOURCE_COUNTS = {"AWAN": 10, "Lecturer": 19}
TRAINING_DATASET_IDENTIFIER = (
    "backend/data/processed/geometric_classification/production_module2_features/"
    "revised_confirmed_module2_geometric_features.csv"
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND_ROOT.parent
TRAINING_CSV = REPOSITORY_ROOT / TRAINING_DATASET_IDENTIFIER
MODEL_DIRECTORY = BACKEND_ROOT / "models" / "geometric_classification"
MODEL_ARTIFACT = MODEL_DIRECTORY / f"{MODEL_VERSION}.joblib"
METADATA_ARTIFACT = MODEL_DIRECTORY / f"{MODEL_VERSION}.metadata.json"
TRAINING_SUMMARY = MODEL_DIRECTORY / f"{MODEL_VERSION}_training_summary.txt"


@dataclass(frozen=True)
class DevelopmentArtifacts:
    model_path: Path
    metadata_path: Path
    summary_path: Path
    metadata: dict[str, Any]


def load_authorized_training_data(path: str | Path = TRAINING_CSV) -> pd.DataFrame:
    """Load and strictly gate the fixed 29-row development training dataset."""
    source = Path(path)
    if source.resolve() != TRAINING_CSV.resolve():
        raise ValueError(f"Development training is authorized only for: {TRAINING_CSV}")
    frame = pd.read_csv(source)
    required = [
        "ground_truth_class",
        "source_dataset",
        "segmentation_source",
        "processing_status",
        *AUTHORIZED_FEATURE_NAMES,
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Authorized training CSV is missing column(s): {', '.join(missing)}.")
    if len(frame) != sum(EXPECTED_CLASS_COUNTS.values()):
        raise ValueError(f"Authorized training CSV must contain exactly 29 rows; received {len(frame)}.")
    if frame["ground_truth_class"].isna().any():
        raise ValueError("Authorized training CSV contains a missing class label.")
    class_counts = {
        name: int((frame["ground_truth_class"] == name).sum()) for name in AUTHORIZED_CLASS_NAMES
    }
    if class_counts != EXPECTED_CLASS_COUNTS or set(frame["ground_truth_class"]) != set(AUTHORIZED_CLASS_NAMES):
        raise ValueError(
            f"Authorized training class counts must be {EXPECTED_CLASS_COUNTS}; received {class_counts}."
        )
    source_counts = {
        name: int((frame["source_dataset"] == name).sum()) for name in EXPECTED_SOURCE_COUNTS
    }
    if source_counts != EXPECTED_SOURCE_COUNTS or set(frame["source_dataset"]) != set(EXPECTED_SOURCE_COUNTS):
        raise ValueError(
            f"Authorized training source counts must be {EXPECTED_SOURCE_COUNTS}; received {source_counts}."
        )
    if set(frame["segmentation_source"]) != {"production_module2"}:
        raise ValueError("Authorized training rows must all use production_module2 segmentation.")
    if set(frame["processing_status"]) != {"success"}:
        raise ValueError("Authorized training rows must all have processing_status=success.")
    numeric = frame.loc[:, AUTHORIZED_FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Authorized training features must all be finite numeric values.")
    validated = frame.copy()
    validated.loc[:, AUTHORIZED_FEATURE_NAMES] = numeric
    return validated


def train_development_model(frame: pd.DataFrame) -> RandomForestClassifier:
    """Fit the fixed conservative baseline without evaluation or tuning."""
    validate_feature_names(list(frame.loc[:, AUTHORIZED_FEATURE_NAMES].columns))
    model = train_classifier(frame, label_column="ground_truth_class", **RF_PARAMETERS)
    validate_model_contract(model)
    return model


def current_git_commit(repository_root: str | Path = REPOSITORY_ROOT) -> str | None:
    """Return the current commit when Git is available; never invent a value."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def build_metadata(
    frame: pd.DataFrame,
    *,
    training_timestamp: str | None = None,
    training_code_version: str | None = None,
) -> dict[str, Any]:
    """Build complete metadata from the already-validated training rows."""
    timestamp = training_timestamp or datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_type": AUTHORIZED_MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "model_stage": MODEL_STAGE,
        "feature_names": list(AUTHORIZED_FEATURE_NAMES),
        "class_names": list(AUTHORIZED_CLASS_NAMES),
        "random_state": RF_PARAMETERS["random_state"],
        "n_estimators": RF_PARAMETERS["n_estimators"],
        "class_weight": RF_PARAMETERS["class_weight"],
        "training_timestamp": timestamp,
        "training_dataset_identifier": TRAINING_DATASET_IDENTIFIER,
        "training_sample_count": int(len(frame)),
        "training_class_counts": {
            name: int((frame["ground_truth_class"] == name).sum()) for name in AUTHORIZED_CLASS_NAMES
        },
        "training_source_counts": {
            name: int((frame["source_dataset"] == name).sum()) for name in EXPECTED_SOURCE_COUNTS
        },
        "training_code_version": training_code_version,
    }
    return validate_artifact_metadata(metadata)


def _preflight_new_paths(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing development artifact(s): {joined}")


def _training_summary(metadata: Mapping[str, Any], validation_passed: bool) -> str:
    features = metadata["feature_names"]
    return "\n".join(
        [
            f"Training samples: {metadata['training_sample_count']}",
            f"Alpha: {metadata['training_class_counts']['Alpha']}",
            f"Electron: {metadata['training_class_counts']['Electron']}",
            "",
            "Source counts:",
            f"AWAN: {metadata['training_source_counts']['AWAN']}",
            f"Lecturer: {metadata['training_source_counts']['Lecturer']}",
            "",
            f"Features: {len(features)}",
            "Feature order: " + ", ".join(features),
            "",
            "Parameters:",
            f"n_estimators: {metadata['n_estimators']}",
            f"random_state: {metadata['random_state']}",
            f"class_weight: {metadata['class_weight']}",
            "",
            f"Model stage: {metadata['model_stage']}",
            f"Artifact validation: {'PASS' if validation_passed else 'FAIL'}",
            "",
            "THIS MODEL IS A DEVELOPMENT ARTIFACT.",
            "THE GROUPED OOF PILOT, NOT TRAINING RESUBSTITUTION, IS THE CURRENT",
            "PERFORMANCE EVIDENCE.",
            "",
        ]
    )


def validate_saved_artifact(
    model_path: str | Path,
    metadata_path: str | Path,
    sample_rows: pd.DataFrame,
) -> ClassifierArtifact:
    """Reload and exercise class probabilities on valid named sample rows."""
    artifact = load_classifier(model_path, metadata_path)
    validate_model_contract(artifact.model)
    validate_feature_names(artifact.metadata["feature_names"])
    validate_class_names(artifact.metadata["class_names"])
    samples = sample_rows.loc[:, AUTHORIZED_FEATURE_NAMES]
    predictions = artifact.model.predict(samples)
    probabilities = artifact.model.predict_proba(samples)
    if predictions.shape != (len(samples),):
        raise ValueError("Reloaded development model returned an unexpected prediction shape.")
    if probabilities.shape != (len(samples), len(AUTHORIZED_CLASS_NAMES)):
        raise ValueError("Reloaded development model returned an unexpected probability shape.")
    if tuple(str(name) for name in artifact.model.classes_) != tuple(AUTHORIZED_CLASS_NAMES):
        raise ValueError("Reloaded development model probability order does not match the class contract.")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Reloaded development model probabilities do not sum to one.")
    return artifact


def persist_development_artifacts(
    model: RandomForestClassifier,
    metadata: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    model_path: str | Path = MODEL_ARTIFACT,
    metadata_path: str | Path = METADATA_ARTIFACT,
    summary_path: str | Path = TRAINING_SUMMARY,
) -> DevelopmentArtifacts:
    """Write new versioned outputs and immediately reload/validate them."""
    target_model = Path(model_path)
    target_metadata = Path(metadata_path)
    target_summary = Path(summary_path)
    _preflight_new_paths((target_model, target_metadata, target_summary))
    target_model.parent.mkdir(parents=True, exist_ok=True)
    save_classifier(model, metadata, target_model)
    target_metadata.write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")
    try:
        validate_saved_artifact(target_model, target_metadata, frame.iloc[: min(3, len(frame))])
    except Exception:
        target_summary.write_text(_training_summary(metadata, False), encoding="utf-8")
        raise
    target_summary.write_text(_training_summary(metadata, True), encoding="utf-8")
    return DevelopmentArtifacts(target_model, target_metadata, target_summary, dict(metadata))


def train_and_persist_development_model() -> DevelopmentArtifacts:
    """Run the single authorized v1 development training workflow."""
    _preflight_new_paths((MODEL_ARTIFACT, METADATA_ARTIFACT, TRAINING_SUMMARY))
    frame = load_authorized_training_data()
    model = train_development_model(frame)
    metadata = build_metadata(frame, training_code_version=current_git_commit())
    return persist_development_artifacts(model, metadata, frame)
