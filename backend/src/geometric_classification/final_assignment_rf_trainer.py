"""Controlled full-fit trainer for the Module 3 Development RF v2 artifact.

The grouped 34-row OOF evaluation remains the performance evidence.  This
module fits one deployable assignment artifact on all frozen eligible rows and
does not calculate training-set performance.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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


MODEL_VERSION = "module3_rf_development_v2"
MODEL_STAGE = "development"
RF_PARAMETERS = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}
EXPECTED_CLASS_COUNTS = {"Alpha": 15, "Electron": 19}
EXPECTED_SOURCE_COUNTS = {"AWAN": 6, "Lecturer": 28}
EXPECTED_SOURCE_CLASS_COUNTS = {
    "AWAN|Alpha": 6,
    "AWAN|Electron": 0,
    "Lecturer|Alpha": 9,
    "Lecturer|Electron": 19,
}
EXPECTED_DEPENDENCY_GROUPS = 29
EXPECTED_TRAINING_ROWS = 34

TRAINING_DATASET_IDENTIFIER = (
    "backend/data/processed/geometric_classification/"
    "expanded_human_approved_module3_features/final_independent/"
    "final_independent_module3_features.csv"
)
PERFORMANCE_EVIDENCE_IDENTIFIER = (
    "backend/data/processed/geometric_classification/"
    "expanded_independent_rf_reevaluation/rf_reevaluation_summary.txt"
)
KNOWN_LIMITATIONS = [
    "AWAN Electron = 0.",
    (
        "The grouped OOF evaluation is exploratory evidence and does not establish "
        "source-generalizable performance."
    ),
    (
        "The deployed full-fit artifact has no held-out predictions of its own; the preserved "
        "34-row grouped OOF evaluation remains the performance evidence."
    ),
]

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND_ROOT.parent
TRAINING_CSV = REPOSITORY_ROOT / TRAINING_DATASET_IDENTIFIER
PERFORMANCE_EVIDENCE = REPOSITORY_ROOT / PERFORMANCE_EVIDENCE_IDENTIFIER
MODEL_DIRECTORY = BACKEND_ROOT / "models" / "geometric_classification"
MODEL_ARTIFACT = MODEL_DIRECTORY / f"{MODEL_VERSION}.joblib"
METADATA_ARTIFACT = MODEL_DIRECTORY / f"{MODEL_VERSION}.metadata.json"
TRAINING_SUMMARY = MODEL_DIRECTORY / f"{MODEL_VERSION}_training_summary.txt"


@dataclass(frozen=True)
class FinalAssignmentArtifacts:
    model_path: Path
    metadata_path: Path
    summary_path: Path
    metadata: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(repository_root: str | Path = REPOSITORY_ROOT) -> str | None:
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
    return completed.stdout.strip() or None


def _source_class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        f"{source}|{class_name}": int(
            ((frame["source_dataset"] == source) & (frame["ground_truth_class"] == class_name)).sum()
        )
        for source in ("AWAN", "Lecturer")
        for class_name in AUTHORIZED_CLASS_NAMES
    }


def load_final_assignment_training_data(path: str | Path = TRAINING_CSV) -> pd.DataFrame:
    """Load and strictly gate the frozen final 34-row assignment dataset."""
    source = Path(path)
    if source.resolve() != TRAINING_CSV.resolve():
        raise ValueError(f"Development RF v2 training is authorized only for: {TRAINING_CSV}")
    frame = pd.read_csv(source, keep_default_na=False)
    required = [
        "ground_truth_track_id",
        "ground_truth_class",
        "source_dataset",
        "physical_track_group_id",
        "dependency_group_id",
        "selected_candidate",
        "segmentation_source",
        "mapping_decision",
        "mask_quality_decision",
        "processing_status",
        *AUTHORIZED_FEATURE_NAMES,
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Final assignment training CSV is missing: {', '.join(missing)}")
    if len(frame) != EXPECTED_TRAINING_ROWS:
        raise ValueError(f"Final assignment training CSV must contain 34 rows; received {len(frame)}.")
    if frame["ground_truth_track_id"].duplicated().any():
        raise ValueError("Final assignment training data contains duplicate ground-truth IDs.")
    if frame["selected_candidate"].duplicated().any():
        raise ValueError("Final assignment training data contains duplicate Module 2 masks.")
    if frame["ground_truth_track_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Final assignment training data contains a blank ground-truth ID.")
    if frame["dependency_group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Final assignment training data contains a blank dependency group.")

    class_counts = {
        name: int((frame["ground_truth_class"] == name).sum()) for name in AUTHORIZED_CLASS_NAMES
    }
    source_counts = {
        name: int((frame["source_dataset"] == name).sum()) for name in EXPECTED_SOURCE_COUNTS
    }
    if class_counts != EXPECTED_CLASS_COUNTS or set(frame["ground_truth_class"]) != set(AUTHORIZED_CLASS_NAMES):
        raise ValueError(f"Final assignment class counts changed: {class_counts}")
    if source_counts != EXPECTED_SOURCE_COUNTS or set(frame["source_dataset"]) != set(EXPECTED_SOURCE_COUNTS):
        raise ValueError(f"Final assignment source counts changed: {source_counts}")
    if _source_class_counts(frame) != EXPECTED_SOURCE_CLASS_COUNTS:
        raise ValueError("Final assignment source/class distribution changed.")
    if frame["dependency_group_id"].nunique() != EXPECTED_DEPENDENCY_GROUPS:
        raise ValueError("Final assignment dependency-group count changed.")
    if not frame["mapping_decision"].eq("CONFIRMED").all():
        raise ValueError("Every final assignment mapping must be CONFIRMED.")
    if not frame["mask_quality_decision"].eq("PASS").all():
        raise ValueError("Every final assignment mask-quality decision must be PASS.")
    if not frame["segmentation_source"].eq("production_module2_human_selected").all():
        raise ValueError("Every v2 row must use a human-selected production Module 2 mask.")
    if not frame["processing_status"].eq("SUCCESS").all():
        raise ValueError("Every final assignment feature row must have processing_status=SUCCESS.")

    physical = frame[frame["physical_track_group_id"].astype(str).str.strip().ne("")]
    if physical["physical_track_group_id"].duplicated().any():
        raise ValueError("Final assignment training data retains duplicate physical tracks.")

    numeric = frame.loc[:, AUTHORIZED_FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if values.shape != (EXPECTED_TRAINING_ROWS, len(AUTHORIZED_FEATURE_NAMES)):
        raise ValueError("Final assignment feature matrix has an unexpected shape.")
    if not np.isfinite(values).all():
        raise ValueError("Final assignment features contain NaN or infinity.")
    for feature in ("track_length_px", "mean_width_px", "median_width_px", "max_width_px"):
        if (numeric[feature] <= 0).any():
            raise ValueError(f"Final assignment feature {feature} must be positive.")
    if (numeric["tortuosity"] < 1).any():
        raise ValueError("Final assignment tortuosity must be at least one.")

    validated = frame.copy()
    validated.loc[:, AUTHORIZED_FEATURE_NAMES] = numeric
    return validated.sort_values("ground_truth_track_id").reset_index(drop=True)


def train_final_assignment_model(frame: pd.DataFrame) -> RandomForestClassifier:
    """Fit the frozen RF recipe on all eligible rows without evaluating training performance."""
    validate_feature_names(list(frame.loc[:, AUTHORIZED_FEATURE_NAMES].columns))
    model = train_classifier(frame, label_column="ground_truth_class", **RF_PARAMETERS)
    validate_model_contract(model)
    return model


def build_metadata(
    frame: pd.DataFrame,
    *,
    training_timestamp: str | None = None,
    training_code_version: str | None = None,
) -> dict[str, Any]:
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
        "training_dataset_sha256": sha256_file(TRAINING_CSV),
        "training_sample_count": int(len(frame)),
        "training_class_counts": {
            name: int((frame["ground_truth_class"] == name).sum())
            for name in AUTHORIZED_CLASS_NAMES
        },
        "training_source_counts": {
            name: int((frame["source_dataset"] == name).sum()) for name in EXPECTED_SOURCE_COUNTS
        },
        "training_source_class_counts": _source_class_counts(frame),
        "dependency_group_count": int(frame["dependency_group_id"].nunique()),
        "training_gt_ids": sorted(frame["ground_truth_track_id"].astype(str).tolist()),
        "rf_parameters": dict(RF_PARAMETERS),
        "preprocessing": "none",
        "performance_evidence_identifier": PERFORMANCE_EVIDENCE_IDENTIFIER,
        "performance_evidence_sha256": sha256_file(PERFORMANCE_EVIDENCE),
        "known_limitations": list(KNOWN_LIMITATIONS),
        "training_code_version": training_code_version,
    }
    return validate_artifact_metadata(metadata)


def _preflight_new_paths(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing Development RF v2 output(s): "
            + ", ".join(str(path) for path in existing)
        )


def validate_saved_artifact(
    model_path: str | Path,
    metadata_path: str | Path,
    sample_rows: pd.DataFrame,
) -> ClassifierArtifact:
    """Reload through the shared loader and exercise prediction/probability contracts."""
    artifact = load_classifier(model_path, metadata_path)
    validate_model_contract(artifact.model)
    validate_feature_names(artifact.metadata["feature_names"])
    validate_class_names(artifact.metadata["class_names"])
    if artifact.metadata["model_version"] != MODEL_VERSION:
        raise ValueError("Reloaded artifact is not Development RF v2.")
    if artifact.metadata["model_stage"] != MODEL_STAGE:
        raise ValueError("Reloaded artifact is not at the development stage.")
    samples = sample_rows.loc[:, AUTHORIZED_FEATURE_NAMES]
    predictions = np.asarray(artifact.model.predict(samples))
    probabilities = np.asarray(artifact.model.predict_proba(samples), dtype=float)
    if predictions.shape != (len(samples),):
        raise ValueError("Reloaded v2 model returned an unexpected prediction shape.")
    if probabilities.shape != (len(samples), len(AUTHORIZED_CLASS_NAMES)):
        raise ValueError("Reloaded v2 model must return two probabilities per row.")
    if tuple(str(name) for name in artifact.model.classes_) != tuple(AUTHORIZED_CLASS_NAMES):
        raise ValueError("Reloaded v2 probability order does not match the class contract.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Reloaded v2 model returned invalid probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Reloaded v2 probabilities do not sum to one.")
    return artifact


def _training_summary(metadata: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "MODULE 3 FINAL ASSIGNMENT MODEL - DEVELOPMENT RF V2",
            "",
            f"Model version: {metadata['model_version']}",
            f"Model stage: {metadata['model_stage']}",
            f"Training dataset: {metadata['training_dataset_identifier']}",
            f"Training dataset SHA-256: {metadata['training_dataset_sha256']}",
            f"Training samples: {metadata['training_sample_count']}",
            f"Alpha: {metadata['training_class_counts']['Alpha']}",
            f"Electron: {metadata['training_class_counts']['Electron']}",
            f"AWAN: {metadata['training_source_counts']['AWAN']}",
            f"Lecturer: {metadata['training_source_counts']['Lecturer']}",
            f"Dependency groups: {metadata['dependency_group_count']}",
            "",
            "Feature order: " + ", ".join(metadata["feature_names"]),
            f"RF parameters: {metadata['rf_parameters']}",
            "Preprocessing: none",
            "",
            "Training GT IDs:",
            ", ".join(metadata["training_gt_ids"]),
            "",
            "Performance evidence:",
            str(metadata["performance_evidence_identifier"]),
            f"Performance evidence SHA-256: {metadata['performance_evidence_sha256']}",
            "The preserved grouped OOF evaluation, not full-fit training predictions, is the performance evidence.",
            "Training-set accuracy was not calculated or reported.",
            "",
            "Known limitations:",
            *[f"- {limitation}" for limitation in metadata["known_limitations"]],
            "",
            f"Training timestamp: {metadata['training_timestamp']}",
            f"Training code version: {metadata['training_code_version'] or 'UNAVAILABLE'}",
            "Artifact reload validation: PASS",
            "",
        ]
    )


def persist_final_assignment_artifacts(
    model: RandomForestClassifier,
    metadata: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    model_path: str | Path = MODEL_ARTIFACT,
    metadata_path: str | Path = METADATA_ARTIFACT,
    summary_path: str | Path = TRAINING_SUMMARY,
) -> FinalAssignmentArtifacts:
    target_model = Path(model_path)
    target_metadata = Path(metadata_path)
    target_summary = Path(summary_path)
    _preflight_new_paths((target_model, target_metadata, target_summary))
    target_model.parent.mkdir(parents=True, exist_ok=True)
    save_classifier(model, metadata, target_model)
    target_metadata.write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")
    validate_saved_artifact(target_model, target_metadata, frame)
    target_summary.write_text(_training_summary(metadata), encoding="utf-8")
    return FinalAssignmentArtifacts(target_model, target_metadata, target_summary, dict(metadata))


def train_and_persist_final_assignment_model() -> FinalAssignmentArtifacts:
    """Run the single authorized full-fit Development RF v2 workflow."""
    _preflight_new_paths((MODEL_ARTIFACT, METADATA_ARTIFACT, TRAINING_SUMMARY))
    frame = load_final_assignment_training_data()
    model = train_final_assignment_model(frame)
    metadata = build_metadata(frame, training_code_version=current_git_commit())
    return persist_final_assignment_artifacts(model, metadata, frame)
