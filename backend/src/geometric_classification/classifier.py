"""Seven-feature Random Forest infrastructure; no production artifact exists yet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
    validate_artifact_metadata,
    validate_feature_values,
    validate_model_contract,
)


# Backwards-compatible name, now an alias to the single authoritative contract.
CLASSIFIER_FEATURES = AUTHORIZED_FEATURE_NAMES


@dataclass(frozen=True)
class ClassifierArtifact:
    model: RandomForestClassifier
    metadata: dict[str, Any]


def validate_classifier_features(features: Mapping[str, Any]) -> np.ndarray:
    """Validate one future inference row without classifying it."""
    return validate_feature_values(features)


def train_classifier(feature_table: pd.DataFrame, label_column: str = "label", **forest_options: Any) -> RandomForestClassifier:
    required = [*CLASSIFIER_FEATURES, label_column]
    missing = [column for column in required if column not in feature_table]
    if missing:
        raise ValueError(f"Training table is missing required column(s): {', '.join(missing)}.")
    labelled = feature_table.dropna(subset=required).copy()
    labelled = labelled[labelled[label_column].astype(str).str.strip() != ""]
    if labelled.empty or set(labelled[label_column].astype(str)) != set(AUTHORIZED_CLASS_NAMES):
        raise ValueError("Training data must contain exactly the authorized Alpha and Electron classes.")
    numeric = labelled.loc[:, CLASSIFIER_FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Training features must contain finite values for all seven authorized predictors.")
    options = {"n_estimators": 300, "random_state": 42, "class_weight": "balanced"}
    options.update(forest_options)
    model = RandomForestClassifier(**options)
    model.fit(numeric, labelled[label_column])
    validate_model_contract(model)
    return model


def predict_particle(model: RandomForestClassifier, features: pd.DataFrame | pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Generic validated probability helper; not wired as production inference."""
    validate_model_contract(model)
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("predict_particle requires exactly one feature row.")
        feature_mapping: Mapping[str, Any] = features.iloc[0].to_dict()
    elif isinstance(features, (dict, pd.Series)):
        feature_mapping = features
    else:
        raise ValueError("Features must be a mapping, Series, or one-row DataFrame.")
    values = validate_classifier_features(feature_mapping)
    frame = pd.DataFrame([values], columns=CLASSIFIER_FEATURES)
    probabilities = model.predict_proba(frame)[0]
    prediction = model.classes_[int(probabilities.argmax())]
    return {
        "predicted_class": str(prediction),
        "confidence": float(probabilities.max()),
        "probabilities": {
            str(label): float(probability) for label, probability in zip(model.classes_, probabilities)
        },
    }


def save_classifier(model: RandomForestClassifier, metadata: Mapping[str, Any], path: str | Path) -> None:
    """Persist only a fitted model that satisfies the future artifact contract."""
    validate_model_contract(model)
    validated_metadata = validate_artifact_metadata(metadata)
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing classifier artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": validated_metadata}, target)


def load_classifier(path: str | Path, metadata_path: str | Path | None = None) -> ClassifierArtifact:
    """Load an artifact and optionally require a matching JSON metadata sidecar."""
    payload = joblib.load(Path(path))
    if not isinstance(payload, Mapping) or set(payload) != {"model", "metadata"}:
        raise ValueError("Classifier artifact must contain exactly 'model' and 'metadata'.")
    model = payload["model"]
    validate_model_contract(model)
    metadata = validate_artifact_metadata(payload["metadata"])
    if tuple(str(name) for name in model.feature_names_in_) != tuple(metadata["feature_names"]):
        raise ValueError("Classifier model and artifact metadata feature contracts disagree.")
    if tuple(str(name) for name in model.classes_) != tuple(metadata["class_names"]):
        raise ValueError("Classifier model and artifact metadata class contracts disagree.")
    if metadata_path is not None:
        sidecar = Path(metadata_path)
        if not sidecar.is_file():
            raise FileNotFoundError(f"Classifier metadata sidecar not found: {sidecar}")
        try:
            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Classifier metadata sidecar is unreadable or malformed: {sidecar}") from exc
        validated_sidecar = validate_artifact_metadata(sidecar_payload)
        if validated_sidecar != metadata:
            raise ValueError("Classifier embedded metadata and JSON sidecar disagree.")
    return ClassifierArtifact(model=model, metadata=metadata)
