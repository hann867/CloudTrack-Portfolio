"""Authoritative contract for future production Module 3 RF artifacts.

This module defines structure and validation only.  It does not train, load,
save, or run a classifier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np


AUTHORIZED_FEATURE_NAMES = [
    "track_length_px",
    "mean_width_px",
    "median_width_px",
    "max_width_px",
    "mean_curvature",
    "max_curvature",
    "tortuosity",
]
AUTHORIZED_CLASS_NAMES = ("Alpha", "Electron")
AUTHORIZED_MODEL_TYPE = "RandomForestClassifier"

REQUIRED_MODEL_METADATA_FIELDS = (
    "model_type",
    "model_version",
    "model_stage",
    "feature_names",
    "class_names",
    "random_state",
    "n_estimators",
    "class_weight",
    "training_timestamp",
    "training_dataset_identifier",
    "training_sample_count",
    "training_class_counts",
    "training_source_counts",
    "training_code_version",
)


def _ordered_names(values: Sequence[Any], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be an ordered sequence of names.")
    return tuple(str(value) for value in values)


def validate_feature_names(feature_names: Sequence[Any]) -> tuple[str, ...]:
    """Require exactly the authorized seven predictors in exact order."""
    actual = _ordered_names(feature_names, "feature_names")
    if actual != tuple(AUTHORIZED_FEATURE_NAMES):
        raise ValueError(
            "Unexpected Module 3 feature contract. Expected exactly, in order: "
            f"{list(AUTHORIZED_FEATURE_NAMES)}; received: {list(actual)}."
        )
    return actual


def validate_class_names(class_names: Sequence[Any]) -> tuple[str, ...]:
    """Require exactly Alpha then Electron."""
    actual = _ordered_names(class_names, "class_names")
    if actual != tuple(AUTHORIZED_CLASS_NAMES):
        raise ValueError(
            "Unexpected Module 3 class contract. Expected exactly, in order: "
            f"{list(AUTHORIZED_CLASS_NAMES)}; received: {list(actual)}."
        )
    return actual


def validate_feature_values(
    features: Mapping[str, Any],
    *,
    require_contract_order: bool = True,
) -> np.ndarray:
    """Return one finite seven-value vector after strict contract validation."""
    if not isinstance(features, Mapping):
        raise ValueError("Module 3 features must be supplied as a named mapping.")
    missing = [name for name in AUTHORIZED_FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing required Module 3 feature(s): {', '.join(missing)}.")
    unexpected = [name for name in features if name not in AUTHORIZED_FEATURE_NAMES]
    if unexpected:
        raise ValueError(f"Unexpected Module 3 feature(s): {', '.join(map(str, unexpected))}.")
    present_contract_names = tuple(name for name in features if name in AUTHORIZED_FEATURE_NAMES)
    if require_contract_order:
        validate_feature_names(present_contract_names)
    try:
        values = np.asarray([features[name] for name in AUTHORIZED_FEATURE_NAMES], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Module 3 feature values must be numeric: {exc}") from exc
    if values.shape != (len(AUTHORIZED_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Module 3 feature values must contain seven finite numbers.")
    return values


def _positive_integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Artifact metadata '{field}' must be an integer.")
    minimum = 0 if allow_zero else 1
    if int(value) < minimum:
        raise ValueError(f"Artifact metadata '{field}' must be >= {minimum}.")
    return int(value)


def _count_mapping(value: Any, field: str, *, required_keys: tuple[str, ...] | None = None) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Artifact metadata '{field}' must be a mapping of names to counts.")
    counts = {str(name): _positive_integer(count, f"{field}.{name}", allow_zero=True) for name, count in value.items()}
    if required_keys is not None and tuple(counts) != required_keys:
        raise ValueError(f"Artifact metadata '{field}' must have keys in order: {list(required_keys)}.")
    if not counts:
        raise ValueError(f"Artifact metadata '{field}' cannot be empty.")
    return counts


def validate_artifact_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate future persisted-model metadata without manufacturing defaults."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Classifier artifact metadata must be a mapping.")
    missing = [field for field in REQUIRED_MODEL_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"Classifier artifact metadata is missing: {', '.join(missing)}.")
    if metadata["model_type"] != AUTHORIZED_MODEL_TYPE:
        raise ValueError(f"Artifact model_type must be {AUTHORIZED_MODEL_TYPE}.")
    validate_feature_names(metadata["feature_names"])
    validate_class_names(metadata["class_names"])
    for field in ("model_version", "model_stage", "training_dataset_identifier"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"Artifact metadata '{field}' must be a non-empty string.")
    if not isinstance(metadata["random_state"], (int, np.integer)) or isinstance(metadata["random_state"], bool):
        raise ValueError("Artifact metadata 'random_state' must be an integer.")
    _positive_integer(metadata["n_estimators"], "n_estimators")
    if not isinstance(metadata["class_weight"], (str, Mapping)) or not metadata["class_weight"]:
        raise ValueError("Artifact metadata 'class_weight' must be a non-empty string or mapping.")
    if not isinstance(metadata["training_timestamp"], str) or not metadata["training_timestamp"].strip():
        raise ValueError("Artifact metadata 'training_timestamp' must be a non-empty ISO-8601 string.")
    try:
        datetime.fromisoformat(metadata["training_timestamp"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Artifact metadata 'training_timestamp' must be valid ISO-8601.") from exc
    sample_count = _positive_integer(metadata["training_sample_count"], "training_sample_count")
    class_counts = _count_mapping(
        metadata["training_class_counts"], "training_class_counts", required_keys=tuple(AUTHORIZED_CLASS_NAMES)
    )
    source_counts = _count_mapping(metadata["training_source_counts"], "training_source_counts")
    if sum(class_counts.values()) != sample_count:
        raise ValueError("training_class_counts must sum to training_sample_count.")
    if sum(source_counts.values()) != sample_count:
        raise ValueError("training_source_counts must sum to training_sample_count.")
    code_version = metadata["training_code_version"]
    if code_version is not None and (not isinstance(code_version, str) or not code_version.strip()):
        raise ValueError("training_code_version must be a non-empty string when available, or null.")
    return dict(metadata)


def validate_model_contract(model: Any) -> None:
    """Validate feature/class declarations on a future fitted sklearn model."""
    if type(model).__name__ != AUTHORIZED_MODEL_TYPE:
        raise ValueError(f"Model must be a fitted {AUTHORIZED_MODEL_TYPE}.")
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model lacks feature_names_in_; it was not fitted with the named production contract.")
    if not hasattr(model, "classes_"):
        raise ValueError("Model lacks classes_; it is not a fitted classifier.")
    validate_feature_names(tuple(str(name) for name in model.feature_names_in_))
    validate_class_names(tuple(str(name) for name in model.classes_))
