"""Production-style single-mask Module 3 inference with a development model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .classifier import ClassifierArtifact, load_classifier
from .final_assignment_rf_trainer import (
    METADATA_ARTIFACT,
    MODEL_ARTIFACT,
    MODEL_STAGE,
    MODEL_VERSION,
)
from .feature_extractor import extract_geometric_features
from .model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
    validate_class_names,
    validate_feature_names,
    validate_feature_values,
    validate_model_contract,
)


@dataclass(frozen=True)
class _ArtifactFingerprint:
    model_path: Path
    model_size: int
    model_mtime_ns: int
    metadata_path: Path
    metadata_size: int
    metadata_mtime_ns: int


class ArtifactLoadError(RuntimeError):
    """An expected, structured model-loading failure."""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code


def _contract_error_code(message: str) -> str:
    lowered = message.lower()
    if "feature contract" in lowered or "feature contracts disagree" in lowered:
        return "feature_contract_mismatch"
    if "class contract" in lowered or "class contracts disagree" in lowered:
        return "class_contract_mismatch"
    if "metadata" in lowered or "sidecar" in lowered:
        return "corrupt_model_metadata"
    if "randomforestclassifier" in lowered or "model must" in lowered:
        return "model_type_mismatch"
    return "corrupt_model_artifact"


class ModelArtifactCache:
    """Small injectable cache invalidated by path, size, or modification time."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._fingerprint: _ArtifactFingerprint | None = None
        self._artifact: ClassifierArtifact | None = None

    def clear(self) -> None:
        with self._lock:
            self._fingerprint = None
            self._artifact = None

    def load(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        expected_version: str = MODEL_VERSION,
        expected_stage: str = MODEL_STAGE,
    ) -> ClassifierArtifact:
        model_file = Path(model_path)
        metadata_file = Path(metadata_path)
        if not model_file.is_file():
            raise ArtifactLoadError("missing_model_artifact", f"Model artifact not found: {model_file}")
        if not metadata_file.is_file():
            raise ArtifactLoadError("missing_model_metadata", f"Model metadata not found: {metadata_file}")
        try:
            model_stat = model_file.stat()
            metadata_stat = metadata_file.stat()
        except OSError as exc:
            raise ArtifactLoadError("artifact_access_failure", f"Could not inspect model artifacts: {exc}") from exc
        fingerprint = _ArtifactFingerprint(
            model_file.resolve(),
            model_stat.st_size,
            model_stat.st_mtime_ns,
            metadata_file.resolve(),
            metadata_stat.st_size,
            metadata_stat.st_mtime_ns,
        )
        with self._lock:
            if fingerprint == self._fingerprint and self._artifact is not None:
                return self._artifact
            try:
                artifact = load_classifier(model_file, metadata_file)
            except FileNotFoundError as exc:
                code = "missing_model_metadata" if "metadata" in str(exc).lower() else "missing_model_artifact"
                raise ArtifactLoadError(code, str(exc)) from exc
            except ValueError as exc:
                raise ArtifactLoadError(_contract_error_code(str(exc)), str(exc)) from exc
            except Exception as exc:
                raise ArtifactLoadError(
                    "corrupt_model_artifact",
                    f"Model artifact could not be loaded: {type(exc).__name__}: {exc}",
                ) from exc
            try:
                validate_model_contract(artifact.model)
                validate_feature_names(artifact.metadata["feature_names"])
                validate_class_names(artifact.metadata["class_names"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactLoadError(_contract_error_code(str(exc)), str(exc)) from exc
            if artifact.metadata["model_stage"] != expected_stage:
                raise ArtifactLoadError(
                    "model_stage_mismatch",
                    f"Model stage must be {expected_stage}; received {artifact.metadata['model_stage']}.",
                )
            if artifact.metadata["model_version"] != expected_version:
                raise ArtifactLoadError(
                    "model_version_mismatch",
                    f"Model version must be {expected_version}; received {artifact.metadata['model_version']}.",
                )
            self._fingerprint = fingerprint
            self._artifact = artifact
            return artifact


DEFAULT_ARTIFACT_CACHE = ModelArtifactCache()

BATCH_STATUS_SUCCESS = "SUCCESS"
BATCH_STATUS_PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
BATCH_STATUS_FAILED = "FAILED"


def _reporting_physical_features(
    geometry: Mapping[str, Any],
    mm_per_pixel: float | None,
    *,
    mask: np.ndarray | None = None,
    measurement_homography: Any = None,
    roi_x0: int | float = 0,
    roi_y0: int | float = 0,
) -> dict[str, float]:
    """Add measurements without altering the RF's authoritative pixel features."""
    if measurement_homography is not None and mask is not None and geometry.get("principal_path") is not None:
        try:
            matrix = np.asarray(measurement_homography, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                return {}
            path_yx = np.asarray(geometry["principal_path"], dtype=np.float64)
            if path_yx.ndim != 2 or path_yx.shape[0] < 2 or path_yx.shape[1] != 2:
                return {}
            path_xy = np.column_stack((path_yx[:, 1] + float(roi_x0), path_yx[:, 0] + float(roi_y0)))
            physical_path = cv2.perspectiveTransform(path_xy.reshape((-1, 1, 2)), matrix).reshape((-1, 2))
            physical_length = float(np.linalg.norm(np.diff(physical_path, axis=0), axis=1).sum())

            binary = (np.asarray(mask) > 0).astype(np.uint8)
            radii = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[
                path_yx[:, 0].astype(int), path_yx[:, 1].astype(int)
            ]
            tangents = np.gradient(path_xy, axis=0)
            tangent_norms = np.linalg.norm(tangents, axis=1)
            valid = tangent_norms > 1e-9
            normals = np.zeros_like(tangents)
            normals[valid, 0] = -tangents[valid, 1] / tangent_norms[valid]
            normals[valid, 1] = tangents[valid, 0] / tangent_norms[valid]
            first_edges = path_xy - normals * radii[:, None]
            second_edges = path_xy + normals * radii[:, None]
            physical_first = cv2.perspectiveTransform(first_edges.reshape((-1, 1, 2)), matrix).reshape((-1, 2))
            physical_second = cv2.perspectiveTransform(second_edges.reshape((-1, 1, 2)), matrix).reshape((-1, 2))
            widths_mm = np.linalg.norm(physical_second - physical_first, axis=1)[valid]
            if not np.isfinite(physical_length) or widths_mm.size == 0 or not np.isfinite(widths_mm).all():
                return {}
            return {
                "track_length_mm": physical_length,
                "mean_width_mm": float(np.mean(widths_mm)),
                "median_width_mm": float(np.median(widths_mm)),
                "max_width_mm": float(np.max(widths_mm)),
            }
        except (cv2.error, IndexError, MemoryError, TypeError, ValueError):
            return {}
    if mm_per_pixel is None:
        return {}
    scale = float(mm_per_pixel)
    return {
        "track_length_mm": float(geometry["track_length_px"]) * scale,
        "mean_width_mm": float(geometry["mean_width_px"]) * scale,
        "median_width_mm": float(geometry["median_width_px"]) * scale,
        "max_width_mm": float(geometry["max_width_px"]) * scale,
    }


def _failure(
    error_code: str,
    message: str,
    *,
    provenance: Mapping[str, Any] | None = None,
    features: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "processing_status": "failed",
        "error_code": error_code,
        "error_message": message,
        "predicted_class": None,
        "confidence": None,
        "probability_alpha": None,
        "probability_electron": None,
        "probabilities": {},
        "features": dict(features or {}),
        "physical_features": {},
        "model": dict(model or {}),
        "provenance": dict(provenance or {}),
    }


def classify_module2_mask(
    mask: Any,
    *,
    provenance: Mapping[str, Any] | None = None,
    mm_per_pixel: float | None = None,
    model_path: str | Path = MODEL_ARTIFACT,
    metadata_path: str | Path = METADATA_ARTIFACT,
    artifact_cache: ModelArtifactCache = DEFAULT_ARTIFACT_CACHE,
    geometric_extractor: Callable[..., Mapping[str, Any]] = extract_geometric_features,
) -> dict[str, Any]:
    """Classify one existing 2-D Module 2 mask without re-segmentation."""
    if provenance is not None and not isinstance(provenance, Mapping):
        return _failure("invalid_provenance", "Provenance must be a mapping when supplied.")
    preserved_provenance = dict(provenance or {})
    if mm_per_pixel is not None:
        try:
            scale = float(mm_per_pixel)
        except (TypeError, ValueError):
            return _failure(
                "invalid_calibration",
                "mm_per_pixel must be a positive finite number.",
                provenance=preserved_provenance,
            )
        if not np.isfinite(scale) or scale <= 0:
            return _failure(
                "invalid_calibration",
                "mm_per_pixel must be a positive finite number.",
                provenance=preserved_provenance,
            )
        preserved_provenance.setdefault("mm_per_pixel", scale)
    else:
        scale = None
    if mask is None:
        return _failure("missing_mask", "Module 2 mask is missing.", provenance=preserved_provenance)
    try:
        mask_array = np.asarray(mask)
    except (TypeError, ValueError) as exc:
        return _failure(
            "malformed_mask",
            f"Module 2 mask is not array-like: {exc}",
            provenance=preserved_provenance,
        )
    if mask_array.ndim != 2:
        return _failure(
            "malformed_mask",
            f"Module 2 mask must be 2-D; received shape {mask_array.shape}.",
            provenance=preserved_provenance,
        )
    if mask_array.size == 0:
        return _failure("empty_mask", "Module 2 mask is empty.", provenance=preserved_provenance)
    try:
        numeric_mask = mask_array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        return _failure(
            "malformed_mask",
            f"Module 2 mask values must be numeric: {exc}",
            provenance=preserved_provenance,
        )
    if not np.isfinite(numeric_mask).all():
        return _failure(
            "malformed_mask",
            "Module 2 mask values must be finite.",
            provenance=preserved_provenance,
        )
    if not np.any(numeric_mask > 0):
        return _failure(
            "no_foreground",
            "Module 2 mask contains no foreground.",
            provenance=preserved_provenance,
        )

    try:
        geometry = geometric_extractor(mask_array.copy(), mm_per_pixel=None)
    except Exception as exc:
        return _failure(
            "geometric_extraction_failure",
            f"Module 3 geometric extraction raised {type(exc).__name__}: {exc}",
            provenance=preserved_provenance,
        )
    if not isinstance(geometry, Mapping):
        return _failure(
            "geometric_extraction_failure",
            "Module 3 geometric extraction returned no result mapping.",
            provenance=preserved_provenance,
        )
    if geometry.get("processing_status") != "success":
        return _failure(
            "geometric_extraction_failure",
            str(geometry.get("error_message") or "Module 3 geometric extraction failed."),
            provenance=preserved_provenance,
        )
    missing_features = [name for name in AUTHORIZED_FEATURE_NAMES if name not in geometry]
    if missing_features:
        return _failure(
            "feature_contract_mismatch",
            f"Geometric extraction is missing required feature(s): {', '.join(missing_features)}.",
            provenance=preserved_provenance,
        )
    features = {name: geometry[name] for name in AUTHORIZED_FEATURE_NAMES}
    physical_features = _reporting_physical_features(
        geometry,
        scale,
        mask=mask_array,
        measurement_homography=preserved_provenance.get("measurement_homography"),
        roi_x0=preserved_provenance.get("roi_x0", 0),
        roi_y0=preserved_provenance.get("roi_y0", 0),
    )
    try:
        feature_vector = validate_feature_values(features)
    except ValueError as exc:
        code = "non_finite_feature_vector" if "finite" in str(exc).lower() else "feature_contract_mismatch"
        return _failure(code, str(exc), provenance=preserved_provenance, features=features)

    try:
        artifact = artifact_cache.load(
            model_path,
            metadata_path,
            expected_version=MODEL_VERSION,
            expected_stage=MODEL_STAGE,
        )
    except ArtifactLoadError as exc:
        return _failure(
            exc.error_code,
            str(exc),
            provenance=preserved_provenance,
            features=features,
        )
    except Exception as exc:
        return _failure(
            "artifact_load_failure",
            f"Unexpected model loading failure: {type(exc).__name__}: {exc}",
            provenance=preserved_provenance,
            features=features,
        )

    model_info = {
        "model_version": artifact.metadata["model_version"],
        "model_stage": artifact.metadata["model_stage"],
    }
    frame = pd.DataFrame([feature_vector], columns=AUTHORIZED_FEATURE_NAMES)
    try:
        raw_prediction = np.asarray(artifact.model.predict(frame))
        raw_probabilities = np.asarray(artifact.model.predict_proba(frame), dtype=float)
    except Exception as exc:
        return _failure(
            "predict_failure",
            f"Model prediction failed: {type(exc).__name__}: {exc}",
            provenance=preserved_provenance,
            features=features,
            model=model_info,
        )
    if raw_prediction.shape != (1,) or str(raw_prediction[0]) not in AUTHORIZED_CLASS_NAMES:
        return _failure(
            "malformed_prediction_output",
            "Model returned an invalid prediction output.",
            provenance=preserved_provenance,
            features=features,
            model=model_info,
        )
    if (
        raw_probabilities.shape != (1, len(AUTHORIZED_CLASS_NAMES))
        or not np.isfinite(raw_probabilities).all()
        or np.any(raw_probabilities < 0)
        or np.any(raw_probabilities > 1)
        or not np.isclose(raw_probabilities[0].sum(), 1.0)
    ):
        return _failure(
            "malformed_probability_output",
            "Model returned invalid binary probabilities.",
            provenance=preserved_provenance,
            features=features,
            model=model_info,
        )
    predicted_class = str(raw_prediction[0])
    probabilities = {
        name: float(raw_probabilities[0, index]) for index, name in enumerate(AUTHORIZED_CLASS_NAMES)
    }
    return {
        "processing_status": "success",
        "error_code": None,
        "error_message": None,
        "predicted_class": predicted_class,
        "confidence": probabilities[predicted_class],
        "probability_alpha": probabilities["Alpha"],
        "probability_electron": probabilities["Electron"],
        "probabilities": probabilities,
        "features": features,
        "physical_features": physical_features,
        "model_version": model_info["model_version"],
        "model_stage": model_info["model_stage"],
        "model": model_info,
        "provenance": preserved_provenance,
    }


def summarize_module3_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return deterministic aggregate counts for ordered per-track results."""
    records = list(results)
    if any(not isinstance(result, Mapping) for result in records):
        raise ValueError("Every Module 3 batch result must be a mapping.")

    successful = [result for result in records if result.get("processing_status") == "success"]
    alpha_count = sum(result.get("predicted_class") == "Alpha" for result in successful)
    electron_count = sum(result.get("predicted_class") == "Electron" for result in successful)
    if alpha_count + electron_count != len(successful):
        raise ValueError("A successful Module 3 batch result has an invalid predicted class.")

    total = len(records)
    successful_count = len(successful)
    failed_count = total - successful_count
    if successful_count == total and total > 0:
        status = BATCH_STATUS_SUCCESS
    elif successful_count > 0:
        status = BATCH_STATUS_PARTIAL_SUCCESS
    else:
        status = BATCH_STATUS_FAILED

    return {
        "processing_status": status,
        "summary": {
            "total_tracks": total,
            "successful_tracks": successful_count,
            "failed_tracks": failed_count,
            "alpha_count": alpha_count,
            "electron_count": electron_count,
            "success_rate": successful_count / total if total else 0.0,
        },
        "results": records,
    }


def classify_module2_masks(
    masks: Sequence[Any],
    *,
    provenances: Sequence[Mapping[str, Any] | None] | None = None,
    mm_per_pixels: Sequence[float | None] | None = None,
    model_path: str | Path = MODEL_ARTIFACT,
    metadata_path: str | Path = METADATA_ARTIFACT,
    artifact_cache: ModelArtifactCache = DEFAULT_ARTIFACT_CACHE,
) -> dict[str, Any]:
    """Classify multiple existing Module 2 masks sequentially and independently.

    Single track::

        Module 2 mask -> classify_module2_mask() -> Alpha/Electron result

    Multi-track::

        Module 2 masks -> this batch layer -> classify_module2_mask() per track
        -> ordered results plus consolidated counts

    The batch layer performs no segmentation and does not replace or duplicate
    the authoritative single-track classifier. One unexpected track failure is
    converted into that track's failure record and never aborts sibling tracks.
    """
    if isinstance(masks, (str, bytes, bytearray, np.ndarray)) or not isinstance(masks, Sequence):
        raise TypeError("Module 3 batch masks must be supplied as an ordered sequence of masks.")
    ordered_masks = list(masks)

    if provenances is None:
        ordered_provenances: list[Mapping[str, Any] | None] = [None] * len(ordered_masks)
    else:
        if isinstance(provenances, (str, bytes, bytearray)) or not isinstance(provenances, Sequence):
            raise TypeError("Module 3 batch provenances must be an ordered sequence when supplied.")
        ordered_provenances = list(provenances)
        if len(ordered_provenances) != len(ordered_masks):
            raise ValueError("Module 3 batch provenance count must equal mask count.")

    if mm_per_pixels is None:
        ordered_scales: list[float | None] = [None] * len(ordered_masks)
    else:
        if isinstance(mm_per_pixels, (str, bytes, bytearray)) or not isinstance(mm_per_pixels, Sequence):
            raise TypeError("Module 3 batch calibrations must be an ordered sequence when supplied.")
        ordered_scales = list(mm_per_pixels)
        if len(ordered_scales) != len(ordered_masks):
            raise ValueError("Module 3 batch calibration count must equal mask count.")

    results: list[dict[str, Any]] = []
    for mask, provenance, scale in zip(ordered_masks, ordered_provenances, ordered_scales):
        try:
            result = classify_module2_mask(
                mask,
                provenance=provenance,
                mm_per_pixel=scale,
                model_path=model_path,
                metadata_path=metadata_path,
                artifact_cache=artifact_cache,
            )
        except Exception as exc:  # Defensive isolation around the authoritative function.
            preserved = dict(provenance) if isinstance(provenance, Mapping) else {}
            result = _failure(
                "unexpected_inference_failure",
                f"Unexpected Module 3 inference failure: {type(exc).__name__}: {exc}",
                provenance=preserved,
            )
        results.append(result)

    batch = summarize_module3_results(results)
    if not ordered_masks:
        batch["error_code"] = "empty_batch"
        batch["error_message"] = "At least one Module 2 mask is required for batch classification."
    else:
        batch["error_code"] = None
        batch["error_message"] = None
    return batch
