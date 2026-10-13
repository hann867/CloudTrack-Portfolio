from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.geometric_classification.model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
)
from src.geometric_classification.final_assignment_rf_trainer import MODEL_STAGE, MODEL_VERSION
from src.geometric_classification.production_inference import (
    BATCH_STATUS_FAILED,
    BATCH_STATUS_PARTIAL_SUCCESS,
    BATCH_STATUS_SUCCESS,
    ModelArtifactCache,
    classify_module2_mask,
    classify_module2_masks,
)


def line_mask(*, thickness: int = 5) -> np.ndarray:
    mask = np.zeros((80, 140), dtype=np.uint8)
    cv2.line(mask, (15, 40), (125, 40), 255, thickness)
    return mask


def test_all_success_batch_has_exact_counts_contract_and_provenance() -> None:
    provenances = [
        {"track_id": "track_1", "component_id": 11},
        {"track_id": "track_2", "component_id": 12},
        {"track_id": "track_3", "component_id": 13},
    ]
    result = classify_module2_masks(
        [line_mask(thickness=value) for value in (3, 5, 7)],
        provenances=provenances,
        artifact_cache=ModelArtifactCache(),
    )

    assert result["processing_status"] == BATCH_STATUS_SUCCESS
    assert result["summary"]["total_tracks"] == 3
    assert result["summary"]["successful_tracks"] == 3
    assert result["summary"]["failed_tracks"] == 0
    assert result["summary"]["alpha_count"] + result["summary"]["electron_count"] == 3
    assert result["summary"]["success_rate"] == 1.0
    assert [record["provenance"] for record in result["results"]] == provenances

    for record in result["results"]:
        assert record["predicted_class"] in AUTHORIZED_CLASS_NAMES
        assert list(record["features"]) == AUTHORIZED_FEATURE_NAMES
        assert record["model"] == {
            "model_version": MODEL_VERSION,
            "model_stage": MODEL_STAGE,
        }
        assert sum(record["probabilities"].values()) == pytest.approx(1.0)
        assert record["confidence"] == pytest.approx(
            record["probabilities"][record["predicted_class"]]
        )


def test_mixed_batch_isolates_middle_failure_and_preserves_order() -> None:
    result = classify_module2_masks(
        [line_mask(thickness=3), np.zeros((20, 20), dtype=np.uint8), line_mask(thickness=7)],
        provenances=[{"track_id": value} for value in ("first", "middle", "third")],
        artifact_cache=ModelArtifactCache(),
    )

    assert result["processing_status"] == BATCH_STATUS_PARTIAL_SUCCESS
    assert result["summary"]["successful_tracks"] == 2
    assert result["summary"]["failed_tracks"] == 1
    assert [record["provenance"]["track_id"] for record in result["results"]] == [
        "first",
        "middle",
        "third",
    ]
    assert [record["processing_status"] for record in result["results"]] == [
        "success",
        "failed",
        "success",
    ]
    assert result["results"][1]["error_code"] == "no_foreground"


def test_all_failure_and_empty_batches_are_deterministic() -> None:
    failed = classify_module2_masks(
        [None, np.empty((0, 0), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8)]
    )
    assert failed["processing_status"] == BATCH_STATUS_FAILED
    assert failed["summary"] == {
        "total_tracks": 3,
        "successful_tracks": 0,
        "failed_tracks": 3,
        "alpha_count": 0,
        "electron_count": 0,
        "success_rate": 0.0,
    }
    assert len(failed["results"]) == 3

    empty = classify_module2_masks([])
    assert empty["processing_status"] == BATCH_STATUS_FAILED
    assert empty["error_code"] == "empty_batch"
    assert empty["summary"]["total_tracks"] == 0
    assert empty["results"] == []


def test_batch_reuses_one_cached_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.geometric_classification.production_inference as inference

    actual_loader = inference.load_classifier
    calls = 0

    def counting_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual_loader(*args, **kwargs)

    monkeypatch.setattr(inference, "load_classifier", counting_loader)
    result = classify_module2_masks(
        [line_mask(thickness=value) for value in (3, 5, 7)],
        artifact_cache=ModelArtifactCache(),
    )
    assert result["processing_status"] == BATCH_STATUS_SUCCESS
    assert calls == 1


def test_one_item_batch_matches_authoritative_single_track_result() -> None:
    mask = line_mask()
    provenance = {"track_id": "single_regression", "bbox_width": 111}
    cache = ModelArtifactCache()
    single = classify_module2_mask(mask, provenance=provenance, artifact_cache=cache)
    batch = classify_module2_masks([mask], provenances=[provenance], artifact_cache=cache)
    assert batch["results"] == [single]
