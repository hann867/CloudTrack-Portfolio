"""Focused end-to-end regression for the real Module 3 runtime chain."""

from __future__ import annotations

import hashlib

import backend.app  # Apply the Python 3.14rc2 compatibility gate before FastAPI import.
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.main import app
from backend.src.geometric_classification.final_assignment_rf_trainer import (
    METADATA_ARTIFACT,
    MODEL_ARTIFACT,
    MODEL_STAGE,
    MODEL_VERSION,
)
from backend.src.geometric_classification.model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
)
from backend.src.geometric_classification.production_inference import DEFAULT_ARTIFACT_CACHE


def _module2_style_mask_png() -> bytes:
    mask = np.zeros((80, 140), dtype=np.uint8)
    cv2.line(mask, (15, 40), (125, 40), 255, 5)
    encoded, png = cv2.imencode(".png", mask)
    assert encoded
    return png.tobytes()


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_module3_mask_to_api_to_development_rf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real API, geometry, contract, artifact, and probability path."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Training and temporary Module 3 segmentation must not run during inference.")

    monkeypatch.setattr(
        "backend.src.geometric_classification.final_assignment_rf_trainer.train_final_assignment_model",
        forbidden,
    )
    monkeypatch.setattr(
        "backend.src.geometric_classification.feature_extractor.segment_track",
        forbidden,
    )

    artifact_hash_before = _sha256(MODEL_ARTIFACT)
    metadata_hash_before = _sha256(METADATA_ARTIFACT)
    DEFAULT_ARTIFACT_CACHE.clear()

    response = TestClient(app).post(
        "/api/classification/classify",
        files={"file": ("representative_module2_mask.png", _module2_style_mask_png(), "image/png")},
        data={"track_id": "integration_track_0001", "component_id": "1", "label_id": "1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["processing_status"] == "success"
    assert payload["predicted_class"] in AUTHORIZED_CLASS_NAMES
    assert 0.0 <= payload["confidence"] <= 1.0

    assert tuple(payload["probabilities"]) == AUTHORIZED_CLASS_NAMES
    probability_alpha = payload["probabilities"]["Alpha"]
    probability_electron = payload["probabilities"]["Electron"]
    assert 0.0 <= probability_alpha <= 1.0
    assert 0.0 <= probability_electron <= 1.0
    assert probability_alpha + probability_electron == pytest.approx(1.0)
    assert payload["confidence"] == pytest.approx(
        payload["probabilities"][payload["predicted_class"]]
    )

    assert list(payload["features"]) == AUTHORIZED_FEATURE_NAMES
    assert len(payload["features"]) == 7
    assert payload["model"] == {
        "model_version": MODEL_VERSION,
        "model_stage": MODEL_STAGE,
    }

    assert _sha256(MODEL_ARTIFACT) == artifact_hash_before
    assert _sha256(METADATA_ARTIFACT) == metadata_hash_before
