from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import backend.app  # Applies the Python 3.14rc2/FastAPI compatibility gate before FastAPI import.
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.main import app
from backend.src.geometric_classification.model_contract import AUTHORIZED_CLASS_NAMES, AUTHORIZED_FEATURE_NAMES


client = TestClient(app)


def png_bytes(mask: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", mask)
    assert success
    return encoded.tobytes()


def line_mask() -> np.ndarray:
    mask = np.zeros((80, 140), dtype=np.uint8)
    cv2.line(mask, (15, 40), (125, 40), 255, 5)
    return mask


def post_mask(mask: np.ndarray, *, data: dict[str, str] | None = None):
    return client.post(
        "/api/classification/classify",
        files={"file": ("track_mask.png", png_bytes(mask), "image/png")},
        data=data or {},
    )


def test_valid_mask_returns_complete_development_classification() -> None:
    response = post_mask(
        line_mask(),
        data={"track_id": "frame_track_0007", "component_id": "17", "label_id": "2"},
    )
    assert response.status_code == 200
    assert response.status_code != 501
    payload = response.json()
    assert payload["processing_status"] == "success"
    assert payload["predicted_class"] in AUTHORIZED_CLASS_NAMES
    assert tuple(payload["probabilities"]) == AUTHORIZED_CLASS_NAMES
    assert set(payload["features"]) == set(AUTHORIZED_FEATURE_NAMES)
    assert len(payload["features"]) == 7
    assert payload["model"]["model_stage"] == "development"
    assert payload["model"]["model_version"] == "module3_rf_development_v2"
    assert payload["confidence"] == pytest.approx(payload["probabilities"][payload["predicted_class"]])
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0)
    assert payload["provenance"] == {
        "source_filename": "track_mask.png",
        "track_id": "frame_track_0007",
        "component_id": 17,
        "label_id": 2,
    }


def test_malformed_upload_returns_machine_readable_400() -> None:
    response = client.post(
        "/api/classification/classify",
        files={"file": ("track_mask.png", b"not a png", "image/png")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "processing_status": "failed",
        "error_code": "unreadable_mask",
        "error_message": "The uploaded PNG could not be decoded as an image.",
    }


def test_empty_upload_returns_machine_readable_400() -> None:
    response = client.post(
        "/api/classification/classify",
        files={"file": ("track_mask.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "empty_mask"


def test_no_foreground_returns_422() -> None:
    response = post_mask(np.zeros((20, 30), dtype=np.uint8))
    assert response.status_code == 422
    assert response.json()["error_code"] == "no_foreground"


def test_colour_mask_is_rejected_instead_of_converted() -> None:
    response = post_mask(np.zeros((20, 30, 3), dtype=np.uint8))
    assert response.status_code == 400
    assert response.json()["error_code"] == "malformed_mask"


def test_model_unavailable_returns_sanitized_503(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        return {
            "processing_status": "failed",
            "error_code": "missing_model_artifact",
            "error_message": "Model artifact not found: C:\\Users\\private\\secret.joblib",
        }

    monkeypatch.setattr("backend.app.routes.classification.classify_module2_mask", unavailable)
    response = post_mask(line_mask())
    assert response.status_code == 503
    body = response.text
    assert response.json()["error_code"] == "missing_model_artifact"
    assert "C:\\Users" not in body
    assert "secret.joblib" not in body


def test_unexpected_inference_exception_returns_sanitized_500(monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("private C:\\workspace\\model path")

    monkeypatch.setattr("backend.app.routes.classification.classify_module2_mask", explode)
    response = post_mask(line_mask())
    assert response.status_code == 500
    assert response.json()["error_code"] == "unexpected_inference_failure"
    assert "workspace" not in response.text


def test_endpoint_never_invokes_training(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("RF training must not be invoked by inference API")

    monkeypatch.setattr(
        "backend.src.geometric_classification.final_assignment_rf_trainer.train_final_assignment_model", forbidden
    )
    assert post_mask(line_mask()).status_code == 200


def test_endpoint_does_not_modify_module2_core_files() -> None:
    module2_root = Path(__file__).resolve().parents[2] / "src" / "detection_segmentation"

    def hashes() -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in module2_root.glob("*.py")
        }

    before = hashes()
    assert post_mask(line_mask()).status_code == 200
    assert hashes() == before
