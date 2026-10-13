from __future__ import annotations

import json

import backend.app  # Apply the Python 3.14rc2 compatibility gate before FastAPI import.
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.main import app
from backend.app.routes.classification import (
    MAX_CLASSIFICATION_BATCH_TRACKS,
    MAX_MASK_UPLOAD_BYTES,
)
from backend.src.geometric_classification.model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
)
from backend.src.geometric_classification.final_assignment_rf_trainer import (
    MODEL_STAGE,
    MODEL_VERSION,
)


client = TestClient(app)


def png_bytes(mask: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", mask)
    assert success
    return encoded.tobytes()


def line_mask(thickness: int = 5) -> np.ndarray:
    mask = np.zeros((80, 140), dtype=np.uint8)
    cv2.line(mask, (15, 40), (125, 40), 255, thickness)
    return mask


def post_batch(payloads: list[tuple[str, bytes]], provenance: list[dict] | None = None):
    files = [("files", (name, content, "image/png")) for name, content in payloads]
    data = {} if provenance is None else {"provenance_json": json.dumps(provenance)}
    return client.post("/api/classification/classify-batch", files=files, data=data)


def test_batch_api_all_success_preserves_probabilities_features_and_order() -> None:
    provenance = [
        {"track_id": "api_1", "component_id": 21},
        {"track_id": "api_2", "component_id": 22},
        {"track_id": "api_3", "component_id": 23},
    ]
    response = post_batch(
        [(f"api_{index}.png", png_bytes(line_mask(thickness))) for index, thickness in enumerate((3, 5, 7), 1)],
        provenance,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["processing_status"] == "SUCCESS"
    assert payload["summary"]["total_tracks"] == 3
    assert payload["summary"]["successful_tracks"] == 3
    assert payload["summary"]["failed_tracks"] == 0
    assert payload["summary"]["alpha_count"] + payload["summary"]["electron_count"] == 3
    assert [result["provenance"]["track_id"] for result in payload["results"]] == [
        "api_1",
        "api_2",
        "api_3",
    ]
    for result in payload["results"]:
        assert result["predicted_class"] in AUTHORIZED_CLASS_NAMES
        assert list(result["features"]) == AUTHORIZED_FEATURE_NAMES
        assert result["model"] == {
            "model_version": MODEL_VERSION,
            "model_stage": MODEL_STAGE,
        }
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)
        assert result["confidence"] == pytest.approx(
            result["probabilities"][result["predicted_class"]]
        )


def test_batch_api_mixed_upload_failure_does_not_abort_siblings() -> None:
    response = post_batch(
        [
            ("first.png", png_bytes(line_mask(3))),
            ("middle.png", b"not a PNG"),
            ("third.png", png_bytes(line_mask(7))),
        ],
        [{"track_id": value} for value in ("first", "middle", "third")],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["processing_status"] == "PARTIAL_SUCCESS"
    assert payload["summary"]["successful_tracks"] == 2
    assert payload["summary"]["failed_tracks"] == 1
    assert [record["processing_status"] for record in payload["results"]] == [
        "success",
        "failed",
        "success",
    ]
    assert payload["results"][1]["error_code"] == "unreadable_mask"
    assert payload["results"][2]["provenance"]["track_id"] == "third"


def test_batch_api_all_failure_and_empty_batch_contracts() -> None:
    all_failed = post_batch(
        [("blank_1.png", png_bytes(np.zeros((20, 20), np.uint8))), ("blank_2.png", b"")],
        [{"track_id": "blank_1"}, {"track_id": "blank_2"}],
    )
    assert all_failed.status_code == 200
    assert all_failed.json()["processing_status"] == "FAILED"
    assert all_failed.json()["summary"]["failed_tracks"] == 2
    assert len(all_failed.json()["results"]) == 2

    empty = client.post("/api/classification/classify-batch")
    assert empty.status_code == 400
    assert empty.json()["processing_status"] == "FAILED"
    assert empty.json()["error_code"] == "empty_batch"
    assert empty.json()["results"] == []


def test_batch_api_rejects_duplicate_identity_and_excessive_size() -> None:
    duplicate = post_batch(
        [("one.png", png_bytes(line_mask())), ("two.png", png_bytes(line_mask()))],
        [{"track_id": "same"}, {"track_id": "same"}],
    )
    assert duplicate.status_code == 400
    assert duplicate.json()["error_code"] == "duplicate_batch_identity"

    excessive_files = [
        ("files", (f"track_{index}.png", png_bytes(line_mask()), "image/png"))
        for index in range(MAX_CLASSIFICATION_BATCH_TRACKS + 1)
    ]
    excessive = client.post("/api/classification/classify-batch", files=excessive_files)
    assert excessive.status_code == 413
    assert excessive.json()["error_code"] == "batch_too_large"


def test_batch_api_isolates_unsupported_and_oversized_uploads() -> None:
    response = post_batch(
        [
            ("wrong.jpg", png_bytes(line_mask())),
            ("oversized.png", b"x" * (MAX_MASK_UPLOAD_BYTES + 1)),
        ],
        [{"track_id": "wrong_type"}, {"track_id": "too_large"}],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["processing_status"] == "FAILED"
    assert payload["summary"]["failed_tracks"] == 2
    assert [result["error_code"] for result in payload["results"]] == [
        "unsupported_mask_format",
        "mask_too_large",
    ]
