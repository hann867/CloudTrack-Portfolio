from __future__ import annotations

import cv2
import numpy as np
import pytest

import backend.app  # Applies the FastAPI compatibility gate before FastAPI import.
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def calibration_payload(**overrides):
    payload = {
        "point_1_x": 20, "point_1_y": 10,
        "point_2_x": 220, "point_2_y": 20,
        "point_3_x": 210, "point_3_y": 120,
        "point_4_x": 30, "point_4_y": 110,
        "chamber_width_mm": 100,
        "chamber_height_mm": 50,
        "image_filename": "reference.png",
        "profile_name": "Cloud Chamber Setup A",
        "source_image_width": 240,
        "source_image_height": 140,
    }
    payload.update(overrides)
    return payload


def test_four_point_calibration_maps_corners_to_physical_chamber(monkeypatch) -> None:
    saved = {}
    monkeypatch.setattr(
        "backend.app.routes.calibration.save_calibration_profile",
        lambda metadata: {**metadata, "profile_id": "cloud-chamber-setup-a"},
    )
    monkeypatch.setattr(
        "backend.app.routes.calibration.save_calibration_metadata",
        lambda metadata, filename: saved.update({"metadata": dict(metadata), "filename": filename}),
    )

    response = client.post("/api/calibration/calculate", json=calibration_payload())
    assert response.status_code == 200
    payload = response.json()
    assert payload["calibration_type"] == "four_point_measurement"
    assert payload["selected_corners"]["A"] == {"x": 20.0, "y": 10.0}
    assert payload["source_image_resolution"] == {"width": 240, "height": 140}
    assert saved["filename"] == "reference.png_calibration.json"
    assert "rectification_enabled" not in payload
    assert "transform_metadata" not in payload

    source = np.asarray([[20, 10], [220, 20], [210, 120], [30, 110]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(
        source.reshape((-1, 1, 2)), np.asarray(payload["measurement_homography"], dtype=np.float64)
    ).reshape((-1, 2))
    assert mapped == pytest.approx(np.asarray([[0, 0], [100, 0], [100, 50], [0, 50]]), abs=1e-5)


def test_four_point_calibration_rejects_duplicate_or_crossed_corners(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.routes.calibration.save_calibration_metadata", lambda *_args: None)
    duplicate = client.post("/api/calibration/calculate", json=calibration_payload(point_4_x=210, point_4_y=120))
    crossed = client.post("/api/calibration/calculate", json=calibration_payload(
        point_2_x=30, point_2_y=110, point_4_x=220, point_4_y=20,
    ))
    assert duplicate.status_code == 400
    assert crossed.status_code == 400


def test_missing_fourth_point_and_invalid_dimensions_are_rejected() -> None:
    missing = calibration_payload()
    del missing["point_4_x"]
    assert client.post("/api/calibration/calculate", json=missing).status_code == 422
    assert client.post("/api/calibration/calculate", json=calibration_payload(chamber_width_mm=0)).status_code == 422


def test_saved_profiles_are_returned_for_reuse(monkeypatch) -> None:
    profile = {
        "calibration_type": "four_point_measurement",
        "profile_id": "setup-a",
        "profile_name": "Setup A",
        "chamber_width_mm": 100,
        "chamber_height_mm": 50,
        "measurement_homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    monkeypatch.setattr("backend.app.routes.calibration.list_calibration_profiles", lambda: [profile])
    response = client.get("/api/calibration/profiles")
    assert response.status_code == 200
    assert response.json()["profiles"][0]["profile_name"] == "Setup A"


def test_legacy_four_corner_profile_is_exposed_as_measurement_only(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.routes.calibration.list_calibration_profiles", lambda: [{
        "calibration_mode": "chamber_dimensions",
        "profile_id": "setup-a", "profile_name": "Setup A",
        "point_1": {"x": 0, "y": 0}, "point_2": {"x": 100, "y": 0},
        "point_3": {"x": 100, "y": 200}, "point_4": {"x": 0, "y": 200},
        "chamber_width_mm": 50, "chamber_height_mm": 80,
        "source_image_resolution": {"width": 100, "height": 200},
        "rectification_enabled": True,
    }])
    payload = client.get("/api/calibration/profiles").json()["profiles"][0]
    assert payload["calibration_type"] == "four_point_measurement"
    assert payload["horizontal_mm_per_pixel"] == pytest.approx(0.5)
    assert payload["vertical_mm_per_pixel"] == pytest.approx(0.4)
    assert "rectification_enabled" not in payload
