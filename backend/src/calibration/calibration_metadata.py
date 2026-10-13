"""JSON persistence and validation for CloudTrack calibration metadata."""

import json
import math
import re
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..utils.path_utils import CALIBRATED_DIR


REQUIRED_CALIBRATION_FIELDS = frozenset(
    {
        "point_1",
        "point_2",
        "pixel_distance",
        "real_world_distance_mm",
        "mm_per_pixel",
        "pixels_per_mm",
    }
)

PROFILE_FILE_PREFIX = "calibration_profile_"


def validate_calibration_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate persisted spatial calibration metadata."""
    if not isinstance(metadata, Mapping):
        raise TypeError("Calibration metadata must be a mapping.")

    if metadata.get("calibration_type") == "four_point_measurement":
        required = {
            "selected_corners", "chamber_width_mm", "chamber_height_mm",
            "source_image_resolution", "reference_width_px", "reference_height_px",
            "horizontal_mm_per_pixel", "vertical_mm_per_pixel", "mm_per_pixel",
            "measurement_homography",
        }
        missing = required - set(metadata)
        if missing:
            raise ValueError(f"Calibration metadata is missing required field(s): {', '.join(sorted(missing))}")
        corners = metadata["selected_corners"]
        if not isinstance(corners, Mapping):
            raise ValueError("Calibration selected_corners must be a mapping.")
        for label in ("A", "B", "C", "D"):
            point = corners.get(label)
            if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
                raise ValueError(f"Calibration corner {label} must contain x and y.")
            try:
                coordinates = (float(point["x"]), float(point["y"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Calibration corner {label} coordinates must be numeric.") from error
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(f"Calibration corner {label} coordinates must be finite.")
        numeric_fields = (
            "chamber_width_mm", "chamber_height_mm", "reference_width_px", "reference_height_px",
            "horizontal_mm_per_pixel", "vertical_mm_per_pixel", "mm_per_pixel",
        )
        try:
            values = {field: float(metadata[field]) for field in numeric_fields}
        except (TypeError, ValueError) as error:
            raise ValueError("Calibration dimensions and scales must be numeric.") from error
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("Calibration dimensions and scales must be positive finite values.")
        if not math.isclose(values["horizontal_mm_per_pixel"], values["chamber_width_mm"] / values["reference_width_px"], rel_tol=1e-9):
            raise ValueError("Horizontal calibration scale is inconsistent.")
        if not math.isclose(values["vertical_mm_per_pixel"], values["chamber_height_mm"] / values["reference_height_px"], rel_tol=1e-9):
            raise ValueError("Vertical calibration scale is inconsistent.")
        if not math.isclose(values["mm_per_pixel"], (values["horizontal_mm_per_pixel"] + values["vertical_mm_per_pixel"]) / 2, rel_tol=1e-9):
            raise ValueError("Representative calibration scale is inconsistent.")
        resolution = metadata["source_image_resolution"]
        try:
            width, height = int(resolution["width"]), int(resolution["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Calibration source resolution must contain positive width and height.") from error
        if width <= 0 or height <= 0:
            raise ValueError("Calibration source resolution must contain positive width and height.")
        matrix = metadata["measurement_homography"]
        if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in matrix):
            raise ValueError("Measurement homography must be a 3 by 3 matrix.")
        try:
            coefficients = [float(value) for row in matrix for value in row]
        except (TypeError, ValueError) as error:
            raise ValueError("Measurement homography coefficients must be numeric.") from error
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("Measurement homography coefficients must be finite.")
        return

    missing_fields = REQUIRED_CALIBRATION_FIELDS - set(metadata)
    if missing_fields:
        missing_names = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Calibration metadata is missing required field(s): {missing_names}"
        )

    for point_name in ("point_1", "point_2"):
        point = metadata[point_name]
        if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
            raise ValueError(f"Calibration metadata '{point_name}' must contain x and y.")
        try:
            coordinates = (float(point["x"]), float(point["y"]))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Calibration metadata '{point_name}' coordinates must be numeric."
            ) from error
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"Calibration metadata '{point_name}' coordinates must be finite."
            )

    numeric_fields = (
        "pixel_distance",
        "real_world_distance_mm",
        "mm_per_pixel",
        "pixels_per_mm",
    )
    try:
        values = {field: float(metadata[field]) for field in numeric_fields}
    except (TypeError, ValueError) as error:
        raise ValueError("Calibration distances and scales must be numeric.") from error
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("Calibration distances and scales must be positive finite values.")
    mode = metadata.get("calibration_mode")
    if mode != "chamber_dimensions" and not math.isclose(
        values["mm_per_pixel"],
        values["real_world_distance_mm"] / values["pixel_distance"],
        rel_tol=1e-9,
    ):
        raise ValueError("Calibration mm_per_pixel is inconsistent with the distances.")
    if mode != "chamber_dimensions" and not math.isclose(
        values["pixels_per_mm"],
        values["pixel_distance"] / values["real_world_distance_mm"],
        rel_tol=1e-9,
    ):
        raise ValueError("Calibration pixels_per_mm is inconsistent with the distances.")

    resolution = metadata.get("source_image_resolution")
    if resolution is not None:
        if not isinstance(resolution, Mapping):
            raise ValueError("Calibration source_image_resolution must be a mapping.")
        try:
            width = int(resolution["width"])
            height = int(resolution["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Calibration source_image_resolution must contain positive width and height."
            ) from error
        if width <= 0 or height <= 0:
            raise ValueError(
                "Calibration source_image_resolution must contain positive width and height."
            )

    if mode is not None and mode not in {"reference_distance", "chamber_dimensions"}:
        raise ValueError("Calibration mode is not supported.")
    if mode == "chamber_dimensions":
        for point_name in ("point_3", "point_4"):
            point = metadata.get(point_name)
            if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
                raise ValueError(f"Chamber calibration metadata '{point_name}' must contain x and y.")
        chamber_numeric_fields = (
            "chamber_width_mm", "chamber_height_mm", "width_px", "height_px",
            "horizontal_mm_per_pixel", "vertical_mm_per_pixel",
            "horizontal_pixels_per_mm", "vertical_pixels_per_mm",
        )
        try:
            chamber_values = {field: float(metadata[field]) for field in chamber_numeric_fields}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Chamber calibration dimensions and scales must be numeric.") from error
        if any(not math.isfinite(value) or value <= 0 for value in chamber_values.values()):
            raise ValueError("Chamber calibration dimensions and scales must be positive finite values.")
        horizontal = chamber_values["chamber_width_mm"] / chamber_values["width_px"]
        vertical = chamber_values["chamber_height_mm"] / chamber_values["height_px"]
        representative = (horizontal + vertical) / 2
        if not math.isclose(chamber_values["horizontal_mm_per_pixel"], horizontal, rel_tol=1e-9):
            raise ValueError("Horizontal chamber scale is inconsistent with width measurements.")
        if not math.isclose(chamber_values["vertical_mm_per_pixel"], vertical, rel_tol=1e-9):
            raise ValueError("Vertical chamber scale is inconsistent with height measurements.")
        if not math.isclose(values["mm_per_pixel"], representative, rel_tol=1e-9):
            raise ValueError("Representative chamber scale is inconsistent with axis scales.")
        if not math.isclose(values["pixels_per_mm"], 1 / representative, rel_tol=1e-9):
            raise ValueError("Representative pixels_per_mm is inconsistent with axis scales.")
        if metadata.get("rectification_enabled"):
            transform = metadata.get("transform_metadata")
            if not isinstance(transform, Mapping):
                raise ValueError("Enabled rectification requires transform metadata.")
            matrix = transform.get("homography")
            if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in matrix):
                raise ValueError("Rectification homography must be a 3 by 3 matrix.")


def _resolve_metadata_path(filename: str | Path) -> Path:
    """Resolve a JSON filename inside the calibrated-data directory."""
    output_name = Path(filename).name
    if not output_name:
        raise ValueError("Calibration metadata filename must not be empty.")

    output_path = Path(output_name)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".json")
    elif output_path.suffix.lower() != ".json":
        raise ValueError("Calibration metadata filename must use the .json extension.")

    return CALIBRATED_DIR / output_path.name


def save_calibration_metadata(
    metadata: Mapping[str, Any],
    filename: str | Path = "calibration_metadata.json",
) -> Path:
    """Validate and save calibration metadata as formatted JSON."""
    validate_calibration_metadata(metadata)
    output_path = _resolve_metadata_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(dict(metadata), json_file, indent=4)
        json_file.write("\n")

    return output_path


def load_calibration_metadata(
    filename: str | Path = "calibration_metadata.json",
) -> dict[str, Any]:
    """Load and validate calibration metadata from calibrated-data storage."""
    input_path = _resolve_metadata_path(filename)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Calibration metadata file does not exist: {input_path}"
        )

    try:
        with input_path.open("r", encoding="utf-8") as json_file:
            metadata = json.load(json_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Calibration metadata file contains invalid JSON: {input_path}"
        ) from error

    validate_calibration_metadata(metadata)
    return dict(metadata)


def calibration_profile_id(profile_name: str) -> str:
    """Return a stable, filesystem-safe identifier for a named setup profile."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(profile_name).strip().lower()).strip("-")
    if not cleaned:
        raise ValueError("Calibration profile name must not be empty.")
    return cleaned[:80]


def save_calibration_profile(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Create or update a reusable camera/chamber calibration profile."""
    profile_name = str(metadata.get("profile_name") or "").strip()
    profile_id = calibration_profile_id(profile_name)
    filename = f"{PROFILE_FILE_PREFIX}{profile_id}.json"
    now = datetime.now(timezone.utc).isoformat()
    created_at = now
    try:
        existing = load_calibration_metadata(filename)
        created_at = str(existing.get("created_at") or now)
    except FileNotFoundError:
        pass

    profile = {
        **dict(metadata),
        "profile_name": profile_name,
        "profile_id": profile_id,
        "created_at": created_at,
        "updated_at": now,
    }
    save_calibration_metadata(profile, filename)
    return profile


def load_calibration_profile(profile_id: str) -> dict[str, Any]:
    """Load one reusable profile by its stable identifier."""
    safe_id = calibration_profile_id(profile_id)
    profile = load_calibration_metadata(f"{PROFILE_FILE_PREFIX}{safe_id}.json")
    if profile.get("profile_id") != safe_id or not profile.get("profile_name"):
        raise ValueError("Saved calibration profile metadata is incomplete.")
    return profile


def list_calibration_profiles() -> list[dict[str, Any]]:
    """List valid reusable profiles, newest update first."""
    profiles: list[dict[str, Any]] = []
    if not CALIBRATED_DIR.exists():
        return profiles
    for path in CALIBRATED_DIR.glob(f"{PROFILE_FILE_PREFIX}*.json"):
        try:
            profile = load_calibration_metadata(path.name)
        except (FileNotFoundError, TypeError, ValueError):
            continue
        if profile.get("profile_id") and profile.get("profile_name"):
            profiles.append(profile)
    return sorted(profiles, key=lambda item: str(item.get("updated_at") or ""), reverse=True)
