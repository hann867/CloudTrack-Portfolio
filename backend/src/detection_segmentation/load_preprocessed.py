"""Discovery, loading, conversion, and input quality diagnostics."""

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def discover_images(input_dir, recursive=True):
    """Discover supported images with deterministic, case-insensitive sorting."""
    input_dir = Path(input_dir)
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    paths = sorted(
        (p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda p: str(p.relative_to(input_dir)).casefold(),
    )
    stems = {}
    for path in paths:
        stems.setdefault(path.stem.casefold(), []).append(path)
    duplicates = {stem: group for stem, group in stems.items() if len(group) > 1}
    if duplicates:
        collisions = "; ".join(
            f"{stem}: {[str(path) for path in group]}" for stem, group in duplicates.items()
        )
        raise ValueError(f"Duplicate source stems would overwrite outputs: {collisions}")
    return paths


def _to_uint8(image):
    """Convert a numeric image to uint8 while preserving available dynamic range."""
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.copy()
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Unsupported image dtype: {array.dtype}")
    numeric = array.astype(np.float64)
    finite = np.isfinite(numeric)
    if not finite.any():
        raise ValueError("Image contains no finite intensity values.")
    low, high = float(numeric[finite].min()), float(numeric[finite].max())
    numeric = np.nan_to_num(numeric, nan=low, posinf=high, neginf=low)
    if low >= 0.0 and high <= 1.0:
        scaled = numeric * 255.0
    elif low >= 0.0 and high <= 255.0:
        scaled = numeric
    elif high > low:
        scaled = (numeric - low) * (255.0 / (high - low))
    else:
        scaled = np.full_like(numeric, 0 if high <= 0 else 255, dtype=np.float64)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def load_grayscale_image(path):
    """Load an image as 8-bit grayscale, a BGR copy, and source metadata."""
    path = Path(path)
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Could not read bytes from {path.name}: {exc}") from exc
    if encoded.size == 0:
        raise ValueError(f"File is empty: {path.name}")
    raw = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"OpenCV could not decode: {path.name}")
    original_shape = tuple(int(v) for v in raw.shape)
    original_dtype = str(raw.dtype)
    original_min, original_max = float(np.nanmin(raw)), float(np.nanmax(raw))
    if raw.ndim == 2:
        gray = _to_uint8(raw)
        display_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        source_kind = "grayscale"
    elif raw.ndim == 3 and raw.shape[2] == 1:
        gray = _to_uint8(raw[:, :, 0])
        display_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        source_kind = "single-channel"
    elif raw.ndim == 3 and raw.shape[2] == 3:
        gray = _to_uint8(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY))
        display_bgr = _to_uint8(raw)
        source_kind = "BGR colour"
    elif raw.ndim == 3 and raw.shape[2] == 4:
        gray = _to_uint8(cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY))
        display_bgr = _to_uint8(cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR))
        source_kind = "BGRA colour"
    else:
        raise ValueError(f"Unsupported image shape {raw.shape} for {path.name}")
    metadata = {
        "source_filename": path.name, "source_kind": source_kind,
        "original_shape": original_shape, "original_dtype": original_dtype,
        "original_min": original_min, "original_max": original_max,
        "working_dtype": str(gray.dtype), "working_min": int(gray.min()),
        "working_max": int(gray.max()),
    }
    return gray, display_bgr, metadata


def is_nearly_binary(gray, tolerance=5, required_fraction=0.995, min_endpoint_fraction=0.0001):
    """Return True when an 8-bit image is exact or convincingly near binary."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("is_nearly_binary expects a 2-D uint8 image.")
    unique_values = np.unique(gray)
    if unique_values.size <= 2:
        return True
    low, high = gray <= tolerance, gray >= (255 - tolerance)
    return bool(
        np.mean(low | high) >= required_fraction
        and np.mean(low) >= min_endpoint_fraction
        and np.mean(high) >= min_endpoint_fraction
    )


def infer_foreground_polarity(gray):
    """Infer whether sparse objects are likely brighter or darker than the background."""
    if int(gray.max()) == int(gray.min()):
        return "undetermined", int(gray.flat[0])
    dominant = int(np.bincount(gray.ravel(), minlength=256).argmax())
    return ("bright" if dominant < 128 else "dark"), dominant


def input_intensity_record(path, config):
    """Build one intensity/polarity quality-control record."""
    gray, _, metadata = load_grayscale_image(path)
    unique_values = np.unique(gray)
    polarity, dominant = infer_foreground_polarity(gray)
    tolerance = config["near_binary_tolerance"]
    return {
        **metadata, "unique_values": int(unique_values.size),
        "exact_binary": bool(unique_values.size <= 2),
        "nearly_binary": is_nearly_binary(
            gray, tolerance, config["near_binary_required_fraction"],
            config["near_binary_min_endpoint_fraction"]),
        "median_intensity": float(np.median(gray)), "dominant_intensity": dominant,
        "low_endpoint_percent": 100.0 * float(np.mean(gray <= tolerance)),
        "high_endpoint_percent": 100.0 * float(np.mean(gray >= 255 - tolerance)),
        "inferred_foreground_polarity": polarity,
        "configured_foreground_polarity": config["foreground_polarity"],
    }

