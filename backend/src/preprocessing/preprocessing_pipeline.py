"""Pre-processing pipelines for CloudTrack Module 1.

Two internal branches run automatically on every image:

* **Electron branch** — tight ROI, white top-hat, high-percentile
  threshold, small-component filter.
* **Alpha/proton branch** — wide ROI, broad Gaussian background
  subtraction, moderate CLAHE, lower-percentile threshold, elongation-
  based component filter.

The two masks are combined with bitwise OR.  No user interaction is
required to select the particle type.
"""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contrast_enhancement import apply_clahe
from .illumination_correction import correct_uneven_illumination
from .image_loader import load_image
from .noise_reduction import apply_median_filter



# Default configs


DEFAULT_PREPROCESSING_CONFIG: dict[str, Any] = {
    "median_kernel_size": 5,
    "illumination_method": "gaussian",
    "illumination_kernel_size": 51,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
}

CLEAN_PIPELINE_CONFIG: dict[str, Any] = {
    "median_kernel_size": 3,
    # ---- electron branch ------------------------------------------------
    "electron": {
        "roi": {
            "left": 0.08,
            "right": 0.92,
            "top": 0.16,
            "bottom": 0.88,
        },
        "top_hat_kernel_size": 31,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
        "threshold_percentile": 99.2,
        "component_filter": {
            "min_area": 2,
            "max_area": 120,
            "max_width": 60,
            "max_height": 120,
        },
    },
    # ---- alpha / proton branch ------------------------------------------
    "alpha_proton": {
        "roi": {
            "left": 0.04,
            "right": 0.98,
            "top": 0.09,
            "bottom": 0.90,
        },
        "gaussian_blur_sigma": 45,
        "clahe_clip_limit": 1.5,
        "clahe_tile_grid_size": (8, 8),
        "threshold_percentile": 98.2,
        "component_filter": {
            "min_area": 80,
            "max_area": 25000,
            # Axis-aligned bounding-box aspect ratio is not a reliable
            # particle-track discriminator for curved, thick, or short tracks.
            # Keep this optional for explicit experiments, but do not reject
            # valid Alpha/proton candidates by shape in the default pipeline.
            "min_aspect_ratio": None,
            "max_roi_area_ratio": 0.18,
        },
    },
}



# Config resolution



def _resolve_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge optional settings with the original processing defaults."""
    if config is None:
        return DEFAULT_PREPROCESSING_CONFIG.copy()
    if not isinstance(config, Mapping):
        raise TypeError("Pipeline config must be a mapping or None.")

    unknown_keys = set(config) - set(DEFAULT_PREPROCESSING_CONFIG)
    if unknown_keys:
        unknown_names = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ValueError(f"Unknown pipeline config option(s): {unknown_names}")

    resolved_config = DEFAULT_PREPROCESSING_CONFIG.copy()
    resolved_config.update(config)

    tile_grid_size = resolved_config["clahe_tile_grid_size"]
    if isinstance(tile_grid_size, list):
        resolved_config["clahe_tile_grid_size"] = tuple(tile_grid_size)

    return resolved_config


def _deep_merge(base: dict, overrides: Mapping) -> dict:
    """Recursively merge *overrides* into a shallow copy of *base*."""
    merged = base.copy()
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_clean_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Deep-merge optional overrides with the dual-branch defaults."""
    resolved = deepcopy(CLEAN_PIPELINE_CONFIG)
    if config is None:
        return resolved
    if not isinstance(config, Mapping):
        raise TypeError("Clean pipeline config must be a mapping or None.")

    resolved = _deep_merge(resolved, dict(config))

    # Normalise list → tuple for tile-grid sizes in each branch.
    for branch_key in ("electron", "alpha_proton"):
        branch = resolved.get(branch_key)
        if isinstance(branch, dict):
            tgs = branch.get("clahe_tile_grid_size")
            if isinstance(tgs, list):
                branch["clahe_tile_grid_size"] = tuple(tgs)

    return resolved



# Validation helpers



def _validate_grayscale_image(image: np.ndarray) -> None:
    """Validate a non-empty two-dimensional image."""
    if image is None:
        raise ValueError("Input grayscale image must not be empty or None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Input grayscale image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Input grayscale image must not be empty.")
    if image.ndim != 2:
        raise ValueError("Input image must be a two-dimensional grayscale image.")



# ROI mask construction



def _build_roi_mask(
    image_shape: tuple[int, ...],
    roi_fractions: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, int]]:
    """Create a binary ROI mask from fractional bounds.

    Parameters
    ----------
    image_shape : (height, width, ...)
    roi_fractions : mapping with ``left``, ``right``, ``top``, ``bottom``
        as fractions of image width / height.

    Returns
    -------
    roi_mask : uint8 array, 0 or 255
    roi_details : dict with pixel coordinates x1, x2, y1, y2, width, height
    """
    height, width = image_shape[:2]
    required = {"left", "right", "top", "bottom"}
    missing = required - set(roi_fractions)
    if missing:
        raise ValueError(
            f"ROI config is missing field(s): {', '.join(sorted(missing))}"
        )

    left = float(roi_fractions["left"])
    right = float(roi_fractions["right"])
    top = float(roi_fractions["top"])
    bottom = float(roi_fractions["bottom"])
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(
            "ROI fractions must satisfy 0 <= left < right <= 1 and "
            "0 <= top < bottom <= 1."
        )

    x1 = int(width * left)
    x2 = int(width * right)
    y1 = int(height * top)
    y2 = int(height * bottom)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("ROI fractions produce an empty image region.")

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    details = {
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
    }
    return mask, details


def create_chamber_roi_mask(
    grayscale_image: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    """Public wrapper — accepts a config mapping with an ``"roi"`` key."""
    _validate_grayscale_image(grayscale_image)
    roi_cfg = config.get("roi")
    if roi_cfg is None or not isinstance(roi_cfg, Mapping):
        raise ValueError("Config must contain an 'roi' mapping.")
    return _build_roi_mask(grayscale_image.shape, roi_cfg)



# Chamber-frame artefact removal (electron path only)



def _remove_chamber_frame_artefacts(
    binary: np.ndarray,
    roi_details: Mapping[str, int],
    *,
    boundary_fraction: float = 0.06,
    min_span_fraction: float = 0.20,
    min_aspect_ratio: float = 5.0,
    image_shape: tuple[int, ...] | None = None,
    max_frame_thickness: int = 25,
) -> np.ndarray:
    """Remove bright lines caused by the physical cloud-chamber frame.

    The chamber edges appear as long, thin, horizontal or vertical
    connected components located very close to the ROI boundary.  Real
    electron traces are much smaller and randomly positioned, so the
    two can be separated by combining *location* and *geometry*.

    A component is classified as a **frame artefact** and erased when
    **all** of the following conditions are met:

    Top / bottom horizontal frame line
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Its vertical centre is within ``boundary_fraction`` of the ROI
      height from the top or bottom ROI edge.
    * Its bounding-box width  ≥ ``min_span_fraction`` × ROI width.
    * width / height  ≥ ``min_aspect_ratio``  (strongly horizontal).

    Left / right vertical frame line
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Its horizontal centre is within ``boundary_fraction`` of the ROI
      width from the left or right ROI edge.
    * Its bounding-box height ≥ ``min_span_fraction`` × ROI height.
    * height / width  ≥ ``min_aspect_ratio``  (strongly vertical).

    Parameters
    ----------
    binary : uint8 mask (0 / 255)
    roi_details : pixel-coordinate dict from ``_build_roi_mask``
    boundary_fraction : how close (fraction of ROI size) a component
        must be to the ROI edge to be considered a frame candidate.
    min_span_fraction : minimum width (or height) of a candidate
        expressed as a fraction of the ROI width (or height).
    min_aspect_ratio : minimum aspect ratio (long / short side) to
        distinguish frame lines from compact particle traces.

    Returns
    -------
    Cleaned binary mask with frame artefacts set to 0.
    """
    x1 = int(roi_details["x1"])
    x2 = int(roi_details["x2"])
    y1 = int(roi_details["y1"])
    y2 = int(roi_details["y2"])
    roi_w = x2 - x1
    roi_h = y2 - y1

    # Pixel margins for "near the boundary".
    y_margin = int(roi_h * boundary_fraction)
    x_margin = int(roi_w * boundary_fraction)

    # Minimum absolute span to qualify as a frame line.
    min_horiz_span = int(roi_w * min_span_fraction)
    min_vert_span = int(roi_h * min_span_fraction)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = binary.copy()

    # Image dimensions for border-touch detection.
    if image_shape is not None:
        img_h, img_w = image_shape[:2]
    else:
        img_h, img_w = binary.shape[:2]

    for i in range(1, n_labels):
        cx, cy, cw, ch, _area = stats[i]
        # Component centres.
        centre_x = cx + cw / 2.0
        centre_y = cy + ch / 2.0

        # Components centred outside the ROI: remove small debris and
        # anything touching the actual image border.  Large components
        # that don't touch the image border fall through to the
        # frame-line geometry checks below — this preserves legitimate
        # alpha/proton tracks while still catching the main frame line.
        if centre_x < x1 or centre_x > x2 or centre_y < y1 or centre_y > y2:
            touches_image_border = (
                cx <= 1 or cy <= 1
                or cx + cw >= img_w - 1
                or cy + ch >= img_h - 1
            )
            if touches_image_border or _area < 500:
                cleaned[labels == i] = 0
                continue

        safe_cw = max(cw, 1)
        safe_ch = max(ch, 1)

        # Horizontal frame line (top / bottom or outside ROI vertically).
        # A component spanning > 80% of the ROI width is a full-width
        # frame line regardless of thickness.  Shorter components near
        # the boundary must also be thin (≤ max_frame_thickness) to be
        # classified as frame debris — this preserves thick tracks.
        near_top = centre_y < y1 + y_margin
        near_bottom = centre_y > y2 - y_margin
        if (near_top or near_bottom) and cw >= min_horiz_span:
            if cw / safe_ch >= min_aspect_ratio:
                if cw >= roi_w * 0.80 or ch <= max_frame_thickness:
                    cleaned[labels == i] = 0
                    continue

        # Vertical frame line (left / right or outside ROI horizontally).
        near_left = centre_x < x1 + x_margin
        near_right = centre_x > x2 - x_margin
        if (near_left or near_right) and ch >= min_vert_span:
            if ch / safe_cw >= min_aspect_ratio:
                if ch >= roi_h * 0.80 or cw <= max_frame_thickness:
                    cleaned[labels == i] = 0
                    continue

    return cleaned



# Component filters



def _filter_electron_components(
    binary: np.ndarray,
    roi_mask: np.ndarray,
    roi_details: Mapping[str, int],
    filter_config: Mapping[str, Any],
) -> np.ndarray:
    """Keep small, thin components inside the ROI (electron-like traces).

    Rejection criteria:
    * touches the *ROI* border (± 2 px margin)
    * area outside [min_area, max_area]
    * bounding-box width  > max_width
    * bounding-box height > max_height
    """
    min_area = int(filter_config["min_area"])
    max_area = int(filter_config["max_area"])
    max_width = int(filter_config["max_width"])
    max_height = int(filter_config["max_height"])

    x1 = int(roi_details["x1"])
    x2 = int(roi_details["x2"])
    y1 = int(roi_details["y1"])
    y2 = int(roi_details["y2"])
    margin = 2

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary, dtype=np.uint8)

    for i in range(1, n_labels):
        cx, cy, cw, ch, area = stats[i]
        # Reject if touching ROI border.
        if (
            cx <= x1 + margin
            or cy <= y1 + margin
            or cx + cw >= x2 - margin
            or cy + ch >= y2 - margin
        ):
            continue
        if min_area <= area <= max_area and cw <= max_width and ch <= max_height:
            out[labels == i] = 255

    return cv2.bitwise_and(out, out, mask=roi_mask)


def _filter_alpha_proton_components(
    binary: np.ndarray,
    roi_mask: np.ndarray,
    roi_details: Mapping[str, int],
    image_shape: tuple[int, ...],
    filter_config: Mapping[str, Any],
) -> np.ndarray:
    """Keep bright, elongated components (alpha / proton-like tracks).

    Rejection criteria:
    * touches the **original image** border (frame / edge artefact)
    * area outside [min_area, max_area]
    * area > max_roi_area_ratio × ROI pixel area
    * optional aspect-ratio filter when ``min_aspect_ratio`` is configured
    * horizontal reflection line near the bottom of the image
      (width > 30% image width, height < 5% image height,
       aspect ratio > 6, located in the bottom 25%)

    No max_width / max_height limit is applied — alpha / proton tracks
    can be legitimately tall or wide.
    """
    img_h, img_w = image_shape[:2]
    roi_pixel_area = int(roi_details["width"]) * int(roi_details["height"])

    min_area = int(filter_config["min_area"])
    max_area = int(filter_config["max_area"])
    configured_min_ar = filter_config.get("min_aspect_ratio")
    min_ar = (
        None
        if configured_min_ar is None
        else float(configured_min_ar)
    )
    max_ratio = float(filter_config["max_roi_area_ratio"])

    # Thresholds for bottom horizontal-line rejection.
    bottom_y_threshold = img_h * 0.75
    min_width_for_line = img_w * 0.30
    max_height_for_line = img_h * 0.05
    horizontal_ar_limit = 6.0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary, dtype=np.uint8)

    for i in range(1, n_labels):
        cx, cy, cw, ch, area = stats[i]

        # Reject if touching original image border.
        if cx <= 1 or cy <= 1 or cx + cw >= img_w - 1 or cy + ch >= img_h - 1:
            continue

        # Area bounds.
        if area < min_area or area > max_area:
            continue

        # Reject extremely large components relative to the ROI.
        if roi_pixel_area > 0 and area / roi_pixel_area > max_ratio:
            continue

        # Optional legacy experiment. Disabled by default because an
        # axis-aligned bounding box makes curved, thick, and short tracks look
        # compact even when they are valid particle trajectories.
        if min_ar is not None:
            long_side = max(cw, ch)
            short_side = max(min(cw, ch), 1)
            if long_side / short_side < min_ar:
                continue

        # Reject horizontal reflection / frame lines near the bottom.
        if (
            cy >= bottom_y_threshold
            and cw > min_width_for_line
            and ch < max_height_for_line
            and cw / max(ch, 1) > horizontal_ar_limit
        ):
            continue

        out[labels == i] = 255

    return cv2.bitwise_and(out, out, mask=roi_mask)



# Threshold helpers



def _compute_threshold(
    enhanced: np.ndarray,
    roi_mask: np.ndarray,
    percentile: float,
) -> float:
    """Return the intensity threshold for a given ROI percentile."""
    if isinstance(percentile, bool) or not isinstance(percentile, Real):
        raise TypeError("Threshold percentile must be a number.")
    percentile = float(percentile)
    if not 0 <= percentile <= 100:
        raise ValueError("Threshold percentile must be between 0 and 100.")

    roi_values = enhanced[roi_mask > 0]
    roi_values = roi_values[roi_values > 0]
    if roi_values.size == 0:
        return 255.0
    return float(np.percentile(roi_values, percentile))


def _threshold_to_binary(
    enhanced: np.ndarray,
    threshold_value: float,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """Binarise an enhanced image and clip to the ROI."""
    _, binary = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(binary, binary, mask=roi_mask)



# Public pipeline entry points



def run_clean_particle_preprocessing_pipeline(
    image_path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dual-branch preprocessing: electron + alpha/proton.

    Shared steps
    ~~~~~~~~~~~~
    1. Load image → BGR, RGB, grayscale.
    2. Median-filter the grayscale.

    Electron branch
    ~~~~~~~~~~~~~~~
    Tight ROI → white top-hat → CLAHE → 99.2-percentile threshold →
    small-component filter.

    Alpha / proton branch
    ~~~~~~~~~~~~~~~~~~~~~
    Wide ROI → broad Gaussian background subtraction (σ ≈ 45) → moderate
    CLAHE → 98.2-percentile threshold → elongation-based component
    filter.  Rejects only components that touch the original image
    border or occupy an extreme fraction of the ROI area.

    The two masks are combined with ``cv2.bitwise_or``.
    """
    cfg = _resolve_clean_config(config)
    e_cfg = cfg["electron"]
    ap_cfg = cfg["alpha_proton"]

    
    # Shared: load → grayscale → median filter
    
    original_bgr, original_rgb, grayscale = load_image(image_path)
    noise_reduced = apply_median_filter(
        grayscale,
        kernel_size=cfg["median_kernel_size"],
    )

    # ================================================================
    # ELECTRON BRANCH
    # ================================================================
    e_roi_mask, e_roi_details = _build_roi_mask(grayscale.shape, e_cfg["roi"])
    e_roi_masked = cv2.bitwise_and(noise_reduced, noise_reduced, mask=e_roi_mask)

    # White top-hat enhancement.
    e_illumination = correct_uneven_illumination(
        e_roi_masked,
        method="top_hat",
        kernel_size=e_cfg["top_hat_kernel_size"],
    )

    # CLAHE.
    e_enhanced = apply_clahe(
        e_illumination,
        clip_limit=e_cfg["clahe_clip_limit"],
        tile_grid_size=e_cfg["clahe_tile_grid_size"],
    )

    # Threshold + component filter.
    e_thresh = _compute_threshold(
        e_enhanced, e_roi_mask, e_cfg["threshold_percentile"]
    )
    e_binary = _threshold_to_binary(e_enhanced, e_thresh, e_roi_mask)
    electron_mask = _filter_electron_components(
        e_binary, e_roi_mask, e_roi_details, e_cfg["component_filter"]
    )

    # Remove chamber-frame artefacts from the electron mask.
    electron_mask = _remove_chamber_frame_artefacts(
        electron_mask, e_roi_details,
    )

    # ================================================================
    # ALPHA / PROTON BRANCH
    # ================================================================
    ap_roi_mask, ap_roi_details = _build_roi_mask(grayscale.shape, ap_cfg["roi"])
    ap_roi_masked = cv2.bitwise_and(
        noise_reduced, noise_reduced, mask=ap_roi_mask
    )

    # Broad Gaussian background subtraction.
    sigma = float(ap_cfg["gaussian_blur_sigma"])
    ksize = int(sigma * 6) | 1  # ≈ 6σ rule, ensure odd
    background = cv2.GaussianBlur(
        ap_roi_masked, (ksize, ksize), sigmaX=sigma
    )
    ap_foreground = cv2.subtract(ap_roi_masked, background)
    ap_illumination = cv2.normalize(
        ap_foreground, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )

    # Moderate CLAHE.
    ap_enhanced = apply_clahe(
        ap_illumination,
        clip_limit=ap_cfg["clahe_clip_limit"],
        tile_grid_size=ap_cfg["clahe_tile_grid_size"],
    )

    # Threshold + component filter.
    ap_thresh = _compute_threshold(
        ap_enhanced, ap_roi_mask, ap_cfg["threshold_percentile"]
    )
    ap_binary = _threshold_to_binary(ap_enhanced, ap_thresh, ap_roi_mask)
    alpha_proton_mask = _filter_alpha_proton_components(
        ap_binary,
        ap_roi_mask,
        ap_roi_details,
        grayscale.shape,
        ap_cfg["component_filter"],
    )

    # ================================================================
    # COMBINE
    # ================================================================
    combined_particle_mask = cv2.bitwise_or(electron_mask, alpha_proton_mask)

    # Final safety pass: remove any frame artefacts that survived the OR
    # combination.  The alpha/proton branch can contribute small edge
    # debris that the electron-only pass never saw.  Use more relaxed
    # thresholds here — wider boundary zone, lower span and AR limits —
    # to catch thicker or shorter frame fragments while still requiring
    # all three conditions (location + span + orientation) together.
    combined_particle_mask = _remove_chamber_frame_artefacts(
        combined_particle_mask,
        e_roi_details,
        boundary_fraction=0.10,
        min_span_fraction=0.05,
        min_aspect_ratio=2.5,
        image_shape=grayscale.shape,
    )
    final_image = combined_particle_mask

    return {
        # Shared
        "original_bgr": original_bgr,
        "original_rgb": original_rgb,
        "grayscale": grayscale,
        "noise_reduced": noise_reduced,
        # Electron branch
        "electron_roi_mask": e_roi_mask,
        "electron_roi_details": e_roi_details,
        "electron_enhanced": e_enhanced,
        "electron_mask": electron_mask,
        "electron_threshold_value": e_thresh,
        # Alpha / proton branch
        "alpha_proton_roi_mask": ap_roi_mask,
        "alpha_proton_roi_details": ap_roi_details,
        "alpha_proton_enhanced": ap_enhanced,
        "alpha_proton_mask": alpha_proton_mask,
        "alpha_proton_threshold_value": ap_thresh,
        # Combined
        "combined_particle_mask": combined_particle_mask,
        "final_image": final_image,
        # Backward-compatibility aliases used by the preprocessing CLI.
        "roi_mask": e_roi_mask,
        "roi_details": e_roi_details,
        "roi_masked": e_roi_masked,
        "illumination_corrected": e_illumination,
        "particle_enhanced": e_enhanced,
        "particle_binary": e_binary,
        "threshold_value": e_thresh,
        "config": cfg,
    }


def run_preprocessing_pipeline(
    image_path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified public entry point — delegates to the dual-branch pipeline."""
    return run_clean_particle_preprocessing_pipeline(image_path, config)
