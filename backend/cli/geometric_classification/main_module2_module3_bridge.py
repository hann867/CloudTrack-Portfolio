"""Isolated command-line demonstration of the Module 2 to Module 3 bridge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from backend.src.detection_segmentation import CONFIG, process_image, validate_config
from backend.src.geometric_classification import adapt_module2_result, export_module2_bridge_diagnostics


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTICS_ROOT = (
    BACKEND_ROOT / "data" / "processed" / "geometric_classification" / "module2_bridge_diagnostics"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run existing Module 2 and extract Module 3 geometry directly from its ROI masks."
    )
    parser.add_argument("image", type=Path, help="PNG or JPEG image accepted by production Module 2")
    parser.add_argument(
        "--export-diagnostics",
        action="store_true",
        help="Export analysis-only copies of Module 2 image/mask ROIs, overlays, and a manifest",
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    image = arguments.image.expanduser().resolve()
    if not image.is_file():
        print(json.dumps({"status": "error", "error_message": f"Image not found: {image}"}, indent=2))
        return 1
    try:
        validate_config(CONFIG)
        module2_result = process_image(image, CONFIG)
        tracks = adapt_module2_result(module2_result)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2))
        return 1
    summary = {
        "status": "success",
        "source_image": str(image),
        "module2_accepted_track_count": int(module2_result["accepted_component_count"]),
        "module3_success_count": sum(record["processing_status"] == "success" for record in tracks),
        "module3_failure_count": sum(record["processing_status"] != "success" for record in tracks),
        "tracks": tracks,
    }
    if arguments.export_diagnostics:
        try:
            summary["diagnostic_export"] = export_module2_bridge_diagnostics(
                module2_result,
                tracks,
                arguments.diagnostics_root.expanduser().resolve(),
                source_stem=image.stem,
            )
        except (OSError, TypeError, ValueError) as exc:
            print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2))
            return 1
    print(json.dumps(_json_safe(summary), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
