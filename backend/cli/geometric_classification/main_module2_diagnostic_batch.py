"""Generate missing Module 2 bridge diagnostics for labelled source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.detection_segmentation import CONFIG, process_image, validate_config
from backend.src.geometric_classification.module2_adapter import adapt_module2_result
from backend.src.geometric_classification.module2_bridge_diagnostics import export_module2_bridge_diagnostics
from backend.src.geometric_classification.module2_diagnostic_batch import generate_diagnostic_batch


BACKEND_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate missing production Module 2 diagnostics for labelled source images."
    )
    parser.add_argument("--labels", type=Path, default=BACKEND_ROOT / "data/processed/labelled_tracks/labels.csv")
    parser.add_argument("--source-root", action="append", type=Path, dest="source_roots")
    parser.add_argument("--diagnostics-root", type=Path,
                        default=BACKEND_ROOT / "data/processed/geometric_classification/module2_bridge_diagnostics")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = args.source_roots or [
        BACKEND_ROOT / "data/raw/labelled_sources",
        BACKEND_ROOT / "data/raw/cloud_chamber",
    ]
    try:
        validate_config(CONFIG)
        result = generate_diagnostic_batch(
            args.labels, roots, args.diagnostics_root,
            process_image_fn=process_image,
            module2_config=CONFIG,
            adapter_fn=adapt_module2_result,
            exporter_fn=export_module2_bridge_diagnostics,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2))
        return 1
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
