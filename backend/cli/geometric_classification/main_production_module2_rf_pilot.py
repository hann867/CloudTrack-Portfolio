"""Run the isolated production-Module-2 Random Forest exploratory pilot."""

from __future__ import annotations

import json
from pathlib import Path

from backend.src.geometric_classification.production_module2_rf_pilot import (
    run_production_module2_rf_pilot,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
FEATURES = (
    BACKEND_ROOT
    / "data/processed/geometric_classification/production_module2_features"
    / "revised_confirmed_module2_geometric_features.csv"
)
OUTPUT = (
    BACKEND_ROOT
    / "data/processed/geometric_classification/production_module2_rf_pilot"
)


def main() -> int:
    try:
        result = run_production_module2_rf_pilot(FEATURES, OUTPUT)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}, indent=2))
        return 1
    primary = result["primary"]
    print(json.dumps({
        "status": "success",
        "title": "PRODUCTION-MODULE-2 RF EXPLORATORY PILOT",
        "n": len(result["frame"]),
        "groups": primary.group_count,
        "folds": primary.fold_count,
        "leakage_detected": primary.leakage_detected,
        "metrics": primary.metrics,
        "misclassified": len(result["misclassified"]),
        "output_dir": str(result["output_dir"].resolve()),
        "production_module2_rf_pilot_trained": True,
        "final_classifier_trained": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
