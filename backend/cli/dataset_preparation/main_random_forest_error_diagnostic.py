"""Run the read-only diagnostic for existing Random Forest pilot errors."""

from __future__ import annotations

from pathlib import Path

from backend.src.dataset_preparation.random_forest_error_diagnostic import run_error_diagnostic


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = BACKEND_ROOT / "data" / "processed" / "labelled_tracks"
PILOT_ROOT = DATASET_ROOT / "analysis" / "random_forest_pilot"


def main() -> int:
    try:
        result = run_error_diagnostic(
            DATASET_ROOT / "labels.csv",
            DATASET_ROOT / "features" / "labelled_geometric_features.csv",
            PILOT_ROOT / "pilot_oof_predictions.csv",
            DATASET_ROOT,
            PILOT_ROOT / "error_diagnostic",
        )
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 1
    print(result["report"], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

