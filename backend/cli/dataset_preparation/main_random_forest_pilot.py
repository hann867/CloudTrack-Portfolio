"""Run the isolated first Random Forest pilot evaluation."""

from __future__ import annotations

from pathlib import Path

from backend.src.dataset_preparation.random_forest_pilot import run_random_forest_pilot


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = BACKEND_ROOT / "data" / "processed" / "labelled_tracks"


def main() -> int:
    try:
        result = run_random_forest_pilot(
            DATASET_ROOT / "labels.csv",
            DATASET_ROOT / "features" / "labelled_geometric_features.csv",
            DATASET_ROOT / "analysis" / "random_forest_pilot",
        )
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 1
    print(result["summary"], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

