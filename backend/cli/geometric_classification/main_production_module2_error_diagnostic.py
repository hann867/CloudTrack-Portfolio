"""Generate the isolated production Module-2 RF error diagnostic."""

from pathlib import Path

from backend.src.geometric_classification.production_module2_error_diagnostic import run_production_error_diagnostic


if __name__ == "__main__":
    backend = Path(__file__).resolve().parents[2]
    processed = backend / "data" / "processed"
    geometric = processed / "geometric_classification"
    result = run_production_error_diagnostic(
        geometric / "production_module2_features" / "revised_confirmed_module2_geometric_features.csv",
        geometric / "production_module2_rf_pilot" / "production_rf_oof_predictions.csv",
        processed / "labelled_tracks" / "labels.csv",
        geometric / "ground_truth_candidate_matching" / "confirmed_ground_truth_module2_mapping_revised.csv",
        backend / "data" / "raw" / "labelled_sources",
        geometric / "production_module2_rf_pilot" / "error_diagnostic",
    )
    print(result["report"])
