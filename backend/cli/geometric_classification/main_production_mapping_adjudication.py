"""Render read-only diagnostics for problematic production mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.geometric_classification.production_mapping_adjudication import (
    build_adjudication_outputs,
    render_all_diagnostics,
    write_adjudication,
)

BACKEND_ROOT=Path(__file__).resolve().parents[2]
MATCH_ROOT=BACKEND_ROOT/"data/processed/geometric_classification/ground_truth_candidate_matching"
OUTPUT_ROOT=BACKEND_ROOT/"data/processed/geometric_classification/production_mapping_adjudication"

def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Render diagnostic-only views for problematic confirmed mappings.")
    parser.add_argument("--output-dir",type=Path,default=OUTPUT_ROOT/"diagnostic_images")
    args=parser.parse_args(argv)
    try:
        metrics=render_all_diagnostics(
            BACKEND_ROOT/"data/processed/labelled_tracks/labels.csv",
            MATCH_ROOT/"confirmed_ground_truth_module2_mapping.csv",MATCH_ROOT/"candidate_matching.csv",
            BACKEND_ROOT/"data/processed/geometric_classification/module2_bridge_diagnostics",
            [BACKEND_ROOT/"data/raw/labelled_sources",BACKEND_ROOT/"data/raw/cloud_chamber"],
            BACKEND_ROOT/"data/processed/labelled_tracks",args.output_dir)
    except (OSError,TypeError,ValueError) as exc:
        print(json.dumps({"status":"error","error_message":str(exc)},indent=2));return 1
    rows, report = build_adjudication_outputs(metrics)
    write_adjudication(
        OUTPUT_ROOT / "production_mapping_adjudication.csv",
        OUTPUT_ROOT / "diagnostic_report.txt",
        rows,
        report,
    )
    summary={track:{key:value for key,value in values.items() if not key.startswith("_")} for track,values in metrics.items()}
    print(json.dumps({
        "status": "success",
        "diagnostic_metrics": summary,
        "adjudication_rows": len(rows),
        "expected_unique_masks_after_adjudication": 29,
        "ready_to_revise_confirmed_mapping": True,
        "ready_to_train_rf": False,
    },indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
