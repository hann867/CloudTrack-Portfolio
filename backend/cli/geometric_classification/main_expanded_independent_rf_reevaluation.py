"""Run the isolated grouped RF re-evaluation on the finalized 34-row dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.geometric_classification.expanded_independent_rf_reevaluation import run_grouped_reevaluation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/geometric_classification/expanded_human_approved_module3_features/final_independent/final_independent_module3_features.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/geometric_classification/expanded_independent_rf_reevaluation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grouped exploratory RF re-evaluation; does not persist a model artifact.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--permutation-repeats", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        result = run_grouped_reevaluation(args.input, args.output_dir,
                                          bootstrap_repetitions=args.bootstrap_repetitions,
                                          permutation_repeats=args.permutation_repeats)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status":"error", "error_message":str(exc)}, indent=2)); return 1
    metrics = result.pop("metrics")
    print(json.dumps({"status":"success", **result,
                      "accuracy":metrics["accuracy"], "balanced_accuracy":metrics["balanced_accuracy"],
                      "macro_f1":metrics["macro_f1"], "mcc":metrics["mcc"],
                      "ci":{name:[metrics[f"{name}_ci_low"],metrics[f"{name}_ci_high"]]
                            for name in ("accuracy","balanced_accuracy","macro_f1","mcc")},
                      "output_dir":str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
