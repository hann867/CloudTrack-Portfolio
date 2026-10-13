"""Regression tests for diagnostic-only source-confound analysis."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dataset_preparation.source_confound_analysis import (
    FEATURES,
    analyse_source_confound,
    categorical_association,
    class_roi_contingency,
    class_source_contingency,
    identify_cross_source_samples,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"track_id": "A001", "class": "Alpha", "source_dataset": "AWAN", "roi_type": "rotated", "ground_truth_quality": "high", "crop_filename": "A001.png"},
            {"track_id": "A002", "class": "Alpha", "source_dataset": "AWAN", "roi_type": "rotated", "ground_truth_quality": "high", "crop_filename": "A002.png"},
            {"track_id": "A003", "class": "Alpha", "source_dataset": "Lecturer", "roi_type": "axis_aligned", "ground_truth_quality": "high", "crop_filename": "A003.png"},
            {"track_id": "E001", "class": "Electron", "source_dataset": "Lecturer", "roi_type": "axis_aligned", "ground_truth_quality": "high", "crop_filename": "E001.png"},
            {"track_id": "E002", "class": "Electron", "source_dataset": "Lecturer", "roi_type": "axis_aligned", "ground_truth_quality": "medium", "crop_filename": "E002.png"},
            {"track_id": "E003", "class": "Electron", "source_dataset": "AWAN", "roi_type": "rotated", "ground_truth_quality": "high", "crop_filename": "E003.png"},
        ]
    )


def _features() -> pd.DataFrame:
    rows = []
    for index, track_id in enumerate(("A001", "A002", "A003", "E001", "E002", "E003"), start=1):
        row = {"track_id": track_id, "processing_status": "success"}
        row.update({feature: float(index + offset / 10) for offset, feature in enumerate(FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


class SourceConfoundAnalysisTests(unittest.TestCase):
    def test_contingency_table_contains_counts_and_percentages(self):
        table = class_source_contingency(_labels()).set_index("source_dataset")
        self.assertEqual(table.loc["AWAN", "alpha_count"], 2)
        self.assertEqual(table.loc["AWAN", "electron_count"], 1)
        self.assertAlmostEqual(table.loc["AWAN", "alpha_row_pct"], 200 / 3)
        self.assertAlmostEqual(table.loc["Lecturer", "electron_column_pct"], 200 / 3)

    def test_class_source_association_uses_fisher_and_phi(self):
        labels = pd.DataFrame(
            [{"class": "Alpha", "source_dataset": "AWAN"}] * 3
            + [{"class": "Electron", "source_dataset": "Lecturer"}] * 3
        )
        association = categorical_association(class_source_contingency(labels))
        self.assertAlmostEqual(association["phi_cramers_v"], 1.0)
        self.assertIsNotNone(association["fisher_p_value"])

    def test_cross_source_samples_are_identified(self):
        frame = _labels().merge(_features(), on="track_id")
        cross = identify_cross_source_samples(frame)
        self.assertEqual(set(cross["track_id"]), {"A003", "E003"})

    def test_roi_type_contingency_is_generated(self):
        table = class_roi_contingency(_labels()).set_index("roi_type")
        self.assertEqual(table.loc["rotated", "alpha_count"], 2)
        self.assertEqual(table.loc["axis_aligned", "electron_count"], 2)

    def test_analysis_preserves_label_and_feature_csv_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path, features_path = root / "labels.csv", root / "features.csv"
            _labels().to_csv(labels_path, index=False)
            _features().to_csv(features_path, index=False)
            label_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
            feature_hash = hashlib.sha256(features_path.read_bytes()).hexdigest()

            result = analyse_source_confound(labels_path, features_path, root / "analysis", create_plots=False)

            self.assertEqual(hashlib.sha256(labels_path.read_bytes()).hexdigest(), label_hash)
            self.assertEqual(hashlib.sha256(features_path.read_bytes()).hexdigest(), feature_hash)
            self.assertTrue((root / "analysis" / "source_confound_report.txt").is_file())
            self.assertIn(result["risk"], {"LOW", "MODERATE", "HIGH"})

    def test_small_sample_association_fails_safely(self):
        labels = pd.DataFrame([{"class": "Alpha", "source_dataset": "AWAN"}])
        association = categorical_association(class_source_contingency(labels))
        self.assertIsNone(association["fisher_p_value"])
        self.assertIsNone(association["phi_cramers_v"])


if __name__ == "__main__":
    unittest.main()

