"""Focused regression tests for labelled-dataset audit and feature extraction."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.dataset_preparation.dataset_audit import audit_labelled_dataset
from src.dataset_preparation.dataset_metadata import LABEL_COLUMNS, build_label_record
from src.dataset_preparation.labelled_feature_extraction import (
    GEOMETRIC_FEATURE_COLUMNS,
    SEGMENTATION_SOURCE,
    extract_labelled_geometric_features,
)


def _record(track_id: str, class_name: str, filename: str, event_id: str) -> dict[str, object]:
    label = {"Alpha": "alpha", "Electron": "electron", "Uncertain": "uncertain"}.get(class_name, "alpha")
    record = build_label_record(
        track_id=track_id,
        label=label,
        source_dataset="lecturer",
        source_image="source.jpg",
        roi=(1, 2, 8, 5),
        event_id=event_id,
    )
    record["class"] = class_name
    record["crop_filename"] = filename
    return record


def _write_labels(root: Path, records: list[dict[str, object]]) -> Path:
    path = root / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return path


class DatasetAuditTests(unittest.TestCase):
    def test_audit_detects_missing_crop_and_preserves_labels_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = _write_labels(root, [_record("A001", "Alpha", "A001.png", "EVENT_1")])
            original = labels.read_bytes()

            result = audit_labelled_dataset(root, root / "audit")

            self.assertTrue(any(issue["issue_code"] == "missing_crop" for issue in result.issues))
            self.assertEqual(result.record_statuses[0], "ERROR")
            self.assertEqual(labels.read_bytes(), original)

    def test_audit_detects_invalid_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_labels(root, [_record("A001", "Muon", "A001.png", "EVENT_1")])

            result = audit_labelled_dataset(root)

            self.assertTrue(any(issue["issue_code"] == "invalid_class" for issue in result.issues))

    def test_audit_reports_duplicate_event_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                _record("A001", "Alpha", "A001.png", "SHARED_EVENT"),
                _record("E001", "Electron", "E001.png", "SHARED_EVENT"),
            ]
            _write_labels(root, records)

            result = audit_labelled_dataset(root)

            duplicate = [row for row in result.duplicates if row["duplicate_type"] == "event_id"]
            self.assertEqual(len(duplicate), 1)
            self.assertIn("different labels", duplicate[0]["details"])

    def test_audit_detects_exact_duplicate_crop_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "alpha").mkdir()
            (root / "electron").mkdir()
            image = np.full((12, 18, 3), 77, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "alpha" / "A001.png"), image))
            self.assertTrue(cv2.imwrite(str(root / "electron" / "E001.png"), image))
            _write_labels(
                root,
                [
                    _record("A001", "Alpha", "A001.png", "EVENT_A"),
                    _record("E001", "Electron", "E001.png", "EVENT_E"),
                ],
            )

            result = audit_labelled_dataset(root)

            duplicates = [row for row in result.duplicates if row["duplicate_type"] == "exact_crop_sha256"]
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(set(duplicates[0]["track_ids"].split(";")), {"A001", "E001"})


class FeatureExtractionTests(unittest.TestCase):
    def test_failed_sample_does_not_stop_batch_and_uncertain_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_dir in ("alpha", "electron", "uncertain"):
                (root / class_dir).mkdir()
            for path, value in (
                (root / "alpha" / "A001.png", 10),
                (root / "electron" / "E001.png", 20),
                (root / "uncertain" / "U001.png", 30),
            ):
                self.assertTrue(cv2.imwrite(str(path), np.full((30, 50, 3), value, dtype=np.uint8)))
            records = [
                _record("A001", "Alpha", "A001.png", "EVENT_A"),
                _record("E001", "Electron", "E001.png", "EVENT_E"),
                _record("U001", "Uncertain", "U001.png", "EVENT_U"),
            ]

            def segmenter(image):
                if int(image[0, 0, 0]) == 20:
                    return {"processing_status": "failed", "error_message": "synthetic failure", "mask": None}
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.line(mask, (5, 15), (44, 15), 255, 2)
                return {"processing_status": "success", "error_message": None, "mask": mask}

            def extractor(_mask):
                values = {column: 1.0 for column in GEOMETRIC_FEATURE_COLUMNS}
                return {**values, "processing_status": "success", "error_message": None}

            rows = extract_labelled_geometric_features(
                root,
                records,
                root / "features",
                segmenter=segmenter,
                geometric_extractor=extractor,
            )

            self.assertEqual([row["class"] for row in rows], ["Alpha", "Electron"])
            self.assertEqual(rows[0]["processing_status"], "success")
            self.assertEqual(rows[1]["processing_status"], "failed")
            self.assertEqual(rows[1]["error_message"], "synthetic failure")
            self.assertEqual({row["segmentation_source"] for row in rows}, {SEGMENTATION_SOURCE})
            self.assertEqual(rows[0]["track_id"], "A001")
            self.assertEqual(rows[1]["track_id"], "E001")
            self.assertTrue((root / "features" / "labelled_geometric_features.csv").is_file())


if __name__ == "__main__":
    unittest.main()

