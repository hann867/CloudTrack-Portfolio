"""Synthetic regression tests for the integrated Module 2 pipeline."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.detection_segmentation import CONFIG, process_image


class DetectionPipelineTests(unittest.TestCase):
    def test_straight_binary_track_produces_one_roi(self):
        image = np.zeros((180, 320), dtype=np.uint8)
        cv2.line(image, (30, 90), (285, 90), 255, 5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_track.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            result = process_image(path, CONFIG)

        self.assertEqual(result["segmentation_mode"], "binary-preserving threshold")
        self.assertEqual(result["raw_component_count"], 1)
        self.assertEqual(result["accepted_component_count"], 1)
        self.assertEqual(len(result["rois"]), 1)
        self.assertGreater(np.count_nonzero(result["rois"][0]["_mask_roi"]), 0)


if __name__ == "__main__":
    unittest.main()
