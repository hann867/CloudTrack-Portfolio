"""Synthetic verification for Module 3 geometry and ROI detection."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.geometric_classification.annotation_roi import crop_inside_rectangle, detect_red_rectangle
from src.geometric_classification.feature_extractor import analyse_annotated_image, extract_geometric_features


class GeometricFeatureTests(unittest.TestCase):
    def test_straight_line_has_unit_tortuosity_and_low_curvature(self):
        mask = np.zeros((80, 140), np.uint8)
        cv2.line(mask, (15, 40), (125, 40), 255, 3)
        result = extract_geometric_features(mask)
        self.assertEqual(result["processing_status"], "success")
        self.assertAlmostEqual(result["tortuosity"], 1.0, places=3)
        self.assertLess(result["mean_curvature"], 0.01)

    def test_curved_line_is_more_tortuous_and_curved(self):
        straight = np.zeros((120, 160), np.uint8)
        curved = np.zeros_like(straight)
        cv2.line(straight, (20, 80), (140, 80), 255, 3)
        cv2.ellipse(curved, (80, 85), (60, 50), 180, 0, 180, 255, 3)
        straight_features = extract_geometric_features(straight)
        curved_features = extract_geometric_features(curved)
        self.assertEqual(curved_features["processing_status"], "success")
        self.assertGreater(curved_features["tortuosity"], straight_features["tortuosity"] + 0.1)
        self.assertGreater(curved_features["mean_curvature"], straight_features["mean_curvature"])

    def test_thick_track_is_wider_than_thin_track(self):
        thin = np.zeros((80, 140), np.uint8)
        thick = np.zeros_like(thin)
        cv2.line(thin, (15, 40), (125, 40), 255, 1)
        cv2.line(thick, (15, 40), (125, 40), 255, 9)
        self.assertGreater(extract_geometric_features(thick)["mean_width_px"], extract_geometric_features(thin)["mean_width_px"])

    def test_diagonal_length_uses_sqrt_two_weights(self):
        mask = np.zeros((50, 50), np.uint8)
        cv2.line(mask, (10, 10), (30, 30), 255, 1)
        self.assertAlmostEqual(extract_geometric_features(mask)["track_length_px"], 20 * math.sqrt(2), places=5)

    def test_empty_mask_fails_gracefully(self):
        result = extract_geometric_features(np.zeros((20, 20), np.uint8))
        self.assertEqual(result["processing_status"], "failed")
        self.assertIn("empty", result["error_message"].lower())


class AnnotationTests(unittest.TestCase):
    def test_missing_rectangle_fails_gracefully(self):
        self.assertEqual(detect_red_rectangle(np.zeros((100, 120, 3), np.uint8))["processing_status"], "failed")

    def test_jpeg_red_rectangle_and_inward_crop(self):
        image = np.full((120, 180, 3), 30, np.uint8)
        cv2.rectangle(image, (25, 20), (155, 100), (0, 0, 255), 3)
        cv2.line(image, (45, 65), (135, 55), (230, 230, 230), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compressed.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            compressed = cv2.imread(str(path))
            rectangle = detect_red_rectangle(compressed)
            self.assertEqual(rectangle["processing_status"], "success")
            self.assertLess(abs(rectangle["x"] - 25), 5)
            roi = crop_inside_rectangle(compressed, rectangle, inward_margin=5)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red = cv2.inRange(hsv, (0, 65, 55), (12, 255, 255)) | cv2.inRange(hsv, (165, 65, 55), (179, 255, 255))
            self.assertLess(np.count_nonzero(red), roi.size * 0.002)

    def test_annotated_image_without_rectangle_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.png"
            cv2.imwrite(str(path), np.zeros((50, 50, 3), np.uint8))
            self.assertEqual(analyse_annotated_image(path, directory)["processing_status"], "failed")
