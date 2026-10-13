import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.src.detection_segmentation import CONFIG, process_image
from backend.src.geometric_classification.production_inference import classify_module2_masks
from backend.src.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline
from backend.src.video_processing.video_frame_pipeline import process_extracted_frames


def frame_mapping(output_index, source_index):
    return {
        "output_frame_index": output_index,
        "source_frame_index": source_index,
        "timestamp_seconds": output_index / 30,
        "filename": f"frame_{output_index + 1:06d}.png",
    }


def detection_result(image, with_track=True):
    height, width = image.shape[:2]
    labels = np.zeros((height, width), dtype=np.int32)
    rois = []
    if with_track:
        labels[8:16, 10:42] = 1
        mask = np.where(labels == 1, 255, 0).astype(np.uint8)
        rois.append(
            {
                "track_id": "ignored_source_track_0001",
                "component_id": 1,
                "label_id": 1,
                "bbox_x": 10,
                "bbox_y": 8,
                "bbox_width": 32,
                "bbox_height": 8,
                "area_pixels": 256,
                "_image_roi": image[6:18, 8:44].copy(),
                "_mask_roi": mask[6:18, 8:44].copy(),
            }
        )
    return {
        "annotated": image.copy(),
        "binary": np.where(labels > 0, 255, 0).astype(np.uint8),
        "filtered_mask": np.where(labels > 0, 255, 0).astype(np.uint8),
        "accepted_labels": labels,
        "rois": rois,
    }


def classification_result():
    return {
        "processing_status": "SUCCESS",
        "results": [
            {
                "processing_status": "success",
                "error_code": None,
                "error_message": None,
                "predicted_class": "Alpha",
                "confidence": 0.9,
                "probabilities": {"Alpha": 0.9, "Electron": 0.1},
                "features": {
                    "track_length_px": 30.0,
                    "mean_width_px": 4.0,
                    "median_width_px": 4.0,
                    "max_width_px": 5.0,
                    "mean_curvature": 0.1,
                    "max_curvature": 0.2,
                    "tortuosity": 1.1,
                },
                "physical_features": {},
                "model": {"model_version": "test", "model_stage": "development"},
                "provenance": {},
            }
        ],
    }


class VideoFramePipelineTests(unittest.TestCase):
    def make_job(self, root, mappings):
        job = root / "job123"
        frames = job / "frames"
        frames.mkdir(parents=True)
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        for mapping in mappings:
            cv2.imwrite(str(frames / mapping["filename"]), image)
        return job, image

    def test_duplicate_source_is_processed_once_but_all_outputs_exist(self):
        mappings = [frame_mapping(0, 0), frame_mapping(1, 0)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            job, image = self.make_job(Path(temporary_directory), mappings)
            with (
                patch(
                    "backend.src.video_processing.video_frame_pipeline.run_preprocessing_pipeline",
                    return_value={"final_image": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)},
                ) as preprocess,
                patch(
                    "backend.src.video_processing.video_frame_pipeline.process_image",
                    return_value=detection_result(image),
                ) as detect,
                patch(
                    "backend.src.video_processing.video_frame_pipeline.classify_module2_masks",
                    return_value=classification_result(),
                ) as classify,
            ):
                results = process_extracted_frames(job, mappings)

            self.assertEqual(preprocess.call_count, 1)
            self.assertEqual(detect.call_count, 1)
            self.assertEqual(classify.call_count, 1)
            self.assertTrue(results["frames"][1]["is_duplicate_source_frame"])
            self.assertEqual(results["unique_source_frames_processed"], 1)
            self.assertEqual(results["particles"], {"total_tracks": 1, "alpha": 1, "electron": 0})
            self.assertEqual(len(list((job / "preprocessed").glob("frame_*.png"))), 2)
            self.assertEqual(len(list((job / "classified").glob("frame_*.png"))), 2)
            self.assertEqual(len(json.loads((job / "results.json").read_text())["frames"]), 2)

    def test_module1_and_module2_outputs_match_direct_image_processing_calls(self):
        mappings = [frame_mapping(0, 0)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            job, _ = self.make_job(root, mappings)
            raw_path = job / "frames" / "frame_000001.png"
            direct_preprocessed = run_preprocessing_pipeline(raw_path)["final_image"]
            direct_path = root / "direct_preprocessed.png"
            cv2.imwrite(str(direct_path), direct_preprocessed)
            direct_detection = process_image(direct_path, CONFIG)

            results = process_extracted_frames(job, mappings)
            video_preprocessed = cv2.imread(
                str(job / "preprocessed" / "frame_000001.png"), cv2.IMREAD_GRAYSCALE
            )

            self.assertTrue(np.array_equal(video_preprocessed, direct_preprocessed))
            self.assertEqual(
                results["frames"][0]["tracks_detected"],
                direct_detection["accepted_component_count"],
            )

    def test_zero_track_frame_is_successful_and_skips_classifier(self):
        mappings = [frame_mapping(0, 0)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            job, image = self.make_job(Path(temporary_directory), mappings)
            with (
                patch(
                    "backend.src.video_processing.video_frame_pipeline.run_preprocessing_pipeline",
                    return_value={"final_image": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)},
                ),
                patch(
                    "backend.src.video_processing.video_frame_pipeline.process_image",
                    return_value=detection_result(image, with_track=False),
                ),
                patch(
                    "backend.src.video_processing.video_frame_pipeline.classify_module2_masks"
                ) as classify,
            ):
                results = process_extracted_frames(job, mappings)

            self.assertEqual(results["failed_frames"], 0)
            self.assertEqual(results["frames"][0]["tracks_detected"], 0)
            self.assertEqual(results["frames"][0]["classification_status"], "SUCCESS")
            classify.assert_not_called()

    def test_failed_frame_is_recorded_and_classified_sequence_is_preserved(self):
        mappings = [frame_mapping(0, 0), frame_mapping(1, 0)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            job, _ = self.make_job(Path(temporary_directory), mappings)
            with patch(
                "backend.src.video_processing.video_frame_pipeline.run_preprocessing_pipeline",
                side_effect=RuntimeError("test preprocessing failure"),
            ):
                results = process_extracted_frames(job, mappings)

            self.assertEqual(results["failed_frames"], 2)
            self.assertIn("test preprocessing failure", results["frames"][0]["error"])
            self.assertTrue(results["frames"][1]["is_duplicate_source_frame"])
            self.assertEqual(len(list((job / "classified").glob("frame_*.png"))), 2)
            self.assertEqual(len(list((job / "preprocessed").glob("frame_*.png"))), 0)

    def test_track_classification_failure_marks_frame_failed_without_dropping_output(self):
        mappings = [frame_mapping(0, 0)]
        failed_classification = classification_result()
        failed_classification["processing_status"] = "FAILED"
        failed_classification["results"][0].update(
            {
                "processing_status": "failed",
                "error_message": "Geometric extraction failed.",
                "predicted_class": None,
                "confidence": None,
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            job, image = self.make_job(Path(temporary_directory), mappings)
            with (
                patch(
                    "backend.src.video_processing.video_frame_pipeline.run_preprocessing_pipeline",
                    return_value={"final_image": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)},
                ),
                patch(
                    "backend.src.video_processing.video_frame_pipeline.process_image",
                    return_value=detection_result(image),
                ),
                patch(
                    "backend.src.video_processing.video_frame_pipeline.classify_module2_masks",
                    return_value=failed_classification,
                ),
            ):
                results = process_extracted_frames(job, mappings)

            self.assertEqual(results["failed_frames"], 1)
            self.assertEqual(results["frames"][0]["failed_track_classifications"], 1)
            self.assertIn("Geometric extraction failed", results["frames"][0]["error"])
            self.assertTrue((job / "classified" / "frame_000001.png").is_file())

    def test_module3_result_matches_direct_classifier_for_same_module2_mask(self):
        mappings = [frame_mapping(0, 0)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            job, image = self.make_job(Path(temporary_directory), mappings)
            detection = detection_result(image)
            mask = detection["rois"][0]["_mask_roi"]
            direct = classify_module2_masks([mask])["results"][0]
            with (
                patch(
                    "backend.src.video_processing.video_frame_pipeline.run_preprocessing_pipeline",
                    return_value={"final_image": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)},
                ),
                patch(
                    "backend.src.video_processing.video_frame_pipeline.process_image",
                    return_value=detection,
                ),
            ):
                results = process_extracted_frames(job, mappings)

            video_classification = results["frames"][0]["tracks"][0]["classification"]
            self.assertEqual(video_classification["processing_status"], direct["processing_status"])
            self.assertEqual(video_classification["predicted_class"], direct["predicted_class"])
            self.assertEqual(video_classification["features"], direct["features"])
            self.assertEqual(video_classification["probabilities"], direct["probabilities"])


if __name__ == "__main__":
    unittest.main()
