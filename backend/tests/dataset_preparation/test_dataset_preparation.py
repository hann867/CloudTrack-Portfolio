"""GUI-independent verification for manual track dataset preparation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import src.dataset_preparation.track_annotation_tool as annotation_tool

from src.dataset_preparation.crop_utils import (
    crop_original_pixels,
    extract_rotated_crop,
    map_display_roi_to_original,
    rotated_roi_bounding_box,
    rotated_roi_from_axis,
    rotated_roi_from_center,
    validate_roi,
    validate_rotated_roi,
)
from src.dataset_preparation.dataset_metadata import (
    LABEL_COLUMNS,
    LEGACY_LABEL_COLUMNS,
    append_label_record,
    build_label_record,
    class_metadata,
    create_event_id,
    ensure_dataset_structure,
    next_track_id,
    save_annotation,
)
from src.dataset_preparation.track_annotation_tool import (
    AnnotationDefaults,
    PreviewState,
    build_display_crop,
    build_crop_preview_frame,
    calculate_crop_preview_geometry,
    discover_image_files,
    is_save_confirmation,
    navigate_image_index,
)


class CropUtilityTests(unittest.TestCase):
    def test_display_coordinates_map_to_original_resolution(self):
        mapped = map_display_roi_to_original(
            (100, 100, 300, 200),
            original_size=(5696, 4288),
            display_size=(1424, 1072),
        )
        self.assertEqual(mapped, (400, 400, 1200, 800))

    def test_zoomed_panned_viewport_maps_to_original_coordinates(self):
        mapped = map_display_roi_to_original(
            (100, 80, 200, 160),
            original_size=(1000, 800),
            display_size=(500, 400),
            original_offset=(100, 150),
            viewport_original_size=(250, 200),
        )
        self.assertEqual(mapped, (150, 190, 100, 80))

    def test_invalid_rois_are_rejected(self):
        for roi in ((0, 0, 0, 5), (-1, 0, 5, 5), (8, 8, 3, 3)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                validate_roi(roi, 10, 10)

    def test_crop_preserves_original_resolution_and_pixels(self):
        image = np.arange(12 * 16 * 3, dtype=np.uint16).reshape(12, 16, 3)
        crop = crop_original_pixels(image, (3, 2, 7, 5))
        self.assertEqual(crop.shape, (5, 7, 3))
        self.assertEqual(crop.dtype, image.dtype)
        np.testing.assert_array_equal(crop, image[2:7, 3:10])

    def test_rotated_roi_coordinates_and_bounding_box(self):
        roi = rotated_roi_from_center((50, 40), width=40, height=10, angle_degrees=30)
        validate_rotated_roi(roi, 100, 80)
        self.assertAlmostEqual(roi.center_x, 50)
        self.assertAlmostEqual(roi.center_y, 40)
        self.assertAlmostEqual(roi.width, 40)
        self.assertAlmostEqual(roi.height, 10)
        self.assertEqual(len(roi.corners), 4)
        bounding = rotated_roi_bounding_box(roi, 100, 80)
        self.assertLessEqual(bounding[0], min(point[0] for point in roi.corners))
        self.assertGreaterEqual(bounding[0] + bounding[2], max(point[0] for point in roi.corners))

    def test_rotated_crop_uses_requested_output_geometry(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(120, dtype=np.uint8)
        image[:, :, 1] = np.arange(100, dtype=np.uint8)[:, None]
        roi = rotated_roi_from_center((60, 50), width=40, height=12, angle_degrees=35)
        crop = extract_rotated_crop(image, roi)
        self.assertEqual(crop.shape, (12, 40, 3))
        self.assertAlmostEqual(int(crop[6, 20, 0]), 60, delta=2)
        self.assertAlmostEqual(int(crop[6, 20, 1]), 50, delta=2)

    def test_rotated_axis_gesture_preserves_direction_angle(self):
        roi = rotated_roi_from_axis((20, 20), (60, 60), half_height=6)
        self.assertAlmostEqual(roi.angle_degrees, 45)
        self.assertAlmostEqual(roi.width, np.hypot(40, 40))
        self.assertAlmostEqual(roi.height, 12)


class MetadataTests(unittest.TestCase):
    def test_labels_csv_and_output_directories_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_dataset_structure(root)
            for name in ("alpha", "electron", "uncertain"):
                self.assertTrue((root / name).is_dir())
            with (root / "labels.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), LABEL_COLUMNS)
            self.assertTrue((root / "source_metadata.csv").is_file())

    def test_sequential_naming_uses_files_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_dataset_structure(root)
            (root / "alpha" / "A001.png").touch()
            record = build_label_record(
                track_id="A004",
                label="alpha",
                source_dataset="test",
                source_image="source.png",
                roi=(0, 0, 1, 1),
                event_id="EVENT_4",
            )
            append_label_record(root / "labels.csv", record)
            self.assertEqual(next_track_id(root, "alpha"), "A005")
            self.assertEqual(next_track_id(root, "electron"), "E001")

    def test_append_preserves_entry_and_refuses_duplicate_id(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.csv"
            record = build_label_record(
                track_id="U001",
                label="uncertain",
                source_dataset="AWAN",
                source_image="source.jpg",
                roi=(2, 3, 4, 5),
                event_id="AWAN_EVENT_1",
            )
            append_label_record(labels_path, record)
            with self.assertRaises(FileExistsError):
                append_label_record(labels_path, record)
            with labels_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["track_id"], "U001")

    def test_legacy_csv_rows_are_preserved_during_additive_schema_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.csv"
            legacy_record = {
                "track_id": "A001",
                "class": "Alpha",
                "crop_filename": "A001.png",
                "source_dataset": "AWAN",
                "source_image": "source.jpg",
                "source_isotope": "Am-241",
                "physical_origin": "alpha_decay",
                "source_panel": "left",
                "x": "10",
                "y": "20",
                "width": "30",
                "height": "40",
                "label_basis": "known_source_manual_track_selection",
                "ground_truth_quality": "high",
                "event_id": "AWAN_EVENT_OLD",
                "notes": "keep this text",
            }
            with labels_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LEGACY_LABEL_COLUMNS)
                writer.writeheader()
                writer.writerow(legacy_record)
            complete_new_record = build_label_record(
                track_id="E001",
                label="electron",
                source_dataset="McGill",
                source_image="new.jpg",
                roi=(1, 2, 3, 4),
                event_id="NEW_EVENT",
            )
            legacy_style_new_record = {
                column: complete_new_record[column] for column in LEGACY_LABEL_COLUMNS
            }
            append_label_record(labels_path, legacy_style_new_record)
            with labels_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, LABEL_COLUMNS)
            for column in LEGACY_LABEL_COLUMNS:
                self.assertEqual(rows[0][column], legacy_record[column])
            self.assertEqual(rows[0]["roi_type"], "axis_aligned")
            self.assertEqual(rows[0]["rotation_angle_degrees"], "0")
            self.assertEqual(rows[1]["roi_type"], "axis_aligned")

    def test_alpha_metadata_is_manual_known_source(self):
        self.assertEqual(
            class_metadata("A"),
            {
                "class": "Alpha",
                "label_basis": "known_source_manual_track_selection",
                "ground_truth_quality": "high",
            },
        )

    def test_electron_metadata_is_manual_known_source(self):
        self.assertEqual(class_metadata("E")["class"], "Electron")
        self.assertEqual(class_metadata("E")["label_basis"], "known_source_manual_track_selection")

    def test_uncertain_metadata_is_low_quality_and_uncertain(self):
        metadata = class_metadata("U")
        self.assertEqual(metadata["class"], "Uncertain")
        self.assertEqual(metadata["label_basis"], "uncertain")
        self.assertEqual(metadata["ground_truth_quality"], "low")

    def test_event_ids_are_unique_and_dataset_scoped(self):
        first = create_event_id("AWAN")
        second = create_event_id("AWAN")
        self.assertTrue(first.startswith("AWAN_EVENT_"))
        self.assertNotEqual(first, second)


class SaveWorkflowTests(unittest.TestCase):
    def test_contrast_preview_does_not_modify_original_crop(self):
        crop = np.full((40, 90, 3), 82, dtype=np.uint8)
        cv2.line(crop, (8, 30), (82, 12), (91, 91, 91), 1)
        original = crop.copy()
        display = build_display_crop(crop, PreviewState(contrast_enabled=True))
        np.testing.assert_array_equal(crop, original)
        self.assertFalse(np.shares_memory(display, crop))
        self.assertFalse(np.array_equal(display, crop))

    def test_grayscale_preview_does_not_modify_original_crop(self):
        crop = np.zeros((30, 70, 3), dtype=np.uint8)
        crop[:, :, 0] = 35
        crop[:, :, 1] = 70
        crop[:, :, 2] = 120
        original = crop.copy()
        display = build_display_crop(crop, PreviewState(grayscale_enabled=True))
        np.testing.assert_array_equal(crop, original)
        np.testing.assert_array_equal(display[:, :, 0], display[:, :, 1])
        np.testing.assert_array_equal(display[:, :, 1], display[:, :, 2])

    def test_zoomed_preview_does_not_modify_original_crop(self):
        crop = np.arange(35 * 80 * 3, dtype=np.uint8).reshape(35, 80, 3)
        original = crop.copy()
        state = PreviewState()
        state.cycle_zoom()
        state.cycle_zoom()
        frame = build_crop_preview_frame(crop, "electron", state=state)
        np.testing.assert_array_equal(crop, original)
        self.assertEqual(state.magnification, 2.0)
        self.assertEqual(frame.shape, (700, 1000, 3))

    def test_restore_original_disables_all_display_modes(self):
        state = PreviewState(zoom_index=3, contrast_enabled=True, grayscale_enabled=True)
        state.restore_original()
        self.assertEqual(state.magnification, 1.0)
        self.assertFalse(state.contrast_enabled)
        self.assertFalse(state.grayscale_enabled)
        self.assertEqual(state.description(), "ORIGINAL")

    def test_crop_preview_is_usable_proportional_and_non_destructive(self):
        crop = np.arange(24 * 120 * 3, dtype=np.uint8).reshape(24, 120, 3)
        original = crop.copy()
        geometry = calculate_crop_preview_geometry((120, 24))
        frame = build_crop_preview_frame(crop, "alpha")

        self.assertEqual(frame.shape, (700, 1000, 3))
        self.assertGreater(geometry.width, 120)
        self.assertAlmostEqual(geometry.width / geometry.height, 120 / 24, delta=0.05)
        self.assertGreaterEqual(geometry.y, 84)
        np.testing.assert_array_equal(crop, original)

    def test_preview_modes_cannot_change_saved_axis_aligned_crop(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(100, dtype=np.uint8)
        image[:, :, 1] = np.arange(60, dtype=np.uint8)[:, None]
        image[:, :, 2] = 143
        roi = (12, 9, 65, 34)
        extracted_crop = crop_original_pixels(image, roi)
        original = extracted_crop.copy()

        states = (
            PreviewState(contrast_enabled=True),
            PreviewState(grayscale_enabled=True),
            PreviewState(zoom_index=3),
            PreviewState(zoom_index=2, contrast_enabled=True, grayscale_enabled=True),
        )
        for state in states:
            build_crop_preview_frame(extracted_crop, "electron", state=state)
        np.testing.assert_array_equal(extracted_crop, original)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = save_annotation(
                image=image,
                roi=roi,
                prepared_crop=extracted_crop,
                label="electron",
                output_root=root,
                source_dataset="McGill",
                source_image="faint_pb210.jpg",
                source_isotope="Pb-210",
                physical_origin="beta_minus_decay",
                event_id="PB210_FAINT_001",
            )
            saved = cv2.imread(str(root / "electron" / record["crop_filename"]), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(saved, original)

    def test_saved_png_contains_exact_original_pixels(self):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(12, dtype=np.uint8)
        image[:, :, 1] = np.arange(10, dtype=np.uint8)[:, None]
        image[:, :, 2] = 177
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = save_annotation(
                image=image,
                roi=(3, 2, 6, 5),
                label="electron",
                output_root=root,
                source_dataset="McGill",
                source_image="known.png",
                source_isotope="Pb-210",
                physical_origin="beta_minus_decay",
                event_id="MCGILL_EVENT_001",
            )
            saved = cv2.imread(str(root / "electron" / record["crop_filename"]), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(saved, image[2:7, 3:9])

    def test_rotated_preview_crop_is_saved_and_provenance_persisted(self):
        image = np.zeros((90, 110, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(110, dtype=np.uint8)
        image[:, :, 1] = np.arange(90, dtype=np.uint8)[:, None]
        rotated_roi = rotated_roi_from_center((55, 45), width=36, height=14, angle_degrees=-28)
        previewed_crop = extract_rotated_crop(image, rotated_roi)
        bounding_roi = rotated_roi_bounding_box(rotated_roi, 110, 90)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = save_annotation(
                image=image,
                roi=bounding_roi,
                rotated_roi=rotated_roi,
                prepared_crop=previewed_crop,
                label="alpha",
                output_root=root,
                source_dataset="AWAN",
                source_image="known.jpg",
                event_id="ROTATED_EVENT_001",
            )
            saved = cv2.imread(str(root / "alpha" / record["crop_filename"]), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(saved, previewed_crop)
            with (root / "labels.csv").open(newline="", encoding="utf-8") as handle:
                persisted = next(csv.DictReader(handle))
            self.assertEqual(persisted["roi_type"], "rotated")
            self.assertEqual(float(persisted["center_x"]), 55)
            self.assertEqual(float(persisted["center_y"]), 45)
            self.assertEqual(float(persisted["roi_width"]), 36)
            self.assertEqual(float(persisted["roi_height"]), 14)
            self.assertEqual(float(persisted["rotation_angle_degrees"]), -28)
            self.assertTrue(all(persisted[f"corner_{index}_x"] for index in range(1, 5)))

    def test_cancellation_is_safe_without_explicit_save(self):
        for decision in (None, "quit", "redraw", "uncertain", "S"):
            with self.subTest(decision=decision):
                self.assertFalse(is_save_confirmation(decision))
        self.assertTrue(is_save_confirmation("save"))


class BatchWorkflowTests(unittest.TestCase):
    def test_both_roi_headers_show_batch_navigation_controls(self):
        image = np.zeros((140, 900, 3), dtype=np.uint8)
        expected = "N next image | P previous image | K skip image | R reset | Q quit"

        for selector_type in (annotation_tool.ViewportROISelector, annotation_tool.OrientedROISelector):
            with self.subTest(selector=selector_type.__name__), mock.patch.object(
                annotation_tool.cv2, "putText", wraps=annotation_tool.cv2.putText
            ) as put_text:
                selector_type(image, batch_mode=True)._render()
                displayed_text = [call.args[1] for call in put_text.call_args_list]

            self.assertIn(expected, displayed_text)

    def test_n_p_k_navigate_without_saving_from_both_roi_selectors(self):
        image = np.arange(120 * 180 * 3, dtype=np.uint8).reshape(120, 180, 3)
        original = image.copy()
        defaults = AnnotationDefaults("lecturer", "", "unknown")

        for roi_mode in ("axis_aligned", "rotated"):
            for key in ("n", "p", "k"):
                with self.subTest(roi_mode=roi_mode, key=key), tempfile.TemporaryDirectory() as directory, mock.patch.object(
                    annotation_tool, "prompt_batch_action", return_value=roi_mode
                ), mock.patch.object(annotation_tool.cv2, "namedWindow"), mock.patch.object(
                    annotation_tool.cv2, "setMouseCallback"
                ), mock.patch.object(annotation_tool.cv2, "imshow"), mock.patch.object(
                    annotation_tool.cv2, "destroyWindow"
                ), mock.patch.object(
                    annotation_tool.cv2, "waitKey", return_value=ord(key)
                ), mock.patch.object(
                    annotation_tool, "choose_manual_label"
                ) as choose_label, mock.patch.object(annotation_tool, "save_annotation") as save:
                    command = annotation_tool._run_image_session(
                        Path(directory) / "source.jpg",
                        image,
                        Path(directory) / "output",
                        defaults,
                        batch_mode=True,
                    )

                self.assertEqual(command, key)
                choose_label.assert_not_called()
                save.assert_not_called()
                np.testing.assert_array_equal(image, original)

    def test_single_image_selector_ignores_batch_navigation_safely(self):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        selector = annotation_tool.ViewportROISelector(image, batch_mode=False)
        with mock.patch.object(annotation_tool.cv2, "namedWindow"), mock.patch.object(
            annotation_tool.cv2, "setMouseCallback"
        ), mock.patch.object(annotation_tool.cv2, "imshow"), mock.patch.object(
            annotation_tool.cv2, "destroyWindow"
        ), mock.patch.object(annotation_tool.cv2, "waitKey", side_effect=(ord("n"), ord("q"))):
            result = selector.select()

        self.assertIsNone(result)

    def test_folder_discovery_uses_deterministic_filename_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("frame10.JPG", "frame02.jpeg", "Frame01.png", "ignore.txt"):
                (root / name).touch()
            (root / "nested.jpg").mkdir()

            discovered = discover_image_files(root)

            self.assertEqual(
                [path.name for path in discovered],
                ["Frame01.png", "frame02.jpeg", "frame10.JPG"],
            )

    def test_next_previous_and_skip_navigation(self):
        self.assertEqual(navigate_image_index(1, 3, "n"), 2)
        self.assertEqual(navigate_image_index(1, 3, "p"), 0)
        self.assertEqual(navigate_image_index(0, 3, "k"), 1)
        self.assertEqual(navigate_image_index(0, 3, "p"), 0)
        self.assertEqual(navigate_image_index(2, 3, "n"), 2)
        self.assertIsNone(navigate_image_index(1, 3, "q"))

    def test_folder_coordinator_can_move_forward_and_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            output = root / "output"
            source.mkdir()
            pixels = np.zeros((4, 5, 3), dtype=np.uint8)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                self.assertTrue(cv2.imwrite(str(source / name), pixels))
            commands = iter(("n", "p", "n", "n", "q"))
            visited: list[str] = []

            def session(path, _image, _output, _defaults):
                visited.append(path.name)
                return next(commands)

            defaults = AnnotationDefaults("lecturer", "", "unknown")
            with mock.patch.object(annotation_tool, "_prompt_annotation_defaults", return_value=defaults):
                result = annotation_tool.run_folder_annotation(
                    source,
                    output,
                    "lecturer",
                    session_runner=session,
                )

            self.assertEqual(result, 0)
            self.assertEqual(visited, ["a.jpg", "b.jpg", "a.jpg", "b.jpg", "c.jpg"])

    def test_multiple_crops_stay_on_same_source_and_record_exact_filename(self):
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        source_path = Path("Lecturer Electron Frame 07.JPG")
        defaults = AnnotationDefaults("lecturer", "", "unknown")
        selector = mock.Mock()
        selector.select.side_effect = [(1, 1, 4, 3), (7, 5, 5, 4)]
        prompts = iter(("EVENT_1", "high", "first", "EVENT_2", "high", "second"))

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            annotation_tool, "ViewportROISelector", return_value=selector
        ), mock.patch.object(annotation_tool, "OrientedROISelector"), mock.patch.object(
            annotation_tool, "prompt_batch_action", side_effect=("axis_aligned", "axis_aligned", "n")
        ), mock.patch.object(
            annotation_tool, "choose_manual_label", side_effect=("electron", "electron")
        ), mock.patch.object(
            annotation_tool, "preview_before_save", side_effect=("save", "save")
        ), mock.patch.object(annotation_tool, "_prompt", side_effect=lambda *_args: next(prompts)):
            command = annotation_tool._run_image_session(
                source_path,
                image,
                Path(directory),
                defaults,
                batch_mode=True,
            )
            with (Path(directory) / "labels.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(command, "n")
        self.assertEqual([row["track_id"] for row in rows], ["E001", "E002"])
        self.assertEqual([row["source_image"] for row in rows], [source_path.name, source_path.name])

    def test_single_image_mode_still_dispatches_to_original_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "one.jpg"
            image_path.touch()
            output = Path(directory) / "output"
            with mock.patch.object(annotation_tool, "run_annotation", return_value=0) as run_single, mock.patch.object(
                annotation_tool, "run_folder_annotation"
            ) as run_folder:
                result = annotation_tool.run_interactive(image_path, output, "lecturer")

            self.assertEqual(result, 0)
            run_single.assert_called_once_with(image_path, output, "lecturer")
            run_folder.assert_not_called()

    def test_electron_folder_name_does_not_bypass_manual_class_selection(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        selector = mock.Mock()
        selector.select.return_value = (1, 1, 4, 3)
        defaults = AnnotationDefaults("lecturer", "", "unknown")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            annotation_tool, "ViewportROISelector", return_value=selector
        ), mock.patch.object(annotation_tool, "OrientedROISelector"), mock.patch.object(
            annotation_tool, "prompt_batch_action", return_value="axis_aligned"
        ), mock.patch.object(
            annotation_tool, "choose_manual_label", return_value=None
        ) as choose_label, mock.patch.object(annotation_tool, "save_annotation") as save:
            command = annotation_tool._run_image_session(
                Path(directory) / "electron" / "frame.jpg",
                image,
                Path(directory) / "output",
                defaults,
                batch_mode=True,
            )

        self.assertEqual(command, "q")
        choose_label.assert_called_once()
        save.assert_not_called()

    def test_saving_new_crop_preserves_existing_crop_and_csv_row(self):
        image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_annotation(
                image=image,
                roi=(1, 1, 4, 3),
                label="electron",
                output_root=root,
                source_dataset="lecturer",
                source_image="first.jpg",
                event_id="EVENT_FIRST",
            )
            first_crop = root / "electron" / str(first["crop_filename"])
            original_bytes = first_crop.read_bytes()
            with (root / "labels.csv").open(newline="", encoding="utf-8") as handle:
                original_row = next(csv.DictReader(handle))

            second = save_annotation(
                image=image,
                roi=(6, 5, 4, 3),
                label="electron",
                output_root=root,
                source_dataset="lecturer",
                source_image="second.jpg",
                event_id="EVENT_SECOND",
            )
            with (root / "labels.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(second["track_id"], "E002")
            self.assertEqual(first_crop.read_bytes(), original_bytes)
            self.assertEqual(rows[0], original_row)
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
