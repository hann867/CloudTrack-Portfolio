import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from backend.src.video_processing.frame_extractor import TARGET_FPS, build_frame_plan, extract_frames
from backend.src.video_processing.video_reader import VideoMetadata, VideoValidationError, read_video_metadata


def metadata(fps: float, frame_count: int) -> VideoMetadata:
    duration = frame_count / fps
    return VideoMetadata(
        filename="test.avi",
        original_fps=fps,
        original_frames=frame_count,
        width=64,
        height=48,
        duration_seconds=duration,
        target_fps=TARGET_FPS,
        expected_extracted_frames=max(1, int(np.ceil(duration * TARGET_FPS - 1e-9))),
    )


class FrameExtractionTests(unittest.TestCase):
    def test_frame_plan_uses_target_timestamps(self):
        for fps, source_frames, expected_frames in (
            (30.0, 150, 150),
            (60.0, 300, 150),
            (24.0, 120, 150),
            (59.94, 300, 151),
        ):
            with self.subTest(fps=fps):
                plan = build_frame_plan(metadata(fps, source_frames))
                self.assertEqual(len(plan), expected_frames)
                self.assertEqual(
                    plan[0],
                    {
                        "output_frame_index": 0,
                        "source_frame_index": 0,
                        "timestamp_seconds": 0.0,
                        "filename": "frame_000001.png",
                    },
                )
                self.assertTrue(
                    all(item["timestamp_seconds"] == index / 30 for index, item in enumerate(plan))
                )
                self.assertEqual(plan[-1]["filename"], f"frame_{expected_frames:06d}.png")

    def test_below_30_fps_plan_records_duplicate_source_frames(self):
        plan = build_frame_plan(metadata(24.0, 120))
        source_indices = [item["source_frame_index"] for item in plan]

        self.assertEqual(len(plan), 150)
        self.assertEqual(len(set(source_indices)), 120)
        self.assertTrue(any(left == right for left, right in zip(source_indices, source_indices[1:])))

    def test_extracts_pngs_and_writes_frame_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "source.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for index in range(24):
                writer.write(np.full((48, 64, 3), index * 8, dtype=np.uint8))
            writer.release()

            video_metadata = read_video_metadata(video_path, "source.avi", TARGET_FPS)
            frames_directory = root / "job" / "frames"
            mapping = extract_frames(video_path, frames_directory, video_metadata)

            self.assertEqual(len(mapping), 30)
            self.assertEqual(len(list(frames_directory.glob("frame_*.png"))), 30)
            self.assertTrue((frames_directory / "frame_000001.png").is_file())
            self.assertTrue((frames_directory / "frame_000030.png").is_file())
            self.assertEqual(
                json.loads((frames_directory.parent / "frames.json").read_text()), mapping
            )

    def test_non_video_file_has_clear_validation_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "invalid.mp4"
            invalid_path.write_text("not a video", encoding="utf-8")

            with self.assertRaisesRegex(VideoValidationError, "Unable to open uploaded video"):
                read_video_metadata(invalid_path, invalid_path.name, TARGET_FPS)


if __name__ == "__main__":
    unittest.main()
