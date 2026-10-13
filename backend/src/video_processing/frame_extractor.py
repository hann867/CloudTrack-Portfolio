"""Convert a constant-frame-rate source timeline to a 30 FPS PNG sequence."""

import json
import math
from pathlib import Path
from typing import Callable

import cv2

from .video_reader import VideoMetadata, VideoValidationError


TARGET_FPS = 30


def build_frame_plan(metadata: VideoMetadata) -> list[dict]:
    """Build a stable, timestamp-based nearest-source-frame mapping."""
    plan = []
    for output_index in range(metadata.expected_extracted_frames):
        timestamp = output_index / metadata.target_fps
        source_index = min(
            metadata.original_frames - 1,
            math.floor(timestamp * metadata.original_fps + 0.5),
        )
        plan.append(
            {
                "output_frame_index": output_index,
                "source_frame_index": source_index,
                "timestamp_seconds": timestamp,
                "filename": f"frame_{output_index + 1:06d}.png",
            }
        )
    return plan


def extract_frames(
    video_path: Path,
    frames_directory: Path,
    metadata: VideoMetadata,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Save the planned source frames as PNGs and return their mapping."""
    plan = build_frame_plan(metadata)
    frames_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError("Unable to reopen uploaded video for frame extraction.")

    current_source_index = -1
    current_frame = None
    try:
        for completed, item in enumerate(plan, start=1):
            wanted_source_index = item["source_frame_index"]
            while current_source_index < wanted_source_index:
                readable, current_frame = capture.read()
                current_source_index += 1
                if not readable or current_frame is None:
                    raise VideoValidationError(
                        f"Video frame {wanted_source_index} could not be read during extraction."
                    )

            output_path = frames_directory / item["filename"]
            if current_frame is None or not cv2.imwrite(str(output_path), current_frame):
                raise OSError(f"Failed to save extracted frame '{item['filename']}'.")
            if progress_callback is not None:
                progress_callback(completed, len(plan))
    finally:
        capture.release()

    frames_json = frames_directory.parent / "frames.json"
    with frames_json.open("w", encoding="utf-8") as file_handle:
        json.dump(plan, file_handle, indent=2)
    return plan
