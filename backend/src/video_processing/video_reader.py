"""Open and validate uploaded videos and read their source metadata."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import cv2


SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}


class VideoValidationError(ValueError):
    """An uploaded file is not a usable video."""


@dataclass(frozen=True)
class VideoMetadata:
    filename: str
    original_fps: float
    original_frames: int
    width: int
    height: int
    duration_seconds: float
    target_fps: int
    expected_extracted_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


def read_video_metadata(video_path: Path, original_filename: str, target_fps: int) -> VideoMetadata:
    """Validate a video with OpenCV and return timeline metadata."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError("Unable to open uploaded video.")

    try:
        original_fps = float(capture.get(cv2.CAP_PROP_FPS))
        original_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        readable, frame = capture.read()
        if not readable or frame is None:
            raise VideoValidationError("Video contains no readable frames.")
        if not math.isfinite(original_fps) or original_fps <= 0:
            raise VideoValidationError("Unable to determine the uploaded video's frame rate.")

        # Fall back to a sequential count for containers/codecs that expose a
        # valid stream but do not populate CAP_PROP_FRAME_COUNT.
        if original_frames <= 0:
            original_frames = 1
            while True:
                readable, next_frame = capture.read()
                if not readable or next_frame is None:
                    break
                original_frames += 1

        # Some containers omit dimensions even though the first frame is valid.
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]

        duration_seconds = original_frames / original_fps
        # Target timestamps are n / target_fps for all n where t is before the
        # end of the source timeline. The epsilon only suppresses binary-float
        # noise when the mathematical result is already an integer.
        expected_frames = max(1, math.ceil(duration_seconds * target_fps - 1e-9))
        return VideoMetadata(
            filename=original_filename,
            original_fps=original_fps,
            original_frames=original_frames,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            target_fps=target_fps,
            expected_extracted_frames=expected_frames,
        )
    finally:
        capture.release()
