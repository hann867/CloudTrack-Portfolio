"""Path definitions and directory setup for CloudTrack Module 1."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_IMAGE_DIR = BASE_DIR / "data" / "raw" / "cloud_chamber"
CALIBRATION_DIR = BASE_DIR / "data" / "calibration"
PREPROCESSED_DIR = BASE_DIR / "data" / "processed" / "preprocessed"
CALIBRATED_DIR = BASE_DIR / "data" / "processed" / "calibrated"
OUTPUT_IMAGE_DIR = BASE_DIR / "outputs" / "images"
OUTPUT_VIDEO_DIR = BASE_DIR / "outputs" / "videos"
MODULE2_OUTPUT_DIR = OUTPUT_IMAGE_DIR / "module2_outputs"
LOG_DIR = BASE_DIR / "outputs" / "logs"
_REQUIRED_DIRECTORIES = (
    RAW_IMAGE_DIR,
    CALIBRATION_DIR,
    PREPROCESSED_DIR,
    CALIBRATED_DIR,
    OUTPUT_IMAGE_DIR,
    OUTPUT_VIDEO_DIR,
    MODULE2_OUTPUT_DIR,
    LOG_DIR,
)


def create_required_directories() -> None:
    """Create all directories required by CloudTrack Module 1."""
    for directory in _REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
