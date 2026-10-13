"""Batch processing support for CloudTrack particle-only preprocessing."""

from pathlib import Path
from time import perf_counter

import cv2
import pandas as pd
from tqdm import tqdm

from .preprocessing_pipeline import run_preprocessing_pipeline


SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)

INTERMEDIATE_OUTPUT_NAMES = (
    "grayscale",
    "noise_reduced",
    "electron_enhanced",
    "electron_mask",
    "alpha_proton_enhanced",
    "alpha_proton_mask",
    "combined_particle_mask",
)

LOG_COLUMNS = (
    "input_file",
    "output_file",
    "status",
    "error_type",
    "error_message",
    "processing_seconds",
)


def _find_supported_images(input_directory: Path) -> list[Path]:
    """Return supported files in deterministic filename order."""
    return sorted(
        (
            path
            for path in input_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def _save_image(image, output_path: Path, overwrite: bool) -> None:
    """Save an OpenCV image and raise a clear error if writing fails."""
    if output_path.exists() and not overwrite:
        return
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to save processed image: {output_path}")


def batch_process_images(
    input_dir,
    output_dir,
    log_dir,
    save_intermediate=False,
    overwrite=True,
):
    """Process every supported image while isolating per-image failures.

    Existing final outputs are recorded as ``skipped`` when ``overwrite`` is
    false. A skipped image counts as successful because its requested output
    already exists and no processing error occurred.

    The final combined particle-only image is saved as::

        <original_filename>_particle_only_preprocessed.png
    """
    if not isinstance(save_intermediate, bool):
        raise TypeError("save_intermediate must be a boolean.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")

    input_directory = Path(input_dir).expanduser()
    output_directory = Path(output_dir).expanduser()

    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    image_paths = _find_supported_images(input_directory)

    records: list[dict[str, object]] = []
    successful = 0
    failed = 0

    for image_path in tqdm(
        image_paths,
        desc="Preprocessing cloud chamber images",
        unit="image",
    ):
        started_at = perf_counter()
        final_output_path = (
            output_directory
            / f"{image_path.stem}_particle_only_preprocessed.png"
        )

        if final_output_path.exists() and not overwrite:
            successful += 1
            records.append(
                {
                    "input_file": str(image_path),
                    "output_file": str(final_output_path),
                    "status": "skipped",
                    "error_type": "",
                    "error_message": "Output already exists and overwrite is disabled.",
                    "processing_seconds": round(perf_counter() - started_at, 6),
                }
            )
            continue

        try:
            pipeline_outputs = run_preprocessing_pipeline(image_path)
            if "final_image" not in pipeline_outputs:
                raise KeyError("Pipeline output is missing 'final_image'.")

            _save_image(
                pipeline_outputs["final_image"],
                final_output_path,
                overwrite=overwrite,
            )

            if save_intermediate:
                for output_name in INTERMEDIATE_OUTPUT_NAMES:
                    if output_name not in pipeline_outputs:
                        raise KeyError(
                            f"Pipeline output is missing '{output_name}'."
                        )
                    intermediate_path = (
                        output_directory
                        / f"{image_path.stem}_{output_name}.png"
                    )
                    _save_image(
                        pipeline_outputs[output_name],
                        intermediate_path,
                        overwrite=overwrite,
                    )

            successful += 1
            records.append(
                {
                    "input_file": str(image_path),
                    "output_file": str(final_output_path),
                    "status": "success",
                    "error_type": "",
                    "error_message": "",
                    "processing_seconds": round(perf_counter() - started_at, 6),
                }
            )
        except Exception as error:
            failed += 1
            records.append(
                {
                    "input_file": str(image_path),
                    "output_file": str(final_output_path),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "processing_seconds": round(perf_counter() - started_at, 6),
                }
            )

    log_directory = Path(log_dir).expanduser()
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "batch_preprocessing_log.csv"
    log_table = pd.DataFrame.from_records(records, columns=LOG_COLUMNS)
    log_table.to_csv(log_path, index=False, encoding="utf-8")

    return {
        "total_images": len(image_paths),
        "successful": successful,
        "failed": failed,
        "output_dir": output_directory,
        "log_path": log_path,
    }
