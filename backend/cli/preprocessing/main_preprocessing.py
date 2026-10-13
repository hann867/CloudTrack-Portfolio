"""Interactive entry point for CloudTrack Module 1."""

from pathlib import Path

import cv2

from backend.src.calibration.calibration_metadata import save_calibration_metadata
from backend.src.calibration.calibration_tool import run_manual_calibration
from backend.src.preprocessing.batch_preprocessing import batch_process_images
from backend.src.preprocessing.image_loader import SUPPORTED_IMAGE_FORMATS
from backend.src.preprocessing.preprocessing_pipeline import (
    CLEAN_PIPELINE_CONFIG,
    run_clean_particle_preprocessing_pipeline,
    run_preprocessing_pipeline,
)
from backend.src.utils.path_utils import (
    BASE_DIR,
    CALIBRATION_DIR,
    LOG_DIR,
    PREPROCESSED_DIR,
    RAW_IMAGE_DIR,
    create_required_directories,
)
from backend.src.utils.visualisation import display_images_side_by_side


def _list_available_images(directory: Path) -> list[Path]:
    """Print and return supported image files in a directory."""
    images = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ),
        key=lambda path: path.name.casefold(),
    )

    print(f"\nAvailable images in {directory}:")
    if not images:
        print("  No supported images found.")
        return []

    for index, image_path in enumerate(images, start=1):
        print(f"  {index}. {image_path.name}")

    return images


def _choose_image(images: list[Path]) -> Path:
    """Prompt until the user selects a valid numbered image."""
    while True:
        response = input(f"Choose an image [1-{len(images)}]: ").strip()
        try:
            selected_index = int(response)
        except ValueError:
            print("Invalid selection. Enter one of the listed numbers.")
            continue

        if 1 <= selected_index <= len(images):
            return images[selected_index - 1]

        print("Invalid selection. Enter one of the listed numbers.")


def _ask_yes_no(prompt: str) -> bool:
    """Prompt until the user answers yes or no."""
    while True:
        response = input(f"{prompt} [y/n]: ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Enter 'y' for yes or 'n' for no.")


def _save_preprocessed_image(source_path: Path, final_image) -> Path:
    """Save the clean particle-only output as a lossless PNG."""
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        PREPROCESSED_DIR
        / f"{source_path.stem}_particle_only_preprocessed.png"
    )
    if not cv2.imwrite(str(output_path), final_image):
        raise OSError(f"Failed to save preprocessed image: {output_path}")
    return output_path


def _choose_main_menu_option() -> str:
    """Display the application menu and return a valid selection."""
    while True:
        print("\nChoose an option:")
        print("  1. Process Images")
        print("  2. Calibrate Imaging Setup")
        print("  3. Exit")
        selection = input("Enter option [1-3]: ").strip()
        if selection in {"1", "2", "3"}:
            return selection
        print("Invalid option. Enter 1, 2, or 3.")


def _choose_process_images_menu_option() -> str:
    """Display the process images menu and return a valid selection."""
    while True:
        print("\nProcess Images:")
        print("  1. Process one image")
        print("  2. Batch process all images")
        print("  3. Back")
        selection = input("Enter option [1-3]: ").strip()
        if selection in {"1", "2", "3"}:
            return selection
        print("Invalid option. Enter 1, 2, or 3.")


def _run_single_image_flow() -> None:
    """Run the existing one-image preprocessing and calibration workflow."""
    raw_images = _list_available_images(RAW_IMAGE_DIR)
    if not raw_images:
        print(f"Add a supported image to {RAW_IMAGE_DIR} and try again.")
        return

    selected_image = _choose_image(raw_images)
    print(f"\nProcessing: {selected_image.name}")
    pipeline_outputs = run_clean_particle_preprocessing_pipeline(
        selected_image,
        config=CLEAN_PIPELINE_CONFIG,
    )

    preprocessed_path = _save_preprocessed_image(
        selected_image,
        pipeline_outputs["final_image"],
    )
    print(f"Particle-only image saved to: {preprocessed_path}")
    print(
        f"Electron threshold: "
        f"{pipeline_outputs['electron_threshold_value']:.2f}"
    )
    print(
        f"Alpha/proton threshold: "
        f"{pipeline_outputs['alpha_proton_threshold_value']:.2f}"
    )
    print(f"Electron ROI: {pipeline_outputs['electron_roi_details']}")
    print(f"Alpha/proton ROI: {pipeline_outputs['alpha_proton_roi_details']}")

    display_images_side_by_side(
        [
            pipeline_outputs["original_rgb"],
            pipeline_outputs["electron_mask"],
            pipeline_outputs["alpha_proton_mask"],
            pipeline_outputs["final_image"],
        ],
        titles=[
            "Original",
            "Electron mask",
            "Alpha/proton mask",
            "Final combined",
        ],
    )

def _run_calibration_flow() -> None:
    """Run the manual calibration setup."""
    calibration_images = _list_available_images(CALIBRATION_DIR)
    if not calibration_images:
        print(
            f"Add a supported calibration image to {CALIBRATION_DIR} "
            "and try again."
        )
        return

    selected_calibration_image = _choose_image(calibration_images)
    try:
        calibration_results = run_manual_calibration(selected_calibration_image)
    except RuntimeError as error:
        print(f"Calibration was not completed: {error}")
        return
    except Exception as error:
        print(f"An error occurred during calibration: {error}")
        return

    metadata_path = save_calibration_metadata(
        calibration_results,
        filename=f"{selected_calibration_image.stem}_calibration.json",
    )
    
    # Print metrics
    pixel_dist = calibration_results.get("pixel_distance", 0.0)
    mm_per_px = calibration_results.get("mm_per_pixel", 0.0)
    px_per_mm = calibration_results.get("pixels_per_mm", 0.0)
    
    print("\nCalibration Results:")
    print(f"  Pixel distance: {pixel_dist:.2f} px")
    print(f"  mm per pixel:   {mm_per_px:.4f} mm/px")
    print(f"  pixels per mm:  {px_per_mm:.4f} px/mm")
    print(f"Calibration metadata saved to: {metadata_path}")


def _run_batch_image_flow() -> None:
    """Process every supported raw image and print the batch summary."""
    summary = batch_process_images(
        input_dir=RAW_IMAGE_DIR,
        output_dir=PREPROCESSED_DIR,
        log_dir=LOG_DIR,
        save_intermediate=False,
        overwrite=True,
    )

    print("\nBatch preprocessing finished.")
    print(f"Total images: {summary['total_images']}")
    print(f"Successful images: {summary['successful']}")
    print(f"Failed images: {summary['failed']}")
    print(f"Output folder: {summary['output_dir']}")
    print(f"CSV log: {summary['log_path']}")


def main() -> None:
    """Run the menu-driven preprocessing and calibration application."""
    create_required_directories()
    print("CloudTrack Module 1: Image Pre-processing and Calibration")
    print(f"Project root: {BASE_DIR}")

    while True:
        selected_option = _choose_main_menu_option()
        if selected_option == "1":
            while True:
                process_option = _choose_process_images_menu_option()
                if process_option == "1":
                    _run_single_image_flow()
                elif process_option == "2":
                    _run_batch_image_flow()
                else:
                    break
        elif selected_option == "2":
            _run_calibration_flow()
        else:
            print("Exiting CloudTrack Module 1.")
            return


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nOperation cancelled.")
