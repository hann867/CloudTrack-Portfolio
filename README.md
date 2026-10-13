# CloudTrack

CloudTrack is a full-stack application for processing and analysing cloud chamber images.

## Project Structure

This project is organised into a full-stack structure:

* `backend/` — Contains the FastAPI backend, image preprocessing algorithms (Module 1), track detection and segmentation (Module 2), calibration tools, and all data.
* `frontend/` — Contains the React interface for preprocessing, calibration, and track detection.

For specific instructions, refer to the README files within each directory:
- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)

## Image Pre-processing (Module 1)

The interactive Module 1 preprocessing pipeline is still accessible from the project root. It handles separating alpha/proton tracks from electron traces, filtering chamber frame artefacts, and reducing noise.

### Usage

Ensure the Python virtual environment is activated, then run the main menu script:

```bash
python -m backend.cli.preprocessing.main_preprocessing
```

### Options

1. **Process Images**: Run the full preprocessing pipeline to extract particle tracks.
   - Process one image (interactive preview)
   - Batch process all images in `backend/data/raw/cloud_chamber`
2. **Calibrate Imaging Setup**: Perform a manual 2-point calibration to calculate `pixels_per_mm`.
3. **Exit**

## Preprocessing Pipeline Details

The pipeline uses two independent branches to extract different particle types:
- **Electron branch**: Tight ROI, white top-hat, high-percentile threshold, small-component filter.
- **Alpha/proton branch**: Wide ROI, broad Gaussian background subtraction, moderate CLAHE, lower-percentile threshold, elongation-based component filter.

Both branches are combined, and a final frame-artefact filter removes bright physical edges while preserving particle traces.

## Track Detection and Segmentation (Module 2)

Start the backend and frontend, then open `/detection` in the UI. Upload one or
more particle-only images produced by Module 1. Module 2 generates binary,
cleaned, and labelled masks; an annotated image; individual track ROI crops;
and a downloadable track-region CSV file. Outputs mirror the standalone
assignment layout under `backend/outputs/images/module2_outputs/`:

- `binary_masks/`
- `cleaned_masks/`
- `labelled_masks/`
- `annotated_images/`
- `rois/images/` and `rois/masks/`
- `track_regions.csv`

The API endpoints are:

- `POST /api/detection/detect` for one preprocessed image.
- `POST /api/detection/batch` for multiple preprocessed images.
