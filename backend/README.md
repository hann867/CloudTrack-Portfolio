# CloudTrack Backend

This directory contains the Python backend for CloudTrack, built with FastAPI, together with the image-processing modules used by the application.

## Prerequisites

Before starting the backend, make sure the following are installed on your computer:

- Python
- `pip` (included with most Python installations)
- PowerShell or another Windows terminal

The commands below assume the project is located at:

```text
D:\cloudchamber
```

## Setup & Startup Instructions

Open a PowerShell terminal and run the following commands from the project root.

### 1. Go to the Project Folder

```powershell
cd D:\cloudchamber
```

### 2. Create a Virtual Environment

Create the project virtual environment in `.venv`:

```powershell
py -m venv .venv
```

> You only need to create the virtual environment the first time. If `.venv` already exists, you can skip this step.

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, your terminal should normally show `(.venv)` at the beginning of the command prompt.

### 4. Install Backend Dependencies

Install the required Python packages from `backend\requirements.txt`:

```powershell
python -m pip install -r backend\requirements.txt
```

Run this command again whenever the backend requirements are updated.

Video output uses the FFmpeg executable supplied by `imageio-ffmpeg` to encode
H.264 / yuv420p fast-start MP4. Video PDF reports use `reportlab`. Both are included
in `backend/requirements.txt`; no separate FFmpeg or PDF installation is needed.
Always install with the same virtual environment used to start Uvicorn. In this
document that is `D:\cloudchamber\.venv`, not `backend\.venv`.
To avoid accidentally launching global Python, the equivalent explicit startup
command from the repository root is:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Processing Duration is measured by a monotonic backend clock from receipt of the
processing request (before multipart parsing) until the encoded video has been
validated and published. It includes upload parsing and queue time, but not the
network delay before the request reaches the backend. Older jobs without timing
data display "Not recorded" rather than an estimated duration.

### 5. Start the FastAPI Backend

Run the backend from the repository root because the application imports modules using the `backend.*` package namespace:

```powershell
python -m uvicorn backend.app.main:app --reload
```

The development server will normally be available at:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

Keep this terminal open while using the CloudTrack frontend.

## Quick Start

For a new setup, the complete backend command sequence is:

```powershell
cd D:\cloudchamber
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --reload
```

For later runs, if the virtual environment and dependencies are already installed, you normally only need:

```powershell
cd D:\cloudchamber
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --reload
```

## Test the API

You can test the backend health endpoint by opening:

- `http://127.0.0.1:8000/api/health`

Or, in another terminal, run:

```powershell
curl http://127.0.0.1:8000/api/health
```

A successful response should look like:

```json
{"status": "ok", "service": "CloudTrack API"}
```

## Module 1: Pre-processing and Calibration

The application includes the image pre-processing and calibration workflow used before later image-analysis stages.

If you need to run the standalone interactive pre-processing CLI directly, run it from the repository root:

```powershell
cd D:\cloudchamber
python -m backend.cli.preprocessing.main_preprocessing
```

## Module 2: Detection and Segmentation

Preprocessed PNG or JPEG images can be submitted through:

- `POST /api/detection/detect`
- `POST /api/detection/batch`

Generated files use the same layout as the standalone assignment under:

```text
backend/outputs/images/module2_outputs/
```

This includes separate mask and annotation folders, `rois/images/`, `rois/masks/`, and a cumulative `track_regions.csv`.

Generated files are served through:

```text
/api/static/
```

## Stopping the Backend

To stop the FastAPI development server, return to the terminal running Uvicorn and press:

```text
Ctrl + C
```

To leave the Python virtual environment afterward, run:

```powershell
deactivate
```

## Running Backend and Frontend Together

CloudTrack requires the backend and frontend development servers to run at the same time during development.

Use two terminals:

1. **Backend terminal** — run the FastAPI server from `D:\cloudchamber`.
2. **Frontend terminal** — run the Vite development server from `D:\cloudchamber\frontend`.

See `frontendREADME.md` for the frontend setup instructions.
