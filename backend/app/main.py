from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.routes.preprocessing import router as preprocessing_router
from backend.app.routes.calibration import router as calibration_router
from backend.app.routes.detection import router as detection_router
from backend.app.routes.classification import router as classification_router
from backend.app.routes.analytics import router as analytics_router
from backend.app.routes.video import router as video_router
from backend.app.routes.image_jobs import router as image_jobs_router
from backend.src.utils.path_utils import OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR

# Ensure output directory exists before mounting
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CloudTrack API",
    version="1.0.0",
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for processed images
app.mount("/api/static", StaticFiles(directory=str(OUTPUT_IMAGE_DIR)), name="static")
app.mount("/api/video/static", StaticFiles(directory=str(OUTPUT_VIDEO_DIR)), name="video-static")

# Include routers
app.include_router(preprocessing_router, prefix="/api")
app.include_router(calibration_router, prefix="/api")

# Placeholder routers for future modules
app.include_router(detection_router, prefix="/api")
app.include_router(classification_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(video_router, prefix="/api")
app.include_router(image_jobs_router, prefix="/api")

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "CloudTrack API"
    }
