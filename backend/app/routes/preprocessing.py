import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.src.preprocessing.image_loader import SUPPORTED_IMAGE_FORMATS
from backend.src.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline
from backend.src.utils.path_utils import OUTPUT_IMAGE_DIR

router = APIRouter(
    prefix="/preprocessing",
    tags=["preprocessing"]
)

@router.post("/image")
async def process_image(file: UploadFile = File(...)):
    # 1. Validate file format
    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_IMAGE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{extension}'. Supported: {supported}"
        )

    # 2. Save upload temporarily
    try:
        fd, temp_path = tempfile.mkstemp(suffix=extension)
        os.close(fd)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    try:
        # 3. Call existing pipeline
        try:
            results = run_preprocessing_pipeline(Path(temp_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

        final_image = results.get("final_image")
        if final_image is None:
            raise HTTPException(status_code=500, detail="Pipeline did not return final_image.")

        # 4. Save processed result
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = Path(file.filename).stem
        output_filename = f"processed_{safe_filename}_{unique_id}.png"
        output_path = OUTPUT_IMAGE_DIR / output_filename

        OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), final_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save processed image.")

        # Return response
        return {
            "original_filename": file.filename,
            "status": "success",
            "processed_image_url": f"http://127.0.0.1:8000/api/static/{output_filename}"
        }
    finally:
        # 5. Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/batch")
async def process_batch(files: List[UploadFile] = File(...)):
    results = []
    successful_count = 0
    failed_count = 0

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_result = {
            "original_filename": file.filename,
            "status": "error",
            "processed_image_url": None,
            "error_message": None
        }

        extension = Path(file.filename).suffix.lower()
        if extension not in SUPPORTED_IMAGE_FORMATS:
            file_result["error_message"] = f"Unsupported format '{extension}'"
            failed_count += 1
            results.append(file_result)
            continue

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=extension)
            os.close(fd)
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            pipeline_results = run_preprocessing_pipeline(Path(temp_path))
            final_image = pipeline_results.get("final_image")
            
            if final_image is None:
                raise ValueError("Pipeline did not return final_image.")

            unique_id = uuid.uuid4().hex[:8]
            safe_filename = Path(file.filename).stem
            output_filename = f"processed_{safe_filename}_{unique_id}.png"
            output_path = OUTPUT_IMAGE_DIR / output_filename

            success = cv2.imwrite(str(output_path), final_image)
            if not success:
                raise ValueError("Failed to save processed image.")

            file_result["status"] = "success"
            file_result["processed_image_url"] = f"http://127.0.0.1:8000/api/static/{output_filename}"
            successful_count += 1

        except Exception as e:
            file_result["error_message"] = str(e)
            failed_count += 1
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        results.append(file_result)

    return {
        "total_images": len(files),
        "successful_count": successful_count,
        "failed_count": failed_count,
        "results": results
    }
