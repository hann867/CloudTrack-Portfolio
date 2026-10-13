"""Validate classified frames, finalise results, and build the 30 FPS MP4."""

import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

import cv2

from backend.src.detection_segmentation import save_png


FRAME_NAME_PATTERN = re.compile(r"^frame_(\d{6})\.png$")
OUTPUT_VIDEO_FILENAME = "classified_video.mp4"


class ClassifiedSequenceError(RuntimeError):
    """The classified presentation sequence cannot safely be encoded."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2)


def _append_error(existing: Any, message: str) -> str:
    return "; ".join(part for part in (str(existing or "").strip(), message) if part)


def finalise_classified_sequence(
    job_directory: Path,
    frame_mapping: Sequence[Mapping[str, Any]],
    results: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[Path]:
    """Return a complete ordered sequence, repairing slots from raw frames when safe."""
    expected_frames = list(frame_mapping)
    expected_names = [f"frame_{index:06d}.png" for index in range(1, len(expected_frames) + 1)]
    mapping_names = [str(item.get("filename", "")) for item in expected_frames]
    if mapping_names != expected_names:
        raise ClassifiedSequenceError("Frame mapping filenames are not contiguous and numerically ordered.")

    frame_results = results.get("frames")
    if not isinstance(frame_results, list) or len(frame_results) != len(expected_frames):
        raise ClassifiedSequenceError("Frame-level result count does not match the extracted sequence.")
    by_output_index = {int(frame["output_frame_index"]): frame for frame in frame_results}
    if sorted(by_output_index) != list(range(len(expected_frames))):
        raise ClassifiedSequenceError("Frame-level results are not contiguous and numerically ordered.")

    classified_directory = job_directory / "classified"
    classified_directory.mkdir(parents=True, exist_ok=True)
    ordered_paths: list[Path] = []
    fallback_count = 0
    resized_fallback_count = 0

    for output_index, (mapping, frame_name) in enumerate(zip(expected_frames, expected_names)):
        if int(mapping["output_frame_index"]) != output_index:
            raise ClassifiedSequenceError("Frame mapping indices are not contiguous and numerically ordered.")
        classified_path = classified_directory / frame_name
        frame = cv2.imread(str(classified_path), cv2.IMREAD_COLOR)
        issues: list[str] = []
        used_fallback = False
        resized_fallback = False

        if frame is None:
            issues.append("Classified frame was missing or unreadable.")
        elif frame.shape[1] != width or frame.shape[0] != height:
            issues.append(
                f"Classified frame resolution was {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}."
            )
            frame = None

        if frame is None:
            raw_path = job_directory / "frames" / frame_name
            fallback = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if fallback is None:
                raise ClassifiedSequenceError(
                    f"Neither classified nor extracted frame {frame_name} is readable."
                )
            if fallback.shape[1] != width or fallback.shape[0] != height:
                issues.append(
                    f"Fallback frame was resized from {fallback.shape[1]}x{fallback.shape[0]} to {width}x{height}."
                )
                fallback = cv2.resize(fallback, (width, height), interpolation=cv2.INTER_AREA)
                resized_fallback = True
                resized_fallback_count += 1
            save_png(classified_path, fallback)
            frame = fallback
            used_fallback = True
            fallback_count += 1

        record = by_output_index[output_index]
        record["classified_frame"] = f"classified/{frame_name}"
        record["used_fallback_frame"] = used_fallback
        record["resized_fallback_frame"] = resized_fallback
        record["frame_width"] = int(frame.shape[1])
        record["frame_height"] = int(frame.shape[0])
        record["sequence_validation_issues"] = issues
        if used_fallback:
            record["status"] = "failed"
            record["error"] = _append_error(record.get("error"), " ".join(issues))
        ordered_paths.append(classified_path)

    discovered_names = []
    invalid_names = []
    for path in classified_directory.glob("*.png"):
        match = FRAME_NAME_PATTERN.fullmatch(path.name)
        if match is None:
            invalid_names.append(path.name)
        else:
            discovered_names.append(path.name)
    if invalid_names:
        raise ClassifiedSequenceError(
            f"Classified directory contains invalid frame filename(s): {', '.join(sorted(invalid_names))}."
        )
    if sorted(discovered_names) != expected_names:
        raise ClassifiedSequenceError(
            f"Classified frame count mismatch: expected {len(expected_names)}, found {len(discovered_names)}."
        )

    failed_frames = sum(frame.get("status") == "failed" for frame in frame_results)
    detected_frames = sum(bool(frame.get("detection_outputs")) for frame in frame_results)
    results["failed_frames"] = failed_frames
    results["sequence_validation"] = {
        "expected_frames": len(expected_names),
        "valid_frames": len(ordered_paths),
        "fallback_frames": fallback_count,
        "resized_fallback_frames": resized_fallback_count,
        "width": width,
        "height": height,
        "status": "success",
    }
    results["processing"] = {
        "extracted_frames": len(expected_names),
        "preprocessed_frames": int(results.get("preprocessed_frames", 0)),
        "detected_frames": detected_frames,
        "classified_frames": int(results.get("classified_frames", 0)),
        "video_frames_ready": len(ordered_paths),
        "failed_frames": failed_frames,
    }
    return ordered_paths


def build_classified_video(
    job_directory: Path,
    ordered_frames: Sequence[Path],
    *,
    width: int,
    height: int,
    target_fps: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Encode each validated classified PNG exactly once, then verify the MP4."""
    output_directory = job_directory / "output"
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / OUTPUT_VIDEO_FILENAME
    output_path = output_directory / f".encoding-{uuid.uuid4().hex}.mp4"
    if not ordered_frames or width <= 0 or height <= 0:
        raise ClassifiedSequenceError("No valid classified frame sequence to encode.")
    if width % 2 or height % 2:
        raise ClassifiedSequenceError("Browser-compatible H.264 requires even frame dimensions; refusing to crop or distort the video.")
    try:
        # pyrefly: ignore [missing-import]
        import imageio_ffmpeg
    except ImportError as error:
        raise ClassifiedSequenceError(
            "Video encoding requires imageio-ffmpeg. Install backend/requirements.txt in the environment running Uvicorn."
        ) from error
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(target_fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-tag:v", "avc1", "-movflags", "+faststart", str(output_path),
    ]
    # A temporary stderr file avoids pipe deadlocks; each PNG is read only as needed.
    with tempfile.TemporaryFile() as encoder_errors:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=encoder_errors,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            _write_encoder_frames(encoder, ordered_frames, width, height, progress_callback)
            encoder.stdin.close()
            if encoder.wait(timeout=300) != 0:
                encoder_errors.seek(0)
                raise ClassifiedSequenceError(f"H.264 encoding failed: {encoder_errors.read().decode(errors='replace')[-2000:]}")
        except Exception:
            if encoder.poll() is None:
                encoder.kill()
            encoder.wait()
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            output_path.unlink(missing_ok=True)
            raise

    try:
        output = _validate_encoded_video(output_path, len(ordered_frames), width, height, target_fps)
        # Only publish a complete, verified file. Existing output survives a failed encode.
        output_path.replace(final_path)
        return output
    finally:
        output_path.unlink(missing_ok=True)


def _write_encoder_frames(encoder, ordered_frames, width, height, progress_callback):
    for expected_index, frame_path in enumerate(ordered_frames, start=1):
        match = FRAME_NAME_PATTERN.fullmatch(frame_path.name)
        if match is None or int(match.group(1)) != expected_index:
            raise ClassifiedSequenceError("Classified frames are not in numerical sequence order.")
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ClassifiedSequenceError(f"Classified frame became unreadable: {frame_path.name}")
        if frame.shape[1] != width or frame.shape[0] != height:
            raise ClassifiedSequenceError(f"Classified frame has inconsistent dimensions: {frame_path.name}")
        encoder.stdin.write(frame.tobytes())
        if progress_callback is not None:
            progress_callback(expected_index, len(ordered_frames))

def _validate_encoded_video(output_path, expected_count, width, height, target_fps):
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ClassifiedSequenceError("The MP4 encoder did not create a valid output file.")

    capture = cv2.VideoCapture(str(output_path))
    if not capture.isOpened():
        capture.release()
        raise ClassifiedSequenceError("Unable to open the generated classified video.")
    try:
        output_fps = float(capture.get(cv2.CAP_PROP_FPS))
        output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 255) for index in range(4))
        decoded_frames = 0
        while True:
            readable, frame = capture.read()
            if not readable or frame is None:
                break
            decoded_frames += 1
    finally:
        capture.release()

    if not math.isfinite(output_fps) or not math.isclose(output_fps, target_fps, abs_tol=0.05):
        raise ClassifiedSequenceError(
            f"Generated video FPS is {output_fps}, expected {target_fps}."
        )
    if output_frames != expected_count or decoded_frames != expected_count:
        raise ClassifiedSequenceError(
            f"Generated video contains {output_frames} frames ({decoded_frames} readable), expected {expected_count}."
        )
    if (output_width, output_height) != (width, height):
        raise ClassifiedSequenceError(
            f"Generated video resolution is {output_width}x{output_height}, expected {width}x{height}."
        )
    if codec.lower() not in {"avc1", "h264"}:
        raise ClassifiedSequenceError(f"Generated video uses {codec}, not browser-compatible H.264.")

    return {
        "video_filename": OUTPUT_VIDEO_FILENAME,
        "relative_path": f"output/{OUTPUT_VIDEO_FILENAME}",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "faststart": True,
        "fps": output_fps,
        "frame_count": output_frames,
        "width": output_width,
        "height": output_height,
        "duration_seconds": output_frames / output_fps,
    }


def finalise_video_job(
    job_directory: Path,
    frame_mapping: Sequence[Mapping[str, Any]],
    results: dict[str, Any],
    *,
    width: int,
    height: int,
    source_duration_seconds: float,
    target_fps: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Complete Steps 7-9 and update the existing job result in place."""
    ordered_frames = finalise_classified_sequence(
        job_directory,
        frame_mapping,
        results,
        width=width,
        height=height,
    )
    output = build_classified_video(
        job_directory,
        ordered_frames,
        width=width,
        height=height,
        target_fps=target_fps,
        progress_callback=progress_callback,
    )
    particles = results["particles"]
    results["summary"] = {
        "target_fps": target_fps,
        "source_duration_seconds": source_duration_seconds,
        "output_duration_seconds": output["duration_seconds"],
        "total_output_frames": len(ordered_frames),
        "successfully_processed_frames": len(ordered_frames) - results["failed_frames"],
        "failed_frames": results["failed_frames"],
        "unique_source_frames_processed": results["unique_source_frames_processed"],
        "total_track_detections_across_unique_analysed_frames": particles["total_tracks"],
        "alpha_track_detections": particles["alpha"],
        "electron_track_detections": particles["electron"],
        "scientific_limitation": (
            "Frames are analysed independently; track detections are not unique physical particles "
            "because temporal particle tracking has not been implemented."
        ),
    }
    results["output"] = output
    results["current_stage"] = "COMPLETED"
    results["current_frame"] = len(ordered_frames)
    _write_json(job_directory / "results.json", results)
    return results
