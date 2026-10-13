"""Interactive, persistent review of pre-ranked ground-truth candidate matches."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .ground_truth_candidate_matcher import manual_polygon


REVIEW_RESULT_COLUMNS = [
    "ground_truth_track_id", "ground_truth_class", "source_image", "review_status",
    "selected_rank", "selected_candidate_track_id", "selected_candidate_iou",
    "selected_candidate_manual_coverage", "selected_candidate_candidate_coverage",
    "selected_candidate_centroid_distance", "selected_candidate_spatial_score",
    "review_timestamp", "notes",
]

MAPPING_COLUMNS = [
    "ground_truth_track_id", "ground_truth_class", "source_image", "match_method",
    "module2_track_id", "iou", "manual_roi_coverage", "candidate_coverage",
    "normalized_centroid_distance", "spatial_score",
]

COMPLETED_STATUSES = {"CONFIRMED", "NONE"}
ALLOWED_REVIEW_STATUSES = COMPLETED_STATUSES | {"SKIPPED"}
CANDIDATE_COLOURS = [(0, 255, 0), (0, 165, 255), (255, 0, 255)]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    with file_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_write_csv(path: str | Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def ensure_review_results(path: str | Path) -> None:
    if not Path(path).exists():
        _atomic_write_csv(path, REVIEW_RESULT_COLUMNS, [])


def load_review_results(path: str | Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    results: dict[str, dict[str, str]] = {}
    for row in rows:
        track_id = str(row.get("ground_truth_track_id", "")).strip()
        status = str(row.get("review_status", "")).strip()
        if not track_id or status not in ALLOWED_REVIEW_STATUSES:
            raise ValueError(f"Invalid manual review row for {track_id or '<blank track>'}.")
        if track_id in results:
            raise ValueError(f"Duplicate manual review result for {track_id}.")
        results[track_id] = row
    return results


def load_review_queue(
    labels_csv: str | Path,
    candidate_matching_csv: str | Path,
    review_results_csv: str | Path | None = None,
    *,
    include_completed: bool = False,
) -> list[dict[str, Any]]:
    """Return REVIEW items in dataset order with candidate rank order preserved."""

    labels = [row for row in _read_csv(labels_csv) if row.get("class") in {"Alpha", "Electron"}]
    label_index = {str(row["track_id"]): row for row in labels}
    candidate_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    review_ids: set[str] = set()
    for row in _read_csv(candidate_matching_csv):
        if row.get("confidence_status") != "REVIEW":
            continue
        track_id = str(row.get("labelled_track_id", ""))
        review_ids.add(track_id)
        if str(row.get("candidate_track_id", "")).strip():
            candidate_groups[track_id].append(row)
    completed = load_review_results(review_results_csv) if review_results_csv else {}
    queue = []
    position = 0
    for annotation in labels:
        track_id = str(annotation["track_id"])
        if track_id not in review_ids:
            continue
        position += 1
        candidates = sorted(candidate_groups[track_id], key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in candidates]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"Duplicate candidate rank for {track_id}.")
        result = completed.get(track_id)
        if not include_completed and result and result["review_status"] in COMPLETED_STATUSES:
            continue
        queue.append({
            "annotation": label_index[track_id],
            "candidates": candidates[:3],
            "review_position": position,
            "review_total": len(review_ids),
            "existing_result": result,
        })
    return queue


def record_review_decision(
    results_csv: str | Path,
    item: Mapping[str, Any],
    review_status: str,
    *,
    selected_rank: int | None = None,
    notes: str = "",
    replace_existing: bool = False,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Persist one decision; existing rows require an explicit replacement opt-in."""

    if review_status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unsupported review status: {review_status}")
    annotation = item["annotation"]
    track_id = str(annotation["track_id"])
    existing = load_review_results(results_csv)
    if track_id in existing and not replace_existing:
        raise FileExistsError(f"A review result already exists for {track_id}; explicit replacement is required.")
    candidate = None
    if review_status == "CONFIRMED":
        if selected_rank not in {1, 2, 3}:
            raise ValueError("CONFIRMED requires selected_rank 1, 2, or 3.")
        candidate = next((row for row in item["candidates"] if int(row["rank"]) == selected_rank), None)
        if candidate is None:
            raise ValueError(f"Candidate rank {selected_rank} is unavailable for {track_id}.")
    elif selected_rank is not None:
        raise ValueError(f"{review_status} must not select a candidate rank.")
    row = {column: "" for column in REVIEW_RESULT_COLUMNS}
    row.update({
        "ground_truth_track_id": track_id,
        "ground_truth_class": annotation.get("class", ""),
        "source_image": annotation.get("source_image", ""),
        "review_status": review_status,
        "review_timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    })
    if candidate:
        row.update({
            "selected_rank": selected_rank,
            "selected_candidate_track_id": candidate["candidate_track_id"],
            "selected_candidate_iou": candidate["iou"],
            "selected_candidate_manual_coverage": candidate["manual_roi_coverage"],
            "selected_candidate_candidate_coverage": candidate["candidate_bbox_coverage"],
            "selected_candidate_centroid_distance": candidate["normalized_centroid_distance"],
            "selected_candidate_spatial_score": candidate["spatial_score"],
        })
    existing[track_id] = row
    ordered = sorted(existing.values(), key=lambda value: str(value["ground_truth_track_id"]))
    _atomic_write_csv(results_csv, REVIEW_RESULT_COLUMNS, ordered)
    return row


def generate_confirmed_mapping(
    labels_csv: str | Path,
    candidate_matching_csv: str | Path,
    review_results_csv: str | Path,
    output_csv: str | Path,
) -> list[dict[str, Any]]:
    """Combine automatic HIGH and manually confirmed REVIEW matches only."""

    labels = {row["track_id"]: row for row in _read_csv(labels_csv) if row.get("class") in {"Alpha", "Electron"}}
    matches = _read_csv(candidate_matching_csv)
    by_pair = {(row["labelled_track_id"], row["candidate_track_id"]): row for row in matches
               if row.get("candidate_track_id")}
    mapping: list[dict[str, Any]] = []

    def mapped_row(track_id: str, candidate: Mapping[str, str], method: str) -> dict[str, Any]:
        annotation = labels[track_id]
        return {
            "ground_truth_track_id": track_id,
            "ground_truth_class": annotation["class"],
            "source_image": annotation["source_image"],
            "match_method": method,
            "module2_track_id": candidate["candidate_track_id"],
            "iou": candidate["iou"],
            "manual_roi_coverage": candidate["manual_roi_coverage"],
            "candidate_coverage": candidate["candidate_bbox_coverage"],
            "normalized_centroid_distance": candidate["normalized_centroid_distance"],
            "spatial_score": candidate["spatial_score"],
        }

    for candidate in matches:
        track_id = candidate.get("labelled_track_id", "")
        if candidate.get("confidence_status") == "HIGH" and candidate.get("rank") == "1":
            mapping.append(mapped_row(track_id, candidate, "AUTO_HIGH"))
    for result in load_review_results(review_results_csv).values():
        if result["review_status"] != "CONFIRMED":
            continue
        key = result["ground_truth_track_id"], result["selected_candidate_track_id"]
        candidate = by_pair.get(key)
        if candidate is None or candidate.get("confidence_status") != "REVIEW":
            raise ValueError(f"Confirmed manual selection is absent from REVIEW candidates: {key[0]} / {key[1]}")
        mapping.append(mapped_row(key[0], candidate, "MANUAL_REVIEW"))
    mapping.sort(key=lambda row: str(row["ground_truth_track_id"]))
    _atomic_write_csv(output_csv, MAPPING_COLUMNS, mapping)
    return mapping


def progress_summary(labels_csv: str | Path, candidate_matching_csv: str | Path,
                     review_results_csv: str | Path) -> dict[str, int]:
    labels = [row for row in _read_csv(labels_csv) if row.get("class") in {"Alpha", "Electron"}]
    matches = _read_csv(candidate_matching_csv)
    status_ids: dict[str, set[str]] = defaultdict(set)
    for row in matches:
        status_ids[row.get("confidence_status", "")].add(row.get("labelled_track_id", ""))
    results = load_review_results(review_results_csv)
    confirmed = sum(row["review_status"] == "CONFIRMED" for row in results.values())
    none = sum(row["review_status"] == "NONE" for row in results.values())
    review_total = len(status_ids["REVIEW"])
    return {
        "total_labelled": len(labels), "auto_high": len(status_ids["HIGH"]),
        "manual_review_queue": review_total, "confirmed_manually": confirmed, "none": none,
        "skipped_or_unresolved": review_total - confirmed - none,
        "no_match": len(status_ids["NO_MATCH"]), "total_confirmed_mapping": len(status_ids["HIGH"]) + confirmed,
    }


def exact_file_index(roots: Sequence[str | Path], wanted_names: set[str]) -> dict[str, Path]:
    matches: dict[str, list[Path]] = defaultdict(list)
    for root_value in roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in wanted_names:
                matches[path.name].append(path)
    return {name: paths[0] for name, paths in matches.items() if len(paths) == 1}


def diagnostic_overlay_index(diagnostics_root: str | Path) -> dict[tuple[str, str], Path]:
    index = {}
    for manifest in Path(diagnostics_root).rglob("manifest.csv"):
        for row in _read_csv(manifest):
            relative = str(row.get("diagnostic_overlay_file", "")).strip()
            if relative:
                index[(row.get("source_filename", ""), row.get("track_id", ""))] = manifest.parent / relative
    return index


def _read_image(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _placeholder(width: int, height: int, text: str) -> np.ndarray:
    image = np.full((height, width, 3), 35, np.uint8)
    cv2.putText(image, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return image


def _fit_panel(image: np.ndarray | None, width: int, height: int, title: str) -> np.ndarray:
    panel = np.full((height, width, 3), 24, np.uint8)
    if image is None:
        image = _placeholder(width, height - 36, "Image unavailable")
    scale = min(width / image.shape[1], (height - 36) / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x = (width - resized.shape[1]) // 2; y = 36 + (height - 36 - resized.shape[0]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(panel, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _annotated_source(item: Mapping[str, Any], source: np.ndarray | None) -> np.ndarray | None:
    if source is None:
        return None
    display = source.copy()
    manual = np.rint(manual_polygon(item["annotation"])).astype(np.int32)
    cv2.polylines(display, [manual], True, (255, 255, 0), 4, cv2.LINE_AA)
    for index, candidate in enumerate(item["candidates"]):
        colour = CANDIDATE_COLOURS[index]
        manifest = Path(candidate["manifest_file"])
        mask_path = manifest.parent / candidate["diagnostic_mask_file"]
        mask = _read_image(mask_path)
        if mask is not None:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            x0, y0 = int(float(candidate["roi_x0"])), int(float(candidate["roi_y0"]))
            x1 = min(display.shape[1], x0 + mask_gray.shape[1]); y1 = min(display.shape[0], y0 + mask_gray.shape[0])
            if x0 < x1 and y0 < y1:
                foreground = mask_gray[:y1-y0, :x1-x0] > 0
                region = display[y0:y1, x0:x1]
                tint = np.empty_like(region); tint[:] = colour
                region[foreground] = cv2.addWeighted(region[foreground], 0.55, tint[foreground], 0.45, 0)
        x, y = int(float(candidate["bbox_x"])), int(float(candidate["bbox_y"]))
        width, height = int(float(candidate["bbox_width"])), int(float(candidate["bbox_height"]))
        cv2.rectangle(display, (x, y), (x + width, y + height), colour, 3)
        cv2.putText(display, f"C{index + 1}", (x + 4, max(24, y + 25)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colour, 2, cv2.LINE_AA)
    return display


def render_review_screen(item: Mapping[str, Any], source_path: Path | None, crop_path: Path | None,
                         overlay_index: Mapping[tuple[str, str], Path]) -> np.ndarray:
    annotation = item["annotation"]
    source = _read_image(source_path); crop = _read_image(crop_path)
    canvas = np.full((1020, 1500, 3), 18, np.uint8)
    cv2.putText(canvas, f"Ground truth: {annotation['track_id']} / {annotation['class']}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Source: {annotation['source_image']}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Review: {item['review_position']} / {item['review_total']}", (1180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for index, candidate in enumerate(item["candidates"]):
        line = (f"#{index+1} {candidate['candidate_track_id']} | IoU {float(candidate['iou']):.4f} | "
                f"manual {float(candidate['manual_roi_coverage']):.4f} | candidate {float(candidate['candidate_bbox_coverage']):.4f} | "
                f"centroid/diag {float(candidate['normalized_centroid_distance']):.4f} | score {float(candidate['spatial_score']):.4f}")
        cv2.putText(canvas, line, (15, 92 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    CANDIDATE_COLOURS[index], 1, cv2.LINE_AA)
    cv2.putText(canvas, "1/2/3 select | N none | S skip | P previous | Q quit", (15, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    source_panel = _fit_panel(_annotated_source(item, source), 930, 500, "Original source: manual ROI + C1/C2/C3")
    crop_panel = _fit_panel(crop, 540, 500, "Stored ground-truth crop (display-only resize)")
    canvas[195:695, 15:945] = source_panel; canvas[195:695, 955:1495] = crop_panel
    for index in range(3):
        candidate = item["candidates"][index] if index < len(item["candidates"]) else None
        overlay = None
        title = f"Candidate {index + 1}: unavailable"
        if candidate:
            overlay = _read_image(overlay_index.get((annotation["source_image"], candidate["candidate_track_id"])))
            title = f"Candidate {index + 1}: {candidate['candidate_track_id']}"
        panel = _fit_panel(overlay, 480, 305, title)
        x0 = 15 + index * 495
        canvas[705:1010, x0:x0 + 480] = panel
    return canvas


def confirmation_screen(screen: np.ndarray, message: str) -> np.ndarray:
    display = screen.copy()
    cv2.rectangle(display, (160, 400), (1340, 620), (0, 0, 0), -1)
    cv2.rectangle(display, (160, 400), (1340, 620), (0, 200, 255), 3)
    cv2.putText(display, "Confirm:", (200, 465), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(display, message, (200, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(display, "Y confirm | N cancel | Q quit", (200, 580), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (220, 220, 220), 1)
    return display
