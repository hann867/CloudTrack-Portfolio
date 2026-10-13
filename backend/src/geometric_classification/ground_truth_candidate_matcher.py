"""Spatial-only matching of manual annotations to Module 2 diagnostic candidates."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


SCORE_WEIGHTS = {
    "iou": 0.40,
    "manual_roi_coverage": 0.25,
    "candidate_bbox_coverage": 0.20,
    "centroid_proximity": 0.15,
}
NO_MATCH_THRESHOLDS = {"iou": 0.05, "manual_roi_coverage": 0.10, "candidate_bbox_coverage": 0.20}
HIGH_THRESHOLDS = {
    "spatial_score": 0.70,
    "iou": 0.50,
    "manual_roi_coverage": 0.60,
    "candidate_bbox_coverage": 0.60,
    "normalized_centroid_distance": 0.25,
    "dominance_margin": 0.10,
}

MATCH_COLUMNS = [
    "labelled_track_id", "source_image", "roi_type", "confidence_status", "status_reason",
    "rank", "candidate_track_id", "component_id", "label_id", "intersection_area",
    "iou", "manual_roi_coverage", "candidate_bbox_coverage", "centroid_distance_px",
    "normalized_centroid_distance", "spatial_score", "dominance_margin", "manual_area",
    "candidate_bbox_area", "manual_centroid_x", "manual_centroid_y", "candidate_centroid_x",
    "candidate_centroid_y", "candidate_centroid_source", "bbox_x", "bbox_y", "bbox_width",
    "bbox_height", "roi_x0", "roi_y0", "roi_width", "roi_height", "diagnostic_mask_file",
    "manifest_file",
]


def _number(record: Mapping[str, Any], key: str) -> float:
    value = str(record.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing required annotation value: {key}")
    return float(value)


def manual_polygon(record: Mapping[str, Any]) -> np.ndarray:
    """Return the exact manual annotation quadrilateral in source-image coordinates."""

    roi_type = str(record.get("roi_type", "")).strip().lower()
    if roi_type == "axis_aligned":
        x, y = _number(record, "x"), _number(record, "y")
        width, height = _number(record, "width"), _number(record, "height")
        points = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    elif roi_type == "rotated":
        points = [[_number(record, f"corner_{index}_x"), _number(record, f"corner_{index}_y")]
                  for index in range(1, 5)]
    else:
        raise ValueError(f"Unsupported roi_type {record.get('roi_type')!r}")
    polygon = np.asarray(points, dtype=np.float32)
    if abs(cv2.contourArea(polygon)) <= 0:
        raise ValueError("Manual annotation polygon has zero area.")
    return polygon


def _centroid(polygon: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(polygon)
    if moments["m00"] == 0:
        return tuple(np.mean(polygon, axis=0))  # type: ignore[return-value]
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _bbox_polygon(candidate: Mapping[str, Any]) -> np.ndarray:
    x, y = _number(candidate, "bbox_x"), _number(candidate, "bbox_y")
    width, height = _number(candidate, "bbox_width"), _number(candidate, "bbox_height")
    return np.asarray([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32)


def _read_mask(candidate: Mapping[str, Any]) -> np.ndarray | None:
    path = Path(str(candidate.get("_resolved_mask_file", "")))
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def _candidate_centroid(candidate: Mapping[str, Any]) -> tuple[float, float, str]:
    mask = _read_mask(candidate)
    if mask is not None:
        moments = cv2.moments((mask > 0).astype(np.uint8))
        if moments["m00"] > 0:
            return (
                _number(candidate, "roi_x0") + moments["m10"] / moments["m00"],
                _number(candidate, "roi_y0") + moments["m01"] / moments["m00"],
                "mask_moments",
            )
    return (
        _number(candidate, "bbox_x") + _number(candidate, "bbox_width") / 2,
        _number(candidate, "bbox_y") + _number(candidate, "bbox_height") / 2,
        "bbox_center_fallback",
    )


def spatial_metrics(manual: np.ndarray, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate matching metrics without reading class or geometric/RF features."""

    candidate_polygon = _bbox_polygon(candidate)
    manual_area = abs(float(cv2.contourArea(manual)))
    candidate_area = abs(float(cv2.contourArea(candidate_polygon)))
    intersection_area, _ = cv2.intersectConvexConvex(manual, candidate_polygon)
    intersection_area = max(0.0, float(intersection_area))
    union = manual_area + candidate_area - intersection_area
    iou = intersection_area / union if union else 0.0
    manual_coverage = intersection_area / manual_area if manual_area else 0.0
    candidate_coverage = intersection_area / candidate_area if candidate_area else 0.0
    manual_cx, manual_cy = _centroid(manual)
    candidate_cx, candidate_cy, centroid_source = _candidate_centroid(candidate)
    distance = math.hypot(candidate_cx - manual_cx, candidate_cy - manual_cy)
    min_xy, max_xy = manual.min(axis=0), manual.max(axis=0)
    diagonal = math.hypot(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]))
    normalized_distance = distance / diagonal if diagonal else math.inf
    proximity = max(0.0, 1.0 - normalized_distance)
    score = (
        SCORE_WEIGHTS["iou"] * iou
        + SCORE_WEIGHTS["manual_roi_coverage"] * manual_coverage
        + SCORE_WEIGHTS["candidate_bbox_coverage"] * candidate_coverage
        + SCORE_WEIGHTS["centroid_proximity"] * proximity
    )
    return {
        "intersection_area": intersection_area,
        "iou": iou,
        "manual_roi_coverage": manual_coverage,
        "candidate_bbox_coverage": candidate_coverage,
        "centroid_distance_px": distance,
        "normalized_centroid_distance": normalized_distance,
        "spatial_score": score,
        "manual_area": manual_area,
        "candidate_bbox_area": candidate_area,
        "manual_centroid_x": manual_cx,
        "manual_centroid_y": manual_cy,
        "candidate_centroid_x": candidate_cx,
        "candidate_centroid_y": candidate_cy,
        "candidate_centroid_source": centroid_source,
    }


def rank_candidates(annotation: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    manual = manual_polygon(annotation)
    ranked = []
    for candidate in candidates:
        row = dict(candidate)
        row.update(spatial_metrics(manual, candidate))
        ranked.append(row)
    ranked.sort(key=lambda row: (
        -row["spatial_score"], -row["iou"], -row["manual_roi_coverage"],
        -row["candidate_bbox_coverage"], str(row.get("track_id", "")),
    ))
    return ranked


def confidence_status(ranked: list[Mapping[str, Any]]) -> tuple[str, str, float]:
    if not ranked:
        return "NO_MATCH", "No Module 2 diagnostics exist for the exact source_image.", 0.0
    top = ranked[0]
    margin = float(top["spatial_score"]) - (float(ranked[1]["spatial_score"]) if len(ranked) > 1 else 0.0)
    if float(top["intersection_area"]) <= 0 or (
        float(top["iou"]) < NO_MATCH_THRESHOLDS["iou"]
        and float(top["manual_roi_coverage"]) < NO_MATCH_THRESHOLDS["manual_roi_coverage"]
        and float(top["candidate_bbox_coverage"]) < NO_MATCH_THRESHOLDS["candidate_bbox_coverage"]
    ):
        return "NO_MATCH", "No candidate meets the reported minimum spatial-overlap rule.", margin
    high = (
        float(top["spatial_score"]) >= HIGH_THRESHOLDS["spatial_score"]
        and float(top["iou"]) >= HIGH_THRESHOLDS["iou"]
        and float(top["manual_roi_coverage"]) >= HIGH_THRESHOLDS["manual_roi_coverage"]
        and float(top["candidate_bbox_coverage"]) >= HIGH_THRESHOLDS["candidate_bbox_coverage"]
        and float(top["normalized_centroid_distance"]) <= HIGH_THRESHOLDS["normalized_centroid_distance"]
        and margin >= HIGH_THRESHOLDS["dominance_margin"]
    )
    if high:
        return "HIGH", "Top candidate meets every reported overlap, proximity, and dominance threshold.", margin
    return "REVIEW", "Spatial overlap exists, but the top candidate does not meet every HIGH threshold.", margin


def load_diagnostic_candidates(diagnostics_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for manifest in sorted(Path(diagnostics_root).rglob("manifest.csv"), key=lambda path: path.as_posix().casefold()):
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                source = str(row.get("source_filename", "")).strip()
                track_id = str(row.get("track_id", "")).strip()
                if not source or not track_id:
                    continue
                key = source, track_id
                if key in seen:
                    raise ValueError(f"Duplicate diagnostic candidate for exact source: {source} / {track_id}")
                seen.add(key)
                row["manifest_file"] = str(manifest.resolve())
                mask = str(row.get("diagnostic_mask_file", "")).strip()
                row["_resolved_mask_file"] = str((manifest.parent / mask).resolve()) if mask else ""
                by_source[source].append(row)
    return by_source


def _find_source_images(source_root: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = defaultdict(list)
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            found[path.name].append(path)
    return {name: paths[0] for name, paths in found.items() if len(paths) == 1}


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(str(path))


def _visual(annotation: Mapping[str, Any], candidate: Mapping[str, Any], source: Path, output: Path) -> None:
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not decode source image: {source}")
    manual = np.rint(manual_polygon(annotation)).astype(np.int32)
    overlay = image.copy()
    mask = _read_mask(candidate)
    if mask is not None:
        x0, y0 = int(float(candidate["roi_x0"])), int(float(candidate["roi_y0"]))
        y1, x1 = min(image.shape[0], y0 + mask.shape[0]), min(image.shape[1], x0 + mask.shape[1])
        if x0 < x1 and y0 < y1:
            local = mask[: y1 - y0, : x1 - x0] > 0
            colour = np.zeros_like(overlay[y0:y1, x0:x1]); colour[:] = (0, 165, 255)
            region = overlay[y0:y1, x0:x1]
            region[local] = cv2.addWeighted(region[local], 0.55, colour[local], 0.45, 0)
    bx, by = int(float(candidate["bbox_x"])), int(float(candidate["bbox_y"]))
    bw, bh = int(float(candidate["bbox_width"])), int(float(candidate["bbox_height"]))
    cv2.polylines(overlay, [manual], True, (255, 255, 0), 4, cv2.LINE_AA)
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 4)
    text = (f"GT {annotation['track_id']} | M2 {candidate['track_id']} | "
            f"IoU {candidate['iou']:.4f} | manual cov {candidate['manual_roi_coverage']:.4f} | "
            f"bbox cov {candidate['candidate_bbox_coverage']:.4f} | centroid/diag {candidate['normalized_centroid_distance']:.4f}")
    scale = max(0.55, min(1.1, image.shape[1] / 1800))
    cv2.rectangle(overlay, (0, 0), (image.shape[1], int(48 * scale + 18)), (0, 0, 0), -1)
    cv2.putText(overlay, text, (12, int(38 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    _write_png(output, overlay)


def _review_montage(annotation: Mapping[str, Any], candidates: list[Mapping[str, Any]], source: Path, output: Path) -> None:
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not decode source image: {source}")
    manual = manual_polygon(annotation)
    x0, y0 = np.floor(manual.min(axis=0)).astype(int)
    x1, y1 = np.ceil(manual.max(axis=0)).astype(int)
    pad = max(30, int(0.15 * max(x1 - x0, y1 - y0)))
    panels = []
    for rank, candidate in enumerate(candidates, 1):
        bx, by = int(float(candidate["bbox_x"])), int(float(candidate["bbox_y"]))
        bw, bh = int(float(candidate["bbox_width"])), int(float(candidate["bbox_height"]))
        left, top = max(0, min(x0, bx) - pad), max(0, min(y0, by) - pad)
        right, bottom = min(image.shape[1], max(x1, bx + bw) + pad), min(image.shape[0], max(y1, by + bh) + pad)
        panel = image[top:bottom, left:right].copy()
        shifted = np.rint(manual - [left, top]).astype(np.int32)
        cv2.polylines(panel, [shifted], True, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.rectangle(panel, (bx - left, by - top), (bx + bw - left, by + bh - top), (0, 255, 0), 3)
        title = f"#{rank} {candidate['track_id']} IoU={candidate['iou']:.3f} score={candidate['spatial_score']:.3f}"
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, title, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    if not panels:
        return
    height, width = 360, 520
    resized = [cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA) for panel in panels]
    blank = np.zeros_like(resized[0])
    while len(resized) < 6:
        resized.append(blank.copy())
    montage = np.vstack([np.hstack(resized[:3]), np.hstack(resized[3:6])])
    _write_png(output, montage)


def _output_row(annotation: Mapping[str, Any], candidate: Mapping[str, Any] | None, status: str,
                reason: str, rank: int | str, margin: float) -> dict[str, Any]:
    row = {column: "" for column in MATCH_COLUMNS}
    row.update({
        "labelled_track_id": annotation.get("track_id", ""), "source_image": annotation.get("source_image", ""),
        "roi_type": annotation.get("roi_type", ""), "confidence_status": status,
        "status_reason": reason, "rank": rank, "dominance_margin": margin,
    })
    if candidate:
        row["candidate_track_id"] = candidate.get("track_id", "")
        for key in MATCH_COLUMNS:
            if key in candidate:
                row[key] = candidate[key]
    return row


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _write_method(path: Path) -> None:
    path.write_text(
        "Spatial-only ground-truth-to-Module-2 candidate matching\n"
        "Candidate eligibility: diagnostic manifest source_filename must exactly equal labels.csv source_image.\n"
        "Candidate geometry: Module 2 bbox; candidate centroid: mask moments in source coordinates, bbox center fallback.\n"
        "Manual geometry: axis-aligned x/y/width/height or exact rotated corner_1..corner_4 polygon.\n"
        "Manual ROI diagonal: diagonal of the manual polygon's axis-aligned extent.\n"
        "Spatial score = 0.40*IoU + 0.25*manual ROI coverage + 0.20*candidate bbox coverage "
        "+ 0.15*max(0, 1-normalized centroid distance).\n"
        "NO_MATCH: no exact-source candidates, zero top intersection, or all of top IoU<0.05, manual coverage<0.10, bbox coverage<0.20.\n"
        "HIGH: score>=0.70, IoU>=0.50, manual coverage>=0.60, bbox coverage>=0.60, "
        "normalized centroid distance<=0.25, and score lead over second candidate>=0.10.\n"
        "REVIEW: reasonable overlap exists but not every HIGH condition is met.\n"
        "Class, geometric features, and Random Forest outputs are neither loaded nor scored.\n",
        encoding="utf-8",
    )


def match_dataset(labels_csv: str | Path, diagnostics_root: str | Path, source_root: str | Path,
                  output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    visuals = output / "visual_confirmations"; visuals.mkdir(exist_ok=True)
    with Path(labels_csv).open(encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    candidates_by_source = load_diagnostic_candidates(diagnostics_root)
    source_images = _find_source_images(Path(source_root))
    all_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    no_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        source_name = str(annotation.get("source_image", ""))
        ranked = rank_candidates(annotation, candidates_by_source.get(source_name, []))
        status, reason, margin = confidence_status(ranked)
        top_five = ranked[:5]
        rows = [_output_row(annotation, candidate, status, reason, rank, margin)
                for rank, candidate in enumerate(top_five, 1)]
        if not rows:
            rows = [_output_row(annotation, None, status, reason, "", margin)]
        all_rows.extend(rows)
        if status == "HIGH":
            high_rows.append(rows[0])
        elif status == "REVIEW":
            review_rows.extend(rows)
        else:
            no_rows.append(rows[0])
        summaries[str(annotation.get("track_id"))] = rows[0]
        source_path = source_images.get(source_name)
        if source_path and top_five:
            safe_id = str(annotation.get("track_id", "unknown")).replace("/", "_").replace("\\", "_")
            if status == "HIGH":
                _visual(annotation, top_five[0], source_path, visuals / f"{safe_id}_confirmation.png")
            elif status == "REVIEW":
                _review_montage(annotation, top_five, source_path, visuals / f"{safe_id}_review_montage.png")
    _write_csv(output / "candidate_matching.csv", all_rows)
    _write_csv(output / "high_confidence_matches.csv", high_rows)
    _write_csv(output / "review_matches.csv", review_rows)
    _write_csv(output / "no_matches.csv", no_rows)
    _write_method(output / "matching_method.txt")
    counts = {status: sum(row["confidence_status"] == status for row in summaries.values())
              for status in ("HIGH", "REVIEW", "NO_MATCH")}
    return {"total_labelled_tracks": len(annotations), "high_confidence_matches": counts["HIGH"],
            "review_required": counts["REVIEW"], "no_match": counts["NO_MATCH"],
            "output_dir": str(output.resolve()), "track_summaries": summaries}
