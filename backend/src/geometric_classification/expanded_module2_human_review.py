"""Resumable human adjudication of expanded production Module 2 mappings."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .ground_truth_candidate_matcher import manual_polygon


MAPPING_DECISIONS = {"CONFIRMED", "REJECTED", "UNCERTAIN", "UNREVIEWED"}
MASK_DECISIONS = {"PASS", "FAIL", "UNCERTAIN", "UNREVIEWED"}
COMPLETED_MAPPING = {"CONFIRMED", "REJECTED", "UNCERTAIN"}
COMPLETED_MASK = {"PASS", "FAIL", "UNCERTAIN"}
HISTORY_COLUMNS = [
    "track_id", "ground_truth_class", "source", "source_dataset", "source_video",
    "frame_number", "timestamp_seconds", "module2_event_id", "module2_component_id",
    "selected_candidate", "mapping_decision", "mask_quality_decision", "reviewed_at",
    "review_method", "notes", "auto_accepted_existing",
]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    with file_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _index(rows: Sequence[Mapping[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw in rows:
        value = str(raw.get(key, "")).strip()
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key}: {value or '<blank>'}")
        result[value] = dict(raw)
    return result


def load_history(path: str | Path) -> dict[str, dict[str, str]]:
    history = _index(read_csv(path), "track_id") if Path(path).is_file() else {}
    for track_id, row in history.items():
        if row.get("mapping_decision") not in MAPPING_DECISIONS:
            raise ValueError(f"Invalid mapping decision for {track_id}.")
        if row.get("mask_quality_decision") not in MASK_DECISIONS:
            raise ValueError(f"Invalid mask-quality decision for {track_id}.")
        if row["mapping_decision"] != "CONFIRMED" and row["mask_quality_decision"] != "UNREVIEWED":
            raise ValueError(f"Mask decision is invalid without confirmed identity for {track_id}.")
    return history


def initialize_history(
    manifest_csv: str | Path,
    mapping_csv: str | Path,
    quality_csv: str | Path,
    history_csv: str | Path,
) -> list[dict[str, str]]:
    """Create/reconcile exactly one isolated history row per independent track."""

    manifests = read_csv(manifest_csv)
    mappings = _index(read_csv(mapping_csv), "ground_truth_track_id")
    qualities = _index(read_csv(quality_csv), "ground_truth_track_id")
    track_ids = [row["ground_truth_track_id"] for row in manifests]
    if len(track_ids) != 114 or len(set(track_ids)) != 114:
        raise ValueError("Expected exactly 114 unique independent ground-truth tracks.")
    if set(track_ids) != set(mappings) or set(track_ids) != set(qualities):
        raise ValueError("Manifest, mapping, and mask-quality track IDs do not align.")
    if {"A018", "A027"} & set(track_ids):
        raise ValueError("Excluded physical-track duplicates were reintroduced.")

    existing = load_history(history_csv)
    unknown = set(existing) - set(track_ids)
    if unknown:
        raise ValueError(f"History contains non-independent track IDs: {sorted(unknown)}")
    output = []
    for manifest in manifests:
        track_id = manifest["ground_truth_track_id"]
        old = existing.get(track_id)
        if old:
            if old["ground_truth_class"] != manifest["ground_truth_class"] or old["source"] != manifest["source_image"]:
                raise ValueError(f"Protected history provenance changed for {track_id}.")
            output.append(old)
            continue
        mapping = mappings[track_id]
        quality = qualities[track_id]
        auto = mapping["mapping_status"] == "CONFIRMED_EXISTING" and quality["mask_quality_status"] == "PASS"
        output.append({
            "track_id": track_id,
            "ground_truth_class": manifest["ground_truth_class"],
            "source": manifest["source_image"],
            "source_dataset": manifest["source_dataset"],
            "source_video": manifest.get("source_video", ""),
            "frame_number": manifest.get("frame_number", ""),
            "timestamp_seconds": manifest.get("timestamp_seconds", ""),
            "module2_event_id": mapping["candidate_track_id"] if auto else "",
            "module2_component_id": mapping["component_id"] if auto else "",
            "selected_candidate": mapping["candidate_track_id"] if auto else "",
            "mapping_decision": "CONFIRMED" if auto else "UNREVIEWED",
            "mask_quality_decision": "PASS" if auto else "UNREVIEWED",
            "reviewed_at": "",
            "review_method": "HUMAN_VISUAL_REVIEW",
            "notes": (
                "Preserved previously authoritative confirmed mapping with existing PASS screening; "
                "no historical review timestamp fabricated."
                if auto else ""
            ),
            "auto_accepted_existing": "YES" if auto else "NO",
        })
    atomic_write_csv(history_csv, output)
    return output


def requires_human_review(mapping: Mapping[str, str], quality: Mapping[str, str]) -> bool:
    return not (
        mapping.get("mapping_status") == "CONFIRMED_EXISTING"
        and quality.get("mask_quality_status") == "PASS"
    )


def is_complete(row: Mapping[str, str]) -> bool:
    mapping = row.get("mapping_decision")
    if mapping in {"REJECTED", "UNCERTAIN"}:
        return True
    return mapping == "CONFIRMED" and row.get("mask_quality_decision") in COMPLETED_MASK


def _load_diagnostics(manifest_paths: Sequence[str | Path]) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifests = sorted({Path(value) for value in manifest_paths}, key=lambda p: p.as_posix().casefold())
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing Module 2 diagnostic manifest: {manifest}")
        for raw in read_csv(manifest):
            row: dict[str, Any] = dict(raw)
            row["manifest_file"] = str(manifest.resolve())
            row["candidate_track_id"] = row.get("track_id", "")
            row["diagnostic_mask_path"] = str((manifest.parent / row.get("diagnostic_mask_file", "")).resolve())
            row["diagnostic_image_path"] = str((manifest.parent / row.get("diagnostic_image_file", "")).resolve())
            row["diagnostic_overlay_path"] = str((manifest.parent / row.get("diagnostic_overlay_file", "")).resolve())
            row["_resolved_mask_file"] = row["diagnostic_mask_path"]
            by_source[row.get("source_filename", "")].append(row)
    return by_source


def build_review_queue(
    manifest_csv: str | Path,
    mapping_csv: str | Path,
    quality_csv: str | Path,
    candidate_csv: str | Path,
    diagnostics_root: str | Path,
    history_csv: str | Path,
) -> list[dict[str, Any]]:
    manifests = read_csv(manifest_csv)
    mappings = _index(read_csv(mapping_csv), "ground_truth_track_id")
    qualities = _index(read_csv(quality_csv), "ground_truth_track_id")
    history = load_history(history_csv)
    top_by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(candidate_csv):
        top_by_track[row["labelled_track_id"]].append(row)
    review_manifests = [
        row for row in manifests
        if requires_human_review(mappings[row["ground_truth_track_id"]], qualities[row["ground_truth_track_id"]])
    ]
    # Use authoritative queue manifest paths directly; never scan or regenerate Module 2 output trees.
    diagnostics_by_source = _load_diagnostics(row["diagnostic_manifest_path"] for row in review_manifests)
    queue = []
    for manifest in manifests:
        track_id = manifest["ground_truth_track_id"]
        mapping, quality = mappings[track_id], qualities[track_id]
        if not requires_human_review(mapping, quality):
            continue
        annotation = {
            "track_id": track_id, "class": manifest["ground_truth_class"],
            "source_image": manifest["source_image"], "source_dataset": manifest["source_dataset"],
        }
        queue.append({"manifest": manifest, "mapping": mapping, "quality": quality,
                      "annotation_stub": annotation, "top_candidates": top_by_track.get(track_id, []),
                      "source_diagnostics": diagnostics_by_source.get(manifest["source_image"], []),
                      "history": history[track_id]})

    label_index = _index(read_csv(Path(manifest_csv).with_name("independent_ground_truth_labels.csv")), "track_id")
    for item in queue:
        annotation = label_index[item["manifest"]["ground_truth_track_id"]]
        diagnostics = item.pop("source_diagnostics")
        diagnostic_index = {row["candidate_track_id"]: row for row in diagnostics}
        # The existing matcher already ranks five spatial alternatives. Reuse those metrics.
        ranked = []
        for top in sorted(item["top_candidates"], key=lambda row: int(row["rank"])):
            diagnostic = diagnostic_index.get(top["candidate_track_id"])
            if diagnostic:
                ranked.append({**diagnostic, **top})
        if item["mapping"]["mapping_status"] == "NO_MATCH":
            # Keep all remaining detections inspectable without decoding/re-ranking thousands of masks.
            seen = {row["candidate_track_id"] for row in ranked}
            for position, diagnostic in enumerate(sorted(diagnostics, key=lambda row: row["candidate_track_id"]), 1):
                if diagnostic["candidate_track_id"] not in seen:
                    ranked.append({**diagnostic, "rank": f"ALL_{position}"})
        # Put a prior/current selection first and ensure it remains inspectable if outside the top five.
        selected = item["history"].get("selected_candidate", "") or item["mapping"].get("candidate_track_id", "")
        if selected and not any(row["candidate_track_id"] == selected for row in ranked):
            diagnostic = diagnostic_index.get(selected)
            if diagnostic:
                candidate = {**diagnostic, "rank": "PRIOR_SELECTED"}
                ranked.append(candidate)
        if selected:
            ranked.sort(key=lambda row: (row["candidate_track_id"] != selected, str(row["rank"])))
        item["annotation"] = annotation
        item["candidates"] = ranked
        item.pop("annotation_stub", None)
        item.pop("top_candidates", None)
    return queue


def cycle_candidate(current: int, total: int, delta: int) -> int:
    if total <= 0:
        return 0
    return (current + delta) % total


def previous_index(current: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, current - 1)


def next_unresolved_index(queue: Sequence[Mapping[str, Any]], current: int) -> int | None:
    if not queue:
        return None
    for offset in range(1, len(queue) + 1):
        index = (current + offset) % len(queue)
        if not is_complete(queue[index]["history"]):
            return index
    return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _persist_row(history_csv: str | Path, updated: Mapping[str, str]) -> dict[str, str]:
    history = load_history(history_csv)
    history[updated["track_id"]] = dict(updated)
    ordered = list(history.values())
    atomic_write_csv(history_csv, ordered)
    return dict(updated)


def record_mapping_decision(
    history_csv: str | Path,
    track_id: str,
    decision: str,
    candidate: Mapping[str, Any] | None = None,
    *,
    notes: str = "",
    replace_existing: bool = False,
    timestamp: str | None = None,
) -> dict[str, str]:
    if decision not in {"CONFIRMED", "REJECTED", "UNCERTAIN"}:
        raise ValueError(f"Unsupported mapping decision: {decision}")
    history = load_history(history_csv)
    row = history[track_id]
    if row["mapping_decision"] != "UNREVIEWED" and not replace_existing:
        raise FileExistsError(f"A mapping decision already exists for {track_id}; explicit replacement is required.")
    if decision == "CONFIRMED" and not candidate:
        raise ValueError("CONFIRMED requires an exact selected Module 2 candidate.")
    updated = dict(row)
    updated.update({
        "mapping_decision": decision,
        "mask_quality_decision": "UNREVIEWED",
        "module2_event_id": str(candidate.get("candidate_track_id", "")) if candidate else "",
        "module2_component_id": str(candidate.get("component_id", "")) if candidate else "",
        "selected_candidate": str(candidate.get("candidate_track_id", "")) if candidate else "",
        "reviewed_at": timestamp or _timestamp(),
        "review_method": "HUMAN_VISUAL_REVIEW",
        "notes": notes,
        "auto_accepted_existing": "NO",
    })
    return _persist_row(history_csv, updated)


def record_mask_decision(
    history_csv: str | Path,
    track_id: str,
    decision: str,
    *,
    notes: str = "",
    replace_existing: bool = False,
    timestamp: str | None = None,
) -> dict[str, str]:
    if decision not in {"PASS", "FAIL", "UNCERTAIN"}:
        raise ValueError(f"Unsupported mask-quality decision: {decision}")
    history = load_history(history_csv)
    row = history[track_id]
    if row["mapping_decision"] != "CONFIRMED":
        raise ValueError("Mask quality can only be decided after mapping identity is CONFIRMED.")
    if row["mask_quality_decision"] != "UNREVIEWED" and not replace_existing:
        raise FileExistsError(f"A mask decision already exists for {track_id}; explicit replacement is required.")
    updated = dict(row)
    updated.update({
        "mask_quality_decision": decision,
        "reviewed_at": timestamp or _timestamp(),
        "review_method": "HUMAN_VISUAL_REVIEW",
        "notes": notes or row.get("notes", ""),
        "auto_accepted_existing": "NO",
    })
    return _persist_row(history_csv, updated)


def status_summary(manifest_csv: str | Path, mapping_csv: str | Path, quality_csv: str | Path,
                   history_csv: str | Path) -> dict[str, Any]:
    manifests = read_csv(manifest_csv)
    mappings = _index(read_csv(mapping_csv), "ground_truth_track_id")
    qualities = _index(read_csv(quality_csv), "ground_truth_track_id")
    history = load_history(history_csv)
    required = [row for row in manifests if requires_human_review(mappings[row["ground_truth_track_id"]], qualities[row["ground_truth_track_id"]])]
    required_ids = {row["ground_truth_track_id"] for row in required}
    completed = sum(is_complete(history[track_id]) for track_id in required_ids)
    mapping_counts = Counter(row["mapping_decision"] for row in history.values())
    mask_counts = Counter(row["mask_quality_decision"] for row in history.values())
    usable_ids = {track_id for track_id, row in history.items()
                  if row["mapping_decision"] == "CONFIRMED" and row["mask_quality_decision"] == "PASS"}
    manifest_by_id = {row["ground_truth_track_id"]: row for row in manifests}
    usable = [manifest_by_id[track_id] for track_id in usable_ids]
    def breakdown(field: str) -> dict[str, int]:
        return dict(sorted(Counter((row.get(field) or "UNAVAILABLE") for row in usable).items()))
    return {
        "total_independent_gt": len(manifests),
        "auto_accepted_existing": sum(row.get("auto_accepted_existing") == "YES" for row in history.values()),
        "human_review_required": len(required),
        "human_review_completed": completed,
        "human_review_remaining": len(required) - completed,
        "mapping": {key: mapping_counts[key] for key in ("CONFIRMED", "REJECTED", "UNCERTAIN", "UNREVIEWED")},
        "mask": {key: mask_counts[key] for key in ("PASS", "FAIL", "UNCERTAIN", "UNREVIEWED")},
        "provisional_module3_usable": len(usable),
        "provisional_by_class": breakdown("ground_truth_class"),
        "provisional_by_source_dataset": breakdown("source_dataset"),
        "provisional_by_source_video": breakdown("source_video"),
    }


def format_status(summary: Mapping[str, Any]) -> str:
    lines = [
        f"TOTAL INDEPENDENT GT: {summary['total_independent_gt']}",
        f"AUTO-ACCEPTED EXISTING: {summary['auto_accepted_existing']}",
        f"HUMAN REVIEW REQUIRED: {summary['human_review_required']}",
        f"HUMAN REVIEW COMPLETED: {summary['human_review_completed']}",
        f"HUMAN REVIEW REMAINING: {summary['human_review_remaining']}", "", "MAPPING:",
    ]
    lines += [f"{key}: {summary['mapping'][key]}" for key in ("CONFIRMED", "REJECTED", "UNCERTAIN", "UNREVIEWED")]
    lines += ["", "MASK:"]
    lines += [f"{key}: {summary['mask'][key]}" for key in ("PASS", "FAIL", "UNCERTAIN", "UNREVIEWED")]
    lines += ["", f"PROVISIONAL MODULE-3-USABLE: {summary['provisional_module3_usable']}",
              "This count is PROVISIONAL until human review is complete.", "", "BY CLASS:"]
    lines += [f"{key}: {value}" for key, value in summary["provisional_by_class"].items()]
    lines += ["", "BY SOURCE DATASET:"]
    lines += [f"{key}: {value}" for key, value in summary["provisional_by_source_dataset"].items()]
    lines += ["", "BY SOURCE VIDEO:"]
    lines += [f"{key}: {value}" for key, value in summary["provisional_by_source_video"].items()]
    return "\n".join(lines)


def _decode(path: str | Path | None, grayscale: bool = False) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)


def _fit(image: np.ndarray | None, width: int, height: int, title: str) -> np.ndarray:
    panel = np.full((height, width, 3), 25, np.uint8)
    cv2.putText(panel, title[:95], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
    if image is None:
        cv2.putText(panel, "UNAVAILABLE", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, .7, (140, 140, 140), 2)
        return panel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], (height - 38) / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    x, y = (width - resized.shape[1]) // 2, 38 + (height - 38 - resized.shape[0]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def render_review_screen(item: Mapping[str, Any], candidate_index: int, source_path: Path | None,
                         crop_path: Path | None, *, zoom_roi: bool = False) -> np.ndarray:
    annotation, candidates = item["annotation"], item["candidates"]
    candidate = candidates[candidate_index] if candidates else None
    source = _decode(source_path)
    crop = _decode(crop_path)
    mask = _decode(candidate.get("diagnostic_mask_path") if candidate else None, True)
    component = _decode(candidate.get("diagnostic_image_path") if candidate else None)
    overlay = _decode(candidate.get("diagnostic_overlay_path") if candidate else None)
    source_view = source.copy() if source is not None else None
    if source_view is not None:
        polygon = np.rint(manual_polygon(annotation)).astype(np.int32)
        cv2.polylines(source_view, [polygon], True, (255, 255, 0), 4, cv2.LINE_AA)
        # Every detection is visible in gray; selected detection is green.
        for row in candidates:
            x, y, w, h = [int(float(row[key])) for key in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")]
            cv2.rectangle(source_view, (x, y), (x + w, y + h), (90, 90, 90), 1)
        if candidate:
            x, y, w, h = [int(float(candidate[key])) for key in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")]
            cv2.rectangle(source_view, (x, y), (x + w, y + h), (0, 255, 0), 4)
            if mask is not None:
                x0, y0 = int(float(candidate["roi_x0"])), int(float(candidate["roi_y0"]))
                y1, x1 = min(source_view.shape[0], y0 + mask.shape[0]), min(source_view.shape[1], x0 + mask.shape[1])
                if x0 < x1 and y0 < y1:
                    foreground = mask[:y1-y0, :x1-x0] > 0
                    region = source_view[y0:y1, x0:x1]
                    tint = np.zeros_like(region); tint[:] = (0, 0, 255)
                    region[foreground] = cv2.addWeighted(region[foreground], .55, tint[foreground], .45, 0)
            if zoom_roi:
                gt = polygon.reshape(-1, 2)
                xs = list(gt[:, 0]) + [x, x + w]; ys = list(gt[:, 1]) + [y, y + h]
                pad = 50; left, right = max(0, min(xs)-pad), min(source_view.shape[1], max(xs)+pad)
                top, bottom = max(0, min(ys)-pad), min(source_view.shape[0], max(ys)+pad)
                source_view = source_view[top:bottom, left:right]

    canvas = np.full((1120, 1800, 3), 16, np.uint8)
    manifest, mapping, quality, history = item["manifest"], item["mapping"], item["quality"], item["history"]
    selected_number = candidate_index + 1 if candidate else 0
    cv2.putText(canvas, f"{annotation['track_id']} | {annotation['class']} | {annotation['source_dataset']} | candidate {selected_number}/{len(candidates)}",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"source: {annotation['source_image']} | video/frame/time: {manifest.get('source_video','')} / {manifest.get('frame_number','')} / {manifest.get('timestamp_seconds','')}",
                (15, 62), cv2.FONT_HERSHEY_SIMPLEX, .5, (210,210,210), 1, cv2.LINE_AA)
    identity = candidate.get("candidate_track_id", "NO CANDIDATE") if candidate else "NO CANDIDATE"
    component_id = candidate.get("component_id", "") if candidate else ""
    cv2.putText(canvas, f"Module 2: {identity} | component {component_id} | AUTO MAP {mapping['mapping_status']} | AUTO MASK {quality['mask_quality_status']}: {quality['quality_flags']}",
                (15, 91), cv2.FONT_HERSHEY_SIMPLEX, .48, (210,210,210), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"HUMAN MAP {history['mapping_decision']} | HUMAN MASK {history['mask_quality_decision']}",
                (15, 120), cv2.FONT_HERSHEY_SIMPLEX, .58, (0,230,255), 1, cv2.LINE_AA)
    canvas[140:760, 10:1010] = _fit(source_view, 1000, 620, "SOURCE: cyan=GT ROI; green=selected M2 bbox; red=M2 mask; gray=all detections")
    canvas[140:440, 1020:1405] = _fit(crop, 385, 300, "AUTHORITATIVE GT CROP")
    canvas[140:440, 1415:1790] = _fit(component, 375, 300, "MODULE 2 ROI / COMPONENT")
    canvas[450:750, 1020:1405] = _fit(mask, 385, 300, "MODULE 2 BINARY MASK")
    canvas[450:750, 1415:1790] = _fit(overlay, 375, 300, "MODULE 2 OVERLAY")
    help_lines = [
        "MAPPING: C confirmed | R rejected | U uncertain     CANDIDATES: [ previous | ] next",
        "MASK (only after confirmed): P pass | F fail | U uncertain     B previous case | N next unresolved | Q safe quit",
        "VIEW: Z zoom GT/candidate | 0 full source     Existing decisions require Y confirmation before replacement.",
        "Press M to return explicitly to the mapping stage when revisiting a confirmed case.",
        "MASK PASS = enough intended track for meaningful geometry. FAIL = border/background/wrong, severe clipping, missing, merged, or degenerate.",
    ]
    for index, line in enumerate(help_lines):
        cv2.putText(canvas, line, (18, 790 + index*38), cv2.FONT_HERSHEY_SIMPLEX, .54, (230,230,230), 1, cv2.LINE_AA)
    return canvas
