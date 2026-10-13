from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.geometric_classification.expanded_human_approved_features import (
    FEATURES,
    authorized_history,
    run_expanded_feature_stage,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fixture(root: Path) -> dict[str, Path]:
    ids = [f"A{i:03d}" for i in range(1, 54) if i not in {18, 27}] + [f"E{i:03d}" for i in range(1, 64)]
    authorized = [track_id for track_id in ids if track_id.startswith("A")][:16] + [track_id for track_id in ids if track_id.startswith("E")][:19]
    confirmed = ids[:108]
    # Ensure all authorized tracks are in the confirmed set.
    confirmed = authorized + [track_id for track_id in ids if track_id not in authorized][:73]
    rejected = [track_id for track_id in ids if track_id not in confirmed]
    assert len(confirmed) == 108 and len(rejected) == 6
    history, labels, batch, diagnostic_rows = [], [], [], []
    manifest_path = root / "module2_diagnostics/source/manifest.csv"
    for index, track_id in enumerate(ids):
        class_name = "Alpha" if track_id.startswith("A") else "Electron"
        is_authorized = track_id in authorized
        is_confirmed = track_id in confirmed
        candidate = f"candidate_{track_id}"
        source_dataset = "AWAN" if is_authorized and authorized.index(track_id) < 7 else "Lecturer"
        source = f"source_{track_id}.png"
        history.append({
            "track_id": track_id, "ground_truth_class": class_name, "source": source,
            "source_dataset": source_dataset, "source_video": "video.mp4" if source_dataset == "Lecturer" else "",
            "frame_number": str(index), "timestamp_seconds": str(index / 10),
            "module2_event_id": candidate if is_authorized else "", "module2_component_id": str(index + 1) if is_authorized else "",
            "selected_candidate": candidate if is_authorized else "",
            "mapping_decision": "CONFIRMED" if is_confirmed else "REJECTED",
            "mask_quality_decision": "PASS" if is_authorized else ("FAIL" if is_confirmed else "UNREVIEWED"),
            "reviewed_at": "2026-01-01T00:00:00+00:00", "review_method": "HUMAN_VISUAL_REVIEW",
            "notes": "fixture", "auto_accepted_existing": "NO",
        })
        labels.append({
            "track_id": track_id, "class": class_name, "source_dataset": source_dataset, "source_image": source,
            "event_id": f"EVENT_{track_id}", "roi_type": "axis_aligned", "ground_truth_quality": "high",
        })
        batch.append({
            "ground_truth_track_id": track_id, "diagnostic_manifest_path": str(manifest_path),
            "physical_track_group_id": "PTG001" if track_id == "A020" else "",
        })
        if is_authorized:
            mask_rel = f"masks/{candidate}_mask.png"
            diagnostic_rows.append({
                "track_id": candidate, "source_filename": source, "component_id": str(index + 1),
                "label_id": str(index + 1), "diagnostic_mask_file": mask_rel,
            })
            mask_path = manifest_path.parent / mask_rel; mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((80, 40), np.uint8)
            cv2.line(mask, (20, 8), (20 + index % 4, 70), 255, 3 + index % 3)
            cv2.imwrite(str(mask_path), mask)
    previous = []
    for index in range(29):
        previous.append({
            "ground_truth_track_id": f"OLD{index:03d}", "ground_truth_class": "Alpha" if index < 14 else "Electron",
            "source_dataset": "AWAN" if index < 10 else "Lecturer", "source_image": f"old_{index}.png",
            "source_video": "", "roi_type": "axis_aligned", "dependency_group_id": f"OLD_DEP_{index}",
            "processing_status": "success", **{feature: str(index + 1) for feature in FEATURES},
        })
    paths = {
        "history": root / "history.csv", "labels": root / "independent_ground_truth_labels.csv",
        "batch": root / "module2_batch_manifest.csv", "previous": root / "previous.csv", "output": root / "output",
    }
    for key, rows in (("history", history), ("labels", labels), ("batch", batch), ("previous", previous)):
        _write(paths[key], rows)
    _write(manifest_path, diagnostic_rows)
    return paths


def test_authorization_gate_selects_only_exact_completed_intersection(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    selected = authorized_history(paths["history"])
    assert len(selected) == 35
    assert sum(row["ground_truth_class"] == "Alpha" for row in selected) == 16
    assert sum(row["source_dataset"] == "AWAN" for row in selected) == 7
    assert not {"A018", "A027"} & {row["track_id"] for row in selected}
    rows = _read(paths["history"]); rows[0]["mapping_decision"] = "UNCERTAIN"; _write(paths["history"], rows)
    with pytest.raises(ValueError, match="not at the verified completed state"):
        authorized_history(paths["history"])


def test_stage_uses_exact_seven_features_and_builds_audits_without_mutating_inputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    protected = [paths[key] for key in ("history", "labels", "batch", "previous")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    result = run_expanded_feature_stage(paths["history"], paths["labels"], paths["batch"], paths["previous"], paths["output"])
    assert result["authorized"] == 35
    assert result["successful"] == 35
    feature_rows = _read(paths["output"] / "expanded_module3_features.csv")
    assert len(feature_rows) == 35
    assert all(set(FEATURES) <= set(row) for row in feature_rows)
    assert "endpoint_distance_px" not in feature_rows[0]
    assert len(_read(paths["output"] / "expanded_feature_failures.csv")) == 0
    independence = _read(paths["output"] / "expanded_independence_audit.csv")
    assert all(row["duplicate_module2_mask_id"] == "NO" for row in independence)
    assert (paths["output"] / "expanded_feature_audit.txt").is_file()
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}


def test_failure_and_duplicate_mask_are_reported_without_relaxed_extraction(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    history = _read(paths["history"])
    selected = [row for row in history if row["mapping_decision"] == "CONFIRMED" and row["mask_quality_decision"] == "PASS"]
    selected[1]["selected_candidate"] = selected[0]["selected_candidate"]
    selected[1]["module2_event_id"] = selected[0]["module2_event_id"]
    _write(paths["history"], history)
    result = run_expanded_feature_stage(paths["history"], paths["labels"], paths["batch"], paths["previous"], paths["output"])
    # Source provenance rejects the shared mask for the second record; duplicate identity is still audited.
    assert result["failed"] == 1
    assert result["duplicate_mask_rows"] == 2
    assert result["readiness"].startswith("C.")
    failures = _read(paths["output"] / "expanded_feature_failures.csv")
    assert "source does not match" in failures[0]["error_message"]

