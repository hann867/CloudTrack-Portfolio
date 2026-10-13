from __future__ import annotations

import csv
from pathlib import Path

from src.geometric_classification.module2_diagnostic_batch import (
    BATCH_SUMMARY_COLUMNS,
    discover_labelled_sources,
    generate_diagnostic_batch,
    resolve_source_files,
)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _labels(path: Path, sources: list[str]) -> None:
    rows = [{"track_id": f"A{index:03d}", "class": "Alpha", "source_image": source}
            for index, source in enumerate(sources, 1)]
    rows.append({"track_id": "U001", "class": "Uncertain", "source_image": "ignored.png"})
    _write_csv(path, ["track_id", "class", "source_image"], rows)


def _exporter(module2_result, results, output_root, *, source_stem):
    root = Path(output_root) / source_stem
    rows = [{"track_id": result["track_id"], "source_filename": module2_result["path"].name,
             "processing_status": result["processing_status"]} for result in results]
    _write_csv(root / "manifest.csv", ["track_id", "source_filename", "processing_status"], rows)
    return {"manifest": str(root / "manifest.csv")}


def _adapter(result):
    return [{"track_id": roi["track_id"], "processing_status": roi.get("status", "success")}
            for roi in result["rois"]]


def test_discovers_unique_valid_labelled_source_images(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _labels(labels, ["two.png", "one.png", "two.png"])
    rows, counts = discover_labelled_sources(labels)
    assert len(rows) == 3
    assert counts == {"one.png": 1, "two.png": 2}


def test_source_resolution_requires_exact_unambiguous_filename(tmp_path: Path) -> None:
    root = tmp_path / "raw"; (root / "nested").mkdir(parents=True)
    (root / "nested" / "Exact.png").write_bytes(b"source")
    resolved, errors = resolve_source_files(["Exact.png", "exact.png", "absent.png"], [root])
    assert resolved["Exact.png"].name == "Exact.png"
    assert "exact.png" in errors
    assert "absent.png" in errors


def test_existing_valid_diagnostics_are_skipped(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"; _labels(labels, ["frame.png"])
    raw = tmp_path / "raw"; raw.mkdir(); (raw / "frame.png").write_bytes(b"source")
    diagnostics = tmp_path / "diagnostics"
    _write_csv(diagnostics / "frame" / "manifest.csv",
               ["track_id", "source_filename", "processing_status"],
               [{"track_id": "track_1", "source_filename": "frame.png", "processing_status": "success"}])
    calls = []
    result = generate_diagnostic_batch(
        labels, [raw], diagnostics, process_image_fn=lambda *_: calls.append(True), module2_config={},
        adapter_fn=_adapter, exporter_fn=_exporter,
    )
    assert not calls
    assert result["status_counts"] == {"ALREADY_EXISTS": 1}
    assert result["total_module2_candidates"] == 1


def test_failed_source_does_not_abort_and_summary_is_complete(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"; _labels(labels, ["bad.png", "good.png", "missing.png"])
    labels_before = labels.read_bytes()
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "bad.png").write_bytes(b"bad"); (raw / "good.png").write_bytes(b"good")
    module2_owned = tmp_path / "module2_owned.txt"; module2_owned.write_bytes(b"unchanged")
    module2_before = module2_owned.read_bytes()

    def process(path, _config):
        if path.name == "bad.png":
            raise RuntimeError("isolated failure")
        return {"path": path, "rois": [{"track_id": "track_1"},
                                         {"track_id": "track_2", "status": "failed"}]}

    diagnostics = tmp_path / "diagnostics"
    result = generate_diagnostic_batch(
        labels, [raw], diagnostics, process_image_fn=process, module2_config={},
        adapter_fn=_adapter, exporter_fn=_exporter,
    )
    assert result["status_counts"] == {"MODULE2_FAILED": 1, "GENERATED": 1, "MISSING_SOURCE": 1}
    assert result["module3_success_count"] == 1
    assert result["module3_failure_count"] == 1
    assert labels.read_bytes() == labels_before
    assert module2_owned.read_bytes() == module2_before
    with Path(result["summary_path"]).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); rows = list(reader)
    assert reader.fieldnames == BATCH_SUMMARY_COLUMNS
    assert [row["source_image"] for row in rows] == ["bad.png", "good.png", "missing.png"]


def test_export_failure_is_recorded_without_aborting_later_sources(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"; _labels(labels, ["a.png", "b.png"])
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "a.png").write_bytes(b"a"); (raw / "b.png").write_bytes(b"b")

    def exporter(result, tracks, output, *, source_stem):
        if result["path"].name == "a.png":
            raise OSError("export failure")
        return _exporter(result, tracks, output, source_stem=source_stem)

    result = generate_diagnostic_batch(
        labels, [raw], tmp_path / "diagnostics",
        process_image_fn=lambda path, _: {"path": path, "rois": [{"track_id": "track_1"}]},
        module2_config={}, adapter_fn=_adapter, exporter_fn=exporter,
    )
    assert result["status_counts"] == {"EXPORT_FAILED": 1, "GENERATED": 1}
