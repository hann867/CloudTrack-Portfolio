"""Module-3-side orchestration for batch generation of Module 2 diagnostics."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


BATCH_SUMMARY_COLUMNS = [
    "source_image",
    "resolved_source_path",
    "labelled_track_count",
    "diagnostic_status",
    "module2_candidate_count",
    "module3_success_count",
    "module3_failure_count",
    "diagnostic_manifest_path",
    "error_message",
]

VALID_CLASSES = {"Alpha", "Electron"}
REQUIRED_MANIFEST_COLUMNS = {"track_id", "source_filename", "processing_status"}


def discover_labelled_sources(labels_csv: str | Path) -> tuple[list[dict[str, str]], Counter[str]]:
    """Load valid labelled rows and count their distinct source filenames."""

    with Path(labels_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("class") in VALID_CLASSES]
    missing = [row.get("track_id", "<unknown>") for row in rows if not str(row.get("source_image", "")).strip()]
    if missing:
        raise ValueError(f"Valid labelled rows have blank source_image: {', '.join(missing)}")
    return rows, Counter(str(row["source_image"]).strip() for row in rows)


def resolve_source_files(source_names: Sequence[str], search_roots: Sequence[str | Path]) -> tuple[dict[str, Path], dict[str, str]]:
    """Resolve exact filenames recursively; ambiguous filenames are not substituted."""

    matches: dict[str, list[Path]] = defaultdict(list)
    wanted = set(source_names)
    for root_value in search_roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in wanted:
                matches[path.name].append(path.resolve())
    resolved: dict[str, Path] = {}
    errors: dict[str, str] = {}
    for name in source_names:
        exact = sorted(set(matches.get(name, [])), key=lambda path: path.as_posix().casefold())
        if len(exact) == 1:
            resolved[name] = exact[0]
        elif not exact:
            errors[name] = "Exact source filename was not found under the configured raw-source roots."
        else:
            errors[name] = "Exact source filename is ambiguous: " + " | ".join(map(str, exact))
    return resolved, errors


def read_valid_manifest(manifest: str | Path, source_image: str) -> list[dict[str, str]] | None:
    """Return rows only when an existing diagnostic manifest is structurally valid."""

    path = Path(manifest)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not REQUIRED_MANIFEST_COLUMNS.issubset(reader.fieldnames):
                return None
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return None
    if any(str(row.get("source_filename", "")) != source_image for row in rows):
        return None
    return rows


def _manifest_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    success = sum(str(row.get("processing_status", "")).lower() == "success" for row in rows)
    return len(rows), success, len(rows) - success


def _summary_row(source_image: str, source_path: Path | None, count: int) -> dict[str, Any]:
    row = {column: "" for column in BATCH_SUMMARY_COLUMNS}
    row.update({
        "source_image": source_image,
        "resolved_source_path": str(source_path) if source_path else "",
        "labelled_track_count": count,
        "module2_candidate_count": 0,
        "module3_success_count": 0,
        "module3_failure_count": 0,
    })
    return row


def write_batch_summary(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_diagnostic_batch(
    labels_csv: str | Path,
    search_roots: Sequence[str | Path],
    diagnostics_root: str | Path,
    *,
    process_image_fn: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]],
    module2_config: Mapping[str, Any],
    adapter_fn: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    exporter_fn: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Generate missing diagnostics while isolating every source-level failure."""

    labelled_rows, source_counts = discover_labelled_sources(labels_csv)
    source_names = sorted(source_counts, key=lambda value: (value.casefold(), value))
    resolved, resolution_errors = resolve_source_files(source_names, search_roots)
    output_root = Path(diagnostics_root)
    summary_rows: list[dict[str, Any]] = []
    for source_image in source_names:
        source_path = resolved.get(source_image)
        row = _summary_row(source_image, source_path, source_counts[source_image])
        if source_path is None:
            row["diagnostic_status"] = "MISSING_SOURCE"
            row["error_message"] = resolution_errors[source_image]
            summary_rows.append(row)
            continue
        manifest = output_root / source_path.stem / "manifest.csv"
        existing_rows = read_valid_manifest(manifest, source_image)
        if existing_rows is not None:
            candidates, successes, failures = _manifest_counts(existing_rows)
            row.update({
                "diagnostic_status": "ALREADY_EXISTS",
                "module2_candidate_count": candidates,
                "module3_success_count": successes,
                "module3_failure_count": failures,
                "diagnostic_manifest_path": str(manifest.resolve()),
            })
            summary_rows.append(row)
            continue
        try:
            module2_result = process_image_fn(source_path, module2_config)
        except Exception as exc:  # source-level isolation is intentional
            row["diagnostic_status"] = "MODULE2_FAILED"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
            summary_rows.append(row)
            continue
        try:
            track_results = list(adapter_fn(module2_result))
            export = exporter_fn(module2_result, track_results, output_root, source_stem=source_path.stem)
            generated_rows = read_valid_manifest(export["manifest"], source_image)
            if generated_rows is None:
                raise ValueError("Exporter did not create a valid exact-source diagnostic manifest.")
            candidates, successes, failures = _manifest_counts(generated_rows)
            row.update({
                "diagnostic_status": "GENERATED",
                "module2_candidate_count": candidates,
                "module3_success_count": successes,
                "module3_failure_count": failures,
                "diagnostic_manifest_path": str(Path(export["manifest"]).resolve()),
            })
        except Exception as exc:  # source-level isolation is intentional
            row["diagnostic_status"] = "EXPORT_FAILED"
            row["module2_candidate_count"] = len(module2_result.get("rois", []))
            row["error_message"] = f"{type(exc).__name__}: {exc}"
        summary_rows.append(row)
    summary_path = output_root / "batch_generation_summary.csv"
    write_batch_summary(summary_path, summary_rows)
    status_counts = Counter(str(row["diagnostic_status"]) for row in summary_rows)
    return {
        "total_labelled_records": len(labelled_rows),
        "unique_source_images": len(source_names),
        "resolved_source_images": len(resolved),
        "missing_source_images": len(source_names) - len(resolved),
        "status_counts": dict(status_counts),
        "total_module2_candidates": sum(int(row["module2_candidate_count"]) for row in summary_rows),
        "module3_success_count": sum(int(row["module3_success_count"]) for row in summary_rows),
        "module3_failure_count": sum(int(row["module3_failure_count"]) for row in summary_rows),
        "summary_path": str(summary_path.resolve()),
        "rows": summary_rows,
    }
