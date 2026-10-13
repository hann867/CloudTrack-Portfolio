"""Create the standalone CloudTrack video-analysis PDF from saved job outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape
import uuid


REPORT_FILENAME = "CloudTrack_Video_Analysis_Report.pdf"


def _preview_indices(total_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    return sorted({round((total_frames - 1) * fraction) for fraction in (0, 0.25, 0.5, 0.75, 1)})


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload


def _report_is_current(report_path: Path, source_paths: Sequence[Path]) -> bool:
    return report_path.is_file() and all(
        not path.is_file() or path.stat().st_mtime <= report_path.stat().st_mtime for path in source_paths
    )


def generate_video_report(job_directory: Path, *, force: bool = False) -> Path:
    """Generate or reuse a compact report without rerunning image-analysis modules."""
    metadata_path = job_directory / "metadata.json"
    frames_path = job_directory / "frames.json"
    results_path = job_directory / "results.json"
    for source_path in (metadata_path, frames_path, results_path):
        if not source_path.is_file():
            raise FileNotFoundError(f"Required report source is missing: {source_path.name}")

    report_directory = job_directory / "report"
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / REPORT_FILENAME
    if not force and _report_is_current(report_path, (metadata_path, frames_path, results_path, Path(__file__))):
        return report_path

    metadata = _load_json(metadata_path)
    results = _load_json(results_path)
    frame_results = results.get("frames", [])
    if not isinstance(frame_results, list) or not frame_results:
        raise ValueError("Video results contain no frame analysis records.")

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as error:
        raise RuntimeError(
            "PDF dependency import failed. Install backend/requirements.txt using the Python environment "
            f"that starts Uvicorn. Import error: {error}"
        ) from error

    palette = {
        "ink": colors.HexColor("#172033"),
        "muted": colors.HexColor("#5F6B7A"),
        "blue": colors.HexColor("#2563EB"),
        "panel": colors.HexColor("#F3F6FB"),
        "line": colors.HexColor("#D8E0EC"),
        "white": colors.white,
    }
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VideoTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23,
        leading=28, textColor=palette["ink"], spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "VideoSubtitle", parent=styles["Normal"], fontSize=10, leading=14,
        textColor=palette["blue"], spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "VideoHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11,
        leading=14, textColor=palette["ink"], spaceBefore=9, spaceAfter=7,
    )
    body_style = ParagraphStyle(
        "VideoBody", parent=styles["BodyText"], fontSize=8.5, leading=12,
        textColor=palette["ink"],
    )
    note_style = ParagraphStyle(
        "VideoNote", parent=body_style, fontSize=8, leading=11, textColor=palette["muted"],
        backColor=palette["panel"], borderColor=palette["line"], borderWidth=0.5,
        borderPadding=8, spaceBefore=8,
    )
    caption_style = ParagraphStyle(
        "FrameCaption", parent=body_style, fontSize=7.5, leading=10, alignment=TA_CENTER,
    )

    temporary_report = report_directory / f".report-{uuid.uuid4().hex}.pdf"
    document = SimpleDocTemplate(
        str(temporary_report), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=17 * mm, bottomMargin=16 * mm,
        title="CloudTrack Video Analysis Report",
        author="CloudTrack",
    )

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(palette["line"])
        canvas.line(16 * mm, 12 * mm, A4[0] - 16 * mm, 12 * mm)
        canvas.setFillColor(palette["muted"])
        canvas.setFont("Helvetica", 7)
        canvas.drawString(16 * mm, 8 * mm, f"CloudTrack · Job {job_directory.name}")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    def information_table(rows: list[tuple[str, Any]]) -> Table:
        table = Table([[label, Paragraph(escape(str(value)), body_style)] for label, value in rows],
                      colWidths=[59 * mm, 102 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), palette["panel"]),
            ("TEXTCOLOR", (0, 0), (-1, -1), palette["ink"]),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.45, palette["line"]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return table

    particles = results.get("particles", {})
    processing = results.get("processing", {})
    output = results.get("output", {})
    total_tracks = int(particles.get("total_tracks", particles.get("total_track_detections", 0)) or 0)
    story = [
        Paragraph("CloudTrack", subtitle_style),
        Paragraph("CloudTrack Video Analysis Report", title_style),
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Job {job_directory.name}",
            body_style,
        ),
        Spacer(1, 8),
        Paragraph("VIDEO INFORMATION", heading_style),
        information_table([
            ("Filename", metadata.get("filename", "—")),
            ("Video Duration", f"{float(metadata.get('duration_seconds', 0)):.2f} seconds"),
            ("Processing Duration", (
                f"{float(results['processing_duration_seconds']):.2f} seconds"
                if results.get("processing_duration_seconds") is not None else "Not recorded for this job"
            )),
            ("Original FPS", f"{float(metadata.get('original_fps', 0)):.3f}".rstrip("0").rstrip(".")),
            ("Processing FPS", f"{float(metadata.get('target_fps', 30)):.0f}"),
            ("Resolution", f"{metadata.get('width', 0)} × {metadata.get('height', 0)}"),
            ("Original Frames", metadata.get("original_frames", 0)),
            ("Processed Frames", output.get("frame_count", results.get("total_frames", 0))),
        ]),
        Paragraph("PARTICLE TRACK DETECTIONS ACROSS ANALYSED FRAMES", heading_style),
        information_table([
            ("Alpha", int(particles.get("alpha", 0) or 0)),
            ("Electron", int(particles.get("electron", 0) or 0)),
            ("Total Track Detections", total_tracks),
            ("Unique Analysed Source Frames", int(results.get("unique_source_frames_processed", 0) or 0)),
        ]),
        Paragraph("PROCESSING SUMMARY", heading_style),
        information_table([
            ("Frames Extracted", int(processing.get("extracted_frames", 0) or 0)),
            ("Frames Preprocessed", int(processing.get("preprocessed_frames", 0) or 0)),
            ("Frames Analysed", int(processing.get("detected_frames", 0) or 0)),
            ("Frames Classified", int(processing.get("classified_frames", 0) or 0)),
            ("Failed Frames", int(processing.get("failed_frames", 0) or 0)),
        ]),
        Paragraph(
            "Scientific note: Video frames are analysed independently. Track detections across adjacent "
            "frames may represent the same physical particle; these values are not unique physical-particle counts.",
            note_style,
        ),
        PageBreak(),
        Paragraph("Representative Classified Frames", title_style),
        Paragraph("Five points sampled across the 30 FPS presentation timeline.", subtitle_style),
    ]

    representative_frames = [frame_results[index] for index in _preview_indices(len(frame_results))]
    cards = []
    summary_rows = [["Frame", "Time", "Tracks", "Alpha", "Electron"]]
    for frame in representative_frames:
        frame_number = int(frame.get("output_frame_index", 0)) + 1
        image_path = job_directory / "classified" / str(frame.get("filename", f"frame_{frame_number:06d}.png"))
        if image_path.is_file():
            report_image = Image(str(image_path), width=65 * mm, height=36 * mm, kind="proportional")
        else:
            report_image = Paragraph("Classified image unavailable", note_style)
        caption = Paragraph(
            f"<b>Frame {frame_number}</b><br/>{float(frame.get('timestamp_seconds', 0)):.2f} sec · "
            f"Tracks {int(frame.get('tracks_detected', 0))} · Alpha {int(frame.get('alpha_count', 0))} · "
            f"Electron {int(frame.get('electron_count', 0))}",
            caption_style,
        )
        cards.append([report_image, Spacer(1, 3), caption])
        summary_rows.append([
            frame_number,
            f"{float(frame.get('timestamp_seconds', 0)):.2f}",
            int(frame.get("tracks_detected", 0)),
            int(frame.get("alpha_count", 0)),
            int(frame.get("electron_count", 0)),
        ])

    for offset in range(0, len(cards), 2):
        row = cards[offset:offset + 2]
        if len(row) == 1:
            row.append("")
        frame_table = Table([row], colWidths=[82 * mm, 82 * mm], hAlign="LEFT")
        frame_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOX", (0, 0), (0, 0), 0.5, palette["line"]),
            ("BOX", (1, 0), (1, 0), 0.5, palette["line"]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([frame_table, Spacer(1, 7)])

    story.extend([Paragraph("Representative Frame Summary", heading_style)])
    summary_table = Table(summary_rows, colWidths=[30 * mm, 30 * mm, 34 * mm, 34 * mm, 34 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), palette["blue"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), palette["white"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.45, palette["line"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [palette["white"], palette["panel"]]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    try:
        document.build(story, onFirstPage=footer, onLaterPages=footer)
        temporary_report.replace(report_path)
    finally:
        temporary_report.unlink(missing_ok=True)
    return report_path
