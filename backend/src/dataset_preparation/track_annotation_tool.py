"""Interactive OpenCV tool for manual known-source track annotation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import cv2
import numpy as np

from .crop_utils import crop_original_pixels, validate_roi
from .crop_utils import (
    RotatedROI,
    extract_rotated_crop,
    rotated_roi_bounding_box,
    rotated_roi_from_axis,
    validate_rotated_roi,
)
from .dataset_metadata import (
    AWAN_IMAGE_NAME,
    AWAN_PANEL_METADATA,
    create_event_id,
    ensure_dataset_structure,
    save_annotation,
)
from .video_review_history import format_source_frame_warning


SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def discover_image_files(folder: Path) -> list[Path]:
    """Return supported files directly in ``folder`` in deterministic name order."""

    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a source image directory: {folder}")
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda path: (path.name.casefold(), path.name),
    )


def navigate_image_index(current: int, total: int, command: str) -> int | None:
    """Apply a batch navigation command without wrapping at either boundary."""

    if total <= 0 or not 0 <= current < total:
        raise ValueError("Current image index is outside the batch")
    normalized = command.strip().lower()
    if normalized in {"n", "next", "k", "skip"}:
        return min(current + 1, total - 1)
    if normalized in {"p", "previous"}:
        return max(current - 1, 0)
    if normalized in {"q", "quit"}:
        return None
    raise ValueError(f"Unknown batch navigation command: {command}")


def _display_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


@dataclass
class ViewportROISelector:
    """Rectangle selector with zoom and pan while retaining original coordinates."""

    image: np.ndarray
    max_width: int = 1400
    max_height: int = 850
    window_name: str = "Select one complete track"
    batch_mode: bool = False

    def __post_init__(self) -> None:
        self.display_image = _display_bgr(self.image)
        self.image_height, self.image_width = self.display_image.shape[:2]
        self.canvas_width = min(self.max_width, self.image_width)
        self.canvas_height = min(self.max_height, self.image_height)
        self.fit_zoom = min(self.canvas_width / self.image_width, self.canvas_height / self.image_height)
        self.zoom = self.fit_zoom
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.pan_start: tuple[int, int] | None = None
        self.pan_origin = (0.0, 0.0)
        self.selected_roi: tuple[int, int, int, int] | None = None

    def _clip_offsets(self) -> None:
        visible_width = self.canvas_width / self.zoom
        visible_height = self.canvas_height / self.zoom
        self.offset_x = min(max(0.0, self.offset_x), max(0.0, self.image_width - visible_width))
        self.offset_y = min(max(0.0, self.offset_y), max(0.0, self.image_height - visible_height))

    def _screen_to_original(self, x: int, y: int) -> tuple[int, int]:
        original_x = int(round(self.offset_x + x / self.zoom))
        original_y = int(round(self.offset_y + y / self.zoom))
        return min(max(original_x, 0), self.image_width), min(max(original_y, 0), self.image_height)

    def _zoom_at(self, x: int, y: int, factor: float) -> None:
        before_x = self.offset_x + x / self.zoom
        before_y = self.offset_y + y / self.zoom
        self.zoom = min(max(self.fit_zoom, self.zoom * factor), max(8.0, self.fit_zoom))
        self.offset_x = before_x - x / self.zoom
        self.offset_y = before_y - y / self.zoom
        self._clip_offsets()

    def _mouse(self, event: int, x: int, y: int, flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x1, y1 = self._screen_to_original(*self.drag_start)
            x2, y2 = self._screen_to_original(x, y)
            candidate = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            self.drag_start = None
            self.drag_current = None
            try:
                self.selected_roi = validate_roi(candidate, self.image_width, self.image_height)
            except ValueError:
                self.selected_roi = None
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self.pan_start = (x, y)
            self.pan_origin = (self.offset_x, self.offset_y)
        elif event == cv2.EVENT_MOUSEMOVE and self.pan_start is not None:
            self.offset_x = self.pan_origin[0] - (x - self.pan_start[0]) / self.zoom
            self.offset_y = self.pan_origin[1] - (y - self.pan_start[1]) / self.zoom
            self._clip_offsets()
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self.pan_start = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._zoom_at(x, y, 1.25 if flags > 0 else 0.8)

    def _render(self) -> np.ndarray:
        transform = np.array(
            [[self.zoom, 0.0, -self.offset_x * self.zoom], [0.0, self.zoom, -self.offset_y * self.zoom]],
            dtype=np.float32,
        )
        canvas = cv2.warpAffine(
            self.display_image,
            transform,
            (self.canvas_width, self.canvas_height),
            flags=cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_LINEAR,
        )
        if self.drag_start and self.drag_current:
            cv2.rectangle(canvas, self.drag_start, self.drag_current, (0, 255, 0), 2)
        header_height = 76 if self.batch_mode else 50
        cv2.rectangle(canvas, (0, 0), (self.canvas_width, header_height), (0, 0, 0), -1)
        cv2.putText(canvas, "Left-drag ROI | wheel +/- zoom | right/middle-drag pan | 0 fit", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        controls = (
            "N next image | P previous image | K skip image | R reset | Q quit"
            if self.batch_mode
            else "R reset | Q quit"
        )
        cv2.putText(canvas, controls, (12, 55 if self.batch_mode else 43), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def select(self) -> tuple[int, int, int, int] | str | None:
        self.selected_roi = None
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse)
        try:
            while True:
                cv2.imshow(self.window_name, self._render())
                key = cv2.waitKey(20) & 0xFF
                if self.batch_mode and key in (ord("n"), ord("N"), ord("p"), ord("P"), ord("k"), ord("K")):
                    self.drag_start = None
                    self.drag_current = None
                    self.selected_roi = None
                    return chr(key).lower()
                if key in (ord("q"), ord("Q"), 27):
                    return None
                if key in (ord("r"), ord("R")):
                    self.drag_start = None
                    self.drag_current = None
                    self.selected_roi = None
                elif key == ord("0"):
                    self.zoom = self.fit_zoom
                    self.offset_x = self.offset_y = 0.0
                elif key in (ord("+"), ord("=")):
                    self._zoom_at(self.canvas_width // 2, self.canvas_height // 2, 1.25)
                elif key in (ord("-"), ord("_")):
                    self._zoom_at(self.canvas_width // 2, self.canvas_height // 2, 0.8)
                if self.selected_roi is not None:
                    return self.selected_roi
        finally:
            cv2.destroyWindow(self.window_name)


class OrientedROISelector(ViewportROISelector):
    """Select an oriented rectangle using a centreline and perpendicular width."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.window_name = "Select one oriented track ROI"
        self.axis_start: tuple[float, float] | None = None
        self.axis_end: tuple[float, float] | None = None
        self.width_point: tuple[float, float] | None = None
        self.axis_dragging = False
        self.selected_rotated_roi: RotatedROI | None = None
        self.selection_error = ""

    def _original_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        return (
            round((point[0] - self.offset_x) * self.zoom),
            round((point[1] - self.offset_y) * self.zoom),
        )

    def _candidate(self) -> RotatedROI | None:
        if self.axis_start is None or self.axis_end is None or self.width_point is None:
            return None
        delta_x = self.axis_end[0] - self.axis_start[0]
        delta_y = self.axis_end[1] - self.axis_start[1]
        axis_length = float(np.hypot(delta_x, delta_y))
        if axis_length < 2:
            return None
        half_height = abs(
            delta_x * (self.width_point[1] - self.axis_start[1])
            - delta_y * (self.width_point[0] - self.axis_start[0])
        ) / axis_length
        if half_height < 1:
            return None
        return rotated_roi_from_axis(self.axis_start, self.axis_end, half_height)

    def _reset_oriented_selection(self) -> None:
        self.axis_start = None
        self.axis_end = None
        self.width_point = None
        self.axis_dragging = False
        self.selected_rotated_roi = None
        self.selection_error = ""

    def _mouse(self, event: int, x: int, y: int, flags: int, _parameter: object) -> None:
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self.pan_start = (x, y)
            self.pan_origin = (self.offset_x, self.offset_y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self.pan_start is not None:
            self.offset_x = self.pan_origin[0] - (x - self.pan_start[0]) / self.zoom
            self.offset_y = self.pan_origin[1] - (y - self.pan_start[1]) / self.zoom
            self._clip_offsets()
            return
        if event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self.pan_start = None
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            self._zoom_at(x, y, 1.25 if flags > 0 else 0.8)
            return

        original_point = tuple(float(value) for value in self._screen_to_original(x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.axis_start is not None and self.axis_end is not None and not self.axis_dragging:
                self.width_point = original_point
                candidate = self._candidate()
                if candidate is None:
                    self.selection_error = "Width is too small; move farther from the centreline and click."
                    return
                try:
                    self.selected_rotated_roi = validate_rotated_roi(candidate, self.image_width, self.image_height)
                except ValueError:
                    self.selection_error = "Oriented ROI crosses the image edge; press R and redraw."
                return
            self.axis_start = original_point
            self.axis_end = original_point
            self.width_point = None
            self.axis_dragging = True
            self.selection_error = ""
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.axis_dragging:
                self.axis_end = original_point
            elif self.axis_start is not None and self.axis_end is not None:
                self.width_point = original_point
        elif event == cv2.EVENT_LBUTTONUP and self.axis_dragging:
            self.axis_end = original_point
            self.axis_dragging = False
            if self._candidate() is None and np.hypot(
                self.axis_end[0] - self.axis_start[0],
                self.axis_end[1] - self.axis_start[1],
            ) < 2:
                self.selection_error = "Centreline is too short; press R and redraw."

    def _render(self) -> np.ndarray:
        canvas = super()._render()
        candidate = self._candidate()
        if candidate is not None:
            points = np.asarray([self._original_to_screen(point) for point in candidate.corners], dtype=np.int32)
            cv2.polylines(canvas, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
        if self.axis_start is not None and self.axis_end is not None:
            cv2.line(
                canvas,
                self._original_to_screen(self.axis_start),
                self._original_to_screen(self.axis_end),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        header_height = 102 if self.batch_mode else 76
        cv2.rectangle(canvas, (0, 0), (self.canvas_width, header_height), (0, 0, 0), -1)
        cv2.putText(canvas, "Drag slightly beyond both track ends, then move outward and click for margin", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        controls = (
            "N next image | P previous image | K skip image | R reset | Q quit"
            if self.batch_mode
            else "Wheel zoom | right/middle-drag pan | R reset | Q quit"
        )
        cv2.putText(canvas, controls, (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        if self.batch_mode:
            cv2.putText(canvas, "Wheel zoom | right/middle-drag pan", (12, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        if self.selection_error:
            cv2.putText(canvas, self.selection_error, (12, 98 if self.batch_mode else 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
        return canvas

    def select(self) -> RotatedROI | str | None:
        self._reset_oriented_selection()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse)
        try:
            while True:
                cv2.imshow(self.window_name, self._render())
                key = cv2.waitKey(20) & 0xFF
                if self.batch_mode and key in (ord("n"), ord("N"), ord("p"), ord("P"), ord("k"), ord("K")):
                    self._reset_oriented_selection()
                    return chr(key).lower()
                if key in (ord("q"), ord("Q"), 27):
                    return None
                if key in (ord("r"), ord("R")):
                    self._reset_oriented_selection()
                elif key == ord("0"):
                    self.zoom = self.fit_zoom
                    self.offset_x = self.offset_y = 0.0
                elif key in (ord("+"), ord("=")):
                    self._zoom_at(self.canvas_width // 2, self.canvas_height // 2, 1.25)
                elif key in (ord("-"), ord("_")):
                    self._zoom_at(self.canvas_width // 2, self.canvas_height // 2, 0.8)
                if self.selected_rotated_roi is not None:
                    return self.selected_rotated_roi
        finally:
            cv2.destroyWindow(self.window_name)


def _fit_preview(image: np.ndarray, max_width: int = 1200, max_height: int = 800) -> np.ndarray:
    display = _display_bgr(image)
    height, width = display.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return display.copy()
    return cv2.resize(display, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class PreviewGeometry:
    """Display-only placement of a crop inside the preview canvas."""

    x: int
    y: int
    width: int
    height: int
    scale: float


@dataclass
class PreviewState:
    """Display-only crop preview options; never contains dataset pixels."""

    zoom_index: int = 0
    contrast_enabled: bool = False
    grayscale_enabled: bool = False

    ZOOM_LEVELS: ClassVar[tuple[float, ...]] = (1.0, 1.5, 2.0, 3.0)

    @property
    def magnification(self) -> float:
        return self.ZOOM_LEVELS[self.zoom_index]

    def cycle_zoom(self) -> None:
        self.zoom_index = (self.zoom_index + 1) % len(self.ZOOM_LEVELS)

    def restore_original(self) -> None:
        self.zoom_index = 0
        self.contrast_enabled = False
        self.grayscale_enabled = False

    def description(self) -> str:
        if self.contrast_enabled and self.grayscale_enabled:
            return "CONTRAST ENHANCED + GRAYSCALE (DISPLAY ONLY)"
        if self.contrast_enabled:
            return "CONTRAST ENHANCED (DISPLAY ONLY)"
        if self.grayscale_enabled:
            return "GRAYSCALE (DISPLAY ONLY)"
        return "ORIGINAL"


def calculate_crop_preview_geometry(
    crop_size: tuple[int, int],
    canvas_size: tuple[int, int] = (1000, 700),
    *,
    header_height: int = 138,
    margin: int = 24,
    magnification: float = 1.0,
) -> PreviewGeometry:
    """Fit the complete crop below the header while preserving aspect ratio."""

    crop_width, crop_height = crop_size
    canvas_width, canvas_height = canvas_size
    available_width = canvas_width - 2 * margin
    available_height = canvas_height - header_height - 2 * margin
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Preview crop dimensions must be positive")
    if not np.isfinite(magnification) or magnification <= 0:
        raise ValueError("Preview magnification must be positive")
    if available_width <= 0 or available_height <= 0:
        raise ValueError("Preview canvas is too small")

    scale = min(available_width / crop_width, available_height / crop_height) * magnification
    display_width = max(1, round(crop_width * scale))
    display_height = max(1, round(crop_height * scale))
    x = margin + (available_width - display_width) // 2
    y = header_height + margin + (available_height - display_height) // 2
    return PreviewGeometry(x, y, display_width, display_height, scale)


def _preview_uint8(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit display copy without mutating the extracted crop."""

    if image.dtype == np.uint8:
        return image.copy()
    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
        return cv2.convertScaleAbs(image, alpha=255.0 / maximum)
    return np.clip(image, 0.0, 1.0).astype(np.float32) * 255.0


def build_display_crop(crop: np.ndarray, state: PreviewState) -> np.ndarray:
    """Create a temporary BGR preview; ``crop`` remains byte-for-byte unchanged.

    Contrast mode applies conservative CLAHE to luminance only (clip limit 2.0,
    8x8 tiles). Grayscale and CLAHE outputs are strictly display artifacts.
    """

    if not isinstance(crop, np.ndarray) or crop.size == 0 or crop.ndim not in (2, 3):
        raise ValueError("Cannot preview an empty or invalid crop")
    display = _preview_uint8(crop).astype(np.uint8, copy=False)
    if display.ndim == 3 and display.shape[2] == 4:
        display = cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)

    if display.ndim == 3 and not state.grayscale_enabled:
        if state.contrast_enabled:
            lab = cv2.cvtColor(display, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(lightness)
            display = cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
        return display

    grayscale = display if display.ndim == 2 else cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    if state.contrast_enabled:
        grayscale = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale)
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def build_crop_preview_frame(
    crop: np.ndarray,
    label: str,
    canvas_size: tuple[int, int] = (1000, 700),
    state: PreviewState | None = None,
) -> np.ndarray:
    """Build a readable display-only preview without changing ``crop``."""

    preview_state = state or PreviewState()
    display_crop = build_display_crop(crop, preview_state)
    canvas_width, canvas_height = canvas_size
    geometry = calculate_crop_preview_geometry(
        (display_crop.shape[1], display_crop.shape[0]),
        canvas_size,
        magnification=preview_state.magnification,
    )
    interpolation = cv2.INTER_LINEAR if geometry.scale > 1.0 else cv2.INTER_AREA
    rendered_crop = cv2.resize(
        display_crop,
        (geometry.width, geometry.height),
        interpolation=interpolation,
    )
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    destination_left = max(0, geometry.x)
    destination_top = max(138, geometry.y)
    destination_right = min(canvas_width, geometry.x + geometry.width)
    destination_bottom = min(canvas_height, geometry.y + geometry.height)
    if destination_right > destination_left and destination_bottom > destination_top:
        source_left = destination_left - geometry.x
        source_top = destination_top - geometry.y
        source_right = source_left + destination_right - destination_left
        source_bottom = source_top + destination_bottom - destination_top
        canvas[destination_top:destination_bottom, destination_left:destination_right] = rendered_crop[
            source_top:source_bottom,
            source_left:source_right,
        ]

    label_name = {
        "alpha": "Alpha",
        "electron": "Electron",
        "uncertain": "Uncertain",
    }.get(label.strip().lower(), label.strip().title())
    cv2.putText(
        canvas,
        f"Current manual label: {label_name}",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Preview: {preview_state.description()} | Zoom: {preview_state.magnification:g}x",
        (24, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "DISPLAY ENHANCEMENT IS NOT SAVED",
        (24, 99),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "S save | Z zoom | C contrast | B grayscale | O original | U uncertain | R redraw | Q quit",
        (24, 127),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (max(0, geometry.x), max(138, geometry.y)),
        (min(canvas_width - 1, geometry.x + geometry.width - 1), min(canvas_height - 1, geometry.y + geometry.height - 1)),
        (255, 255, 255),
        1,
    )
    return canvas


def choose_manual_label(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    rotated_roi: RotatedROI | None = None,
) -> str | None:
    frame = _fit_preview(image)
    scale_x = frame.shape[1] / image.shape[1]
    scale_y = frame.shape[0] / image.shape[0]
    x, y, width, height = roi
    if rotated_roi is None:
        cv2.rectangle(frame, (round(x * scale_x), round(y * scale_y)), (round((x + width) * scale_x), round((y + height) * scale_y)), (0, 255, 0), 2)
    else:
        points = np.asarray(
            [(round(point_x * scale_x), round(point_y * scale_y)) for point_x, point_y in rotated_roi.corners],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(frame, "A Alpha | E Electron | U Uncertain | R redraw | Q quit", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    window = "Manual label"
    cv2.imshow(window, frame)
    try:
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("a"), ord("A")):
                return "alpha"
            if key in (ord("e"), ord("E")):
                return "electron"
            if key in (ord("u"), ord("U")):
                return "uncertain"
            if key in (ord("r"), ord("R")):
                return "redraw"
            if key in (ord("q"), ord("Q"), 27):
                return None
    finally:
        cv2.destroyWindow(window)


def preview_before_save(crop: np.ndarray, label: str) -> str:
    state = PreviewState()
    window = "Original-pixel crop preview"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            cv2.imshow(window, build_crop_preview_frame(crop, label, state=state))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("s"), ord("S")):
                return "save"
            if key in (ord("z"), ord("Z")):
                state.cycle_zoom()
                continue
            if key in (ord("c"), ord("C")):
                state.contrast_enabled = not state.contrast_enabled
                continue
            if key in (ord("b"), ord("B")):
                state.grayscale_enabled = not state.grayscale_enabled
                continue
            if key in (ord("o"), ord("O")):
                state.restore_original()
                continue
            if key in (ord("u"), ord("U")):
                return "uncertain"
            if key in (ord("r"), ord("R")):
                return "redraw"
            if key in (ord("q"), ord("Q"), 27):
                return "quit"
    finally:
        cv2.destroyWindow(window)


def is_save_confirmation(decision: str | None) -> bool:
    """Return true only for an explicit preview-save confirmation."""

    return decision == "save"


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    response = input(f"{text}{suffix}: ").strip()
    return response or default


def prompt_roi_mode() -> str | None:
    """Ask for an explicit geometric selection mode without affecting labels."""

    while True:
        response = _prompt("ROI mode: 1 axis-aligned, 2 rotated/oriented, Q quit", "1").lower()
        if response in {"1", "axis", "axis_aligned", "rect", "rectangle"}:
            return "axis_aligned"
        if response in {"2", "rotated", "oriented"}:
            return "rotated"
        if response in {"q", "quit"}:
            return None
        print("Choose 1 for axis-aligned, 2 for rotated/oriented, or Q to quit.")


def prompt_batch_action() -> str:
    """Ask whether to annotate the current image or navigate the batch."""

    while True:
        response = _prompt(
            "ROI/navigation: 1 axis-aligned, 2 rotated/oriented, N next, P previous, K skip, Q quit",
            "1",
        ).lower()
        if response in {"1", "axis", "axis_aligned", "rect", "rectangle"}:
            return "axis_aligned"
        if response in {"2", "rotated", "oriented"}:
            return "rotated"
        if response in {"n", "next"}:
            return "n"
        if response in {"p", "previous"}:
            return "p"
        if response in {"k", "skip"}:
            return "k"
        if response in {"q", "quit"}:
            return "q"
        print("Choose ROI mode 1/2, N next, P previous, K skip, or Q quit.")


def prompt_saved_frame_action() -> str:
    """Require an explicit finish decision in the resumable Lecturer queue."""

    while True:
        response = _prompt(
            "ROI/frame: 1 axis-aligned, 2 rotated/oriented, F finish frame, K skip unresolved, Q quit",
            "1",
        ).lower()
        if response in {"1", "axis", "axis_aligned", "rect", "rectangle"}:
            return "axis_aligned"
        if response in {"2", "rotated", "oriented"}:
            return "rotated"
        if response in {"f", "finish", "done"}:
            return "f"
        if response in {"k", "skip"}:
            return "k"
        if response in {"q", "quit"}:
            return "q"
        print("Choose ROI mode 1/2, F finish frame, K skip unresolved, or Q quit.")


@dataclass(frozen=True)
class AnnotationDefaults:
    source_dataset: str
    source_isotope: str
    physical_origin: str


def _prompt_annotation_defaults(image_path: Path, source_dataset: str) -> AnnotationDefaults:
    dataset = source_dataset or _prompt(
        "Source dataset",
        "AWAN" if image_path.name == AWAN_IMAGE_NAME else "",
    )
    return AnnotationDefaults(
        source_dataset=dataset,
        source_isotope=_prompt("Default source isotope (blank if panel-specific)", ""),
        physical_origin=_prompt("Default physical origin", "unknown"),
    )


def _read_source_image(image_path: Path) -> tuple[Path, np.ndarray]:
    resolved = Path(image_path).expanduser().resolve()
    image = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read source image: {resolved}")
    return resolved, image


def _run_image_session(
    image_path: Path,
    image: np.ndarray,
    output_root: Path,
    defaults: AnnotationDefaults,
    *,
    batch_mode: bool = False,
    saved_frame_mode: bool = False,
    before_save: Callable[[tuple[int, int, int, int], np.ndarray], bool] | None = None,
) -> str:
    """Annotate zero or more crops from one source, returning navigation only on request."""

    dataset = defaults.source_dataset
    default_isotope = defaults.source_isotope
    default_origin = defaults.physical_origin
    last_event_id = ""
    selector = ViewportROISelector(image, batch_mode=batch_mode)
    oriented_selector = OrientedROISelector(image, batch_mode=batch_mode)

    print("\n" + format_source_frame_warning(output_root / "labels.csv", image_path.name))
    print("\nNo class is inferred. Draw one complete track and choose A, E, or U manually.")
    print("Rotated crops use one perspective warp; geometric bilinear interpolation is unavoidable.")
    print("No separate resizing, enhancement, filtering, thresholding, or orientation normalization is applied.")
    while True:
        action = (
            prompt_saved_frame_action()
            if saved_frame_mode
            else prompt_batch_action() if batch_mode else prompt_roi_mode()
        )
        if action is None or action == "q":
            print("Annotation cancelled safely.")
            return "q"
        if action in {"n", "p", "k", "f"}:
            return action
        roi_mode = action
        rotated_roi = oriented_selector.select() if roi_mode == "rotated" else None
        if roi_mode == "rotated":
            if isinstance(rotated_roi, str):
                return rotated_roi
            if rotated_roi is None:
                print("Annotation cancelled safely.")
                return "q"
            roi = rotated_roi_bounding_box(rotated_roi, image.shape[1], image.shape[0])
        else:
            roi = selector.select()
            if isinstance(roi, str):
                return roi
        if roi is None:
            print("Annotation cancelled safely.")
            return "q"
        label = choose_manual_label(image, roi, rotated_roi)
        if label is None:
            print("Annotation cancelled safely.")
            return "q"
        if label == "redraw":
            continue
        crop = crop_original_pixels(image, roi) if rotated_roi is None else extract_rotated_crop(image, rotated_roi)
        while True:
            decision = preview_before_save(crop, label)
            if decision == "uncertain":
                label = "uncertain"
                continue
            break
        if decision == "quit":
            print("Annotation cancelled safely; current ROI was not saved.")
            return "q"
        if decision == "redraw":
            continue
        if not is_save_confirmation(decision):
            continue
        if before_save is not None and not before_save(roi, crop):
            continue

        metadata: dict[str, str] = {
            "source_isotope": default_isotope,
            "physical_origin": default_origin,
            "source_panel": "full",
        }
        if image_path.name == AWAN_IMAGE_NAME and dataset.upper() == "AWAN":
            roi_center_x = rotated_roi.center_x if rotated_roi else roi[0] + roi[2] / 2
            panel = "left" if roi_center_x < image.shape[1] / 2 else "right"
            metadata.update(source_panel=panel, **AWAN_PANEL_METADATA[panel])
            print(f"AWAN provenance: {panel} panel, {metadata['source_isotope']}, {metadata['physical_origin']}")
            print("This provenance did not assign the manual particle label.")

        event_entry = _prompt("Event ID (Enter generates unique; '.' reuses previous)", "")
        if event_entry == "." and last_event_id:
            event_id = last_event_id
        else:
            event_id = event_entry or create_event_id(dataset)
        quality_default = "low" if label == "uncertain" else "high"
        quality = _prompt("Ground-truth quality (high/medium/low)", quality_default).lower()
        if quality not in {"high", "medium", "low"}:
            print("Invalid quality; using the safe default.")
            quality = quality_default
        notes = _prompt("Notes (optional)", "")
        record = save_annotation(
            image=image,
            roi=roi,
            label=label,
            output_root=output_root,
            source_dataset=dataset,
            source_image=image_path.name,
            event_id=event_id,
            ground_truth_quality=quality,
            notes=notes,
            rotated_roi=rotated_roi,
            prepared_crop=crop,
            **metadata,
        )
        last_event_id = event_id
        print(f"Saved {record['crop_filename']} as {record['class']} ({record['event_id']}).")
        if saved_frame_mode:
            print("ANNOTATE ANOTHER TRACK FROM THIS FRAME: choose ROI mode 1/2.")
            print("FINISH THIS FRAME: choose F only after all tracks are annotated.")


def run_annotation(image_path: Path, output_root: Path, source_dataset: str = "") -> int:
    """Run the original single-image workflow."""

    image_path, image = _read_source_image(image_path)
    ensure_dataset_structure(output_root)
    defaults = _prompt_annotation_defaults(image_path, source_dataset)
    _run_image_session(image_path, image, output_root, defaults)
    return 0


def run_folder_annotation(
    folder: Path,
    output_root: Path,
    source_dataset: str = "",
    *,
    session_runner: Callable[[Path, np.ndarray, Path, AnnotationDefaults], str] | None = None,
) -> int:
    """Annotate a deterministically ordered folder without changing its files."""

    images = discover_image_files(folder)
    if not images:
        raise ValueError(f"No supported images found in directory: {Path(folder).expanduser().resolve()}")
    ensure_dataset_structure(output_root)
    defaults = _prompt_annotation_defaults(images[0], source_dataset)
    runner = session_runner or (
        lambda path, image, root, configured: _run_image_session(
            path,
            image,
            root,
            configured,
            batch_mode=True,
        )
    )
    index = 0
    while True:
        image_path, image = _read_source_image(images[index])
        print(f"\nSource: {image_path.name}")
        print(f"Image {index + 1} / {len(images)}")
        command = runner(image_path, image, output_root, defaults)
        next_index = navigate_image_index(index, len(images), command)
        if next_index is None:
            print("Batch annotation quit safely.")
            return 0
        if next_index == index:
            at_first = index == 0 and command.strip().lower() in {"p", "previous"}
            print(f"Already at the {'first' if at_first else 'last'} image.")
        index = next_index


def run_interactive(image_path: Path | None, output_root: Path, source_dataset: str = "") -> int:
    if image_path is None:
        entered = input("Source image path: ").strip().strip('"')
        if not entered:
            print("No source image supplied; nothing was changed.")
            return 0
        image_path = Path(entered)
    if Path(image_path).expanduser().is_dir():
        return run_folder_annotation(image_path, output_root, source_dataset)
    return run_annotation(image_path, output_root, source_dataset)
