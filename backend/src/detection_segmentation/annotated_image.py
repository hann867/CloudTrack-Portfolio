"""Final accepted-component annotation."""

import cv2


def annotate_components(source_bgr, contour_records):
    """Draw only 2-pixel red component bounding boxes on a source-image copy."""
    annotated = source_bgr.copy()
    for record in contour_records:
        x, y = record["bbox_x"], record["bbox_y"]
        width, height = record["bbox_width"], record["bbox_height"]
        cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), (0, 0, 255), 2)
    return annotated
