"""Skeletonisation and weighted principal-path extraction."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import cv2
import numpy as np
from skimage.morphology import skeletonize


NEIGHBOURS = [
    (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)), (1, 0, 1.0), (1, 1, math.sqrt(2.0)),
]


def skeletonise_mask(mask: np.ndarray) -> np.ndarray:
    """Return the largest one-pixel-wide connected skeleton component."""
    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.zeros(binary.shape, dtype=bool)
    skeleton = skeletonize(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), 8)
    if count <= 1:
        return skeleton
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _neighbours(node: tuple[int, int], nodes: set[tuple[int, int]]) -> Iterable[tuple[tuple[int, int], float]]:
    y, x = node
    for dy, dx, weight in NEIGHBOURS:
        adjacent = (y + dy, x + dx)
        if adjacent in nodes:
            yield adjacent, weight


def _shortest_paths(start: tuple[int, int], nodes: set[tuple[int, int]]):
    distances = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        for adjacent, weight in _neighbours(node, nodes):
            candidate = distance + weight
            if candidate < distances.get(adjacent, math.inf):
                distances[adjacent] = candidate
                previous[adjacent] = node
                heapq.heappush(queue, (candidate, adjacent))
    return distances, previous


def principal_skeleton_path(skeleton: np.ndarray) -> dict[str, object]:
    """Find the longest endpoint-to-endpoint weighted geodesic path."""
    nodes = set(map(tuple, np.argwhere(np.asarray(skeleton, dtype=bool))))
    if len(nodes) < 2:
        return {"processing_status": "failed", "error_message": "Skeleton has fewer than two pixels."}
    endpoints = [node for node in nodes if sum(1 for _ in _neighbours(node, nodes)) == 1]
    if len(endpoints) < 2:
        message = "Skeleton contains a loop with no valid endpoint pair." if not endpoints else "Skeleton has only one endpoint."
        return {"processing_status": "failed", "error_message": message, "endpoints": endpoints}
    if len(endpoints) > 200:
        return {"processing_status": "failed", "error_message": f"Skeleton is implausibly branch-heavy ({len(endpoints)} endpoints).", "endpoints": endpoints}

    longest = (-1.0, None, None, None)
    for index, start in enumerate(endpoints[:-1]):
        distances, previous = _shortest_paths(start, nodes)
        for end in endpoints[index + 1:]:
            distance = distances.get(end, -1.0)
            if distance > longest[0]:
                longest = (distance, start, end, previous)

    length, start, end, previous = longest
    if start is None or end is None or previous is None:
        return {"processing_status": "failed", "error_message": "No connected endpoint pair was found."}
    path = [end]
    while path[-1] != start:
        if path[-1] not in previous:
            return {"processing_status": "failed", "error_message": "Principal path reconstruction failed."}
        path.append(previous[path[-1]])
    path.reverse()
    return {"processing_status": "success", "error_message": None, "path": np.asarray(path, dtype=np.int32), "endpoints": [start, end], "all_endpoints": endpoints, "track_length_px": float(length)}
