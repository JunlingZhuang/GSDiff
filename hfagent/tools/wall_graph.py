"""Raster wall-segment primitives for the linework tracer.

Directional morphology extracts thick horizontal/vertical wall strokes from a
line-plan raster; the helpers here turn those masks into axis-aligned
``WallSegment`` centre lines (trace, cluster, merge, junction-snap). The
higher-level engine — door detection, bridging, post-processing, polygonize —
lives in ``tools/linework_tracer.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from shapely.geometry import LineString


@dataclass
class WallSegment:
    orientation: str
    axis: float
    start: float
    end: float
    thickness: float

    def line(self) -> LineString:
        if self.orientation == "horizontal":
            return LineString(((self.start, self.axis), (self.end, self.axis)))
        return LineString(((self.axis, self.start), (self.axis, self.end)))


def _estimate_stroke_thickness(dark: np.ndarray) -> int:
    """Thickness of the heaviest common stroke in the raw ink — the wall bands.

    Wall bands dominate a line plan's ink, so the 99th percentile of the ink's
    distance transform (x2) tracks the band thickness even with text/arc strokes
    present. Needed BEFORE the directional masks: their opening kernels must be
    longer than the wall thickness, or perpendicular wall bands leak into both
    masks (a drawing with fat walls then traces almost nothing).
    """
    distance = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    values = distance[distance > 0]
    if not values.size:
        return 3
    thickness = int(round(float(np.percentile(values, 99)) * 2.0))
    return max(3, min(thickness, round(min(dark.shape) * 0.06)))


def _directional_masks(dark: np.ndarray, min_len: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Long horizontal / vertical wall strokes. ``min_len`` floors the opening
    kernels; it must exceed the wall thickness so a perpendicular band cannot pass."""
    height, width = dark.shape
    horizontal = np.zeros_like(dark)
    vertical = np.zeros_like(dark)
    for divisor in (80, 50, 30):
        h_len = max(min_len, width // divisor)
        v_len = max(min_len, height // divisor)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        horizontal |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
        vertical |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
    return horizontal, vertical


def _estimate_wall_width(horizontal: np.ndarray, vertical: np.ndarray, image_shape,
                         max_width: int | None = None) -> int:
    combined = ((horizontal > 0) | (vertical > 0)).astype(np.uint8)
    distance = cv2.distanceTransform(combined, cv2.DIST_L2, 3)
    values = distance[distance > 0]
    fallback = max(3, round(min(image_shape) * 0.008))
    if not values.size:
        return fallback
    width = int(round(float(np.percentile(values, 82)) * 2.0))
    cap = max_width if max_width is not None else round(min(image_shape) * 0.03)
    return max(3, min(width, cap))


def _raw_segments(mask: np.ndarray, orientation: str, wall_width: int) -> list[WallSegment]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    minimum_length = max(8, wall_width * 2)
    minimum_cross = max(2, math.ceil(wall_width * 0.60))
    segments: list[WallSegment] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        length = width if orientation == "horizontal" else height
        cross = height if orientation == "horizontal" else width
        density = area / max(1, width * height)
        if length < minimum_length or cross < minimum_cross or density < 0.45:
            continue
        pixels_y, pixels_x = np.where(labels == index)
        if orientation == "horizontal":
            axis = float(np.median(pixels_y))
            segments.append(WallSegment(orientation, axis, float(x), float(x + width - 1), float(cross)))
        else:
            axis = float(np.median(pixels_x))
            segments.append(WallSegment(orientation, axis, float(y), float(y + height - 1), float(cross)))
    return segments


def _cluster_axes(segments: list[WallSegment], tolerance: float) -> list[WallSegment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda segment: segment.axis)
    groups: list[list[WallSegment]] = [[ordered[0]]]
    for segment in ordered[1:]:
        weighted_axis = sum(item.axis * max(1.0, item.end - item.start) for item in groups[-1]) / sum(
            max(1.0, item.end - item.start) for item in groups[-1]
        )
        if abs(segment.axis - weighted_axis) <= tolerance:
            groups[-1].append(segment)
        else:
            groups.append([segment])

    clustered: list[WallSegment] = []
    for group in groups:
        weights = [max(1.0, item.end - item.start) for item in group]
        axis = sum(item.axis * weight for item, weight in zip(group, weights)) / sum(weights)
        for item in group:
            clustered.append(WallSegment(
                item.orientation,
                axis,
                item.start,
                item.end,
                item.thickness,
            ))
    return clustered


def _merge_overlaps(segments: list[WallSegment], tolerance: float) -> list[WallSegment]:
    by_axis: dict[tuple[str, float], list[WallSegment]] = {}
    for segment in segments:
        by_axis.setdefault((segment.orientation, segment.axis), []).append(segment)

    merged: list[WallSegment] = []
    for (orientation, axis), items in by_axis.items():
        items.sort(key=lambda item: item.start)
        current = items[0]
        for item in items[1:]:
            if item.start <= current.end + tolerance:
                current.end = max(current.end, item.end)
                current.thickness = max(current.thickness, item.thickness)
            else:
                merged.append(current)
                current = item
        merged.append(current)
    return merged


def _snap_junctions(segments: list[WallSegment], tolerance: float) -> list[WallSegment]:
    horizontal = [segment for segment in segments if segment.orientation == "horizontal"]
    vertical = [segment for segment in segments if segment.orientation == "vertical"]
    for h_segment in horizontal:
        for v_segment in vertical:
            x, y = v_segment.axis, h_segment.axis
            if not (h_segment.start - tolerance <= x <= h_segment.end + tolerance):
                continue
            if not (v_segment.start - tolerance <= y <= v_segment.end + tolerance):
                continue
            if abs(h_segment.start - x) <= tolerance:
                h_segment.start = x
            if abs(h_segment.end - x) <= tolerance:
                h_segment.end = x
            if abs(v_segment.start - y) <= tolerance:
                v_segment.start = y
            if abs(v_segment.end - y) <= tolerance:
                v_segment.end = y
    return segments
