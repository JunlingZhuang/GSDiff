"""Deterministic Manhattan wall graph reconstruction from a line-plan raster.

Directional morphology is used only to extract thick horizontal/vertical wall
strokes. Gaps are classified individually from local residual ink; no global
closing operation is applied. The resulting center-line segments form a planar
graph that Shapely polygonizes into room regions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union


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


@dataclass
class OpeningCandidate:
    orientation: str
    axis: float
    start: float
    end: float
    kind: str
    residual_pixels: int

    @property
    def center(self) -> tuple[float, float]:
        middle = (self.start + self.end) / 2.0
        return (middle, self.axis) if self.orientation == "horizontal" else (self.axis, middle)


@dataclass
class WallGraphResult:
    segments: list[WallSegment]
    openings: list[OpeningCandidate]
    polygons: list[Polygon]
    wall_width: int
    dark: np.ndarray
    wall_pixels: np.ndarray


def _directional_masks(dark: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = dark.shape
    horizontal = np.zeros_like(dark)
    vertical = np.zeros_like(dark)
    for divisor in (80, 50, 30):
        h_len = max(15, width // divisor)
        v_len = max(15, height // divisor)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        horizontal |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
        vertical |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
    return horizontal, vertical


def _estimate_wall_width(horizontal: np.ndarray, vertical: np.ndarray, image_shape) -> int:
    combined = ((horizontal > 0) | (vertical > 0)).astype(np.uint8)
    distance = cv2.distanceTransform(combined, cv2.DIST_L2, 3)
    values = distance[distance > 0]
    fallback = max(3, round(min(image_shape) * 0.008))
    if not values.size:
        return fallback
    width = int(round(float(np.percentile(values, 82)) * 2.0))
    return max(3, min(width, round(min(image_shape) * 0.03)))


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


def _residual_ink(
    dark: np.ndarray,
    wall_pixels: np.ndarray,
    orientation: str,
    axis: float,
    start: float,
    end: float,
    wall_width: int,
) -> tuple[int, int]:
    height, width = dark.shape
    span = max(1, int(round(end - start)))
    radius = max(span, wall_width * 3)
    if orientation == "horizontal":
        x0, x1 = max(0, round(start - wall_width)), min(width, round(end + wall_width + 1))
        y0, y1 = max(0, round(axis - radius)), min(height, round(axis + radius + 1))
        strip_y0, strip_y1 = max(0, round(axis - wall_width)), min(height, round(axis + wall_width + 1))
        strip = (slice(strip_y0, strip_y1), slice(x0, x1))
    else:
        x0, x1 = max(0, round(axis - radius)), min(width, round(axis + radius + 1))
        y0, y1 = max(0, round(start - wall_width)), min(height, round(end + wall_width + 1))
        strip_x0, strip_x1 = max(0, round(axis - wall_width)), min(width, round(axis + wall_width + 1))
        strip = (slice(y0, y1), slice(strip_x0, strip_x1))
    residual = ((dark[y0:y1, x0:x1] > 0) & (wall_pixels[y0:y1, x0:x1] == 0)).astype(np.uint8)
    total = int(residual.sum())
    aligned = int(((dark[strip] > 0) & (wall_pixels[strip] == 0)).sum())
    return total, aligned


def _bridge_symbol_gaps(
    dark: np.ndarray,
    wall_pixels: np.ndarray,
    segments: list[WallSegment],
    wall_width: int,
) -> tuple[list[WallSegment], list[OpeningCandidate]]:
    maximum_gap = wall_width * 12
    by_axis: dict[tuple[str, float], list[WallSegment]] = {}
    for segment in segments:
        by_axis.setdefault((segment.orientation, segment.axis), []).append(segment)

    output: list[WallSegment] = []
    openings: list[OpeningCandidate] = []
    for (orientation, axis), items in by_axis.items():
        items.sort(key=lambda item: item.start)
        current = items[0]
        for item in items[1:]:
            gap_start, gap_end = current.end, item.start
            gap = gap_end - gap_start
            if gap <= wall_width * 0.75:
                current.end = max(current.end, item.end)
                continue
            bridge = False
            kind = "passage"
            residual = aligned = 0
            if gap <= maximum_gap:
                residual, aligned = _residual_ink(
                    dark, wall_pixels, orientation, axis, gap_start, gap_end, wall_width
                )
                perpendicular = max(0, residual - aligned)
                if perpendicular >= max(5, round(gap * 0.20)):
                    bridge, kind = True, "door"
                elif aligned >= max(5, round(gap * 0.25)):
                    bridge, kind = True, "window"
            if bridge:
                openings.append(OpeningCandidate(
                    orientation, axis, gap_start, gap_end, kind, residual
                ))
                current.end = item.end
                current.thickness = max(current.thickness, item.thickness)
            else:
                output.append(current)
                current = item
        output.append(current)
    return output, openings


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


def _polygonize(segments: list[WallSegment], image_shape, wall_width: int) -> list[Polygon]:
    if not segments:
        return []
    network = unary_union([segment.line() for segment in segments])
    minimum_area = max(100.0, image_shape[0] * image_shape[1] * 0.00035)
    polygons = []
    for polygon in polygonize(network):
        if polygon.area < minimum_area:
            continue
        simplified = polygon.simplify(max(0.5, wall_width * 0.15), preserve_topology=True)
        if simplified.geom_type == "Polygon":
            polygons.append(simplified)
    return sorted(polygons, key=lambda polygon: (polygon.centroid.y, polygon.centroid.x))


def reconstruct_wall_graph(gray: np.ndarray) -> WallGraphResult:
    """Extract wall center lines, classify local gaps and polygonize the graph."""
    _, dark = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_mask, vertical_mask = _directional_masks(dark)
    wall_width = _estimate_wall_width(horizontal_mask, vertical_mask, gray.shape)

    raw = _raw_segments(horizontal_mask, "horizontal", wall_width)
    raw += _raw_segments(vertical_mask, "vertical", wall_width)
    clustered = _cluster_axes(raw, tolerance=max(1.0, wall_width * 0.5))
    merged = _merge_overlaps(clustered, tolerance=max(1.0, wall_width * 0.5))

    wall_pixels = ((horizontal_mask > 0) | (vertical_mask > 0)).astype(np.uint8)
    bridged, openings = _bridge_symbol_gaps(dark, wall_pixels, merged, wall_width)
    snapped = _snap_junctions(bridged, tolerance=max(1.0, wall_width * 0.5))
    polygons = _polygonize(snapped, gray.shape, wall_width)
    return WallGraphResult(snapped, openings, polygons, wall_width, dark, wall_pixels)
