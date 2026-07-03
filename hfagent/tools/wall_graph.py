"""Raster wall-segment primitives for the linework tracer.

Directional morphology extracts thick horizontal/vertical wall strokes from a
line-plan raster; the helpers here turn those masks into axis-aligned
``WallSegment`` centre lines (trace, cluster, merge, junction-snap). The
higher-level engine — door detection, bridging, post-processing, polygonize —
lives in ``tools/linework_tracer.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def _solidify(dark: np.ndarray) -> np.ndarray:
    """Fuse hairline splits in the ink before tracing (measured-safe close).

    A small 3x3 close welds anti-aliasing seams and wall faces drawn a pixel or
    two apart into one solid band. The kernel is deliberately tiny and fixed:
    door leaves are drawn hollow with a ~6 px white core and letters sit ~8 px
    apart, so any larger close would weld door symbols / label glyphs into
    wall-thick marks (measured across the 2026-07 eval corpus).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)


def _estimate_stroke_thickness(dark: np.ndarray) -> int:
    """Thickness of the heaviest stroke in the raw ink — the thickest wall band.

    The 99th percentile of the ink's distance transform (x2) tracks the thickest
    band even with text/arc strokes present. Needed BEFORE the directional masks:
    their opening kernels must be longer than the thickest wall, or perpendicular
    wall bands leak into both masks (a drawing with fat walls then traces almost
    nothing).
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


def _median_wall_width(segments: list[WallSegment], image_shape) -> int:
    """Per-image wall width = MEDIAN cross thickness of the accepted directional runs.

    Image models draw wall thickness inconsistently — thick exteriors over thin
    interiors in one drawing — so any ink-mass percentile lands between the modes
    and poisons every gate derived from it. The per-segment median follows the
    dominant (interior partition) mode instead: partitions outnumber the few fat
    perimeter bands. Used ONLY for scale factors (door/gap sizes, stub reach,
    postprocess tolerances) — never as an acceptance gate on wall thickness.
    """
    fallback = max(3, round(min(image_shape) * 0.008))
    if not segments:
        return fallback
    width = int(round(float(np.median([segment.thickness for segment in segments]))))
    cap = round(min(image_shape) * 0.03)
    return max(3, min(width, cap))


def _raw_segments(mask: np.ndarray, orientation: str, min_cross: int) -> list[WallSegment]:
    """Directional runs accepted from a small ABSOLUTE thickness floor upward.

    Wall acceptance is thickness-agnostic: ``min_cross`` only needs to clear the
    thin symbol strokes (door arcs/leaves, ~2-4 px), NOT any global wall-width
    estimate — an 8 px partition and a 57 px perimeter band are both walls. Each
    run keeps its own measured cross thickness. A run must be longer than it is
    thick (walls are elongated) and solidly filled — except image-scale runs, where
    the density floor drops: a hollow double-line band with window slots (a common
    exterior style) fills only ~0.4 of its bbox, and nothing but a wall line ever
    produces a directional run a quarter of the image long.
    """
    span = mask.shape[1] if orientation == "horizontal" else mask.shape[0]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    segments: list[WallSegment] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        length = width if orientation == "horizontal" else height
        cross = height if orientation == "horizontal" else width
        density = area / max(1, width * height)
        min_density = 0.25 if length >= 0.25 * span else 0.45
        if length < max(8, round(1.5 * cross)) or cross < min_cross or density < min_density:
            continue
        pixels_y, pixels_x = np.where(labels == index)
        if orientation == "horizontal":
            axis = float(np.median(pixels_y))
            segments.append(WallSegment(orientation, axis, float(x), float(x + width - 1), float(cross)))
        else:
            axis = float(np.median(pixels_x))
            segments.append(WallSegment(orientation, axis, float(y), float(y + height - 1), float(cross)))
    return segments


def _ink_anchor(dark: np.ndarray, min_reach: float, min_area_frac: float,
                max_symbol_density: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ink component labels + two per-label verdicts: wall-network? door-symbol-like?

    The wall network of a drawing is usually one image-spanning component holding
    almost all the ink (walls touch walls); label text and stray symbols float in
    room interiors as small isolated marks. A component is wall-network ink when
    its bbox reach spans ``min_reach`` AND it holds ``min_area_frac`` of the
    largest component's ink — the area test catches long-but-light marks (a whole
    label word chained together by its underscores). Some drawings detach a whole
    BLOCK of rooms from the rest (a central core ringed by corridor): its ink can
    be well under the area fraction, but unlike a text line it spans ``min_reach``
    in BOTH bbox dimensions (text lines are glyph-high), so a 2D-spanning
    component is wall network regardless of its area share. Components are taken
    on a 1px-dilated copy so door symbols whose hinge kisses the wall across an
    anti-aliasing seam still count as wall-connected.

    Some styles draw door symbols fully DETACHED from the walls, so arc evidence
    cannot demand wall connection. A detached swing arc is a thin curve — its ink
    fills only a few percent of its bbox — where a label glyph fills a dense
    fraction: ``symbol_like`` marks sparse components (<= ``max_symbol_density``).
    """
    joined = cv2.dilate(dark, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    reach = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    spread = np.minimum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    area = stats[:, cv2.CC_STAT_AREA]
    bbox_area = np.maximum(1, stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT])
    largest = area[1:].max() if count > 1 else 0
    anchored = (reach >= min_reach) & ((area >= min_area_frac * largest)
                                       | (spread >= min_reach))
    anchored[0] = False
    # density from the UNDILATED ink: dilation triples a hairline stroke's area and
    # would make a thin arc read as dense as a glyph
    ink_area = np.bincount(labels[dark > 0], minlength=count)
    symbol_like = (~anchored) & (ink_area / bbox_area <= max_symbol_density)
    symbol_like[0] = False
    return labels, anchored, symbol_like


def _is_anchored(segment: WallSegment, anchor) -> bool:
    """True when the run's ink belongs to a wall-network component (see _ink_anchor).

    Every pixel of a directional run comes from ONE ink component, so the first ink
    sample inside the run's band identifies it. The probe scans the full cross
    thickness at each step because a hollow band's centre line can be white (its
    axis is the median of two face lines).
    """
    labels, anchored, _ = anchor
    height, width = labels.shape
    steps = np.linspace(segment.start, segment.end, num=7)
    half = max(1, int(round(segment.thickness / 2)))
    axis = int(round(segment.axis))
    band = range(max(0, axis - half), min(axis + half + 1,
                 height if segment.orientation == "horizontal" else width))
    for position in steps:
        p = int(round(position))
        for a in band:
            x, y = (p, a) if segment.orientation == "horizontal" else (a, p)
            if 0 <= x < width and 0 <= y < height and labels[y, x] > 0:
                return bool(anchored[labels[y, x]])
    return False


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
    """Snap segment endpoints onto crossing perpendicular axes (within tolerance).

    Each endpoint snaps to its NEAREST crossing axis, and a snap that would
    collapse the segment (both endpoints onto one axis) is refused: a short jog
    connector between two nearly-coaxial walls is itself shorter than the
    tolerance, and the naive per-pair snap zeroed it out, reopening the jog.
    """
    horizontal = [segment for segment in segments if segment.orientation == "horizontal"]
    vertical = [segment for segment in segments if segment.orientation == "vertical"]

    def snap(segment: WallSegment, crossers: list[WallSegment]) -> None:
        axes = [
            crosser.axis for crosser in crossers
            if segment.start - tolerance <= crosser.axis <= segment.end + tolerance
            and crosser.start - tolerance <= segment.axis <= crosser.end + tolerance
        ]
        if not axes:
            return
        new_start = min(axes, key=lambda a: abs(segment.start - a))
        new_end = min(axes, key=lambda a: abs(segment.end - a))
        start_ok = abs(segment.start - new_start) <= tolerance
        end_ok = abs(segment.end - new_end) <= tolerance
        if start_ok and end_ok and new_end - new_start < 1.0:
            # keep only the tighter snap - never collapse the segment
            if abs(segment.start - new_start) <= abs(segment.end - new_end):
                end_ok = False
            else:
                start_ok = False
        if start_ok and new_start < segment.end:
            segment.start = new_start
        if end_ok and new_end > segment.start:
            segment.end = new_end

    for h_segment in horizontal:
        snap(h_segment, vertical)
    for v_segment in vertical:
        snap(v_segment, horizontal)
    return segments
