"""Shared raster-to-orthogonal-polygon helpers for structure parsers."""
from __future__ import annotations

import cv2
import numpy as np


def grid_contour(mask: np.ndarray, grid: int) -> np.ndarray | None:
    """Trace a component boundary on a coarse occupancy grid."""
    h, w = mask.shape
    small = cv2.resize(
        mask.astype(np.float32),
        (max(1, w // grid), max(1, h // grid)),
        interpolation=cv2.INTER_AREA,
    )
    small = (small > 0.10).astype(np.uint8)
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < 3:
        return None
    points: list[tuple[float, float]] = []
    for index in range(len(contour)):
        a, b = contour[index], contour[(index + 1) % len(contour)]
        points.append((a[0], a[1]))
        if a[0] != b[0] and a[1] != b[1]:
            points.append((b[0], a[1]))
    return (np.array(points) + 0.5) * grid


def snap_axes(polygons: list[np.ndarray], tolerance: float) -> None:
    """Snap nearby x/y coordinates plan-wide without chained clusters."""
    if not polygons:
        return
    for axis in (0, 1):
        values = np.sort(np.unique(np.concatenate([polygon[:, axis] for polygon in polygons])))
        mapping: dict[float, float] = {}
        group: list[float] = [float(values[0])]
        for value in list(values[1:]) + [float("inf")]:
            if value - group[0] <= tolerance:
                group.append(float(value))
            else:
                mean = float(np.mean(group))
                mapping.update({member: mean for member in group})
                group = [float(value)]
        for polygon in polygons:
            polygon[:, axis] = [mapping[float(value)] for value in polygon[:, axis]]


def clean_ring(points: np.ndarray) -> np.ndarray:
    """Remove duplicate and collinear vertices from a polygon ring."""
    deduplicated: list[np.ndarray] = []
    for point in points:
        if deduplicated and np.allclose(point, deduplicated[-1]):
            continue
        deduplicated.append(point)
    points = np.array(deduplicated)
    keep = []
    for index in range(len(points)):
        a, b, c = points[(index - 1) % len(points)], points[index], points[(index + 1) % len(points)]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > 1e-6:
            keep.append(b)
    return np.array(keep) if len(keep) >= 3 else points
