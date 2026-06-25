# -*- coding: utf-8 -*-
"""Derive a canonical wall list from room polygons.

The parser reconstructs rooms independently, so shared boundaries often arrive
as overlapping or slightly offset polygon edges. Turning those edges directly
into walls makes large plans explode into hundreds of duplicate fragments. This
module snaps near-collinear rectilinear edges onto global axes, unions their
intervals, and emits one wall per maximal occupied segment.
"""
from __future__ import annotations

from collections import defaultdict

from hfagent.schema.plan import Plan, Wall

_MIN_LEN = 1e-6


def _auto_snap_tol(plan: Plan) -> float:
    minx, miny, maxx, maxy = plan.bounds()
    span = max(maxx - minx, maxy - miny)
    if span <= 0:
        return 1e-3
    return max(1e-3, span * 0.003)


def _axis_mapping(values: list[float], tol: float) -> dict[float, float]:
    if not values:
        return {}
    vals = sorted(set(float(v) for v in values))
    mapping: dict[float, float] = {}
    group = [vals[0]]
    for v in vals[1:] + [float("inf")]:
        if v - group[0] <= tol:
            group.append(v)
            continue
        snapped = sum(group) / len(group)
        for g in group:
            mapping[g] = snapped
        group = [v]
    return mapping


def _merge_intervals(intervals: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals if abs(b - a) > _MIN_LEN)
    if not ordered:
        return []

    merged: list[tuple[float, float]] = []
    start, end = ordered[0]
    for a, b in ordered[1:]:
        if a <= end + tol:
            end = max(end, b)
        else:
            merged.append((start, end))
            start, end = a, b
    merged.append((start, end))
    return merged


def plan_to_walls(plan: Plan, snap_tol: float | None = None) -> list[Wall]:
    """Return canonical rectilinear walls for all room boundaries.

    Non-axis-aligned edges are rare in this pipeline; they are still preserved
    with endpoint dedupe so malformed VLM polygons remain inspectable.
    """
    tol = _auto_snap_tol(plan) if snap_tol is None else snap_tol
    xs = [x for r in plan.rooms for x, _ in r.polygon]
    ys = [y for r in plan.rooms for _, y in r.polygon]
    xmap = _axis_mapping(xs, tol)
    ymap = _axis_mapping(ys, tol)

    horiz: dict[float, list[tuple[float, float]]] = defaultdict(list)
    vert: dict[float, list[tuple[float, float]]] = defaultdict(list)
    diagonals: dict[tuple[tuple[float, float], tuple[float, float]], tuple[float, float, float, float]] = {}

    for room in plan.rooms:
        pts = room.polygon
        for k in range(len(pts)):
            x0, y0 = pts[k]
            x1, y1 = pts[(k + 1) % len(pts)]
            x0, x1 = xmap.get(float(x0), float(x0)), xmap.get(float(x1), float(x1))
            y0, y1 = ymap.get(float(y0), float(y0)), ymap.get(float(y1), float(y1))
            if abs(x1 - x0) <= _MIN_LEN and abs(y1 - y0) <= _MIN_LEN:
                continue

            if abs(y1 - y0) <= tol:
                y = (y0 + y1) / 2.0
                horiz[y].append((x0, x1))
            elif abs(x1 - x0) <= tol:
                x = (x0 + x1) / 2.0
                vert[x].append((y0, y1))
            else:
                p0 = (round(x0, 6), round(y0, 6))
                p1 = (round(x1, 6), round(y1, 6))
                key = (p0, p1) if p0 <= p1 else (p1, p0)
                diagonals[key] = (x0, y0, x1, y1)

    walls: list[Wall] = []
    for y in sorted(horiz):
        for x0, x1 in _merge_intervals(horiz[y], tol):
            walls.append(Wall(id=f"w{len(walls)}", start=(x0, y), end=(x1, y)))
    for x in sorted(vert):
        for y0, y1 in _merge_intervals(vert[x], tol):
            walls.append(Wall(id=f"w{len(walls)}", start=(x, y0), end=(x, y1)))
    for x0, y0, x1, y1 in diagonals.values():
        walls.append(Wall(id=f"w{len(walls)}", start=(x0, y0), end=(x1, y1)))
    return walls
