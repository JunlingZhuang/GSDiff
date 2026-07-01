# -*- coding: utf-8 -*-
"""Tool: rectify VLM-read room rectangles into a clean orthogonal Plan.

The structure-JSON path's deterministic replacement for cv_parse. Each room comes in as
one or more normalized rectangles (see structure_reader). We snap rectangle edges onto
shared lines plan-wide (so neighbours share exact edges — closing the small overlaps/gaps
the model leaves), union each room's rectangles into one orthogonal polygon, then scale to
pixels at the building's MEASURED aspect ratio.

The aspect ratio is measured from the realistic image (`building_aspect`), never taken from
the model: VLMs skew extents (they once read a full-width building as 66% wide), but the
non-white footprint in the image is exact. So the model only supplies the layout while the
output building keeps the true proportions. A bent corridor / L-shaped room (several rects)
keeps its shape.

Public API:
    building_aspect(real_png) -> float
    rectify(room_shapes, aspect, base_height=BASE_H) -> Plan
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from shapely.geometry import box
from shapely.ops import unary_union

from hfagent.schema.plan import Plan, Room
from hfagent.tools.structure_reader import RoomShape

BASE_H = 720.0    # output building height in px; width = aspect * BASE_H
SNAP_TOL = 0.015  # edge-cluster tolerance in normalized space


def building_aspect(real_png: bytes) -> float:
    """Width / height of the building footprint (non-white region) in the realistic image."""
    arr = np.asarray(Image.open(BytesIO(real_png)).convert("RGB"))
    ys, xs = np.where((arr < 240).any(axis=2))
    if len(xs) == 0:
        return 1.0
    h = ys.max() - ys.min()
    return float(xs.max() - xs.min()) / float(h) if h > 0 else 1.0


def _snap(values: list[float], tol: float) -> dict[float, float]:
    """Cluster sorted values within tol (anchored on the group's first), map each to its mean."""
    mapping: dict[float, float] = {}
    vals = sorted(set(values))
    group = [vals[0]]
    for v in vals[1:] + [float("inf")]:
        if v - group[0] <= tol:
            group.append(v)
        else:
            mean = sum(group) / len(group)
            mapping.update({g: mean for g in group})
            group = [v]
    return mapping


def _clean_ring(coords) -> list[tuple[float, float]]:
    """Drop the closing duplicate and any collinear vertices from a polygon ring."""
    pts = [(float(x), float(y)) for x, y in coords[:-1]]
    n = len(pts)
    keep = []
    for i in range(n):
        (ax, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[(i + 1) % n]
        if abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > 1e-9:
            keep.append((bx, by))
    return keep if len(keep) >= 3 else pts


def rectify(room_shapes: list[RoomShape], aspect: float, base_height: float = BASE_H) -> Plan:
    if not room_shapes:
        return Plan(units="px", rooms=[])

    # snap every rectangle edge onto shared lines so neighbours abut exactly
    sx = _snap([c for rs in room_shapes for r in rs.rects for c in (r[0], r[2])], SNAP_TOL)
    sy = _snap([c for rs in room_shapes for r in rs.rects for c in (r[1], r[3])], SNAP_TOL)

    # building bbox in normalized space -> force the measured aspect on output
    minx, maxx = min(sx.values()), max(sx.values())
    miny, maxy = min(sy.values()), max(sy.values())
    span_x, span_y = (maxx - minx) or 1.0, (maxy - miny) or 1.0
    out_w, out_h = aspect * base_height, base_height

    def to_px(x: float, y: float) -> tuple[float, float]:
        return ((x - minx) / span_x * out_w, (y - miny) / span_y * out_h)

    rooms: list[Room] = []
    for rs in room_shapes:
        boxes = [box(sx[x1], sy[y1], sx[x2], sy[y2]) for x1, y1, x2, y2 in rs.rects]
        merged = unary_union([b for b in boxes if not b.is_empty])
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)  # MVP: keep the largest block
        if merged.is_empty or merged.geom_type != "Polygon":
            continue
        polygon = [to_px(x, y) for x, y in _clean_ring(list(merged.exterior.coords))]
        if len(polygon) >= 3:
            rooms.append(Room(id=f"r{len(rooms) + 1}", type=rs.type, polygon=polygon))
    return Plan(units="px", rooms=rooms)
