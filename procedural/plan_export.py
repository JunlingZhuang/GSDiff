"""Derive a renderable/editable Plan (walls + openings) from procedural output.

The procedural generator emits room polygons. The CAD editor needs walls
(double-line) and openings (doors/passages). This module bridges the two:
shared polygon edges become interior walls, lone edges become exterior walls,
and graph access edges (door/passage/entrance) become openings on the shared
wall between the two rooms.
"""

from __future__ import annotations

import math

ROUND = 3  # coordinate rounding (mm) for coincident-edge matching


def _rounded_pt(p):
    """Return p rounded to ROUND decimal places as a tuple."""
    return (round(p[0], ROUND), round(p[1], ROUND))


def _seg_key(a, b):
    """Order-independent rounded key for an edge segment."""
    return tuple(sorted([_rounded_pt(a), _rounded_pt(b)]))


def derive_walls(rooms: dict[str, list[tuple[float, float]]],
                 thickness: float = 0.2) -> list[dict]:
    """Collect polygon edges across all rooms; dedup coincident edges.

    `rooms` maps room_id -> closed polygon (list of (x, y), first point repeated
    at the end). Returns a list of wall dicts: {id, a:[x,y], b:[x,y], thickness}.
    """
    seen: dict[tuple, dict] = {}
    for poly in rooms.values():
        for i in range(len(poly) - 1):
            a, b = poly[i], poly[i + 1]
            if _rounded_pt(a) == _rounded_pt(b):
                continue  # skip zero-length edge
            k = _seg_key(a, b)
            if k not in seen:
                seen[k] = {"a": [float(a[0]), float(a[1])],
                           "b": [float(b[0]), float(b[1])]}
    walls = []
    # ids are stable only within one call; callers must not rely on them across calls
    for i, (_, ab) in enumerate(seen.items()):
        walls.append({"id": f"w{i}", "a": ab["a"], "b": ab["b"], "thickness": thickness})
    return walls


ACCESS_KINDS = {"door", "passage", "entrance"}
OPENING_WIDTH = {"door": 0.9, "passage": 1.2, "entrance": 1.0, "window": 1.0}


def _room_edge_keys(poly: list[tuple[float, float]]) -> set:
    keys = set()
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        keys.add(_seg_key(a, b))
    return keys


def derive_openings(rooms: dict[str, list[tuple[float, float]]],
                    walls: list[dict],
                    edges: list[dict]) -> list[dict]:
    """One opening per access edge (door/passage/entrance) on the wall shared by
    the two rooms. wall-type edges produce nothing. If the two rooms share no
    coincident wall (adjacency not realised geometrically), the edge is skipped.
    """
    wall_by_key = {_seg_key(w["a"], w["b"]): w for w in walls}
    openings: list[dict] = []
    idx = 0
    for e in edges:
        kind = e.get("connectivity")
        if kind not in ACCESS_KINDS:
            continue
        ra, rb = e.get("source"), e.get("target")
        if ra not in rooms or rb not in rooms:
            continue
        shared = _room_edge_keys(rooms[ra]) & _room_edge_keys(rooms[rb])
        wall = None
        for k in shared:
            if k in wall_by_key:
                wall = wall_by_key[k]
                break
        if wall is None:
            continue  # adjacency not realised — designer fixes by hand later
        openings.append({
            "id": f"o{idx}",
            "wallId": wall["id"],
            "t": 0.5,
            "width": OPENING_WIDTH.get(kind, 0.9),
            "kind": "door" if kind in ("door", "entrance") else "passage",
        })
        idx += 1
    return openings


import networkx as nx
from shapely.geometry import Polygon

from procedural import generator, rules


def _poly_to_pts(poly: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in poly.exterior.coords]


def build_plan(graph: nx.Graph, boundary: Polygon, seed: int = 0) -> dict:
    """Run the procedural generator, then derive a Plan dict (walls/openings/
    rooms/grid). JSON-serializable; this is the HTTP response body.
    """
    layout = generator.generate(graph, boundary, seed=seed)
    angle = generator.dominant_angle(boundary)

    # rooms: stable string ids r{node}
    rooms_poly: dict[str, list[tuple[float, float]]] = {}
    room_meta: dict[str, dict] = {}
    for n, room in layout.rooms.items():
        if room.polygon.is_empty:
            continue
        rid = f"r{n}"
        pts = [(float(x), float(y)) for x, y in room.polygon.exterior.coords]
        rooms_poly[rid] = pts
        room_meta[rid] = {"type": room.room_type, "node": n}

    walls = derive_walls(rooms_poly, thickness=0.2)

    # graph edges keyed by the same r{node} ids
    edges = []
    for u, v, d in graph.edges(data=True):
        edges.append({"source": f"r{u}", "target": f"r{v}",
                      "connectivity": d.get("connectivity", "wall")})
    openings = derive_openings(rooms_poly, walls, edges)

    # attach wallIds to each room (walls whose segment lies on the room polygon)
    wall_by_key = {_seg_key(w["a"], w["b"]): w for w in walls}
    rooms_out = []
    for rid, pts in rooms_poly.items():
        wall_ids = []
        for k in _room_edge_keys(pts):
            w = wall_by_key.get(k)
            if w:
                wall_ids.append(w["id"])
        rooms_out.append({
            "id": rid,
            "type": room_meta[rid]["type"],
            "poly": [[float(x), float(y)] for x, y in pts],
            "wallIds": wall_ids,
        })

    return {
        "walls": walls,
        "openings": openings,
        "rooms": rooms_out,
        "grid": {"originX": 0.0, "originY": 0.0,
                 "spacingX": 1.0, "spacingY": 1.0, "angleDeg": float(angle)},
        "unit": "m",
    }
