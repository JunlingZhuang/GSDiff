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
