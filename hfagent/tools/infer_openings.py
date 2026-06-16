# -*- coding: utf-8 -*-
"""Door inference from room adjacency — plan docs/agent §11.2.1 strategy C.

Doors are NEVER parsed from pixels (literature: 0.09-0.39 exact match). After
cv_parse + plan_fixes produce clean room polygons, adjacency is computed
deterministically (two rooms sharing a long-enough boundary segment), and doors
are placed by rules:

  * every non-circulation room gets a door to an adjacent corridor (longest
    shared segment wins); waiting counts as circulation fallback
  * a room with no circulation neighbour opens to its largest-boundary
    neighbour (e.g. en-suite toilet -> patient room)
  * corridor-corridor and corridor-waiting junctions get a passage

Each door hangs on a party-wall segment (relational modelling: wall_id +
position 0..1 + width), so geometry edits keep doors attached.
"""
from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import linemerge

from hfagent.schema.plan import Door, Plan, Room, Wall

CIRCULATION = ("corridor", "waiting")

# leaf widths in mm by the room the door SERVES (healthcare-ish defaults)
DOOR_WIDTH_MM = {
    "patient_room": 1200.0, "exam_room": 1000.0, "nurse_station": 1000.0,
    "office": 900.0, "toilet": 900.0, "storage": 900.0,
    "waiting": 1500.0, "corridor": 1200.0,
}


def _poly(room: Room) -> Polygon:
    p = Polygon(room.polygon)
    return p if p.is_valid else p.buffer(0)


def _shared_segment(a: Polygon, b: Polygon) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Longest straight piece of the boundary two rooms share, or None."""
    inter = a.intersection(b)
    if inter.is_empty or inter.length == 0:
        return None
    merged = linemerge(inter) if inter.geom_type == "MultiLineString" else inter
    lines = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    lines = [l for l in lines if l.geom_type == "LineString" and l.length > 0]
    if not lines:
        return None
    seg = max(lines, key=lambda l: l.length)
    (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
    return (float(x0), float(y0)), (float(x1), float(y1)), float(seg.length)


def infer_openings(plan: Plan, min_shared: float | None = None) -> Plan:
    """Returns a copy of `plan` with walls (party-wall segments) and doors
    populated from adjacency rules. `min_shared`: minimum shared boundary
    length to count as adjacency (defaults to ~1.2x the widest door)."""
    out = plan.model_copy(deep=True)
    mm = plan.units == "mm"
    polys = [(r, _poly(r)) for r in out.rooms]

    if min_shared is None:
        if mm:
            min_shared = 1000.0
        else:  # unit-agnostic: fraction of the typical room size
            sizes = sorted(p.area ** 0.5 for _, p in polys)
            min_shared = 0.35 * sizes[len(sizes) // 2]

    # adjacency: (i, j) -> shared wall segment
    adj: dict[tuple[int, int], tuple] = {}
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            seg = _shared_segment(polys[i][1], polys[j][1])
            if seg and seg[2] >= min_shared:
                adj[(i, j)] = seg

    def neighbours(i: int):
        for (a, b), seg in adj.items():
            if a == i:
                yield b, seg
            elif b == i:
                yield a, seg

    walls: list[Wall] = []
    doors: list[Door] = []
    done: set[tuple[int, int]] = set()

    def add_door(i: int, j: int, seg, kind: str, serve_type: str) -> None:
        key = (min(i, j), max(i, j))
        if key in done:
            return
        done.add(key)
        p0, p1, length = seg
        wall = Wall(id=f"w{len(walls) + 1}", start=p0, end=p1, thickness=200.0 if mm else 0.0)
        width_mm = DOOR_WIDTH_MM.get(serve_type, 900.0)
        width = min(width_mm if mm else 0.3 * length, 0.8 * length)
        walls.append(wall)
        doors.append(Door(id=f"d{len(doors) + 1}", wall_id=wall.id, position=0.5, width=width, type=kind))

    for i, (room, _) in enumerate(polys):
        nbrs = list(neighbours(i))
        if not nbrs:
            continue
        if room.type == "corridor":
            # junctions between circulation spaces become open passages
            for j, seg in nbrs:
                if polys[j][0].type in CIRCULATION:
                    add_door(i, j, seg, "passage", "corridor")
            continue
        # prefer corridor, then waiting, else largest-boundary neighbour
        for want in ("corridor", "waiting", None):
            cands = [(j, seg) for j, seg in nbrs if want is None or polys[j][0].type == want]
            if cands:
                j, seg = max(cands, key=lambda c: c[1][2])
                add_door(i, j, seg, "door", room.type)
                break

    out.walls = walls
    out.doors = doors
    return out
