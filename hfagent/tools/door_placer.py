# -*- coding: utf-8 -*-
"""Tool: place physical doors from the room adjacency graph.

The LLM gives only connectivity (room_a connects room_b). This turns that into
geometry: for every pair of reconstructed rooms whose types are connected in the
graph and which physically share a wall, hang a door on that shared wall, **centred**
on the overlap. Exterior edges hang a door on the room's longest outside wall.

Doors hang on the plan's real walls (`plan_to_walls`): each Door.wall_id points to an
actual Wall, `position` is the 0..1 parameter of the door centre along it. The
returned door→room-pair map lets the caller build the adjacency graph's `via` links.

Matching is by room TYPE (not instance): instance identity is lost in the colour
block, but "every patient_room–corridor shared wall gets a door" is exactly the
intent and needs no fragile instance disambiguation.

Public API:
    place_doors(plan, room_graph) -> (Plan, door_pairs)
        Plan: copy with walls + doors populated
        door_pairs: list[(door_id, room_a_id, room_b_id)]  # room_b_id may be "exterior"
"""
from __future__ import annotations

from shapely.geometry import Point, Polygon

from hfagent.schema.plan import Door, Plan, Room
from hfagent.tools.plan_to_walls import plan_to_walls
from hfagent.schema.roomgraph import RoomGraph

# door opening length as a fraction of the wall it sits on (centred)
DOOR_FRACTION = 0.34
# ignore slivers: a shared boundary shorter than this fraction of the smaller
# room's size is not a real adjacency
MIN_SHARE_FRACTION = 0.25


def _poly(room: Room) -> Polygon:
    p = Polygon(room.polygon)
    return p if p.is_valid else p.buffer(0)


def _shared_segment(a: Polygon, b: Polygon, tol: float):
    """Longest STRAIGHT wall shared by a and b → (p0, p1, length).

    Robust to the small gaps/overlaps left by rasterised parsing: instead of a
    strict polygon intersection (which needs the two rooms to touch EXACTLY), take
    the part of a's boundary running within tol of b, then pick the longest straight
    2-point segment — never the chord across a corner, which would be diagonal — so
    the door sits flat on one wall.
    """
    near = a.boundary.intersection(b.buffer(tol))
    if near.is_empty:
        return None
    geoms = list(near.geoms) if hasattr(near, "geoms") else [near]

    best = None  # (p0, p1, length)
    for line in geoms:
        if getattr(line, "geom_type", "") != "LineString":
            continue
        coords = list(line.coords)
        for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
            length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if length > 0 and (best is None or length > best[2]):
                best = ((float(x0), float(y0)), (float(x1), float(y1)), float(length))
    return best


def _type_lookup(room_graph: RoomGraph):
    by_id = {n.id: n.type for n in room_graph.rooms}
    return lambda rid: by_id.get(rid, rid.rsplit("_", 1)[0])


def _hang_on_wall(seg, walls):
    """Map a shared-wall segment to the real wall it lies on → (wall_id, position).

    The segment is (part of) a room-polygon edge, so it runs along one of the walls
    `plan_to_walls` produced. Pick the collinear wall nearest the segment midpoint;
    `position` is where that midpoint projects onto the wall (0..1).
    """
    (x0, y0), (x1, y1), length = seg
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = (x1 - x0) / length, (y1 - y0) / length

    best = None  # (dist, wall_id, position)
    for w in walls:
        (ax, ay), (bx, by) = w.start, w.end
        wl = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if wl == 0:
            continue
        ux, uy = (bx - ax) / wl, (by - ay) / wl
        if abs(ux * dx + uy * dy) < 0.98:        # not parallel → not this wall
            continue
        t = (mx - ax) * ux + (my - ay) * uy       # projection of midpoint along wall
        if not (-1e-6 <= t <= wl + 1e-6):         # midpoint must fall on the wall span
            continue
        px, py = ax + t * ux, ay + t * uy
        dist = ((mx - px) ** 2 + (my - py) ** 2) ** 0.5
        if best is None or dist < best[0]:
            best = (dist, w.id, max(0.0, min(1.0, t / wl)))
    if best is None:
        return None
    return best[1], best[2]


def place_doors(plan: Plan, room_graph: RoomGraph) -> tuple[Plan, list[tuple[str, str, str]]]:
    type_of = _type_lookup(room_graph)
    connected: set[frozenset] = set()   # {typeA, typeB} interior pairs
    exterior_types: set[str] = set()
    for e in room_graph.doors:
        if e.room_a == "exterior" or e.room_b == "exterior":
            other = e.room_b if e.room_a == "exterior" else e.room_a
            exterior_types.add(type_of(other))
        else:
            connected.add(frozenset((type_of(e.room_a), type_of(e.room_b))))

    out = plan.model_copy(deep=True)
    out.walls = plan_to_walls(out)   # explicit geometry-layer wall list; doors hang here

    polys = [(r, _poly(r)) for r in out.rooms]
    typical = sorted(p.area ** 0.5 for _, p in polys)
    median_size = typical[len(typical) // 2] if typical else 0.0
    min_share = MIN_SHARE_FRACTION * median_size
    adj_tol = max(2.0, 0.05 * median_size)  # bridge sub-grid gaps between parsed rooms

    doors: list[Door] = []
    door_pairs: list[tuple[str, str, str]] = []

    def hang(seg, room_a_id: str, room_b_id: str) -> None:
        hung = _hang_on_wall(seg, out.walls)
        if hung is None:
            return
        wall_id, position = hung
        did = f"d{len(doors)}"
        doors.append(Door(id=did, wall_id=wall_id, position=round(position, 4),
                          width=DOOR_FRACTION * seg[2]))
        door_pairs.append((did, room_a_id, room_b_id))

    # interior doors: connected type-pairs that share a wall
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            (ri, pi), (rj, pj) = polys[i], polys[j]
            if frozenset((ri.type, rj.type)) not in connected:
                continue
            seg = _shared_segment(pi, pj, adj_tol)
            if seg and seg[2] >= min_share:
                hang(seg, ri.id, rj.id)

    # exterior doors: centred on the longest outside wall of each exterior-typed room
    if exterior_types:
        eps = max(min_share * 0.1, 1e-6)
        for idx, (room, poly) in enumerate(polys):
            if room.type not in exterior_types:
                continue
            best = None
            pts = list(poly.exterior.coords)
            for k in range(len(pts) - 1):
                (x0, y0), (x1, y1) = pts[k], pts[k + 1]
                seg_len = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                if seg_len < min_share:
                    continue
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                # outward normal: the perpendicular direction that leaves this room
                nx, ny = -(y1 - y0) / seg_len, (x1 - x0) / seg_len
                if poly.contains(Point(mx + nx * eps, my + ny * eps)):
                    nx, ny = -nx, -ny
                probe = Point(mx + nx * eps, my + ny * eps)
                # exterior edge = nothing else lies just outside it
                if any(o != idx and op.contains(probe) for o, (_, op) in enumerate(polys)):
                    continue
                if best is None or seg_len > best[2]:
                    best = ((x0, y0), (x1, y1), seg_len)
            if best:
                hang(best, room.id, "exterior")

    out.doors = doors
    return out, door_pairs
