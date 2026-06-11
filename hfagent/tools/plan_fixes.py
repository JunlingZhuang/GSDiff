# -*- coding: utf-8 -*-
"""Deterministic room-count repair in Plan (JSON) space.

Pixel-space correction (asking the VLM to edit the image) oscillates on complex
plans. Counts are a STRUCTURED constraint, so they are fixed here by code —
zero API cost, guaranteed termination, geometry untouched except where strictly
needed:

  relabel  surplus of type A while type B is missing -> retype the smallest
           surplus-A room to B (no geometry change)
  merge    surplus with nothing missing -> union two ADJACENT same-type rooms
  split    missing with no surplus anywhere -> cut the largest room of that
           type in half across its longer axis

Each operation strictly reduces total count mismatch, so the loop terminates.
"""
from __future__ import annotations

from collections import Counter

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from hfagent.schema.plan import Plan, Room


def _poly(room: Room) -> Polygon:
    p = Polygon(room.polygon)
    return p if p.is_valid else p.buffer(0)


def _ring(p: Polygon) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in p.exterior.coords[:-1]]


def fix_room_counts(plan: Plan, requested: dict[str, int]) -> tuple[Plan, dict]:
    """Returns (fixed_plan, report). report = {ops, fixed, fixed_rooms}."""
    rooms = [r.model_copy(deep=True) for r in plan.rooms]
    ops: list[str] = []

    def counts() -> Counter:
        return Counter(r.type for r in rooms)

    for _ in range(200):  # hard backstop; ops strictly reduce mismatch
        c = counts()
        missing = [t for t, want in requested.items() if c.get(t, 0) < want]
        surplus = [t for t in c if c[t] > requested.get(t, 0)]
        if not missing and not surplus:
            break

        if missing and surplus:  # relabel: free, try first
            st, mt = surplus[0], missing[0]
            cand = min((r for r in rooms if r.type == st), key=lambda r: r.area())
            cand.type = mt
            ops.append(f"relabel {st}->{mt} ({cand.id})")
            continue

        if surplus:  # merge two adjacent same-type rooms
            st = surplus[0]
            same = [r for r in rooms if r.type == st]
            pair = None
            for i in range(len(same)):
                for j in range(i + 1, len(same)):
                    pi = _poly(same[i]).buffer(1.0)
                    pj = _poly(same[j]).buffer(1.0)
                    if pi.intersects(pj):
                        u = unary_union([pi, pj]).buffer(-1.0)
                        if u.geom_type == "Polygon":
                            pair = (same[i], same[j], u)
                            break
                if pair:
                    break
            if not pair:
                break  # no adjacent pair to merge — cannot fix deterministically
            a, b, u = pair
            a.polygon = _ring(u.simplify(0.5))
            rooms.remove(b)
            ops.append(f"merge {st} ({a.id}+{b.id})")
            continue

        # missing only: split the largest room of that type across its longer axis
        mt = missing[0]
        cands = [r for r in rooms if r.type == mt]
        if not cands:
            break
        r0 = max(cands, key=lambda r: r.area())
        p = _poly(r0)
        minx, miny, maxx, maxy = p.bounds
        if maxx - minx >= maxy - miny:
            mid = (minx + maxx) / 2
            h1, h2 = box(minx, miny, mid, maxy), box(mid, miny, maxx, maxy)
        else:
            mid = (miny + maxy) / 2
            h1, h2 = box(minx, miny, maxx, mid), box(minx, mid, maxx, maxy)
        pa, pb = p.intersection(h1), p.intersection(h2)
        if pa.geom_type != "Polygon" or pb.geom_type != "Polygon" or pa.area < 1 or pb.area < 1:
            break  # exotic shape — refuse rather than corrupt geometry
        r0.polygon = _ring(pa)
        rooms.append(Room(id=f"{r0.id}b", type=mt, polygon=_ring(pb)))
        ops.append(f"split {mt} ({r0.id})")

    final = counts()
    fixed = all(final.get(t, 0) == n for t, n in requested.items()) and all(
        t in requested for t in final
    )
    out = plan.model_copy(deep=True)
    out.rooms = rooms
    return out, {"ops": ops, "fixed": fixed, "fixed_rooms": dict(final)}
