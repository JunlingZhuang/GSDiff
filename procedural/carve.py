"""Post-BSP carve pass: transfer area between graph-adjacent rooms.

Side effects:
  * Creates L-shaped rooms (BSP only outputs rectangles by construction).
  * Reduces R3 area-band violations (rooms that BSP made too big or small).

Operation:
  1. Find biggest R3 violator (room with area furthest from p50).
  2. If too big: pick a graph-neighbour that is too small; donor → recipient.
     If too small: pick a graph-neighbour that is too big.
  3. Carve a rectangular notch from donor's corner adjacent to recipient.
  4. Recipient absorbs the notch; if accept (area band improves), keep.

Preserves: 100% boundary coverage (sum of areas unchanged), 0 overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from . import rules

EPS = 0.02


@dataclass
class _EdgeInfo:
    side: str   # 'right' | 'left' | 'top' | 'bot' — which side of donor touches recipient
    lo: float   # overlap start along orthogonal axis
    hi: float   # overlap end


def _shared_edge(donor: Polygon, recipient: Polygon) -> _EdgeInfo | None:
    """Detect axis-aligned shared edge between two rectangles. Returns None if not adjacent."""
    a = donor.bounds
    b = recipient.bounds

    # donor right ↔ recipient left
    if abs(a[2] - b[0]) < EPS:
        lo, hi = max(a[1], b[1]), min(a[3], b[3])
        if hi - lo > EPS:
            return _EdgeInfo("right", lo, hi)
    # donor left ↔ recipient right
    if abs(a[0] - b[2]) < EPS:
        lo, hi = max(a[1], b[1]), min(a[3], b[3])
        if hi - lo > EPS:
            return _EdgeInfo("left", lo, hi)
    # donor top ↔ recipient bot
    if abs(a[3] - b[1]) < EPS:
        lo, hi = max(a[0], b[0]), min(a[2], b[2])
        if hi - lo > EPS:
            return _EdgeInfo("top", lo, hi)
    # donor bot ↔ recipient top
    if abs(a[1] - b[3]) < EPS:
        lo, hi = max(a[0], b[0]), min(a[2], b[2])
        if hi - lo > EPS:
            return _EdgeInfo("bot", lo, hi)
    return None


def _carve_rect(donor: Polygon, edge: _EdgeInfo, area: float) -> Polygon:
    """Build the carve rectangle: sits at one of donor's corners on the shared edge."""
    a = donor.bounds
    shared_len = edge.hi - edge.lo

    if edge.side in ("right", "left"):
        # carve has width w (into donor along x) and height h (along shared edge)
        h = min(shared_len, math.sqrt(2 * area))
        w = area / max(h, 1e-6)
        w = min(w, (a[2] - a[0]) * 0.45)
        y0 = edge.lo
        if edge.side == "right":
            return box(a[2] - w, y0, a[2], y0 + h)
        return box(a[0], y0, a[0] + w, y0 + h)

    # top / bot — carve has height h (into donor along y) and width w (along shared edge)
    w = min(shared_len, math.sqrt(2 * area))
    h = area / max(w, 1e-6)
    h = min(h, (a[3] - a[1]) * 0.45)
    x0 = edge.lo
    if edge.side == "top":
        return box(x0, a[3] - h, x0 + w, a[3])
    return box(x0, a[1], x0 + w, a[1] + h)


def try_carve(donor: Polygon, recipient: Polygon, area: float) -> tuple[Polygon, Polygon] | None:
    """Transfer `area` m² from donor to recipient via a corner notch.

    Both polygons should be reasonably rectangular (BSP output). Returns
    `(new_donor, new_recipient)` if the transfer succeeds (donor stays a single
    polygon with sane area, recipient stays a single polygon), else None.
    """
    edge = _shared_edge(donor, recipient)
    if edge is None:
        return None

    notch = _carve_rect(donor, edge, area)
    if notch.area < EPS:
        return None

    new_donor = donor.difference(notch)
    new_recipient = unary_union([recipient, notch])

    if new_donor.is_empty or new_donor.area < 0.5:
        return None
    if isinstance(new_donor, MultiPolygon):
        return None  # donor would split — refuse
    if isinstance(new_recipient, MultiPolygon):
        return None  # recipient would have a disconnected blob — refuse
    if not isinstance(new_donor, Polygon) or not isinstance(new_recipient, Polygon):
        return None

    return new_donor, new_recipient


def _r3_deviation(area: float, room_type: str) -> float:
    """How much (signed, normalized) the area deviates from p50."""
    target = rules.area_target(room_type)
    return (area - target) / max(target, 1e-6)


def carve_pass(layout, graph, max_iters: int = 30, min_transfer: float = 0.5) -> tuple:
    """Run carve loop on `layout`. Returns (layout, n_accepts)."""
    accepts = 0
    for _ in range(max_iters):
        # rank rooms by |R3 deviation| descending
        ranked = sorted(
            layout.rooms.items(),
            key=lambda kv: -abs(_r3_deviation(kv[1].polygon.area, kv[1].room_type)),
        )

        moved = False
        for n, r in ranked:
            dev_n = _r3_deviation(r.polygon.area, r.room_type)
            if abs(dev_n) < 0.10:
                break  # nothing significantly off — done

            neighbours = [m for m in graph.neighbors(n) if m in layout.rooms]
            if not neighbours:
                continue

            if dev_n > 0:
                # n is too big — donate to a smaller neighbour
                donor_id, recipient_id = n, None
                best_dev = 0
                for m in neighbours:
                    dev_m = _r3_deviation(layout.rooms[m].polygon.area, layout.rooms[m].room_type)
                    if dev_m < best_dev:
                        best_dev = dev_m
                        recipient_id = m
            else:
                # n is too small — receive from a bigger neighbour
                recipient_id, donor_id = n, None
                best_dev = 0
                for m in neighbours:
                    dev_m = _r3_deviation(layout.rooms[m].polygon.area, layout.rooms[m].room_type)
                    if dev_m > best_dev:
                        best_dev = dev_m
                        donor_id = m

            if donor_id is None or recipient_id is None:
                continue

            donor = layout.rooms[donor_id].polygon
            recipient = layout.rooms[recipient_id].polygon
            target_donor = rules.area_target(layout.rooms[donor_id].room_type)
            target_recipient = rules.area_target(layout.rooms[recipient_id].room_type)
            # transfer half of (donor surplus, recipient deficit), whichever is smaller
            amount = min(donor.area - target_donor,
                          target_recipient - recipient.area) * 0.5
            if amount < min_transfer:
                continue

            res = try_carve(donor, recipient, amount)
            if res is None:
                continue
            new_donor, new_recipient = res

            layout.rooms[donor_id].polygon = new_donor
            layout.rooms[recipient_id].polygon = new_recipient
            accepts += 1
            moved = True
            break  # restart from the new biggest violator

        if not moved:
            break

    return layout, accepts
