# -*- coding: utf-8 -*-
"""infer_openings: doors from adjacency rules (strategy C), never from pixels."""
from shapely.geometry import LineString, Polygon

from hfagent.tests.synth import simple_clinic, ward_wing
from hfagent.tools.infer_openings import infer_openings


def _door_segment(plan, door):
    w = {w.id: w for w in plan.walls}[door.wall_id]
    (x0, y0), (x1, y1) = w.start, w.end
    mx, my = x0 + (x1 - x0) * door.position, y0 + (y1 - y0) * door.position
    L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    h = door.width / 2
    return LineString([(mx - ux * h, my - uy * h), (mx + ux * h, my + uy * h)])


def test_every_room_reaches_circulation():
    plan = infer_openings(simple_clinic())
    rooms_with_door = set()
    for d in plan.doors:
        seg = _door_segment(plan, d)
        for r in plan.rooms:
            if Polygon(r.polygon).buffer(1.0).intersection(seg).length > d.width * 0.9:
                rooms_with_door.add(r.id)
    non_circ = [r for r in plan.rooms if r.type not in ("corridor", "waiting")]
    for r in non_circ:
        assert r.id in rooms_with_door, f"{r.id} ({r.type}) has no door"


def test_doors_hang_on_valid_walls():
    plan = infer_openings(ward_wing())
    wall_ids = {w.id for w in plan.walls}
    assert plan.doors, "no doors inferred"
    for d in plan.doors:
        assert d.wall_id in wall_ids
        assert 0.0 <= d.position <= 1.0
        w = {w.id: w for w in plan.walls}[d.wall_id]
        L = ((w.end[0] - w.start[0]) ** 2 + (w.end[1] - w.start[1]) ** 2) ** 0.5
        assert d.width <= L, "door wider than its wall"


def test_door_segment_lies_on_shared_boundary():
    plan = infer_openings(simple_clinic())
    for d in plan.doors:
        seg = _door_segment(plan, d)
        touching = sum(
            1 for r in plan.rooms
            if Polygon(r.polygon).buffer(1.0).intersection(seg).length > d.width * 0.9
        )
        assert touching >= 2, "door not on a party wall between two rooms"


def test_geometry_untouched():
    src = simple_clinic()
    plan = infer_openings(src)
    assert [r.polygon for r in plan.rooms] == [r.polygon for r in src.rooms]
