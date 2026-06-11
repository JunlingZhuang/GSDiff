# -*- coding: utf-8 -*-
"""fix_room_counts: deterministic relabel / merge / split in Plan space."""
from hfagent.schema.plan import Plan, Room
from hfagent.tests.synth import rect
from hfagent.tools.plan_fixes import fix_room_counts


def test_relabel_surplus_to_missing():
    plan = Plan(rooms=[
        Room(id="a", type="exam_room", polygon=rect(0, 0, 4000, 4000)),
        Room(id="b", type="exam_room", polygon=rect(4000, 0, 3000, 4000)),  # smaller
    ])
    fixed, rep = fix_room_counts(plan, {"exam_room": 1, "office": 1})
    assert rep["fixed"], rep
    types = sorted(r.type for r in fixed.rooms)
    assert types == ["exam_room", "office"]
    # the smaller surplus room was retyped, geometry untouched
    assert next(r for r in fixed.rooms if r.type == "office").id == "b"
    assert rep["ops"] == ["relabel exam_room->office (b)"]


def test_merge_adjacent_surplus():
    plan = Plan(rooms=[
        Room(id="a", type="exam_room", polygon=rect(0, 0, 4000, 4000)),
        Room(id="b", type="exam_room", polygon=rect(4000, 0, 4000, 4000)),  # touches a
        Room(id="c", type="corridor", polygon=rect(0, 4000, 8000, 2000)),
    ])
    fixed, rep = fix_room_counts(plan, {"exam_room": 1, "corridor": 1})
    assert rep["fixed"], rep
    exams = [r for r in fixed.rooms if r.type == "exam_room"]
    assert len(exams) == 1
    # merged area ~= sum of both
    assert abs(exams[0].area() - 4000 * 4000 * 2) / (4000 * 4000 * 2) < 0.05


def test_split_when_missing():
    plan = Plan(rooms=[Room(id="a", type="patient_room", polygon=rect(0, 0, 8000, 4000))])
    fixed, rep = fix_room_counts(plan, {"patient_room": 2})
    assert rep["fixed"], rep
    rooms = [r for r in fixed.rooms if r.type == "patient_room"]
    assert len(rooms) == 2
    # split across the longer (x) axis into equal halves
    assert all(abs(r.area() - 8000 * 4000 / 2) < 1.0 for r in rooms)


def test_unfixable_reports_false():
    # surplus room not adjacent to any same-type room -> merge impossible
    plan = Plan(rooms=[
        Room(id="a", type="toilet", polygon=rect(0, 0, 2000, 2000)),
        Room(id="b", type="toilet", polygon=rect(9000, 9000, 2000, 2000)),  # far away
    ])
    fixed, rep = fix_room_counts(plan, {"toilet": 1})
    assert not rep["fixed"]
    assert len(fixed.rooms) == 2  # nothing corrupted


def test_already_exact_is_noop():
    plan = Plan(rooms=[Room(id="a", type="office", polygon=rect(0, 0, 3000, 3000))])
    fixed, rep = fix_room_counts(plan, {"office": 1})
    assert rep["fixed"] and rep["ops"] == []
    assert fixed.rooms[0].polygon == plan.rooms[0].polygon
