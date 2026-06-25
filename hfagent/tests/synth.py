# -*- coding: utf-8 -*-
"""Synthetic ground-truth plans (mm) for round-trip fixtures."""
from hfagent.schema.plan import Plan, Room


def rect(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def simple_clinic() -> Plan:
    """2x2 grid + corridor: 5 rooms, all rectangles."""
    return Plan(
        plan_id="synth-simple",
        rooms=[
            Room(id="r1", type="waiting", polygon=rect(0, 0, 6000, 5000)),
            Room(id="r2", type="exam_room", polygon=rect(6000, 0, 6000, 5000)),
            Room(id="r3", type="corridor", polygon=rect(0, 5000, 12000, 2400)),
            Room(id="r4", type="exam_room", polygon=rect(0, 7400, 6000, 5000)),
            Room(id="r5", type="toilet", polygon=rect(6000, 7400, 6000, 5000)),
        ],
    )


def ward_wing() -> Plan:
    """Corridor spine with patient rooms both sides + nurse station; 8 rooms."""
    rooms = [Room(id="c", type="corridor", polygon=rect(0, 4000, 20000, 2400))]
    for i in range(3):
        rooms.append(Room(id=f"p{i}", type="patient_room", polygon=rect(i * 5000, 0, 5000, 4000)))
    rooms.append(Room(id="ns", type="nurse_station", polygon=rect(15000, 0, 5000, 4000)))
    for i in range(2):
        rooms.append(Room(id=f"q{i}", type="patient_room", polygon=rect(i * 7000, 6400, 7000, 4500)))
    rooms.append(Room(id="st", type="storage", polygon=rect(14000, 6400, 6000, 4500)))
    return Plan(plan_id="synth-ward", rooms=rooms)


def l_shape() -> Plan:
    """Contains one non-rectangular (L-shaped) room."""
    return Plan(
        plan_id="synth-lshape",
        rooms=[
            Room(
                id="r1",
                type="waiting",
                polygon=[(0, 0), (8000, 0), (8000, 4000), (4000, 4000), (4000, 8000), (0, 8000)],
            ),
            Room(id="r2", type="office", polygon=rect(4000, 4000, 4000, 4000)),
        ],
    )


ALL = {"simple_clinic": simple_clinic, "ward_wing": ward_wing, "l_shape": l_shape}


def make_plan_for_program(program: dict, cell: int = 5000) -> Plan:
    """Grid layout — one square cell per room instance — matching the program exactly.

    Used to build a 'perfect' MockVLM script for any programs.json entry without
    hand-writing a synthetic plan per program.
    """
    total = sum(r.get("count", 1) for r in program["rooms"])
    cols = max(3, int(total ** 0.5) + 1)
    col = row = 0
    rooms = []
    for r_spec in program["rooms"]:
        for _ in range(r_spec.get("count", 1)):
            rooms.append(Room(
                id=f"r{len(rooms)}",
                type=r_spec["type"],
                polygon=rect(col * cell, row * cell, cell, cell),
            ))
            col += 1
            if col >= cols:
                col = 0
                row += 1
    return Plan(plan_id="synth-prog", rooms=rooms)
