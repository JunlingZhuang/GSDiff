# -*- coding: utf-8 -*-
from hfagent.schema.plan import Plan, Room
from hfagent.tests.synth import rect
from hfagent.tools.plan_to_walls import plan_to_walls


def test_plan_to_walls_unions_shared_t_junction_edges():
    plan = Plan(
        rooms=[
            Room(id="c", type="corridor", polygon=rect(0, 10, 100, 10)),
            Room(id="a", type="exam_room", polygon=rect(0, 0, 50, 10)),
            Room(id="b", type="exam_room", polygon=rect(50, 0, 50, 10)),
        ]
    )

    walls = plan_to_walls(plan, snap_tol=0.5)
    spans = {(w.start, w.end) for w in walls}

    assert len(walls) == 6
    assert ((0.0, 10.0), (100.0, 10.0)) in spans
    assert ((0.0, 0.0), (100.0, 0.0)) in spans


def test_plan_to_walls_snaps_slightly_misaligned_shared_edges():
    plan = Plan(
        rooms=[
            Room(id="l", type="exam_room", polygon=rect(0, 0, 50, 40)),
            Room(id="r", type="office", polygon=rect(50.4, 0, 50, 40)),
        ]
    )

    walls = plan_to_walls(plan, snap_tol=1.0)
    verticals = [w for w in walls if w.start[0] == w.end[0]]
    shared = [w for w in verticals if abs(w.start[0] - 50.2) < 1e-6]

    assert len(walls) == 5
    assert len(shared) == 1
    assert shared[0].start == (50.2, 0.0)
    assert shared[0].end == (50.2, 40.0)
