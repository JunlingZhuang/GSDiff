# -*- coding: utf-8 -*-
"""The boundary entry: program + building outline -> realflow plan.

Only pass-1 differs from the text entry; everything downstream of the realflow
plan (colour-block, parse, repair, doors, walls, adjacency) is identical.
"""
from hfagent.floor_plan_generate import generate_plan
from hfagent.tools.floor_plan_generator import build_boundary_prompt, build_real_prompt
from hfagent.tests.mock_vlm import MockVLM
from hfagent.tests.synth import simple_clinic

PROGRAM = {
    "building_type": "mock clinic",
    "rooms": [
        {"type": "waiting", "count": 1},
        {"type": "exam_room", "count": 2},
        {"type": "corridor", "count": 1},
        {"type": "toilet", "count": 1},
    ],
    "adjacency": [],
}


def test_boundary_prompt_states_the_outline_constraint():
    p = build_boundary_prompt(PROGRAM)
    assert "BOUNDARY" in p and "outline" in p.lower()
    assert "exam room" in p  # the room program is still embedded
    # shares the same drawing rules as the free-footprint prompt
    assert "TOP-DOWN" in p and "TOP-DOWN" in build_real_prompt(PROGRAM)


def test_boundary_entry_runs_pass1_then_shared_downstream(out_dir):
    client = MockVLM([simple_clinic()])
    r, plan, g = generate_plan(
        PROGRAM, client, out_dir, name="bound", max_rounds=1, boundary=b"\x89PNG fake-outline"
    )
    # pass-1 used the boundary prompt (the mock records every prompt string it gets)
    assert any("BOUNDARY" in f for f in client.feedbacks)
    # downstream identical to the text entry: unified plan + room graph produced
    assert (out_dir / "bound" / "plan.json").exists()
    assert len(g["rooms"]) == 5
    assert plan["walls"]  # full geometry layer still built
