# -*- coding: utf-8 -*-
"""Validate the correction-loop orchestration against a scripted mock VLM.
No API calls — deterministic. Disciplines under test:

  1. converge when the model improves on feedback
  2. violation feedback is actually sent to the VLM
  3. always keep the best (least-violating) round
  4. deterministic plan_fixes repairs what the VLM loop could not
  5. perfect first round skips all remaining rounds
  6. door extraction produces recon_with_door.png
"""
import copy

from hfagent.floor_plan_generate import generate_plan
from hfagent.schema.plan import Plan
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
GOOD = simple_clinic()  # waiting 1, exam 2, corridor 1, toilet 1 — matches PROGRAM


def bad_missing_exam() -> Plan:
    p = copy.deepcopy(GOOD)
    p.rooms[1].type = "office"  # one exam_room mislabelled -> exam 1/2 + office surplus
    return p


def worse() -> Plan:
    p = bad_missing_exam()
    p.rooms[4].type = "office"  # toilet also wrong -> 2 violations
    return p


def test_perfect_first_round_skips_remaining(out_dir):
    client = MockVLM([GOOD])
    r, _, _ = generate_plan(PROGRAM, client, out_dir, name="perfect", max_rounds=3)
    assert r["converged_in"] == 1
    assert len(r["rounds"]) == 1   # second round never ran
    assert client.calls == 2       # 2 image calls per round (real2color)


def test_converges_when_model_improves(out_dir):
    client = MockVLM([bad_missing_exam(), GOOD])
    r, _, _ = generate_plan(PROGRAM, client, out_dir, name="conv", max_rounds=3)
    assert r["converged_in"] == 2
    assert any("exam_room" in f for f in client.feedbacks)


def test_keeps_best_round_when_model_regresses(out_dir):
    # round 1: 1 violation (best), rounds 2-3: 2 violations (worse)
    client = MockVLM([bad_missing_exam(), worse(), worse()])
    r, _, _ = generate_plan(PROGRAM, client, out_dir, name="regress", max_rounds=3)
    assert r["best_round"] == 1
    assert not r["room_count_exact"]


def test_deterministic_fix_repairs_what_vlm_could_not(out_dir):
    client = MockVLM([bad_missing_exam(), worse(), worse(), worse()])
    r, _, _ = generate_plan(PROGRAM, client, out_dir, name="fix", max_rounds=4)
    assert not r["room_count_exact"]
    assert r["count_fix"]["fixed"]
    assert r["final_count_exact"]


def test_recon_with_door_is_produced(out_dir):
    """Pipeline always writes recon_with_door.png + graph.json, even with no doors."""
    client = MockVLM([GOOD])
    r, _, g = generate_plan(PROGRAM, client, out_dir, name="doors_none", max_rounds=1)
    assert (out_dir / "doors_none" / "recon_with_door.png").exists()
    assert (out_dir / "doors_none" / "graph.json").exists()
    # one room-graph node per room instance: waiting1 + exam2 + corridor1 + toilet1 = 5
    assert len(g["rooms"]) == 5


def test_room_adjacency_extracted_from_image(out_dir):
    """Scripted graph edges drive door placement; doors hang on real walls and the
    adjacency graph links each edge to a door (02-data-model §3.2)."""
    # exam_room has count 2 -> instances are exam_room_1 / exam_room_2
    door_data = [
        {"room_a": "exam_room_1", "room_b": "corridor"},
        {"room_a": "waiting", "room_b": "corridor"},
    ]
    client = MockVLM([GOOD], door_script=door_data)
    r, plan, g = generate_plan(PROGRAM, client, out_dir, name="doors_matched", max_rounds=1)
    assert (out_dir / "doors_matched" / "recon_with_door.png").exists()
    assert (out_dir / "doors_matched" / "plan.json").exists()
    assert r["room_graph"]["doors"] == 2                 # logical edges from the LLM
    assert r["room_graph"]["placed_doors"] >= 1          # doors hung on walls
    assert {"exam_room_1", "corridor"} in [set(p) for p in g_adjacency(g)]

    # unified plan: complete walls, every door hangs on a real wall
    wall_ids = {w["id"] for w in plan["walls"]}
    assert plan["walls"]
    assert all(d["wall_id"] in wall_ids for d in plan["doors"])
    # adjacency graph edges link to real doors via door id
    door_ids = {d["id"] for d in plan["doors"]}
    edges = plan["adjacency_graph"]["edges"]
    assert edges and all(e["via"] in door_ids for e in edges)
    assert plan["adjacency_graph"]["nodes"] == [room["id"] for room in plan["rooms"]]


def g_adjacency(graph_dict) -> list[tuple[str, str]]:
    return [tuple(sorted((d["room_a"], d["room_b"]))) for d in graph_dict["doors"]]
