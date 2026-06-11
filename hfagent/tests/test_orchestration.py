# -*- coding: utf-8 -*-
"""Phase 0-B: validate the correction-loop ORCHESTRATION against a scripted
mock VLM (no API, deterministic). Disciplines under test (docs/agent §5.3):

  1. converge when the model improves on feedback
  2. quantified feedback is actually sent
  3. violations not strictly decreasing -> stop early, don't burn rounds
  4. always keep the best (least-violating) round
  5. deterministic plan_fixes still repairs what the loop could not
"""
import copy

from hfagent.run_phase0a import evaluate_one
from hfagent.schema.plan import Plan, Room
from hfagent.tests.mock_vlm import MockVLM
from hfagent.tests.synth import rect, simple_clinic

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
    p.rooms[4].type = "office"  # toilet also wrong -> 2 violations... and worse
    return p


def test_converges_when_model_improves(tmp_path):
    client = MockVLM([bad_missing_exam(), GOOD])
    r = evaluate_one("conv", PROGRAM, client, tmp_path, max_rounds=3)
    assert r["converged_in"] == 2
    assert not r["stopped_early"]
    # quantified feedback reached the model
    assert any("exam_room" in f for f in client.feedbacks)


def test_perfect_first_round(tmp_path):
    client = MockVLM([GOOD])
    r = evaluate_one("perfect", PROGRAM, client, tmp_path, max_rounds=3)
    assert r["converged_in"] == 1 and client.calls == 1


def test_oscillation_stops_early_and_keeps_best(tmp_path):
    # round1: 2 violations; rounds 2..n: 3 violations forever
    client = MockVLM([bad_missing_exam(), worse(), worse(), worse(), worse()])
    r = evaluate_one("osc", PROGRAM, client, tmp_path, max_rounds=5)
    assert r["stopped_early"], r
    assert len(r["rounds"]) == 3  # r2 no improvement (stall 1), r3 no improvement (stall 2) -> stop
    assert r["best_round"] == 1  # least-violating round kept, not the last one
    assert not r["room_count_exact"]


def test_deterministic_fix_repairs_what_loop_could_not(tmp_path):
    client = MockVLM([bad_missing_exam(), worse(), worse(), worse()])
    r = evaluate_one("fix", PROGRAM, client, tmp_path, max_rounds=4)
    # loop failed, but relabel fix (office surplus -> exam missing) succeeds
    assert not r["room_count_exact"]
    assert r["count_fix"]["fixed"], r["count_fix"]
    assert r["final_count_exact"]
