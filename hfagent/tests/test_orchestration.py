# -*- coding: utf-8 -*-
"""Validate the correction-loop orchestration against a scripted mock VLM.
No API calls — deterministic. Disciplines under test:

  1. converge when the model improves on feedback
  2. violation feedback is actually sent to the VLM
  3. always keep the best (least-violating) round
  4. deterministic plan_fixes repairs what the VLM loop could not
  5. perfect first round skips all remaining rounds
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


def test_perfect_first_round_skips_remaining(tmp_path):
    client = MockVLM([GOOD])
    r, _, _ = generate_plan(PROGRAM, client, tmp_path, name="perfect", max_rounds=3)
    assert r["converged_in"] == 1
    assert len(r["rounds"]) == 1   # second round never ran
    assert client.calls == 2       # 2 image calls per round (real2color)


def test_converges_when_model_improves(tmp_path):
    client = MockVLM([bad_missing_exam(), GOOD])
    r, _, _ = generate_plan(PROGRAM, client, tmp_path, name="conv", max_rounds=3)
    assert r["converged_in"] == 2
    assert any("exam_room" in f for f in client.feedbacks)


def test_keeps_best_round_when_model_regresses(tmp_path):
    # round 1: 1 violation (best), rounds 2-3: 2 violations (worse)
    client = MockVLM([bad_missing_exam(), worse(), worse()])
    r, _, _ = generate_plan(PROGRAM, client, tmp_path, name="regress", max_rounds=3)
    assert r["best_round"] == 1
    assert not r["room_count_exact"]


def test_deterministic_fix_repairs_what_vlm_could_not(tmp_path):
    client = MockVLM([bad_missing_exam(), worse(), worse(), worse()])
    r, _, _ = generate_plan(PROGRAM, client, tmp_path, name="fix", max_rounds=4)
    assert not r["room_count_exact"]
    assert r["count_fix"]["fixed"]
    assert r["final_count_exact"]
