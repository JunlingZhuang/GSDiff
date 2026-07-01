# -*- coding: utf-8 -*-
"""structure_mode="linework": deterministic offline tests of the faithful-trace engine.

A synthetic clean line plan (black wall bands, four door openings with quarter-circle
swing arcs + leaves, white interiors) stands in for the Gemini drawing: the tracer must
close every drawn room, confirm exactly the drawn doors, and — through the pipeline —
produce the unified Plan and the fixed trace artifacts. No model calls.
"""
import cv2
import numpy as np

from hfagent.floor_plan_generate import generate_plan
from hfagent.tools.linework_tracer import trace_linework

TRACE_ARTIFACTS = [
    "walls_overlay.png",
    "doors_overlay.png",
    "recon.png",
    "recon_post.png",
    "rooms_colorful.png",
]

# 5 rooms total (2 top, corridor, 2 bottom) — matches the synthetic drawing below
PROGRAM = {
    "building_type": "test clinic",
    "rooms": [
        {"type": "exam_room", "count": 2},
        {"type": "office", "count": 2},
        {"type": "corridor", "count": 1},
    ],
    "adjacency": [["exam_room", "corridor"], ["office", "corridor"]],
}


def synthetic_line_plan_png() -> bytes:
    """5 rooms around a corridor; each room has one door (gap + swing arc + leaf)."""
    image = np.full((540, 820, 3), 255, np.uint8)
    black = (0, 0, 0)
    t = 10
    cv2.rectangle(image, (30, 30), (790, 510), black, t)          # outer walls
    for y in (220, 320):                                          # corridor walls, 2 gaps each
        cv2.line(image, (30, y), (160, y), black, t)
        cv2.line(image, (200, y), (560, y), black, t)
        cv2.line(image, (600, y), (790, y), black, t)
    cv2.line(image, (410, 30), (410, 220), black, t)              # top-room partition
    cv2.line(image, (410, 320), (410, 510), black, t)             # bottom-room partition
    for hinge_x in (200, 600):                                    # thin swing arcs + leaves
        cv2.ellipse(image, (hinge_x, 220), (40, 40), 0, 180, 270, black, 2)
        cv2.line(image, (hinge_x, 220), (hinge_x, 180), black, 2)
        cv2.ellipse(image, (hinge_x, 320), (40, 40), 0, 90, 180, black, 2)
        cv2.line(image, (hinge_x, 320), (hinge_x, 360), black, 2)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


class LineworkImageStub:
    """Minimal image client: every generate_image call returns the synthetic drawing."""
    image_model = "stub-linework-image"
    text_model = "stub-text"

    def __init__(self, png: bytes):
        self.png = png
        self.calls = 0

    def generate_image(self, contents) -> bytes:
        self.calls += 1
        return self.png


def test_trace_linework_closes_rooms_and_confirms_doors():
    trace = trace_linework(synthetic_line_plan_png())

    # geometry-only plan: 5 untyped rooms in px units
    assert trace.plan.units == "px"
    assert len(trace.plan.rooms) == 5
    assert all(room.type == "unknown" for room in trace.plan.rooms)
    assert [room.id for room in trace.plan.rooms] == ["r1", "r2", "r3", "r4", "r5"]

    # exactly the 4 drawn doors, every edge through the one corridor room
    assert trace.diagnostics["doors_detected"] == 4
    edges = [(door.room_a, door.room_b) for door in trace.room_graph.doors]
    assert len(edges) == 4
    common = set.intersection(*(set(edge) for edge in edges))
    assert len(common) == 1                       # the corridor appears in every edge
    assert {room for edge in edges for room in edge} == {f"r{i}" for i in range(1, 6)}

    # all five fixed artifacts, each a decodable PNG
    assert sorted(trace.artifacts) == sorted(TRACE_ARTIFACTS)
    assert all(payload.startswith(b"\x89PNG") for payload in trace.artifacts.values())


def test_linework_pipeline_single_round_and_fixed_artifacts(out_dir):
    """generate_plan(linework): one image call, one round, unified Plan + artifacts."""
    client = LineworkImageStub(synthetic_line_plan_png())
    report, plan, graph = generate_plan(
        PROGRAM, client, out_dir, name="lw", max_rounds=3, structure_mode="linework"
    )
    work = out_dir / "lw"

    # single round regardless of max_rounds; exactly one image call, no colour-block
    assert report["structure_mode"] == "linework"
    assert len(report["rounds"]) == 1
    assert client.calls == 1
    assert (work / "gemini_r1.real.png").exists()
    assert not (work / "gemini_r1.png").exists()

    # verification is total rooms traced vs the program total (rooms are untyped)
    assert report["rounds"][0]["rooms_traced"] == 5
    assert report["rounds"][0]["rooms_required_total"] == 5
    assert report["room_count_exact"]
    assert report["final_count_exact"]
    assert report["count_fix"]["ops"] == []

    # the fixed trace artifacts land in the work dir every run
    for artifact in [
        "walls_overlay.png", "doors_overlay.png", "recon.png",
        "recon_post.png", "rooms_colorful.png",
    ]:
        assert (work / artifact).exists(), artifact
    assert (work / "graph.json").exists()
    assert (work / "plan.json").exists()
    assert (work / "recon_with_door.png").exists()

    # unified Plan: untyped rooms + walls from the traced polygons + doors hung on
    # the exact walls the room pairs share (exact_connected), linked via adjacency
    assert len(plan["rooms"]) == 5
    assert all(room["type"] == "unknown" for room in plan["rooms"])
    assert plan["walls"] and plan["doors"]
    wall_ids = {w["id"] for w in plan["walls"]}
    assert all(d["wall_id"] in wall_ids for d in plan["doors"])
    door_ids = {d["id"] for d in plan["doors"]}
    edges = plan["adjacency_graph"]["edges"]
    assert edges and all(e["via"] in door_ids for e in edges)
    assert len(graph["doors"]) == 4
