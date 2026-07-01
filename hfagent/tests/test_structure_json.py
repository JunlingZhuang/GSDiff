# -*- coding: utf-8 -*-
"""structure_mode="json": the VLM reads the realistic plan into rooms+doors JSON and
`rectify` builds a clean Plan deterministically — no to_colorblock, no cv_parse. The
default colorblock path stays unchanged. Both share the whole downstream.
"""
from hfagent.floor_plan_generate import generate_plan
from hfagent.tools.rectify import building_aspect, rectify
from hfagent.tools.structure_reader import RoomShape
from hfagent.schema.roomgraph import RoomGraph, RoomNode
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


def test_structure_json_mode_builds_unified_plan(out_dir):
    client = MockVLM([simple_clinic()], door_script=[{"room_a": "exam_room_1", "room_b": "corridor"}])
    r, plan, g = generate_plan(
        PROGRAM, client, out_dir, name="sj", max_rounds=1, structure_mode="json"
    )
    work = out_dir / "sj"
    # json path writes the structured read, NOT a colour-block image
    assert (work / "structure_r1.json").exists()
    assert not (work / "gemini_r1.png").exists()
    assert r["structure_mode"] == "json"
    # same unified Plan as the colorblock path: rooms + walls + doors-on-walls + adjacency
    assert len(plan["rooms"]) == 5
    assert plan["walls"]
    wall_ids = {w["id"] for w in plan["walls"]}
    assert all(d["wall_id"] in wall_ids for d in plan["doors"])
    door_ids = {d["id"] for d in plan["doors"]}
    assert all(e["via"] in door_ids for e in plan["adjacency_graph"]["edges"])
    # the scripted door edge survived into the room graph
    assert r["room_graph"]["doors"] == 1
    # aspect ratio is set by the MEASURED building footprint, not by drifting VLM coords
    real_png = (work / "gemini_r1.real.png").read_bytes()
    xs = [p[0] for rm in plan["rooms"] for p in rm["polygon"]]
    ys = [p[1] for rm in plan["rooms"] for p in rm["polygon"]]
    ratio = (max(xs) - min(xs)) / (max(ys) - min(ys))
    assert abs(ratio - building_aspect(real_png)) < 0.05


def test_rectify_keeps_l_shaped_room():
    """A room given as two rectangles traces a non-rectangular (>4-vertex) polygon —
    the win over a single bbox: a bent corridor / L-shaped room no longer collapses to a box."""
    rect_room = RoomShape(id="exam_room", type="exam_room", rects=[(0.5, 0.5, 0.8, 0.8)])
    l_room = RoomShape(id="corridor", type="corridor", rects=[(0.0, 0.0, 0.4, 0.2), (0.0, 0.2, 0.2, 0.5)])
    plan = rectify([rect_room, l_room], aspect=2.0)
    polys = {r.type: r.polygon for r in plan.rooms}
    assert len(polys["exam_room"]) == 4   # plain rectangle
    assert len(polys["corridor"]) > 4     # L-shape preserved


def test_colorblock_mode_unaffected(out_dir):
    """Default path still works unchanged and writes a colour-block image."""
    client = MockVLM([simple_clinic()])
    r, plan, g = generate_plan(PROGRAM, client, out_dir, name="cb", max_rounds=1)
    assert r["structure_mode"] == "colorblock"
    assert (out_dir / "cb" / "gemini_r1.png").exists()
    assert not (out_dir / "cb" / "structure_r1.json").exists()
    assert len(plan["rooms"]) == 5


def test_linework_mode_skips_ai_colorblock(out_dir, monkeypatch):
    parsed = simple_clinic()
    graph = RoomGraph(
        rooms=[RoomNode(id=room.id, type=room.type) for room in parsed.rooms],
        doors=[],
    )
    diagnostics = {
        "room_regions": 5,
        "labelled_rooms": 5,
        "unlabelled_rooms": 0,
        "door_candidates": [],
        "parser_confident": True,
    }

    monkeypatch.setattr(
        "hfagent.floor_plan_generate.parse_linework",
        lambda image, program: (parsed, graph, diagnostics, b"\x89PNG debug"),
    )
    client = MockVLM([parsed])
    report, plan, _ = generate_plan(
        PROGRAM, client, out_dir, name="lw", max_rounds=1, structure_mode="linework"
    )

    work = out_dir / "lw"
    assert report["structure_mode"] == "linework"
    assert client.calls == 1
    assert (work / "linework_r1.json").exists()
    assert (work / "linework_r1.debug.png").exists()
    assert (work / "gemini_r1.png").exists()  # deterministic compatibility mask
    assert len(plan["rooms"]) == 5
