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
from hfagent.tools.linework_tracer import find_gaps, trace_linework, window_break_fills
from hfagent.tools.wall_graph import WallSegment

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


def _png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def two_room_plan(draw_extras) -> bytes:
    """Outer rect + one vertical partition with a door gap; ``draw_extras(image)``
    adds the style variation under test (leaf bar, label text, second arc...)."""
    image = np.full((540, 820, 3), 255, np.uint8)
    black = (0, 0, 0)
    cv2.rectangle(image, (30, 30), (790, 510), black, 10)
    cv2.line(image, (410, 30), (410, 240), black, 10)             # partition above the door
    cv2.line(image, (410, 280), (410, 510), black, 10)            # partition below the door
    draw_extras(image)
    return _png(image)


def test_mixed_wall_thickness_is_accepted_and_ww_is_the_thin_mode():
    """Fat perimeter bands over thin partitions (one drawing) must all trace.

    The old estimate-then-gate design took wall_width from an ink percentile (pulled
    up by the fat exterior) and rejected every wall thinner than 0.6x it — exactly
    the thin partitions. Acceptance is now thickness-agnostic and wall_width is the
    median of the accepted runs' own thickness (the partition mode).
    """
    image = np.full((540, 820, 3), 255, np.uint8)
    black = (0, 0, 0)
    cv2.rectangle(image, (30, 30), (790, 510), black, 24)         # fat exterior
    for y in (220, 320):                                          # thin corridor walls
        cv2.line(image, (42, y), (160, y), black, 7)
        cv2.line(image, (200, y), (560, y), black, 7)
        cv2.line(image, (600, y), (778, y), black, 7)
    cv2.line(image, (410, 42, ), (410, 220), black, 7)            # thin partitions
    cv2.line(image, (410, 320), (410, 498), black, 7)
    for hinge_x in (200, 600):                                    # arcs + leaves at the gaps
        cv2.ellipse(image, (hinge_x, 220), (40, 40), 0, 180, 270, black, 2)
        cv2.line(image, (hinge_x, 220), (hinge_x, 180), black, 2)
        cv2.ellipse(image, (hinge_x, 320), (40, 40), 0, 90, 180, black, 2)
        cv2.line(image, (hinge_x, 320), (hinge_x, 360), black, 2)

    trace = trace_linework(_png(image))
    assert len(trace.plan.rooms) == 5                             # nothing lost to the fat bands
    assert trace.diagnostics["doors_detected"] == 4
    assert 6 <= trace.diagnostics["wall_width"] <= 10             # thin (partition) mode, not 24


def test_solid_leaf_bar_is_not_traced_as_a_wall():
    """Some styles draw the open door leaf as a SOLID bar; it must not become a wall
    (a leaf-wall cuts the gap into pier fragments and the door is lost)."""
    def extras(image):
        black = (0, 0, 0)
        cv2.line(image, (370, 240), (410, 240), black, 6)         # leaf bar into the left room
        cv2.ellipse(image, (410, 240), (40, 40), 0, 90, 180, black, 2)

    trace = trace_linework(two_room_plan(extras))
    assert len(trace.plan.rooms) == 2
    assert trace.diagnostics["doors_detected"] == 1


def test_floating_label_text_is_not_wall_and_not_arc_evidence():
    """Fat label glyphs float in room interiors: they must neither trace as walls nor
    score as swing-arc ink (phantom doors bridged mid-room on label strokes)."""
    def door_only(image):
        cv2.ellipse(image, (410, 240), (40, 40), 0, 90, 180, (0, 0, 0), 2)

    def door_and_text(image):
        door_only(image)
        cv2.putText(image, "exam_room_1", (100, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 5)                            # fat isolated glyph strokes

    clean = trace_linework(two_room_plan(door_only))
    labelled = trace_linework(two_room_plan(door_and_text))
    assert len(labelled.plan.rooms) == len(clean.plan.rooms) == 2
    assert labelled.diagnostics["doors_detected"] == clean.diagnostics["doors_detected"] == 1
    assert labelled.diagnostics["walls_traced"] == clean.diagnostics["walls_traced"]


def test_double_leaf_door_confirms_on_half_gap_arc():
    """A double door draws two mirrored quarter arcs, each spanning HALF the opening;
    either leaf's arc alone must confirm the door (r ~= gap/2 radius regime)."""
    image = np.full((540, 820, 3), 255, np.uint8)
    black = (0, 0, 0)
    cv2.rectangle(image, (30, 30), (790, 510), black, 10)
    cv2.line(image, (410, 30), (410, 230), black, 10)             # wide 80px opening
    cv2.line(image, (410, 310), (410, 510), black, 10)
    cv2.line(image, (410, 230), (370, 230), black, 2)                 # upper leaf + arc
    cv2.ellipse(image, (410, 230), (40, 40), 0, 90, 180, black, 2)
    cv2.line(image, (410, 310), (370, 310), black, 2)                 # lower leaf + arc
    cv2.ellipse(image, (410, 310), (40, 40), 0, 180, 270, black, 2)

    trace = trace_linework(_png(image))
    assert len(trace.plan.rooms) == 2
    assert trace.diagnostics["doors_detected"] == 1


def test_detached_room_block_is_traced_as_walls():
    """A block of rooms drawn DETACHED from the outer wall network (central core
    ringed by corridor) holds well under 10% of the drawing's ink, but spans
    room scale in BOTH bbox dimensions - it must anchor as wall network, not be
    dropped like label text (which is glyph-high)."""
    image = np.full((540, 820, 3), 255, np.uint8)
    black = (0, 0, 0)
    cv2.rectangle(image, (30, 30), (790, 510), black, 14)         # heavy outer ring
    cv2.rectangle(image, (330, 200), (490, 340), black, 5)        # floating thin block
    cv2.line(image, (410, 200), (410, 340), black, 5)             # block partition

    trace = trace_linework(_png(image))
    # the block's two cells close; without block anchoring they vanish entirely
    assert len(trace.plan.rooms) >= 3


def test_jogged_wall_door_confirms_and_bridge_connects():
    """A door whose two jambs sit on slightly OFFSET axes (wall thickness changes
    across the opening) has no collinear gap candidate at all; the jogged-gap
    source must pair the walls (their bands overlap) and the bridge must close
    the room via the jog connectors."""
    image = np.full((540, 820, 3), 255, np.uint8)
    black = (0, 0, 0)
    cv2.rectangle(image, (30, 30), (790, 510), black, 10)
    cv2.line(image, (410, 30), (410, 240), black, 10)             # upper partition
    cv2.line(image, (399, 300), (399, 510), black, 22)            # fatter, jogged lower
    cv2.ellipse(image, (407, 240), (60, 60), 0, 90, 180, black, 2)  # arc at upper jamb
    cv2.line(image, (407, 240), (407, 300), black, 2)             # leaf

    trace = trace_linework(_png(image))
    assert trace.diagnostics["doors_detected"] == 1
    assert len(trace.plan.rooms) == 2


def test_wide_double_door_confirms_beyond_single_leaf_cap():
    """A waiting-room double door can be far wider than the single-leaf cap
    (10 * wall_width); only the mirrored half-gap arcs confirm it out there."""
    image = np.full((540, 820, 3), 255, np.uint8)
    black = (0, 0, 0)
    cv2.rectangle(image, (30, 30), (790, 510), black, 10)
    cv2.line(image, (410, 30), (410, 205), black, 10)             # 130px opening (13 ww)
    cv2.line(image, (410, 335), (410, 510), black, 10)
    cv2.line(image, (410, 205), (345, 205), black, 2)             # two mirrored leaves
    cv2.ellipse(image, (410, 205), (65, 65), 0, 90, 180, black, 2)
    cv2.line(image, (410, 335), (345, 335), black, 2)
    cv2.ellipse(image, (410, 335), (65, 65), 0, 180, 270, black, 2)

    trace = trace_linework(_png(image))
    assert trace.diagnostics["doors_detected"] == 1
    assert len(trace.plan.rooms) == 2


def test_undersized_leaf_in_oversized_mouth_confirms():
    """Models sometimes draw a small door in an OVERSIZED mouth (opening just
    past the single-leaf cap, arc radius only ~a third of it); the undersized
    regimes confirm it there - and only there, so tiny arcs cannot hijack
    normal doors' arc ink at door scale."""
    image = np.full((540, 820, 3), 255, np.uint8)
    black = (0, 0, 0)
    cv2.rectangle(image, (30, 30), (790, 510), black, 20)
    cv2.line(image, (410, 30), (410, 232), black, 20)             # ~172px opening, just
    cv2.line(image, (410, 425), (410, 510), black, 20)            # past the 10*ww cap
    cv2.ellipse(image, (410, 415), (57, 57), 0, 180, 270, black, 2)  # r = 0.33 * gap
    cv2.line(image, (410, 415), (353, 415), black, 2)             # open leaf, into the room

    trace = trace_linework(_png(image))
    assert trace.diagnostics["doors_detected"] == 1
    assert len(trace.plan.rooms) == 2


def test_window_break_fill_requires_continuous_face_ink():
    """A traced-band break whose cross band keeps CONTINUOUS (wall-anchored) ink
    - hollow window faces - is drawn wall and gets filled; the same break with a
    plain white opening must stay open (it may only close as an arc-confirmed
    door)."""
    ww = 10
    walls = [
        WallSegment("vertical", 410.0, 30.0, 200.0, 10.0),
        WallSegment("vertical", 410.0, 280.0, 510.0, 10.0),
    ]
    dark = np.zeros((540, 820), np.uint8)
    dark[30:200, 405:416] = 1                                     # band above the break
    dark[280:510, 405:416] = 1                                    # band below the break
    with_faces = dark.copy()
    with_faces[200:280, 406:408] = 1                              # window face lines
    with_faces[200:280, 413:415] = 1

    def anchor_for(ink):
        labels = (ink > 0).astype(np.int32)
        anchored = np.array([False, True])
        symbol_like = np.array([False, False])
        return labels, anchored, symbol_like

    gaps = find_gaps(walls, ww)
    assert len(gaps) == 1
    filled = window_break_fills(gaps, [], with_faces, anchor_for(with_faces), ww)
    assert len(filled) == 1
    assert filled[0].start == 200.0 and filled[0].end == 280.0
    open_break = window_break_fills(gaps, [], dark, anchor_for(dark), ww)
    assert open_break == []


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
