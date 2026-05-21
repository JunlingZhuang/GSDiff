import json
import pickle
from pathlib import Path

from procedural.plan_export import derive_walls, _seg_key, derive_openings, build_plan
from procedural import io_msd


def _square(x0, y0, x1, y1):
    """Closed CCW polygon as list of (x, y), first point repeated at end."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def test_derive_walls_two_adjacent_squares_dedup_shared_edge():
    # Two unit squares sharing the edge x=1, y in [0,1].
    rooms = {
        "rA": _square(0, 0, 1, 1),
        "rB": _square(1, 0, 2, 1),
    }
    walls = derive_walls(rooms, thickness=0.2)
    # 4 + 4 edges, the shared (1,0)-(1,1) edge deduped to one → 7 unique walls.
    assert len(walls) == 7
    # every wall has the requested thickness and non-zero length
    for w in walls:
        assert w["thickness"] == 0.2
        ax, ay = w["a"]
        bx, by = w["b"]
        assert (ax, ay) != (bx, by)
    # the shared segment appears exactly once
    def key(w):
        return tuple(sorted([tuple(w["a"]), tuple(w["b"])]))
    keys = [key(w) for w in walls]
    shared = tuple(sorted([(1.0, 0.0), (1.0, 1.0)]))
    assert keys.count(shared) == 1


def test_derive_openings_door_edge_makes_one_opening():
    rooms = {"rA": _square(0, 0, 1, 1), "rB": _square(1, 0, 2, 1)}
    walls = derive_walls(rooms, thickness=0.2)
    # graph edge between rA and rB with a door → one opening on the shared wall
    edges = [{"source": "rA", "target": "rB", "connectivity": "door"}]
    openings = derive_openings(rooms, walls, edges)
    assert len(openings) == 1
    o = openings[0]
    assert o["kind"] == "door"
    # the opening's wall is the shared (1,0)-(1,1) segment
    wall = next(w for w in walls if w["id"] == o["wallId"])
    assert _seg_key(wall["a"], wall["b"]) == tuple(sorted([(1.0, 0.0), (1.0, 1.0)]))
    assert 0.0 <= o["t"] <= 1.0


def test_derive_openings_wall_edge_makes_none():
    rooms = {"rA": _square(0, 0, 1, 1), "rB": _square(1, 0, 2, 1)}
    walls = derive_walls(rooms, thickness=0.2)
    edges = [{"source": "rA", "target": "rB", "connectivity": "wall"}]
    openings = derive_openings(rooms, walls, edges)
    assert openings == []


def test_derive_openings_entrance_maps_to_door_kind():
    rooms = {"rA": _square(0, 0, 1, 1), "rB": _square(1, 0, 2, 1)}
    walls = derive_walls(rooms, thickness=0.2)
    edges = [{"source": "rA", "target": "rB", "connectivity": "entrance"}]
    openings = derive_openings(rooms, walls, edges)
    assert len(openings) == 1
    assert openings[0]["kind"] == "door"


def _load_multiapt_graph():
    p = Path(__file__).resolve().parents[2] / "digress" / "data" / "msd_wall_v6" / "graphs.p"
    with open(p, "rb") as f:
        graphs = pickle.load(f)
    # smallest graph for a fast test
    return min(graphs, key=lambda g: g.number_of_nodes())


def test_build_plan_shape_and_json_serializable():
    g = _load_multiapt_graph()
    boundary = io_msd.boundary_for_processed_graph(g)
    assert boundary is not None
    plan = build_plan(g, boundary, seed=0)

    for key in ("walls", "openings", "rooms", "grid", "unit"):
        assert key in plan
    assert plan["unit"] == "m"
    assert len(plan["rooms"]) > 0
    assert len(plan["walls"]) > 0
    # every room references walls and has a polygon + type
    r = plan["rooms"][0]
    for key in ("id", "type", "poly", "wallIds"):
        assert key in r
    # grid carries the building axis angle
    assert "angleDeg" in plan["grid"]
    # at least one room must actually link to walls (guards silent key mismatch)
    assert any(len(r["wallIds"]) > 0 for r in plan["rooms"])
    # whole thing must be JSON-serializable (it crosses the HTTP boundary)
    json.dumps(plan)
