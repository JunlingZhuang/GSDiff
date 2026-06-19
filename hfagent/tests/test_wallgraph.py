# -*- coding: utf-8 -*-
"""Wall graph lift: shared nodes, T-junction splits, derived adjacency,
lossless polygon round-trip."""
from shapely.geometry import Polygon

from hfagent.tests.synth import simple_clinic, ward_wing
from hfagent.tools.wallgraph import plan_to_wallgraph, wallgraph_to_plan


def test_coincident_corners_become_shared_nodes():
    plan = simple_clinic()
    g = plan_to_wallgraph(plan)
    total_vertices = sum(len(r.polygon) for r in plan.rooms)  # 20
    assert len(g.nodes) < total_vertices  # corners merged
    # corner (6000, 0) is shared by r1 and r2 -> same node id in both loops
    shared = [n.id for n in g.nodes.values() if (n.x, n.y) == (6000.0, 0.0)]
    assert len(shared) == 1
    r1 = next(r for r in g.rooms if r.id == "r1")
    r2 = next(r for r in g.rooms if r.id == "r2")
    assert shared[0] in r1.loop and shared[0] in r2.loop


def test_t_junction_nodes_inserted_into_long_edge():
    g = plan_to_wallgraph(ward_wing())
    corridor = next(r for r in g.rooms if r.id == "c")
    # the corridor rectangle has 4 corners, but rooms above/below end on its
    # edges -> their corner nodes must be inserted into the corridor loop
    assert len(corridor.loop) > 4


def test_party_wall_recorded_once_with_both_faces():
    g = plan_to_wallgraph(simple_clinic())
    party = [w for w in g.walls if sorted(w.rooms) == ["r1", "r2"]]
    assert party, "no party wall between r1 and r2"
    assert all(len(w.rooms) == 2 for w in party)


def test_adjacency_is_derived_correctly():
    g = plan_to_wallgraph(ward_wing())
    pairs = {(a, b) for a, b, _ in g.adjacency()}
    # every top-row patient room touches the corridor
    for pid in ("p0", "p1", "p2"):
        assert tuple(sorted((pid, "c"))) in pairs
    # exterior walls exist (building boundary)
    assert g.exterior_walls()


def test_polygon_roundtrip_lossless():
    plan = ward_wing()
    g = plan_to_wallgraph(plan)
    back = wallgraph_to_plan(g)
    for orig, der in zip(plan.rooms, back.rooms):
        assert orig.id == der.id and orig.type == der.type
        # extra collinear vertices are fine; the SHAPE must be identical
        assert Polygon(orig.polygon).equals(Polygon(der.polygon)), orig.id


def test_node_edit_propagates_to_all_referencing_rooms():
    g = plan_to_wallgraph(simple_clinic())
    shared = next(n.id for n in g.nodes.values() if (n.x, n.y) == (6000.0, 0.0))
    g.nodes[shared].x = 6500.0  # drag the corner
    p1 = Polygon(g.room_polygon("r1"))
    p2 = Polygon(g.room_polygon("r2"))
    # both rooms followed the node — no tear, no overlap beyond numeric noise
    assert p1.intersection(p2).area < 1.0
    assert abs((p1.area + p2.area) - 2 * 6000 * 5000) < 1.0  # area conserved
