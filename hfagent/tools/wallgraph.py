# -*- coding: utf-8 -*-
"""Lift room polygons (cv_parse output) into the authoritative wall graph.

Deterministic, three steps:
  1. merge coincident vertices across ALL rooms into shared nodes
  2. split edges at T-junctions: a node sitting mid-edge of another room's
     wall is inserted into that loop, so both sides reference the same nodes
  3. walls = unique consecutive node pairs across loops; each wall records the
     rooms using it (1 = exterior, 2 = party wall -> adjacency for free)

The reverse direction (wallgraph_to_plan) derives room polygons back; nothing
is lost going either way.
"""
from __future__ import annotations

from hfagent.schema.plan import Plan, Room
from hfagent.schema.wallgraph import RoomFace, WallGraph, WallNode, WallSeg

QUANT = 1e-3  # vertex merge precision in plan units


def plan_to_wallgraph(plan: Plan, quant: float = QUANT) -> WallGraph:
    nodes: dict[str, WallNode] = {}
    key_to_id: dict[tuple[int, int], str] = {}

    def node_id(x: float, y: float) -> str:
        k = (round(x / quant), round(y / quant))
        nid = key_to_id.get(k)
        if nid is None:
            nid = f"n{len(nodes)}"
            key_to_id[k] = nid
            nodes[nid] = WallNode(id=nid, x=k[0] * quant, y=k[1] * quant)
        return nid

    # step 1: shared nodes
    loops: dict[str, list[str]] = {}
    for r in plan.rooms:
        loop = [node_id(x, y) for x, y in r.polygon]
        loops[r.id] = [n for i, n in enumerate(loop) if n != loop[(i - 1) % len(loop)]]

    # step 2: T-junction split — insert nodes lying strictly inside an edge
    all_nodes = list(nodes.values())

    def on_edge(n: WallNode, a: WallNode, b: WallNode) -> float | None:
        """Parameter t in (0,1) if n lies on segment a-b, else None."""
        dx, dy = b.x - a.x, b.y - a.y
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return None
        cross = (n.x - a.x) * dy - (n.y - a.y) * dx
        if abs(cross) / (L2 ** 0.5) > quant * 2:
            return None
        t = ((n.x - a.x) * dx + (n.y - a.y) * dy) / L2
        return t if 1e-9 < t < 1 - 1e-9 else None

    for rid, loop in loops.items():
        out: list[str] = []
        for i in range(len(loop)):
            a, b = nodes[loop[i]], nodes[loop[(i + 1) % len(loop)]]
            out.append(a.id)
            mids = [(t, n.id) for n in all_nodes if n.id not in (a.id, b.id)
                    for t in [on_edge(n, a, b)] if t is not None]
            out.extend(nid for _, nid in sorted(mids))
        loops[rid] = out

    # step 3: walls with face bookkeeping
    walls: dict[tuple[str, str], WallSeg] = {}
    for rid, loop in loops.items():
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            if a == b:
                continue
            k = (a, b) if a < b else (b, a)
            w = walls.get(k)
            if w is None:
                w = WallSeg(id=f"w{len(walls)}", n0=k[0], n1=k[1], rooms=[])
                walls[k] = w
            if rid not in w.rooms:
                w.rooms.append(rid)

    return WallGraph(
        plan_id=plan.plan_id,
        units=plan.units,
        nodes=nodes,
        walls=list(walls.values()),
        rooms=[RoomFace(id=r.id, type=r.type, loop=loops[r.id]) for r in plan.rooms],
    )


def wallgraph_to_plan(g: WallGraph) -> Plan:
    """Derived view: rooms with concrete polygons (for rendering / export)."""
    return Plan(
        plan_id=g.plan_id,
        units=g.units,
        rooms=[Room(id=r.id, type=r.type, polygon=g.room_polygon(r.id)) for r in g.rooms],
    )
