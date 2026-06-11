# -*- coding: utf-8 -*-
"""Authoritative plan structure: the wall graph (docs/agent §3, 地基层).

Walls are first-class; rooms no longer own coordinates — they are loops of
node ids. Moving a node moves every wall and room that references it, so edits
never tear the plan apart (the same shape already proven in the frontend CAD
kernel, app/components/floorplan/kernel.ts).

2D is the source of truth; 3D is a derived view — walls carry thickness (and
later height) attributes, so extrusion is free (plan §3.3, ResPlan).
Room polygons remain derivable: see room_polygon().
"""
from __future__ import annotations

from pydantic import BaseModel

SCHEMA_VERSION = "0.1"


class WallNode(BaseModel):
    id: str
    x: float
    y: float


class WallSeg(BaseModel):
    id: str
    n0: str
    n1: str
    thickness: float = 200.0
    height: float = 2800.0  # extrusion attribute: 2D data, 3D for free
    rooms: list[str] = []   # faces: 1 room = exterior wall, 2 rooms = party wall


class RoomFace(BaseModel):
    id: str
    type: str
    loop: list[str]  # ordered node ids; coordinates live in the node table


class WallGraph(BaseModel):
    schema_version: str = SCHEMA_VERSION
    plan_id: str = "p0"
    version: int = 0
    units: str = "mm"
    nodes: dict[str, WallNode] = {}
    walls: list[WallSeg] = []
    rooms: list[RoomFace] = []

    # ── derived views ────────────────────────────────────────────────────────
    def room_polygon(self, room_id: str) -> list[tuple[float, float]]:
        room = next(r for r in self.rooms if r.id == room_id)
        return [(self.nodes[n].x, self.nodes[n].y) for n in room.loop]

    def adjacency(self) -> list[tuple[str, str, str]]:
        """(room_a, room_b, wall_id) for every party wall — the topology layer
        is derived, never stored separately, so it can't go stale."""
        out = []
        for w in self.walls:
            if len(w.rooms) == 2:
                a, b = sorted(w.rooms)
                out.append((a, b, w.id))
        return out

    def exterior_walls(self) -> list[WallSeg]:
        return [w for w in self.walls if len(w.rooms) == 1]
