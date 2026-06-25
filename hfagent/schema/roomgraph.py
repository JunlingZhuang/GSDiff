# -*- coding: utf-8 -*-
"""Room graph: the structural output of generation.

Nodes are rooms (one per instance, e.g. patient_room_1..patient_room_12).
Edges are doors, read directly from the realistic floor-plan image — never
inferred from geometry (doors are invisible in the colour-block plan; the only
reliable source is the original architectural drawing).

This replaces the earlier wall graph: we care about *which rooms connect through
a door*, not about wall segment topology.
"""
from __future__ import annotations

from pydantic import BaseModel

SCHEMA_VERSION = "0.1"


class RoomNode(BaseModel):
    id: str    # instance label, e.g. "patient_room_3" (matches the image labels)
    type: str  # base room type, e.g. "patient_room"


class Door(BaseModel):
    """A logical door edge: rooms A and B connect. No geometry — the LLM returns
    only the connectivity; post-processing (door_placer) puts the physical door
    centred on the wall the two rooms share."""
    room_a: str  # RoomNode.id, or "exterior" for an external door
    room_b: str


class RoomGraph(BaseModel):
    schema_version: str = SCHEMA_VERSION
    plan_id: str = "p0"
    units: str = "logical"
    rooms: list[RoomNode] = []
    doors: list[Door] = []

    def adjacency(self) -> list[tuple[str, str]]:
        """Unordered (room_a, room_b) pairs connected by a door."""
        return [tuple(sorted((d.room_a, d.room_b))) for d in self.doors]
