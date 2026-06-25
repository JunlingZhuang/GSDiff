# -*- coding: utf-8 -*-
"""Building plan data model (Phase 0 minimal subset of docs/agent/02-data-model.md).

Units are millimetres, coordinates are (x, y) with y growing downwards (image
convention). Forward-compatible: walls / doors / adjacency are present as
optional fields so later phases extend this schema instead of replacing it.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1"


class Room(BaseModel):
    id: str
    type: str
    polygon: list[tuple[float, float]] = Field(min_length=3)  # closed implied (last != first)
    name: str | None = None   # human label (docs/agent §3.2); optional
    zone: str | None = None   # functional zone, e.g. "ward"; optional

    def area(self) -> float:
        pts = self.polygon
        s = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            s += x0 * y1 - x1 * y0
        return abs(s) / 2.0


class Wall(BaseModel):  # Phase >=1
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float = 200.0
    load_bearing: bool = False


class Door(BaseModel):  # Phase >=1; hangs on a wall (wall_id + position 0..1)
    id: str
    wall_id: str
    position: float = 0.5
    width: float = 1200.0
    type: str = "door"


class AdjEdge(BaseModel):
    """Topology-layer edge: two rooms connected, realised by a door (docs §3.2)."""
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")  # room id (or "exterior")
    to: str                            # room id (or "exterior")
    type: str = "door"
    via: str | None = None             # the Door.id that realises this edge


class AdjacencyGraph(BaseModel):
    nodes: list[str] = []   # room ids
    edges: list[AdjEdge] = []


class Plan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    plan_id: str = "p0"
    version: int = 0
    units: str = "mm"
    building_type: str = "healthcare_generic"
    rooms: list[Room] = []
    walls: list[Wall] = []
    doors: list[Door] = []
    adjacency_graph: AdjacencyGraph = Field(default_factory=AdjacencyGraph)

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [x for r in self.rooms for x, _ in r.polygon]
        ys = [y for r in self.rooms for _, y in r.polygon]
        if not xs:
            return (0.0, 0.0, 1.0, 1.0)
        return (min(xs), min(ys), max(xs), max(ys))
