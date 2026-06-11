"""Procedural (rule-based) floor-plan generation endpoint."""

import sys

import networkx as nx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from shapely.geometry import Polygon

from app.config import PROJECT_ROOT

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from procedural.plan_export import build_plan  # noqa: E402

router = APIRouter(prefix="/api/generate", tags=["procedural"])


class GraphNode(BaseModel):
    id: int
    room_type: str


class GraphEdge(BaseModel):
    source: int
    target: int
    connectivity: str = "wall"


class GraphIn(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ProceduralRequest(BaseModel):
    graph: GraphIn
    boundary: list[list[float]]
    seed: int = 0
    # Optional building-axis angle in degrees; None auto-detects from boundary.
    axis_angle: float | None = None


def _to_nx(graph: GraphIn) -> nx.Graph:
    g = nx.Graph()
    for node in graph.nodes:
        g.add_node(node.id, room_type=node.room_type)
    for e in graph.edges:
        g.add_edge(e.source, e.target, connectivity=e.connectivity)
    return g


@router.post("/procedural")
def procedural(req: ProceduralRequest):
    if len(req.boundary) < 3:
        raise HTTPException(400, "boundary needs at least 3 points")
    if any(len(p) != 2 for p in req.boundary):
        raise HTTPException(400, "each boundary point must have exactly 2 coordinates")
    try:
        g = _to_nx(req.graph)
        boundary = Polygon(req.boundary)
        if not boundary.is_valid:
            boundary = boundary.buffer(0)
        return build_plan(g, boundary, seed=req.seed, axis_angle=req.axis_angle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
