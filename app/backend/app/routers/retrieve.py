"""Graph retrieval API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.graph_retrieval import msd_retriever


router = APIRouter(prefix="/api/retrieve", tags=["retrieve"])


class EdgeIn(BaseModel):
    source: int
    target: int
    edge_type: int = 0


class RetrieveRequest(BaseModel):
    rooms: list[int]
    edges: list[EdgeIn] = Field(default_factory=list)
    mode: str = "two_stage"
    k: int = 5
    recall_k: int = 50


class GraphNode(BaseModel):
    id: int
    attr: int
    room_type: str


class GraphEdge(BaseModel):
    source: int
    target: int
    edge_type: int
    edge_label: str


class RetrievedGraph(BaseModel):
    num_nodes: int
    num_edges: int
    rooms: list[int]
    adjacency: list[list[int]]
    edge_types: list[list[int]]
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class RetrievedItem(BaseModel):
    idx: int
    score: float
    graph: RetrievedGraph
    floorplan_image: str  # base64 PNG: room polygons (draw_floor_shapes style)
    bubble_image: str     # base64 PNG: bubble graph at centroid positions, typed edges


class RetrieveResponse(BaseModel):
    mode: str
    k: int
    retrieval_seconds: float
    room_types: list[str]
    edge_types: list[str]
    results: list[RetrievedItem]


@router.get("/modes")
def modes():
    return {"modes": msd_retriever.supported_modes()}


@router.post("", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if not req.rooms:
        raise HTTPException(400, "rooms must contain at least one entry")
    if req.k < 1 or req.k > 50:
        raise HTTPException(400, "k must be 1-50")
    if req.recall_k < req.k or req.recall_k > 500:
        raise HTTPException(400, "recall_k must be in [k, 500]")
    try:
        return msd_retriever.retrieve(
            rooms=list(req.rooms),
            edges_in=[e.model_dump() for e in req.edges],
            mode=req.mode,
            k=req.k,
            recall_k=req.recall_k,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))
